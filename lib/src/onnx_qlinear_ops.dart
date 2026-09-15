/// QOperator-format quantized operators: `MatMulInteger`, `ConvInteger`,
/// `QLinearMatMul`, `QLinearConv` (per the public ONNX operator spec).
///
/// Quantized operands are read through `Tensor.intData`, which covers both
/// the compact uint8/int8 storage (weights, quantized activations) and
/// int64; accumulation is exact int32 semantics with headroom to spare.
/// Requantization uses round-half-to-even on the float multiplier, matching
/// the ONNX reference / ORT CPU behavior.
library;

import 'dart:typed_data' hide Int64List;
import 'int64_storage.dart';

import 'gemm_kernel_scalar.dart' if (dart.library.ffi) 'gemm_kernel_simd.dart'
    as gemm;
import 'tensor.dart';
import 'execution_control.dart';

int _roundEven(double v) {
  final f = v.floorToDouble();
  final frac = v - f;
  if (frac > 0.5) return f.toInt() + 1;
  if (frac < 0.5) return f.toInt();
  final i = f.toInt();
  return i.isEven ? i : i + 1;
}

/// `MatMulInteger`: `y_int32 = (A - aZp) · (B - bZp)`. 2-D or batched like
/// MatMul (batch dims must match exactly here — quantized graphs don't
/// broadcast matmul batches in practice). Zero points may be scalar,
/// per-row `[M]` for A, or per-column `[N]` for B.
Tensor opMatMulInteger(Tensor a, Tensor b, Tensor? aZp, Tensor? bZp) {
  final aRank = a.rank, bRank = b.rank;
  final m = a.shape[aRank - 2], k = a.shape[aRank - 1];
  final n = b.shape[bRank - 1];
  assert(b.shape[bRank - 2] == k, 'MatMulInteger inner dim mismatch');
  if ((aZp != null && aZp.length != 1 && aZp.length != m) ||
      (bZp != null && bZp.length != 1 && bZp.length != n)) {
    throw UnsupportedError('MatMulInteger: zero points must be scalar, '
        'per-row [M] for A or per-column [N] for B');
  }
  final az = aZp == null || aZp.length != 1 ? 0 : aZp.getI(0);
  final azRow = aZp != null && aZp.length == m && m != 1 ? aZp.intData : null;
  final bz = bZp == null || bZp.length != 1 ? 0 : bZp.getI(0);
  final bzCol = bZp != null && bZp.length == n && n != 1 ? bZp.intData : null;
  final aBatch = a.length ~/ (m * k), bBatch = b.length ~/ (k * n);
  final batch = aBatch > bBatch ? aBatch : bBatch;
  assert(aBatch == batch || aBatch == 1, 'MatMulInteger batch mismatch');
  assert(bBatch == batch || bBatch == 1, 'MatMulInteger batch mismatch');

  final ai = a.intData, bi = b.intData;
  final out = Int64List(batch * m * n);
  for (int bb = 0; bb < batch; bb++) {
    final aOff = (aBatch == 1 ? 0 : bb) * m * k;
    final bOff = (bBatch == 1 ? 0 : bb) * k * n;
    final oOff = bb * m * n;
    for (int i = 0; i < m; i++) {
      final azi = azRow != null ? azRow[i] : az;
      for (int kk = 0; kk < k; kk++) {
        final av = ai[aOff + i * k + kk] - azi;
        // Zero contributions are exact to skip even with per-column bZp.
        if (av == 0) continue;
        final bRow = bOff + kk * n;
        final oRow = oOff + i * n;
        if (bzCol == null) {
          for (int j = 0; j < n; j++) {
            out[oRow + j] += av * (bi[bRow + j] - bz);
          }
        } else {
          for (int j = 0; j < n; j++) {
            out[oRow + j] += av * (bi[bRow + j] - bzCol[j]);
          }
        }
      }
    }
  }
  final outBatchShape = aBatch >= bBatch
      ? a.shape.sublist(0, aRank - 2)
      : b.shape.sublist(0, bRank - 2);
  return Tensor.int64(out, [...outBatchShape, m, n]);
}

/// `QLinearMatMul`: MatMulInteger + requantize with `aS*bS/yS`, saturating
/// to [lo], [hi] (from the output zero-point's dtype).
Tensor opQLinearMatMul(Tensor a, Tensor aS, Tensor? aZp, Tensor b, Tensor bS,
    Tensor? bZp, Tensor yS, Tensor? yZp,
    {required int lo, required int hi}) {
  if (aS.length != 1 || bS.length != 1 || yS.length != 1) {
    throw UnsupportedError('QLinearMatMul: only per-tensor scales supported');
  }
  final acc = opMatMulInteger(a, b, aZp, bZp);
  // The requant multiplier and product are float32 in the reference / ORT,
  // rounded at each step — a single rounding from double lands on the other
  // side of .5 boundaries (observed on real data).
  final f32 = Float32List(1);
  f32[0] = aS.getD(0) * bS.getD(0);
  f32[0] = f32[0] / yS.getD(0);
  final mult = f32[0].toDouble();
  final yz = yZp?.getI(0) ?? 0;
  final ai = acc.i!;
  final out = Int64List(ai.length);
  for (int i = 0; i < ai.length; i++) {
    f32[0] = ai[i] * mult;
    final q = _roundEven(f32[0].toDouble()) + yz;
    out[i] = q < lo ? lo : (q > hi ? hi : q);
  }
  return Tensor.int64(out, acc.shape);
}

/// `MatMulNBits` (com.microsoft): `y = A · dequant(B)ᵀ`, B block-quantized
/// to 4 bits. B is `[N, ceil(K/block_size), block_size/2]` packed low-nibble
/// first; scales are one float per (column, block); zero points default to 8
/// or come packed 4-bit the same way. Token decoding reads packed weights
/// directly; multi-row execution dequantizes bounded column tiles for GEMM.
Tensor opMatMulNBits(
  Tensor a,
  Tensor bQ,
  Tensor scales,
  Tensor? zeroPoints, {
  required int k,
  required int n,
  required int bits,
  required int blockSize,
  Tensor? bias,
}) {
  if (bits != 4) {
    throw UnsupportedError('MatMulNBits: only bits=4 supported, got $bits');
  }
  if (k <= 0 ||
      n <= 0 ||
      blockSize < 16 ||
      (blockSize & (blockSize - 1)) != 0 ||
      a.rank < 1 ||
      a.shape.last != k) {
    throw ArgumentError('MatMulNBits: invalid K, N, block_size or input shape');
  }
  final nBlocks = (k + blockSize - 1) ~/ blockSize;
  final blobBytes = blockSize ~/ 2;
  final zpBytes = zeroPoints?.u8;
  final zpStride = (nBlocks + 1) ~/ 2;
  if (bQ.rank != 3 ||
      bQ.shape[0] != n ||
      bQ.shape[1] != nBlocks ||
      bQ.shape[2] != blobBytes ||
      bQ.dtype != DType.uint8 ||
      !a.isFloat ||
      !scales.isFloat ||
      scales.length != n * nBlocks ||
      (scales.rank != 1 && scales.rank != 2) ||
      (scales.rank == 2 &&
          (scales.shape[0] != n || scales.shape[1] != nBlocks))) {
    throw ArgumentError('MatMulNBits: invalid packed weights or scales');
  }
  final bBytes = bQ.u8!;
  final sf = scales.f!;
  if (zeroPoints != null &&
      ((zeroPoints.rank != 1 && zeroPoints.rank != 2) ||
          (zeroPoints.rank == 2 && zeroPoints.shape[0] != n) ||
          (zpBytes != null
              ? zeroPoints.length != n * zpStride
              : !zeroPoints.isFloat || zeroPoints.length != n * nBlocks))) {
    throw ArgumentError('MatMulNBits: invalid zero_points');
  }
  if (bias != null && (!bias.isFloat || bias.rank != 1 || bias.length != n)) {
    throw ArgumentError('MatMulNBits: bias must be floating point [N]');
  }

  final af = a.asFloatList();
  final m = a.length ~/ k;
  if (m == 0)
    return Tensor.float(Float32List(0), [...a.shape.sublist(0, a.rank - 1), n]);
  activeOnnxControl?.checkAdditionalBytes(
      4 * (m * n + (m == 1 ? 0 : (k + m) * (n < 32 ? n : 32))));
  final out = Float32List(m * n);
  final rounded = Float32List(1);

  double weight(int col, int kk) {
    final blk = kk ~/ blockSize;
    final byte =
        bBytes[(col * nBlocks + blk) * blobBytes + (kk % blockSize) ~/ 2];
    final q = (kk & 1) == 0 ? byte & 15 : byte >> 4;
    double zp = 8;
    if (zpBytes != null) {
      final z = zpBytes[col * zpStride + blk ~/ 2];
      zp = ((blk & 1) == 0 ? z & 15 : z >> 4).toDouble();
    } else if (zeroPoints != null) {
      zp = zeroPoints.getD(col * nBlocks + blk);
    }
    rounded[0] = (q - zp) * sf[col * nBlocks + blk];
    return rounded[0];
  }

  // Decode directly from packed weights for token-at-a-time inference.
  if (m == 1) {
    for (var col = 0; col < n; col++) {
      activeOnnxControl?.checkpoint();
      double sum = 0;
      for (var kk = 0; kk < k; kk++) {
        if ((kk & 4095) == 0) activeOnnxControl?.checkpoint();
        sum += af[kk] * weight(col, kk);
      }
      out[col] = sum + (bias?.getD(col) ?? 0);
    }
    return Tensor.float(out, [...a.shape.sublist(0, a.rank - 1), n]);
  }

  // Bounded column tiles avoid a full K*N floating-point weight copy.
  const tileColumns = 32;
  for (var start = 0; start < n; start += tileColumns) {
    activeOnnxControl?.checkpoint();
    final columns = n - start < tileColumns ? n - start : tileColumns;
    final bt = Float32List(k * columns);
    for (var col = 0; col < columns; col++) {
      for (var kk = 0; kk < k; kk++) {
        bt[kk * columns + col] = weight(start + col, kk);
      }
    }
    final tile = Float32List(m * columns);
    gemm.matmulKernel(af, 0, bt, 0, tile, 0, m, k, columns);
    for (var row = 0; row < m; row++) {
      for (var col = 0; col < columns; col++) {
        out[row * n + start + col] =
            tile[row * columns + col] + (bias?.getD(start + col) ?? 0);
      }
    }
  }
  return Tensor.float(out, [...a.shape.sublist(0, a.rank - 1), n]);
}

/// Integer 2-D convolution accumulator shared by ConvInteger / QLinearConv:
/// `acc[n,m,oy,ox] = Σ (x - xZp)(w - wZp[m])`, int32-exact.
Int64List _convIntAcc(
  Tensor x,
  Tensor w,
  int xZp,
  List<int> wZpPerChannel, {
  required List<int> strides,
  required List<int> pads,
  required List<int> dilations,
  required int group,
  required List<int> outSp,
}) {
  final n = x.shape[0], cIn = x.shape[1];
  final h = x.shape[2], wd = x.shape[3];
  final m = w.shape[0], cPerGroup = w.shape[1];
  final kh = w.shape[2], kw = w.shape[3];
  final oh = outSp[0], ow = outSp[1];
  final sh = strides[0], sw = strides[1];
  final dh = dilations[0], dw = dilations[1];
  final ph = pads[0], pw = pads[1];
  final mPerGroup = m ~/ group;
  final xi = x.intData, wi = w.intData;
  final out = Int64List(n * m * oh * ow);

  for (int b = 0; b < n; b++) {
    for (int om = 0; om < m; om++) {
      final g = om ~/ mPerGroup;
      final wz = wZpPerChannel[om];
      final wBase = om * cPerGroup * kh * kw;
      final outBase = (b * m + om) * oh * ow;
      for (int oy = 0; oy < oh; oy++) {
        final iy0 = oy * sh - ph;
        for (int ox = 0; ox < ow; ox++) {
          final ix0 = ox * sw - pw;
          int acc = 0;
          for (int c = 0; c < cPerGroup; c++) {
            final xBase = (b * cIn + g * cPerGroup + c) * h * wd;
            final wcBase = wBase + c * kh * kw;
            for (int ky = 0; ky < kh; ky++) {
              final iy = iy0 + ky * dh;
              // Padding contributes (xZp - xZp) = 0... only when the pad
              // value equals the zero point, which is exactly the ONNX
              // definition — so out-of-range taps are simply skipped, but
              // the weight-side zero point must still see the pad: the
              // (x - xZp) factor is 0 there, so skipping is exact.
              if (iy < 0 || iy >= h) continue;
              final xRow = xBase + iy * wd;
              final wRow = wcBase + ky * kw;
              for (int kx = 0; kx < kw; kx++) {
                final ix = ix0 + kx * dw;
                if (ix < 0 || ix >= wd) continue;
                acc += (xi[xRow + ix] - xZp) * (wi[wRow + kx] - wz);
              }
            }
          }
          out[outBase + oy * ow + ox] = acc;
        }
      }
    }
  }
  return out;
}

List<int> _convOutSpatial(Tensor x, Tensor w, List<int> strides, List<int> pads,
    List<int> dilations) {
  final out = <int>[];
  for (int a = 0; a < 2; a++) {
    final window = dilations[a] * (w.shape[2 + a] - 1) + 1;
    out.add(
        (x.shape[2 + a] + pads[a] + pads[2 + a] - window) ~/ strides[a] + 1);
  }
  return out;
}

/// `ConvInteger` (1-D or 2-D): int32 accumulator output. 1-D convs run as
/// 2-D with a singleton height.
Tensor opConvInteger(Tensor x, Tensor w, Tensor? xZp, Tensor? wZp,
    {List<int>? strides,
    List<int>? pads,
    List<int>? dilations,
    int group = 1}) {
  if (x.rank == 3) {
    final y = opConvInteger(
      x.reshape([x.shape[0], x.shape[1], 1, x.shape[2]]),
      w.reshape([w.shape[0], w.shape[1], 1, w.shape[2]]),
      xZp,
      wZp,
      strides: [1, strides?.first ?? 1],
      pads: pads == null ? null : [0, pads[0], 0, pads[1]],
      dilations: [1, dilations?.first ?? 1],
      group: group,
    );
    return y.reshape([y.shape[0], y.shape[1], y.shape[3]]);
  }
  assert(x.rank == 4, 'ConvInteger implemented for 1-D/2-D convs');
  if (wZp != null && wZp.length != 1 && wZp.length != w.shape[0]) {
    throw UnsupportedError('ConvInteger: weight zero point must be scalar '
        'or per-output-channel');
  }
  final s = strides ?? const [1, 1];
  final p = pads ?? const [0, 0, 0, 0];
  final d = dilations ?? const [1, 1];
  final outSp = _convOutSpatial(x, w, s, p, d);
  final m = w.shape[0];
  final wz = wZp == null
      ? List.filled(m, 0)
      : [for (int c = 0; c < m; c++) wZp.getI(wZp.length == 1 ? 0 : c)];
  final acc = _convIntAcc(x, w, xZp?.getI(0) ?? 0, wz,
      strides: s, pads: p, dilations: d, group: group, outSp: outSp);
  return Tensor.int64(acc, [x.shape[0], m, ...outSp]);
}

/// `QLinearConv` (2-D): integer conv + int32 bias + per-channel requantize.
Tensor opQLinearConv(Tensor x, Tensor xS, Tensor? xZp, Tensor w, Tensor wS,
    Tensor? wZp, Tensor yS, Tensor? yZp, Tensor? bias,
    {List<int>? strides,
    List<int>? pads,
    List<int>? dilations,
    int group = 1,
    required int lo,
    required int hi}) {
  assert(x.rank == 4, 'QLinearConv implemented for 2-D convs');
  final s = strides ?? const [1, 1];
  final p = pads ?? const [0, 0, 0, 0];
  final d = dilations ?? const [1, 1];
  final outSp = _convOutSpatial(x, w, s, p, d);
  final m = w.shape[0];
  final wz = wZp == null
      ? List.filled(m, 0)
      : [for (int c = 0; c < m; c++) wZp.getI(wZp.length == 1 ? 0 : c)];
  final acc = _convIntAcc(x, w, xZp?.getI(0) ?? 0, wz,
      strides: s, pads: p, dilations: d, group: group, outSp: outSp);

  final n = x.shape[0];
  final spN = outSp[0] * outSp[1];
  final xs = xS.getD(0), ys = yS.getD(0);
  final yz = yZp?.getI(0) ?? 0;
  final bi = bias?.asIntList();
  final out = Int64List(acc.length);
  for (int b = 0; b < n; b++) {
    for (int om = 0; om < m; om++) {
      final mult = xs * wS.getD(wS.length == 1 ? 0 : om) / ys;
      final bv = bi == null ? 0 : bi[om];
      final base = (b * m + om) * spN;
      for (int k = 0; k < spN; k++) {
        final q = _roundEven((acc[base + k] + bv) * mult) + yz;
        out[base + k] = q < lo ? lo : (q > hi ? hi : q);
      }
    }
  }
  return Tensor.int64(out, [n, m, ...outSp]);
}
