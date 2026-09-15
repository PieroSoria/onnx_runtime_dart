/// Standard ONNX transformer operators. Microsoft variants live separately.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'tensor.dart';
import 'execution_control.dart';
import 'onnx_proto_loader.dart' show halfToFloat32Bits, float32ToHalfBits;

bool _same(List<int> a, List<int> b) =>
    a.length == b.length &&
    List.generate(a.length, (i) => a[i] == b[i]).every((v) => v);

void _float(Tensor t, String name) {
  if (!t.isFloat) throw ArgumentError('$name must be floating point');
}

/// Broadcast an input into a fixed output shape without materializing it.
List<int> _broadcastStrides(Tensor t, List<int> shape) {
  if (t.rank > shape.length) {
    throw ArgumentError('Invalid broadcast ${t.shape} to $shape');
  }
  final offset = shape.length - t.rank;
  final strides = List<int>.filled(shape.length, 0);
  for (var i = 0; i < t.rank; i++) {
    if (t.shape[i] != 1 && t.shape[i] != shape[offset + i]) {
      throw ArgumentError('Invalid broadcast ${t.shape} to $shape');
    }
    if (t.shape[i] != 1) strides[offset + i] = t.strides[i];
  }
  return strides;
}

/// FLOAT implementation of RMSNormalization-23, including suffix reduction.
Tensor standardRMSNormalization(
  Tensor x,
  Tensor scale, {
  int axis = -1,
  double epsilon = 1e-5,
  bool halfInput = false,
}) {
  _float(x, 'X');
  _float(scale, 'scale');
  final ax = axis < 0 ? axis + x.rank : axis;
  if (ax < 0 || ax >= x.rank) {
    throw ArgumentError('RMSNormalization: invalid axis $axis');
  }
  final strides = _broadcastStrides(scale, x.shape);
  final width = x.shape.sublist(ax).fold(1, (int a, int b) => a * b);
  activeOnnxControl?.checkAdditionalBytes(x.length * 4);
  final out = Float32List(x.length);
  final rounded = Float32List(1);
  final bits = Uint32List.view(rounded.buffer);
  double fp32(double value) {
    rounded[0] = value;
    return rounded[0];
  }

  if (width == 0) return Tensor.float(out, x.shape);
  for (var base = 0; base < x.length; base += width) {
    activeOnnxControl?.checkpoint();
    double sum = 0;
    for (var j = 0; j < width; j++) {
      if ((j & 4095) == 0) activeOnnxControl?.checkpoint();
      final v = x.f![base + j];
      sum += fp32(v * v);
    }
    final rms = fp32(math.sqrt(fp32(fp32(sum / width) + epsilon)));
    for (var j = 0; j < width; j++) {
      var index = base + j, s = 0;
      for (var d = x.rank - 1; d >= 0; d--) {
        s += (index % x.shape[d]) * strides[d];
        index ~/= x.shape[d];
      }
      var normalized = fp32(x.f![base + j] / rms);
      if (halfInput) {
        bits[0] = halfToFloat32Bits(float32ToHalfBits(bits[0]));
        normalized = rounded[0];
      }
      out[base + j] = normalized * scale.f![s];
    }
  }
  return Tensor.float(out, x.shape);
}

/// RotaryEmbedding-23 input order and optional pre-indexed caches.
Tensor standardRotaryEmbedding(
  Tensor x,
  Tensor cos,
  Tensor sin,
  Tensor? positions, {
  int numHeads = 0,
  int rotaryDim = 0,
  bool interleaved = false,
}) {
  _float(x, 'X');
  _float(cos, 'cos_cache');
  _float(sin, 'sin_cache');
  if (x.rank != 3 && x.rank != 4) {
    throw ArgumentError('RotaryEmbedding: X must have rank 3 or 4');
  }
  final batch = x.shape[0], seq = x.shape[x.rank == 3 ? 1 : 2];
  final heads = x.rank == 4 ? x.shape[1] : numHeads;
  if (heads <= 0 || (x.rank == 3 && x.shape[2] % heads != 0)) {
    throw ArgumentError('RotaryEmbedding: invalid num_heads');
  }
  final size = x.rank == 4 ? x.shape[3] : x.shape[2] ~/ heads;
  final dim = rotaryDim == 0 ? size : rotaryDim;
  if (size % 2 != 0 || dim <= 0 || dim > size || dim % 2 != 0) {
    throw ArgumentError('RotaryEmbedding: invalid rotary dimension $dim');
  }
  if (!_same(cos.shape, sin.shape) ||
      cos.rank != (positions == null ? 3 : 2) ||
      cos.shape.last != dim ~/ 2) {
    throw ArgumentError('RotaryEmbedding: invalid cosine/sine cache shapes');
  }
  if (positions == null) {
    if (cos.shape[0] != batch || cos.shape[1] != seq) {
      throw ArgumentError('RotaryEmbedding: cache batch/sequence mismatch');
    }
  } else if (positions.dtype != DType.int64 ||
      !_same(positions.shape, [batch, seq])) {
    throw ArgumentError(
      'RotaryEmbedding: position_ids must be int64 [batch, sequence]',
    );
  }
  activeOnnxControl?.checkAdditionalBytes(x.length * 4);
  final out = Float32List.fromList(x.f!);
  for (var b = 0; b < batch; b++) {
    for (var s = 0; s < seq; s++) {
      activeOnnxControl?.checkpoint();
      final pos = positions == null ? b * seq + s : positions.getI(b * seq + s);
      if (pos < 0 || pos >= (positions == null ? batch * seq : cos.shape[0])) {
        throw ArgumentError('RotaryEmbedding: position out of range: $pos');
      }
      for (var h = 0; h < heads; h++) {
        final base = (x.rank == 4
                ? (b * heads + h) * seq + s
                : (b * seq + s) * heads + h) *
            size;
        for (var d = 0; d < dim ~/ 2; d++) {
          final i = base + (interleaved ? 2 * d : d);
          final j = base + (interleaved ? 2 * d + 1 : d + dim ~/ 2);
          final c = cos.getD(pos * (dim ~/ 2) + d),
              sn = sin.getD(pos * (dim ~/ 2) + d);
          out[i] = x.getD(i) * c - x.getD(j) * sn;
          out[j] = x.getD(i) * sn + x.getD(j) * c;
        }
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// FLOAT Attention-23. Scores use one row of scratch unless debug output is
/// requested. Past KV is read directly; present tensors are allocated only
/// when requested, and never mutate caller-owned inputs.
///
/// When [cacheK]/[cacheV] (and [cacheRows]) are provided, the kernel runs
/// against a caller-owned persistent cache: rows `[0, cacheRows)` are resident
/// in the caches and the past inputs are ignored, new rows are appended
/// in place (growing the caches if their capacity is exhausted), and the
/// returned present tensors are offset views over the (possibly new) cache
/// buffers. Successive decode steps then reuse one buffer per layer instead of
/// re-copying the whole growing history.
List<Tensor> standardAttention(
  Tensor q,
  Tensor k,
  Tensor v, {
  Tensor? mask,
  Tensor? pastKey,
  Tensor? pastValue,
  int qHeads = 0,
  int kvHeads = 0,
  bool causal = false,
  double? scale,
  double softcap = 0,
  int debugMode = 0,
  bool present = false,
  bool debug = false,
  Float32List? cacheK,
  Float32List? cacheV,
  int cacheRows = 0,
  int leftWindow = -1,
  int rightWindow = -1,
  void Function()? checkpoint,
}) {
  for (final t in [q, k, v]) {
    _float(t, 'Q/K/V');
  }
  if ((q.rank != 3 && q.rank != 4) || k.rank != q.rank || v.rank != q.rank) {
    throw ArgumentError('Attention: Q/K/V must all have rank 3 or 4');
  }
  final three = q.rank == 3;
  final batch = q.shape[0],
      qs = q.shape[three ? 1 : 2],
      ks = k.shape[three ? 1 : 2];
  final nh = three ? qHeads : q.shape[1], nk = three ? kvHeads : k.shape[1];
  if (nh <= 0 ||
      nk <= 0 ||
      nh % nk != 0 ||
      k.shape[0] != batch ||
      v.shape[0] != batch ||
      v.shape[three ? 1 : 2] != ks ||
      (!three && v.shape[1] != nk)) {
    throw ArgumentError(
      'Attention: incompatible batches, heads or sequence lengths',
    );
  }
  if (three &&
      (q.shape[2] % nh != 0 || k.shape[2] % nk != 0 || v.shape[2] % nk != 0)) {
    throw ArgumentError('Attention: hidden size is not divisible by heads');
  }
  final size = three ? q.shape[2] ~/ nh : q.shape[3];
  final ksize = three ? k.shape[2] ~/ nk : k.shape[3];
  final vs = three ? v.shape[2] ~/ nk : v.shape[3];
  if (size <= 0 ||
      size != ksize ||
      vs <= 0 ||
      softcap < 0 ||
      debugMode < 0 ||
      debugMode > 3) {
    throw ArgumentError('Attention: invalid head size, softcap or debug mode');
  }
  if ((pastKey == null) != (pastValue == null)) {
    throw ArgumentError('Attention: past_key and past_value must be paired');
  }
  final cached = cacheK != null || cacheV != null;
  if (cached && (cacheK == null || cacheV == null || !present)) {
    throw ArgumentError(
        'Attention: persistent cache needs both caches and present=true');
  }
  var past = 0;
  if (pastKey != null) {
    _float(pastKey, 'past_key');
    _float(pastValue!, 'past_value');
    if (pastKey.rank != 4) {
      throw ArgumentError('Attention: past_key must have rank 4');
    }
    past = pastKey.shape[2];
    if (!_same(pastKey.shape, [batch, nk, past, size]) ||
        !_same(pastValue.shape, [batch, nk, past, vs])) {
      throw ArgumentError('Attention: invalid past cache shapes');
    }
  }
  final total = (cached ? cacheRows : past) + ks,
      factor = scale ?? 1 / math.sqrt(size);
  if (!factor.isFinite || factor < 0) {
    throw ArgumentError('Attention: scale must be finite and nonnegative');
  }
  final scoreShape = [batch, nh, qs, total];
  final residentRows = cached ? cacheRows : past;
  var kCache = cacheK, vCache = cacheV;
  var capK = kCache == null ? total : kCache.length ~/ (batch * nk * size);
  var capV = vCache == null ? total : vCache.length ~/ (batch * nk * vs);
  if (cached && (capK * batch * nk * size != kCache!.length ||
      capV * batch * nk * vs != vCache!.length)) {
    throw ArgumentError('Attention: cache buffers are not row-aligned');
  }
  if (cached && total > capK) {
    // Grow geometrically and carry the resident rows over.
    activeOnnxControl?.checkAdditionalBytes(
        4 * (batch * nk * (total - capK) * (size + vs)));
    final newCap = math.max(total, capK * 2);
    final nk_ = Float32List(batch * nk * newCap * size);
    final nv_ = Float32List(batch * nk * newCap * vs);
    for (var b = 0; b < batch; b++) {
      for (var h = 0; h < nk; h++) {
        for (var s = 0; s < capK; s++) {
          final src = (b * nk + h) * capK + s, dst = (b * nk + h) * newCap + s;
          for (var d = 0; d < size; d++) {
            nk_[(dst * size) + d] = kCache![(src * size) + d];
          }
          for (var d = 0; d < vs; d++) {
            nv_[(dst * vs) + d] = vCache![(src * vs) + d];
          }
        }
      }
    }
    kCache = nk_;
    vCache = nv_;
    capK = newCap;
  }
  final scoreBytes = 4 *
      (batch * nh * qs * vs +
          total +
          (present && !cached ? batch * nk * total * (size + vs) : 0) +
          (cached ? batch * nk * ks * (size + vs) : 0) +
          (debug ? batch * nh * qs * total : 0));
  activeOnnxControl?.checkAdditionalBytes(scoreBytes);
  checkpoint ??= activeOnnxControl?.checkpoint;
  final ms = mask == null ? null : _broadcastStrides(mask, scoreShape);
  final y = Float32List(batch * nh * qs * vs);
  final pk = present && !cached ? Float32List(batch * nk * total * size) : Float32List(0);
  final pv = present && !cached ? Float32List(batch * nk * total * vs) : Float32List(0);
  final dbg = debug ? Float32List(batch * nh * qs * total) : Float32List(0);
  final scores = Float32List(total);
  double kv(
    Tensor current,
    Tensor? cache,
    int b,
    int h,
    int s,
    int d,
    int width,
  ) {
    if (s < residentRows) {
      if (cached) {
        final cap = current == k ? capK : capV,
            buf = current == k ? kCache! : vCache!;
        return buf[((b * nk + h) * cap + s) * width + d];
      }
      return cache!.getD(((b * nk + h) * past + s) * width + d);
    }
    return current.getD(
            (three
                        ? (b * ks + s - residentRows) * nk + h
                        : (b * nk + h) * ks + s - residentRows) *
                    width +
                d);
  }

  if (present) {
    if (cached) {
      // Append only the new rows; resident history already lives in the cache,
      // so no copy of the growing prefix is made.
      for (var b = 0; b < batch; b++) {
        for (var h = 0; h < nk; h++) {
          for (var s = residentRows; s < total; s++) {
            checkpoint?.call();
            for (var d = 0; d < size; d++) {
              kCache![((b * nk + h) * capK + s) * size + d] =
                  kv(k, pastKey, b, h, s, d, size);
            }
            for (var d = 0; d < vs; d++) {
              vCache![((b * nk + h) * capV + s) * vs + d] =
                  kv(v, pastValue, b, h, s, d, vs);
            }
          }
        }
      }
    } else {
      for (var b = 0; b < batch; b++) {
        for (var h = 0; h < nk; h++) {
          for (var s = 0; s < total; s++) {
            checkpoint?.call();
            for (var d = 0; d < size; d++) {
              pk[((b * nk + h) * total + s) * size + d] =
                  kv(k, pastKey, b, h, s, d, size);
            }
            for (var d = 0; d < vs; d++) {
              pv[((b * nk + h) * total + s) * vs + d] =
                  kv(v, pastValue, b, h, s, d, vs);
            }
          }
        }
      }
    }
  }
  for (var b = 0; b < batch; b++) {
    for (var h = 0; h < nh; h++) {
      for (var i = 0; i < qs; i++) {
        checkpoint?.call();
        final kh = h ~/ (nh ~/ nk), db = ((b * nh + h) * qs + i) * total;
        var maxScore = double.negativeInfinity;
        for (var j = 0; j < total; j++) {
          if ((j & 255) == 0) checkpoint?.call();
          double dot = 0;
          for (var d = 0; d < size; d++) {
            dot += q.getD(
                  (three ? (b * qs + i) * nh + h : (b * nh + h) * qs + i) *
                          size +
                      d,
                ) *
                kv(k, pastKey, b, kh, j, d, size);
          }
          var score = dot * factor;
          if (debug && debugMode == 0) dbg[db + j] = score;
          if (softcap > 0) {
            final z = score / softcap;
            score = softcap *
                (z >= 0
                    ? (1 - math.exp(-2 * z)) / (1 + math.exp(-2 * z))
                    : (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1));
          }
          if (debug && debugMode == 1) dbg[db + j] = score;
          if (mask != null) {
            final m = mask.getD(b * ms![0] + h * ms[1] + i * ms[2] + j * ms[3]);
            score += mask.isFloat ? m : (m != 0 ? 0 : double.negativeInfinity);
          }
          final p = i + residentRows;
          if (causal && j > p) score = double.negativeInfinity;
          if (leftWindow >= 0 && j < p - leftWindow) {
            score = double.negativeInfinity;
          }
          if (rightWindow >= 0 && j > p + rightWindow) {
            score = double.negativeInfinity;
          }
          scores[j] = score;
          if (debug && debugMode == 2) dbg[db + j] = score;
          maxScore = math.max(maxScore, scores[j]);
        }
        double sum = 0;
        for (var j = 0; j < total; j++) {
          scores[j] = maxScore == double.negativeInfinity
              ? 0
              : math.exp(scores[j] - maxScore);
          sum += scores[j];
        }
        for (var j = 0; j < total; j++) {
          scores[j] = sum == 0 ? 0 : scores[j] / sum;
          if (debug && debugMode == 3) dbg[db + j] = scores[j];
        }
        for (var d = 0; d < vs; d++) {
          double value = 0;
          for (var j = 0; j < total; j++) {
            if (scores[j] != 0) {
              value += scores[j] * kv(v, pastValue, b, kh, j, d, vs);
            }
          }
          y[(three ? (b * qs + i) * nh + h : (b * nh + h) * qs + i) * vs + d] =
              value;
        }
      }
    }
  }
  return [
    Tensor.float(y, three ? [batch, qs, nh * vs] : [batch, nh, qs, vs]),
    cached
        ? Tensor.floatView(kCache!, 0, [batch, nk, total, size])
        : Tensor.float(pk, present ? [batch, nk, total, size] : [0]),
    cached
        ? Tensor.floatView(vCache!, 0, [batch, nk, total, vs])
        : Tensor.float(pv, present ? [batch, nk, total, vs] : [0]),
    Tensor.float(dbg, debug ? scoreShape : [0]),
  ];
}
