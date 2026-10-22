/// Implementations of the standard ONNX operators (per the public ONNX
/// operator specification, https://onnx.ai/onnx/operators/) used by
/// transformer embedding / reranking graphs. Mechanical execution only — each
/// op is run the same way any ONNX runtime would, with no knowledge of why the
/// graph is shaped the way it is.
library;

import 'dart:math' as math;
import 'dart:typed_data' hide Int64List;

import 'execution_control.dart';
import 'int64_storage.dart';

// SIMD GEMM kernel on native targets (VM/AOT, where Float32x4 maps to real
// SSE/NEON); portable scalar kernel on web targets where it's emulated.
import 'gemm_kernel_scalar.dart'
    if (dart.library.ffi) 'gemm_kernel_simd.dart'
    as gemm;
import 'onnx_proto_loader.dart' show float32ToHalfBits, halfToFloat32Bits;
import 'tensor.dart';

// ---------------------------------------------------------------------------
// Broadcasting helpers (numpy-style, as used throughout the ONNX spec)
// ---------------------------------------------------------------------------

List<int> _broadcastShape(List<int> a, List<int> b) {
  final rank = math.max(a.length, b.length);
  final ap = List<int>.filled(rank - a.length, 1, growable: true)..addAll(a);
  final bp = List<int>.filled(rank - b.length, 1, growable: true)..addAll(b);
  final out = List<int>.filled(rank, 1);
  for (int k = 0; k < rank; k++) {
    final ad = ap[k], bd = bp[k];
    if (ad == bd) {
      out[k] = ad;
    } else if (ad == 1) {
      out[k] = bd;
    } else if (bd == 1) {
      out[k] = ad;
    } else {
      throw ArgumentError('Cannot broadcast shapes $a and $b');
    }
  }
  return out;
}

List<int> _unflatten(int flat, List<int> shape) {
  final coords = List<int>.filled(shape.length, 0);
  int rem = flat;
  for (int k = shape.length - 1; k >= 0; k--) {
    final d = shape[k] == 0 ? 1 : shape[k];
    coords[k] = rem % d;
    rem ~/= d;
  }
  return coords;
}

int _flattenBroadcast(List<int> outCoords, List<int> srcShape) {
  final rank = outCoords.length;
  final srcRank = srcShape.length;
  int flat = 0;
  int stride = 1;
  for (int k = srcRank - 1; k >= 0; k--) {
    final outK = k + (rank - srcRank);
    final dim = srcShape[k];
    final coord = dim == 1 ? 0 : outCoords[outK];
    flat += coord * stride;
    stride *= dim;
  }
  return flat;
}

bool _shapeEq(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (int k = 0; k < a.length; k++) {
    if (a[k] != b[k]) return false;
  }
  return true;
}

enum _Arith { add, sub, mul, div }

/// True if [small] (leading 1-dims ignored) equals the trailing dims of [big]
/// — i.e. broadcasting [small] against [big] just tiles it along the leading
/// axes, so no per-element coordinate math is needed.
bool _isTrailingSuffix(List<int> small, List<int> big) {
  int s = 0;
  while (s < small.length && small[s] == 1) {
    s++;
  }
  final n = small.length - s;
  if (n > big.length) return false;
  for (int k = 0; k < n; k++) {
    if (small[small.length - 1 - k] != big[big.length - 1 - k]) return false;
  }
  return true;
}

/// Monomorphic float32 paths for Add/Sub/Mul/Div — the hottest elementwise
/// ops. Direct Float32List indexing (no per-element closure or dtype branch)
/// for the three layouts that cover essentially all transformer traffic:
/// same-shape, scalar, and suffix-tile (bias adds like `[B,T,N] + [N]`).
/// Returns null when the layout (or dtype) needs the general path.
Tensor? _arithFloatFast(Tensor a, Tensor b, _Arith op) {
  final af = a.f, bf = b.f;
  if (af == null || bf == null) return null;
  final an = a.length, bn = b.length;

  if (_shapeEq(a.shape, b.shape)) {
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] + bf[k];
        }
      case _Arith.sub:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] - bf[k];
        }
      case _Arith.mul:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] * bf[k];
        }
      case _Arith.div:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] / bf[k];
        }
    }
    return Tensor.float(out, a.shape);
  }

  if (bn == 1 && a.rank >= b.rank) {
    final s = bf[0];
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] + s;
        }
      case _Arith.sub:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] - s;
        }
      case _Arith.mul:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] * s;
        }
      case _Arith.div:
        for (int k = 0; k < an; k++) {
          out[k] = af[k] / s;
        }
    }
    return Tensor.float(out, a.shape);
  }
  if (an == 1 && b.rank >= a.rank) {
    final s = af[0];
    final out = Float32List(bn);
    switch (op) {
      case _Arith.add:
        for (int k = 0; k < bn; k++) {
          out[k] = s + bf[k];
        }
      case _Arith.sub:
        for (int k = 0; k < bn; k++) {
          out[k] = s - bf[k];
        }
      case _Arith.mul:
        for (int k = 0; k < bn; k++) {
          out[k] = s * bf[k];
        }
      case _Arith.div:
        for (int k = 0; k < bn; k++) {
          out[k] = s / bf[k];
        }
    }
    return Tensor.float(out, b.shape);
  }

  // Per-row scalar: b matches a except the last axis is 1 (LayerNorm's
  // `x - mean` / `x / std` pattern, [B,T,D] op [B,T,1]).
  if (a.rank >= 1 &&
      b.rank == a.rank &&
      b.shape[a.rank - 1] == 1 &&
      a.shape[a.rank - 1] > 1 &&
      _shapeEq(
        a.shape.sublist(0, a.rank - 1),
        b.shape.sublist(0, b.rank - 1),
      )) {
    final d = a.shape[a.rank - 1];
    final rows = bn; // = an ~/ d
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] + s;
          }
        }
      case _Arith.sub:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] - s;
          }
        }
      case _Arith.mul:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] * s;
          }
        }
      case _Arith.div:
        for (int r = 0; r < rows; r++) {
          final s = bf[r], base = r * d;
          for (int j = 0; j < d; j++) {
            out[base + j] = af[base + j] / s;
          }
        }
    }
    return Tensor.float(out, a.shape);
  }

  // Suffix tile, b smaller: out[off+j] = a[off+j] op b[j]. Output shape is
  // the broadcast shape, which here is a.shape (possibly with b's extra
  // leading 1-dims — only possible when b.rank <= a.rank, so require that).
  if (bn < an && b.rank <= a.rank && _isTrailingSuffix(b.shape, a.shape)) {
    final out = Float32List(an);
    switch (op) {
      case _Arith.add:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] + bf[j];
          }
        }
      case _Arith.sub:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] - bf[j];
          }
        }
      case _Arith.mul:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] * bf[j];
          }
        }
      case _Arith.div:
        for (int off = 0; off < an; off += bn) {
          for (int j = 0; j < bn; j++) {
            out[off + j] = af[off + j] / bf[j];
          }
        }
    }
    return Tensor.float(out, a.shape);
  }
  // Suffix tile, a smaller (kept separate: Sub/Div aren't commutative).
  if (an < bn && a.rank <= b.rank && _isTrailingSuffix(a.shape, b.shape)) {
    final out = Float32List(bn);
    switch (op) {
      case _Arith.add:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] + bf[off + j];
          }
        }
      case _Arith.sub:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] - bf[off + j];
          }
        }
      case _Arith.mul:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] * bf[off + j];
          }
        }
      case _Arith.div:
        for (int off = 0; off < bn; off += an) {
          for (int j = 0; j < an; j++) {
            out[off + j] = af[j] / bf[off + j];
          }
        }
    }
    return Tensor.float(out, b.shape);
  }

  return null;
}

Tensor _elementwiseBinary(
  Tensor a,
  Tensor b,
  double Function(double, double) op,
) {
  final bothInt = !a.isFloat && !b.isFloat;

  // Fast path: identical shapes (the common case for residual adds etc.) —
  // no coordinate decomposition needed, just walk both flat buffers in lockstep.
  if (_shapeEq(a.shape, b.shape)) {
    final n = a.length;
    if (bothInt) {
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = op(a.getD(k), b.getD(k)).round();
      }
      return Tensor.int64(out, a.shape);
    }
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(a.getD(k), b.getD(k));
    }
    return Tensor.float(out, a.shape);
  }

  // Fast path: b is a scalar (very common for bias/constant ops). Only valid
  // when a's rank is >= b's — otherwise the broadcast output shape is b's,
  // not a's (e.g. a: rank-0 `[]` vs b: `[1]` broadcasts to `[1]`, not `[]`).
  if (b.length == 1 && a.rank >= b.rank) {
    final n = a.length;
    final scalar = b.getD(0);
    if (bothInt) {
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = op(a.getD(k), scalar).round();
      }
      return Tensor.int64(out, a.shape);
    }
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(a.getD(k), scalar);
    }
    return Tensor.float(out, a.shape);
  }
  if (a.length == 1 && b.rank >= a.rank) {
    final n = b.length;
    final scalar = a.getD(0);
    if (bothInt) {
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = op(scalar, b.getD(k)).round();
      }
      return Tensor.int64(out, b.shape);
    }
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(scalar, b.getD(k));
    }
    return Tensor.float(out, b.shape);
  }

  // General broadcasting path: incremental odometer with per-axis source
  // strides (stride 0 on broadcast axes) — no per-element coordinate
  // decomposition.
  final outShape = _broadcastShape(a.shape, b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final rank = outShape.length;
  final aStridesFull = List<int>.filled(rank, 0);
  final bStridesFull = List<int>.filled(rank, 0);
  final aStrides = a.strides, bStrides = b.strides;
  for (int k = 0; k < a.rank; k++) {
    if (a.shape[k] != 1) aStridesFull[k + rank - a.rank] = aStrides[k];
  }
  for (int k = 0; k < b.rank; k++) {
    if (b.shape[k] != 1) bStridesFull[k + rank - b.rank] = bStrides[k];
  }
  final coords = List<int>.filled(rank, 0);
  int aOff = 0, bOff = 0;

  void advance() {
    for (int k = rank - 1; k >= 0; k--) {
      aOff += aStridesFull[k];
      bOff += bStridesFull[k];
      if (++coords[k] < outShape[k]) return;
      coords[k] = 0;
      aOff -= outShape[k] * aStridesFull[k];
      bOff -= outShape[k] * bStridesFull[k];
    }
  }

  if (bothInt) {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = op(a.getD(aOff), b.getD(bOff)).round();
      advance();
    }
    return Tensor.int64(out, outShape);
  }
  final out = Float32List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = op(a.getD(aOff), b.getD(bOff));
    advance();
  }
  return Tensor.float(out, outShape);
}

// ---------------------------------------------------------------------------
// Elementwise ops
// ---------------------------------------------------------------------------

Tensor opAdd(Tensor a, Tensor b) =>
    _arithFloatFast(a, b, _Arith.add) ??
    _elementwiseBinary(a, b, (x, y) => x + y);

/// Fused `Relu(Add(a, b))` used by residual CNN blocks.
///
/// Equal-shape float tensors are overwhelmingly the common case and can be
/// produced in one allocation and one pass. Other dtypes/broadcast shapes
/// retain the exact generic operator semantics.
Tensor opAddRelu(Tensor a, Tensor b) {
  final af = a.f, bf = b.f;
  if (af != null && bf != null && _shapeEq(a.shape, b.shape)) {
    final out = Float32List(af.length);
    for (int k = 0; k < af.length; k++) {
      final v = af[k] + bf[k];
      out[k] = v < 0.0 ? 0.0 : v;
    }
    return Tensor.float(out, a.shape);
  }
  final sum = opAdd(a, b);
  final sf = sum.f;
  if (sf != null) {
    for (int k = 0; k < sf.length; k++) {
      if (sf[k] < 0.0) sf[k] = 0.0;
    }
    return sum;
  }
  return opRelu(sum);
}

Tensor opAddReluInPlace(Tensor target, Tensor other) {
  final tf = target.f, of = other.f;
  if (tf == null || of == null || !_shapeEq(target.shape, other.shape)) {
    return opAddRelu(target, other);
  }
  for (int k = 0; k < tf.length; k++) {
    final v = tf[k] + of[k];
    tf[k] = v < 0.0 ? 0.0 : v;
  }
  return target;
}

Tensor opSub(Tensor a, Tensor b) =>
    _arithFloatFast(a, b, _Arith.sub) ??
    _elementwiseBinary(a, b, (x, y) => x - y);
Tensor opMul(Tensor a, Tensor b) =>
    _arithFloatFast(a, b, _Arith.mul) ??
    _elementwiseBinary(a, b, (x, y) => x * y);

/// `Mod` — elementwise modulo with broadcasting. [fmod] true = C `fmod`
/// (result takes the dividend's sign, `remainder`); false = integer/numpy mod
/// (result takes the divisor's sign). The NSF source generator in RVC's decoder
/// wraps sine phase with Mod.
Tensor opMod(Tensor a, Tensor b, bool fmod) =>
    _elementwiseBinary(a, b, fmod ? (x, y) => x.remainder(y) : (x, y) => x % y);
Tensor opDiv(Tensor a, Tensor b) {
  // Integer division truncates toward zero per the ONNX spec (the generic
  // path would round the float quotient — off by one on length arithmetic
  // like ceil-div chains).
  if (!a.isFloat && !b.isFloat) {
    if (_shapeEq(a.shape, b.shape) || (b.length == 1 && a.rank >= b.rank)) {
      final n = a.length;
      final ai = a.intData, bi = b.intData;
      final scalar = b.length == 1 ? bi[0] : 0;
      final out = Int64List(n);
      for (int k = 0; k < n; k++) {
        out[k] = ai[k] ~/ (b.length == 1 ? scalar : bi[k]);
      }
      return Tensor.int64(out, a.shape);
    }
    // General broadcast: divide via coordinates but truncate.
    final r = _elementwiseBinary(a, b, (x, y) => (x / y).truncateToDouble());
    return r;
  }
  return _arithFloatFast(a, b, _Arith.div) ??
      _elementwiseBinary(a, b, (x, y) => x / y);
}

Tensor opPow(Tensor a, Tensor b) {
  // Scalar exponent (LayerNorm's `(x-mean)^2`, sqrt-as-pow etc.) — direct
  // loop, and plain multiply for the ubiquitous square.
  if (a.f != null && b.length == 1 && a.rank >= b.rank) {
    final e = b.getD(0);
    final af = a.f!;
    final out = Float32List(af.length);
    if (e == 2.0) {
      for (int k = 0; k < af.length; k++) {
        out[k] = af[k] * af[k];
      }
    } else {
      for (int k = 0; k < af.length; k++) {
        out[k] = math.pow(af[k], e).toDouble();
      }
    }
    return Tensor.float(out, a.shape);
  }
  return _elementwiseBinary(a, b, (x, y) => math.pow(x, y).toDouble());
}

Tensor _elementwiseUnary(Tensor a, double Function(double) op) {
  // Hoist the length (Tensor.length is a computed getter — in the loop
  // condition it would re-fold the shape every iteration) and skip getD's
  // per-element dtype branch.
  final n = a.length;
  final out = Float32List(n);
  final af = a.f;
  if (af != null) {
    for (int k = 0; k < n; k++) {
      out[k] = op(af[k]);
    }
  } else {
    final ai = a.intData;
    for (int k = 0; k < n; k++) {
      out[k] = op(ai[k].toDouble());
    }
  }
  return Tensor.float(out, a.shape);
}

Tensor opSqrt(Tensor a) => _elementwiseUnary(a, (x) => math.sqrt(x));
Tensor opReciprocal(Tensor a) => _elementwiseUnary(a, (x) => 1.0 / x);
// Direct loop: Relu runs over every activation map in CNNs.
Tensor opRelu(Tensor a) {
  final af = a.f;
  if (af == null) return _elementwiseUnary(a, (x) => x < 0 ? 0.0 : x);
  final n = af.length;
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    final v = af[k];
    out[k] = v < 0 ? 0.0 : v;
  }
  return Tensor.float(out, a.shape);
}

Tensor opReluInPlace(Tensor a) {
  final af = a.f;
  if (af == null) return opRelu(a);
  for (int k = 0; k < af.length; k++) {
    if (af[k] < 0.0) af[k] = 0.0;
  }
  return a;
}

Tensor opLeakyRelu(Tensor a, double alpha) =>
    _elementwiseUnary(a, (x) => x < 0 ? alpha * x : x);
Tensor opElu(Tensor a, double alpha) =>
    _elementwiseUnary(a, (x) => x < 0 ? alpha * (math.exp(x) - 1) : x);
Tensor opHardSigmoid(Tensor a, double alpha, double beta) =>
    _elementwiseUnary(a, (x) => (alpha * x + beta).clamp(0.0, 1.0));

/// `HardSwish`: `x * hardSigmoid(x)` with the spec-fixed alpha=1/6, beta=0.5.
Tensor opHardSwish(Tensor a) =>
    _elementwiseUnary(a, (x) => x * (x / 6 + 0.5).clamp(0.0, 1.0));
Tensor opSoftplus(Tensor a) =>
    _elementwiseUnary(a, (x) => math.log(1 + math.exp(x)));
Tensor opAtan(Tensor a) => _elementwiseUnary(a, math.atan);
Tensor opAcos(Tensor a) => _elementwiseUnary(a, math.acos);
Tensor opAsin(Tensor a) => _elementwiseUnary(a, math.asin);
Tensor opTan(Tensor a) => _elementwiseUnary(a, math.tan);
Tensor opCosh(Tensor a) =>
    _elementwiseUnary(a, (x) => (math.exp(x) + math.exp(-x)) * 0.5);
Tensor opSinh(Tensor a) =>
    _elementwiseUnary(a, (x) => (math.exp(x) - math.exp(-x)) * 0.5);
Tensor opAcosh(Tensor a) =>
    _elementwiseUnary(a, (x) => math.log(x + math.sqrt(x * x - 1)));
Tensor opAsinh(Tensor a) =>
    _elementwiseUnary(a, (x) => math.log(x + math.sqrt(x * x + 1)));
Tensor opAtanh(Tensor a) =>
    _elementwiseUnary(a, (x) => 0.5 * math.log((1 + x) / (1 - x)));
Tensor opMish(Tensor a) => _elementwiseUnary(a, (x) {
  final sp = x > 20 ? x : math.log(1 + math.exp(x));
  return x * _tanh(sp);
});
Tensor opSoftsign(Tensor a) => _elementwiseUnary(a, (x) => x / (1 + x.abs()));
Tensor opCelu(Tensor a, double alpha) => _elementwiseUnary(
  a,
  (x) => math.max(0.0, x) + math.min(0.0, alpha * (math.exp(x / alpha) - 1)),
);
Tensor opSelu(Tensor a, double alpha, double gamma) => _elementwiseUnary(
  a,
  (x) => gamma * (x > 0 ? x : alpha * (math.exp(x) - 1)),
);
Tensor opShrink(Tensor a, double lambd, double bias) => _elementwiseUnary(
  a,
  (x) => x < -lambd ? x + bias : (x > lambd ? x - bias : 0.0),
);
Tensor opSwish(Tensor a, double alpha) =>
    _elementwiseUnary(a, (x) => x / (1 + math.exp(-alpha * x)));
Tensor opThresholdedRelu(Tensor a, double alpha) =>
    _elementwiseUnary(a, (x) => x > alpha ? x : 0.0);

Tensor opHardmax(Tensor x, int axis) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = x.shape[ax];
  final innerSize = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final outF = x.isFloat ? Float32List(x.length) : null;
  final outI = x.isFloat ? null : Int64List(x.length);
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      final base = outer * axisSize * innerSize + inner;
      var best = x.getD(base);
      var bestIdx = 0;
      for (int a = 1; a < axisSize; a++) {
        final v = x.getD(base + a * innerSize);
        if (v > best) {
          best = v;
          bestIdx = a;
        }
      }
      if (outF != null) {
        outF[base + bestIdx * innerSize] = 1;
      } else {
        outI![base + bestIdx * innerSize] = 1;
      }
    }
  }
  return x.isFloat
      ? Tensor.float(outF!, x.shape)
      : Tensor.int64(outI!, x.shape);
}

Tensor opLpNormalization(Tensor x, int axis, int p) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = x.shape[ax];
  final innerSize = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final out = Float32List(x.length);
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      final base = outer * axisSize * innerSize + inner;
      double norm = 0;
      for (int a = 0; a < axisSize; a++) {
        norm += math.pow(x.getD(base + a * innerSize).abs(), p).toDouble();
      }
      norm = math.pow(norm, 1.0 / p).toDouble();
      if (norm == 0) continue;
      for (int a = 0; a < axisSize; a++) {
        final idx = base + a * innerSize;
        out[idx] = x.getD(idx) / norm;
      }
    }
  }
  return Tensor.float(out, x.shape);
}

Tensor opMeanVarianceNormalization(Tensor x, List<int> axes) {
  final rank = x.shape.length;
  final ax = axes.map((a) => a < 0 ? a + rank : a).toSet();
  final counts = ax.fold<int>(1, (acc, a) => acc * x.shape[a]);
  final outShape = [
    for (int a = 0; a < rank; a++) ax.contains(a) ? 1 : x.shape[a],
  ];
  final outStrides = Tensor.filledFloat(outShape, 0).strides;
  final sums = Float64List(outShape.fold<int>(1, (a, b) => a * b));
  final squares = Float64List(sums.length);
  final coords = List<int>.filled(rank, 0);
  for (int idx = 0; idx < x.length; idx++) {
    var outFlat = 0;
    for (int a = 0; a < rank; a++) {
      if (!ax.contains(a)) outFlat += coords[a] * outStrides[a];
    }
    final v = x.getD(idx);
    sums[outFlat] += v;
    squares[outFlat] += v * v;
    for (int a = rank - 1; a >= 0; a--) {
      if (++coords[a] < x.shape[a]) break;
      coords[a] = 0;
    }
  }
  final out = Float32List(x.length);
  final outCoords = List<int>.filled(rank, 0);
  for (int idx = 0; idx < x.length; idx++) {
    var outFlat = 0;
    for (int a = 0; a < rank; a++) {
      if (!ax.contains(a)) outFlat += outCoords[a] * outStrides[a];
    }
    final mean = sums[outFlat] / counts;
    final variance = squares[outFlat] / counts - mean * mean;
    out[idx] = (x.getD(idx) - mean) / math.sqrt(variance + 1e-9);
    for (int a = rank - 1; a >= 0; a--) {
      if (++outCoords[a] < x.shape[a]) break;
      outCoords[a] = 0;
    }
  }
  return Tensor.float(out, x.shape);
}

Tensor opSign(Tensor a) => a.isFloat
    ? _elementwiseUnary(a, (x) => x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0))
    : _unaryInt(a, (x) => x > 0 ? 1 : (x < 0 ? -1 : 0));
Tensor opFloor(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => x.floorToDouble()) : a;
Tensor opCeil(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => x.ceilToDouble()) : a;

/// `Round` — half to even, per the ONNX spec.
Tensor opRound(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => _roundEven(x).toDouble()) : a;

/// `Gelu`: exact (erf) form, or the tanh approximation when
/// `approximate="tanh"`.
Tensor opGelu(Tensor a, {bool tanhApprox = false}) => tanhApprox
    ? _elementwiseUnary(a, (x) {
        const c = 0.7978845608028654; // sqrt(2/pi)
        return 0.5 * x * (1 + _tanh(c * (x + 0.044715 * x * x * x)));
      })
    : _elementwiseUnary(a, (x) => 0.5 * x * (1 + _erf(x / math.sqrt2)));

/// `PRelu`: `x < 0 ? slope * x : x`, with [slope] broadcast onto [x].
Tensor opPRelu(Tensor x, Tensor slope) =>
    _elementwiseBinary(x, slope, (v, s) => v < 0 ? s * v : v);

// Erf via the Abramowitz & Stegun 7.1.26 approximation (public numerical
// analysis formula, max abs error ~1.5e-7) — needed for the GELU activation
// (gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))), which the graph expresses directly
// as an Erf node rather than a fused GELU op.
double _erf(double x) {
  final sign = x < 0 ? -1.0 : 1.0;
  x = x.abs();
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  final t = 1.0 / (1.0 + p * x);
  final y =
      1.0 -
      (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x);
  return sign * y;
}

// Direct loop rather than _elementwiseUnary: Erf is hot (GELU runs it over
// the whole FFN activation) and the per-element closure call costs as much
// as the math at this size.
Tensor opErf(Tensor a) {
  final af = a.f;
  if (af == null) return _elementwiseUnary(a, _erf);
  final out = Float32List(af.length);
  for (int k = 0; k < af.length; k++) {
    out[k] = _erf(af[k]);
  }
  return Tensor.float(out, a.shape);
}

Tensor opClip(Tensor x, Tensor? min, Tensor? max) {
  final lo = min != null ? min.getD(0) : double.negativeInfinity;
  final hi = max != null ? max.getD(0) : double.infinity;
  // Preserve integer dtype: Clip runs on int64 shape components (e.g. the
  // VITS/RVC relative-attention padding builds `Concat(Unsqueeze(Clip(len-1)),
  // ...)` over int64s) — a float result would break the downstream int64 Concat
  // (`intData` null). Float inputs keep the fast float path.
  if (!x.isFloat) {
    final ai = x.intData;
    final out = Int64List(ai.length);
    for (var k = 0; k < ai.length; k++) {
      out[k] = ai[k].toDouble().clamp(lo, hi).toInt();
    }
    return Tensor.int64(out, x.shape);
  }
  return _elementwiseUnary(x, (v) => v.clamp(lo, hi));
}

// ---------------------------------------------------------------------------
// Cast
// ---------------------------------------------------------------------------

Tensor opCast(Tensor x, int to) {
  // ONNX TensorProto.DataType: 1=FLOAT, 6=INT32, 7=INT64, 9=BOOL,
  // 10=FLOAT16, 11=DOUBLE. int32/int64 share our int64 tensor, bool is
  // int64 0/1, double is carried as float32.
  if (to == 9) {
    final n = x.length;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = x.getD(k) != 0 ? 1 : 0;
    }
    return Tensor.int64(out, x.shape);
  }
  if (to == 6 || to == 7) return Tensor.int64(x.asIntList(), x.shape);
  if (to == 10) {
    // fp16: values must round through half precision (fp16-casting exports
    // depend on it; treating this as identity drifts vs ORT), even though
    // the result is carried in a float32 tensor.
    final src = x.asFloatList();
    final out = Float32List(src.length);
    final bits = out.buffer.asUint32List();
    final srcBits = src.buffer.asUint32List(src.offsetInBytes, src.length);
    for (int k = 0; k < src.length; k++) {
      bits[k] = halfToFloat32Bits(float32ToHalfBits(srcBits[k]));
    }
    return Tensor.float(out, x.shape);
  }
  return Tensor.float(x.asFloatList(), x.shape);
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

/// `Shape(x)` — the dimensions of [x] as an int64 vector. The optional
/// [start] / [end] (opset 15+) return a slice of the shape, e.g.
/// `Shape(x, start: 1)` drops the batch dim.
Tensor opShape(Tensor x, {int? start, int? end}) {
  final rank = x.shape.length;
  var s = start ?? 0;
  var e = end ?? rank;
  if (s < 0) s += rank;
  if (e < 0) e += rank;
  s = s.clamp(0, rank);
  e = e.clamp(0, rank);
  final slice = e > s ? x.shape.sublist(s, e) : <int>[];
  return Tensor.int64(Int64List.fromList(slice), [slice.length]);
}

Tensor opReshape(Tensor x, Tensor shapeT) => x.reshape(shapeT.asIntList());

Tensor opTranspose(Tensor x, List<int> perm) {
  final newShape = [for (final p in perm) x.shape[p]];
  final n = x.length;
  final oldStrides = x.strides;
  final rank = perm.length;

  // Permuted source strides in output-axis order, so walking the output in
  // order just adds permStrides[k] when output coordinate k increments.
  final permStrides = [for (final p in perm) oldStrides[p]];

  // Fast path: the last axis stays last (e.g. attention's [B,T,H,D] ->
  // [B,H,T,D]) — both source and destination rows of length `last` are
  // contiguous, so copy row-blocks instead of single elements.
  if (x.isFloat && rank >= 2 && perm[rank - 1] == rank - 1) {
    final last = newShape[rank - 1];
    final out = Float32List(n);
    final xf = x.f!;
    final coords = List<int>.filled(rank - 1, 0);
    int srcOff = 0;
    for (int dst = 0; dst < n; dst += last) {
      out.setRange(dst, dst + last, xf, srcOff);
      for (int k = rank - 2; k >= 0; k--) {
        srcOff += permStrides[k];
        if (++coords[k] < newShape[k]) break;
        coords[k] = 0;
        srcOff -= newShape[k] * permStrides[k];
      }
    }
    return Tensor.float(out, newShape);
  }

  // General path: incremental coordinate walk, no per-element allocation.
  final coords = List<int>.filled(rank, 0);
  int srcOff = 0;
  if (x.isFloat) {
    final out = Float32List(n);
    final xf = x.f!;
    for (int idx = 0; idx < n; idx++) {
      out[idx] = xf[srcOff];
      for (int k = rank - 1; k >= 0; k--) {
        srcOff += permStrides[k];
        if (++coords[k] < newShape[k]) break;
        coords[k] = 0;
        srcOff -= newShape[k] * permStrides[k];
      }
    }
    return Tensor.float(out, newShape);
  }
  final out = Int64List(n);
  final xi = x.intData;
  for (int idx = 0; idx < n; idx++) {
    out[idx] = xi[srcOff];
    for (int k = rank - 1; k >= 0; k--) {
      srcOff += permStrides[k];
      if (++coords[k] < newShape[k]) break;
      coords[k] = 0;
      srcOff -= newShape[k] * permStrides[k];
    }
  }
  return Tensor.int64(out, newShape);
}

Tensor opSqueeze(Tensor x, List<int>? axes) {
  final ax =
      (axes ??
              [
                for (int k = 0; k < x.shape.length; k++)
                  if (x.shape[k] == 1) k,
              ])
          .map((a) => a < 0 ? a + x.shape.length : a)
          .toSet();
  final newShape = [
    for (int k = 0; k < x.shape.length; k++)
      if (!ax.contains(k)) x.shape[k],
  ];
  return x.reshape(newShape);
}

Tensor opUnsqueeze(Tensor x, List<int> axes) {
  final outRank = x.shape.length + axes.length;
  final normAxes = axes.map((a) => a < 0 ? a + outRank : a).toSet();
  final newShape = <int>[];
  int srcIdx = 0;
  for (int k = 0; k < outRank; k++) {
    if (normAxes.contains(k)) {
      newShape.add(1);
    } else {
      newShape.add(x.shape[srcIdx++]);
    }
  }
  return x.reshape(newShape);
}

Tensor opConcat(List<Tensor> inputs, int axis) {
  final rank = inputs.first.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outShape = List<int>.from(inputs.first.shape);
  outShape[ax] = inputs.fold<int>(0, (sum, t) => sum + t.shape[ax]);

  if (inputs.first is StringTensor) {
    final outerSize = outShape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
    final innerSize = outShape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
    final out = <String>[];
    for (int outer = 0; outer < outerSize; outer++) {
      for (final t in inputs) {
        final values = (t as StringTensor).strings;
        final blockSize = t.shape[ax] * innerSize;
        final srcStart = outer * blockSize;
        out.addAll(values.sublist(srcStart, srcStart + blockSize));
      }
    }
    return StringTensor(out, outShape);
  }

  final isFloat = inputs.first.isFloat;
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  final outerSize = outShape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final innerSize = outShape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  for (int outer = 0; outer < outerSize; outer++) {
    int axOffset = 0;
    for (final t in inputs) {
      final tAx = t.shape[ax];
      final blockSize = tAx * innerSize;
      final srcStart = outer * blockSize;
      final dstStart = outer * outShape[ax] * innerSize + axOffset * innerSize;
      for (int k = 0; k < blockSize; k++) {
        if (isFloat) {
          outF![dstStart + k] = t.f![srcStart + k];
        } else {
          outI![dstStart + k] = t.intData[srcStart + k];
        }
      }
      axOffset += tAx;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Gather / GatherND
// ---------------------------------------------------------------------------

Tensor opGather(Tensor data, Tensor indices, int axis) {
  final rank = data.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final idxShape = indices.shape;
  final outShape = [
    ...data.shape.sublist(0, ax),
    ...idxShape,
    ...data.shape.sublist(ax + 1),
  ];

  final outerSize = data.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = data.shape[ax];
  final innerSize = data.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final idxCount = idxShape.fold<int>(1, (a, b) => a * b);

  final n = outShape.fold<int>(1, (a, b) => a * b);
  final isFloat = data.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  int outPos = 0;
  for (int outer = 0; outer < outerSize; outer++) {
    for (int ii = 0; ii < idxCount; ii++) {
      int idx = indices.getI(ii);
      if (idx < 0) idx += axisSize;
      final srcStart = outer * axisSize * innerSize + idx * innerSize;
      for (int k = 0; k < innerSize; k++) {
        if (isFloat) {
          outF![outPos + k] = data.f![srcStart + k];
        } else {
          outI![outPos + k] = data.intData[srcStart + k];
        }
      }
      outPos += innerSize;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

Tensor opGatherND(Tensor data, Tensor indices, int batchDims) {
  // batch_dims == 0 is the only case this graph uses.
  assert(batchDims == 0);
  final idxRank = indices.shape.length;
  final k = indices.shape[idxRank - 1];
  final batchIdxShape = indices.shape.sublist(0, idxRank - 1);
  final remainingDataShape = data.shape.sublist(k);
  final outShape = [...batchIdxShape, ...remainingDataShape];

  final dataStrides = data.strides;
  final numIdxTuples = batchIdxShape.fold<int>(1, (a, b) => a * b);
  final innerSize = remainingDataShape.fold<int>(1, (a, b) => a * b);

  final isFloat = data.isFloat;
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  for (int t = 0; t < numIdxTuples; t++) {
    int dataFlatStart = 0;
    for (int d = 0; d < k; d++) {
      final idxVal = indices.getI(t * k + d);
      dataFlatStart += idxVal * dataStrides[d];
    }
    for (int j = 0; j < innerSize; j++) {
      if (isFloat) {
        outF![t * innerSize + j] = data.f![dataFlatStart + j];
      } else {
        outI![t * innerSize + j] = data.intData[dataFlatStart + j];
      }
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Expand / Slice
// ---------------------------------------------------------------------------

Tensor opExpand(Tensor x, Tensor shapeT) {
  final targetShape = shapeT.asIntList().toList();
  final outShape = _broadcastShape(x.shape, targetShape);
  final n = outShape.fold<int>(1, (a, b) => a * b);
  if (x.isFloat) {
    final out = Float32List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = x.f![_flattenBroadcast(_unflatten(idx, outShape), x.shape)];
    }
    return Tensor.float(out, outShape);
  } else {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] =
          x.intData[_flattenBroadcast(_unflatten(idx, outShape), x.shape)];
    }
    return Tensor.int64(out, outShape);
  }
}

Tensor opSlice(
  Tensor x,
  List<int> starts,
  List<int> ends,
  List<int>? axes,
  List<int>? steps,
) {
  final rank = x.shape.length;
  final ax = axes ?? List<int>.generate(starts.length, (k) => k);
  final st = steps ?? List<int>.filled(starts.length, 1);

  final normStart = List<int>.from(x.shape.map((d) => 0));
  final normEnd = List<int>.from(x.shape);
  final normStep = List<int>.filled(rank, 1);

  for (int j = 0; j < ax.length; j++) {
    int a = ax[j];
    if (a < 0) a += rank;
    final dim = x.shape[a];
    int s = starts[j];
    int e = ends[j];
    final step = st[j];
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = s.clamp(0, step < 0 ? dim - 1 : dim);
    e = e.clamp(step < 0 ? -1 : 0, dim);
    normStart[a] = s;
    normEnd[a] = e;
    normStep[a] = step;
  }

  final outShape = <int>[];
  for (int a = 0; a < rank; a++) {
    final cnt = normStep[a] > 0
        ? math.max(
            0,
            (normEnd[a] - normStart[a] + normStep[a] - 1) ~/ normStep[a],
          )
        : math.max(
            0,
            (normStart[a] - normEnd[a] - normStep[a] - 1) ~/ (-normStep[a]),
          );
    outShape.add(cnt);
  }

  final n = outShape.fold<int>(1, (a, b) => a * b);
  final srcStrides = x.strides;

  if (x is StringTensor) {
    final out = List<String>.filled(n, '');
    for (int idx = 0; idx < n; idx++) {
      final outCoords = _unflatten(idx, outShape);
      int srcFlat = 0;
      for (int a = 0; a < rank; a++) {
        final coord = normStart[a] + outCoords[a] * normStep[a];
        srcFlat += coord * srcStrides[a];
      }
      out[idx] = x.strings[srcFlat];
    }
    return StringTensor(out, outShape);
  }

  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  for (int idx = 0; idx < n; idx++) {
    final outCoords = _unflatten(idx, outShape);
    int srcFlat = 0;
    for (int a = 0; a < rank; a++) {
      final coord = normStart[a] + outCoords[a] * normStep[a];
      srcFlat += coord * srcStrides[a];
    }
    if (isFloat) {
      outF![idx] = x.f![srcFlat];
    } else {
      outI![idx] = x.intData[srcFlat];
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

Tensor opReduceMean(Tensor x, List<int>? axes, bool keepdims) {
  final rank = x.shape.length;
  final ax = (axes ?? List<int>.generate(rank, (k) => k))
      .map((a) => a < 0 ? a + rank : a)
      .toSet();

  // Fast path: reducing a contiguous trailing block of axes. This covers both
  // LayerNorm's last-axis mean and CNN global-average pooling over NCHW's
  // spatial axes: each output is the mean of one contiguous row.
  final firstReducedAxis = rank - ax.length;
  final reducesTrailingBlock =
      ax.isNotEmpty &&
      firstReducedAxis >= 0 &&
      List.generate(ax.length, (k) => firstReducedAxis + k).every(ax.contains);
  if (x.isFloat && reducesTrailingBlock) {
    final d = x.shape
        .sublist(firstReducedAxis)
        .fold<int>(1, (product, dimension) => product * dimension);
    final rows = d == 0 ? 0 : x.length ~/ d;
    final out = Float32List(rows);
    final xf = x.f!;
    for (int r = 0; r < rows; r++) {
      final base = r * d;
      double sum = 0;
      for (int k = 0; k < d; k++) {
        sum += xf[base + k];
      }
      out[r] = sum / d;
    }
    final shape = keepdims
        ? [
            ...x.shape.sublist(0, firstReducedAxis),
            ...List.filled(ax.length, 1),
          ]
        : x.shape.sublist(0, firstReducedAxis);
    return Tensor.float(out, shape);
  }

  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k],
  ];
  final reducedCount = ax.fold<int>(1, (acc, k) => acc * x.shape[k]);

  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final sums = Float64List(n);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;

  for (int idx = 0; idx < x.length; idx++) {
    final coords = _unflatten(idx, x.shape);
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      final c = ax.contains(k) ? 0 : coords[k];
      outFlat += c * outStridesFull[k];
    }
    sums[outFlat] += x.getD(idx);
  }
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    out[k] = sums[k] / reducedCount;
  }

  if (keepdims) return Tensor.float(out, outShapeFull);
  final squeezedShape = [
    for (int k = 0; k < rank; k++)
      if (!ax.contains(k)) x.shape[k],
  ];
  return Tensor.float(out, squeezedShape);
}

/// `ReduceProd` — product over the given axes.
Tensor opReduceProd(Tensor x, List<int>? axes, bool keepdims) {
  final rank = x.shape.length;
  final ax = (axes ?? List<int>.generate(rank, (k) => k))
      .map((a) => a < 0 ? a + rank : a)
      .toSet();
  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k],
  ];
  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final prods = Float64List(n)..fillRange(0, n, 1);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;
  final coords = List<int>.filled(rank, 0);
  for (int idx = 0; idx < x.length; idx++) {
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      if (!ax.contains(k)) outFlat += coords[k] * outStridesFull[k];
    }
    prods[outFlat] *= x.getD(idx);
    for (int k = rank - 1; k >= 0; k--) {
      if (++coords[k] < x.shape[k]) break;
      coords[k] = 0;
    }
  }
  final shape = keepdims
      ? outShapeFull
      : [
          for (int k = 0; k < rank; k++)
            if (!ax.contains(k)) x.shape[k],
        ];
  if (x.isFloat) {
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = prods[k];
    }
    return Tensor.float(out, shape);
  }
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = prods[k].toInt();
  }
  return Tensor.int64(out, shape);
}

/// `ReduceSumSquare` — sum of squares over the given axes.
Tensor opReduceSumSquare(Tensor x, List<int>? axes, bool keepdims) {
  final squared = opMul(x, x);
  return opReduceSum(squared, axes, keepdims);
}

/// Process-global injection for the stochastic ops ([opRandomNormalFill] /
/// [opRandomUniform]). When [provider] is set, each such op calls it with its op
/// name + output shape; a non-null `Float32List` **of the matching flat length**
/// is used verbatim instead of the default fill, so a determinism harness can
/// feed the exact buffer a reference used — e.g. RVC's Site-B SineGen additive
/// noise. Routing is by length: give the provider the one buffer sized to the
/// target node and other Random nodes (a different length) fall through to their
/// default. Null return / length-mismatch → the default (seeded uniform,
/// mean-fill normal). Set it around ONE run and clear it after (see rvc.dart).
class OnnxRandomInject {
  OnnxRandomInject._();

  static Float32List? Function(String op, List<int> shape)? provider;

  /// The injected buffer for [op] at [shape], or null to use the default fill.
  static Float32List? draw(String op, List<int> shape) {
    final p = provider;
    if (p == null) return null;
    final buf = p(op, shape);
    if (buf == null) return null;
    final n = shape.fold<int>(1, (a, b) => a * b);
    return buf.length == n ? buf : null;
  }
}

/// `RandomUniform` — a [shape]-sized tensor of uniform `[low, high)` samples.
/// Seeded (deterministic) so a run is reproducible (pooled == sync); the NSF
/// source generator in RVC's decoder draws a little phase noise from it. Won't
/// bit-match onnxruntime's RNG, but the sine synthesis dominates the output.
/// An [OnnxRandomInject] provider overrides the draw for a determinism harness.
Tensor opRandomUniform(List<int> shape, double low, double high, int? seed) {
  final inj = OnnxRandomInject.draw('RandomUniform', shape);
  if (inj != null) return Tensor.float(inj, shape);
  final n = shape.fold<int>(1, (a, b) => a * b);
  final rng = math.Random(seed ?? 1234567);
  final out = Float32List(n);
  final span = high - low;
  for (var k = 0; k < n; k++) {
    out[k] = low + span * rng.nextDouble();
  }
  return Tensor.float(out, shape);
}

Tensor opRandomUniformLike(Tensor x, double low, double high, int? seed) =>
    opRandomUniform(x.shape, low, high, seed);

Tensor opBernoulli(Tensor x, {int? seed, int? dtype}) {
  final n = x.length;
  final rng = math.Random(seed ?? 1234567);
  final asFloat = dtype == null || dtype == 1 || dtype == 10 || dtype == 11;
  if (asFloat) {
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = rng.nextDouble() < x.getD(k) ? 1 : 0;
    }
    return Tensor.float(out, x.shape);
  }
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = rng.nextDouble() < x.getD(k) ? 1 : 0;
  }
  return Tensor.int64(out, x.shape);
}

Tensor opMultinomial(Tensor x, int sampleSize, {int? seed}) {
  if (x.rank != 2) {
    throw ArgumentError('Multinomial requires a rank-2 input, got ${x.shape}');
  }
  final batch = x.shape[0], classes = x.shape[1];
  final rng = math.Random(seed ?? 1234567);
  final out = Int64List(batch * sampleSize);
  final cumulative = Float64List(classes);
  for (int b = 0; b < batch; b++) {
    final base = b * classes;
    double total = 0;
    for (int c = 0; c < classes; c++) {
      total += x.getD(base + c);
      cumulative[c] = total;
    }
    if (total <= 0) {
      throw ArgumentError('Multinomial input probabilities must sum > 0');
    }
    for (int s = 0; s < sampleSize; s++) {
      final r = rng.nextDouble() * total;
      var chosen = classes - 1;
      for (int c = 0; c < classes; c++) {
        if (r < cumulative[c]) {
          chosen = c;
          break;
        }
      }
      out[b * sampleSize + s] = chosen;
    }
  }
  return Tensor.int64(out, [batch, sampleSize]);
}

/// `ReduceL2` — the L2 norm `sqrt(sum(x^2))` over the given axes (VITS/RVC flow
/// weight-norm uses it heavily).
Tensor opReduceL2(Tensor x, List<int>? axes, bool keepdims) {
  final ss = opReduceSumSquare(x, axes, keepdims);
  final f = ss.f!;
  final out = Float32List(f.length);
  for (var k = 0; k < f.length; k++) {
    out[k] = math.sqrt(f[k]);
  }
  return Tensor.float(out, ss.shape);
}

/// `Split` — slices [x] along [axis] into [numOutputs] parts, sized by
/// [splitSizes] when given, else evenly (ceil-divided, last part smaller —
/// the opset-18 `num_outputs` rule, which reduces to equal parts when the
/// axis divides evenly).
List<Tensor> opSplit(
  Tensor x,
  int axis,
  int numOutputs, [
  List<int>? splitSizes,
]) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final dim = x.shape[ax];
  final sizes =
      splitSizes ??
      () {
        final chunk = (dim + numOutputs - 1) ~/ numOutputs;
        return [
          for (int i = 0; i < numOutputs; i++)
            i < numOutputs - 1 ? chunk : dim - chunk * (numOutputs - 1),
        ];
      }();
  final outs = <Tensor>[];
  int start = 0;
  for (final size in sizes) {
    outs.add(opSlice(x, [start], [start + size], [ax], null));
    start += size;
  }
  return outs;
}

/// GLU gate — the fused form of `Split`-in-half + `Sigmoid(second)` +
/// `Mul(first, ·)`: splits [x] in half along [axis] into `(a, b)` and returns
/// `a * sigmoid(b)` in one pass (one memory read/write instead of three).
/// Bitwise identical to the decomposition (same `1/(1+exp(-x))` sigmoid).
Tensor opGlu(Tensor x, int axis) {
  final rank = x.rank;
  final ax = axis < 0 ? axis + rank : axis;
  final axisSize = x.shape[ax];
  final half = axisSize ~/ 2;
  final inner = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final outer = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final xf = x.f ?? x.asFloatList();
  final out = Float32List(outer * half * inner);
  for (int o = 0; o < outer; o++) {
    for (int h = 0; h < half; h++) {
      final aBase = (o * axisSize + h) * inner;
      final bBase = (o * axisSize + half + h) * inner;
      final oBase = (o * half + h) * inner;
      for (int i = 0; i < inner; i++) {
        // s = Sigmoid(b) then a * s — matches the decomposed op bit-for-bit.
        final s = 1.0 / (1.0 + math.exp(-xf[bBase + i]));
        out[oBase + i] = xf[aBase + i] * s;
      }
    }
  }
  return Tensor.float(out, [...x.shape]..[ax] = half);
}

/// `STFT` (opset 17) — real input only, frame-major output
/// `[batch, frames, bins, 2]` (re/im). No centering or padding: frames start
/// at multiples of [frameStep] and must fit entirely inside the signal (the
/// ONNX graph does its own `Pad` beforehand). [onesided] keeps
/// `dftSize/2 + 1` bins. Odd or non-power-of-2 frame lengths are fine — bins
/// come from a direct DFT over a single-period twiddle table.
Tensor opSTFT(
  Tensor signal,
  int frameStep,
  Tensor? window,
  int? frameLength, {
  bool onesided = true,
}) {
  // Signal is [batch, len] or [batch, len, 1].
  assert(
    signal.rank == 2 || (signal.rank == 3 && signal.shape[2] == 1),
    'STFT supports real signals only',
  );
  final batch = signal.shape[0], len = signal.shape[1];
  final n = frameLength ?? window!.length;
  final win = window?.asFloatList();
  final frames = (len - n) ~/ frameStep + 1;
  final bins = onesided ? n ~/ 2 + 1 : n;

  final cosT = Float64List(n), sinT = Float64List(n);
  for (int i = 0; i < n; i++) {
    final a = -2 * math.pi * i / n;
    cosT[i] = math.cos(a);
    sinT[i] = math.sin(a);
  }

  final sf = signal.asFloatList();
  final out = Float32List(batch * frames * bins * 2);
  final frame = Float64List(n);
  for (int b = 0; b < batch; b++) {
    for (int f = 0; f < frames; f++) {
      final src = b * len + f * frameStep;
      for (int i = 0; i < n; i++) {
        frame[i] = win == null ? sf[src + i] : sf[src + i] * win[i];
      }
      final outBase = ((b * frames) + f) * bins * 2;
      for (int k = 0; k < bins; k++) {
        double re = 0, im = 0;
        int idx = 0;
        for (int i = 0; i < n; i++) {
          final v = frame[i];
          re += v * cosT[idx];
          im += v * sinT[idx];
          idx += k;
          if (idx >= n) idx -= n;
        }
        out[outBase + 2 * k] = re;
        out[outBase + 2 * k + 1] = im;
      }
    }
  }
  return Tensor.float(out, [batch, frames, bins, 2]);
}

Tensor opWindow(String type, int size, {bool periodic = true}) {
  if (size < 0) throw ArgumentError('window size must be nonnegative');
  final out = Float32List(size);
  if (size == 1) {
    out[0] = 1;
    return Tensor.float(out, [size]);
  }
  final den = periodic ? size : size - 1;
  for (int n = 0; n < size; n++) {
    final x = 2 * math.pi * n / den;
    out[n] = switch (type) {
      'Hann' => 0.5 - 0.5 * math.cos(x),
      'Hamming' => 0.54 - 0.46 * math.cos(x),
      'Blackman' => 0.42 - 0.5 * math.cos(x) + 0.08 * math.cos(2 * x),
      _ => throw ArgumentError('Unknown window type "$type"'),
    };
  }
  return Tensor.float(out, [size]);
}

/// `ReduceMax` / `ReduceMin` (dtype-preserving, unlike the mean/sum
/// reductions which are float by definition).
Tensor opReduceMinMax(
  Tensor x,
  List<int>? axes,
  bool keepdims, {
  required bool isMax,
}) {
  final rank = x.shape.length;
  final ax =
      (axes == null || axes.isEmpty ? List<int>.generate(rank, (k) => k) : axes)
          .map((a) => a < 0 ? a + rank : a)
          .toSet();
  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k],
  ];
  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;
  final best = Float64List(n)
    ..fillRange(0, n, isMax ? double.negativeInfinity : double.infinity);

  final coords = List<int>.filled(rank, 0);
  final total = x.length;
  for (int idx = 0; idx < total; idx++) {
    int outFlat = 0;
    for (int k = 0; k < rank; k++) {
      if (!ax.contains(k)) outFlat += coords[k] * outStridesFull[k];
    }
    final v = x.getD(idx);
    if (isMax ? v > best[outFlat] : v < best[outFlat]) best[outFlat] = v;
    for (int k = rank - 1; k >= 0; k--) {
      if (++coords[k] < x.shape[k]) break;
      coords[k] = 0;
    }
  }
  final shape = keepdims
      ? outShapeFull
      : [
          for (int k = 0; k < rank; k++)
            if (!ax.contains(k)) x.shape[k],
        ];
  if (x.isFloat) {
    final out = Float32List(n);
    for (int k = 0; k < n; k++) {
      out[k] = best[k];
    }
    return Tensor.float(out, shape);
  }
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = best[k].toInt();
  }
  return Tensor.int64(out, shape);
}

Tensor opSoftmax(Tensor x, int axis) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = x.shape[ax];
  final innerSize = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  final out = Float32List(x.length);
  final xf = x.f ?? x.asFloatList(); // attention scores are always float
  // Contiguous last-axis softmax (the overwhelmingly common case: attention
  // rows, classifier heads) — walk each row linearly instead of striding.
  if (innerSize == 1) {
    final rows = x.length ~/ axisSize;
    for (int r = 0; r < rows; r++) {
      final base = r * axisSize;
      double maxV = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        final v = xf[base + a];
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        final e = math.exp(xf[base + a] - maxV);
        out[base + a] = e;
        sum += e;
      }
      final inv = 1.0 / sum;
      for (int a = 0; a < axisSize; a++) {
        out[base + a] *= inv;
      }
    }
    return Tensor.float(out, x.shape);
  }
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      final b0 = outer * axisSize * innerSize + inner;
      double maxV = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        final v = xf[b0 + a * innerSize];
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        final e = math.exp(xf[b0 + a * innerSize] - maxV);
        out[b0 + a * innerSize] = e;
        sum += e;
      }
      final inv = 1.0 / sum;
      for (int a = 0; a < axisSize; a++) {
        out[b0 + a * innerSize] *= inv;
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `LogSoftmax(x) = x - max - log(sum(exp(x - max)))` — numerically stable
/// via the same max-shift as Softmax, but keeping the result in log space
/// (CTC / LID heads use this).
Tensor opLogSoftmax(Tensor x, int axis) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final axisSize = x.shape[ax];
  final innerSize = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  final out = Float32List(x.length);
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      double maxV = double.negativeInfinity;
      for (int a = 0; a < axisSize; a++) {
        final v = x.getD(outer * axisSize * innerSize + a * innerSize + inner);
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int a = 0; a < axisSize; a++) {
        sum += math.exp(
          x.getD(outer * axisSize * innerSize + a * innerSize + inner) - maxV,
        );
      }
      final logSum = maxV + math.log(sum);
      for (int a = 0; a < axisSize; a++) {
        final idx = outer * axisSize * innerSize + a * innerSize + inner;
        out[idx] = x.getD(idx) - logSum;
      }
    }
  }
  return Tensor.float(out, x.shape);
}

Tensor opLayerNormalization(
  Tensor x,
  Tensor scale,
  Tensor bias,
  int axis,
  double epsilon,
) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final outerSize = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final normSize = x.shape.sublist(ax).fold<int>(1, (a, b) => a * b);

  final out = Float32List(x.length);
  // LayerNorm inputs are always float; direct-index the backing buffer to
  // skip getD's per-element dtype branch (this runs per token per layer).
  final xf = x.f ?? x.asFloatList();
  final sf = scale.asFloatList(), bf = bias.asFloatList();
  for (int outer = 0; outer < outerSize; outer++) {
    final base = outer * normSize;
    double mean = 0;
    for (int k = 0; k < normSize; k++) {
      mean += xf[base + k];
    }
    mean /= normSize;
    double variance = 0;
    for (int k = 0; k < normSize; k++) {
      final d = xf[base + k] - mean;
      variance += d * d;
    }
    variance /= normSize;
    final invStd = 1.0 / math.sqrt(variance + epsilon);
    for (int k = 0; k < normSize; k++) {
      out[base + k] = (xf[base + k] - mean) * invStd * sf[k] + bf[k];
    }
  }
  return Tensor.float(out, x.shape);
}

// ---------------------------------------------------------------------------
// MatMul / Gemm
// ---------------------------------------------------------------------------

/// numpy-style matmul: last two dims are the matrix dims, leading dims batch
/// (broadcastable).
Tensor opMatMul(Tensor a, Tensor b) {
  final aRank = a.shape.length, bRank = b.shape.length;
  assert(aRank >= 2 && bRank >= 2);
  final m = a.shape[aRank - 2], k = a.shape[aRank - 1];
  final k2 = b.shape[bRank - 2], n = b.shape[bRank - 1];
  assert(k == k2, 'MatMul inner dims mismatch: $k vs $k2');

  final aBatch = a.shape.sublist(0, aRank - 2);
  final bBatch = b.shape.sublist(0, bRank - 2);
  final outBatch = _broadcastShape(aBatch, bBatch);
  final outShape = [...outBatch, m, n];

  final batchCount = outBatch.fold<int>(1, (x, y) => x * y);

  // B without batch dims (a shared weight): every batch of A multiplies the
  // same matrix, and A's batches are contiguous rows — collapse them into a
  // single GEMM. Otherwise a [64,1,k] @ [k,n] would re-pack the weight 64
  // times to compute one row each (observed: 3x whole-model slowdowns).
  if (a.f != null &&
      b.f != null &&
      batchCount > 1 &&
      bBatch.fold<int>(1, (x, y) => x * y) == 1) {
    final out = Float32List(batchCount * m * n);
    gemm.matmulKernel(a.f!, 0, b.f!, 0, out, 0, batchCount * m, k, n);
    return Tensor.float(out, outShape);
  }

  final out = Float32List(batchCount * m * n);

  final aMatSize = m * k, bMatSize = k2 * n;
  final af = a.f,
      bf = b.f; // non-null in the (only actually occurring) all-float case
  for (int batch = 0; batch < batchCount; batch++) {
    final batchCoords = _unflatten(batch, outBatch);
    final aBatchFlat = _flattenBroadcast(batchCoords, aBatch);
    final bBatchFlat = _flattenBroadcast(batchCoords, bBatch);
    final aOff = aBatchFlat * aMatSize;
    final bOff = bBatchFlat * bMatSize;
    final outOff = batch * m * n;
    if (af != null && bf != null) {
      // Tiled (and, on native targets, SIMD) micro-kernel — see
      // gemm_kernel_simd.dart for the strategy.
      gemm.matmulKernel(af, aOff, bf, bOff, out, outOff, m, k, n);
    } else {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          double sum = 0;
          for (int kk = 0; kk < k; kk++) {
            sum += a.getD(aOff + i * k + kk) * b.getD(bOff + kk * n + j);
          }
          out[outOff + i * n + j] = sum;
        }
      }
    }
  }
  return Tensor.float(out, outShape);
}

/// Fused RMSNorm: `y = x * gamma / sqrt(mean(x², axis) + eps)` — the
/// 7-node Pow/ReduceMean/Add/Sqrt/Reciprocal/Mul/Mul chain emitted by
/// RMSNorm exports (Qwen-style transformers, Maia3).
Tensor opRMSNorm(Tensor x, Tensor gamma, int axis, double eps) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final xf = x.f!;
  final gf = gamma.asFloatList();
  // The original chain's rank-1 Mul(gamma) broadcasts along the LAST axis
  // regardless of the reduce axis — enforce the shapes that makes valid.
  if (gf.length != x.shape[x.rank - 1]) {
    throw ArgumentError(
      'RMSNorm gamma length ${gf.length} != last dim '
      '${x.shape[x.rank - 1]}',
    );
  }
  final out = Float32List(xf.length);
  if (ax == x.rank - 1) {
    final d = x.shape[ax];
    final rows = xf.length ~/ d;
    for (int r = 0; r < rows; r++) {
      final base = r * d;
      double ss = 0;
      for (int j = 0; j < d; j++) {
        final v = xf[base + j];
        ss += v * v;
      }
      final inv = 1.0 / math.sqrt(ss / d + eps);
      for (int j = 0; j < d; j++) {
        out[base + j] = xf[base + j] * inv * gf[j];
      }
    }
    return Tensor.float(out, x.shape);
  }
  // General axis: strided walk.
  final dim = x.shape[ax];
  int outer = 1, inner = 1;
  for (int a = 0; a < ax; a++) {
    outer *= x.shape[a];
  }
  for (int a = ax + 1; a < x.rank; a++) {
    inner *= x.shape[a];
  }
  final lastDim = x.shape[x.rank - 1];
  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      final base = o * dim * inner + i;
      double ss = 0;
      for (int j = 0; j < dim; j++) {
        final v = xf[base + j * inner];
        ss += v * v;
      }
      final inv = 1.0 / math.sqrt(ss / dim + eps);
      for (int j = 0; j < dim; j++) {
        final idx = base + j * inner;
        out[idx] = xf[idx] * inv * gf[idx % lastDim];
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `GroupQueryAttention` (com.microsoft) — grouped-query attention over
/// already-projected `[batch, seq, *]` Q/K/V, the fused op most modern LLM
/// exports use (Llama3 / Qwen2 / Gemma3). [numHeads] query heads share
/// [kvNumHeads] KV heads (each query head `h` attends to KV head
/// `h ~/ (numHeads/kvNumHeads)`). When [doRotary], RoPE is applied to Q and K
/// inside the op from [cosCache]/[sinCache] at positions `pastLen..pastLen+seq`.
/// [attentionBias] is the additive mask (typically the causal mask).
/// `softcap` and `local_window` are not supported.
///
/// KV cache: [pastKey]/[pastValue] (`[batch, kvNumHeads, pastLen, headSize]`)
/// are prepended to the current K/V; the op returns
/// `[output, presentKey, presentValue]` where present KV is the full
/// `[batch, kvNumHeads, pastLen+seq, headSize]` concatenation (the RoPE'd K and
/// the raw V), ready to feed back as the next step's past. Query token `i` has
/// absolute position `pastLen+i` and attends causally to keys `0..pastLen+i`.
List<Tensor> opGroupQueryAttention(
  Tensor q,
  Tensor k,
  Tensor v, {
  required int numHeads,
  required int kvNumHeads,
  double? scale,
  bool doRotary = false,
  bool rotaryInterleaved = false,
  Tensor? cosCache,
  Tensor? sinCache,
  Tensor? pastKey,
  Tensor? pastValue,
  Tensor? attentionBias,
  double softcap = 0,
  int localWindow = -1,
}) {
  if (softcap != 0 || localWindow > 0) {
    throw UnsupportedError(
      'GroupQueryAttention: softcap / local_window not supported',
    );
  }
  final batch = q.shape[0], seq = q.shape[1];
  final headSize = q.shape[2] ~/ numHeads;
  final group = numHeads ~/ kvNumHeads;
  final kvHidden = kvNumHeads * headSize;
  final pastLen = (pastKey != null && pastKey.shape.length == 4)
      ? pastKey.shape[2]
      : 0;
  final totalLen = pastLen + seq;

  var qq = q, kk = k;
  if (doRotary) {
    final pos = Tensor.int64(
      Int64List.fromList([for (int s = 0; s < seq; s++) pastLen + s]),
      [1, seq],
    );
    qq = opRotaryEmbedding(
      q,
      pos,
      cosCache!,
      sinCache!,
      interleaved: rotaryInterleaved,
      numHeads: numHeads,
    );
    kk = opRotaryEmbedding(
      k,
      pos,
      cosCache,
      sinCache,
      interleaved: rotaryInterleaved,
      numHeads: kvNumHeads,
    );
  }
  final s = scale ?? (1.0 / math.sqrt(headSize));
  final qf = qq.f!, kf = kk.f!, vf = v.f ?? v.asFloatList();
  final pkf = pastKey?.f ?? pastKey?.asFloatList();
  final pvf = pastValue?.f ?? pastValue?.asFloatList();
  final bias = attentionBias?.asFloatList();
  final biasHeads = attentionBias == null ? 0 : attentionBias.shape[1];

  // present_key / present_value: [batch, kvNumHeads, totalLen, headSize],
  // the past cache (if any) followed by the current step's K/V.
  final presentK = Float32List(batch * kvNumHeads * totalLen * headSize);
  final presentV = Float32List(batch * kvNumHeads * totalLen * headSize);
  for (int b = 0; b < batch; b++) {
    for (int kvh = 0; kvh < kvNumHeads; kvh++) {
      final baseP = ((b * kvNumHeads + kvh) * totalLen) * headSize;
      for (int j = 0; j < pastLen; j++) {
        final src = ((b * kvNumHeads + kvh) * pastLen + j) * headSize;
        final dst = baseP + j * headSize;
        for (int d = 0; d < headSize; d++) {
          presentK[dst + d] = pkf![src + d];
          presentV[dst + d] = pvf![src + d];
        }
      }
      for (int j = 0; j < seq; j++) {
        final src = (b * seq + j) * kvHidden + kvh * headSize;
        final dst = baseP + (pastLen + j) * headSize;
        for (int d = 0; d < headSize; d++) {
          presentK[dst + d] = kf[src + d];
          presentV[dst + d] = vf[src + d];
        }
      }
    }
  }

  final out = Float32List(batch * seq * numHeads * headSize);
  final qh = Float32List(seq * headSize);
  final kt = Float32List(headSize * totalLen);
  final scores = Float32List(seq * totalLen);
  final ctx = Float32List(seq * headSize);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < numHeads; h++) {
      final kvh = h ~/ group;
      final baseP = ((b * kvNumHeads + kvh) * totalLen) * headSize;
      for (int i = 0; i < seq; i++) {
        final qs = (b * seq + i) * numHeads * headSize + h * headSize;
        for (int d = 0; d < headSize; d++) {
          qh[i * headSize + d] = qf[qs + d];
        }
      }
      // K^T for this KV head: [headSize, totalLen], read from present cache.
      for (int j = 0; j < totalLen; j++) {
        final ks = baseP + j * headSize;
        for (int d = 0; d < headSize; d++) {
          kt[d * totalLen + j] = presentK[ks + d];
        }
      }
      scores.fillRange(0, seq * totalLen, 0);
      gemm.matmulKernel(qh, 0, kt, 0, scores, 0, seq, headSize, totalLen);
      final biasBase = bias == null
          ? 0
          : (b * biasHeads + (h % biasHeads)) * seq * totalLen;
      for (int i = 0; i < seq; i++) {
        final row = i * totalLen;
        // token i (absolute position pastLen+i) attends causally to keys
        // 0..pastLen+i, plus any additive bias.
        final lastKey = pastLen + i;
        double maxV = double.negativeInfinity;
        for (int j = 0; j <= lastKey; j++) {
          var sc = scores[row + j] * s;
          if (bias != null) sc += bias[biasBase + row + j];
          scores[row + j] = sc;
          if (sc > maxV) maxV = sc;
        }
        double sum = 0;
        for (int j = 0; j <= lastKey; j++) {
          final e = math.exp(scores[row + j] - maxV);
          scores[row + j] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        for (int j = 0; j <= lastKey; j++) {
          scores[row + j] *= inv;
        }
        for (int j = lastKey + 1; j < totalLen; j++) {
          scores[row + j] = 0; // masked
        }
      }
      // ctx = scores · present_V  ([seq,totalLen] · [totalLen,headSize]); the
      // per-head V slice is contiguous at baseP.
      ctx.fillRange(0, seq * headSize, 0);
      gemm.matmulKernel(
        scores,
        0,
        presentV,
        baseP,
        ctx,
        0,
        seq,
        totalLen,
        headSize,
      );
      for (int i = 0; i < seq; i++) {
        final dst = (b * seq + i) * numHeads * headSize + h * headSize;
        for (int d = 0; d < headSize; d++) {
          out[dst + d] = ctx[i * headSize + d];
        }
      }
    }
  }
  return [
    Tensor.float(out, [batch, seq, numHeads * headSize]),
    Tensor.float(presentK, [batch, kvNumHeads, totalLen, headSize]),
    Tensor.float(presentV, [batch, kvNumHeads, totalLen, headSize]),
  ];
}

/// `MultiHeadAttention` (com.microsoft) — fused multi-head attention over
/// already-projected `[batch, seq, hidden]` Q/K/V. [attentionBias] (optional)
/// is an additive `[batch, heads, seq, kvSeq]` mask added to the scaled
/// scores before softmax. Q/K/V may carry different head counts already
/// expanded to match (grouped-query attention is expanded upstream here).
Tensor opMultiHeadAttention(
  Tensor q,
  Tensor k,
  Tensor v,
  Tensor? attentionBias, {
  required int numHeads,
  double? scale,
}) {
  final batch = q.shape[0], seq = q.shape[1], hidden = q.shape[2];
  final headSize = hidden ~/ numHeads;
  final kvSeq = k.shape[1];
  final s = scale ?? (1.0 / math.sqrt(headSize));
  final qf = q.f ?? q.asFloatList();
  final kf = k.f ?? k.asFloatList();
  final vf = v.f ?? v.asFloatList();
  final bias = attentionBias?.asFloatList();
  final biasHeads = attentionBias == null ? 0 : attentionBias.shape[1];

  final out = Float32List(batch * seq * hidden);
  final qh = Float32List(seq * headSize);
  final kt = Float32List(headSize * kvSeq); // K_h transposed [d, kvSeq]
  final vh = Float32List(kvSeq * headSize);
  final scores = Float32List(seq * kvSeq);
  final ctx = Float32List(seq * headSize);

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < numHeads; h++) {
      // Gather this head's Q [seq,d], Kᵀ [d,kvSeq], V [kvSeq,d].
      for (int i = 0; i < seq; i++) {
        final src = (b * seq + i) * hidden + h * headSize;
        for (int d = 0; d < headSize; d++) {
          qh[i * headSize + d] = qf[src + d];
        }
      }
      for (int j = 0; j < kvSeq; j++) {
        final src = (b * kvSeq + j) * hidden + h * headSize;
        for (int d = 0; d < headSize; d++) {
          kt[d * kvSeq + j] = kf[src + d];
          vh[j * headSize + d] = vf[src + d];
        }
      }
      // scores = Q·Kᵀ, then scale + bias + row softmax.
      scores.fillRange(0, seq * kvSeq, 0);
      gemm.matmulKernel(qh, 0, kt, 0, scores, 0, seq, headSize, kvSeq);
      final biasBase = bias == null
          ? 0
          : (b * biasHeads + (h % biasHeads)) * seq * kvSeq;
      for (int i = 0; i < seq; i++) {
        final row = i * kvSeq;
        double maxV = double.negativeInfinity;
        for (int j = 0; j < kvSeq; j++) {
          var sc = scores[row + j] * s;
          if (bias != null) sc += bias[biasBase + row + j];
          scores[row + j] = sc;
          if (sc > maxV) maxV = sc;
        }
        double sum = 0;
        for (int j = 0; j < kvSeq; j++) {
          final e = math.exp(scores[row + j] - maxV);
          scores[row + j] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        for (int j = 0; j < kvSeq; j++) {
          scores[row + j] *= inv;
        }
      }
      // context = P·V, scattered back to [b, seq, h*d].
      ctx.fillRange(0, seq * headSize, 0);
      gemm.matmulKernel(scores, 0, vh, 0, ctx, 0, seq, kvSeq, headSize);
      for (int i = 0; i < seq; i++) {
        final dst = (b * seq + i) * hidden + h * headSize;
        for (int d = 0; d < headSize; d++) {
          out[dst + d] = ctx[i * headSize + d];
        }
      }
    }
  }
  return Tensor.float(out, [batch, seq, hidden]);
}

/// Fused scaled-dot-product-attention epilogue:
/// `MatMul(Softmax(MatMul(a, b) * scale + mask, axis: -1), v)`.
///
/// The scale, mask-add and row softmax happen in one pass over the attention
/// matrix (no intermediate tensors); [mask] broadcasts against the score
/// shape numpy-style.
Tensor opFusedSDPA(Tensor a, Tensor b, Tensor v, Tensor mask, double scale) {
  final s = opMatMul(a, b);
  final sf = s.f!;
  final t = s.shape.last;
  final rows = s.length ~/ t;

  final mf = mask.asFloatList();
  // Per-row mask offset via broadcast; within a row the mask stride is 1
  // (mask last dim == t) or 0 (mask last dim == 1).
  final rowShape = s.shape.sublist(0, s.rank - 1);
  final maskLastStride = mask.shape.isNotEmpty && mask.shape.last == t ? 1 : 0;
  final rowCoords = List<int>.filled(rowShape.length, 0);
  // Mask shape aligned against the full score shape, minus its last axis.
  final maskRowShape = mask.rank == 0
      ? const <int>[]
      : mask.shape.sublist(0, mask.rank - 1);

  for (int r = 0; r < rows; r++) {
    final base = r * t;
    final mBase =
        _flattenBroadcast(rowCoords, maskRowShape) *
        (mask.rank == 0 ? 0 : mask.shape.last);
    double max = double.negativeInfinity;
    for (int j = 0; j < t; j++) {
      final val = sf[base + j] * scale + mf[mBase + j * maskLastStride];
      sf[base + j] = val;
      if (val > max) max = val;
    }
    double sum = 0;
    for (int j = 0; j < t; j++) {
      final e = math.exp(sf[base + j] - max);
      sf[base + j] = e;
      sum += e;
    }
    for (int j = 0; j < t; j++) {
      sf[base + j] /= sum;
    }
    for (int k = rowShape.length - 1; k >= 0; k--) {
      if (++rowCoords[k] < rowShape[k]) break;
      rowCoords[k] = 0;
    }
  }
  return opMatMul(s, v);
}

/// Gemm: Y = alpha * A' * B' + beta * C  (A'/B' optionally transposed)
Tensor opGemm(
  Tensor a,
  Tensor b,
  Tensor? c, {
  double alpha = 1.0,
  double beta = 1.0,
  bool transA = false,
  bool transB = false,
}) {
  final at = transA ? opTranspose(a, [1, 0]) : a;
  final bt = transB ? opTranspose(b, [1, 0]) : b;
  final mm = opMatMul(at, bt);
  if (alpha != 1.0) {
    for (int k = 0; k < mm.f!.length; k++) {
      mm.f![k] *= alpha;
    }
  }
  if (c == null) return mm;
  final scaledC = beta == 1.0 ? c : _elementwiseUnary(c, (v) => v * beta);
  return opAdd(mm, scaledC);
}

// ---------------------------------------------------------------------------
// Einsum — special-cased for exactly the two equations this graph uses
// (a general einsum parser isn't needed for two known patterns).
// ---------------------------------------------------------------------------

/// General einsum over one or two operands (explicit labels; `...` is not
/// supported). The two transformer-hot equations keep their specialized
/// kernels; everything else runs the direct sum-of-products definition.
Tensor opEinsum(String equation, List<Tensor> inputs) {
  final eq = equation.replaceAll(' ', '');
  if (inputs.length == 2) {
    switch (eq) {
      case 'bhi,oi->bho':
        return _einsumBhiOi(inputs[0], inputs[1]);
      case 'bid,bjd->bij':
        return _einsumBidBjd(inputs[0], inputs[1]);
    }
  }
  if (eq.contains('...')) {
    throw UnsupportedError('Einsum: ellipsis not supported ("$equation")');
  }
  final arrowSplit = eq.split('->');
  final terms = arrowSplit[0].split(',');
  if (terms.length != inputs.length || inputs.length > 2) {
    throw UnsupportedError(
      'Einsum: ${inputs.length} inputs vs '
      '${terms.length} terms in "$equation" (max 2 supported)',
    );
  }

  // Label -> dimension size, checked for consistency across operands.
  final sizeOf = <String, int>{};
  final counts = <String, int>{};
  for (int t = 0; t < terms.length; t++) {
    if (terms[t].length != inputs[t].rank) {
      throw ArgumentError(
        'Einsum term "${terms[t]}" vs rank '
        '${inputs[t].rank} operand',
      );
    }
    for (int a = 0; a < terms[t].length; a++) {
      final l = terms[t][a];
      final d = inputs[t].shape[a];
      final prev = sizeOf[l];
      if (prev != null && prev != d && prev != 1 && d != 1) {
        throw ArgumentError('Einsum label "$l" size mismatch: $prev vs $d');
      }
      sizeOf[l] = math.max(prev ?? 1, d);
      counts.update(l, (v) => v + 1, ifAbsent: () => 1);
    }
  }
  final outTerm = arrowSplit.length > 1
      ? arrowSplit[1]
      : ([
          for (final l in (counts.keys.toList()..sort()))
            if (counts[l] == 1) l,
        ].join());
  if (outTerm.split('').toSet().length != outTerm.length) {
    throw UnsupportedError(
      'Einsum: repeated output labels not supported ("$equation")',
    );
  }
  final sumLabels = [
    for (final l in sizeOf.keys)
      if (!outTerm.contains(l)) l,
  ]..sort();

  // Global coordinate order: output labels then contraction labels.
  final order = [...outTerm.split(''), ...sumLabels];
  final pos = {for (int i = 0; i < order.length; i++) order[i]: i};
  // Per input: stride contribution of each global coordinate (0 when the
  // operand broadcasts along that label).
  final contribs = <List<int>>[];
  for (int t = 0; t < terms.length; t++) {
    final c = List<int>.filled(order.length, 0);
    final strides = inputs[t].strides;
    for (int a = 0; a < terms[t].length; a++) {
      if (inputs[t].shape[a] != 1) c[pos[terms[t][a]]!] += strides[a];
    }
    contribs.add(c);
  }

  final outShape = [for (final l in outTerm.split('')) sizeOf[l]!];
  final outN = outShape.fold<int>(1, (a, b) => a * b);
  final sumShape = [for (final l in sumLabels) sizeOf[l]!];
  final sumN = sumShape.fold<int>(1, (a, b) => a * b);
  final out = Float32List(outN);
  final coords = List<int>.filled(order.length, 0);
  final nOut = outTerm.length;

  for (int oi = 0; oi < outN; oi++) {
    double acc = 0;
    for (int si = 0; si < sumN; si++) {
      double prod = 1;
      for (int t = 0; t < inputs.length; t++) {
        int flat = 0;
        final c = contribs[t];
        for (int p = 0; p < order.length; p++) {
          flat += coords[p] * c[p];
        }
        prod *= inputs[t].getD(flat);
      }
      acc += prod;
      for (int p = order.length - 1; p >= nOut; p--) {
        if (++coords[p] < sizeOf[order[p]]!) break;
        coords[p] = 0;
      }
    }
    out[oi] = acc;
    for (int p = nOut - 1; p >= 0; p--) {
      if (++coords[p] < outShape[p]) break;
      coords[p] = 0;
    }
  }
  // Output dtype follows the inputs (integer einsum stays integer).
  if (inputs.every((t) => !t.isFloat)) {
    return Tensor.int64(
      Int64List.fromList([for (final v in out) v.round()]),
      outShape,
    );
  }
  return Tensor.float(out, outShape);
}

// A:[b,h,i], B:[o,i] -> out[b,h,o] = sum_i A[b,h,i]*B[o,i]  (a linear layer)
Tensor _einsumBhiOi(Tensor a, Tensor b) {
  // B arrives as [o, i]: both operands are already contiguous along the
  // contraction axis, so SIMD row dots beat a transpose+pack GEMM here.
  final bs = a.shape[0], h = a.shape[1], i = a.shape[2];
  final o = b.shape[0];
  assert(b.shape[1] == i);
  final out = Float32List(bs * h * o);
  final af = a.f!, bf = b.f!;
  for (int r = 0; r < bs * h; r++) {
    final aBase = r * i, outBase = r * o;
    for (int oi = 0; oi < o; oi++) {
      out[outBase + oi] = gemm.dotProduct(af, aBase, bf, oi * i, i);
    }
  }
  return Tensor.float(out, [bs, h, o]);
}

// A:[b,i,d], B:[b,j,d] -> out[b,i,j] = sum_d A[b,i,d]*B[b,j,d]  (attention scores)
Tensor _einsumBidBjd(Tensor a, Tensor b) {
  // Both operands contract along their contiguous last axis — SIMD row
  // dots, no transpose needed (attention scores).
  final bs = a.shape[0], i = a.shape[1], d = a.shape[2];
  final j = b.shape[1];
  assert(b.shape[0] == bs && b.shape[2] == d);
  final out = Float32List(bs * i * j);
  final af = a.f!, bf = b.f!;
  for (int bi = 0; bi < bs; bi++) {
    for (int ii = 0; ii < i; ii++) {
      final aBase = (bi * i + ii) * d;
      final outBase = (bi * i + ii) * j;
      final bBatch = bi * j * d;
      for (int ji = 0; ji < j; ji++) {
        out[outBase + ji] = gemm.dotProduct(af, aBase, bf, bBatch + ji * d, d);
      }
    }
  }
  return Tensor.float(out, [bs, i, j]);
}

// ---------------------------------------------------------------------------
// Extended op set (transformer embedders / rerankers): unary math, comparison
// and logical ops, selection, ranges and shape-fills. Added so the interpreter
// runs the common BERT / RoPE embedding & reranking graphs, not just Maia3.
// ---------------------------------------------------------------------------

Tensor opAbs(Tensor a) => a.isFloat
    ? _elementwiseUnary(a, (x) => x.abs())
    : _unaryInt(a, (x) => x.abs());
Tensor opNeg(Tensor a) =>
    a.isFloat ? _elementwiseUnary(a, (x) => -x) : _unaryInt(a, (x) => -x);
Tensor opSigmoid(Tensor a) =>
    _elementwiseUnary(a, (x) => 1.0 / (1.0 + math.exp(-x)));
Tensor opTanh(Tensor a) => _elementwiseUnary(a, _tanh);
Tensor opCos(Tensor a) => _elementwiseUnary(a, math.cos);
Tensor opSin(Tensor a) => _elementwiseUnary(a, math.sin);
Tensor opExp(Tensor a) => _elementwiseUnary(a, math.exp);
Tensor opLog(Tensor a) => _elementwiseUnary(a, math.log);

double _tanh(double x) {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  final e2 = math.exp(2 * x);
  return (e2 - 1) / (e2 + 1);
}

Tensor _unaryInt(Tensor a, int Function(int) op) {
  final n = a.length;
  final out = Int64List(n);
  for (int k = 0; k < n; k++) {
    out[k] = op(a.getI(k));
  }
  return Tensor.int64(out, a.shape);
}

// Comparison / logical ops. ONNX yields BOOL tensors; we carry them as int64
// 0/1 so downstream Where / Cast / And keep working with the two dtypes the
// tensor type supports.
Tensor _boolBinary(Tensor a, Tensor b, bool Function(double, double) p) {
  final same = _shapeEq(a.shape, b.shape);
  if (same || (b.length == 1 && a.rank >= b.rank)) {
    final n = a.length;
    final bScalar = b.length == 1;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = p(a.getD(k), b.getD(bScalar ? 0 : k)) ? 1 : 0;
    }
    return Tensor.int64(out, a.shape);
  }
  if (a.length == 1 && b.rank >= a.rank) {
    final n = b.length;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = p(a.getD(0), b.getD(k)) ? 1 : 0;
    }
    return Tensor.int64(out, b.shape);
  }
  final outShape = _broadcastShape(a.shape, b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final coords = List<int>.filled(outShape.length, 0);
  final out = Int64List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] =
        p(
          a.getD(_flattenBroadcast(coords, a.shape)),
          b.getD(_flattenBroadcast(coords, b.shape)),
        )
        ? 1
        : 0;
    for (int k = outShape.length - 1; k >= 0; k--) {
      if (++coords[k] < outShape[k]) break;
      coords[k] = 0;
    }
  }
  return Tensor.int64(out, outShape);
}

Tensor opEqual(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x == y);
Tensor opGreater(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x > y);
Tensor opLess(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x < y);
Tensor opGreaterOrEqual(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => x >= y);
Tensor opLessOrEqual(Tensor a, Tensor b) => _boolBinary(a, b, (x, y) => x <= y);
Tensor opAnd(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => x != 0 && y != 0);
Tensor opOr(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => x != 0 || y != 0);
Tensor opNot(Tensor a) => _unaryInt(a, (x) => x != 0 ? 0 : 1);
Tensor opXor(Tensor a, Tensor b) =>
    _boolBinary(a, b, (x, y) => (x != 0) != (y != 0));

Tensor _intBinary(Tensor a, Tensor b, int Function(int, int) op) {
  if (_shapeEq(a.shape, b.shape)) {
    final n = a.length;
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = op(a.getI(k), b.getI(k));
    }
    return Tensor.int64(out, a.shape);
  }
  if (b.length == 1 && a.rank >= b.rank) {
    final n = a.length;
    final out = Int64List(n);
    final y = b.getI(0);
    for (int k = 0; k < n; k++) {
      out[k] = op(a.getI(k), y);
    }
    return Tensor.int64(out, a.shape);
  }
  if (a.length == 1 && b.rank >= a.rank) {
    final n = b.length;
    final out = Int64List(n);
    final x = a.getI(0);
    for (int k = 0; k < n; k++) {
      out[k] = op(x, b.getI(k));
    }
    return Tensor.int64(out, b.shape);
  }
  final outShape = _broadcastShape(a.shape, b.shape);
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final coords = List<int>.filled(outShape.length, 0);
  final out = Int64List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = op(
      a.getI(_flattenBroadcast(coords, a.shape)),
      b.getI(_flattenBroadcast(coords, b.shape)),
    );
    for (int k = outShape.length - 1; k >= 0; k--) {
      if (++coords[k] < outShape[k]) break;
      coords[k] = 0;
    }
  }
  return Tensor.int64(out, outShape);
}

Tensor opBitwiseAnd(Tensor a, Tensor b) => _intBinary(a, b, (x, y) => x & y);
Tensor opBitwiseOr(Tensor a, Tensor b) => _intBinary(a, b, (x, y) => x | y);
Tensor opBitwiseXor(Tensor a, Tensor b) => _intBinary(a, b, (x, y) => x ^ y);
Tensor opBitwiseNot(Tensor a) => _unaryInt(a, (x) => ~x);
Tensor opBitShift(Tensor a, Tensor b, {required bool left}) =>
    _intBinary(a, b, (x, y) => left ? x << y : x >> y);

/// `IsNaN` / `IsInf` — bool (int64 0/1) mask over a float tensor. Integer
/// tensors are never NaN/Inf, so the result is all-zero.
Tensor opIsNaN(Tensor a) {
  final n = a.length;
  final out = Int64List(n);
  if (a.isFloat) {
    final af = a.f!;
    for (int k = 0; k < n; k++) {
      out[k] = af[k].isNaN ? 1 : 0;
    }
  }
  return Tensor.int64(out, a.shape);
}

Tensor opIsInf(
  Tensor a, {
  bool detectPositive = true,
  bool detectNegative = true,
}) {
  final n = a.length;
  final out = Int64List(n);
  if (a.isFloat) {
    final af = a.f!;
    for (int k = 0; k < n; k++) {
      final v = af[k];
      out[k] =
          (v.isInfinite &&
              ((v > 0 && detectPositive) || (v < 0 && detectNegative)))
          ? 1
          : 0;
    }
  }
  return Tensor.int64(out, a.shape);
}

Tensor opMax(List<Tensor> ins) =>
    ins.reduce((a, b) => _elementwiseBinary(a, b, math.max));
Tensor opMin(List<Tensor> ins) =>
    ins.reduce((a, b) => _elementwiseBinary(a, b, math.min));
Tensor opSum(List<Tensor> ins) => ins.reduce(opAdd);
Tensor opMean(List<Tensor> ins) {
  final sum = ins.reduce(opAdd);
  final out = Float32List(sum.length);
  final sf = sum.asFloatList();
  for (int k = 0; k < out.length; k++) {
    out[k] = sf[k] / ins.length;
  }
  return Tensor.float(out, sum.shape);
}

Tensor opNegativeLogLikelihoodLoss(
  Tensor input,
  Tensor target,
  Tensor? weight, {
  String reduction = 'mean',
  int? ignoreIndex,
}) {
  final classes = input.shape[1];
  final batch = input.shape[0];
  final spatial = batch == 0 ? 0 : target.length ~/ batch;
  final wf = weight?.asFloatList();
  final none = Float32List(target.length);
  double total = 0;
  int count = 0;
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < spatial; s++) {
      final t = target.getI(b * spatial + s);
      if (ignoreIndex != null && t == ignoreIndex) continue;
      final idx = (b * classes + t) * spatial + s;
      final w = wf == null ? 1.0 : wf[t];
      final loss = -w * input.getD(idx);
      none[b * spatial + s] = loss;
      total += loss;
      count++;
    }
  }
  if (reduction == 'none') return Tensor.float(none, target.shape);
  if (reduction == 'sum') return Tensor.scalarFloat(total);
  return Tensor.scalarFloat(count == 0 ? 0 : total / count);
}

List<Tensor> opSoftmaxCrossEntropyLoss(
  Tensor scores,
  Tensor labels,
  Tensor? weight, {
  String reduction = 'mean',
  int? ignoreIndex,
  bool wantLogProb = false,
}) {
  final batch = scores.shape[0];
  final classes = scores.shape[1];
  final spatial = scores.length ~/ (batch * classes);
  final logProb = Float32List(scores.length);
  final losses = Float32List(batch * spatial);
  final wf = weight?.asFloatList();
  double total = 0;
  int count = 0;
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < spatial; s++) {
      final base = b * classes * spatial + s;
      var maxV = double.negativeInfinity;
      for (int c = 0; c < classes; c++) {
        final v = scores.getD(base + c * spatial);
        if (v > maxV) maxV = v;
      }
      double sum = 0;
      for (int c = 0; c < classes; c++) {
        sum += math.exp(scores.getD(base + c * spatial) - maxV);
      }
      final logSum = maxV + math.log(sum);
      for (int c = 0; c < classes; c++) {
        logProb[base + c * spatial] = scores.getD(base + c * spatial) - logSum;
      }
      final t = labels.getI(b * spatial + s);
      if (ignoreIndex != null && t == ignoreIndex) continue;
      final w = wf == null ? 1.0 : wf[t];
      final loss = -w * logProb[base + t * spatial];
      losses[b * spatial + s] = loss;
      total += loss;
      count++;
    }
  }
  final lossTensor = reduction == 'none'
      ? Tensor.float(losses, labels.shape)
      : Tensor.scalarFloat(
          reduction == 'sum' ? total : (count == 0 ? 0 : total / count),
        );
  return [lossTensor, if (wantLogProb) Tensor.float(logProb, scores.shape)];
}

/// Element-wise select: `cond ? a : b`, broadcasting all three inputs.
Tensor opWhere(Tensor cond, Tensor a, Tensor b) {
  final outShape = _broadcastShape(
    _broadcastShape(cond.shape, a.shape),
    b.shape,
  );
  final n = outShape.fold<int>(1, (x, y) => x * y);
  final bothInt = !a.isFloat && !b.isFloat;
  final coords = List<int>.filled(outShape.length, 0);

  double pick() {
    final c = cond.getD(_flattenBroadcast(coords, cond.shape));
    return c != 0
        ? a.getD(_flattenBroadcast(coords, a.shape))
        : b.getD(_flattenBroadcast(coords, b.shape));
  }

  void advance() {
    for (int k = outShape.length - 1; k >= 0; k--) {
      if (++coords[k] < outShape[k]) return;
      coords[k] = 0;
    }
  }

  if (bothInt) {
    final out = Int64List(n);
    for (int idx = 0; idx < n; idx++) {
      out[idx] = pick().round();
      advance();
    }
    return Tensor.int64(out, outShape);
  }
  final out = Float32List(n);
  for (int idx = 0; idx < n; idx++) {
    out[idx] = pick();
    advance();
  }
  return Tensor.float(out, outShape);
}

/// `Size` — total element count as an int64 scalar.
Tensor opSize(Tensor x) => Tensor.scalarInt(x.length);

/// `RotaryEmbedding` (com.microsoft / onnx opset 23) — fused rotary position
/// embedding. [x] is `[batch, seq, hidden]` (3-D) or
/// `[batch, heads, seq, headSize]` (4-D); [cosCache]/[sinCache] are
/// `[maxPos, rotaryDim/2]`, indexed by [positionIds] `[batch, seq]`. Only the
/// first `rotaryDim` of each head is rotated (the tail passes through);
/// `interleaved` selects the GPT-J pairing, else the GPT-NeoX rotate-half.
Tensor opRotaryEmbedding(
  Tensor x,
  Tensor positionIds,
  Tensor cosCache,
  Tensor sinCache, {
  bool interleaved = false,
  int numHeads = 0,
  int rotaryEmbeddingDim = 0,
}) {
  final xf = x.f ?? x.asFloatList();
  final cos = cosCache.asFloatList(), sin = sinCache.asFloatList();
  final pos = positionIds.asIntList();
  final halfRot = cosCache.shape.last;
  final rotaryDim = rotaryEmbeddingDim > 0 ? rotaryEmbeddingDim : 2 * halfRot;

  int batch, seq, heads, headSize;
  if (x.rank == 4) {
    batch = x.shape[0];
    heads = x.shape[1];
    seq = x.shape[2];
    headSize = x.shape[3];
  } else {
    batch = x.shape[0];
    seq = x.shape[1];
    final hidden = x.shape[2];
    // num_heads=0 → infer from a full-rotation head (headSize == rotaryDim).
    headSize = numHeads > 0 ? hidden ~/ numHeads : rotaryDim;
    heads = hidden ~/ headSize;
  }
  final cosStride = halfRot; // cos/sin row length per position

  final out = Float32List(xf.length);
  // Layout: 4-D is [b,h,s,d] contiguous; 3-D is [b,s,h*d] contiguous.
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < seq; s++) {
      final p = pos[(positionIds.rank == 1 ? 0 : b) * seq + s];
      final cBase = p * cosStride;
      for (int h = 0; h < heads; h++) {
        final base = x.rank == 4
            ? ((b * heads + h) * seq + s) * headSize
            : (b * seq + s) * heads * headSize + h * headSize;
        // Rotated portion.
        if (interleaved) {
          for (int i = 0; i < rotaryDim ~/ 2; i++) {
            final c = cos[cBase + i], sn = sin[cBase + i];
            final x0 = xf[base + 2 * i], x1 = xf[base + 2 * i + 1];
            out[base + 2 * i] = x0 * c - x1 * sn;
            out[base + 2 * i + 1] = x1 * c + x0 * sn;
          }
        } else {
          final half = rotaryDim ~/ 2;
          for (int i = 0; i < half; i++) {
            final c = cos[cBase + i], sn = sin[cBase + i];
            final x0 = xf[base + i], x1 = xf[base + i + half];
            out[base + i] = x0 * c - x1 * sn;
            out[base + i + half] = x1 * c + x0 * sn;
          }
        }
        // Pass-through tail (headSize beyond rotaryDim).
        for (int i = rotaryDim; i < headSize; i++) {
          out[base + i] = xf[base + i];
        }
      }
    }
  }
  return Tensor.float(out, x.shape);
}

/// `RandomNormalLike` / `RandomNormal` — a Gaussian-noise tensor. A
/// pseudo-random draw can never match ORT's Mersenne-Twister stream across
/// implementations, so this returns the distribution *mean* (a deterministic
/// fill). That is exact for the only parity-testable case — models that scale
/// the noise to zero (e.g. VITS/Piper with `noise_scale = 0`) — and finite
/// otherwise. Genuinely stochastic sampling is not reproduced.
Tensor opRandomNormalFill(List<int> shape, double mean) {
  final inj = OnnxRandomInject.draw('RandomNormal', shape);
  return inj != null
      ? Tensor.float(inj, shape)
      : Tensor.filledFloat(shape, mean);
}

/// `ArgMax` / `ArgMin` along [axis]; ties resolve to the first (or, with
/// [selectLastIndex], the last) occurrence.
Tensor opArgMinMax(
  Tensor x,
  int axis,
  bool keepdims, {
  required bool isMax,
  bool selectLastIndex = false,
}) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final dim = x.shape[ax];
  int outer = 1, inner = 1;
  for (int a = 0; a < ax; a++) {
    outer *= x.shape[a];
  }
  for (int a = ax + 1; a < x.rank; a++) {
    inner *= x.shape[a];
  }
  final out = Int64List(outer * inner);
  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      final base = o * dim * inner + i;
      int bestIdx = 0;
      double best = x.getD(base);
      // NaN handling matches ORT's CPU kernel (pinned by the argmax_nan
      // fixture): plain comparisons, so a NaN never wins unless it is the
      // initial element — numpy/the ONNX reference differ here, ORT is the
      // oracle.
      for (int d = 1; d < dim; d++) {
        final v = x.getD(base + d * inner);
        final better = isMax
            ? (selectLastIndex ? v >= best : v > best)
            : (selectLastIndex ? v <= best : v < best);
        if (better) {
          best = v;
          bestIdx = d;
        }
      }
      out[o * inner + i] = bestIdx;
    }
  }
  final shape = [
    for (int a = 0; a < x.rank; a++)
      if (a != ax) x.shape[a] else if (keepdims) 1,
  ];
  return Tensor.int64(out, shape);
}

/// `NonZero` — indices of nonzero elements as `[rank, count]` int64,
/// scanning in row-major order.
Tensor opNonZero(Tensor x) {
  final rank = math.max(x.rank, 1);
  final hits = <int>[];
  for (int k = 0; k < x.length; k++) {
    if (x.getD(k) != 0) hits.add(k);
  }
  final strides = x.strides;
  final out = Int64List(rank * hits.length);
  for (int j = 0; j < hits.length; j++) {
    int rem = hits[j];
    for (int a = 0; a < x.rank; a++) {
      out[a * hits.length + j] = rem ~/ strides[a];
      rem %= strides[a];
    }
  }
  return Tensor.int64(out, [rank, hits.length]);
}

/// `TopK` — the [k] largest (or smallest) values along [axis], sorted, with
/// ties broken toward the lower index. Returns `[values, indices]`.
List<Tensor> opTopK(Tensor x, int k, {int axis = -1, bool largest = true}) {
  final ax = axis < 0 ? axis + x.rank : axis;
  final dim = x.shape[ax];
  // Computed from the shape, not by division — TopK over an empty axis
  // (zero candidates after a filter) is legal and returns empty outputs.
  int outer = 1, inner = 1;
  for (int a = 0; a < ax; a++) {
    outer *= x.shape[a];
  }
  for (int a = ax + 1; a < x.rank; a++) {
    inner *= x.shape[a];
  }
  final outShape = [...x.shape]..[ax] = k;
  final values = Float32List(outer * k * inner);
  final indices = Int64List(outer * k * inner);
  final order = List<int>.generate(dim, (i) => i);
  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      final base = o * dim * inner + i;
      for (int d = 0; d < dim; d++) {
        order[d] = d;
      }
      order.sort((p, q) {
        final vp = x.getD(base + p * inner), vq = x.getD(base + q * inner);
        if (vp != vq) return largest ? vq.compareTo(vp) : vp.compareTo(vq);
        return p.compareTo(q);
      });
      final outBase = o * k * inner + i;
      for (int d = 0; d < k; d++) {
        values[outBase + d * inner] = x.getD(base + order[d] * inner);
        indices[outBase + d * inner] = order[d];
      }
    }
  }
  final vT = Tensor.float(values, outShape);
  return [
    x.isFloat
        ? vT
        : Tensor.int64(
            Int64List.fromList([for (final v in values) v.toInt()]),
            outShape,
          ),
    Tensor.int64(indices, outShape),
  ];
}

/// `NonMaxSuppression` — greedy per-(batch, class) suppression. Returns
/// `[num_selected, 3]` int64 rows of (batch, class, boxIndex), in
/// batch-major, class-major, descending-score order (matching ORT).
Tensor opNonMaxSuppression(
  Tensor boxes,
  Tensor scores, {
  int maxOutputBoxesPerClass = 0,
  double iouThreshold = 0,
  double? scoreThreshold,
  bool centerPointBox = false,
}) {
  final nBatch = boxes.shape[0], nBox = boxes.shape[1];
  final nClass = scores.shape[1];
  final bf = boxes.asFloatList(), sf = scores.asFloatList();
  final selected = <int>[];
  if (maxOutputBoxesPerClass > 0) {
    // Corner coords may come in either order; normalize with min/max.
    (double, double, double, double) corners(int b, int i) {
      final o = (b * nBox + i) * 4;
      if (centerPointBox) {
        final cx = bf[o], cy = bf[o + 1], w = bf[o + 2], h = bf[o + 3];
        return (cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2);
      }
      final y1 = math.min(bf[o], bf[o + 2]);
      final y2 = math.max(bf[o], bf[o + 2]);
      final x1 = math.min(bf[o + 1], bf[o + 3]);
      final x2 = math.max(bf[o + 1], bf[o + 3]);
      return (y1, x1, y2, x2);
    }

    double iou(int b, int i, int j) {
      final (iy1, ix1, iy2, ix2) = corners(b, i);
      final (jy1, jx1, jy2, jx2) = corners(b, j);
      final w = math.min(ix2, jx2) - math.max(ix1, jx1);
      final h = math.min(iy2, jy2) - math.max(iy1, jy1);
      if (w <= 0 || h <= 0) return 0;
      final inter = w * h;
      final areaI = (ix2 - ix1) * (iy2 - iy1);
      final areaJ = (jx2 - jx1) * (jy2 - jy1);
      return inter / (areaI + areaJ - inter);
    }

    final order = List<int>.generate(nBox, (i) => i);
    for (int b = 0; b < nBatch; b++) {
      for (int c = 0; c < nClass; c++) {
        final sBase = (b * nClass + c) * nBox;
        for (int i = 0; i < nBox; i++) {
          order[i] = i;
        }
        order.sort((p, q) {
          final d = sf[sBase + q].compareTo(sf[sBase + p]);
          return d != 0 ? d : p.compareTo(q);
        });
        final picked = <int>[];
        for (final cand in order) {
          if (picked.length >= maxOutputBoxesPerClass) break;
          final s = sf[sBase + cand];
          if (scoreThreshold != null && s <= scoreThreshold) break;
          var keep = true;
          for (final p in picked) {
            if (iou(b, p, cand) > iouThreshold) {
              keep = false;
              break;
            }
          }
          if (keep) {
            picked.add(cand);
            selected.addAll([b, c, cand]);
          }
        }
      }
    }
  }
  return Tensor.int64(Int64List.fromList(selected), [selected.length ~/ 3, 3]);
}

/// `Trilu` — keeps the upper (or lower) triangle of the last two dims,
/// zeroing the rest; [k] shifts the diagonal.
Tensor opDet(Tensor x) {
  final rank = x.rank;
  if (rank < 2 || x.shape[rank - 2] != x.shape[rank - 1]) {
    throw ArgumentError('Det requires square matrices, got ${x.shape}');
  }
  final n = x.shape[rank - 1];
  final batch = x.length ~/ (n * n);
  final out = Float32List(batch);
  final a = Float64List(n * n);
  for (int b = 0; b < batch; b++) {
    for (int k = 0; k < n * n; k++) {
      a[k] = x.getD(b * n * n + k);
    }
    var det = 1.0;
    for (int i = 0; i < n; i++) {
      var pivot = i;
      for (int r = i + 1; r < n; r++) {
        if (a[r * n + i].abs() > a[pivot * n + i].abs()) pivot = r;
      }
      if (a[pivot * n + i] == 0) {
        det = 0;
        break;
      }
      if (pivot != i) {
        for (int c = 0; c < n; c++) {
          final tmp = a[i * n + c];
          a[i * n + c] = a[pivot * n + c];
          a[pivot * n + c] = tmp;
        }
        det = -det;
      }
      final p = a[i * n + i];
      det *= p;
      for (int r = i + 1; r < n; r++) {
        final f = a[r * n + i] / p;
        for (int c = i + 1; c < n; c++) {
          a[r * n + c] -= f * a[i * n + c];
        }
      }
    }
    out[b] = det;
  }
  return Tensor.float(out, x.shape.sublist(0, rank - 2));
}

Tensor opTrilu(Tensor x, {bool upper = true, int k = 0}) {
  final rank = x.rank;
  final rows = x.shape[rank - 2], cols = x.shape[rank - 1];
  final batch = x.length ~/ (rows * cols);
  if (x.isFloat) {
    final out = Float32List(x.length);
    final xf = x.f!;
    for (int b = 0; b < batch; b++) {
      final base = b * rows * cols;
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          final keep = upper ? j - i >= k : j - i <= k;
          if (keep) out[base + i * cols + j] = xf[base + i * cols + j];
        }
      }
    }
    return Tensor.float(out, x.shape);
  }
  final out = Int64List(x.length);
  final xi = x.intData;
  for (int b = 0; b < batch; b++) {
    final base = b * rows * cols;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        final keep = upper ? j - i >= k : j - i <= k;
        if (keep) out[base + i * cols + j] = xi[base + i * cols + j];
      }
    }
  }
  return Tensor.int64(out, x.shape);
}

/// `ScatterND` — copy of [data] with [updates] written at [indices]
/// (index tuples over the leading dims; each update fills the remaining
/// trailing block). No reduction (last write wins, per the default).
Tensor opScatterND(Tensor data, Tensor indices, Tensor updates) {
  final idxDepth = indices.shape.last;
  final nIdx = indices.length ~/ idxDepth;
  final strides = data.strides;
  int blockLen = 1;
  for (int a = idxDepth; a < data.rank; a++) {
    blockLen *= data.shape[a];
  }
  final ii = indices.asIntList();
  if (data.isFloat) {
    final out = Float32List.fromList(data.f!);
    final uf = updates.asFloatList();
    for (int t = 0; t < nIdx; t++) {
      int off = 0;
      for (int d = 0; d < idxDepth; d++) {
        var v = ii[t * idxDepth + d];
        if (v < 0) v += data.shape[d];
        off += v * strides[d];
      }
      out.setRange(off, off + blockLen, uf, t * blockLen);
    }
    return Tensor.float(out, data.shape);
  }
  final out = Int64List.fromList(data.asIntList());
  final ui = updates.asIntList();
  for (int t = 0; t < nIdx; t++) {
    int off = 0;
    for (int d = 0; d < idxDepth; d++) {
      var v = ii[t * idxDepth + d];
      if (v < 0) v += data.shape[d];
      off += v * strides[d];
    }
    out.setRange(
      off,
      off + blockLen,
      Int64List.sublistView(ui, t * blockLen, (t + 1) * blockLen),
    );
  }
  return Tensor.int64(out, data.shape);
}

/// `Tile` — repeats [x] along each axis: `out[c] = x[c % shape]`.
Tensor opTile(Tensor x, List<int> repeats) {
  final rank = x.rank;
  final outShape = [for (int a = 0; a < rank; a++) x.shape[a] * repeats[a]];
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final srcStrides = x.strides;
  final coords = List<int>.filled(rank, 0);
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  final src = isFloat ? null : x.intData;
  for (int idx = 0; idx < n; idx++) {
    int off = 0;
    for (int a = 0; a < rank; a++) {
      off += (coords[a] % x.shape[a]) * srcStrides[a];
    }
    if (isFloat) {
      outF![idx] = x.f![off];
    } else {
      outI![idx] = src![off];
    }
    for (int a = rank - 1; a >= 0; a--) {
      if (++coords[a] < outShape[a]) break;
      coords[a] = 0;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

Tensor opCastLike(Tensor x, Tensor like) {
  switch (like.dtype) {
    case DType.float32:
      return opCast(x, 1);
    case DType.int64:
      return opCast(x, 7);
    case DType.uint8:
      return Tensor.uint8(
        Uint8List.fromList([
          for (int k = 0; k < x.length; k++) x.getD(k).round().clamp(0, 255),
        ]),
        x.shape,
      );
    case DType.int8:
      return Tensor.int8(
        Int8List.fromList([
          for (int k = 0; k < x.length; k++) x.getD(k).round().clamp(-128, 127),
        ]),
        x.shape,
      );
  }
}

Tensor opEyeLike(Tensor x, {int k = 0, int? dtype}) {
  if (x.rank != 2) {
    throw ArgumentError('EyeLike requires a rank-2 input, got ${x.shape}');
  }
  final rows = x.shape[0], cols = x.shape[1];
  final type = dtype;
  if (type == 1 || (type == null && x.isFloat)) {
    final out = Float32List(rows * cols);
    for (int i = 0; i < rows; i++) {
      final j = i + k;
      if (j >= 0 && j < cols) out[i * cols + j] = 1;
    }
    return Tensor.float(out, x.shape);
  }
  if (type == 2) {
    final out = Uint8List(rows * cols);
    for (int i = 0; i < rows; i++) {
      final j = i + k;
      if (j >= 0 && j < cols) out[i * cols + j] = 1;
    }
    return Tensor.uint8(out, x.shape);
  }
  if (type == 3) {
    final out = Int8List(rows * cols);
    for (int i = 0; i < rows; i++) {
      final j = i + k;
      if (j >= 0 && j < cols) out[i * cols + j] = 1;
    }
    return Tensor.int8(out, x.shape);
  }
  final out = Int64List(rows * cols);
  for (int i = 0; i < rows; i++) {
    final j = i + k;
    if (j >= 0 && j < cols) out[i * cols + j] = 1;
  }
  return Tensor.int64(out, x.shape);
}

Tensor opCompress(Tensor x, Tensor condition, {int? axis}) {
  final selected = <int>[];
  for (int i = 0; i < condition.length; i++) {
    if (condition.getI(i) != 0) selected.add(i);
  }
  if (axis == null) {
    final n = selected.length;
    if (x.isFloat) {
      final out = Float32List(n);
      for (int i = 0; i < n; i++) {
        out[i] = x.f![selected[i]];
      }
      return Tensor.float(out, [n]);
    }
    final out = Int64List(n);
    for (int i = 0; i < n; i++) {
      out[i] = x.intData[selected[i]];
    }
    return Tensor.int64(out, [n]);
  }
  final rank = x.rank;
  final ax = axis < 0 ? axis + rank : axis;
  if (condition.length != x.shape[ax]) {
    throw ArgumentError(
      'Compress condition length ${condition.length} != axis dim ${x.shape[ax]}',
    );
  }
  final outShape = List<int>.from(x.shape)..[ax] = selected.length;
  final outer = x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final inner = x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final n = outShape.fold<int>(1, (a, b) => a * b);
  if (x.isFloat) {
    final out = Float32List(n);
    for (int o = 0; o < outer; o++) {
      for (int j = 0; j < selected.length; j++) {
        final src = (o * x.shape[ax] + selected[j]) * inner;
        final dst = (o * selected.length + j) * inner;
        out.setRange(dst, dst + inner, x.f!, src);
      }
    }
    return Tensor.float(out, outShape);
  }
  final out = Int64List(n);
  for (int o = 0; o < outer; o++) {
    for (int j = 0; j < selected.length; j++) {
      final src = (o * x.shape[ax] + selected[j]) * inner;
      final dst = (o * selected.length + j) * inner;
      out.setRange(dst, dst + inner, x.intData, src);
    }
  }
  return Tensor.int64(out, outShape);
}

Tensor opReverseSequence(
  Tensor x,
  Tensor sequenceLens,
  int timeAxis,
  int batchAxis,
) {
  final rank = x.rank;
  final tAx = timeAxis < 0 ? timeAxis + rank : timeAxis;
  final bAx = batchAxis < 0 ? batchAxis + rank : batchAxis;
  final List<num> out = x.isFloat
      ? Float32List.fromList(x.f!)
      : Int64List.fromList(x.intData);
  final strides = x.strides;
  final batchCount = x.shape[bAx];
  final timeCount = x.shape[tAx];
  final otherShape = <int>[
    for (int a = 0; a < rank; a++)
      if (a != tAx && a != bAx) x.shape[a],
  ];
  final otherCoords = List<int>.filled(otherShape.length, 0);
  final otherAxes = [
    for (int a = 0; a < rank; a++)
      if (a != tAx && a != bAx) a,
  ];
  for (int b = 0; b < batchCount; b++) {
    final len = sequenceLens.getI(b);
    final limit = len < timeCount ? len : timeCount;
    for (int j = 0; j < otherShape.length; j++) {
      otherCoords[j] = 0;
    }
    while (true) {
      for (int t = 0; t < limit ~/ 2; t++) {
        final coords = List<int>.filled(rank, 0);
        coords[bAx] = b;
        coords[tAx] = t;
        for (int j = 0; j < otherShape.length; j++) {
          coords[otherAxes[j]] = otherCoords[j];
        }
        final a = _flatIndex(coords, strides);
        coords[tAx] = limit - 1 - t;
        final c = _flatIndex(coords, strides);
        final tmp = out[a];
        out[a] = out[c];
        out[c] = tmp;
      }
      var advanced = false;
      for (int j = otherShape.length - 1; j >= 0; j--) {
        if (++otherCoords[j] < otherShape[j]) {
          advanced = true;
          break;
        }
        otherCoords[j] = 0;
      }
      if (!advanced) break;
    }
  }
  return x.isFloat
      ? Tensor.float(out as Float32List, x.shape)
      : Tensor.int64(out as Int64List, x.shape);
}

int _flatIndex(List<int> coords, List<int> strides) {
  var flat = 0;
  for (int i = 0; i < coords.length; i++) {
    flat += coords[i] * strides[i];
  }
  return flat;
}

List<int> _stridesOf(List<int> shape) {
  final s = List<int>.filled(shape.length, 1);
  for (var i = shape.length - 2; i >= 0; i--) {
    s[i] = s[i + 1] * shape[i + 1];
  }
  return s;
}

Tensor opDepthToSpace(Tensor x, int blockSize, String mode) {
  if (x.rank != 4) {
    throw ArgumentError('DepthToSpace requires rank 4, got ${x.shape}');
  }
  final n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  final b = blockSize;
  if (c % (b * b) != 0) {
    throw ArgumentError('DepthToSpace channels $c not divisible by $b^2');
  }
  final co = c ~/ (b * b);
  if (mode == 'CRD') {
    final r = x.reshape([n, co, b, b, h, w]);
    final t = opTranspose(r, [0, 1, 4, 2, 5, 3]);
    return t.reshape([n, co, h * b, w * b]);
  }
  final r = x.reshape([n, b, b, co, h, w]);
  final t = opTranspose(r, [0, 3, 4, 1, 5, 2]);
  return t.reshape([n, co, h * b, w * b]);
}

Tensor opSpaceToDepth(Tensor x, int blockSize) {
  if (x.rank != 4) {
    throw ArgumentError('SpaceToDepth requires rank 4, got ${x.shape}');
  }
  final n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  final b = blockSize;
  if (h % b != 0 || w % b != 0) {
    throw ArgumentError(
      'SpaceToDepth spatial dims ${x.shape.sublist(2)} '
      'not divisible by $b',
    );
  }
  final hb = h ~/ b, wb = w ~/ b;
  final r = x.reshape([n, c, hb, b, wb, b]);
  final t = opTranspose(r, [0, 3, 5, 1, 2, 4]);
  return t.reshape([n, c * b * b, hb, wb]);
}

Tensor opScatterElements(
  Tensor data,
  Tensor indices,
  Tensor updates,
  int axis,
  String reduction,
) {
  final rank = data.rank;
  final ax = axis < 0 ? axis + rank : axis;
  final List<num> out = data.isFloat
      ? Float32List.fromList(data.f!)
      : Int64List.fromList(data.intData);
  final strides = data.strides;
  for (int idx = 0; idx < indices.length; idx++) {
    final c = _unflatten(idx, indices.shape);
    var j = indices.getI(idx);
    if (j < 0) j += data.shape[ax];
    c[ax] = j;
    final off = _flatIndex(c, strides);
    final u = updates.getD(idx);
    if (data.isFloat) {
      switch (reduction) {
        case 'add':
          out[off] = out[off] + u;
        case 'mul':
          out[off] = out[off] * u;
        case 'min':
          out[off] = math.min(out[off], u);
        case 'max':
          out[off] = math.max(out[off], u);
        default:
          out[off] = u;
      }
    } else {
      final ui = updates.getI(idx);
      switch (reduction) {
        case 'add':
          out[off] = out[off] + ui;
        case 'mul':
          out[off] = out[off] * ui;
        case 'min':
          out[off] = math.min(out[off], ui);
        case 'max':
          out[off] = math.max(out[off], ui);
        default:
          out[off] = ui;
      }
    }
  }
  return data.isFloat
      ? Tensor.float(out as Float32List, data.shape)
      : Tensor.int64(out as Int64List, data.shape);
}

List<Tensor> opUnique(Tensor x, {int? axis, bool sorted = true}) {
  final rank = x.rank;
  final ax = axis == null ? null : (axis < 0 ? axis + rank : axis);
  final outer = ax == null
      ? 1
      : x.shape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final dim = ax == null ? x.length : x.shape[ax];
  final inner = ax == null
      ? 1
      : x.shape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);

  List<dynamic> keyOf(int j) {
    if (ax == null) return [x.isFloat ? x.getD(j) : x.getI(j)];
    return [
      for (int o = 0; o < outer; o++)
        for (int i = 0; i < inner; i++)
          x.isFloat
              ? x.getD((o * dim + j) * inner + i)
              : x.getI((o * dim + j) * inner + i),
    ];
  }

  bool same(List<dynamic> a, List<dynamic> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  int cmp(List<dynamic> a, List<dynamic> b) {
    for (int i = 0; i < a.length; i++) {
      final c = (a[i] as num).compareTo(b[i] as num);
      if (c != 0) return c;
    }
    return 0;
  }

  final uniqueKeys = <List<dynamic>>[];
  final first = <int>[];
  final counts = <int>[];
  final inverse = List<int>.filled(dim, 0);
  for (int j = 0; j < dim; j++) {
    final key = keyOf(j);
    var found = -1;
    for (int u = 0; u < uniqueKeys.length; u++) {
      if (same(uniqueKeys[u], key)) {
        found = u;
        break;
      }
    }
    if (found < 0) {
      found = uniqueKeys.length;
      uniqueKeys.add(key);
      first.add(j);
      counts.add(1);
    } else {
      counts[found]++;
    }
    inverse[j] = found;
  }

  var order = List<int>.generate(uniqueKeys.length, (i) => i);
  if (sorted) {
    order.sort((a, b) => cmp(uniqueKeys[a], uniqueKeys[b]));
  }
  final remap = List<int>.filled(order.length, 0);
  for (int i = 0; i < order.length; i++) {
    remap[order[i]] = i;
  }
  final List<num> y = x.isFloat
      ? Float32List(order.length * outer * inner)
      : Int64List(order.length * outer * inner);
  for (int u = 0; u < order.length; u++) {
    final srcJ = first[order[u]];
    for (int o = 0; o < outer; o++) {
      for (int i = 0; i < inner; i++) {
        final src = (o * dim + srcJ) * inner + i;
        final dst = (o * order.length + u) * inner + i;
        if (x.isFloat) {
          y[dst] = x.f![src];
        } else {
          y[dst] = x.intData[src];
        }
      }
    }
  }
  final yShape = ax == null
      ? [order.length]
      : (List<int>.from(x.shape)..[ax] = order.length);
  final indicesOut = Int64List.fromList([for (final u in order) first[u]]);
  final inverseOut = Int64List.fromList([
    for (int j = 0; j < dim; j++) remap[inverse[j]],
  ]);
  final countsOut = Int64List.fromList([for (final u in order) counts[u]]);
  return [
    x.isFloat
        ? Tensor.float(y as Float32List, yShape)
        : Tensor.int64(y as Int64List, yShape),
    Tensor.int64(indicesOut, [order.length]),
    Tensor.int64(inverseOut, [dim]),
    Tensor.int64(countsOut, [order.length]),
  ];
}

// ---------------------------------------------------------------------------
// Quantization (QDQ format)
// ---------------------------------------------------------------------------

/// Round half to even (banker's rounding), as the ONNX quantization ops
/// require — Dart's `round()` rounds half away from zero.
int _roundEven(double v) {
  final f = v.floorToDouble();
  final frac = v - f;
  if (frac > 0.5) return f.toInt() + 1;
  if (frac < 0.5) return f.toInt();
  final i = f.toInt();
  return i.isEven ? i : i + 1;
}

/// Per-axis channel index of flat element [k]: scale/zero-point entry to use
/// when they are 1-D along [axis].
int _channelOf(int k, List<int> shape, int axis) {
  int stride = 1;
  for (int a = shape.length - 1; a > axis; a--) {
    stride *= shape[a];
  }
  return (k ~/ stride) % shape[axis];
}

/// `QuantizeLinear`: `y = saturate(roundEven(x / scale) + zeroPoint)`,
/// per-tensor or per-axis. [lo]/[hi] are the saturation bounds of the output
/// type (from the zero-point tensor's dtype: uint8 → 0..255, int8 →
/// -128..127).
Tensor opQuantizeLinear(
  Tensor x,
  Tensor scale,
  Tensor? zeroPoint, {
  int axis = 1,
  required int lo,
  required int hi,
}) {
  final n = x.length;
  // Compact output: int8 when the range is signed, uint8 otherwise.
  final signed = lo < 0;
  final outI8 = signed ? Int8List(n) : null;
  final outU8 = signed ? null : Uint8List(n);
  final perAxis = scale.length > 1;
  final ax = axis < 0 ? axis + x.rank : axis;
  final sf = scale.asFloatList();
  final zp = zeroPoint?.asIntList();
  for (int k = 0; k < n; k++) {
    final c = perAxis ? _channelOf(k, x.shape, ax) : 0;
    final q = _roundEven(x.getD(k) / sf[c]) + (zp == null ? 0 : zp[c]);
    final v = q < lo ? lo : (q > hi ? hi : q);
    if (signed) {
      outI8![k] = v;
    } else {
      outU8![k] = v;
    }
  }
  return signed ? Tensor.int8(outI8!, x.shape) : Tensor.uint8(outU8!, x.shape);
}

/// `DequantizeLinear`: `y = (x - zeroPoint) * scale`, per-tensor or per-axis.
Tensor opDequantizeLinear(
  Tensor x,
  Tensor scale,
  Tensor? zeroPoint, {
  int axis = 1,
}) {
  final n = x.length;
  final out = Float32List(n);
  final perAxis = scale.length > 1;
  final ax = axis < 0 ? axis + x.rank : axis;
  final sf = scale.asFloatList();
  final zp = zeroPoint?.asIntList();
  final xi = x.intData;
  for (int k = 0; k < n; k++) {
    final c = perAxis ? _channelOf(k, x.shape, ax) : 0;
    out[k] = (xi[k] - (zp == null ? 0 : zp[c])) * sf[c];
  }
  return Tensor.float(out, x.shape);
}

/// `DynamicQuantizeLinear`: computes uint8 scale/zero-point from the data
/// range (always spanning 0) and quantizes. Returns `[y, scale, zeroPoint]`.
List<Tensor> opDynamicQuantizeLinear(Tensor x) {
  final xf = x.asFloatList();
  double mn = 0, mx = 0; // range must include 0 per spec
  for (final v in xf) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  // The reference computes scale and zero point in float32 — matching that
  // exactly matters, because a scale differing in the last bit shifts every
  // quantized value (the downstream integer math then diverges everywhere).
  final f32 = Float32List(1);
  f32[0] = (mx - mn) / 255.0;
  final scale = f32[0].toDouble();
  f32[0] = scale == 0 ? 0 : -mn / scale;
  final zpF = f32[0].toDouble();
  final zp = _roundEven(zpF.clamp(0.0, 255.0));
  final out = Uint8List(xf.length);
  for (int k = 0; k < xf.length; k++) {
    f32[0] = scale == 0 ? 0 : xf[k] / scale;
    final q = scale == 0 ? zp : _roundEven(f32[0].toDouble()) + zp;
    out[k] = q < 0 ? 0 : (q > 255 ? 255 : q);
  }
  return [
    Tensor.uint8(out, x.shape),
    Tensor.scalarFloat(scale),
    Tensor.scalarInt(zp),
  ];
}

/// `Pad` — constant / reflect / edge modes over any rank. [pads] is
/// `[begin_0..begin_r, end_0..end_r]`; with [axes] (opset 18+) it lists only
/// the entries for those axes, in the same begin-then-end layout.
Tensor opPad(
  Tensor x,
  List<int> pads, {
  String mode = 'constant',
  double constantValue = 0,
  List<int>? axes,
}) {
  final rank = x.rank;
  final beg = List<int>.filled(rank, 0);
  final end = List<int>.filled(rank, 0);
  if (axes == null) {
    for (int a = 0; a < rank; a++) {
      beg[a] = pads[a];
      end[a] = pads[rank + a];
    }
  } else {
    for (int k = 0; k < axes.length; k++) {
      final a = axes[k] < 0 ? axes[k] + rank : axes[k];
      beg[a] = pads[k];
      end[a] = pads[axes.length + k];
    }
  }
  final outShape = [
    for (int a = 0; a < rank; a++) x.shape[a] + beg[a] + end[a],
  ];
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final srcStrides = x.strides;
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);

  // Row-wise copy: map each output coordinate back to a source coordinate
  // once per axis, then move whole contiguous runs with setRange. The generic
  // per-element odometer below is O(n·rank) and dominated multi-hundred-ms
  // reflect paddings on 1024² activations.
  final maps = [
    for (int a = 0; a < rank; a++)
      _padIndexMap(x.shape[a], beg[a], end[a], mode),
  ];
  final last = rank - 1;
  final inner = x.shape[last];
  final outInner = outShape[last];
  final outerCount = outInner == 0 ? 0 : n ~/ outInner;
  final coords = List<int>.filled(rank - 1, 0);
  final lastMap = maps[last];
  final srcF = isFloat ? x.f! : null;
  final srcI = isFloat ? null : x.intData;
  final cv = constantValue.toInt();
  final hasBorder = beg[last] != 0 || end[last] != 0;
  for (int o = 0; o < outerCount; o++) {
    int srcRow = 0;
    bool valid = true;
    for (int a = 0; a < last; a++) {
      final i = maps[a][coords[a]];
      if (i < 0) {
        valid = false;
        break;
      }
      srcRow += i * srcStrides[a];
    }
    final dst = o * outInner;
    if (isFloat) {
      final out = outF!;
      if (!valid) {
        out.fillRange(dst, dst + outInner, constantValue);
      } else if (!hasBorder) {
        out.setRange(dst, dst + outInner, srcF!, srcRow);
      } else {
        out.setRange(dst + beg[last], dst + beg[last] + inner, srcF!, srcRow);
        for (int j = 0; j < beg[last]; j++) {
          final i = lastMap[j];
          out[dst + j] = i < 0 ? constantValue : srcF[srcRow + i];
        }
        for (int j = beg[last] + inner; j < outInner; j++) {
          final i = lastMap[j];
          out[dst + j] = i < 0 ? constantValue : srcF[srcRow + i];
        }
      }
    } else {
      final out = outI!;
      if (!valid) {
        out.fillRange(dst, dst + outInner, cv);
      } else if (!hasBorder) {
        out.setRange(dst, dst + outInner, srcI!, srcRow);
      } else {
        out.setRange(dst + beg[last], dst + beg[last] + inner, srcI!, srcRow);
        for (int j = 0; j < beg[last]; j++) {
          final i = lastMap[j];
          out[dst + j] = i < 0 ? cv : srcI[srcRow + i];
        }
        for (int j = beg[last] + inner; j < outInner; j++) {
          final i = lastMap[j];
          out[dst + j] = i < 0 ? cv : srcI[srcRow + i];
        }
      }
    }
    for (int a = last - 1; a >= 0; a--) {
      if (++coords[a] < outShape[a]) break;
      coords[a] = 0;
    }
  }
  return isFloat
      ? Tensor.float(outF!, outShape)
      : Tensor.int64(outI!, outShape);
}

/// Maps every output index along one axis to a source index, or -1 when
/// `constant` mode leaves it in the pad. Mirrors ONNX reflect/edge semantics.
List<int> _padIndexMap(int dim, int before, int after, String mode) {
  final len = dim + before + after;
  final map = List<int>.filled(len, 0);
  if (mode == 'reflect' && dim <= 1 && (before > 0 || after > 0)) {
    // ONNX requires reflect pad < input dim; without this the mirror loop
    // below never converges for a singleton axis.
    throw ArgumentError.value(
      dim,
      'dim',
      'reflect padding requires an axis longer than 1',
    );
  }
  for (int j = 0; j < len; j++) {
    int i = j - before;
    switch (mode) {
      case 'reflect':
        while (i < 0 || i >= dim) {
          if (i < 0) i = -i;
          if (i >= dim) i = 2 * dim - 2 - i;
        }
      case 'edge':
        if (i < 0) i = 0;
        if (i >= dim) i = dim - 1;
      default:
        if (i < 0 || i >= dim) i = -1;
    }
    map[j] = i;
  }
  return map;
}

/// `Range(start, limit, delta)` — a 1-D sequence, int64 if the bounds are int.
Tensor opRange(Tensor start, Tensor limit, Tensor delta) {
  final s = start.getD(0), l = limit.getD(0), d = delta.getD(0);
  final n = math.max(0, ((l - s) / d).ceil());
  final bothInt = !start.isFloat && !delta.isFloat;
  if (bothInt) {
    final out = Int64List(n);
    for (int k = 0; k < n; k++) {
      out[k] = (start.getI(0) + k * delta.getI(0));
    }
    return Tensor.int64(out, [n]);
  }
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    out[k] = s + k * d;
  }
  return Tensor.float(out, [n]);
}

/// `ConstantOfShape(shape)` filled with [value] (a 1-element tensor; default 0).
Tensor opConstantOfShape(Tensor shapeT, Tensor? value) {
  final shape = shapeT.asIntList().toList();
  final n = shape.fold<int>(1, (a, b) => a * b);
  if (value != null && !value.isFloat) {
    final v = value.getI(0);
    return Tensor.int64(Int64List(n)..fillRange(0, n, v), shape);
  }
  final v = value != null ? value.getD(0) : 0.0;
  return Tensor.float(Float32List(n)..fillRange(0, n, v), shape);
}

/// `ReduceSum` over [axes] (all axes if null), matching [opReduceMean] shape
/// semantics but summing rather than averaging.
Tensor opReduceSum(Tensor x, List<int>? axes, bool keepdims) {
  final rank = x.shape.length;
  final ax = (axes ?? List<int>.generate(rank, (k) => k))
      .map((a) => a < 0 ? a + rank : a)
      .toSet();
  final outShapeFull = [
    for (int k = 0; k < rank; k++) ax.contains(k) ? 1 : x.shape[k],
  ];
  final n = outShapeFull.fold<int>(1, (a, b) => a * b);
  final outStridesFull = Tensor.filledFloat(outShapeFull, 0).strides;
  final shape = keepdims
      ? outShapeFull
      : [
          for (int k = 0; k < rank; k++)
            if (!ax.contains(k)) x.shape[k],
        ];
  if (!x.isFloat) {
    final sums = Int64List(n);
    for (int idx = 0; idx < x.length; idx++) {
      final coords = _unflatten(idx, x.shape);
      var outFlat = 0;
      for (int k = 0; k < rank; k++) {
        outFlat += (ax.contains(k) ? 0 : coords[k]) * outStridesFull[k];
      }
      sums[outFlat] += x.getI(idx);
    }
    return Tensor.int64(sums, shape);
  }
  final sums = Float64List(n);
  for (int idx = 0; idx < x.length; idx++) {
    final coords = _unflatten(idx, x.shape);
    var outFlat = 0;
    for (int k = 0; k < rank; k++) {
      outFlat += (ax.contains(k) ? 0 : coords[k]) * outStridesFull[k];
    }
    sums[outFlat] += x.getD(idx);
  }
  final out = Float32List(n);
  for (int k = 0; k < n; k++) {
    out[k] = sums[k];
  }
  return Tensor.float(out, shape);
}

Tensor opReduceL1(Tensor x, List<int>? axes, bool keepdims) =>
    opReduceSum(opAbs(x), axes, keepdims);

Tensor opReduceLogSum(Tensor x, List<int>? axes, bool keepdims) {
  final sum = opReduceSum(x, axes, keepdims);
  final out = Float32List(sum.length);
  final sf = sum.asFloatList();
  for (int k = 0; k < out.length; k++) {
    out[k] = math.log(sf[k]);
  }
  return Tensor.float(out, sum.shape);
}

Tensor opReduceLogSumExp(Tensor x, List<int>? axes, bool keepdims) {
  final max = opReduceMinMax(x, axes, true, isMax: true);
  final centered = opSub(x, max);
  final exp = opExp(centered);
  final sum = opReduceSum(exp, axes, true);
  final out = Float32List(sum.length);
  final sf = sum.asFloatList();
  final mf = max.asFloatList();
  for (int k = 0; k < out.length; k++) {
    out[k] = math.log(sf[k]) + mf[k];
  }
  var result = Tensor.float(out, sum.shape);
  if (!keepdims) {
    final rank = x.shape.length;
    final reduced = axes ?? List<int>.generate(rank, (k) => k);
    result = opSqueeze(result, reduced);
  }
  return result;
}

/// `GatherElements` — output has the shape of [indices]; each element picks
/// `data` at that coordinate with the [axis] index replaced by the index value.
Tensor opGatherElements(Tensor data, Tensor indices, int axis) {
  final rank = data.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final dStrides = data.strides;
  final n = indices.length;
  final isFloat = data.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  for (int idx = 0; idx < n; idx++) {
    final coords = _unflatten(idx, indices.shape);
    var j = indices.getI(idx);
    if (j < 0) j += data.shape[ax];
    coords[ax] = j;
    int flat = 0;
    for (int k = 0; k < rank; k++) {
      flat += coords[k] * dStrides[k];
    }
    if (isFloat) {
      outF![idx] = data.f![flat];
    } else {
      outI![idx] = data.intData[flat];
    }
  }
  return isFloat
      ? Tensor.float(outF!, indices.shape)
      : Tensor.int64(outI!, indices.shape);
}

/// `CumSum` along [axis] (inclusive, forward by default; [exclusive] shifts the
/// sum, [reverse] accumulates from the end) — used e.g. to derive position ids
/// from an attention mask.
Tensor opCumSum(
  Tensor x,
  int axis, {
  bool exclusive = false,
  bool reverse = false,
}) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final axisStride = x.strides[ax];
  final axisSize = x.shape[ax];
  final n = x.length;
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  // Float accumulation rounds to float32 at every step, matching ORT —
  // long cumulative sums (e.g. vocoder phase over tens of thousands of
  // samples) otherwise drift from the oracle and chaos-amplify through
  // sin() downstream.
  final f32 = Float32List(1);
  for (int i = 0; i < n; i++) {
    if ((i ~/ axisStride) % axisSize != 0) continue; // only line starts
    var accF = 0.0;
    var accI = 0;
    for (int s = 0; s < axisSize; s++) {
      final k = reverse ? axisSize - 1 - s : s;
      final idx = i + k * axisStride;
      final v = x.getD(idx);
      if (exclusive) {
        if (isFloat) {
          outF![idx] = accF;
        } else {
          outI![idx] = accI;
        }
      }
      f32[0] = accF + v;
      accF = f32[0];
      accI += v.round();
      if (!exclusive) {
        if (isFloat) {
          outF![idx] = accF;
        } else {
          outI![idx] = accI;
        }
      }
    }
  }
  return isFloat ? Tensor.float(outF!, x.shape) : Tensor.int64(outI!, x.shape);
}

Tensor opCumProd(
  Tensor x,
  int axis, {
  bool exclusive = false,
  bool reverse = false,
}) {
  final rank = x.shape.length;
  final ax = axis < 0 ? axis + rank : axis;
  final axisStride = x.strides[ax];
  final axisSize = x.shape[ax];
  final n = x.length;
  final isFloat = x.isFloat;
  final outF = isFloat ? Float32List(n) : null;
  final outI = isFloat ? null : Int64List(n);
  final f32 = Float32List(1);
  for (int i = 0; i < n; i++) {
    if ((i ~/ axisStride) % axisSize != 0) continue;
    var accF = 1.0;
    var accI = 1;
    for (int s = 0; s < axisSize; s++) {
      final k = reverse ? axisSize - 1 - s : s;
      final idx = i + k * axisStride;
      final v = x.getD(idx);
      if (exclusive) {
        if (isFloat) {
          outF![idx] = accF;
        } else {
          outI![idx] = accI;
        }
      }
      f32[0] = accF * v;
      accF = f32[0];
      accI *= v.round();
      if (!exclusive) {
        if (isFloat) {
          outF![idx] = accF;
        } else {
          outI![idx] = accI;
        }
      }
    }
  }
  return isFloat ? Tensor.float(outF!, x.shape) : Tensor.int64(outI!, x.shape);
}

/// OneHot: inserts a depth-sized axis into an integer index tensor. Nullable
/// positions are not special-cased — ORT and the ONNX reference implementation
/// both fold every index through `mod(indices, depth)`.
Tensor opOneHot(Tensor indices, Tensor depth, Tensor values, int axis) {
  final rank = indices.rank;
  var ax = axis < 0 ? axis + rank + 1 : axis;
  if (ax < 0 || ax > rank) {
    throw ArgumentError('OneHot: axis $axis is out of range for rank $rank');
  }
  if (depth.length < 1) throw ArgumentError('OneHot: depth must be a scalar');
  final d = depth.asIntList().first;
  if (d < 0) throw ArgumentError('OneHot: depth must be nonnegative');
  if (values.length < 2) {
    throw ArgumentError('OneHot: values must have exactly two elements');
  }
  final off = values.getD(0), on = values.getD(1);
  final mid = indices.shape.sublist(ax).fold(1, (int a, int b) => a * b);
  final shape = [
    ...indices.shape.sublist(0, ax),
    d,
    ...indices.shape.sublist(ax),
  ];
  final isFloat = values.isFloat;
  final outF = isFloat ? Float32List(indices.length * d) : null;
  final outI = isFloat ? null : Int64List(indices.length * d);
  for (var p = 0; p < indices.length; p++) {
    if ((p & 4095) == 0) activeOnnxControl?.checkpoint();
    final cls = indices.getI(p) % d;
    final base = (p ~/ mid) * (d * mid) + (p % mid);
    for (var c = 0; c < d; c++) {
      final idx = base + c * mid;
      if (isFloat) {
        outF![idx] = c == cls ? on : off;
      } else {
        outI![idx] = (c == cls ? on : off).round();
      }
    }
  }
  return isFloat ? Tensor.float(outF!, shape) : Tensor.int64(outI!, shape);
}

List<num> _copyTensorData(Tensor x) => switch (x.dtype) {
  DType.float32 => Float32List.fromList(x.f!),
  DType.int64 => Int64List.fromList(x.i!),
  DType.uint8 => Uint8List.fromList(x.u8!),
  DType.int8 => Int8List.fromList(x.i8!),
};

Tensor _tensorFromData(List<num> data, List<int> shape, DType dtype) =>
    switch (dtype) {
      DType.float32 => Tensor.float(
        Float32List.fromList(data.map((v) => v.toDouble()).toList()),
        shape,
      ),
      DType.int64 => Tensor.int64(
        Int64List.fromList(data.map((v) => v.toInt()).toList()),
        shape,
      ),
      DType.uint8 => Tensor.uint8(
        Uint8List.fromList(data.map((v) => v.toInt()).toList()),
        shape,
      ),
      DType.int8 => Tensor.int8(
        Int8List.fromList(data.map((v) => v.toInt()).toList()),
        shape,
      ),
    };

Tensor opCenterCropPad(Tensor x, Tensor shapeInput, List<int>? axes) {
  final targetDims = shapeInput.asIntList();
  final ax = axes ?? List<int>.generate(targetDims.length, (i) => i);
  if (ax.length != targetDims.length) {
    throw ArgumentError('CenterCropPad: axes and shape length differ');
  }
  if (targetDims.any((d) => d < 0)) {
    throw ArgumentError('CenterCropPad: target dimensions must be nonnegative');
  }
  final normalized = [for (final a in ax) a < 0 ? a + x.rank : a];
  final outShape = List<int>.from(x.shape);
  for (var i = 0; i < normalized.length; i++) {
    final a = normalized[i];
    if (a < 0 || a >= x.rank) {
      throw ArgumentError('CenterCropPad: axis $a is out of range');
    }
    outShape[a] = targetDims[i];
  }
  final outLen = outShape.fold<int>(1, (a, b) => a * b);
  final out = List<num>.filled(outLen, 0);
  final outStrides = _stridesOf(outShape);
  for (var idx = 0; idx < x.length; idx++) {
    if ((idx & 4095) == 0) activeOnnxControl?.checkpoint();
    final coords = _unflatten(idx, x.shape);
    var skip = false;
    for (var i = 0; i < normalized.length; i++) {
      final a = normalized[i], target = targetDims[i], source = x.shape[a];
      if (target >= source) {
        coords[a] += (target - source) ~/ 2;
      } else {
        coords[a] -= (source - target) ~/ 2;
        if (coords[a] < 0 || coords[a] >= target) {
          skip = true;
          break;
        }
      }
    }
    if (skip) continue;
    out[_flatIndex(coords, outStrides)] = x.isFloat ? x.getD(idx) : x.getI(idx);
  }
  return _tensorFromData(out, outShape, x.dtype);
}

Tensor opCol2Im(
  Tensor x,
  Tensor imageShapeInput,
  Tensor blockShapeInput, {
  List<int>? strides,
  List<int>? dilations,
  List<int>? pads,
}) {
  final imageShape = imageShapeInput.asIntList();
  final blockShape = blockShapeInput.asIntList();
  final rank = imageShape.length;
  if (rank == 0 || blockShape.length != rank) {
    throw ArgumentError('Col2Im: image and block ranks must match');
  }
  final stride = strides ?? List<int>.filled(rank, 1);
  final dilation = dilations ?? List<int>.filled(rank, 1);
  final pad = pads ?? List<int>.filled(rank * 2, 0);
  if (stride.length != rank ||
      dilation.length != rank ||
      pad.length != rank * 2) {
    throw ArgumentError('Col2Im: attribute rank mismatch');
  }
  final blockProd = blockShape.fold<int>(1, (a, b) => a * b);
  final imageProd = imageShape.fold<int>(1, (a, b) => a * b);
  final grid = <int>[
    for (var d = 0; d < rank; d++)
      (imageShape[d] +
                  pad[d] +
                  pad[d + rank] -
                  dilation[d] * (blockShape[d] - 1) -
                  1) ~/
              stride[d] +
          1,
  ];
  final gridProd = grid.fold<int>(1, (a, b) => a * b);
  if (x.rank != 3 || x.shape[2] != gridProd) {
    throw ArgumentError('Col2Im: input shape ${x.shape} does not match grid');
  }
  final n = x.shape[0];
  final channels = x.shape[1] ~/ blockProd;
  if (channels * blockProd != x.shape[1]) {
    throw ArgumentError('Col2Im: channels are not divisible by block size');
  }
  final outLen = n * channels * imageProd;
  final out = List<num>.filled(outLen, 0);
  final imageStrides = _stridesOf(imageShape);
  for (var ni = 0; ni < n; ni++) {
    for (var c = 0; c < channels; c++) {
      for (var block = 0; block < gridProd; block++) {
        final blockCoords = _unflatten(block, grid);
        for (var k = 0; k < blockProd; k++) {
          final kernelCoords = _unflatten(k, blockShape);
          var inside = true;
          for (var d = 0; d < rank; d++) {
            kernelCoords[d] =
                blockCoords[d] * stride[d] -
                pad[d] +
                kernelCoords[d] * dilation[d];
            if (kernelCoords[d] < 0 || kernelCoords[d] >= imageShape[d]) {
              inside = false;
              break;
            }
          }
          if (!inside) continue;
          final src = ((ni * channels + c) * blockProd + k) * gridProd + block;
          final dst =
              (ni * channels + c) * imageProd +
              _flatIndex(kernelCoords, imageStrides);
          out[dst] = out[dst] + (x.isFloat ? x.getD(src) : x.getI(src));
        }
      }
    }
  }
  return _tensorFromData(out, [n, channels, ...imageShape], x.dtype);
}

Tensor opMaxUnpool(
  Tensor x,
  Tensor indices,
  List<int> kernelShape, {
  List<int>? strides,
  List<int>? pads,
  Tensor? outputShape,
}) {
  final rank = x.rank;
  final spatialRank = rank - 2;
  final kernel = kernelShape;
  final stride = strides ?? kernel;
  final pad = pads ?? List<int>.filled(spatialRank * 2, 0);
  if (kernel.length != spatialRank ||
      stride.length != spatialRank ||
      pad.length != spatialRank * 2) {
    throw ArgumentError('MaxUnpool: attribute rank mismatch');
  }
  List<int> outShape;
  if (outputShape != null) {
    final requested = outputShape.asIntList();
    outShape = requested.length == rank
        ? requested
        : [x.shape[0], x.shape[1], ...requested];
  } else {
    outShape = [
      x.shape[0],
      x.shape[1],
      for (var d = 0; d < spatialRank; d++)
        (x.shape[d + 2] - 1) * stride[d] +
            kernel[d] -
            pad[d] -
            pad[d + spatialRank],
    ];
  }
  final out = List<num>.filled(outShape.fold<int>(1, (a, b) => a * b), 0);
  final perBatch = x.length ~/ (x.shape[0] * x.shape[1]);
  final outSpatial = outShape.sublist(2).fold<int>(1, (a, b) => a * b);
  for (var idx = 0; idx < x.length; idx++) {
    final n = idx ~/ (x.shape[1] * perBatch);
    final c = (idx ~/ perBatch) % x.shape[1];
    final target = indices.getI(idx);
    if (target < 0 || target >= outSpatial) {
      throw ArgumentError('MaxUnpool: index $target is out of range');
    }
    out[(n * x.shape[1] + c) * outSpatial + target] = x.getD(idx);
  }
  return _tensorFromData(out, outShape, x.dtype);
}

Tensor opMaxRoiPool(
  Tensor x,
  Tensor rois,
  List<int> pooledShape, {
  double spatialScale = 1.0,
}) {
  if (x.rank != 4 || pooledShape.length != 2) {
    throw ArgumentError('MaxRoiPool: expected NCHW input and 2-D pooled shape');
  }
  final n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];
  final numRois = rois.shape[0];
  final ph = pooledShape[0], pw = pooledShape[1];
  final out = List<num>.filled(numRois * c * ph * pw, 0);
  for (var r = 0; r < numRois; r++) {
    final batch = rois.getD(r * 5).round();
    if (batch < 0 || batch >= n) {
      throw ArgumentError('MaxRoiPool: batch index $batch is out of range');
    }
    final startW = (rois.getD(r * 5 + 1) * spatialScale).round();
    final startH = (rois.getD(r * 5 + 2) * spatialScale).round();
    final endW = (rois.getD(r * 5 + 3) * spatialScale).round();
    final endH = (rois.getD(r * 5 + 4) * spatialScale).round();
    final roiW = math.max(endW - startW + 1, 1);
    final roiH = math.max(endH - startH + 1, 1);
    for (var ch = 0; ch < c; ch++) {
      for (var py = 0; py < ph; py++) {
        final hStart = startH + (py * roiH ~/ ph);
        final hEnd = math.min(startH + ((py + 1) * roiH ~/ ph), h);
        for (var px = 0; px < pw; px++) {
          final wStart = startW + (px * roiW ~/ pw);
          final wEnd = math.min(startW + ((px + 1) * roiW ~/ pw), w);
          var best = double.negativeInfinity;
          for (var y = math.max(hStart, 0); y < hEnd; y++) {
            for (var xx = math.max(wStart, 0); xx < wEnd; xx++) {
              final v = x.getD(((batch * c + ch) * h + y) * w + xx);
              if (v > best) best = v;
            }
          }
          out[((r * c + ch) * ph + py) * pw + px] =
              best == double.negativeInfinity ? 0.0 : best;
        }
      }
    }
  }
  return Tensor.float(Float32List.fromList(out.cast<double>()), [
    numRois,
    c,
    ph,
    pw,
  ]);
}

Tensor opMelWeightMatrix(
  Tensor numMelBins,
  Tensor dftLength,
  Tensor sampleRate,
  Tensor lowerEdge,
  Tensor upperEdge, {
  int outputDataType = 1,
}) {
  final bins = numMelBins.getI(0);
  final dft = dftLength.getI(0);
  final rate = sampleRate.getI(0);
  final low = lowerEdge.getD(0);
  final high = upperEdge.getD(0);
  final specBins = dft ~/ 2 + 1;
  final melLow = 2595 * math.log(1 + low / 700) / math.ln10;
  final melHigh = 2595 * math.log(1 + high / 700) / math.ln10;
  final step = (melHigh - melLow) / (bins + 2);
  final edges = <int>[
    for (var i = 0; i < bins + 2; i++)
      (((dft + 1) * (700 * (math.pow(10, (melLow + i * step) / 2595) - 1))) ~/
              rate)
          .toInt(),
  ];
  final matrix = List<double>.filled(specBins * bins, 0);
  for (var i = 0; i < bins; i++) {
    final lower = edges[i], center = edges[i + 1], higher = edges[i + 2];
    final lowToCenter = center - lower;
    if (lowToCenter == 0) {
      if (center >= 0 && center < specBins) {
        matrix[center * bins + i] = 1;
      }
    } else {
      for (var j = lower; j <= center; j++) {
        if (j >= 0 && j < specBins) {
          matrix[j * bins + i] = (j - lower) / lowToCenter;
        }
      }
    }
    final centerToHigh = higher - center;
    if (centerToHigh > 0) {
      for (var j = center; j < higher; j++) {
        if (j >= 0 && j < specBins) {
          matrix[j * bins + i] = (higher - j) / centerToHigh;
        }
      }
    }
  }
  final shape = [specBins, bins];
  if (outputDataType == 2) {
    return Tensor.uint8(
      Uint8List.fromList(matrix.map((v) => v.round()).toList()),
      shape,
    );
  }
  if (outputDataType == 3) {
    return Tensor.int8(
      Int8List.fromList(matrix.map((v) => v.round()).toList()),
      shape,
    );
  }
  if (outputDataType == 7) {
    return Tensor.int64(
      Int64List.fromList(matrix.map((v) => v.round()).toList()),
      shape,
    );
  }
  return Tensor.float(Float32List.fromList(matrix), shape);
}

Tensor opTensorScatter(
  Tensor past,
  Tensor update, {
  Tensor? writeIndices,
  int axis = -2,
  String mode = 'linear',
}) {
  final ax = axis < 0 ? axis + past.rank : axis;
  if (ax < 0 || ax >= past.rank) {
    throw ArgumentError('TensorScatter: axis $axis is out of range');
  }
  if (update.rank != past.rank) {
    throw ArgumentError('TensorScatter: cache and update ranks differ');
  }
  final out = _copyTensorData(past);
  final prefixShape = [
    for (var d = 0; d < past.rank; d++)
      if (d != ax) past.shape[d],
  ];
  final prefixLen = prefixShape.fold<int>(1, (a, b) => a * b);
  final pastStrides = past.strides;
  final updateStrides = update.strides;
  final maxSeq = past.shape[ax];
  final updateSeq = update.shape[ax];
  final prefixStrides = [
    for (var d = 0; d < past.rank; d++)
      if (d != ax) pastStrides[d],
  ];
  final updatePrefixStrides = [
    for (var d = 0; d < past.rank; d++)
      if (d != ax) updateStrides[d],
  ];
  for (var prefix = 0; prefix < prefixLen; prefix++) {
    final prefixCoords = _unflatten(prefix, prefixShape);
    final base = _flatIndex(prefixCoords, prefixStrides);
    final updateBase = _flatIndex(prefixCoords, updatePrefixStrides);
    final write = writeIndices == null
        ? 0
        : writeIndices.length == 1
        ? writeIndices.getI(0)
        : writeIndices.getI(prefixCoords.isEmpty ? 0 : prefixCoords[0]);
    for (var s = 0; s < updateSeq; s++) {
      var cacheSeq = write + s;
      if (mode == 'circular') {
        cacheSeq %= maxSeq;
      } else if (cacheSeq < 0 || cacheSeq >= maxSeq) {
        continue;
      }
      final src = updateBase + s * updateStrides[ax];
      final dst = base + cacheSeq * pastStrides[ax];
      out[dst] = past.isFloat ? update.getD(src) : update.getI(src);
    }
  }
  return _tensorFromData(out, past.shape, past.dtype);
}

Tensor opAffineGrid(Tensor theta, Tensor size, {bool alignCorners = false}) {
  if (theta.rank != 3 || size.rank != 1 || size.length < 4 || size.length > 5) {
    throw ArgumentError(
      'AffineGrid: expected theta NxMx(M+1) and a 4/5-element size vector',
    );
  }
  final n = theta.shape[0];
  final sizeValues = size.asIntList();
  final dims = sizeValues.length - 2;
  if ((dims == 2 && theta.shape[1] != 2) ||
      (dims == 3 && theta.shape[1] != 3) ||
      theta.shape[2] != theta.shape[1] + 1) {
    throw ArgumentError(
      'AffineGrid: theta shape ${theta.shape} does not match size',
    );
  }
  final spatial = sizeValues.sublist(2);
  final outShape = [n, ...spatial, dims];
  final out = Float32List(n * spatial.fold<int>(1, (a, b) => a * b) * dims);
  final coords = List<double>.filled(dims, 0);
  for (var b = 0; b < n; b++) {
    for (var flat = 0; flat < spatial.fold<int>(1, (a, b) => a * b); flat++) {
      final pos = _unflatten(flat, spatial);
      for (var d = 0; d < dims; d++) {
        final sd = dims - 1 - d;
        coords[d] = alignCorners
            ? (spatial[sd] == 1 ? 0.0 : -1 + 2 * pos[sd] / (spatial[sd] - 1))
            : -1 + (2 * pos[sd] + 1) / spatial[sd];
      }
      for (var r = 0; r < dims; r++) {
        var v = theta.getD((b * dims + r) * (dims + 1) + dims);
        for (var d = 0; d < dims; d++) {
          v += theta.getD((b * dims + r) * (dims + 1) + d) * coords[d];
        }
        out[(b * spatial.fold<int>(1, (a, b) => a * b) + flat) * dims + r] = v;
      }
    }
  }
  return Tensor.float(out, outShape);
}

Tensor opDft(
  Tensor input,
  Tensor? dftLength,
  Tensor? axisInput, {
  bool onesided = false,
  bool inverse = false,
}) {
  final rank = input.rank;
  final complex = input.shape.last;
  if (complex != 1 && complex != 2) {
    throw ArgumentError('DFT: last dimension must be 1 or 2');
  }
  final signalShape = input.shape.sublist(0, rank - 1);
  final axis = axisInput == null ? signalShape.length - 1 : axisInput.getI(0);
  final ax = axis < 0 ? axis + signalShape.length : axis;
  if (ax < 0 || ax >= signalShape.length) {
    throw ArgumentError('DFT: axis $axis is out of range');
  }
  final readN = signalShape[ax];
  final n = dftLength == null
      ? (inverse && onesided ? 2 * (readN - 1) : readN)
      : dftLength.getI(0);
  if (n <= 0) throw ArgumentError('DFT: dft_length must be positive');
  final outN = inverse ? (onesided ? n : n) : (onesided ? n ~/ 2 + 1 : n);
  final outComplex = inverse && onesided ? 1 : 2;
  final outShape = List<int>.from(signalShape)
    ..[ax] = outN
    ..add(outComplex);
  final outer = signalShape.sublist(0, ax).fold<int>(1, (a, b) => a * b);
  final inner = signalShape.sublist(ax + 1).fold<int>(1, (a, b) => a * b);
  final out = Float32List(outer * outN * inner * outComplex);
  final re = Float64List(readN);
  final im = Float64List(readN);
  for (var o = 0; o < outer; o++) {
    for (var i = 0; i < inner; i++) {
      for (var k = 0; k < readN; k++) {
        final base = ((o * readN + k) * inner + i) * complex;
        re[k] = input.getD(base);
        im[k] = complex == 2 ? input.getD(base + 1) : 0;
      }
      for (var k = 0; k < outN; k++) {
        var sr = 0.0, si = 0.0;
        if (!inverse) {
          for (var j = 0; j < n; j++) {
            final jr = j < readN ? re[j] : 0.0;
            final ji = j < readN ? im[j] : 0.0;
            final angle = -2 * math.pi * k * j / n;
            final c = math.cos(angle), s = math.sin(angle);
            sr += jr * c - ji * s;
            si += jr * s + ji * c;
          }
        } else if (!onesided) {
          for (var j = 0; j < n; j++) {
            final jr = j < readN ? re[j] : 0.0;
            final ji = j < readN ? im[j] : 0.0;
            final angle = 2 * math.pi * k * j / n;
            final c = math.cos(angle), s = math.sin(angle);
            sr += jr * c - ji * s;
            si += jr * s + ji * c;
          }
          sr /= n;
          si /= n;
        } else {
          for (var j = 0; j < n; j++) {
            int idx;
            bool conjugate;
            if (j < readN) {
              idx = j;
              conjugate = false;
            } else {
              idx = n - j;
              conjugate = true;
            }
            if (idx < 0 || idx >= readN) continue;
            final jr = re[idx];
            final ji = conjugate ? -im[idx] : im[idx];
            final angle = 2 * math.pi * k * j / n;
            final c = math.cos(angle), s = math.sin(angle);
            sr += jr * c - ji * s;
            si += jr * s + ji * c;
          }
          sr /= n;
          si /= n;
        }
        final base = ((o * outN + k) * inner + i) * outComplex;
        out[base] = sr;
        if (outComplex == 2) out[base + 1] = si;
      }
    }
  }
  return Tensor.float(out, outShape);
}

double _sampleMultilinear(
  Tensor x,
  int batch,
  int channel,
  List<double> point,
  List<int> spatial,
) {
  final rank = spatial.length;
  var flatSpatial = 1;
  for (final d in spatial) {
    flatSpatial *= d;
  }
  final lo = List<int>.filled(rank, 0);
  final weightLo = List<double>.filled(rank, 0);
  final weightHi = List<double>.filled(rank, 0);
  for (var d = 0; d < rank; d++) {
    final v = point[d];
    lo[d] = v.floor();
    final frac = v - lo[d];
    weightLo[d] = 1 - frac;
    weightHi[d] = frac;
  }
  var acc = 0.0;
  final corners = 1 << rank;
  final coords = List<int>.filled(rank, 0);
  for (var corner = 0; corner < corners; corner++) {
    var weight = 1.0;
    var valid = true;
    for (var d = 0; d < rank; d++) {
      final upper = (corner >> d) & 1;
      final c = lo[d] + upper;
      if (c < 0 || c >= spatial[d]) {
        valid = false;
        break;
      }
      coords[d] = c;
      weight *= upper == 0 ? weightLo[d] : weightHi[d];
    }
    if (!valid || weight == 0) continue;
    var offset = 0;
    for (var d = 0; d < rank; d++) {
      offset = offset * spatial[d] + coords[d];
    }
    acc +=
        weight *
        x.getD(((batch * x.shape[1] + channel) * flatSpatial) + offset);
  }
  return acc;
}

Tensor opDeformConv(
  Tensor x,
  Tensor weight,
  Tensor offset, {
  Tensor? bias,
  Tensor? mask,
  int group = 1,
  int offsetGroup = 1,
  List<int>? dilations,
  List<int>? kernelShape,
  List<int>? pads,
  List<int>? strides,
}) {
  if (x.rank < 3 || weight.rank != x.rank) {
    throw ArgumentError('DeformConv: expected matching N-D input and weight');
  }
  final rank = x.rank - 2;
  final spatial = x.shape.sublist(2);
  final kernel = kernelShape ?? weight.shape.sublist(2);
  final stride = strides ?? List<int>.filled(rank, 1);
  final dilation = dilations ?? List<int>.filled(rank, 1);
  final pad = pads ?? List<int>.filled(rank * 2, 0);
  if (kernel.length != rank ||
      stride.length != rank ||
      dilation.length != rank ||
      pad.length != rank * 2) {
    throw ArgumentError('DeformConv: attribute rank mismatch');
  }
  final outSpatial = offset.shape.sublist(2);
  final n = x.shape[0];
  final channels = x.shape[1];
  final outChannels = weight.shape[0];
  final icsPerGroup = weight.shape[1];
  if (channels != icsPerGroup * group ||
      outChannels % group != 0 ||
      channels % offsetGroup != 0) {
    throw ArgumentError('DeformConv: inconsistent channel grouping');
  }
  final ocsPerGroup = outChannels ~/ group;
  final icsPerOffsetGroup = channels ~/ offsetGroup;
  final outProd = outSpatial.fold<int>(1, (a, b) => a * b);
  final kernelProd = kernel.fold<int>(1, (a, b) => a * b);
  if (offset.shape[1] != offsetGroup * kernelProd * rank) {
    throw ArgumentError('DeformConv: invalid offset channel count');
  }
  final out = List<double>.filled(n * outChannels * outProd, 0);
  final point = List<double>.filled(rank, 0);
  for (var batch = 0; batch < n; batch++) {
    for (var oc = 0; oc < outChannels; oc++) {
      final groupIndex = oc ~/ ocsPerGroup;
      for (var icg = 0; icg < icsPerGroup; icg++) {
        final ic = groupIndex * icsPerGroup + icg;
        final og = ic ~/ icsPerOffsetGroup;
        for (var outFlat = 0; outFlat < outProd; outFlat++) {
          if ((outFlat & 1023) == 0) activeOnnxControl?.checkpoint();
          final outCoords = _unflatten(outFlat, outSpatial);
          var acc = 0.0;
          for (var k = 0; k < kernelProd; k++) {
            final kCoords = _unflatten(k, kernel);
            for (var d = 0; d < rank; d++) {
              final offBase =
                  ((batch * offsetGroup + og) * kernelProd + k) * rank + d;
              point[d] =
                  outCoords[d] * stride[d] -
                  pad[d] +
                  kCoords[d] * dilation[d] +
                  offset.getD(offBase * outProd + outFlat);
            }
            var modulation = 1.0;
            if (mask != null) {
              final maskBase =
                  ((batch * offsetGroup + og) * kernelProd + k) * outProd +
                  outFlat;
              modulation = mask.getD(maskBase);
            }
            final sampled = _sampleMultilinear(x, batch, ic, point, spatial);
            final w = weight.getD((oc * icsPerGroup + icg) * kernelProd + k);
            acc += sampled * w * modulation;
          }
          out[(batch * outChannels + oc) * outProd + outFlat] += acc;
        }
      }
      if (bias != null) {
        final b = bias.getD(oc);
        for (var outFlat = 0; outFlat < outProd; outFlat++) {
          out[(batch * outChannels + oc) * outProd + outFlat] += b;
        }
      }
    }
  }
  return Tensor.float(Float32List.fromList(out), [
    n,
    outChannels,
    ...outSpatial,
  ]);
}

List<Tensor> opCausalConvWithState(
  Tensor input,
  Tensor weight, {
  Tensor? bias,
  Tensor? pastState,
  String activation = 'none',
}) {
  if (input.rank != 3 || weight.rank != 3) {
    throw ArgumentError(
      'CausalConvWithState: expected rank-3 input and weight',
    );
  }
  final b = input.shape[0], c = input.shape[1], length = input.shape[2];
  final kernelSize = weight.shape[2];
  final stateLength = kernelSize - 1;
  final paddedLen = stateLength + length;
  final padded = List<double>.filled(b * c * paddedLen, 0);
  if (pastState != null) {
    if (pastState.shape[0] != b ||
        pastState.shape[1] != c ||
        pastState.shape[2] != stateLength) {
      throw ArgumentError('CausalConvWithState: invalid past state shape');
    }
    for (var i = 0; i < b * c * stateLength; i++) {
      padded[i] = pastState.getD(i);
    }
  }
  for (var i = 0; i < b * c * length; i++) {
    final bc = i ~/ length;
    final t = i % length;
    padded[bc * paddedLen + stateLength + t] = input.getD(i);
  }
  final out = List<double>.filled(b * c * length, 0);
  for (var bc = 0; bc < b * c; bc++) {
    final channel = bc % c;
    final biasValue = bias == null ? 0.0 : bias.getD(channel);
    for (var t = 0; t < length; t++) {
      var acc = biasValue;
      for (var k = 0; k < kernelSize; k++) {
        acc +=
            weight.getD(channel * kernelSize + k) *
            padded[bc * paddedLen + t + k];
      }
      if (activation == 'silu' || activation == 'swish') {
        acc = acc / (1 + math.exp(-acc));
      }
      out[bc * length + t] = acc;
    }
  }
  final present = List<double>.filled(b * c * stateLength, 0);
  for (var bc = 0; bc < b * c; bc++) {
    for (var i = 0; i < stateLength; i++) {
      present[bc * stateLength + i] = padded[bc * paddedLen + length + i];
    }
  }
  return [
    Tensor.float(Float32List.fromList(out), [b, c, length]),
    Tensor.float(Float32List.fromList(present), [b, c, stateLength]),
  ];
}

List<Tensor> opLinearAttention(
  Tensor query,
  Tensor key,
  Tensor value, {
  Tensor? pastState,
  Tensor? decay,
  Tensor? beta,
  String updateRule = 'gated_delta',
  double scale = 0.0,
  required int kvNumHeads,
  required int qNumHeads,
}) {
  if (query.rank != 3 || key.rank != 3 || value.rank != 3) {
    throw ArgumentError('LinearAttention: query, key and value must be rank 3');
  }
  if (qNumHeads <= 0 || kvNumHeads <= 0 || qNumHeads % kvNumHeads != 0) {
    throw ArgumentError('LinearAttention: invalid head counts');
  }
  final gating = updateRule == 'gated' || updateRule == 'gated_delta';
  final delta = updateRule == 'delta' || updateRule == 'gated_delta';
  if (!const {'linear', 'gated', 'delta', 'gated_delta'}.contains(updateRule)) {
    throw ArgumentError('LinearAttention: unsupported update_rule $updateRule');
  }
  final b = query.shape[0], t = query.shape[1];
  final dk = query.shape[2] ~/ qNumHeads;
  final dv = value.shape[2] ~/ kvNumHeads;
  final groupSize = qNumHeads ~/ kvNumHeads;
  if (query.shape[2] != qNumHeads * dk ||
      key.shape[2] != kvNumHeads * dk ||
      value.shape[2] != kvNumHeads * dv) {
    throw ArgumentError('LinearAttention: head dimension mismatch');
  }
  final state = List<double>.filled(b * kvNumHeads * dk * dv, 0);
  if (pastState != null) {
    if (pastState.shape.length != 4 ||
        pastState.shape[0] != b ||
        pastState.shape[1] != kvNumHeads ||
        pastState.shape[2] != dk ||
        pastState.shape[3] != dv) {
      throw ArgumentError('LinearAttention: invalid past state shape');
    }
    for (var i = 0; i < state.length; i++) {
      state[i] = pastState.getD(i);
    }
  }
  final output = List<double>.filled(b * qNumHeads * t * dv, 0);
  final scaleValue = scale == 0.0 ? 1 / math.sqrt(dk) : scale;
  final vEff = List<double>.filled(dv, 0);
  final decayStride = decay == null
      ? 0
      : (decay.shape[2] == kvNumHeads ? 1 : dk);
  for (var bi = 0; bi < b; bi++) {
    for (var ti = 0; ti < t; ti++) {
      for (var kh = 0; kh < kvNumHeads; kh++) {
        final stateBase = ((bi * kvNumHeads + kh) * dk) * dv;
        final qStride = qNumHeads * dk;
        final kvStride = kvNumHeads * dk;
        final vStride = kvNumHeads * dv;
        if (gating) {
          for (var d = 0; d < dk; d++) {
            final di = decayStride == 1 ? 0 : d;
            final g = decay!.getD(
              (bi * t + ti) * decay.shape[2] + kh * decayStride + di,
            );
            final factor = math.exp(g);
            for (var m = 0; m < dv; m++) {
              state[stateBase + d * dv + m] *= factor;
            }
          }
        }
        if (delta) {
          final betaScale = beta!.shape[2] == 1
              ? beta.getD(bi * t + ti)
              : beta.getD((bi * t + ti) * beta.shape[2] + kh);
          for (var m = 0; m < dv; m++) {
            var retrieved = 0.0;
            for (var d = 0; d < dk; d++) {
              final k = key.getD((bi * t + ti) * kvStride + kh * dk + d);
              retrieved += state[stateBase + d * dv + m] * k;
            }
            final v = value.getD((bi * t + ti) * vStride + kh * dv + m);
            vEff[m] = betaScale * (v - retrieved);
          }
        } else {
          for (var m = 0; m < dv; m++) {
            vEff[m] = value.getD((bi * t + ti) * vStride + kh * dv + m);
          }
        }
        for (var d = 0; d < dk; d++) {
          final k = key.getD((bi * t + ti) * kvStride + kh * dk + d);
          for (var m = 0; m < dv; m++) {
            state[stateBase + d * dv + m] += k * vEff[m];
          }
        }
        for (var qi = 0; qi < groupSize; qi++) {
          final qh = kh * groupSize + qi;
          for (var m = 0; m < dv; m++) {
            var acc = 0.0;
            for (var d = 0; d < dk; d++) {
              final q = query.getD((bi * t + ti) * qStride + qh * dk + d);
              acc += q * state[stateBase + d * dv + m];
            }
            output[((bi * t + ti) * qNumHeads + qh) * dv + m] =
                acc * scaleValue;
          }
        }
      }
    }
  }
  return [
    Tensor.float(Float32List.fromList(output), [b, t, qNumHeads * dv]),
    Tensor.float(Float32List.fromList(state), [b, kvNumHeads, dk, dv]),
  ];
}

Tensor opTfIdfVectorizer(
  Tensor x, {
  required List<int> poolInt64s,
  required List<int> ngramCounts,
  required List<int> ngramIndexes,
  required int minGramLength,
  required int maxGramLength,
  required int maxSkipCount,
  required String mode,
  List<double>? weights,
}) {
  if (poolInt64s.isEmpty) {
    throw ArgumentError('TfIdfVectorizer: only pool_int64s is supported');
  }
  if (mode != 'TF' && mode != 'IDF' && mode != 'TFIDF') {
    throw ArgumentError('TfIdfVectorizer: unsupported mode $mode');
  }
  if (x.rank > 2) {
    throw ArgumentError('TfIdfVectorizer: input must be rank 1 or 2');
  }
  final outputSize = ngramIndexes.reduce(math.max) + 1;
  final grams = <String, int>{};
  final prefixes = <String>{};
  var ngramId = 0;
  for (var size = 1; size <= ngramCounts.length; size++) {
    final start = ngramCounts[size - 1];
    final end = size < ngramCounts.length
        ? ngramCounts[size]
        : poolInt64s.length;
    final count = (end - start) ~/ size;
    final included = size >= minGramLength && size <= maxGramLength;
    for (var g = 0; g < count; g++) {
      final base = start + g * size;
      final key = [
        for (var i = 0; i < size; i++) poolInt64s[base + i].toString(),
      ].join(',');
      for (var p = 1; p <= size; p++) {
        prefixes.add(
          [for (var i = 0; i < p; i++) poolInt64s[base + i].toString()]
              .join(','),
        );
      }
      if (included) {
        grams[key] = ngramIndexes[ngramId];
        ngramId++;
      } else {
        ngramId++;
      }
    }
  }
  final rows = x.rank == 2 ? x.shape[0] : 1;
  final rowSize = x.rank == 2 ? x.shape[1] : x.length;
  final frequencies = List<int>.filled(rows * outputSize, 0);
  for (var row = 0; row < rows; row++) {
    var startSize = minGramLength;
    for (var skip = 1; skip <= maxSkipCount + 1; skip++) {
      for (var start = 0; start < rowSize; start++) {
        final keyParts = <String>[];
        for (var pos = start; pos < rowSize; pos += skip) {
          keyParts.add(x.getI(row * rowSize + pos).toString());
          final size = keyParts.length;
          if (size < startSize) continue;
          if (size > maxGramLength) break;
          final key = keyParts.join(',');
          if (!prefixes.contains(key)) break;
          final idx = grams[key];
          if (idx != null) {
            frequencies[row * outputSize + idx]++;
          }
        }
      }
      if (startSize == 1) {
        startSize = 2;
        if (startSize > maxGramLength) break;
      }
    }
  }
  final out = Float32List(frequencies.length);
  for (var i = 0; i < frequencies.length; i++) {
    final count = frequencies[i];
    if (mode == 'TF') {
      out[i] = count.toDouble();
    } else if (mode == 'IDF') {
      if (weights != null && weights.isNotEmpty) {
        out[i] = count > 0 ? weights[i % outputSize] : 0;
      } else {
        out[i] = count > 0 ? 1 : 0;
      }
    } else {
      out[i] = weights != null && weights.isNotEmpty
          ? weights[i % outputSize] * count
          : count.toDouble();
    }
  }
  return Tensor.float(out, x.rank <= 1 ? [outputSize] : [rows, outputSize]);
}

StringTensor opStringConcat(Tensor x, Tensor y) {
  if (x is! StringTensor || y is! StringTensor) {
    throw ArgumentError('StringConcat: inputs must be string tensors');
  }
  final outShape = _broadcastShape(x.shape, y.shape);
  final n = outShape.fold<int>(1, (a, b) => a * b);
  final coords = List<int>.filled(outShape.length, 0);
  final out = List<String>.filled(n, '');
  for (int idx = 0; idx < n; idx++) {
    final xi = _flattenBroadcast(coords, x.shape);
    final yi = _flattenBroadcast(coords, y.shape);
    out[idx] = x.strings[xi] + y.strings[yi];
    for (int k = outShape.length - 1; k >= 0; k--) {
      if (++coords[k] < outShape[k]) break;
      coords[k] = 0;
    }
  }
  return StringTensor(out, outShape);
}

List<String> _splitString(String value, String? delimiter, int? maxsplit) {
  if (delimiter == null || delimiter.isEmpty) {
    final trimmed = value.trim();
    if (trimmed.isEmpty) return [];
    final limit = maxsplit ?? -1;
    final result = <String>[];
    int start = 0;
    while (start < trimmed.length && (limit < 0 || result.length < limit)) {
      int end = start;
      while (end < trimmed.length && !_isWhitespace(trimmed.codeUnitAt(end))) {
        end++;
      }
      result.add(trimmed.substring(start, end));
      while (end < trimmed.length && _isWhitespace(trimmed.codeUnitAt(end))) {
        end++;
      }
      start = end;
    }
    if (start < trimmed.length) result.add(trimmed.substring(start));
    return result;
  }
  final result = <String>[];
  int start = 0;
  int splits = 0;
  while (maxsplit == null || splits < maxsplit) {
    final at = value.indexOf(delimiter, start);
    if (at < 0) break;
    result.add(value.substring(start, at));
    start = at + delimiter.length;
    splits++;
  }
  result.add(value.substring(start));
  return result;
}

bool _isWhitespace(int c) =>
    c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0B || c == 0x0C || c == 0x0D;

(StringTensor, Tensor) opStringSplit(
  StringTensor x, {
  String? delimiter,
  int? maxsplit,
}) {
  final groups = [
    for (final value in x.strings) _splitString(value, delimiter, maxsplit),
  ];
  var width = 0;
  for (final group in groups) {
    if (group.length > width) width = group.length;
  }
  final out = <String>[];
  for (final group in groups) {
    out.addAll(group);
    for (int k = group.length; k < width; k++) {
      out.add('');
    }
  }
  final lengths = Int64List.fromList([
    for (final group in groups) group.length,
  ]);
  return (
    StringTensor(out, [...x.shape, width]),
    Tensor.int64(lengths, x.shape),
  );
}

const Map<int, String> _accentFolds = {
  0xC0: 'A',
  0xC1: 'A',
  0xC2: 'A',
  0xC3: 'A',
  0xC4: 'A',
  0xC5: 'A',
  0xC7: 'C',
  0xC8: 'E',
  0xC9: 'E',
  0xCA: 'E',
  0xCB: 'E',
  0xCC: 'I',
  0xCD: 'I',
  0xCE: 'I',
  0xCF: 'I',
  0xD1: 'N',
  0xD2: 'O',
  0xD3: 'O',
  0xD4: 'O',
  0xD5: 'O',
  0xD6: 'O',
  0xD9: 'U',
  0xDA: 'U',
  0xDB: 'U',
  0xDC: 'U',
  0xDD: 'Y',
  0xE0: 'a',
  0xE1: 'a',
  0xE2: 'a',
  0xE3: 'a',
  0xE4: 'a',
  0xE5: 'a',
  0xE7: 'c',
  0xE8: 'e',
  0xE9: 'e',
  0xEA: 'e',
  0xEB: 'e',
  0xEC: 'i',
  0xED: 'i',
  0xEE: 'i',
  0xEF: 'i',
  0xF1: 'n',
  0xF2: 'o',
  0xF3: 'o',
  0xF4: 'o',
  0xF5: 'o',
  0xF6: 'o',
  0xF9: 'u',
  0xFA: 'u',
  0xFB: 'u',
  0xFC: 'u',
  0xFD: 'y',
  0xFF: 'y',
};

String _stripAccents(String value) {
  final out = StringBuffer();
  for (final unit in value.codeUnits) {
    out.write(_accentFolds[unit] ?? String.fromCharCode(unit));
  }
  return out.toString();
}

StringTensor opStringNormalizer(
  StringTensor x, {
  String caseChangeAction = 'NONE',
  bool isCaseSensitive = false,
  List<String> stopwords = const [],
}) {
  if (x.rank != 1 && x.rank != 2) {
    throw ArgumentError('StringNormalizer: input must be rank 1 or 2');
  }
  if (caseChangeAction != 'NONE' &&
      caseChangeAction != 'LOWER' &&
      caseChangeAction != 'UPPER') {
    throw ArgumentError(
      'StringNormalizer: invalid case_change_action $caseChangeAction',
    );
  }
  final rawStops = stopwords.toSet();
  final stops = caseChangeAction == 'LOWER'
      ? {for (final word in stopwords) word.toLowerCase()}
      : caseChangeAction == 'UPPER'
      ? {for (final word in stopwords) word.toUpperCase()}
      : rawStops;
  final cleaned = <String>[];
  for (final original in x.strings) {
    var value = _stripAccents(original);
    if (isCaseSensitive && rawStops.isNotEmpty) {
      value = value
          .split(' ')
          .where((word) => !rawStops.contains(word))
          .join(' ');
    }
    if (caseChangeAction == 'LOWER') {
      value = value.toLowerCase();
    } else if (caseChangeAction == 'UPPER') {
      value = value.toUpperCase();
    }
    if (!isCaseSensitive && stops.isNotEmpty) {
      value = value.split(' ').where((word) => !stops.contains(word)).join(' ');
    }
    cleaned.add(value);
  }
  if (x.rank == 1) {
    final out = cleaned.where((value) => value.isNotEmpty).toList();
    return StringTensor(out.isEmpty ? [''] : out, [
      out.isEmpty ? 1 : out.length,
    ]);
  }
  if (x.shape[0] == 1) {
    final out = cleaned.where((value) => value.isNotEmpty).toList();
    return StringTensor(out.isEmpty ? [''] : out, [
      1,
      out.isEmpty ? 1 : out.length,
    ]);
  }
  return StringTensor(cleaned, x.shape);
}

Tensor opRegexFullMatch(StringTensor x, String pattern) {
  final RegExp regex;
  try {
    regex = RegExp('^(?:$pattern)\$');
  } on FormatException catch (e) {
    throw ArgumentError('Invalid regex pattern $pattern: $e');
  }
  final out = Int64List(x.length);
  for (var i = 0; i < x.length; i++) {
    final match = regex.firstMatch(x.strings[i]);
    out[i] =
        match != null && match.start == 0 && match.end == x.strings[i].length
        ? 1
        : 0;
  }
  return Tensor.int64(out, x.shape);
}

SequenceTensor opSequenceConstruct(List<Tensor> items) =>
    SequenceTensor(List<Tensor>.of(items));

Tensor opSequenceAt(SequenceTensor sequence, Tensor position) {
  if (position.length != 1) {
    throw ArgumentError('SequenceAt: position must be a scalar');
  }
  final n = sequence.items.length;
  var at = position.getI(0);
  if (at < 0) at += n;
  if (at < 0 || at >= n) throw RangeError('SequenceAt: position out of range');
  return sequence.items[at];
}

SequenceTensor opSequenceErase(SequenceTensor sequence, Tensor? position) {
  final items = List<Tensor>.of(sequence.items);
  if (items.isEmpty) throw RangeError('SequenceErase: empty sequence');
  if (position == null) {
    items.removeLast();
    return SequenceTensor(items);
  }
  if (position.length != 1) {
    throw ArgumentError('SequenceErase: position must be a scalar');
  }
  var at = position.getI(0);
  if (at < 0) at += items.length;
  if (at < 0 || at >= items.length) {
    throw RangeError('SequenceErase: position out of range');
  }
  items.removeAt(at);
  return SequenceTensor(items);
}

SequenceTensor opSequenceInsert(
  SequenceTensor sequence,
  Tensor value,
  Tensor? position,
) {
  final items = List<Tensor>.of(sequence.items);
  var at = items.length;
  if (position != null) {
    if (position.length != 1) {
      throw ArgumentError('SequenceInsert: position must be a scalar');
    }
    at = position.getI(0);
    if (at < 0) at += items.length;
    if (at < 0 || at > items.length) {
      throw RangeError('SequenceInsert: position out of range');
    }
  }
  items.insert(at, value);
  return SequenceTensor(items);
}

Tensor opSequenceLength(SequenceTensor sequence) =>
    Tensor.scalarInt(sequence.items.length);

Tensor opConcatFromSequence(
  SequenceTensor sequence, {
  int? axis,
  bool newAxis = false,
}) {
  if (sequence.items.isEmpty) {
    throw ArgumentError('ConcatFromSequence: empty sequence');
  }
  if (axis == null) {
    throw ArgumentError('ConcatFromSequence: axis is required');
  }
  final items = sequence.items;
  if (!newAxis) return opConcat(items, axis);
  final rank = items.first.rank;
  var ax = axis;
  if (ax < 0) ax += rank + 1;
  if (ax < 0 || ax > rank) {
    throw RangeError('ConcatFromSequence: axis out of range');
  }
  final expanded = <Tensor>[];
  for (final item in items) {
    if (item.shape.length != rank) {
      throw ArgumentError('ConcatFromSequence: input ranks differ');
    }
    for (var k = 0; k < rank; k++) {
      if (item.shape[k] != items.first.shape[k]) {
        throw ArgumentError('ConcatFromSequence: input shapes differ');
      }
    }
    final shape = List<int>.of(item.shape)..insert(ax, 1);
    expanded.add(item.reshape(shape));
  }
  return opConcat(expanded, ax);
}

SequenceTensor opSplitToSequence(
  Tensor x, {
  Tensor? split,
  int axis = 0,
  bool keepdims = true,
}) {
  final rank = x.rank;
  var ax = axis;
  if (ax < 0) ax += rank;
  if (ax < 0 || ax >= rank) {
    throw RangeError('SplitToSequence: axis out of range');
  }
  final dim = x.shape[ax];
  final sizes = <int>[];
  var explicitSizes = false;
  if (split == null) {
    sizes.addAll(List<int>.filled(dim, 1));
  } else if (split.rank == 0) {
    final chunk = split.getI(0);
    if (chunk <= 0) throw ArgumentError('SplitToSequence: split must be > 0');
    explicitSizes = true;
    for (var start = 0; start < dim; start += chunk) {
      sizes.add(math.min(chunk, dim - start));
    }
  } else {
    explicitSizes = true;
    sizes.addAll(split.asIntList());
    if (sizes.any((size) => size <= 0) ||
        sizes.fold<int>(0, (a, b) => a + b) != dim) {
      throw ArgumentError('SplitToSequence: split sizes do not match input');
    }
  }
  final result = <Tensor>[];
  var start = 0;
  for (final size in sizes) {
    var part = opSlice(x, [start], [start + size], [ax], null);
    if (!explicitSizes && !keepdims) {
      final shape = List<int>.of(part.shape)..removeAt(ax);
      part = part.reshape(shape);
    }
    result.add(part);
    start += size;
  }
  return SequenceTensor(result);
}

OptionalTensor opOptional(Tensor? value) => OptionalTensor(value);

Tensor opOptionalGetElement(Tensor? value) {
  if (value is OptionalTensor) {
    final element = value.value;
    if (element == null) {
      throw StateError('OptionalGetElement: optional has no element');
    }
    return element;
  }
  if (value == null) {
    throw StateError('OptionalGetElement: optional input is absent');
  }
  return value;
}

Tensor opOptionalHasElement(Tensor? value) => Tensor.scalarInt(
  value == null || (value is OptionalTensor && !value.hasElement) ? 0 : 1,
);
