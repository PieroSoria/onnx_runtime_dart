/// SIMD GEMM micro-kernel (`Float32x4`, compiles to SSE/NEON on the native
/// VM and AOT). Selected via conditional import on `dart.library.ffi`; web
/// targets get `gemm_kernel_scalar.dart` instead, where Float32x4 is
/// emulated and slower than plain loops.
///
/// Strategy: an outer loop over **column panels** of B keeps the packed
/// panel (k × panelWidth) inside L2, so every A-row tile reads B from cache
/// rather than re-streaming the whole matrix from memory (the difference
/// between memory- and compute-bound for large `k`). Within a panel, B is
/// packed to a 4-float-aligned row stride and the output is computed in
/// register-blocked 4-row × 8-column tiles — the 8 accumulators live in
/// locals across the whole k loop and are stored once per tile. Panel width,
/// row and column remainders all fall back to narrower work. The panel loop
/// does not change accumulation order, so results are bitwise identical to a
/// single-panel run (the isolate pool relies on this).
library;

import 'dart:typed_data';

/// Target packed-panel footprint in bytes (~half a typical 512 KB L2, leaving
/// room for the A-row tiles and output). Panel width is derived from this and
/// `k` so the panel stays resident across the full m loop.
const int _panelBytes = 192 * 1024;
const int _maxReusablePanelBytes = 1024 * 1024;

// Kernel calls are synchronous and isolates do not share statics, so this
// bounded scratch buffer can be reused without cross-run or cross-worker races.
Float32x4List _panelScratch = Float32x4List(0);

/// Computes `out[m×n] += a[m×k] · b[k×n]` (row-major, at the given flat
/// offsets). `out` is assumed zero-initialized (fresh allocation).
void matmulKernel(
  Float32List a,
  int aOff,
  Float32List b,
  int bOff,
  Float32List out,
  int outOff,
  int m,
  int k,
  int n, {
  bool narrowDirect = false,
  int? outStride,
  int panelBytes = _panelBytes,
}) {
  // `outStride` lets a caller write a column slab into a wider row layout
  // (banded convolution: output rows are `oh*ow` apart, the slab is `n` wide).
  final oStride = outStride ?? n;
  // Single-row GEMV (LLM decode: batch·seq == 1). The register-tiled path
  // below packs a column panel of B before computing — worthwhile when many
  // A-rows reuse it, but for one row that packing is pure overhead (B is read
  // to pack, then read again to compute). Stream B once in register-held
  // column tiles instead. Same k-order accumulation ⇒ bitwise identical.
  if (m == 1) {
    _gemv(a, aOff, b, bOff, out, outOff, k, n);
    return;
  }
  // Very few output rows (e.g. the 3-channel RGB output conv of a CNN): the
  // panel-packing path below pays O(k·n) packing for O(m·k·n) work, and its
  // remainder loop handles fewer than four rows one at a time. Stream B once
  // into per-column accumulators instead; the k-order is unchanged, so the
  // result is bitwise identical.
  if (m < 4) {
    _matmulSmallM(a, aOff, b, bOff, out, outOff, m, k, n, oStride);
    return;
  }
  final bBaseBytes = b.offsetInBytes + bOff * 4;
  if (narrowDirect &&
      oStride == n &&
      m >= 4 &&
      n <= 32 &&
      (n & 3) == 0 &&
      (bBaseBytes & 15) == 0) {
    _matmulNarrowAligned(a, aOff, b, bBaseBytes, out, outOff, m, k, n);
    return;
  }
  // Direct 4x16 path: when B is 16-byte aligned and n is a multiple of four,
  // read B through a Float32x4List view instead of packing a panel. The panel
  // loop still keeps the active k x panel slice in L2, but removes the
  // O(k*n) pack copy on every call. Measured ~15-30% faster on transformer
  // GEMMs; accumulation order is unchanged, so results stay bitwise identical.
  if (m >= 4 && (n & 3) == 0 && (bBaseBytes & 15) == 0) {
    _matmul4x16Direct(
      a,
      aOff,
      b,
      bBaseBytes,
      out,
      outOff,
      m,
      k,
      n,
      oStride,
      panelBytes,
    );
    return;
  }
  // Column-panel width: as many columns as keep k×panel×4 bytes under the
  // target, rounded up to a multiple of 4 (one Float32x4 lane); never wider
  // than n and at least 4.
  int panel = panelBytes ~/ (k * 4);
  panel = (panel >> 2) << 2;
  if (panel < 4) panel = 4;
  if (panel > n) panel = n;
  final singlePanel = panel >= n;

  final panelN4 = (panel + 3) >> 2;
  final panelLength = k * panelN4;
  final Float32x4List bp;
  if (panelLength * 16 <= _maxReusablePanelBytes) {
    if (_panelScratch.length < panelLength) {
      _panelScratch = Float32x4List(panelLength);
    }
    bp = _panelScratch;
  } else {
    bp = Float32x4List(panelLength);
  }
  final bpF = bp.buffer.asFloat32List();

  void storeLane(Float32x4 v, int outIdx, int cols) {
    out[outIdx] = v.x;
    if (cols > 1) out[outIdx + 1] = v.y;
    if (cols > 2) out[outIdx + 2] = v.z;
    if (cols > 3) out[outIdx + 3] = v.w;
  }

  for (int p0 = 0; p0 < n; p0 += panel) {
    final pw = singlePanel ? n : (p0 + panel <= n ? panel : n - p0);
    final pn4 = (pw + 3) >> 2;

    // Pack this column panel of B (zero-filled tail lanes for alignment).
    for (int kk = 0; kk < k; kk++) {
      final src = bOff + kk * n + p0, dst = kk * pn4 * 4;
      int j = 0;
      for (; j < pw; j++) {
        bpF[dst + j] = b[src + j];
      }
      for (; j < pn4 * 4; j++) {
        bpF[dst + j] = 0;
      }
    }

    int i = 0;
    for (; i + 4 <= m; i += 4) {
      final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
      final o0 = outOff + i * oStride + p0,
          o1 = o0 + oStride,
          o2 = o1 + oStride,
          o3 = o2 + oStride;

      int j = 0;
      // Full 4×8 tiles (two lanes entirely inside the panel width).
      for (; (j + 2) * 4 <= pw; j += 2) {
        var c00 = Float32x4.zero(), c01 = Float32x4.zero();
        var c10 = Float32x4.zero(), c11 = Float32x4.zero();
        var c20 = Float32x4.zero(), c21 = Float32x4.zero();
        var c30 = Float32x4.zero(), c31 = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          final bv0 = bp[bRow], bv1 = bp[bRow + 1];
          bRow += pn4;
          final v0 = Float32x4.splat(a[a0 + kk]);
          c00 += v0 * bv0;
          c01 += v0 * bv1;
          final v1 = Float32x4.splat(a[a1 + kk]);
          c10 += v1 * bv0;
          c11 += v1 * bv1;
          final v2 = Float32x4.splat(a[a2 + kk]);
          c20 += v2 * bv0;
          c21 += v2 * bv1;
          final v3 = Float32x4.splat(a[a3 + kk]);
          c30 += v3 * bv0;
          c31 += v3 * bv1;
        }
        final col = j * 4;
        storeLane(c00, o0 + col, 4);
        storeLane(c01, o0 + col + 4, 4);
        storeLane(c10, o1 + col, 4);
        storeLane(c11, o1 + col + 4, 4);
        storeLane(c20, o2 + col, 4);
        storeLane(c21, o2 + col + 4, 4);
        storeLane(c30, o3 + col, 4);
        storeLane(c31, o3 + col + 4, 4);
      }
      // Remaining lanes (≤ 8 tail columns of the panel), one lane at a time.
      for (; j < pn4; j++) {
        var c0 = Float32x4.zero(),
            c1 = Float32x4.zero(),
            c2 = Float32x4.zero(),
            c3 = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          final bv = bp[bRow];
          bRow += pn4;
          c0 += Float32x4.splat(a[a0 + kk]) * bv;
          c1 += Float32x4.splat(a[a1 + kk]) * bv;
          c2 += Float32x4.splat(a[a2 + kk]) * bv;
          c3 += Float32x4.splat(a[a3 + kk]) * bv;
        }
        final col = j * 4;
        final cols = pw - col < 4 ? pw - col : 4;
        storeLane(c0, o0 + col, cols);
        storeLane(c1, o1 + col, cols);
        storeLane(c2, o2 + col, cols);
        storeLane(c3, o3 + col, cols);
      }
    }

    // Remainder rows (m % 4): single-row, two lanes at a time.
    for (; i < m; i++) {
      final aRow = aOff + i * k;
      final oRow = outOff + i * oStride + p0;
      int j = 0;
      for (; (j + 2) * 4 <= pw; j += 2) {
        var c0 = Float32x4.zero(), c1 = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          final v = Float32x4.splat(a[aRow + kk]);
          c0 += v * bp[bRow];
          c1 += v * bp[bRow + 1];
          bRow += pn4;
        }
        final col = j * 4;
        storeLane(c0, oRow + col, 4);
        storeLane(c1, oRow + col + 4, 4);
      }
      for (; j < pn4; j++) {
        var c = Float32x4.zero();
        int bRow = j;
        for (int kk = 0; kk < k; kk++) {
          c += Float32x4.splat(a[aRow + kk]) * bp[bRow];
          bRow += pn4;
        }
        final col = j * 4;
        storeLane(c, oRow + col, pw - col < 4 ? pw - col : 4);
      }
    }
  }
}

/// Direct 4-row x 16-column kernel for aligned B with `n % 4 == 0`. No panel
/// packing: B is read through a vector view, while the panel loop keeps the
/// active slice in L2.
void _matmul4x16Direct(
  Float32List a,
  int aOff,
  Float32List b,
  int bBaseBytes,
  Float32List out,
  int outOff,
  int m,
  int k,
  int n,
  int oStride,
  int panelBytes,
) {
  final n4 = n >> 2;
  final b4 = Float32x4List.view(b.buffer, bBaseBytes, k * n4);
  int panel = panelBytes ~/ (k * 4);
  panel = (panel >> 2) << 2;
  if (panel < 4) panel = 4;
  if (panel > n) panel = n;

  for (int p0 = 0; p0 < n; p0 += panel) {
    final pw = p0 + panel <= n ? panel : n - p0;
    final pn4 = pw >> 2;

    int i = 0;
    for (; i + 4 <= m; i += 4) {
      final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
      final o0 = outOff + i * oStride + p0;
      final o1 = o0 + oStride, o2 = o1 + oStride, o3 = o2 + oStride;

      int j = 0;
      for (; j + 4 <= pn4; j += 4) {
        var c00 = Float32x4.zero(),
            c01 = Float32x4.zero(),
            c02 = Float32x4.zero(),
            c03 = Float32x4.zero();
        var c10 = Float32x4.zero(),
            c11 = Float32x4.zero(),
            c12 = Float32x4.zero(),
            c13 = Float32x4.zero();
        var c20 = Float32x4.zero(),
            c21 = Float32x4.zero(),
            c22 = Float32x4.zero(),
            c23 = Float32x4.zero();
        var c30 = Float32x4.zero(),
            c31 = Float32x4.zero(),
            c32 = Float32x4.zero(),
            c33 = Float32x4.zero();
        int bRow = (p0 >> 2) + j;
        for (int kk = 0; kk < k; kk++) {
          final bv0 = b4[bRow], bv1 = b4[bRow + 1];
          final bv2 = b4[bRow + 2], bv3 = b4[bRow + 3];
          bRow += n4;
          final v0 = Float32x4.splat(a[a0 + kk]);
          c00 += v0 * bv0;
          c01 += v0 * bv1;
          c02 += v0 * bv2;
          c03 += v0 * bv3;
          final v1 = Float32x4.splat(a[a1 + kk]);
          c10 += v1 * bv0;
          c11 += v1 * bv1;
          c12 += v1 * bv2;
          c13 += v1 * bv3;
          final v2 = Float32x4.splat(a[a2 + kk]);
          c20 += v2 * bv0;
          c21 += v2 * bv1;
          c22 += v2 * bv2;
          c23 += v2 * bv3;
          final v3 = Float32x4.splat(a[a3 + kk]);
          c30 += v3 * bv0;
          c31 += v3 * bv1;
          c32 += v3 * bv2;
          c33 += v3 * bv3;
        }
        final col = j << 2;
        _store4(out, o0 + col, c00);
        _store4(out, o0 + col + 4, c01);
        _store4(out, o0 + col + 8, c02);
        _store4(out, o0 + col + 12, c03);
        _store4(out, o1 + col, c10);
        _store4(out, o1 + col + 4, c11);
        _store4(out, o1 + col + 8, c12);
        _store4(out, o1 + col + 12, c13);
        _store4(out, o2 + col, c20);
        _store4(out, o2 + col + 4, c21);
        _store4(out, o2 + col + 8, c22);
        _store4(out, o2 + col + 12, c23);
        _store4(out, o3 + col, c30);
        _store4(out, o3 + col + 4, c31);
        _store4(out, o3 + col + 8, c32);
        _store4(out, o3 + col + 12, c33);
      }
      for (; j < pn4; j++) {
        var c0 = Float32x4.zero(), c1 = Float32x4.zero();
        var c2 = Float32x4.zero(), c3 = Float32x4.zero();
        int bRow = (p0 >> 2) + j;
        for (int kk = 0; kk < k; kk++) {
          final bv = b4[bRow];
          bRow += n4;
          c0 += Float32x4.splat(a[a0 + kk]) * bv;
          c1 += Float32x4.splat(a[a1 + kk]) * bv;
          c2 += Float32x4.splat(a[a2 + kk]) * bv;
          c3 += Float32x4.splat(a[a3 + kk]) * bv;
        }
        final col = j << 2;
        _store4(out, o0 + col, c0);
        _store4(out, o1 + col, c1);
        _store4(out, o2 + col, c2);
        _store4(out, o3 + col, c3);
      }
    }

    for (; i < m; i++) {
      final aRow = aOff + i * k;
      final oRow = outOff + i * oStride + p0;
      int j = 0;
      for (; j + 4 <= pn4; j += 4) {
        var c0 = Float32x4.zero(), c1 = Float32x4.zero();
        var c2 = Float32x4.zero(), c3 = Float32x4.zero();
        int bRow = (p0 >> 2) + j;
        for (int kk = 0; kk < k; kk++) {
          final v = Float32x4.splat(a[aRow + kk]);
          c0 += v * b4[bRow];
          c1 += v * b4[bRow + 1];
          c2 += v * b4[bRow + 2];
          c3 += v * b4[bRow + 3];
          bRow += n4;
        }
        final col = j << 2;
        _store4(out, oRow + col, c0);
        _store4(out, oRow + col + 4, c1);
        _store4(out, oRow + col + 8, c2);
        _store4(out, oRow + col + 12, c3);
      }
      for (; j < pn4; j++) {
        var c = Float32x4.zero();
        int bRow = (p0 >> 2) + j;
        for (int kk = 0; kk < k; kk++) {
          c += Float32x4.splat(a[aRow + kk]) * b4[bRow];
          bRow += n4;
        }
        _store4(out, oRow + (j << 2), c);
      }
    }
  }
}

void _matmulNarrowAligned(
  Float32List a,
  int aOff,
  Float32List b,
  int bBaseBytes,
  Float32List out,
  int outOff,
  int m,
  int k,
  int n,
) {
  final n4 = n >> 2;
  final b4 = Float32x4List.view(b.buffer, bBaseBytes, k * n4);
  int i = 0;
  for (; i + 4 <= m; i += 4) {
    final a0 = aOff + i * k, a1 = a0 + k, a2 = a1 + k, a3 = a2 + k;
    final o0 = outOff + i * n, o1 = o0 + n, o2 = o1 + n, o3 = o2 + n;
    for (int j = 0; j < n4; j++) {
      var c0 = Float32x4.zero(), c1 = Float32x4.zero();
      var c2 = Float32x4.zero(), c3 = Float32x4.zero();
      int br = j;
      for (int kk = 0; kk < k; kk++) {
        final bv = b4[br];
        br += n4;
        c0 += Float32x4.splat(a[a0 + kk]) * bv;
        c1 += Float32x4.splat(a[a1 + kk]) * bv;
        c2 += Float32x4.splat(a[a2 + kk]) * bv;
        c3 += Float32x4.splat(a[a3 + kk]) * bv;
      }
      _store4(out, o0 + (j << 2), c0);
      _store4(out, o1 + (j << 2), c1);
      _store4(out, o2 + (j << 2), c2);
      _store4(out, o3 + (j << 2), c3);
    }
  }
  for (; i < m; i++) {
    final ar = aOff + i * k, or = outOff + i * n;
    for (int j = 0; j < n4; j++) {
      var sum = Float32x4.zero();
      int br = j;
      for (int kk = 0; kk < k; kk++) {
        sum += Float32x4.splat(a[ar + kk]) * b4[br];
        br += n4;
      }
      _store4(out, or + (j << 2), sum);
    }
  }
}

void _store4(Float32List out, int offset, Float32x4 value) {
  out[offset] = value.x;
  out[offset + 1] = value.y;
  out[offset + 2] = value.z;
  out[offset + 3] = value.w;
}

/// `out[1×n] = a[1×k] · b[k×n]`, streaming B exactly once. Columns are
/// processed in 16-wide tiles whose four `Float32x4` accumulators live in
/// registers across the k loop, so each B element is read once (sequentially
/// within a tile) and each output written once. Accumulation is in k order —
/// bitwise identical to the register-tiled path.
///
/// When `n` is a multiple of 4 and B's base is 16-byte aligned (the common
/// case: transformer weight matrices), B is read through a `Float32x4List`
/// view — one vector load per lane instead of the four scalar loads the
/// `Float32x4(...)` constructor compiles to. Measured ~1.9× on large GEMV
/// (bitwise identical: the view yields the same 4 floats). Otherwise the
/// scalar-load fallback runs (still correct for any offset/width).
void _gemv(
  Float32List a,
  int aOff,
  Float32List b,
  int bOff,
  Float32List out,
  int outOff,
  int k,
  int n,
) {
  final baseBytes = b.offsetInBytes + bOff * 4;
  if ((n & 3) == 0 && (baseBytes & 15) == 0) {
    _gemvAligned(a, aOff, b, baseBytes, out, outOff, k, n);
  } else {
    _gemvScalar(a, aOff, b, bOff, out, outOff, k, n);
  }
}

/// Vector-load GEMV: B viewed as `Float32x4List` from the (16-byte aligned)
/// base. `n % 4 == 0`, so the column loop lands exactly on `n` with no scalar
/// tail. Each B row is `n>>2` vectors wide.
void _gemvAligned(
  Float32List a,
  int aOff,
  Float32List b,
  int baseBytes,
  Float32List out,
  int outOff,
  int k,
  int n,
) {
  final n4 = n >> 2;
  final b4 = Float32x4List.view(b.buffer, baseBytes, k * n4);
  int j = 0;
  for (; j + 16 <= n; j += 16) {
    var c0 = Float32x4.zero(),
        c1 = Float32x4.zero(),
        c2 = Float32x4.zero(),
        c3 = Float32x4.zero();
    int br = j >> 2;
    for (int kk = 0; kk < k; kk++) {
      final va = Float32x4.splat(a[aOff + kk]);
      c0 += va * b4[br];
      c1 += va * b4[br + 1];
      c2 += va * b4[br + 2];
      c3 += va * b4[br + 3];
      br += n4;
    }
    final o = outOff + j;
    out[o] = c0.x;
    out[o + 1] = c0.y;
    out[o + 2] = c0.z;
    out[o + 3] = c0.w;
    out[o + 4] = c1.x;
    out[o + 5] = c1.y;
    out[o + 6] = c1.z;
    out[o + 7] = c1.w;
    out[o + 8] = c2.x;
    out[o + 9] = c2.y;
    out[o + 10] = c2.z;
    out[o + 11] = c2.w;
    out[o + 12] = c3.x;
    out[o + 13] = c3.y;
    out[o + 14] = c3.z;
    out[o + 15] = c3.w;
  }
  for (; j + 4 <= n; j += 4) {
    var c0 = Float32x4.zero();
    int br = j >> 2;
    for (int kk = 0; kk < k; kk++) {
      c0 += Float32x4.splat(a[aOff + kk]) * b4[br];
      br += n4;
    }
    final o = outOff + j;
    out[o] = c0.x;
    out[o + 1] = c0.y;
    out[o + 2] = c0.z;
    out[o + 3] = c0.w;
  }
}

/// Scalar-load GEMV fallback for any B offset/width (used when the aligned
/// vector-load preconditions don't hold — e.g. an odd `n` or an isolate
/// worker's unaligned column slice).
void _gemvScalar(
  Float32List a,
  int aOff,
  Float32List b,
  int bOff,
  Float32List out,
  int outOff,
  int k,
  int n,
) {
  int j = 0;
  for (; j + 16 <= n; j += 16) {
    var c0 = Float32x4.zero(),
        c1 = Float32x4.zero(),
        c2 = Float32x4.zero(),
        c3 = Float32x4.zero();
    int br = bOff + j;
    for (int kk = 0; kk < k; kk++) {
      final va = Float32x4.splat(a[aOff + kk]);
      c0 += va * Float32x4(b[br], b[br + 1], b[br + 2], b[br + 3]);
      c1 += va * Float32x4(b[br + 4], b[br + 5], b[br + 6], b[br + 7]);
      c2 += va * Float32x4(b[br + 8], b[br + 9], b[br + 10], b[br + 11]);
      c3 += va * Float32x4(b[br + 12], b[br + 13], b[br + 14], b[br + 15]);
      br += n;
    }
    final o = outOff + j;
    out[o] = c0.x;
    out[o + 1] = c0.y;
    out[o + 2] = c0.z;
    out[o + 3] = c0.w;
    out[o + 4] = c1.x;
    out[o + 5] = c1.y;
    out[o + 6] = c1.z;
    out[o + 7] = c1.w;
    out[o + 8] = c2.x;
    out[o + 9] = c2.y;
    out[o + 10] = c2.z;
    out[o + 11] = c2.w;
    out[o + 12] = c3.x;
    out[o + 13] = c3.y;
    out[o + 14] = c3.z;
    out[o + 15] = c3.w;
  }
  for (; j + 4 <= n; j += 4) {
    var c0 = Float32x4.zero();
    int br = bOff + j;
    for (int kk = 0; kk < k; kk++) {
      c0 +=
          Float32x4.splat(a[aOff + kk]) *
          Float32x4(b[br], b[br + 1], b[br + 2], b[br + 3]);
      br += n;
    }
    final o = outOff + j;
    out[o] = c0.x;
    out[o + 1] = c0.y;
    out[o + 2] = c0.z;
    out[o + 3] = c0.w;
  }
  for (; j < n; j++) {
    double acc = 0;
    int br = bOff + j;
    for (int kk = 0; kk < k; kk++) {
      acc += a[aOff + kk] * b[br];
      br += n;
    }
    out[outOff + j] = acc;
  }
}

/// Streaming kernel for `m < 4`: reads B once and keeps one SIMD accumulator
/// per four output columns, instead of packing a column panel and then walking
/// it one row at a time. The k accumulation order is the same as the tiled
/// path, so results are bitwise identical.
void _matmulSmallM(
  Float32List a,
  int aOff,
  Float32List b,
  int bOff,
  Float32List out,
  int outOff,
  int m,
  int k,
  int n,
  int oStride,
) {
  final n4 = n >> 2;
  final acc = Float32x4List(m * n4);
  final va = List<Float32x4>.filled(m, Float32x4.zero());
  for (int kk = 0; kk < k; kk++) {
    final aRow = aOff + kk;
    for (int r = 0; r < m; r++) {
      va[r] = Float32x4.splat(a[aRow + r * k]);
    }
    final bRow = bOff + kk * n;
    for (int jb = 0; jb < n4; jb++) {
      final col = jb << 2;
      final bv = Float32x4(
        b[bRow + col],
        b[bRow + col + 1],
        b[bRow + col + 2],
        b[bRow + col + 3],
      );
      for (int r = 0; r < m; r++) {
        final i = r * n4 + jb;
        acc[i] += va[r] * bv;
      }
    }
  }
  final lane = Float32x4List(1);
  final laneF = lane.buffer.asFloat32List();
  for (int r = 0; r < m; r++) {
    final oRow = outOff + r * oStride;
    final aRow = aOff + r * k;
    final base = r * n4;
    for (int jb = 0; jb < n4; jb++) {
      lane[0] = acc[base + jb];
      final col = jb << 2;
      out[oRow + col] = laneF[0];
      out[oRow + col + 1] = laneF[1];
      out[oRow + col + 2] = laneF[2];
      out[oRow + col + 3] = laneF[3];
    }
    for (int col = n4 << 2; col < n; col++) {
      double s = 0;
      int br = bOff + col;
      for (int kk = 0; kk < k; kk++) {
        s += a[aRow + kk] * b[br];
        br += n;
      }
      out[oRow + col] = s;
    }
  }
}

/// SIMD dot product of two contiguous rows (used by einsum kernels whose
/// operand layout is already dot-friendly — no transpose or packing needed).
double dotProduct(Float32List a, int aOff, Float32List b, int bOff, int n) {
  var acc = Float32x4.zero();
  int k = 0;
  for (; k + 4 <= n; k += 4) {
    acc +=
        Float32x4(
          a[aOff + k],
          a[aOff + k + 1],
          a[aOff + k + 2],
          a[aOff + k + 3],
        ) *
        Float32x4(
          b[bOff + k],
          b[bOff + k + 1],
          b[bOff + k + 2],
          b[bOff + k + 3],
        );
  }
  double sum = acc.x + acc.y + acc.z + acc.w;
  for (; k < n; k++) {
    sum += a[aOff + k] * b[bOff + k];
  }
  return sum;
}
