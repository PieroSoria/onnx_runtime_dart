/// Isolate worker pool for parallel GEMM (native targets).
///
/// Each worker permanently owns a contiguous **column slice** of every
/// partitioned weight matrix, assigned once at spawn — so a matmul call
/// only ships the (small) activation matrix to each worker and receives its
/// output-column slice back; the (large) weights never cross an isolate
/// boundary again. Column-sliced GEMM computes the exact same dot products
/// as the local kernel, so results are bitwise identical.
library;

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'gemm_kernel_scalar.dart'
    if (dart.library.ffi) 'gemm_kernel_simd.dart'
    as gemm;
import 'onnx_nn_ops.dart' as nn;
import 'tensor.dart';

class _WeightSlice {
  final Float32List data; // [k × nCount], row-major
  final int k, nCount;
  _WeightSlice(this.data, this.k, this.nCount);
}

/// One partitioned weight: k rows, n total columns, and each worker's count
/// (worker w owns columns [colStart(w), colStart(w)+colCount(w))).
class PartitionedWeight {
  final int k, n;
  final List<int> colCounts;
  PartitionedWeight(this.k, this.n, this.colCounts);
  int colStart(int w) => colCounts.take(w).fold(0, (a, b) => a + b);
}

void _workerMain(SendPort toMain) {
  final fromMain = ReceivePort();
  toMain.send(fromMain.sendPort);
  final slices = <String, _WeightSlice>{};
  final convWeights = <String, (Tensor, Tensor?)>{}; // full replicas
  // One reusable Winograd plan per kernel shape. Rebuilding the transform per
  // conv is necessary (each node has distinct weights), but reallocating its
  // planes would churn several GB per inference.
  final winogradPlans = <String, nn.WinogradF2x2Plan>{};
  fromMain.listen((msg) {
    if (msg == null) {
      fromMain.close();
      return;
    }
    final map = msg as Map;
    switch (map['op']) {
      case 'load':
        slices[map['name'] as String] = _WeightSlice(
          map['data'] as Float32List,
          map['k'] as int,
          map['n'] as int,
        );
        (map['ack'] as SendPort).send(true);
      case 'loadconv':
        final bias = map['bias'] as Float32List?;
        convWeights[map['name'] as String] = (
          Tensor.float(
            map['w'] as Float32List,
            (map['wshape'] as List).cast<int>(),
          ),
          bias == null ? null : Tensor.float(bias, [bias.length]),
        );
        (map['ack'] as SendPort).send(true);
      case 'conv':
        final Tensor w;
        final Tensor? bias;
        if (map['w'] != null) {
          final wf = map['w'] as Float32List;
          w = Tensor.float(wf, (map['wshape'] as List).cast<int>());
          final bf = map['bias'] as Float32List?;
          bias = bf == null ? null : Tensor.float(bf, [bf.length]);
        } else {
          final (w0, b0) = convWeights[map['name'] as String]!;
          w = w0;
          bias = b0;
        }
        final raw = map['x'];
        final xf = raw is TransferableTypedData
            ? raw.materialize().asFloat32List()
            : raw as Float32List;
        nn.WinogradF2x2Plan? winograd;
        if (map['winograd'] == true) {
          final key = '${w.shape[0]},${w.shape[1]}';
          winograd = nn.pretransformWinogradF2x2(w, reuse: winogradPlans[key]);
          winogradPlans[key] = winograd;
        }
        final slab = nn.opConv(
          Tensor.float(xf, (map['xshape'] as List).cast<int>()),
          w,
          bias,
          strides: (map['strides'] as List?)?.cast<int>(),
          pads: (map['pads'] as List?)?.cast<int>(),
          dilations: (map['dilations'] as List?)?.cast<int>(),
          group: map['group'] as int,
          autoPad: map['autoPad'] as String,
          bandStart: map['b0'] as int,
          bandEnd: map['b1'] as int,
          srcRowStart: map['srcRowStart'] as int?,
          srcRows: map['srcRows'] as int?,
          winogradPlan: winograd,
        );
        (map['reply'] as SendPort).send({
          'id': map['id'],
          'out': TransferableTypedData.fromList([slab.f!]),
        });
      case 'convtranspose':
        final (w, bias) = convWeights[map['name'] as String]!;
        final raw = map['x'];
        final xf = raw is TransferableTypedData
            ? raw.materialize().asFloat32List()
            : raw as Float32List;
        final slab = nn.opConvTranspose(
          Tensor.float(xf, (map['xshape'] as List).cast<int>()),
          w,
          bias,
          strides: (map['strides'] as List?)?.cast<int>(),
          pads: (map['pads'] as List?)?.cast<int>(),
          dilations: (map['dilations'] as List?)?.cast<int>(),
          outputPadding: (map['outputPadding'] as List?)?.cast<int>(),
          group: map['group'] as int,
          bandStart: map['b0'] as int,
          bandEnd: map['b1'] as int,
          srcRowStart: map['srcRowStart'] as int?,
          srcRows: map['srcRows'] as int?,
        );
        (map['reply'] as SendPort).send({
          'id': map['id'],
          'out': TransferableTypedData.fromList([slab.f!]),
        });
      case 'matmulBatched':
        // A[128×128] times a batched twiddle tensor B[batch×128×65]: split
        // B along its leading batch dim; A is tiny and replicated.
        final rawA = map['a'];
        final a = rawA is TransferableTypedData
            ? rawA.materialize().asFloat32List()
            : rawA as Float32List;
        final rawB = map['b'];
        final b = rawB is TransferableTypedData
            ? rawB.materialize().asFloat32List()
            : rawB as Float32List;
        final batches = map['batches'] as int;
        final m = map['m'] as int;
        final k = map['k'] as int;
        final n = map['n'] as int;
        final out = Float32List(batches * m * n);
        for (int bi = 0; bi < batches; bi++) {
          gemm.matmulKernel(a, 0, b, bi * k * n, out, bi * m * n, m, k, n);
        }
        (map['reply'] as SendPort).send({
          'id': map['id'],
          'out': TransferableTypedData.fromList([out]),
        });
      case 'matmulNow':
        // Runtime B (computed inside the graph, e.g. the FFT twiddle
        // matrices): A is row-sliced by the caller, B is small and repeated.
        final rawA = map['a'];
        final a = rawA is TransferableTypedData
            ? rawA.materialize().asFloat32List()
            : rawA as Float32List;
        final b = map['b'] as Float32List;
        final rows = map['rows'] as int;
        final k = map['k'] as int;
        final n = map['n'] as int;
        final out = Float32List(rows * n);
        gemm.matmulKernel(a, 0, b, 0, out, 0, rows, k, n);
        (map['reply'] as SendPort).send({
          'id': map['id'],
          'out': TransferableTypedData.fromList([out]),
        });
      default: // 'matmul'
        final s = slices[map['name'] as String]!;
        final a = map['a'] as Float32List;
        final m = map['m'] as int;
        final out = Float32List(m * s.nCount);
        gemm.matmulKernel(a, 0, s.data, 0, out, 0, m, s.k, s.nCount);
        (map['reply'] as SendPort).send({'id': map['id'], 'out': out});
    }
  });
}

class GemmPool {
  final List<SendPort> _workers;
  final List<Isolate> _isolates;
  final ReceivePort _replies;
  final Map<String, PartitionedWeight> weights;

  /// Conv weights replicated to every worker (name -> output-row count is
  /// decided per call; membership is what matters here).
  final Set<String> convWeights;
  final Map<int, void Function(int worker, Float32List out)> _pending = {};
  int _nextId = 0;

  GemmPool._(
    this._workers,
    this._isolates,
    this._replies,
    this.weights,
    this.convWeights,
  );

  int get workerCount => _workers.length;

  /// Spawns [workers] isolates, distributes column slices of every entry in
  /// [toPartition] (name -> row-major [k × n] matmul weight), and replicates
  /// every conv weight in [convToReplicate]
  /// (name -> (weight data, weight shape, optional bias)) to all workers.
  ///
  /// [convNames] lists convs that are *eligible* for pooling without shipping
  /// their weights at spawn; their weight travels with each `conv` call. This
  /// keeps the resident set bounded for large CNNs (a 196 MB model replicated
  /// to 16 workers costs ~3 GB) at the price of copying a kernel per call.
  static Future<GemmPool> spawn(
    int workers,
    Map<String, (Float32List, int, int)> toPartition, [
    Map<String, (Float32List, List<int>, Float32List?)> convToReplicate =
        const {},
    Set<String>? convNames,
  ]) async {
    final isolates = <Isolate>[];
    final ports = <SendPort>[];
    for (int w = 0; w < workers; w++) {
      final handshake = ReceivePort();
      isolates.add(await Isolate.spawn(_workerMain, handshake.sendPort));
      ports.add(await handshake.first as SendPort);
      handshake.close();
    }

    final weights = <String, PartitionedWeight>{};
    for (final entry in toPartition.entries) {
      final (data, k, n) = entry.value;
      final base = n ~/ workers, rem = n % workers;
      final counts = [
        for (int w = 0; w < workers; w++) base + (w < rem ? 1 : 0),
      ];
      weights[entry.key] = PartitionedWeight(k, n, counts);
      int col = 0;
      for (int w = 0; w < workers; w++) {
        final nw = counts[w];
        final slice = Float32List(k * nw);
        for (int r = 0; r < k; r++) {
          slice.setRange(r * nw, (r + 1) * nw, data, r * n + col);
        }
        final ack = ReceivePort();
        ports[w].send({
          'op': 'load',
          'name': entry.key,
          'data': slice,
          'k': k,
          'n': nw,
          'ack': ack.sendPort,
        });
        await ack.first;
        ack.close();
        col += nw;
      }
    }

    for (final entry in convToReplicate.entries) {
      final (data, shape, bias) = entry.value;
      for (int w = 0; w < workers; w++) {
        final ack = ReceivePort();
        ports[w].send({
          'op': 'loadconv',
          'name': entry.key,
          'w': data,
          'wshape': shape,
          'bias': bias,
          'ack': ack.sendPort,
        });
        await ack.first;
        ack.close();
      }
    }

    final replies = ReceivePort();
    final pool = GemmPool._(ports, isolates, replies, weights, {
      ...convToReplicate.keys,
      ...?convNames,
    });
    replies.listen((msg) {
      final map = msg as Map;
      final entry = pool._pending.remove(map['id'] as int);
      final raw = map['out'];
      final slab = raw is TransferableTypedData
          ? raw.materialize().asFloat32List()
          : raw as Float32List;
      entry?.call(-1, slab); // worker id baked into id
    });
    return pool;
  }

  /// `out[m × n] = a[m × k] · W` for a partitioned weight, fanned out across
  /// the workers and reassembled by column range.
  Future<Float32List> matmul(String name, Float32List a, int m) {
    final w = weights[name]!;
    final out = Float32List(m * w.n);
    final completer = Completer<Float32List>();
    int remaining = _workers.length;
    for (int wk = 0; wk < _workers.length; wk++) {
      final id = _nextId++;
      final colStart = w.colStart(wk);
      final nCount = w.colCounts[wk];
      _pending[id] = (_, slice) {
        for (int i = 0; i < m; i++) {
          out.setRange(
            i * w.n + colStart,
            i * w.n + colStart + nCount,
            slice,
            i * nCount,
          );
        }
        if (--remaining == 0) completer.complete(out);
      };
      _workers[wk].send({
        'op': 'matmul',
        'id': id,
        'name': name,
        'a': a,
        'm': m,
        'reply': _replies.sendPort,
      });
    }
    return completer.future;
  }

  /// `out[m × n] = a[m × k] · b[k × n]` for a B computed at run time (FFT
  /// twiddles, attention weights, …). B is small and shipped to every worker;
  /// A is row-sliced so each worker receives only its band. Row splitting
  /// leaves the dot-product order untouched, so results stay bitwise equal.
  Future<Float32List> matmulRuntime(
    Float32List a,
    int m,
    Float32List b,
    int k,
    int n,
  ) {
    final out = Float32List(m * n);
    final completer = Completer<Float32List>();
    final w = _workers.length;
    final base = m ~/ w, rem = m % w;
    int remaining = 0, row = 0;
    for (int wk = 0; wk < w; wk++) {
      final rows = base + (wk < rem ? 1 : 0);
      if (rows == 0) continue;
      final id = _nextId++;
      final r0 = row, r1 = row + rows;
      row = r1;
      remaining++;
      _pending[id] = (_, slab) {
        out.setRange(r0 * n, r1 * n, slab, 0);
        if (--remaining == 0) completer.complete(out);
      };
      final slice = Float32List(rows * k);
      slice.setRange(0, rows * k, a, r0 * k);
      _workers[wk].send({
        'op': 'matmulNow',
        'id': id,
        'a': TransferableTypedData.fromList([slice]),
        'b': b,
        'rows': rows,
        'k': k,
        'n': n,
        'reply': _replies.sendPort,
      });
    }
    if (remaining == 0) completer.complete(out);
    return completer.future;
  }

  /// Batched runtime MatMul `A[m×k] · B[batch×k×n]`, splitting B's batch
  /// dimension across workers (FFC's per-batch twiddle matrices).
  Future<Float32List> matmulBatched(
    Float32List a,
    int m,
    int k,
    Float32List b,
    int batch,
    int n,
  ) {
    final out = Float32List(batch * m * n);
    final completer = Completer<Float32List>();
    final w = _workers.length;
    final base = batch ~/ w, rem = batch % w;
    int remaining = 0, b0 = 0;
    for (int wk = 0; wk < w; wk++) {
      final batches = base + (wk < rem ? 1 : 0);
      if (batches == 0) continue;
      final id = _nextId++;
      final start = b0, end = b0 + batches;
      b0 = end;
      remaining++;
      _pending[id] = (_, slab) {
        out.setRange(start * m * n, end * m * n, slab, 0);
        if (--remaining == 0) completer.complete(out);
      };
      final slice = Float32List(batches * k * n);
      slice.setRange(0, slice.length, b, start * k * n);
      _workers[wk].send({
        'op': 'matmulBatched',
        'id': id,
        'a': a,
        'b': TransferableTypedData.fromList([slice]),
        'batches': batches,
        'm': m,
        'k': k,
        'n': n,
        'reply': _replies.sendPort,
      });
    }
    if (remaining == 0) completer.complete(out);
    return completer.future;
  }

  /// 2-D conv fanned out by output-row bands: each worker computes rows
  /// `[cut_w, cut_{w+1})` against its full weight replica, and the slabs are
  /// stitched back into the `[n, m, oh, ow]` output. Each worker receives only
  /// the input rows its band reads (plus the kernel halo), shipped as
  /// `TransferableTypedData` so the slice is moved, not copied twice.
  /// When [weight] is given it travels with the request instead of using a
  /// replica (bounded resident set for large models).
  Future<Float32List> conv(
    String name,
    Float32List x,
    List<int> xShape, {
    required List<int>? strides,
    required List<int>? pads,
    required List<int>? dilations,
    required int group,
    required String autoPad,
    required int kernelH,
    required int n,
    required int m,
    required int oh,
    required int ow,
    bool depthwise = false,
    bool winograd = false,
    Tensor? weight,
    Tensor? bias,
  }) {
    final out = Float32List(n * m * oh * ow);
    final completer = Completer<Float32List>();
    final w = _workers.length;
    final base = oh ~/ w, rem = oh % w;
    int remaining = 0, cut = 0;
    final tasks = <(int, int, int)>[]; // (worker, b0, b1)
    for (int wk = 0; wk < w; wk++) {
      final rows = base + (wk < rem ? 1 : 0);
      if (rows == 0) continue;
      tasks.add((wk, cut, cut + rows));
      cut += rows;
      remaining++;
    }
    final c = xShape[1], inH = xShape[2], inW = xShape[3];
    for (final (wk, b0, b1) in tasks) {
      final id = _nextId++;
      _pending[id] = (_, slab) {
        final rows = b1 - b0;
        for (int img = 0; img < n * m; img++) {
          out.setRange(
            (img * oh + b0) * ow,
            (img * oh + b1) * ow,
            slab,
            img * rows * ow,
          );
        }
        if (--remaining == 0) completer.complete(out);
      };
      final Float32List xPart;
      final List<int> xPartShape;
      int? srcRowStart;
      int? srcRows;
      if (depthwise) {
        xPart = x;
        xPartShape = xShape;
      } else {
        final (gs, ge) = nn.convInputRowRange(
          inH: inH,
          kernelH: kernelH,
          bandStart: b0,
          bandEnd: b1,
          stride: strides?[0] ?? 1,
          dilation: dilations?[0] ?? 1,
          padTop: pads?[0],
          autoPad: autoPad,
        );
        final rows = ge - gs;
        final slice = Float32List(n * c * rows * inW);
        for (int img = 0; img < n; img++) {
          for (int ch = 0; ch < c; ch++) {
            final srcBase = (img * c + ch) * inH * inW + gs * inW;
            final dstBase = (img * c + ch) * rows * inW;
            slice.setRange(dstBase, dstBase + rows * inW, x, srcBase);
          }
        }
        xPart = slice;
        xPartShape = [n, c, rows, inW];
        srcRowStart = gs;
        srcRows = inH;
      }
      _workers[wk].send({
        'op': 'conv',
        'id': id,
        'name': name,
        'x': depthwise ? xPart : TransferableTypedData.fromList([xPart]),
        'xshape': xPartShape,
        'strides': strides,
        'pads': pads,
        'dilations': dilations,
        'group': group,
        'autoPad': autoPad,
        'b0': b0,
        'b1': b1,
        'srcRowStart': srcRowStart,
        'srcRows': srcRows,
        'winograd': winograd,
        'w': weight?.f,
        'wshape': weight?.shape,
        'bias': bias?.f,
        'reply': _replies.sendPort,
      });
    }
    return completer.future;
  }

  /// Transposed 2-D conv fanned out by output-row bands. Each worker gets the
  /// input rows that contribute to its band (kernel halo included) and writes
  /// a [m, bandRows, ow] slab, stitched back into [n, m, oh, ow].
  Future<Float32List> convTranspose(
    String name,
    Float32List x,
    List<int> xShape, {
    required List<int>? strides,
    required List<int>? pads,
    required List<int>? dilations,
    required List<int>? outputPadding,
    required int group,
    required int kernelH,
    required int n,
    required int m,
    required int oh,
    required int ow,
  }) {
    final out = Float32List(n * m * oh * ow);
    final completer = Completer<Float32List>();
    final w = _workers.length;
    final base = oh ~/ w, rem = oh % w;
    int remaining = 0, cut = 0;
    final tasks = <(int, int, int)>[];
    for (int wk = 0; wk < w; wk++) {
      final rows = base + (wk < rem ? 1 : 0);
      if (rows == 0) continue;
      tasks.add((wk, cut, cut + rows));
      cut += rows;
      remaining++;
    }
    final c = xShape[1], inH = xShape[2], inW = xShape[3];
    for (final (wk, b0, b1) in tasks) {
      final id = _nextId++;
      _pending[id] = (_, slab) {
        final rows = b1 - b0;
        for (int img = 0; img < n * m; img++) {
          out.setRange(
            (img * oh + b0) * ow,
            (img * oh + b1) * ow,
            slab,
            img * rows * ow,
          );
        }
        if (--remaining == 0) completer.complete(out);
      };
      final (gs, ge) = nn.convTransposeInputRowRange(
        inH: inH,
        kernelH: kernelH,
        bandStart: b0,
        bandEnd: b1,
        stride: strides?[0] ?? 1,
        dilation: dilations?[0] ?? 1,
        padTop: pads?[0] ?? 0,
      );
      final rows = ge - gs;
      final slice = Float32List(n * c * rows * inW);
      for (int img = 0; img < n; img++) {
        for (int ch = 0; ch < c; ch++) {
          final srcBase = (img * c + ch) * inH * inW + gs * inW;
          final dstBase = (img * c + ch) * rows * inW;
          slice.setRange(dstBase, dstBase + rows * inW, x, srcBase);
        }
      }
      _workers[wk].send({
        'op': 'convtranspose',
        'id': id,
        'name': name,
        'x': TransferableTypedData.fromList([slice]),
        'xshape': [n, c, rows, inW],
        'strides': strides,
        'pads': pads,
        'dilations': dilations,
        'outputPadding': outputPadding,
        'group': group,
        'b0': b0,
        'b1': b1,
        'srcRowStart': gs,
        'srcRows': inH,
        'reply': _replies.sendPort,
      });
    }
    return completer.future;
  }

  void dispose() {
    for (final p in _workers) {
      p.send(null);
    }
    for (final iso in _isolates) {
      iso.kill(priority: Isolate.immediate);
    }
    _replies.close();
  }
}
