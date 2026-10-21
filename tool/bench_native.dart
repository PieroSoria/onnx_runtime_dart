/// Wall-clock benchmark and optional parity check for the native ORT backend.
///
///   dart run tool/bench_native.dart model.onnx --library path/to/onnxruntime.dll
///       [--iters 5] [--seq 16] [--output name] [--compare-dart]
///
/// Inputs are synthesized from the session metadata. Unknown dims use
/// `--seq` (or 1 for a leading batch dim). With `--compare-dart` the same feed
/// is also run through the pure-Dart interpreter and the outputs are compared.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_ort.dart';

Future<void> main(List<String> args) async {
  final positional = args.where((a) => !a.startsWith('--')).toList();
  if (positional.isEmpty) {
    stderr.writeln(
      'Usage: dart run tool/bench_native.dart model.onnx --library <path> '
      '[--iters N] [--seq N] [--output name] [--compare-dart]',
    );
    exitCode = 64;
    return;
  }
  final modelPath = positional.first;
  final iters = _flag(args, 'iters', 5);
  final seq = _flag(args, 'seq', 16);
  final threads = _flag(args, 'threads', 0);
  final interThreads = _flag(args, 'inter-threads', 1);
  final library =
      _flagString(args, 'library') ??
      Platform.environment['ONNX_RUNTIME_DART_ORT_LIBRARY'] ??
      Platform.environment['ORT_LIBRARY'];
  final outputArg = _flagString(args, 'output');
  final cuda = args.contains('--cuda');
  final cudaDir = _flagString(args, 'cuda-dir');
  final dartWorkers = _flag(args, 'dart-workers', 0);

  final options = OrtOptions(
    libraryPath: library,
    intraOpThreads: threads,
    interOpThreads: interThreads,
    providers: cuda ? const [OrtCudaProvider()] : const [OrtCpuProvider()],
    cudaLibraryDirectory: cudaDir,
  );
  final stopwatch = Stopwatch()..start();
  final model = await OnnxSession.fromFile(
    modelPath,
    backend: OnnxBackend.ort,
    ort: options,
  );
  stopwatch.stop();
  if (model is! OrtRunner) {
    throw StateError('Expected OrtRunner, got ${model.runtimeType}');
  }
  stderr.writeln(
    'load: ${stopwatch.elapsedMilliseconds} ms; '
    'native=${model.runtimeVersion}; providers=${model.providers}',
  );
  try {
    for (final spec in model.inputSpecs) {
      stderr.writeln(
        'input ${spec.name}: type=${spec.elemType} '
        'shape=${spec.shape}',
      );
    }
    for (final name in model.outputNames) {
      stderr.writeln('output $name');
    }
    final feed = _makeFeed(model.inputSpecs, seq);
    final outputs = outputArg == null ? model.outputNames : [outputArg];
    if (outputs.isEmpty) {
      throw StateError('Model declares no outputs');
    }

    model.run(feed, outputs);
    final times = <int>[];
    for (var i = 0; i < iters; i++) {
      final watch = Stopwatch()..start();
      model.run(feed, outputs);
      watch.stop();
      times.add(watch.elapsedMicroseconds);
    }
    times.sort();
    final mean = times.reduce((a, b) => a + b) / times.length;
    final result = model.run(feed, outputs);
    for (final name in outputs) {
      final tensor = result[name]!;
      stderr.writeln(
        'output $name: shape=${tensor.shape} '
        'first4=${tensor.asFloatList().take(4).map((v) => v.toStringAsFixed(5)).toList()}',
      );
    }
    print(
      'native wall: min=${(times.first / 1000).toStringAsFixed(1)}ms '
      'mean=${(mean / 1000).toStringAsFixed(1)}ms over ${times.length} iters',
    );
    print(
      'rss: peak=${(ProcessInfo.maxRss / (1 << 20)).toStringAsFixed(0)}MB '
      'end=${(ProcessInfo.currentRss / (1 << 20)).toStringAsFixed(0)}MB',
    );

    if (args.contains('--compare-dart')) {
      stderr.writeln('running the same feed through the Dart interpreter...');
      final dartWatch = Stopwatch()..start();
      final dart = await OnnxSession.fromFile(
        modelPath,
        backend: OnnxBackend.dart,
      );
      if (dartWorkers > 0 && dart is OnnxModel) {
        await dart.parallelize(workers: dartWorkers, poolConv: true);
      }
      final actual = dart.run(feed, outputs);
      dartWatch.stop();
      dart.dispose();
      var maxError = 0.0;
      var sum = 0.0;
      var count = 0;
      var outside = 0;
      for (final name in outputs) {
        final a = actual[name]!;
        final b = result[name]!;
        if (a.shape.join(',') != b.shape.join(',')) {
          throw StateError('$name shape ${a.shape} != ${b.shape}');
        }
        final av = a.asFloatList();
        final bv = b.asFloatList();
        for (var i = 0; i < av.length; i++) {
          final delta = (av[i] - bv[i]).abs();
          maxError = math.max(maxError, delta);
          sum += delta;
          count++;
          if (delta > 1e-4 + bv[i].abs() * 1e-3) outside++;
        }
      }
      print(
        'dart wall: ${(dartWatch.elapsedMicroseconds / 1000).toStringAsFixed(1)}ms; '
        'max_abs_error=${maxError.toStringAsExponential(3)} '
        'mean_abs_error=${(sum / count).toStringAsExponential(3)} '
        'outside_tolerance=$outside/${count ~/ outputs.length}',
      );
    }
  } finally {
    model.dispose();
  }
}

Map<String, Tensor> _makeFeed(List<TensorSpec> specs, int seq) {
  final feed = <String, Tensor>{};
  for (final spec in specs) {
    final shape = [
      for (var i = 0; i < spec.shape.length; i++)
        spec.shape[i] > 0 ? spec.shape[i] : (i == 0 ? 1 : seq),
    ];
    final count = shape.fold<int>(1, (a, b) => a * b);
    final name = spec.name.toLowerCase();
    switch (spec.elemType) {
      case 7:
      case 6:
        final data = Int64List(count);
        if (name.contains('mask')) {
          data.fillRange(0, count, 1);
        } else if (name.contains('position')) {
          for (var i = 0; i < count; i++) {
            data[i] = i % seq;
          }
        } else {
          for (var i = 0; i < count; i++) {
            data[i] = 1000 + (i * 37) % 999;
          }
        }
        feed[spec.name] = Tensor.int64(data, shape);
      case 2:
        feed[spec.name] = Tensor.uint8(Uint8List(count), shape);
      case 3:
        feed[spec.name] = Tensor.int8(Int8List(count), shape);
      default:
        final data = Float32List(count);
        if (name.contains('mask')) {
          final h = shape.length >= 4 ? shape[shape.length - 2] : seq;
          final w = shape.length >= 4 ? shape[shape.length - 1] : seq;
          for (var y = 0; y < h; y++) {
            for (var x = 0; x < w; x++) {
              final inside =
                  x > w * .4 && x < w * .6 && y > h * .4 && y < h * .6;
              data[y * w + x] = inside ? 1 : 0;
            }
          }
        } else {
          for (var i = 0; i < count; i++) {
            data[i] = ((i * 2654435761) & 0xffff) / 32768.0 - 1.0;
          }
        }
        feed[spec.name] = Tensor.float(data, shape);
    }
  }
  return feed;
}

int _flag(List<String> args, String name, int fallback) {
  final index = args.indexOf('--$name');
  if (index < 0 || index + 1 >= args.length) return fallback;
  return int.tryParse(args[index + 1]) ?? fallback;
}

String? _flagString(List<String> args, String name) {
  final index = args.indexOf('--$name');
  return index >= 0 && index + 1 < args.length ? args[index + 1] : null;
}
