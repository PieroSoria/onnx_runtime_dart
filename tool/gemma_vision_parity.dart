/// Runs the Gemma vision encoder against tool/gemma_vision_parity.py's oracle.
library;
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

void main(List<String> args) {
  if (args.length != 2) {
    throw ArgumentError('Usage: gemma_vision_parity.dart MODEL_DIR ORACLE_DIR');
  }
  final root = args[0], oracle = args[1];
  final reference = jsonDecode(File('$oracle/reference.json').readAsStringSync());
  final control = OnnxExecutionControl(
      maxTensorBytes: 3 * 1024 * 1024 * 1024,
      deadline: DateTime.now().add(const Duration(minutes: 20)),
      onProgress: (phase, done, total) {
        stdout.writeln('$phase $done/$total rss=${ProcessInfo.currentRss}');
      });
  final model =
      loadOnnxModel('$root/vision_encoder/model.onnx', control: control);
  final folder = '$oracle/vision_encoder';
  final grid = (reference['grid'] as num).toInt();
  final pixels = File('$folder/pixel-values.bin').readAsBytesSync();
  final pos = File('$folder/pixel-position-ids.bin').readAsBytesSync();
  final x2 = _readFp16(pixels, [1, grid * grid, 768]);
  final p = Tensor.int64(Int64List.view(pos.buffer, pos.offsetInBytes,
      pos.length ~/ 8), [1, grid * grid, 2]);
  final sw = Stopwatch()..start();
  final all = model.run({'pixel_values': x2, 'pixel_position_ids': p}, ['*']);
  final y = all['image_features']!;
  final spec = (reference['outputs'] as Map)['image_features'] as Map;
  final eb = File('$folder/${spec['file']}').readAsBytesSync();
  final expected = Tensor.float(
      Float32List.view(eb.buffer, eb.offsetInBytes, eb.length ~/ 4), y.shape);
  var maxAbs = 0.0, sum = 0.0, norm = 0.0, gn = 0.0, dot = 0.0;
  for (var i = 0; i < y.length; i++) {
    final a = y.getD(i), b = expected.getD(i);
    final delta = (a - b).abs();
    maxAbs = math.max(maxAbs, delta);
    sum += delta * delta;
    norm += b * b;
    gn += a * a;
    dot += a * b;
  }
  final report = {
    'grid': grid,
    'shape': y.shape,
    'maxAbs': maxAbs,
    'rmse': math.sqrt(sum / math.max(1, y.length)),
    'cosine': dot / math.sqrt(norm * gn),
    'seconds': sw.elapsedMilliseconds / 1000,
    'rss': ProcessInfo.currentRss,
  };
  final probeReport = <String, Map<String, double>>{};
  for (final entry in (reference['outputs'] as Map).entries) {
    final name = entry.key as String;
    if (name == 'image_features' || !all.containsKey(name)) continue;
    final cSpec = entry.value as Map;
    final cb = File('$folder/${cSpec['file']}').readAsBytesSync();
    final got = all[name];
    final dims = (cSpec['shape'] as List).cast<int>();
    final dataLen = dims.fold<int>(1, (a, b) => a * b);
    final expected = (cSpec['dtype'] == 'int64')
        ? Tensor.int64(Int64List.view(cb.buffer, cb.offsetInBytes, dataLen), dims)
        : Tensor.float(
            Float32List.view(cb.buffer, cb.offsetInBytes, dataLen), dims);
    if (got!.shape.toString() != dims.toString()) {
      probeReport[name] = {'shape-mismatch': 1};
      continue;
    }
    var maxAbs = 0.0, norm = 0.0, gn = 0.0, dot = 0.0;
    for (var i = 0; i < got.length; i++) {
      final a = got.getD(i), b = expected.getD(i);
      final delta = (a - b).abs();
      maxAbs = math.max(maxAbs, delta);
      norm += b * b;
      gn += a * a;
      dot += a * b;
    }
    probeReport[name] = {
      'maxAbs': maxAbs,
      'cosine': (norm * gn) == 0 ? 0.0 : dot / math.sqrt(norm * gn),
    };
  }
  if (probeReport.isNotEmpty) report['probes'] = probeReport;
  stdout.writeln(jsonEncode(report));
  File('$oracle/dart-report.json')
      .writeAsStringSync(const JsonEncoder.withIndent('  ').convert(report));
}

Tensor _readFp16(Uint8List bytes, List<int> shape) {
  final u16 = Uint16List.view(bytes.buffer, bytes.offsetInBytes, bytes.length ~/ 2);
  final out = Float32List(u16.length);
  final rounded = Float32List(1);
  final bits = Uint32List.view(rounded.buffer);
  for (var i = 0; i < u16.length; i++) {
    final sign = (u16[i] & 0x8000) == 0 ? 0 : 0x80000000;
    final e16 = (u16[i] >> 10) & 0x1f;
    final m = u16[i] & 0x3ff;
    var bits32 = 0;
    if (e16 == 0) {
      if (m == 0) {
        bits32 = sign;
      } else {
        // subnormal: value = m * 2^-24, normalize into fp32.
        var e = -14;
        var mm = m;
        while ((mm & 0x400) == 0) {
          mm <<= 1;
          e -= 1;
        }
        bits32 = sign | ((e + 127) << 23) | ((mm & 0x3ff) << 13);
      }
    } else if (e16 == 0x1f) {
      bits32 = sign | 0x7f800000 | (m << 13);
    } else {
      bits32 = sign | ((e16 + 112) << 23) | (m << 13);
    }
    bits[0] = bits32;
    out[i] = rounded[0];
  }
  return Tensor.float(out, shape);
}