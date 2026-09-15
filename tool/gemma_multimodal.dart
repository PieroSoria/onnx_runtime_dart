/// Runs the Gemma multimodal flow (real image -> vision -> embedding -> decoder)
/// against tool/gemma_multimodal.py's oracle, with persistent KV enabled.
library;
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

void main(List<String> args) {
  if (args.length != 2) {
    throw ArgumentError('Usage: gemma_multimodal.dart MODEL_DIR ORACLE_DIR');
  }
  final root = args[0], oracle = args[1];
  final reference = jsonDecode(File('$oracle/reference.json').readAsStringSync());
  var lastPhase = '';
  final control = OnnxExecutionControl(
      maxTensorBytes: 4 * 1024 * 1024 * 1024,
      deadline: DateTime.now().add(const Duration(minutes: 20)),
      onProgress: (phase, done, total) {
        if (phase != lastPhase || done == total || done % 100 == 0) {
          stdout.writeln('$phase $done/$total rss=${ProcessInfo.currentRss}');
          lastPhase = phase;
        }
      });
  final vision = loadOnnxModel('$root/vision_encoder/model.onnx', control: control);
  final embedding = loadOnnxModel('$root/embedding/model.onnx', control: control);
  final decoder = loadOnnxModel('$root/decoder/model.onnx', control: control);
  decoder.enablePersistentKv();

  Tensor read(Map value, [List<int>? shape]) {
    final bytes = File('$oracle/${value['file']}').readAsBytesSync();
    final dims = shape ?? (value['shape'] as List).cast<int>();
    if (value['dtype'] == 'int64') {
      return Tensor.int64(Int64List.view(bytes.buffer, bytes.offsetInBytes,
          bytes.length ~/ 8), dims);
    }
    return Tensor.float(Float32List.view(
        bytes.buffer, bytes.offsetInBytes, bytes.length ~/ 4), dims);
  }

  Map<String, Object> compare(Tensor got, Tensor expected) {
    if (got.length != expected.length ||
        got.shape.toString() != expected.shape.toString()) {
      throw StateError('Shape mismatch ${got.shape}/${expected.shape}');
    }
    double maxAbs = 0, sum = 0, norm = 0, dot = 0, gn = 0;
    for (var i = 0; i < got.length; i++) {
      final a = got.getD(i), b = expected.getD(i);
      if (!a.isFinite || !b.isFinite) throw StateError('Non-finite output at $i');
      final delta = (a - b).abs();
      maxAbs = math.max(maxAbs, delta);
      sum += delta * delta;
      norm += b * b;
      gn += a * a;
      dot += a * b;
    }
    return {
      'maxAbs': maxAbs,
      'rmse': math.sqrt(sum / math.max(1, got.length)),
      'cosine': dot / math.sqrt(norm * gn),
    };
  }

  // ---- vision encoder on the real image -----------------------------------
  final grid = (reference['grid'] as num).toInt();
  final pixels = _readFp16(
      File('$oracle/pixel-values.bin').readAsBytesSync(), [1, grid * grid, 768]);
  final posBytes = File('$oracle/pixel-position-ids.bin').readAsBytesSync();
  final pos = Tensor.int64(Int64List.view(posBytes.buffer, posBytes.offsetInBytes,
      posBytes.length ~/ 8), [1, grid * grid, 2]);
  final sw = Stopwatch()..start();
  final visionAll = vision.run({'pixel_values': pixels, 'pixel_position_ids': pos}, ['*']);
  final features = visionAll['image_features']!;
  final vReport = compare(features, read(reference['image-features'], features.shape));
  stdout.writeln('image_features ${jsonEncode(vReport)}');

  // ---- embedding graph splices the feature(s) at the image tokens ----------
  final cache = <String, Tensor>{
    for (final spec in decoder.inputSpecs)
      if (spec.name.startsWith('past_key_values.'))
        spec.name: Tensor.float(
            Float32List(0), [1, spec.shape[1], 0, spec.shape[3]])
  };
  final records = <Map<String, Object>>[];
  var past = 0;
  var generated = <int>[];
  for (var step = 0; step < (reference['steps'] as List).length; step++) {
    final record = reference['steps'][step];
    final ids = (record['ids'][0] as List).cast<int>();
    final stepTimer = Stopwatch()..start();
    final imageFeat = step == 0
        ? features
        : Tensor.float(Float32List(features.length), features.shape);
    final e = embedding.run({
      'input_ids': Tensor.int64(ids, [1, ids.length]),
      'image_features': imageFeat,
      'audio_features': Tensor.float(Float32List(0), [0, 1536]),
    }, embedding.outputNames, control: control);
    final embMetrics = <String, Object>{
      for (final name in e.keys)
        name: compare(
            e[name]!, read(record['embedding'][name], e[name]!.shape)),
    };
    final y = decoder.run({
      ...e,
      ...cache,
      'position_ids': Tensor.int64(
          List.generate(ids.length, (i) => past + i), [1, ids.length]),
      'attention_mask':
          Tensor.int64(List.filled(past + ids.length, 1), [1, past + ids.length]),
    }, decoder.outputNames, control: control);
    final logits = y['logits']!;
    final width = logits.shape.last, offset = logits.length - width;
    var token = 0;
    for (var i = 1; i < width; i++) {
      if (logits.getD(offset + i) > logits.getD(offset + token)) token = i;
    }
    final metrics = <String, Object>{
      'embedding': embMetrics,
      'logits': compare(logits, read(record['logits'], logits.shape)),
      'token': token,
      'expectedToken': record['token'],
      'seconds': stepTimer.elapsedMilliseconds / 1000,
      'rss': ProcessInfo.currentRss,
    };
    stdout.writeln('step $step ${jsonEncode(metrics)}');
    records.add(metrics);
    generated.add(token);
    if (token != (record['token'] as num).toInt()) {
      exitCode = 1;
      break;
    }
    for (final name in cache.keys.toList()) {
      cache[name] = y[name.replaceFirst('past_key_values.', 'present.')]!;
    }
    past += ids.length;
  }
  final report = {
    'image_features': vReport,
    'steps': records,
    'generated': generated,
    'expected': reference['generated'],
    'seconds': sw.elapsedMilliseconds / 1000,
    'rss': ProcessInfo.currentRss,
  };
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