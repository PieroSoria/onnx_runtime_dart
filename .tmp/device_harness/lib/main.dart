import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

void main() {
  runApp(const MaterialApp(home: _HarnessScaffold()));
}

class _HarnessScaffold extends StatelessWidget {
  const _HarnessScaffold();
  @override
  Widget build(BuildContext context) => const _Harness();
}

class _Harness extends StatefulWidget {
  const _Harness();
  @override
  State<_Harness> createState() => _HarnessState();
}

class _HarnessState extends State<_Harness> {
  String _log = '';
  bool _started = false;
  void _logLine(String line) {
    setState(() => _log += '$line\n');
    debugPrint(line);
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_started) {
        _started = true;
        _run();
      }
    });
  }

  int rssKb() {
    try {
      final line = File('/proc/self/status').readAsLinesSync().firstWhere(
          (l) => l.startsWith('VmRSS:'));
      return int.parse(line.split(RegExp(r'\s+'))[1]);
    } catch (_) {
      return -1;
    }
  }

  Future<void> _run() async {
    final out = Directory.systemTemp;
    final report = <String, Object>{};
    final models = '/data/local/tmp/gemma';
    final control = OnnxExecutionControl(
        maxTensorBytes: 4 * 1024 * 1024 * 1024,
        deadline: DateTime.now().add(const Duration(minutes: 30)),
        onProgress: (phase, done, total) {
          if (done == total || done % 200 == 0) {
            _logLine('  $phase $done/$total rss=${rssKb()}kB');
          }
        });
    try {
      // ---- vision_encoder: load + grid-4 run --------------------------------
      _logLine('loading vision_encoder...');
      final t0 = DateTime.now();
      final vision =
          loadOnnxModel('$models/vision_encoder/model.onnx', control: control);
      final loadMs = DateTime.now().difference(t0).inMilliseconds;
      _logLine('vision loaded in ${loadMs}ms rss=${rssKb()}kB');
      final pixels = Float32List(1 * 4 * 4 * 768);
      final rng = math.Random(7);
      for (var i = 0; i < pixels.length; i++) {
        pixels[i] = _nextGaussian(rng);
      }
      final pos = Int64List(1 * 4 * 4 * 2);
      for (var r = 0; r < 4; r++) {
        for (var c = 0; c < 4; c++) {
          pos[(r * 4 + c) * 2] = r;
          pos[(r * 4 + c) * 2 + 1] = c;
        }
      }
      _logLine('running vision_encoder...');
      final t1 = DateTime.now();
      final yv = vision.run(
          {'pixel_values': Tensor.float(pixels, [1, 16, 768]),
            'pixel_position_ids': Tensor.int64(pos, [1, 16, 2])}, ['*'],
          control: control);
      final visionMs = DateTime.now().difference(t1).inMilliseconds;
      final visionRss = rssKb();
      report['vision_encoder'] = {
        'load_ms': loadMs,
        'run_ms': visionMs,
        'rss_kb_after': visionRss,
        'image_features': yv['image_features']!.shape,
      };
      _logLine('vision run ${visionMs}ms rss=${visionRss}kB '
          'shape=${yv['image_features']!.shape}');

      // ---- decoder: load + replicate step 0 from the existing oracle -------
      _logLine('loading decoder...');
      final t2 = DateTime.now();
      final decoder =
          loadOnnxModel('$models/decoder/model.onnx', control: control);
      final dLoadMs = DateTime.now().difference(t2).inMilliseconds;
      _logLine('decoder loaded in ${dLoadMs}ms rss=${rssKb()}kB');
      final cache = <String, Tensor>{
        for (final spec in decoder.inputSpecs)
          if (spec.name.startsWith('past_key_values.'))
            spec.name: Tensor.float(
                Float32List(0), [1, spec.shape[1], 0, spec.shape[3]])
      };
      final ids = _readBin('$models/0-embed-inputs_embeds.bin');
      final idsShape = [1, ids.length ~/ 1536, 1536];
      final layer = _readBin('$models/0-embed-per_layer_inputs.bin');
      _logLine('running decoder (prompt=${idsShape[1]} tokens)...');
      final t3 = DateTime.now();
      final y = decoder.run({
        'inputs_embeds': Tensor.float(ids, idsShape),
        'per_layer_inputs': Tensor.float(layer, [1, idsShape[1], 8960]),
        ...cache,
        'position_ids': Tensor.int64(
            Int64List.fromList(List.generate(idsShape[1], (i) => i)),
            [1, idsShape[1]]),
        'attention_mask': Tensor.int64(
            Int64List.fromList(List.filled(idsShape[1], 1)), [1, idsShape[1]]),
      }, decoder.outputNames, control: control);
      final decMs = DateTime.now().difference(t3).inMilliseconds;
      final decRss = rssKb();
      report['decoder'] = {
        'load_ms': dLoadMs,
        'step0_ms': decMs,
        'rss_kb_after': decRss,
        'logits_shape': y['logits']!.shape,
      };
      _logLine('decoder step0 ${decMs}ms rss=${decRss}kB '
          'logits=${y['logits']!.shape}');

      report['device'] = Platform.operatingSystem;
      final json = const JsonEncoder.withIndent('  ').convert(report);
      File('${out.path}/result.json').writeAsStringSync(json);
      _logLine('wrote ${out.path}/result.json:\n$json');
    } catch (e, st) {
      _logLine('ERROR: $e\n$st');
      File('${out.path}/result.json')
          .writeAsStringSync(jsonEncode({'error': '$e', 'trace': '$st'}));
    }
  }

  double _nextGaussian(math.Random rng) {
    double u = 0, v = 0;
    while (u == 0) {
      u = rng.nextDouble();
    }
    v = rng.nextDouble();
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * v);
  }

  Float32List _readBin(String path) {
    final bytes = File(path).readAsBytesSync();
    final list =
        Float32List.view(bytes.buffer, bytes.offsetInBytes, bytes.length ~/ 4);
    return Float32List.fromList(list);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('onnx_runtime_dart device harness')),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton(onPressed: _run, child: const Text('Run')),
            const SizedBox(height: 8),
            Expanded(
              child: SingleChildScrollView(
                child: SelectableText(_log,
                    style: const TextStyle(fontFamily: 'monospace')),
              ),
            ),
          ],
        ),
      ),
    );
  }
}