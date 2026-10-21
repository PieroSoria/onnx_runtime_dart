import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_ort.dart';
import 'package:test/test.dart';

// ONNX Add(image, mask), FP32 [1,1,2,2], opset 17. No weights or DLL needed.
const _graph =
    'CAk6iwEKGgoFaW1hZ2UKBG1hc2sSBm91dHB1dCIDQWRkEgp0d29faW5wdXRzWh8KBWltYWdlEhYKFAgBEhAKAggBCgIIAQoCCAIKAggCWh4KBG1hc2sSFgoUCAESEAoCCAEKAggBCgIIAgoCCAJiIAoGb3V0cHV0EhYKFAgBEhAKAggBCgIIAQoCCAIKAggCQgQKABAR';

void main() {
  final ortLib = Platform.environment['ORT_LIBRARY'];
  final skipReason = ortLib == null || ortLib.isEmpty
      ? 'Set ORT_LIBRARY to an onnxruntime shared library to run native tests'
      : null;

  late Uint8List bytes;
  setUpAll(() => bytes = base64Decode(_graph));

  Tensor image() =>
      Tensor.float(Float32List.fromList([1, 2, 3, 4]), const [1, 1, 2, 2]);
  Tensor mask() =>
      Tensor.float(Float32List.fromList([.5, 0, 1, -1]), const [1, 1, 2, 2]);

  test('native metadata and inference', () async {
    final model = await OnnxSession.fromBytes(
      bytes,
      backend: OnnxBackend.ort,
      ort: OrtOptions(libraryPath: ortLib),
    );
    addTearDown(model.dispose);
    expect(model.inputSpecs.map((s) => s.name), ['image', 'mask']);
    expect(model.inputSpecs.first.shape, [1, 1, 2, 2]);
    expect(model.inputSpecs.first.elemType, 1);
    expect(model.outputNames, ['output']);
    expect((model as OrtRunner).providers, contains('CPUExecutionProvider'));

    final out = model.run({'image': image(), 'mask': mask()}, ['output']);
    expect(out['output']!.shape, [1, 1, 2, 2]);
    expect(out['output']!.asFloatList(), [1.5, 2, 4, 3]);

    final asyncOut = await model.runAsync(
      {'image': image(), 'mask': mask()},
      ['output'],
    );
    expect(asyncOut['output']!.asFloatList(), [1.5, 2, 4, 3]);
  }, skip: skipReason);

  test('native rejects bad shapes and closes idempotently', () async {
    final model = await OnnxSession.fromBytes(
      bytes,
      backend: OnnxBackend.ort,
      ort: OrtOptions(libraryPath: ortLib),
    );
    expect(
      () => model.run(
        {
          'image': Tensor.float(Float32List(4), const [4]),
          'mask': mask(),
        },
        ['output'],
      ),
      throwsArgumentError,
    );
    model.dispose();
    model.dispose();
    expect(
      () => model.run({'image': image(), 'mask': mask()}, ['output']),
      throwsA(isA<OrtException>()),
    );
  }, skip: skipReason);

  test('sync factory and unavailable CUDA warning', () {
    final model = OrtRunner.fromBytesSync(
      bytes,
      options: OrtOptions(
        libraryPath: ortLib,
        providers: const [OrtCudaProvider()],
      ),
    );
    addTearDown(model.dispose);
    if (!model.providers.contains('CUDAExecutionProvider')) {
      expect(model.providers, ['CPUExecutionProvider']);
      expect(model.warnings, isNotEmpty);
    }
    final out = model.run({'image': image(), 'mask': mask()}, ['output']);
    expect(out['output']!.asFloatList(), [1.5, 2, 4, 3]);
  }, skip: skipReason);

  test('native applies execution and memory options', () async {
    final model = await OnnxSession.fromBytes(
      bytes,
      backend: OnnxBackend.ort,
      ort: OrtOptions(
        libraryPath: ortLib,
        executionMode: OrtExecutionMode.parallel,
        enableMemPattern: false,
        enableCpuMemArena: false,
      ),
    );
    addTearDown(model.dispose);
    final out = model.run({'image': image(), 'mask': mask()}, ['output']);
    expect(out['output']!.asFloatList(), [1.5, 2, 4, 3]);
  }, skip: skipReason);

  test('auto falls back to Dart when the native library is missing', () async {
    final model = await OnnxSession.fromBytes(
      bytes,
      backend: OnnxBackend.auto,
      ort: const OrtOptions(libraryPath: 'missing-onnxruntime.dll'),
    );
    addTearDown(model.dispose);
    final out = model.run({'image': image(), 'mask': mask()}, ['output']);
    expect(out['output']!.asFloatList(), [1.5, 2, 4, 3]);
  });

  test('explicit native backend does not silently fall back', () async {
    await expectLater(
      OnnxSession.fromBytes(
        bytes,
        backend: OnnxBackend.ort,
        ort: const OrtOptions(libraryPath: 'missing-onnxruntime.dll'),
      ),
      throwsA(isA<OrtLibraryNotFoundException>()),
    );
  });

  test('fromFile fails for a missing model', () async {
    await expectLater(
      OnnxSession.fromFile('model.onnx', backend: OnnxBackend.dart),
      throwsA(isA<FileSystemException>()),
    );
  });

  test('native returns every requested output', () async {
    final model = await OnnxSession.fromBytes(
      File('test/fixtures/gqa_kvcache/model.onnx').readAsBytesSync(),
      backend: OnnxBackend.ort,
      ort: OrtOptions(libraryPath: ortLib),
    );
    addTearDown(model.dispose);
    expect(model.outputNames, ['out0', 'out1', 'out2']);
    final out = model.run(_feed(model), model.outputNames);
    expect(out.keys, ['out0', 'out1', 'out2']);
    expect(out['out0']!.shape, [1, 5, 32]);
    expect(out['out1']!.shape, [1, 2, 8, 8]);
    expect(out['out2']!.shape, [1, 2, 8, 8]);
  }, skip: skipReason);

  test('native int64 output matches the Dart interpreter', () async {
    final modelBytes = File('test/fixtures/argmax_axis1/model.onnx')
        .readAsBytesSync();
    final native = await OnnxSession.fromBytes(
      modelBytes,
      backend: OnnxBackend.ort,
      ort: OrtOptions(libraryPath: ortLib),
    );
    final dart = await OnnxSession.fromBytes(
      modelBytes,
      backend: OnnxBackend.dart,
    );
    addTearDown(native.dispose);
    addTearDown(dart.dispose);
    final feed = _feed(native);
    final expected = native.run(feed, native.outputNames);
    final actual = dart.run(feed, dart.outputNames);
    for (final name in native.outputNames) {
      expect(actual[name]!.shape, expected[name]!.shape);
      expect(actual[name]!.asIntList(), expected[name]!.asIntList());
    }
  }, skip: skipReason);
}

Map<String, Tensor> _feed(OnnxRunner model) {
  final feed = <String, Tensor>{};
  for (final spec in model.inputSpecs) {
    final shape = [
      for (var i = 0; i < spec.shape.length; i++)
        spec.shape[i] > 0 ? spec.shape[i] : 1,
    ];
    final count = shape.fold<int>(1, (a, b) => a * b);
    if (spec.elemType == 7 || spec.elemType == 6) {
      feed[spec.name] = Tensor.int64(
        Int64List(count)..fillRange(0, count, 1),
        shape,
      );
    } else {
      feed[spec.name] = Tensor.float(
        Float32List(count)..fillRange(0, count, .25),
        shape,
      );
    }
  }
  return feed;
}
