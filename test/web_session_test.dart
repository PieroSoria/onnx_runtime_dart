@TestOn('node')
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

const _addModel =
    'CAk6iwEKGgoFaW1hZ2UKBG1hc2sSBm91dHB1dCIDQWRkEgp0d29faW5wdXRzWh8KBWltYWdlEhYKFAgBEhAKAggBCgIIAQoCCAIKAggCWh4KBG1hc2sSFgoUCAESEAoCCAEKAggBCgIIAgoCCAJiIAoGb3V0cHV0EhYKFAgBEhAKAggBCgIIAQoCCAIKAggCQgQKABAR';

void main() {
  test('web build selects the Dart backend and never exposes ORT', () async {
    expect(OnnxSession.isOrtSupported, isFalse);
    await expectLater(
      OnnxSession.fromBytes(base64Decode(_addModel), backend: OnnxBackend.ort),
      throwsA(isA<UnsupportedError>()),
    );

    final model = await OnnxSession.fromBytes(base64Decode(_addModel));
    addTearDown(model.dispose);
    final out = model.run(
      {
        'image': Tensor.float(Float32List.fromList([1, 2, 3, 4]), [1, 1, 2, 2]),
        'mask': Tensor.float(Float32List.fromList([.5, 0, 1, -1]), [
          1,
          1,
          2,
          2,
        ]),
      },
      ['output'],
    );
    expect(out['output']!.asFloatList(), [1.5, 2, 4, 3]);
  });

  test('fromFile is rejected without dart:io', () async {
    await expectLater(
      OnnxSession.fromFile('model.onnx'),
      throwsA(isA<UnsupportedError>()),
    );
  });
}
