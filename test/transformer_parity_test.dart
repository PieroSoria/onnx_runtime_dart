import 'dart:convert';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';
import 'transformer_oracle_data.dart';

Tensor decode(Map<String, dynamic> value) {
  final shape = (value['shape'] as List).cast<int>();
  final data = value['data'] as List;
  return value['dtype'] == 'int64'
      ? Tensor.int64(data.map((v) => (v as num).toInt()).toList(), shape)
      : Tensor.float(
          Float32List.fromList(data.map((v) => (v as num).toDouble()).toList()),
          shape);
}

void main() {
  final cases = jsonDecode(transformerOraclesJson) as Map<String, dynamic>;
  for (final entry in cases.entries) {
    test('portable ORT parity: ${entry.key}', () {
      final model =
          OnnxModel.fromBytes(base64Decode(entry.value['model'] as String));
      final data = entry.value['case'];
      final inputs = (data['inputs'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, decode(v as Map<String, dynamic>)));
      final expected = (data['expected'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, decode(v as Map<String, dynamic>)));
      final actual = model.run(inputs, expected.keys.toList());
      for (final name in expected.keys) {
        final want = expected[name]!, got = actual[name]!;
        final tolerance = data['expected'][name]['dtype'] == 'float16' ? 2e-3 : 2e-5;
        expect(got.shape, want.shape);
        for (var i = 0; i < want.length; i++) {
          expect(got.getD(i).isFinite, isTrue, reason: '$name[$i]');
          expect(got.getD(i),
              closeTo(want.getD(i), tolerance + want.getD(i).abs() * tolerance),
              reason: '$name[$i]');
        }
      }
    });
  }
}
