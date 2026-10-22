import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:test/test.dart';

NodeProto node(
  String op,
  List<String> ins,
  List<String> outs, [
  List<AttributeProto> attrs = const [],
]) => NodeProto()
  ..opType = op
  ..input.addAll(ins)
  ..output.addAll(outs)
  ..attribute.addAll(attrs);

AttributeProto iAttr(String name, int value) => AttributeProto()
  ..name = name
  ..i = Int64(value);

AttributeProto fAttr(String name, double value) => AttributeProto()
  ..name = name
  ..f = value;

AttributeProto isAttr(String name, List<int> values) => AttributeProto()
  ..name = name
  ..ints.addAll(values.map(Int64.new));

AttributeProto sAttr(String name, String value) => AttributeProto()
  ..name = name
  ..s = utf8.encode(value);

AttributeProto ssAttr(String name, List<String> values) => AttributeProto()
  ..name = name
  ..strings.addAll(values.map(utf8.encode));

AttributeProto gAttr(String name, GraphProto value) => AttributeProto()
  ..name = name
  ..type = AttributeProto_AttributeType.GRAPH
  ..g = value;

GraphProto subgraph(
  List<NodeProto> nodes, {
  required List<String> inputs,
  required List<String> outputs,
}) => GraphProto()
  ..input.addAll(inputs.map((s) => ValueInfoProto()..name = s))
  ..output.addAll(outputs.map((s) => ValueInfoProto()..name = s))
  ..node.addAll(nodes);

OnnxModel modelOf(
  List<NodeProto> nodes, {
  List<String> inputs = const ['X'],
  List<String> outputs = const ['Y'],
  List<TensorProto> initializers = const [],
  int opset = 26,
}) => OnnxModel.fromBytes(
  (ModelProto()
        ..opsetImport.add(
          OperatorSetIdProto()
            ..domain = ''
            ..version = Int64(opset),
        )
        ..graph = (GraphProto()
          ..input.addAll(inputs.map((s) => ValueInfoProto()..name = s))
          ..output.addAll(outputs.map((s) => ValueInfoProto()..name = s))
          ..initializer.addAll(initializers)
          ..node.addAll(nodes)))
      .writeToBuffer(),
);

Tensor ft(List<int> shape, List<double> values) =>
    Tensor.float(Float32List.fromList(values), shape);

Tensor it(List<int> shape, List<int> values) =>
    Tensor.int64(Int64List.fromList(values), shape);

void expectClose(
  List<double> actual,
  List<double> expected, {
  double tol = 1e-5,
}) {
  expect(actual.length, expected.length);
  for (var k = 0; k < actual.length; k++) {
    expect(actual[k], closeTo(expected[k], tol), reason: 'index $k');
  }
}

void main() {
  test('unary math and activation ops', () {
    final cases = <String, (double, double)>{
      'Acos': (0, math.pi / 2),
      'Acosh': (1, 0),
      'Asin': (0, 0),
      'Asinh': (0, 0),
      'Atanh': (0, 0),
      'Cosh': (0, 1),
      'Sinh': (0, 0),
      'Tan': (0, 0),
      'Mish': (0, 0),
      'Softsign': (1, 0.5),
      'Celu': (0, 0),
      'Selu': (0, 0),
      'Shrink': (0.75, 0.75),
      'Swish': (0, 0),
      'ThresholdedRelu': (0.5, 0),
    };
    for (final entry in cases.entries) {
      final model = modelOf([
        node(entry.key, ['X'], ['Y']),
      ]);
      final y = model.run(
        {
          'X': ft([1], [entry.value.$1]),
        },
        ['Y'],
      )['Y']!;
      expect(
        y.asFloatList()[0],
        closeTo(entry.value.$2, 1e-5),
        reason: entry.key,
      );
    }
  });

  test('Hardmax selects the first maximum on the requested axis', () {
    final model = modelOf([
      node('Hardmax', ['X'], ['Y'], [iAttr('axis', -1)]),
    ]);
    final y = model.run(
      {
        'X': ft([2, 3], [1, 3, 3, 2, 1, 0]),
      },
      ['Y'],
    )['Y']!;
    expect(y.asFloatList(), [0, 1, 0, 1, 0, 0]);
  });

  test('Sum and Mean are variadic', () {
    final sum = modelOf(
      [
        node('Sum', ['A', 'B', 'C'], ['Y']),
      ],
      inputs: ['A', 'B', 'C'],
    );
    final sumY = sum.run(
      {
        'A': ft([2], [1, 2]),
        'B': ft([2], [3, 4]),
        'C': ft([2], [5, 6]),
      },
      ['Y'],
    )['Y']!;
    expect(sumY.asFloatList(), [9, 12]);

    final mean = modelOf(
      [
        node('Mean', ['A', 'B', 'C'], ['Y']),
      ],
      inputs: ['A', 'B', 'C'],
    );
    final meanY = mean.run(
      {
        'A': ft([2], [1, 2]),
        'B': ft([2], [3, 4]),
        'C': ft([2], [5, 6]),
      },
      ['Y'],
    )['Y']!;
    expect(meanY.asFloatList(), [3, 4]);
  });

  test('ReduceL1, ReduceLogSum and ReduceLogSumExp', () {
    Tensor evaluate(String op, List<int> input) {
      final model = modelOf(
        [
          node(op, ['X', 'Axes'], ['Y'], [iAttr('keepdims', 0)]),
        ],
        inputs: ['X', 'Axes'],
      );
      return model.run(
        {
          'X': it([2, 2], input),
          'Axes': it([1], [-1]),
        },
        ['Y'],
      )['Y']!;
    }

    final l1 = evaluate('ReduceL1', [-1, 2, 3, -4]);
    expect(l1.asIntList(), [3, 7]);
    final logSum = evaluate('ReduceLogSum', [1, 2, 3, 4]);
    expectClose(logSum.asFloatList(), [math.log(3), math.log(7)]);
    final logSumExp = evaluate('ReduceLogSumExp', [0, 0, 0, 0]);
    expectClose(logSumExp.asFloatList(), [math.log(2), math.log(2)]);
  });

  test('CumProd supports exclusive and reverse', () {
    final model = modelOf(
      [
        node('CumProd', ['X', 'Axis'], ['Y']),
      ],
      inputs: ['X', 'Axis'],
    );
    final y = model.run(
      {
        'X': ft([4], [2, 3, 4, 5]),
        'Axis': it([], [0]),
      },
      ['Y'],
    )['Y']!;
    expect(y.asFloatList(), [2, 6, 24, 120]);

    final exclusive = modelOf(
      [
        node('CumProd', ['X', 'Axis'], ['Y'], [
          iAttr('exclusive', 1),
          iAttr('reverse', 1),
        ]),
      ],
      inputs: ['X', 'Axis'],
    );
    final ey = exclusive.run(
      {
        'X': ft([4], [2, 3, 4, 5]),
        'Axis': it([], [0]),
      },
      ['Y'],
    )['Y']!;
    expect(ey.asFloatList(), [60, 20, 5, 1]);
  });

  test('CastLike and EyeLike honor their target types', () {
    final cast = modelOf(
      [
        node('CastLike', ['X', 'T'], ['Y']),
      ],
      inputs: ['X', 'T'],
    );
    final castY = cast.run(
      {
        'X': ft([3], [1.9, -2.1, 0]),
        'T': it([1], [0]),
      },
      ['Y'],
    )['Y']!;
    expect(castY.dtype, DType.int64);
    expect(castY.asIntList(), [1, -2, 0]);

    final eye = modelOf([
      node('EyeLike', ['X'], ['Y'], [iAttr('k', 1)]),
    ]);
    final eyeY = eye.run(
      {
        'X': ft([2, 3], [0, 0, 0, 0, 0, 0]),
      },
      ['Y'],
    )['Y']!;
    expect(eyeY.asFloatList(), [0, 1, 0, 0, 0, 1]);
  });

  test('Compress selects values globally or along an axis', () {
    final flat = modelOf(
      [
        node('Compress', ['X', 'C'], ['Y']),
      ],
      inputs: ['X', 'C'],
    );
    final flatY = flat.run(
      {
        'X': ft([5], [10, 20, 30, 40, 50]),
        'C': it([5], [0, 1, 1, 0, 1]),
      },
      ['Y'],
    )['Y']!;
    expect(flatY.asFloatList(), [20, 30, 50]);

    final axis = modelOf(
      [
        node('Compress', ['X', 'C'], ['Y'], [iAttr('axis', 1)]),
      ],
      inputs: ['X', 'C'],
    );
    final axisY = axis.run(
      {
        'X': ft([2, 3], [1, 2, 3, 4, 5, 6]),
        'C': it([3], [1, 0, 1]),
      },
      ['Y'],
    )['Y']!;
    expect(axisY.shape, [2, 2]);
    expect(axisY.asFloatList(), [1, 3, 4, 6]);
  });

  test('ReverseSequence reverses each batch prefix', () {
    final model = modelOf(
      [
        node('ReverseSequence', ['X', 'L'], ['Y'], [
          iAttr('time_axis', 0),
          iAttr('batch_axis', 1),
        ]),
      ],
      inputs: ['X', 'L'],
    );
    final y = model.run(
      {
        'X': ft([4, 2], [0, 10, 1, 11, 2, 12, 3, 13]),
        'L': it([2], [2, 3]),
      },
      ['Y'],
    )['Y']!;
    expect(y.asFloatList(), [1, 12, 0, 11, 2, 10, 3, 13]);
  });

  test('DepthToSpace and SpaceToDepth match the ONNX layouts', () {
    final d2s = modelOf([
      node('DepthToSpace', ['X'], ['Y'], [
        iAttr('blocksize', 2),
        sAttr('mode', 'DCR'),
      ]),
    ]);
    final d2sY = d2s.run(
      {
        'X': ft([1, 8, 2, 2], [for (int k = 0; k < 32; k++) k.toDouble()]),
      },
      ['Y'],
    )['Y']!;
    expect(d2sY.shape, [1, 2, 4, 4]);
    expect(d2sY.asFloatList().take(8), [0, 8, 1, 9, 16, 24, 17, 25]);

    final s2d = modelOf([
      node('SpaceToDepth', ['X'], ['Y'], [iAttr('blocksize', 2)]),
    ]);
    final s2dY = s2d.run(
      {
        'X': ft([1, 2, 4, 4], [for (int k = 0; k < 32; k++) k.toDouble()]),
      },
      ['Y'],
    )['Y']!;
    expect(s2dY.shape, [1, 8, 2, 2]);
    expect(s2dY.asFloatList().take(8), [0, 2, 8, 10, 16, 18, 24, 26]);
  });

  test('ScatterElements applies all reductions', () {
    for (final reduction in ['none', 'add', 'mul', 'min', 'max']) {
      final model = modelOf(
        [
          node('ScatterElements', ['D', 'I', 'U'], ['Y'], [
            iAttr('axis', 1),
            sAttr('reduction', reduction),
          ]),
        ],
        inputs: ['D', 'I', 'U'],
      );
      final y = model.run(
        {
          'D': ft([2, 3], [0, 0, 0, 0, 0, 0]),
          'I': it([2, 2], [0, 1, 2, 0]),
          'U': ft([2, 2], [1, 2, 3, 4]),
        },
        ['Y'],
      )['Y']!;
      final want = switch (reduction) {
        'add' => [1, 2, 0, 4, 0, 3],
        'mul' => [0, 0, 0, 0, 0, 0],
        'min' => [0, 0, 0, 0, 0, 0],
        'max' => [1, 2, 0, 4, 0, 3],
        _ => [1, 2, 0, 4, 0, 3],
      };
      expect(y.asFloatList(), want, reason: reduction);
    }
  });

  test('Unique returns values, indices, inverse and counts', () {
    final flat = modelOf(
      [
        node('Unique', ['X'], ['Y', 'I', 'INV', 'C']),
      ],
      outputs: ['Y', 'I', 'INV', 'C'],
    );
    final y = flat.run(
      {
        'X': ft([6], [2, 1, 2, 3, 1, 2]),
      },
      ['Y', 'I', 'INV', 'C'],
    );
    expect(y['Y']!.asFloatList(), [1, 2, 3]);
    expect(y['I']!.asIntList(), [1, 0, 3]);
    expect(y['INV']!.asIntList(), [1, 0, 1, 2, 0, 1]);
    expect(y['C']!.asIntList(), [2, 3, 1]);

    final axis = modelOf(
      [
        node('Unique', ['X'], ['Y', 'I', 'INV', 'C'], [iAttr('axis', 0)]),
      ],
      outputs: ['Y', 'I', 'INV', 'C'],
    );
    final ay = axis.run(
      {
        'X': ft([3, 2], [1, 2, 1, 2, 3, 4]),
      },
      ['Y', 'I', 'INV', 'C'],
    );
    expect(ay['Y']!.shape, [2, 2]);
    expect(ay['Y']!.asFloatList(), [1, 2, 3, 4]);
    expect(ay['INV']!.asIntList(), [0, 0, 1]);
    expect(ay['C']!.asIntList(), [2, 1]);
  });

  test('Det computes determinants over batched matrices', () {
    final model = modelOf([
      node('Det', ['X'], ['Y']),
    ]);
    final y = model.run(
      {
        'X': ft([2, 2, 2], [1, 2, 3, 4, 2, 0, 0, 3]),
      },
      ['Y'],
    )['Y']!;
    expectClose(y.asFloatList(), [-2, 6]);
  });

  test('bitwise and boolean logical ops', () {
    final model = modelOf(
      [
        node('BitwiseAnd', ['A', 'B'], ['And']),
        node('BitwiseOr', ['A', 'B'], ['Or']),
        node('BitwiseXor', ['A', 'B'], ['Xor']),
        node('BitwiseNot', ['A'], ['Not']),
        node('BitShift', ['A', 'S'], ['Shift'], [sAttr('direction', 'LEFT')]),
        node('Xor', ['C', 'D'], ['BoolXor']),
      ],
      inputs: ['A', 'B', 'S', 'C', 'D'],
      outputs: ['And', 'Or', 'Xor', 'Not', 'Shift', 'BoolXor'],
    );
    final y = model.run(
      {
        'A': it([2], [6, 5]),
        'B': it([2], [3, 3]),
        'S': it([2], [1, 1]),
        'C': it([2], [1, 0]),
        'D': it([2], [0, 1]),
      },
      ['And', 'Or', 'Xor', 'Not', 'Shift', 'BoolXor'],
    );
    expect(y['And']!.asIntList(), [2, 1]);
    expect(y['Or']!.asIntList(), [7, 7]);
    expect(y['Xor']!.asIntList(), [5, 6]);
    expect(y['Not']!.asIntList(), [~6, ~5]);
    expect(y['Shift']!.asIntList(), [12, 10]);
    expect(y['BoolXor']!.asIntList(), [1, 1]);
  });

  test('LpNormalization, MeanVarianceNormalization and LRN', () {
    final lp = modelOf([
      node('LpNormalization', ['X'], ['Y'], [iAttr('axis', -1), iAttr('p', 2)]),
    ]);
    final lpY = lp.run(
      {
        'X': ft([2, 2], [3, 4, 0, 0]),
      },
      ['Y'],
    )['Y']!;
    expectClose(lpY.asFloatList(), [0.6, 0.8, 0, 0]);

    final mvn = modelOf([
      node('MeanVarianceNormalization', ['X'], ['Y']),
    ]);
    final mvnY = mvn.run(
      {
        'X': ft([1, 1, 2, 2], [1, 2, 3, 4]),
      },
      ['Y'],
    )['Y']!;
    expectClose(mvnY.asFloatList(), [
      -1.3416407,
      -0.4472136,
      0.4472136,
      1.3416407,
    ]);

    final lrn = modelOf([
      node('LRN', ['X'], ['Y'], [iAttr('size', 3)]),
    ]);
    final lrnY = lrn.run(
      {
        'X': ft([1, 3, 2, 2], [for (int k = 0; k < 12; k++) k.toDouble()]),
      },
      ['Y'],
    )['Y']!;
    expect(lrnY.asFloatList()[1], closeTo(0.9993504, 1e-6));
  });

  test('LpPool and GlobalLpPool', () {
    final lp = modelOf([
      node('LpPool', ['X'], ['Y'], [
        isAttr('kernel_shape', [2, 2]),
        iAttr('p', 2),
      ]),
    ]);
    final lpY = lp.run(
      {
        'X': ft([1, 1, 3, 3], [for (int k = 0; k < 9; k++) k.toDouble()]),
      },
      ['Y'],
    )['Y']!;
    expect(lpY.shape, [1, 1, 2, 2]);
    expectClose(lpY.asFloatList(), [5.0990195, 6.78233, 10.488088, 12.409674]);

    final global = modelOf([
      node('GlobalLpPool', ['X'], ['Y'], [iAttr('p', 2)]),
    ]);
    final gY = global.run(
      {
        'X': ft([1, 2, 2, 2], [1, 2, 2, 0, 3, 4, 0, 0]),
      },
      ['Y'],
    )['Y']!;
    expect(gY.shape, [1, 2, 1, 1]);
    expectClose(gY.asFloatList(), [3, 5]);
  });

  test('random-like ops are seeded and shaped correctly', () {
    final uniform = modelOf([
      node('RandomUniformLike', ['X'], ['Y'], [
        fAttr('low', -1),
        fAttr('high', 2),
        fAttr('seed', 7),
      ]),
    ]);
    final uY = uniform.run(
      {
        'X': ft([2, 3], List.filled(6, 0)),
      },
      ['Y'],
    )['Y']!;
    expect(uY.shape, [2, 3]);
    for (final v in uY.asFloatList()) {
      expect(v, inInclusiveRange(-1, 2));
    }

    final bernoulli = modelOf([
      node('Bernoulli', ['X'], ['Y'], [fAttr('seed', 7)]),
    ]);
    final bY = bernoulli.run(
      {
        'X': ft([20], List.filled(20, 0.5)),
      },
      ['Y'],
    )['Y']!;
    expect(bY.shape, [20]);
    expect(bY.asFloatList().every((v) => v == 0 || v == 1), isTrue);

    final multinomial = modelOf([
      node('Multinomial', ['X'], ['Y'], [
        iAttr('sample_size', 4),
        fAttr('seed', 7),
      ]),
    ]);
    final mY = multinomial.run(
      {
        'X': ft([1, 2], [0, 1]),
      },
      ['Y'],
    )['Y']!;
    expect(mY.shape, [1, 4]);
    expect(mY.asIntList(), everyElement(1));
  });

  test('window functions follow periodic/symmetric definitions', () {
    for (final entry in {
      'HannWindow': [0, 0.5, 1, 0.5],
      'HammingWindow': [0.08, 0.54, 1, 0.54],
      'BlackmanWindow': [0, 0.34, 1, 0.34],
    }.entries) {
      final model = modelOf(
        [
          node(entry.key, ['N'], ['Y'], [iAttr('periodic', 1)]),
        ],
        inputs: ['N'],
      );
      final y = model.run(
        {
          'N': it([], [4]),
        },
        ['Y'],
      )['Y']!;
      expectClose(
        y.asFloatList(),
        entry.value.map((v) => v.toDouble()).toList(),
        tol: 1e-6,
      );
    }
  });

  test('negative log likelihood and softmax cross entropy losses', () {
    final nll = modelOf(
      [
        node('NegativeLogLikelihoodLoss', ['X', 'T'], ['Y'], [
          sAttr('reduction', 'none'),
        ]),
      ],
      inputs: ['X', 'T'],
    );
    final nllY = nll.run(
      {
        'X': ft(
          [2, 3],
          [
            math.log(0.2),
            math.log(0.3),
            math.log(0.5),
            math.log(0.1),
            math.log(0.7),
            math.log(0.2),
          ],
        ),
        'T': it([2], [1, 2]),
      },
      ['Y'],
    )['Y']!;
    expectClose(nllY.asFloatList(), [1.2039728, 1.6094379]);

    final sce = modelOf(
      [
        node('SoftmaxCrossEntropyLoss', ['X', 'T'], ['Y', 'L']),
      ],
      inputs: ['X', 'T'],
      outputs: ['Y', 'L'],
    );
    final sceY = sce.run(
      {
        'X': ft([2, 3], [1, 2, 3, 1, 1, 1]),
        'T': it([2], [2, 0]),
      },
      ['Y', 'L'],
    );
    expect(sceY['Y']!.asFloatList().first, closeTo(0.7531091, 1e-6));
    expectClose(sceY['L']!.asFloatList(), [
      -2.4076059,
      -1.4076059,
      -0.40760595,
      -1.0986123,
      -1.0986123,
      -1.0986123,
    ]);
  });

  test('CenterCropPad crops and pads around the center', () {
    final m = modelOf(
      [
        node('CenterCropPad', ['X', 'S'], ['Y']),
      ],
      inputs: ['X', 'S'],
    );
    final y = m.run(
      {
        'X': ft([2, 3], [0, 1, 2, 3, 4, 5]),
        'S': it([2], [3, 2]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [3, 2]);
    expectClose(y.asFloatList(), [0, 1, 3, 4, 0, 0]);
  });

  test('AffineGrid builds normalized sampling grids', () {
    final m = modelOf(
      [
        node('AffineGrid', ['T', 'S'], ['Y']),
      ],
      inputs: ['T', 'S'],
    );
    final y = m.run(
      {
        'T': ft([1, 2, 3], [1, 0, 0, 0, 1, 0]),
        'S': it([4], [1, 1, 2, 3]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [1, 2, 3, 2]);
    expectClose(y.asFloatList(), [
      -2 / 3,
      -0.5,
      0,
      -0.5,
      2 / 3,
      -0.5,
      -2 / 3,
      0.5,
      0,
      0.5,
      2 / 3,
      0.5,
    ]);
  });

  test('Col2Im folds block columns into an image', () {
    final m = modelOf(
      [
        node('Col2Im', ['X', 'I', 'B'], ['Y']),
      ],
      inputs: ['X', 'I', 'B'],
    );
    final y = m.run(
      {
        'X': ft(
          [1, 5, 5],
          [
            1, 6, 11, 16, 21, //
            2, 7, 12, 17, 22,
            3, 8, 13, 18, 23,
            4, 9, 14, 19, 24,
            5, 0, 15, 20, 25,
          ],
        ),
        'I': it([2], [5, 5]),
        'B': it([2], [1, 5]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [1, 1, 5, 5]);
    expectClose(y.asFloatList(), [
      1, 2, 3, 4, 5, //
      6, 7, 8, 9, 0,
      11, 12, 13, 14, 15,
      16, 17, 18, 19, 20,
      21, 22, 23, 24, 25,
    ]);
  });

  test('DFT, RFFT and IRFFT match the ONNX reference', () {
    final forward = modelOf([
      node('DFT', ['X'], ['Y']),
    ]);
    final y = forward.run(
      {
        'X': ft([1, 4, 1], [1, 2, 3, 4]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [1, 4, 2]);
    expectClose(y.asFloatList(), [10, 0, -2, 2, -2, 0, -2, -2]);

    final rfft = modelOf([
      node('DFT', ['X'], ['Y'], [iAttr('onesided', 1)]),
    ]);
    final yr = rfft.run(
      {
        'X': ft([1, 4, 1], [1, 2, 3, 4]),
      },
      ['Y'],
    )['Y']!;
    expect(yr.shape, [1, 3, 2]);
    expectClose(yr.asFloatList(), [10, 0, -2, 2, -2, 0]);

    final irfft = modelOf([
      node('DFT', ['X'], ['Y'], [iAttr('onesided', 1), iAttr('inverse', 1)]),
    ]);
    final yi = irfft.run(
      {
        'X': ft([1, 3, 2], [10, 0, -2, 2, -2, 0]),
      },
      ['Y'],
    )['Y']!;
    expect(yi.shape, [1, 4, 1]);
    expectClose(yi.asFloatList(), [1, 2, 3, 4]);
  });

  test('MaxUnpool scatters values using spatial indices', () {
    final m = modelOf(
      [
        node('MaxUnpool', ['X', 'I'], ['Y'], [
          isAttr('kernel_shape', [2, 2]),
          isAttr('strides', [2, 2]),
        ]),
      ],
      inputs: ['X', 'I'],
    );
    final y = m.run(
      {
        'X': ft([1, 1, 2, 2], [6, 8, 14, 16]),
        'I': it([1, 1, 2, 2], [5, 7, 13, 15]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [1, 1, 4, 4]);
    expectClose(y.asFloatList(), [
      0, 0, 0, 0, //
      0, 6, 0, 8,
      0, 0, 0, 0,
      0, 14, 0, 16,
    ]);
  });

  test('MaxRoiPool pools each ROI into a fixed grid', () {
    final m = modelOf(
      [
        node('MaxRoiPool', ['X', 'R'], ['Y'], [
          isAttr('pooled_shape', [2, 2]),
        ]),
      ],
      inputs: ['X', 'R'],
    );
    final y = m.run(
      {
        'X': ft([1, 1, 4, 4], List.generate(16, (i) => i.toDouble())),
        'R': ft([1, 5], [0, 0, 0, 3, 3]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [1, 1, 2, 2]);
    expectClose(y.asFloatList(), [5, 7, 13, 15]);
  });

  test('MelWeightMatrix builds triangular filters', () {
    final m = modelOf(
      [
        node('MelWeightMatrix', ['M', 'D', 'SR', 'L', 'H'], ['Y']),
      ],
      inputs: ['M', 'D', 'SR', 'L', 'H'],
    );
    final y = m.run(
      {
        'M': it([], [3]),
        'D': it([], [8]),
        'SR': it([], [8000]),
        'L': ft([], [0]),
        'H': ft([], [4000]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [5, 3]);
    expectClose(y.asFloatList(), [
      1, 1, 0, //
      0, 0, 1,
      0, 0, 0,
      0, 0, 0,
      0, 0, 0,
    ]);
  });

  test('TensorScatter writes updates at batch-local indices', () {
    final m = modelOf(
      [
        node('TensorScatter', ['P', 'U', 'W'], ['Y'], [iAttr('axis', -1)]),
      ],
      inputs: ['P', 'U', 'W'],
    );
    final y = m.run(
      {
        'P': ft([2, 4], List.generate(8, (i) => i.toDouble())),
        'U': ft([2, 2], List.filled(4, 9)),
        'W': it([2], [1, 2]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [2, 4]);
    expectClose(y.asFloatList(), [0, 9, 9, 3, 4, 5, 9, 9]);
  });

  test('DeformConv handles offsets, masks, bias and groups', () {
    final m = modelOf(
      [
        node('DeformConv', ['X', 'W', 'O', 'B', 'M'], ['Y']),
      ],
      inputs: ['X', 'W', 'O', 'B', 'M'],
    );
    final x = ft([1, 1, 3, 3], List.generate(9, (i) => i + 1.0));
    final w = ft([1, 1, 2, 2], [1, 1, 1, 1]);
    final off = ft(
      [1, 8, 2, 2],
      [0.5, 0, 0, 0, 0.5, 0, 0, 0, ...List.filled(24, 0)],
    );
    final mask = ft(
      [1, 4, 2, 2],
      [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    );
    final y = m.run(
      {
        'X': x,
        'W': w,
        'O': off,
        'B': ft([1], [1.5]),
        'M': mask,
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [1, 1, 2, 2]);
    expectClose(y.asFloatList(), [14.0, 17.5, 25.5, 29.5]);

    final noMask = modelOf(
      [
        node('DeformConv', ['X', 'W', 'O'], ['Y']),
      ],
      inputs: ['X', 'W', 'O'],
    );
    final y2 = noMask.run(
      {
        'X': ft(
          [1, 2, 3, 3],
          [
            ...List.generate(9, (i) => i + 1.0),
            ...List.generate(9, (i) => (i + 1) * 10.0),
          ],
        ),
        'W': ft([1, 2, 2, 2], List.filled(8, 1)),
        'O': ft([1, 8, 2, 2], List.filled(32, 0)),
      },
      ['Y'],
    )['Y']!;
    expectClose(y2.asFloatList(), [132, 176, 264, 308]);
  });

  test('CausalConvWithState uses the carry state and fused activation', () {
    final m = modelOf(
      [
        node('CausalConvWithState', ['X', 'W', 'B', 'S'], ['Y', 'P'], [
          sAttr('activation', 'silu'),
        ]),
      ],
      inputs: ['X', 'W', 'B', 'S'],
      outputs: ['Y', 'P'],
      opset: 27,
    );
    final out = m.run(
      {
        'X': ft([1, 1, 3], [1, 2, 3]),
        'W': ft([1, 1, 2], [1, 1]),
        'B': ft([1], [10]),
        'S': ft([1, 1, 1], [0.5]),
      },
      ['Y', 'P'],
    );
    expect(out['Y']!.shape, [1, 1, 3]);
    expectClose(out['Y']!.asFloatList(), [11.499884, 12.99997, 14.999994]);
    expectClose(out['P']!.asFloatList(), [3]);
  });

  test('LinearAttention supports GQA and gated delta updates', () {
    final linear = modelOf(
      [
        node('LinearAttention', ['Q', 'K', 'V'], ['Y', 'S'], [
          sAttr('update_rule', 'linear'),
          iAttr('q_num_heads', 2),
          iAttr('kv_num_heads', 1),
        ]),
      ],
      inputs: ['Q', 'K', 'V'],
      outputs: ['Y', 'S'],
      opset: 27,
    );
    final linearOut = linear.run(
      {
        'Q': ft([1, 1, 4], [1, 0, 1, 0]),
        'K': ft([1, 1, 2], [1, 2]),
        'V': ft([1, 1, 1], [3]),
      },
      ['Y', 'S'],
    );
    expect(linearOut['Y']!.shape, [1, 1, 2]);
    expectClose(linearOut['Y']!.asFloatList(), [2.1213202, 2.1213202]);
    expectClose(linearOut['S']!.asFloatList(), [3, 6]);

    final delta = modelOf(
      [
        node('LinearAttention', ['Q', 'K', 'V', '', 'D', 'B'], ['Y', 'S'], [
          sAttr('update_rule', 'gated_delta'),
          iAttr('q_num_heads', 1),
          iAttr('kv_num_heads', 1),
          fAttr('scale', 1),
        ]),
      ],
      inputs: ['Q', 'K', 'V', 'D', 'B'],
      outputs: ['Y', 'S'],
      opset: 27,
    );
    final deltaOut = delta.run(
      {
        'Q': ft([1, 2, 1], [1, 1]),
        'K': ft([1, 2, 1], [1, 1]),
        'V': ft([1, 2, 1], [2, 3]),
        'D': ft([1, 2, 1], [0, 0]),
        'B': ft([1, 2, 1], [0.5, 0.5]),
      },
      ['Y', 'S'],
    );
    expectClose(deltaOut['Y']!.asFloatList(), [1, 2]);
    expectClose(deltaOut['S']!.asFloatList(), [2]);
  });

  test('TfIdfVectorizer counts n-grams with skips', () {
    final attrs = [
      iAttr('max_gram_length', 2),
      iAttr('max_skip_count', 0),
      iAttr('min_gram_length', 2),
      sAttr('mode', 'TF'),
      isAttr('ngram_counts', [0, 4]),
      isAttr('ngram_indexes', [0, 1, 2, 3, 4, 5, 6]),
      isAttr('pool_int64s', [2, 3, 5, 4, 5, 6, 7, 8, 6, 7]),
    ];
    final batch = modelOf([
      node('TfIdfVectorizer', ['X'], ['Y'], attrs),
    ]);
    final y = batch.run(
      {
        'X': it([2, 6], [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]),
      },
      ['Y'],
    )['Y']!;
    expect(y.shape, [2, 7]);
    expectClose(y.asFloatList(), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]);

    final uni = attrs
        .map(
          (a) => a.name == 'min_gram_length'
              ? iAttr('min_gram_length', 1)
              : a.name == 'max_skip_count'
              ? iAttr('max_skip_count', 5)
              : a,
        )
        .toList();
    final single = modelOf([
      node('TfIdfVectorizer', ['X'], ['Y'], uni),
    ]);
    final y2 = single.run(
      {
        'X': it([12], [1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]),
      },
      ['Y'],
    )['Y']!;
    expect(y2.shape, [7]);
    expectClose(y2.asFloatList(), [0, 3, 1, 0, 1, 3, 1]);
  });

  test('SequenceConstruct, SequenceInsert, SequenceErase, SequenceAt and SequenceLength', () {
    final model = modelOf(
      [
        node('SequenceConstruct', ['A', 'B'], ['S']),
        node('SequenceInsert', ['S', 'C'], ['S2']),
        node('SequenceErase', ['S2'], ['S3']),
        node('SequenceAt', ['S3', 'I'], ['Y']),
        node('SequenceLength', ['S3'], ['L']),
      ],
      inputs: ['A', 'B', 'C', 'I'],
      outputs: ['Y', 'L'],
    );
    final out = model.run(
      {
        'A': ft([1], [1]),
        'B': ft([1], [2]),
        'C': ft([1], [3]),
        'I': it([], [-1]),
      },
      ['Y', 'L'],
    );
    expect(out['Y']!.asFloatList(), [2]);
    expect(out['L']!.asIntList(), [2]);
  });

  test('SequenceEmpty and Optional operators handle absent values', () {
    final model = modelOf(
      [
        node('SequenceEmpty', [], ['S']),
        node('SequenceLength', ['S'], ['L0']),
        node('SequenceInsert', ['S', 'A', 'P'], ['S1']),
        node('SequenceLength', ['S1'], ['L1']),
        node('Optional', [], ['O']),
        node('OptionalHasElement', ['O'], ['H0']),
        node('Optional', ['A'], ['O2']),
        node('OptionalGetElement', ['O2'], ['Y']),
        node('OptionalHasElement', ['O2'], ['H1']),
      ],
      inputs: ['A', 'P'],
      outputs: ['L0', 'L1', 'Y', 'H0', 'H1'],
    );
    final out = model.run(
      {
        'A': ft([1], [7]),
        'P': it([], [0]),
      },
      ['L0', 'L1', 'Y', 'H0', 'H1'],
    );
    expect(out['L0']!.asIntList(), [0]);
    expect(out['L1']!.asIntList(), [1]);
    expect(out['Y']!.asFloatList(), [7]);
    expect(out['H0']!.asIntList(), [0]);
    expect(out['H1']!.asIntList(), [1]);
  });

  test('SplitToSequence and ConcatFromSequence preserve axes and stacking', () {
    final split = modelOf(
      [
        node('SplitToSequence', ['X', 'K'], ['S'], [iAttr('axis', 1)]),
        node('ConcatFromSequence', ['S'], ['Y'], [iAttr('axis', 1)]),
      ],
      inputs: ['X', 'K'],
    );
    final splitOut = split.run(
      {
        'X': ft([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
        'K': it([], [2]),
      },
      ['Y'],
    );
    expect(splitOut['Y']!.shape, [2, 4]);
    expect(splitOut['Y']!.asFloatList(), [1, 2, 3, 4, 5, 6, 7, 8]);

    final stack = modelOf([
      node('SplitToSequence', ['X'], ['S'], [iAttr('axis', 0)]),
      node('ConcatFromSequence', ['S'], ['Y'], [
        iAttr('axis', 1),
        iAttr('new_axis', 1),
      ]),
    ]);
    final stackOut = stack.run(
      {
        'X': ft([2, 3], [1, 2, 3, 4, 5, 6]),
      },
      ['Y'],
    );
    expect(stackOut['Y']!.shape, [1, 2, 3]);
    expect(stackOut['Y']!.asFloatList(), [1, 2, 3, 4, 5, 6]);

    final squeeze = modelOf(
      [
        node('SplitToSequence', ['X'], ['S'], [
          iAttr('axis', 0),
          iAttr('keepdims', 0),
        ]),
        node('SequenceAt', ['S', 'I'], ['Y']),
      ],
      inputs: ['X', 'I'],
    );
    final squeezeOut = squeeze.run(
      {
        'X': ft([2, 3], [1, 2, 3, 4, 5, 6]),
        'I': it([], [1]),
      },
      ['Y'],
    );
    expect(squeezeOut['Y']!.shape, [3]);
    expect(squeezeOut['Y']!.asFloatList(), [4, 5, 6]);
  });

  test('SequenceMap applies a body graph to every element', () {
    final body = subgraph(
      [
        node('Add', ['item', 'bias'], ['out']),
      ],
      inputs: ['item', 'bias'],
      outputs: ['out'],
    );
    final model = modelOf(
      [
        node('SequenceMap', ['S', 'Bias'], ['M'], [gAttr('body', body)]),
        node('SequenceAt', ['M', 'I'], ['Y']),
      ],
      inputs: ['S', 'Bias', 'I'],
    );
    final out = model.run(
      {
        'S': SequenceTensor([
          ft([1], [1]),
          ft([1], [2]),
        ]),
        'Bias': ft([1], [10]),
        'I': it([], [1]),
      },
      ['Y'],
    );
    expect(out['Y']!.asFloatList(), [12]);
  });

  test('StringConcat broadcasts elementwise', () {
    final model = modelOf(
      [
        node('StringConcat', ['X', 'Y'], ['Z']),
      ],
      inputs: ['X', 'Y'],
    );
    final out =
        model.run(
              {
                'X': StringTensor(['cat', 'dog', 'snake'], [3]),
                'Y': StringTensor(['s'], [1]),
              },
              ['Z'],
            )['Z']!
            as StringTensor;
    expect(out.shape, [3]);
    expect(out.strings, ['cats', 'dogs', 'snakes']);

    final scalar =
        model.run(
              {
                'X': StringTensor(['cat'], const []),
                'Y': StringTensor(['s'], const []),
              },
              ['Z'],
            )['Z']!
            as StringTensor;
    expect(scalar.shape, isEmpty);
    expect(scalar.strings, ['cats']);
  });

  test('StringSplit supports delimiters, maxsplit and whitespace', () {
    final delimiter = modelOf(
      [
        node('StringSplit', ['X'], ['Y', 'L'], [sAttr('delimiter', '.')]),
      ],
      outputs: ['Y', 'L'],
    );
    final delimiterOut = delimiter.run(
      {
        'X': StringTensor(['abc.com', 'def.net'], [2]),
      },
      ['Y', 'L'],
    );
    final substrings = delimiterOut['Y']! as StringTensor;
    expect(substrings.shape, [2, 2]);
    expect(substrings.strings, ['abc', 'com', 'def', 'net']);
    expect(delimiterOut['L']!.asIntList(), [2, 2]);

    final maxsplit = modelOf(
      [
        node('StringSplit', ['X'], ['Y', 'L'], [iAttr('maxsplit', 2)]),
      ],
      outputs: ['Y', 'L'],
    );
    final maxsplitOut = maxsplit.run(
      {
        'X': StringTensor(
          ['hello world', 'def.net', 'o n n x', 'the quick brown fox'],
          [2, 2],
        ),
      },
      ['Y', 'L'],
    );
    final padded = maxsplitOut['Y']! as StringTensor;
    expect(padded.shape, [2, 2, 3]);
    expect(padded.strings, [
      'hello',
      'world',
      '',
      'def.net',
      '',
      '',
      'o',
      'n',
      'n x',
      'the',
      'quick',
      'brown fox',
    ]);
    expect(maxsplitOut['L']!.asIntList(), [2, 1, 3, 3]);
  });

  test('StringNormalizer removes stopwords and changes case', () {
    final model = modelOf(
      [
        node('StringNormalizer', ['X'], ['Y'], [
          sAttr('case_change_action', 'UPPER'),
          ssAttr('stopwords', ['monday']),
        ]),
      ],
      outputs: ['Y'],
    );
    final out =
        model.run(
              {
                'X': StringTensor(
                  [
                    'Monday',
                    'tuesday',
                    'wednesday',
                    'Monday',
                    'tuesday',
                    'wednesday',
                  ],
                  [1, 6],
                ),
              },
              ['Y'],
            )['Y']!
            as StringTensor;
    expect(out.shape, [1, 4]);
    expect(out.strings, ['TUESDAY', 'WEDNESDAY', 'TUESDAY', 'WEDNESDAY']);

    final accents = modelOf(
      [
        node('StringNormalizer', ['X'], ['Y'], [
          sAttr('case_change_action', 'UPPER'),
        ]),
      ],
      outputs: ['Y'],
    );
    final accentOut =
        accents.run(
              {
                'X': StringTensor(['caf\u00E9'], [1]),
              },
              ['Y'],
            )['Y']!
            as StringTensor;
    expect(accentOut.strings, ['CAFE']);
  });

  test('RegexFullMatch performs full-string matching', () {
    final model = modelOf(
      [
        node('RegexFullMatch', ['X'], ['Y'], [
          sAttr('pattern', r'www\.[\w.-]+\.\bcom\b'),
        ]),
      ],
      outputs: ['Y'],
    );
    final out = model.run(
      {
        'X': StringTensor(
          ['www.google.com', 'www.facebook.com', 'www.bbc.co.uk'],
          [3],
        ),
      },
      ['Y'],
    )['Y']!;
    expect(out.asIntList(), [1, 1, 0]);
  });

  test('String TensorProto initializers decode as UTF-8', () {
    final initializer = TensorProto()
      ..name = 'P'
      ..dataType = 8
      ..dims.addAll([Int64(2)])
      ..stringData.addAll([utf8.encode('a'), utf8.encode('b')]);
    final model = modelOf(
      [
        node('StringConcat', ['P', 'X'], ['Y']),
      ],
      initializers: [initializer],
    );
    final out =
        model.run(
              {
                'X': StringTensor(['1', '2'], [2]),
              },
              ['Y'],
            )['Y']!
            as StringTensor;
    expect(out.strings, ['a1', 'b2']);
  });
}
