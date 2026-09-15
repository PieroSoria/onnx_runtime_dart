import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:fixnum/fixnum.dart';
import 'package:onnx_runtime_dart/onnx_proto.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/src/onnx_standard_attention.dart';
import 'package:test/test.dart';

ModelProto model(List<NodeProto> nodes,
        {int opset = 23,
        List<TensorProto> weights = const [],
        List<String> inputs = const ['X'],
        List<String> outputs = const ['Y']}) =>
    ModelProto(
        opsetImport: [OperatorSetIdProto(domain: '', version: Int64(opset))],
        graph: GraphProto(
            node: nodes,
            initializer: weights,
            input: inputs.map((s) => ValueInfoProto(name: s)),
            output: outputs.map((s) => ValueInfoProto(name: s))));
NodeProto node(String op, List<String> ins, List<String> outs,
        {String domain = ''}) =>
    NodeProto(opType: op, input: ins, output: outs, domain: domain);
Tensor f(List<int> shape, List<double> values) =>
    Tensor.float(Float32List.fromList(values), shape);
TensorProto externalWeight() => TensorProto(
    name: 'W',
    dims: [Int64(100)],
    dataType: 1,
    dataLocation: TensorProto_DataLocation.EXTERNAL,
    externalData: [
      StringStringEntryProto(key: 'location', value: 'weights.bin')
    ]);

void main() {
  test('memory accounting deduplicates shared reshape buffers', () {
    final x = f([4], [1,2,3,4]);
    final control = OnnxExecutionControl(maxTensorBytes:16);
    control.account([x, x.reshape([2,2]), x]);
    expect(control.peakAccountedBytes, 16);
  });
  test('async source rejects incompatibility before requesting external files', () async {
    final p = model([node('Unknown',['W'],['Y'])],weights:[externalWeight()]);
    final source = MemoryOnnxDataSource({'model.onnx':p.writeToBuffer()});
    await expectLater(OnnxModel.fromSource(source,'model.onnx'), throwsUnsupportedError);
  });
  test('raw int64 constants and shape arithmetic work on browser and VM', () {
    final raw = Uint8List(16);
    ByteData.sublistView(raw)
      ..setUint32(0, 7, Endian.little)
      ..setUint32(8, 0xfffffffe, Endian.little)
      ..setInt32(12, -1, Endian.little);
    final w =
        TensorProto(name: 'W', dataType: 7, dims: [Int64(2)], rawData: raw);
    final p = model([
      node('Add', ['W', 'W'], ['Y'])
    ], weights: [
      w
    ], inputs: []);
    expect(
        OnnxModel.fromBytes(p.writeToBuffer()).run({}, ['Y'])['Y']!.asIntList(),
        [14, -4]);
  });
  test('RMS uses float32 intermediate overflow semantics', () {
    final out =
        standardRMSNormalization(f([2], [1e30, 1e30]), Tensor.scalarFloat(1));
    expect(out.asFloatList(), [0, 0]);
  });
  test('audit rejects ignored training and blocked quantization attributes',
      () {
    final n = node('BatchNormalization', ['X', 'X', 'X', 'X', 'X'], ['Y'])
      ..attribute.add(AttributeProto(name: 'training_mode', i: Int64(1)));
    expect(auditOnnxModel(model([n])).hasErrors, isTrue);
    final q = node('DequantizeLinear', ['X', 'X'], ['Y'])
      ..attribute.add(AttributeProto(name: 'block_size', i: Int64(32)));
    expect(auditOnnxModel(model([q])).hasErrors, isTrue);
  });
  test('portable async source loads external constants and weights', () async {
    final w = externalWeight()..dims.clear();
    w.dims.add(Int64(2));
    final p = model([
      node('Add', ['X', 'W'], ['Y'])
    ], weights: [
      w
    ]);
    final raw = Uint8List(8);
    ByteData.sublistView(raw)
      ..setFloat32(0, 2, Endian.little)
      ..setFloat32(4, 3, Endian.little);
    final source = MemoryOnnxDataSource(
        {'model.onnx': p.writeToBuffer(), 'weights.bin': raw});
    final m = await OnnxModel.fromSource(source, 'model.onnx');
    expect(
        m.run({
          'X': f([2], [10, 20])
        }, [
          'Y'
        ])['Y']!.asFloatList(),
        [12, 23]);
    final constant = node('Constant', [], ['C'])
      ..attribute.add(AttributeProto(name: 'value', t: w));
    final c = model([
      constant,
      node('Add', ['X', 'C'], ['Y'])
    ]);
    final cm = await OnnxModel.fromSource(
        MemoryOnnxDataSource(
            {'model.onnx': c.writeToBuffer(), 'weights.bin': raw}),
        'model.onnx');
    expect(
        cm.run({
          'X': f([2], [10, 20])
        }, [
          'Y'
        ])['Y']!.asFloatList(),
        [12, 23]);
  });
  test('audit aggregates errors before external data is touched', () {
    final n = node('RMSNormalization', ['X', 'W'], ['Y'])
      ..attribute.add(AttributeProto(
          name: 'stash_type',
          i: Int64(10),
          type: AttributeProto_AttributeType.INT));
    final proto = model([
      n,
      node('Unknown', ['X'], ['Z']),
      node('Add', ['X', 'X'], ['C'], domain: 'custom')
    ], weights: [
      externalWeight()
    ]);
    final report = auditOnnxModel(proto);
    expect(report.hasErrors, isTrue);
    expect(
        report.issues.where((i) => i.isError).length, greaterThanOrEqualTo(3));
    expect(report.initializerBytes, 400);
    var reads = 0;
    expect(
        () => OnnxModel.fromBytes(proto.writeToBuffer(),
                externalData: (_, __, ___) {
              reads++;
              return Uint8List(400);
            }),
        throwsUnsupportedError);
    expect(reads, 0);
  });
  test('domain collisions and wrong opsets are rejected', () {
    final p = model([
      node('Attention', ['X', 'X', 'X'], ['Y'], domain: 'com.microsoft')
    ]);
    p.opsetImport
        .add(OperatorSetIdProto(domain: 'com.microsoft', version: Int64(1)));
    expect(auditOnnxModel(p).hasErrors, isTrue);
    expect(
        auditOnnxModel(model([
          node('RMSNormalization', ['X', 'X'], ['Y'])
        ], opset: 22))
            .hasErrors,
        isTrue);
    // Without a declared input shape the audit can only flag data-dependent
    // conditions as warnings; rank-3 Attention without head-count attributes is
    // statically incompatible even when the input size is unknown.
    final rank3 = NodeProto(opType: 'Attention', input: ['X', 'X', 'X'], output: ['Y']);
    final p3 = ModelProto(
        opsetImport: [OperatorSetIdProto(domain: '', version: Int64(24))],
        graph: GraphProto(
            node: [rank3],
            input: [
              ValueInfoProto(name: 'X', type: TypeProto(tensorType: TypeProto_Tensor(elemType: 1, shape: TensorShapeProto(dim: [TensorShapeProto_Dimension(dimValue: Int64(3)), TensorShapeProto_Dimension(dimValue: Int64(7)), TensorShapeProto_Dimension(dimValue: Int64(8))]))))
            ],
            output: [ValueInfoProto(name: 'Y')]));
    expect(auditOnnxModel(p3).hasErrors, isTrue);
    // Unknown shapes degrade to execution-time validation, not a false error.
    final untyped = auditOnnxModel(model([
      node('Attention', ['X', 'X', 'X'], ['Y'])
    ], opset: 24));
    expect(untyped.hasErrors, isFalse);
    expect(
        untyped.issues.where((i) =>
            i.message.contains('validated at execution')).length,
        greaterThanOrEqualTo(3));
  });
  test('audits attributes, tensor types and nested graphs', () {
    final bad = node('Add', ['X', 'X'], ['Z'])
      ..attribute.add(AttributeProto(name: 'made_up', i: Int64(1)));
    final branch = GraphProto(node: [bad], output: [ValueInfoProto(name: 'Z')]);
    final n = node('If', ['X'], ['Y'])
      ..attribute.addAll([
        AttributeProto(name: 'then_branch', g: branch),
        AttributeProto(name: 'else_branch', g: branch)
      ]);
    final w = externalWeight()..dataType = 16;
    final report = auditOnnxModel(model([n], weights: [w]));
    expect(
        report.issues.where((i) => i.isError).length, greaterThanOrEqualTo(3));
  });
  test('fully masked attention rows are zero and inputs are unchanged', () {
    final x = f([1, 1, 2, 2], [1, 2, 3, 4]);
    final mask = Tensor.int64(List<int>.filled(4, 0), [2, 2]);
    final out =
        standardAttention(x, x, x, mask: mask, debug: true, debugMode: 3);
    expect(out[0].asFloatList(), [0, 0, 0, 0]);
    expect(out[3].asFloatList(), [0, 0, 0, 0]);
    expect(x.asFloatList(), [1, 2, 3, 4]);
  });
  test('incremental attention equals the last row of full causal attention',
      () {
    final x = f([1, 1, 3, 2], [1, 0, 0, 1, 1, 1]);
    final first = standardAttention(f([1, 1, 2, 2], [1, 0, 0, 1]),
        f([1, 1, 2, 2], [1, 0, 0, 1]), f([1, 1, 2, 2], [1, 0, 0, 1]),
        causal: true, present: true);
    final last = f([1, 1, 1, 2], [1, 1]);
    final step = standardAttention(last, last, last,
        pastKey: first[1], pastValue: first[2], causal: true, present: true);
    final full = standardAttention(x, x, x, causal: true);
    for (var i = 0; i < 2; i++) {
      expect(step[0].getD(i), closeTo(full[0].getD(4 + i), 1e-6));
    }
    expect(first[1].shape, [1, 1, 2, 2]);
    expect(step[1].shape, [1, 1, 3, 2]);
  });
  test('persistent KV cache matches copy path and reuses buffers', () {
    // batch=1, nk=1, head=2; first query two tokens, then one.
    final k0 = Float32List(1 * 1 * 2 * 2), v0 = Float32List(1 * 1 * 2 * 2);
    final first = standardAttention(f([1, 1, 2, 2], [1, 0, 0, 1]),
        f([1, 1, 2, 2], [1, 0, 0, 1]), f([1, 1, 2, 2], [1, 0, 0, 1]),
        causal: true, present: true, cacheK: k0, cacheV: v0, cacheRows: 0);
    // Capacity 2 was enough: the returned present views alias the cache.
    expect(identical(first[1].f, k0), isTrue);
    expect(first[1].shape, [1, 1, 2, 2]);
    expect(first[1].cacheView, isTrue);
    // Third token forces a grow; history must be carried over, views must
    // reference the new buffer, and the result must match a fresh full run.
    final last = f([1, 1, 1, 2], [1, 1]);
    final step = standardAttention(last, last, last,
        causal: true, present: true, cacheK: k0, cacheV: v0, cacheRows: 2);
    expect(identical(step[1].f, k0), isFalse, reason: 'cache should grow');
    expect(step[1].shape, [1, 1, 3, 2]);
    expect(step[1].cacheView, isTrue);
    final full = standardAttention(
        f([1, 1, 3, 2], [1, 0, 0, 1, 1, 1]), f([1, 1, 3, 2], [1, 0, 0, 1, 1, 1]),
        f([1, 1, 3, 2], [1, 0, 0, 1, 1, 1]),
        causal: true, present: true);
    for (var i = 0; i < 3; i++) {
      expect(step[1].getD(i), closeTo(full[1].getD(i), 0));
    }
    // The stepped output covers only the new token: matches the last row.
    expect(step[0].getD(0), closeTo(full[0].getD(4), 1e-6));
    expect(step[0].getD(1), closeTo(full[0].getD(5), 1e-6));
    // Resident rows 0..1 survive the grow unchanged.
    expect(step[1].getD(0), closeTo(first[1].getD(0), 0));
    expect(step[1].getD(1), closeTo(first[1].getD(1), 0));
  });
  test('executor persistent KV cache appends across decode runs', () {
    final attn = NodeProto(
        opType: 'Attention',
        input: ['X', 'X', 'X', '', 'PK', 'PK'],
        output: ['Y', 'PRK', 'PRK'])
      ..attribute.add(AttributeProto(name: 'q_num_heads', i: Int64(1)))
      ..attribute.add(AttributeProto(name: 'kv_num_heads', i: Int64(1)))
      ..attribute.add(AttributeProto(name: 'is_causal', i: Int64(1)));
    final p = model([attn], inputs: ['X', 'PK'], outputs: ['Y', 'PRK']);
    final withCache = OnnxModel.fromBytes(p.writeToBuffer());
    withCache.enablePersistentKv();
    final without = OnnxModel.fromBytes(p.writeToBuffer());
    Tensor? cachedKeys;
    Tensor? cachedY;
    Tensor? copyKeys;
    Tensor? copyY;
    for (var step = 0; step < 3; step++) {
      final qs = step == 0 ? 2 : 1;
      final x = f([1, 1, qs, 2], [
        for (var i = 0; i < qs; i++) ...[i == 0 ? 1.0 : 0.0, i == 0 ? 0.0 : 1.0]
      ]);
      final pk = step == 0
          ? Tensor.float(Float32List(0), [1, 1, 0, 2])
          : cachedKeys;
      final ck = step == 0
          ? Tensor.float(Float32List(0), [1, 1, 0, 2])
          : copyKeys;
      final co = withCache.run({'X': x, 'PK': pk!}, ['Y', 'PRK']);
      final no = without.run({'X': x, 'PK': ck!}, ['Y', 'PRK']);
      cachedKeys = co['PRK']!;
      copyKeys = no['PRK']!;
      cachedY = co['Y']!;
      copyY = no['Y']!;
      for (var i = 0; i < cachedY.length; i++) {
        expect(cachedY.getD(i), closeTo(copyY.getD(i), 1e-6));
      }
    }
    // The final cache holds the whole 4-token history as one buffer.
    expect(cachedKeys!.shape, [1, 1, 4, 2]);
  });
  test('Attention-25 sliding windows mask rows like the spec', () {
    // right_window_size=0 (causal=0): a query attends keys up to and including
    // its own position. Constant K/V rows make any attended subset yield (1,1),
    // and an entirely missing row would come out zero.
    final x = f([1, 1, 4, 2], List.filled(8, 1.0));
    final y = standardAttention(x, x, x,
        qHeads: 1, kvHeads: 1, rightWindow: 0)[0];
    for (var i = 0; i < 4; i++) {
      expect(y.getD(i * 2 + 0), closeTo(1.0, 0));
      expect(y.getD(i * 2 + 1), closeTo(1.0, 0));
    }
    final rows = [
      for (var i = 0; i < 4; i++) [1.0 + i, 2.0 + i]
    ];
    // left=2,right=1, causal=0: rows attend {0,1},{0,1,2},{0,1,2,3},{1,2,3}.
    final allowed = [
      [0, 1],
      [0, 1, 2],
      [0, 1, 2, 3],
      [1, 2, 3],
    ];
    // Query is the constant direction [1,0] so score(j) = rows[j][0]*factor.
    final q2 = f([1, 1, 4, 2],
        [for (var i = 0; i < 4; i++) ...[1.0, 0.0]]);
    final x2 = f([1, 1, 4, 2], [for (final r in rows) ...r]);
    final y2 = standardAttention(q2, x2, x2,
        qHeads: 1, kvHeads: 1, leftWindow: 2, rightWindow: 1)[0];
    final factor = 1 / math.sqrt(2.0);
    for (var i = 0; i < 4; i++) {
      final weights = allowed[i]
          .map<double>((j) => math.exp(rows[j][0] * factor))
          .toList();
      final s = weights.reduce((a, b) => a + b);
      var exp0 = 0.0, exp1 = 0.0;
      for (var t = 0; t < allowed[i].length; t++) {
        final w = weights[t] / s;
        exp0 += w * rows[allowed[i][t]][0];
        exp1 += w * rows[allowed[i][t]][1];
      }
      expect(y2.getD(i * 2 + 0), closeTo(exp0, 1e-5));
      expect(y2.getD(i * 2 + 1), closeTo(exp1, 1e-5));
    }
  });
  test('rotary checks cache shape and position bounds', () {
    final x = f([1, 1, 1, 4], [1, 2, 3, 4]);
    expect(
        () => standardRotaryEmbedding(
            x, f([1, 2], [1, 1]), f([1, 2], [0, 0]), Tensor.scalarInt(0)),
        throwsArgumentError);
    expect(
        () => standardRotaryEmbedding(
            x, f([1, 2], [1, 1]), f([1, 2], [0, 0]), Tensor.int64([1], [1, 1])),
        throwsArgumentError);
  });
  test('RMS suffix normalization supports scalar scale and rejects expansion',
      () {
    final x = f([1, 2, 2], [1, 1, 1, 1]);
    final y =
        standardRMSNormalization(x, Tensor.scalarFloat(2), axis: 1, epsilon: 0);
    expect(y.asFloatList(), [2, 2, 2, 2]);
    expect(() => standardRMSNormalization(x, f([3, 2, 2], List.filled(12, 1))),
        throwsArgumentError);
  });
  test('load budget and cancellation run before external reads', () {
    final p = model([
      node('Identity', ['W'], ['Y'])
    ], weights: [
      externalWeight()
    ]);
    var reads = 0;
    Uint8List resolve(String _, int __, int ___) {
      reads++;
      return Uint8List(400);
    }

    expect(
        () => OnnxModel.fromBytes(p.writeToBuffer(),
            externalData: resolve,
            control: OnnxExecutionControl(maxTensorBytes: 100)),
        throwsA(isA<OnnxMemoryLimitException>()));
    final control = OnnxExecutionControl()..cancel();
    expect(
        () => OnnxModel.fromBytes(p.writeToBuffer(),
            externalData: resolve, control: control),
        throwsA(isA<OnnxCancelledException>()));
    expect(reads, 0);
  });
  test('intermediates are released, requested debug values retained', () {
    final p = model([
      node('Neg', ['X'], ['A']),
      node('Neg', ['A'], ['B']),
      node('Neg', ['B'], ['Y'])
    ]);
    final m = OnnxModel.fromBytes(p.writeToBuffer(), fuse: false);
    final c = OnnxExecutionControl(maxTensorBytes: 48);
    final x = f([4], [1, 2, 3, 4]);
    expect(m.run({'X': x}, ['Y'], control: c)['Y']!.asFloatList(),
        [-1, -2, -3, -4]);
    expect(c.peakAccountedBytes, lessThanOrEqualTo(48));
    expect(m.run({'X': x}, ['*']).keys, containsAll(['X', 'A', 'B', 'Y']));
    expect(m.run({'X': x}, ['A', 'Y']).keys, containsAll(['A', 'Y']));
  });
  test('async cancellation can arrive from an event and preserves exception',
      () async {
    final m = OnnxModel.fromBytes(model([
      node('Neg', ['X'], ['Y'])
    ]).writeToBuffer());
    final c = OnnxExecutionControl();
    Timer.run(c.cancel);
    await expectLater(
        m.runAsync({
          'X': f([1], [1])
        }, [
          'Y'
        ], control: c),
        throwsA(isA<OnnxCancelledException>()));
    expect(
        m.run({
          'X': f([1], [1])
        }, [
          'Y'
        ])['Y']!.getD(0),
        -1);
  });
  test('softmax honors pre-13 flattened suffix semantics', () {
    final p = model([
      node('Softmax', ['X'], ['Y'])
    ], opset: 11);
    final m = OnnxModel.fromBytes(p.writeToBuffer());
    expect(
        m.run({
          'X': f([1, 2, 2], [0, 0, 0, 0])
        }, [
          'Y'
        ])['Y']!.asFloatList(),
        [.25, .25, .25, .25]);
  });
}
