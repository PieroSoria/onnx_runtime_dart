import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_proto.dart';

void main(List<String> args) {
  final p = args.first;
  final meta = ModelProto.fromBuffer(Uint8List.fromList(File(p).readAsBytesSync()));
  final g = meta.graph;
  final init = <String, List<int>>{
    for (final t in g.initializer) t.name: t.dims.map((d) => d.toInt()).toList()
  };
  print('== inputs ==');
  for (final vi in g.input) {
    if (init.containsKey(vi.name)) continue;
    final t = vi.type.tensorType;
    print('${vi.name}: elem=${t.elemType} dims=${t.shape.dim.map((d) => d.dimValue.toInt()).toList()}');
  }
  print('== outputs ==');
  for (final o in g.output) {
    final t = o.type.tensorType;
    print('${o.name}: elem=${t.elemType} dims=${t.shape.dim.map((d) => d.dimValue.toInt()).toList()}');
  }
  final counts = <String, int>{};
  final convs = <String>[];
  for (final n in g.node) {
    counts[n.opType] = (counts[n.opType] ?? 0) + 1;
    if (n.opType == 'Conv') {
      final w = init[n.input[1]];
      convs.add('${n.name} W=${w} x=${n.input[0]}');
    }
  }
  print('== op counts ==');
  final sorted = counts.entries.toList()..sort((a, b) => b.value.compareTo(a.value));
  for (final e in sorted.take(30)) {
    print('${e.value}\t${e.key}');
  }
  print('== convs (${convs.length}) ==');
  for (final c in convs.take(40)) {
    print(c);
  }
}
