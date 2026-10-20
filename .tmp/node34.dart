import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_proto.dart';
void main(List<String> args) {
  final meta = ModelProto.fromBuffer(Uint8List.fromList(File(args.first).readAsBytesSync()));
  final g = meta.graph;
  final init = {for (final t in g.initializer) t.name: t.dims.map((d)=>d.toInt()).toList()};
  for (final n in g.node) {
    if (n.name.contains('model.34') || n.name.contains('model.33') || n.name.contains('model.32')) {
      final w = n.input.length > 1 ? init[n.input[1]] : null;
      int ai(String name, int dflt) { for (final a in n.attribute) { if (a.name==name) return a.i.toInt(); } return dflt; }
      List<int>? ais(String name) { for (final a in n.attribute) { if (a.name==name) return a.ints.map((e)=>e.toInt()).toList(); } return null; }
      print('${n.opType} ${n.name} W=$w group=${ai('group',1)} s=${ais('strides')} p=${ais('pads')} x=${n.input.isNotEmpty?n.input[0]:""}');
    }
  }
}
