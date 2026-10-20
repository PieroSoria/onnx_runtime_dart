import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_proto.dart';
void main(List<String> args) {
  final meta = ModelProto.fromBuffer(Uint8List.fromList(File(args.first).readAsBytesSync()));
  final g = meta.graph;
  final init = {for (final t in g.initializer) t.name: t.dims.map((d)=>d.toInt()).toList()};
  var i = 0;
  for (final n in g.node) {
    if (n.opType != 'MatMul') continue;
    if (i++ >= 20) break;
    print('${n.name} B=${n.input.length>1?init[n.input[1]]:null} A=${n.input[0]}');
  }
  print('total MatMul=${i}');
}
