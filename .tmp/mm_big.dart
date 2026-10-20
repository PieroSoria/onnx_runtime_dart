import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_proto.dart';
void main(List<String> args) {
  final meta = ModelProto.fromBuffer(Uint8List.fromList(File(args.first).readAsBytesSync()));
  final g = meta.graph;
  final init = {for (final t in g.initializer) t.name: t.dims.map((d)=>d.toInt()).toList()};
  int mm=0,mmBig=0,gemm=0,gemmBig=0;
  final hist=<String,int>{};
  for (final n in g.node) {
    if (n.opType!='MatMul' && n.opType!='Gemm') continue;
    final d = n.input.length>1 ? init[n.input[1]] : null;
    if (d==null) continue;
    final sz = d.fold<int>(1,(a,b)=>a*b);
    hist['${n.opType} ${d.join('x')}'] = (hist['${n.opType} ${d.join('x')}']??0)+1;
    if (n.opType=='MatMul'){mm++; if(sz>=65536) mmBig++;}
    else {gemm++; if(sz>=65536) gemmBig++;}
  }
  print('MatMul=$mm big=$mmBig Gemm=$gemm big=$gemmBig');
  print('Pad nodes: ${g.node.where((n)=>n.opType=='Pad').length}');
  for (final e in (hist.entries.toList()..sort((a,b)=>b.value.compareTo(a.value))).take(8)) print('${e.key} x${e.value}');
}
