import 'dart:io';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_proto.dart';

void main(List<String> args) {
  final meta = ModelProto.fromBuffer(Uint8List.fromList(File(args.first).readAsBytesSync()));
  final g = meta.graph;
  final init = {for (final t in g.initializer) t.name: t.dims.map((d)=>d.toInt()).toList()};
  int winog=0, pw=0, k3=0, k7=0, other=0, grouped=0, strided=0, noPadAttr=0;
  final byKey = <String,int>{};
  for (final n in g.node) {
    if (n.opType != 'Conv') continue;
    int ai(String name, int dflt) {
      for (final a in n.attribute) { if (a.name==name) return a.i.toInt(); }
      return dflt;
    }
    List<int>? ais(String name) {
      for (final a in n.attribute) { if (a.name==name) return a.ints.map((e)=>e.toInt()).toList(); }
      return null;
    }
    final w = init[n.input[1]]!;
    final group = ai('group',1);
    final pads = ais('pads');
    final strides = ais('strides');
    if (group>1) grouped++;
    if (strides != null && strides.any((v)=>v!=1)) strided++;
    if (pads == null) noPadAttr++;
    final kh=w[2], kw=w[3];
    final eligible = group==1 && kh==3 && kw==3 && (strides==null||strides.every((v)=>v==1)) && pads!=null && pads.length==4 && pads.every((v)=>v==1);
    if (eligible) winog++;
    if (kh==1&&kw==1) pw++; else if (kh==3&&kw==3) k3++; else if (kh==7&&kw==7) k7++; else other++;
    final key = 'k=${kh}x$kw g=$group s=${strides} p=${pads}';
    byKey[key] = (byKey[key]??0)+1;
  }
  print('total convs=222 winogradEligible=$winog pointwise=$pw k3=$k3 k7=$k7 other=$other grouped=$grouped strided=$strided noPadsAttr=$noPadAttr');
  final keys = byKey.entries.toList()..sort((a,b)=>b.value.compareTo(a.value));
  for (final e in keys) print('${e.value}\t${e.key}');
}
