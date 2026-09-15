/// Row-wise access to external embedding/rotary tables used only by Gather-0.
library;

import 'dart:async';
import 'dart:typed_data';
import 'package:fixnum/fixnum.dart';
import 'onnx.pb.dart';
import 'onnx_proto_loader.dart';
import 'tensor.dart';
import 'execution_control.dart';

typedef OnnxRangeReader = FutureOr<Uint8List> Function(String location, int offset, int length);

/// Only tables whose every use is a standard Gather along axis zero qualify.
/// Graph inputs/outputs and captures with other uses never become deferred.
Set<String> externalGatherNames(GraphProto graph) {
  final candidates = {for (final t in graph.initializer)
    if (t.dataLocation == TensorProto_DataLocation.EXTERNAL && t.dims.isNotEmpty && const {1,10}.contains(t.dataType)) t.name};
  candidates.removeAll([...graph.input, ...graph.output].map((v) => v.name));
  final seen = <String>{};
  void scan(GraphProto g, {bool nested = false}) {
    for (final n in g.node) {
      final axis = n.attribute.where((a) => a.name == 'axis').map((a) => a.i.toInt()).firstOrNull ?? 0;
      for (var i = 0; i < n.input.length; i++) {
        final name = n.input[i];
        if (!candidates.contains(name)) continue;
        if (!nested && i == 0 && n.opType == 'Gather' && (n.domain.isEmpty || n.domain == 'ai.onnx') && axis == 0) { seen.add(name); }
        else { candidates.remove(name); }
      }
      for (final a in n.attribute) {
        if (a.hasG()) scan(a.g, nested: true);
        for (final sub in a.graphs) { scan(sub, nested: true); }
      }
    }
  }
  scan(graph);
  return candidates.intersection(seen);
}

class ExternalGatherTable {
  final TensorProto proto;
  final OnnxRangeReader reader;
  ExternalGatherTable(this.proto, this.reader);
  List<int> get shape => proto.dims.map((d) => d.toInt()).toList();
  int get rowElements => shape.skip(1).fold(1, (int a, int b) => a*b);
  int get elementBytes => proto.dataType == 10 ? 2 : 4;
  (String,int,int) _range(int index) {
    final dims = shape;
    final row = index < 0 ? index + dims[0] : index;
    if (row < 0 || row >= dims[0]) throw RangeError('Gather index $index outside ${dims[0]} rows');
    final fields = {for (final e in proto.externalData) e.key:e.value};
    return (fields['location']!, (int.tryParse(fields['offset'] ?? '') ?? 0) + row * rowElements * elementBytes, rowElements * elementBytes);
  }
  Tensor _decode(Uint8List bytes) => tensorFromProto(TensorProto(
      dataType:proto.dataType, dims:shape.skip(1).map(Int64.new), rawData:bytes));
  Tensor gather(Tensor indices) {
    final width = rowElements;
    activeOnnxControl?.checkAdditionalBytes(indices.length * width * 4 + width * (4 + elementBytes));
    final out = Float32List(indices.length * width);
    for (var i = 0; i < indices.length; i++) {
      activeOnnxControl?.checkpoint();
      final (location, offset, length) = _range(indices.getI(i));
      final bytes = reader(location,offset,length);
      if (bytes is! Uint8List) throw UnsupportedError('This model source requires runAsync for external Gather');
      if (bytes.length != length) throw FormatException('Short external Gather read');
      out.setRange(i*width,(i+1)*width,_decode(bytes).asFloatList());
    }
    return Tensor.float(out,[...indices.shape,...shape.skip(1)]);
  }
  Future<Tensor> gatherAsync(Tensor indices) async {
    final width = rowElements;
    activeOnnxControl?.checkAdditionalBytes(indices.length * width * 4 + width * (4 + elementBytes));
    final out = Float32List(indices.length * width);
    for (var i = 0; i < indices.length; i++) {
      activeOnnxControl?.checkpoint();
      final (location, offset, length) = _range(indices.getI(i));
      final bytes = await reader(location,offset,length);
      activeOnnxControl?.checkpoint();
      if (bytes.length != length) throw FormatException('Short external Gather read');
      out.setRange(i*width,(i+1)*width,_decode(bytes).asFloatList());
    }
    return Tensor.float(out,[...indices.shape,...shape.skip(1)]);
  }
}
