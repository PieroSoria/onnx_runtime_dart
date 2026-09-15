library;

import 'dart:async';
import 'dart:typed_data';
import 'onnx.pb.dart';
import 'onnx_audit.dart';
import 'onnx_proto_loader.dart';
import 'tensor.dart';
import 'execution_control.dart';
import 'external_gather.dart';

/// Random-access model storage. Browser adapters can implement this with Blob,
/// IndexedDB or OPFS; no dart:io is imported by the portable library.
abstract interface class OnnxDataSource {
  FutureOr<int> length(String location);
  FutureOr<Uint8List> read(String location, int offset, int length);
}

class MemoryOnnxDataSource implements OnnxDataSource {
  final Map<String, Uint8List> files;
  MemoryOnnxDataSource(Map<String, Uint8List> files)
      : files = Map.unmodifiable(files);
  Uint8List _file(String location) =>
      files[location] ?? (throw ArgumentError('Missing model data: $location'));
  @override
  int length(String location) => _file(location).length;
  @override
  Uint8List read(String location, int offset, int length) {
    final bytes = _file(location);
    if (offset < 0 || length < 0 || offset > bytes.length - length)
      throw FormatException('Invalid range for $location');
    return Uint8List.sublistView(bytes, offset, offset + length);
  }
}

class LoadedOnnxSource {
  final ModelProto model;
  final Map<TensorProto, Tensor> tensors;
  final Map<String, ExternalGatherTable> deferred;
  LoadedOnnxSource(this.model, this.tensors, this.deferred);
}

Iterable<TensorProto> _tensors(GraphProto graph) sync* {
  yield* graph.initializer;
  for (final node in graph.node) {
    for (final attr in node.attribute) {
      if (attr.hasT()) yield attr.t;
      yield* attr.tensors;
      if (attr.hasG()) yield* _tensors(attr.g);
      for (final sub in attr.graphs) {
        yield* _tensors(sub);
      }
    }
  }
}

/// Reads and decodes one tensor at a time after auditing the complete graph.
Future<LoadedOnnxSource> readOnnxSource(OnnxDataSource source, String location,
    OnnxExecutionControl? control) async {
  control?.checkpoint();
  final bytes = await source.read(location, 0, await source.length(location));
  control?.checkpoint();
  final ModelProto model;
  try {
    model = ModelProto.fromBuffer(bytes);
  } catch (e) {
    throw FormatException('Malformed ONNX model (protobuf decode): $e');
  }
  final report = auditOnnxModel(model);
  report.throwIfIncompatible();
  control?.account(const []);
  control?.checkAdditionalBytes(report.initializerBytes - report.deferredInitializerBytes);
  final names = externalGatherNames(model.graph);
  final deferred = <String, ExternalGatherTable>{
    for (final t in model.graph.initializer.where((t) => names.contains(t.name)))
      t.name: ExternalGatherTable(t, source.read),
  };
  final tensors = _tensors(model.graph).toList();
  final decoded = Map<TensorProto, Tensor>.identity();
  control?.progress('load', 0, tensors.length);
  for (final t in tensors) {
    if (model.graph.initializer.contains(t) && names.contains(t.name)) continue;
    control?.checkpoint();
    Uint8List? raw;
    if (t.dataLocation == TensorProto_DataLocation.EXTERNAL) {
      final fields = {for (final e in t.externalData) e.key: e.value};
      final ref = fields['location']!;
      final offset = int.tryParse(fields['offset'] ?? '') ?? 0;
      const bits = {
        1: 32,
        2: 8,
        3: 8,
        6: 32,
        7: 64,
        9: 8,
        10: 16,
        11: 64,
        21: 4,
        22: 4
      };
      final elements = t.dims.fold(1, (int a, b) => a * b.toInt());
      final declared = int.tryParse(fields['length'] ?? '') ?? 0;
      final count =
          declared > 0 ? declared : (elements * bits[t.dataType]! + 7) ~/ 8;
      final available = await source.length(ref);
      if (offset < 0 || count < 0 || offset > available - count)
        throw FormatException('External tensor ${t.name} exceeds $ref');
      final decodedSize = elements *
          (const {
            1: 4,
            2: 1,
            3: 1,
            6: 8,
            7: 8,
            9: 8,
            10: 4,
            11: 4,
            21: 1,
            22: 1
          }[t.dataType]!);
      control?.checkAdditionalBytes(count + decodedSize);
      raw = await source.read(ref, offset, count);
      if (raw.length != count)
        throw FormatException('Short read for external tensor ${t.name}');
      control?.checkpoint();
    }
    decoded[t] =
        tensorFromProto(t, ext: raw == null ? null : (_, __, ___) => raw!);
    control?.account(decoded.values);
    control?.progress('load', decoded.length, tensors.length);
  }
  return LoadedOnnxSource(model, decoded, deferred);
}
