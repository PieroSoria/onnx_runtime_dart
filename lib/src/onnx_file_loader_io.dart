/// Native file loading for ONNX models, including external-data companions.
library;

import 'dart:io';
import 'dart:typed_data';

import 'execution_control.dart';
import 'external_data_guard.dart';
import 'onnx_graph.dart' show OnnxExperiment;
import 'onnx_model.dart';

/// Loads an ONNX model from [path], resolving any external-data weights from
/// the companion file(s) named in the model (relative to [path]'s directory).
///
/// External weights are read on demand with random access, so a model with a
/// multi-gigabyte `.onnx.data` file is not loaded into memory all at once.
OnnxModel loadOnnxModel(
  String path, {
  bool fuse = true,
  bool lastTokenLogits = false,
  OnnxExecutionControl? control,
  Set<OnnxExperiment> experiments = const {},
}) {
  final file = File(path);
  final dir = file.parent.path;
  Uint8List resolve(String location, int offset, int length) {
    checkExternalRef(location, 0, 0, 0);
    final raf = File('$dir/$location').openSync();
    try {
      checkExternalRef(location, offset, length, raf.lengthSync());
      raf.setPositionSync(offset);
      return raf.readSync(length);
    } finally {
      raf.closeSync();
    }
  }

  return OnnxModel.fromBytes(
    file.readAsBytesSync(),
    control: control,
    externalData: resolve,
    fuse: fuse,
    lastTokenLogits: lastTokenLogits,
    experiments: experiments,
  );
}
