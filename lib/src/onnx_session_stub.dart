/// Web implementation of [OnnxSession] support.
library;

import 'dart:typed_data';

import 'execution_control.dart';
import 'onnx_graph.dart' show OnnxExperiment;
import 'onnx_model.dart';
import 'onnx_proto_loader.dart';
import 'onnx_runner.dart';
import 'ort/ort_options.dart';

bool get isOrtSupported => false;

Future<OnnxRunner> fromFile(
  String path, {
  OnnxBackend backend = OnnxBackend.auto,
  OrtOptions ort = const OrtOptions(),
  bool fuse = true,
  bool lastTokenLogits = false,
  Set<OnnxExperiment> experiments = const {},
  OnnxExecutionControl? control,
}) async {
  throw UnsupportedError(
    'OnnxSession.fromFile needs dart:io; use OnnxSession.fromBytes on web',
  );
}

Future<OnnxRunner> fromBytes(
  Uint8List bytes, {
  OnnxBackend backend = OnnxBackend.auto,
  OrtOptions ort = const OrtOptions(),
  ExternalDataResolver? externalData,
  bool fuse = true,
  bool lastTokenLogits = false,
  Set<OnnxExperiment> experiments = const {},
  OnnxExecutionControl? control,
}) async {
  if (backend == OnnxBackend.ort) {
    throw UnsupportedError(
      'The native ONNX Runtime backend is unavailable on the web',
    );
  }
  return OnnxModel.fromBytes(
    bytes,
    control: control,
    externalData: externalData,
    fuse: fuse,
    lastTokenLogits: lastTokenLogits,
    experiments: experiments,
  );
}
