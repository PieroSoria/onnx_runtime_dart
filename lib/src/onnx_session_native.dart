/// Native implementation of [OnnxSession] support.
library;

import 'dart:io';
import 'dart:typed_data';

import 'execution_control.dart';
import 'onnx_file_loader_io.dart';
import 'onnx_graph.dart' show OnnxExperiment;
import 'onnx_model.dart';
import 'onnx_proto_loader.dart';
import 'onnx_runner.dart';
import 'ort/ort_backend.dart';
import 'ort/ort_options.dart';

bool get isOrtSupported => true;

Future<OnnxRunner> fromFile(
  String path, {
  OnnxBackend backend = OnnxBackend.auto,
  OrtOptions ort = const OrtOptions(),
  bool fuse = true,
  bool lastTokenLogits = false,
  Set<OnnxExperiment> experiments = const {},
  OnnxExecutionControl? control,
}) async {
  if (backend == OnnxBackend.dart) {
    return loadOnnxModel(
      path,
      fuse: fuse,
      lastTokenLogits: lastTokenLogits,
      control: control,
      experiments: experiments,
    );
  }
  try {
    return await OrtRunner.fromFile(path, options: ort);
  } catch (e) {
    if (backend == OnnxBackend.ort || !ort.fallbackToDart) rethrow;
    stderr.writeln(
      'onnx_runtime_dart: ONNX Runtime unavailable ($e); using Dart backend',
    );
    return loadOnnxModel(
      path,
      fuse: fuse,
      lastTokenLogits: lastTokenLogits,
      control: control,
      experiments: experiments,
    );
  }
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
  if (backend != OnnxBackend.dart) {
    try {
      return await OrtRunner.fromBytes(bytes, options: ort);
    } catch (e) {
      if (backend == OnnxBackend.ort || !ort.fallbackToDart) rethrow;
      stderr.writeln(
        'onnx_runtime_dart: ONNX Runtime unavailable ($e); using Dart backend',
      );
    }
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
