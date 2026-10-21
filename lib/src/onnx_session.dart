/// Backend selection for ONNX models.
///
/// This facade is web-safe: on the web the conditional implementation never
/// imports `dart:ffi` or `dart:io`, and [OnnxBackend.ort] is rejected.
library;

import 'dart:typed_data';

import 'execution_control.dart';
import 'onnx_graph.dart' show OnnxExperiment;
import 'onnx_proto_loader.dart';
import 'onnx_runner.dart';
import 'onnx_session_stub.dart'
    if (dart.library.io) 'onnx_session_native.dart'
    as impl;
import 'ort/ort_options.dart';

/// Loads a model with the pure-Dart interpreter, native ONNX Runtime, or an
/// automatic choice.
final class OnnxSession {
  const OnnxSession._();

  /// Whether this platform can load the native backend at all. On native
  /// targets this is true even if no ONNX Runtime library is installed; use
  /// [OnnxBackend.ort] to require it.
  static bool get isOrtSupported => impl.isOrtSupported;

  /// Loads [path]. On the web use [fromBytes] instead.
  static Future<OnnxRunner> fromFile(
    String path, {
    OnnxBackend backend = OnnxBackend.auto,
    OrtOptions ort = const OrtOptions(),
    bool fuse = true,
    bool lastTokenLogits = false,
    Set<OnnxExperiment> experiments = const {},
    OnnxExecutionControl? control,
  }) => impl.fromFile(
    path,
    backend: backend,
    ort: ort,
    fuse: fuse,
    lastTokenLogits: lastTokenLogits,
    experiments: experiments,
    control: control,
  );

  /// Loads a serialized model. External tensors use [externalData] when
  /// provided; the native backend currently requires a self-contained model
  /// for this entry point (use [fromFile] for external-data models).
  static Future<OnnxRunner> fromBytes(
    Uint8List bytes, {
    OnnxBackend backend = OnnxBackend.auto,
    OrtOptions ort = const OrtOptions(),
    ExternalDataResolver? externalData,
    bool fuse = true,
    bool lastTokenLogits = false,
    Set<OnnxExperiment> experiments = const {},
    OnnxExecutionControl? control,
  }) => impl.fromBytes(
    bytes,
    backend: backend,
    ort: ort,
    externalData: externalData,
    fuse: fuse,
    lastTokenLogits: lastTokenLogits,
    experiments: experiments,
    control: control,
  );
}
