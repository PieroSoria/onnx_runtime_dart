/// Backend-neutral execution interface shared by the pure-Dart interpreter and
/// the optional native ONNX Runtime backend.
library;

import 'execution_control.dart';
import 'onnx_graph.dart' show ExecutionProfile;
import 'tensor.dart';

/// Selects how an [OnnxRunner] is created.
///
/// [auto] prefers the native backend when it is available and falls back to the
/// pure-Dart interpreter. On the web only the Dart backend exists; requesting
/// [ort] there throws [UnsupportedError].
enum OnnxBackend { dart, ort, auto }

/// A loaded model that can execute named inputs and return named outputs.
///
/// [OnnxModel] implements this interface for the pure-Dart interpreter.
/// `OrtRunner` (native only) implements it with ONNX Runtime via FFI.
abstract interface class OnnxRunner {
  /// The graph inputs a caller must feed, excluding initializer defaults.
  List<TensorSpec> get inputSpecs;

  /// The graph's declared output names, in order.
  List<String> get outputNames;

  /// Runs the model synchronously. On the native backend this blocks the
  /// calling isolate for the duration of the inference.
  Map<String, Tensor> run(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  });

  /// Runs the model asynchronously. The Dart backend can execute partitioned
  /// work on its isolate pool; the native backend currently performs the FFI
  /// call on the calling isolate, so callers should use a worker isolate for UI
  /// work.
  Future<Map<String, Tensor>> runAsync(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  });

  /// Releases native resources and worker isolates. Idempotent.
  void dispose();
}
