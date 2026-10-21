/// A parsed ONNX model ready to run on the pure-Dart interpreter.
library;

import 'dart:typed_data';

import 'execution_control.dart';
import 'model_source.dart';
import 'onnx.pb.dart';
import 'onnx_graph.dart';
import 'onnx_proto_loader.dart';
import 'onnx_runner.dart';
import 'tensor.dart';

/// A parsed ONNX model ready to run.
///
/// Wraps an [OnnxGraphExecutor]; construct one with [OnnxModel.fromBytes] from
/// the raw bytes of a `.onnx` file, then call [run].
class OnnxModel implements OnnxRunner {
  final OnnxGraphExecutor _executor;

  OnnxModel._(this._executor);

  /// Portable asynchronous loading, including external tensors in initializers
  /// and subgraph/Constant attributes. Reads and decodes one weight at a time.
  /// Locations used by external tensors are relative to the source's root.
  static Future<OnnxModel> fromSource(
    OnnxDataSource source,
    String location, {
    OnnxExecutionControl? control,
    bool fuse = true,
    bool lastTokenLogits = false,
    Set<OnnxExperiment> experiments = const {},
  }) async {
    final loaded = await readOnnxSource(source, location, control);
    return withOnnxControl(
      control,
      () => OnnxModel._(
        OnnxGraphExecutor(
          loaded.model,
          decodedTensors: loaded.tensors,
          deferredTables: loaded.deferred,
          fuse: fuse,
          lastTokenLogits: lastTokenLogits,
          experiments: experiments,
        ),
      ),
    );
  }

  /// Parses a serialized ONNX model ([bytes] of a `.onnx` file) and prepares
  /// it for execution.
  ///
  /// If the model stores its weights in a companion data file (large models,
  /// `dataLocation == EXTERNAL`), pass an [externalData] resolver that returns
  /// the bytes for a given `(location, offset, length)`. `loadOnnxModel` in
  /// `package:onnx_runtime_dart/onnx_runtime_dart_io.dart` wires this up
  /// automatically for the native (`dart:io`) platforms.
  /// [fuse] controls the load-time pattern fusion (GELU/SDPA/RMSNorm);
  /// disable it to execute the graph exactly node-by-node (diagnostics).
  /// [lastTokenLogits] rewrites a `logits = MatMul(hidden, vocab_weight)`
  /// output so only the final sequence position is projected (output becomes
  /// `[…, 1, vocab]`). For autoregressive generation the prompt's vocab matmul
  /// is the biggest prefill op yet only the last row is ever sampled, so this
  /// skips the wasted `seq-1` rows. Opt-in: it changes the logits' seq extent.
  factory OnnxModel.fromBytes(
    Uint8List bytes, {
    OnnxExecutionControl? control,
    ExternalDataResolver? externalData,
    bool fuse = true,
    bool lastTokenLogits = false,
    Set<OnnxExperiment> experiments = const {},
  }) {
    // ModelProto.fromBuffer parses untrusted bytes. Malformed data surfaces
    // either as the protobuf decoder's own InvalidProtocolBufferException or,
    // for a corrupt length-delimited field, as a leaked RangeError from
    // Uint8List.view. Both mean the same thing — bad input — so normalize every
    // decode failure to the single documented reject type (FormatException), so
    // a caller catching it handles every malformed .onnx without depending on a
    // package:protobuf-internal exception type. (Verified by tool/fuzz/.)
    final ModelProto proto;
    control?.checkpoint();
    try {
      proto = ModelProto.fromBuffer(bytes);
    } catch (e) {
      // GUARD:protobuf_leak >>>
      throw FormatException('Malformed ONNX model (protobuf decode): $e');
      // GUARD:protobuf_leak <<<
    }
    return withOnnxControl(
      control,
      () => OnnxModel._(
        OnnxGraphExecutor(
          proto,
          externalData: externalData,
          fuse: fuse,
          lastTokenLogits: lastTokenLogits,
          experiments: experiments,
        ),
      ),
    );
  }

  /// The graph inputs a caller must feed (excluding those with initializer
  /// defaults), each with its declared shape (symbolic dims as -1) and ONNX
  /// element type. Use this to build correctly-shaped feeds — e.g. the empty
  /// `past_key_values.*` tensors an LLM decoder needs on its first step.
  @override
  List<TensorSpec> get inputSpecs => _executor.inputSpecs;

  /// The graph's declared output names, in order.
  @override
  List<String> get outputNames => _executor.outputNames;

  /// Opt in to capacity-managed persistent KV: the executor keeps one reusable
  /// buffer per `Attention` present output, so successive decode steps append
  /// rows in place and return offset views instead of copying the growing
  /// history. Past-KV inputs are ignored while enabled (the cache is
  /// authoritative). Call before the first `run`.
  void enablePersistentKv() => _executor.enablePersistentKv();

  /// Runs the graph with the given named [inputs] and returns the requested
  /// named [outputNames].
  ///
  /// Pass a [profile] to accumulate per-op-type wall time across the run.
  @override
  Map<String, Tensor> run(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  }) => withOnnxControl(
    control,
    () => _executor.run(inputs, outputNames, profile: profile),
  );

  /// Spawns [workers] isolate workers and partitions the model's large
  /// `MatMul` weights across them by output column; afterwards [runAsync]
  /// executes those matmuls in parallel (results stay bitwise identical to
  /// [run]). Native targets only — throws [UnsupportedError] on the web.
  ///
  /// [poolConv] opts 2-D convolutions into the pool as well (output-row
  /// bands); measure before enabling — for typical CNN latencies the
  /// activation copying outweighs the parallel compute.
  ///
  /// Call [dispose] when done to shut the workers down.
  Future<void> parallelize({
    required int workers,
    int minWeightElements = 65536,
    bool poolConv = false,
  }) => _executor.parallelize(
    workers: workers,
    minWeightElements: minWeightElements,
    poolConv: poolConv,
  );

  /// Like [run], but partitioned matmuls execute on the isolate pool set up
  /// by [parallelize]. Without a prior [parallelize] call it behaves exactly
  /// like [run].
  @override
  Future<Map<String, Tensor>> runAsync(
    Map<String, Tensor> inputs,
    List<String> outputNames, {
    ExecutionProfile? profile,
    OnnxExecutionControl? control,
  }) => withOnnxControl(
    control,
    () => _executor.runAsync(inputs, outputNames, profile: profile),
  );

  /// Shuts down the isolate pool (no-op if [parallelize] was never called).
  @override
  void dispose() => _executor.dispose();
}
