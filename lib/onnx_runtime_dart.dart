/// A dependency-light **pure-Dart ONNX inference runtime** with an optional
/// native ONNX Runtime backend.
///
/// The default path interprets the graph in plain Dart, so the same code runs
/// on every Dart/Flutter target, **including the web and WebAssembly**. On
/// native platforms, [OnnxSession] can opt into ONNX Runtime through FFI when
/// a compatible library is available; the web entrypoint never imports
/// `dart:ffi` and keeps using the Dart interpreter.
///
/// ```dart
/// import 'dart:typed_data';
/// import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
///
/// final model = OnnxModel.fromBytes(bytes); // bytes of a .onnx file
/// final out = model.run(
///   {'input': Tensor.float(Float32List.fromList([1, 2, 3]), [1, 3])},
///   ['output'],
/// );
/// print(out['output']!.asFloatList());
/// ```
///
/// The generated protobuf bindings (`onnx.pb.dart`) derive from the ONNX
/// project's `onnx.proto3` and are used under Apache-2.0; see `NOTICE.md`.
library;

export 'src/tensor.dart' show Tensor, DType, TensorSpec;
export 'src/bpe_tokenizer.dart' show BpeTokenizer;
export 'src/wordpiece_tokenizer.dart' show WordPieceTokenizer;
export 'src/unigram_tokenizer.dart' show UnigramTokenizer;
export 'src/token_template.dart' show TruncationDirection, TruncationStrategy;
export 'src/onnx_model.dart' show OnnxModel;
export 'src/onnx_runner.dart' show OnnxBackend, OnnxRunner;
export 'src/onnx_session.dart' show OnnxSession;
export 'src/ort/ort_options.dart'
    show
        OrtOptions,
        OrtProvider,
        OrtCpuProvider,
        OrtCudaProvider,
        OrtCoreMlProvider,
        OrtDirectMlProvider,
        OrtNnapiProvider,
        OrtGraphOptimization,
        OrtExecutionMode;
export 'src/onnx_graph.dart'
    show OnnxGraphExecutor, ExecutionProfile, OnnxExperiment;
export 'src/onnx_ops.dart' show OnnxRandomInject;
export 'src/onnx_proto_loader.dart' show ExternalDataResolver;
export 'src/model_source.dart' show OnnxDataSource, MemoryOnnxDataSource;
export 'src/execution_control.dart'
    show OnnxExecutionControl, OnnxCancelledException, OnnxMemoryLimitException;
export 'src/onnx_audit.dart'
    show
        OnnxCompatibilityIssue,
        OnnxCompatibilityReport,
        auditOnnxBytes,
        auditOnnxModel;
