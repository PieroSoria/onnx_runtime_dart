/// Configuration for the optional native ONNX Runtime backend.
///
/// This library is intentionally free of `dart:ffi`/`dart:io` imports so the
/// web-safe API can expose the same option types (and reject the backend at
/// runtime when it is unavailable).
library;

/// ONNX Runtime graph optimization level.
enum OrtGraphOptimization {
  disabled(0),
  basic(1),
  extended(2),
  all(99);

  const OrtGraphOptimization(this.ortValue);
  final int ortValue;
}

/// ONNX Runtime execution mode.
enum OrtExecutionMode {
  sequential(0),
  parallel(1);

  const OrtExecutionMode(this.ortValue);
  final int ortValue;
}

/// An execution provider requested for an ONNX Runtime session.
///
/// CPU is always available. CUDA is implemented by this package; CoreML,
/// DirectML and NNAPI are accepted by the API but currently rejected with an
/// [UnsupportedError] by the native loader until their provider bindings are
/// wired up.
sealed class OrtProvider {
  const OrtProvider();
}

/// The CPU execution provider (always available).
final class OrtCpuProvider extends OrtProvider {
  const OrtCpuProvider();
}

/// NVIDIA CUDA execution provider.
final class OrtCudaProvider extends OrtProvider {
  const OrtCudaProvider({
    this.deviceId = 0,
    this.useTf32 = false,
    this.arenaExtendStrategy = OrtCudaArenaExtendStrategy.nextPowerOfTwo,
    this.cudnnConvAlgoSearch = OrtCudnnConvAlgoSearch.exhaustive,
    this.gpuMemLimit,
  });

  final int deviceId;
  final bool useTf32;
  final OrtCudaArenaExtendStrategy arenaExtendStrategy;
  final OrtCudnnConvAlgoSearch cudnnConvAlgoSearch;

  /// Optional CUDA memory limit in bytes. Null leaves ONNX Runtime's default.
  final int? gpuMemLimit;
}

/// CUDA memory arena growth strategy.
enum OrtCudaArenaExtendStrategy {
  nextPowerOfTwo('kNextPowerOfTwo'),
  sameAsRequested('kSameAsRequested');

  const OrtCudaArenaExtendStrategy(this.ortValue);
  final String ortValue;
}

/// cuDNN convolution algorithm search strategy.
enum OrtCudnnConvAlgoSearch {
  exhaustive('EXHAUSTIVE'),
  heuristic('HEURISTIC'),
  defaultAlgorithm('DEFAULT');

  const OrtCudnnConvAlgoSearch(this.ortValue);
  final String ortValue;
}

/// Apple CoreML execution provider (macOS/iOS).
final class OrtCoreMlProvider extends OrtProvider {
  const OrtCoreMlProvider({
    this.computeUnits = 'ALL',
    this.modelFormat = 'MLProgram',
    this.requireStaticInputShapes = false,
    this.enableOnSubgraphs = false,
  });

  final String computeUnits;
  final String modelFormat;
  final bool requireStaticInputShapes;
  final bool enableOnSubgraphs;
}

/// Windows DirectML execution provider.
final class OrtDirectMlProvider extends OrtProvider {
  const OrtDirectMlProvider({this.deviceId = 0});
  final int deviceId;
}

/// Android NNAPI execution provider.
final class OrtNnapiProvider extends OrtProvider {
  const OrtNnapiProvider({
    this.useFp16 = false,
    this.nchw = false,
    this.flags = 0,
  });

  final bool useFp16;
  final bool nchw;
  final int flags;
}

/// Options for loading and running a model with ONNX Runtime.
class OrtOptions {
  const OrtOptions({
    this.libraryPath,
    this.providers = const [OrtCpuProvider()],
    this.intraOpThreads = 0,
    this.interOpThreads = 1,
    this.graphOptimization = OrtGraphOptimization.all,
    this.executionMode = OrtExecutionMode.sequential,
    this.enableCpuMemArena = true,
    this.enableMemPattern = true,
    this.preloadCudaDependencies = true,
    this.cudaLibraryDirectory,
    this.fallbackToDart = true,
  }) : assert(intraOpThreads >= 0),
       assert(interOpThreads >= 0);

  /// Explicit path to `onnxruntime.dll` / `libonnxruntime.so` /
  /// `libonnxruntime.dylib`. When null, the loader checks
  /// `ONNX_RUNTIME_DART_ORT_LIBRARY`, `ORT_LIBRARY`, common names next to the
  /// executable and the platform loader path.
  final String? libraryPath;

  /// Providers to append, in order. An empty list is treated as CPU.
  final List<OrtProvider> providers;

  /// `intra_op_num_threads`; 0 lets ONNX Runtime choose.
  final int intraOpThreads;

  /// `inter_op_num_threads`; 1 is the safe default for sequential graphs.
  final int interOpThreads;

  final OrtGraphOptimization graphOptimization;
  final OrtExecutionMode executionMode;
  final bool enableCpuMemArena;
  final bool enableMemPattern;

  /// On Windows, preload CUDA/cuDNN DLLs from [cudaLibraryDirectory] (or a
  /// discovered sibling `cuda12` directory) before opening the runtime.
  final bool preloadCudaDependencies;
  final String? cudaLibraryDirectory;

  /// When [OnnxBackend.auto] is requested, fall back to the Dart interpreter
  /// if the native library or session cannot be loaded. Explicit
  /// [OnnxBackend.ort] never falls back.
  final bool fallbackToDart;
}
