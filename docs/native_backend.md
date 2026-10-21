# Native ONNX Runtime backend

`onnx_runtime_dart` keeps the pure-Dart interpreter as its portable default.
On native Dart/Flutter targets the package can optionally execute the same
graph with ONNX Runtime through FFI. The native path is selected explicitly or
through `OnnxBackend.auto`; the web/WebAssembly entrypoint never imports
`dart:ffi` and always uses the Dart interpreter.

## Requirements

- ONNX Runtime **1.15 or newer** with the C API v17 table (tested with 1.30).
- A dynamically loadable runtime for the target:
  - Windows: `onnxruntime.dll`
  - Linux/Android: `libonnxruntime.so`
  - macOS/iOS: `libonnxruntime.dylib`
- The runtime architecture must match the Dart process (x64, arm64, ...).

The package does not bundle binaries or download them at runtime. Ship the
library with the application, install it on the system, or provide an explicit
`libraryPath`.

## Loading a model

```dart
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

final model = await OnnxSession.fromFile(
  'model.onnx',
  backend: OnnxBackend.auto,
  ort: const OrtOptions(
    libraryPath: r'C:\app\runtime\onnxruntime.dll',
    intraOpThreads: 8,
  ),
);

final out = model.run(feed, model.outputNames);
model.dispose();
```

`OnnxSession.fromBytes` works for self-contained models. ONNX Runtime resolves
external-data companions only when the model is opened from a file; use
`fromFile` for those models.

`auto` tries the native backend first and falls back to the Dart interpreter if
the library or session cannot be loaded. `dart` skips FFI. `ort` never falls
back and surfaces `OrtLibraryNotFoundException` or `OrtException`.

For synchronous setup, use `OrtRunner.fromFileSync` or
`OrtRunner.fromBytesSync`; the `fromFile`/`fromBytes` futures are convenience
wrappers over the same synchronous FFI setup.

### Library discovery

Without `libraryPath`, the loader checks, in order:

1. `ONNX_RUNTIME_DART_ORT_LIBRARY`
2. `ORT_LIBRARY`
3. the platform library name in the normal dynamic-loader search path
4. the platform library name next to `Platform.resolvedExecutable`

An explicit `libraryPath` must exist, otherwise loading fails immediately.

## Providers and options

```dart
const options = OrtOptions(
  providers: [OrtCudaProvider(useTf32: false)],
  intraOpThreads: 8,
  interOpThreads: 1,
  graphOptimization: OrtGraphOptimization.all,
  executionMode: OrtExecutionMode.sequential,
  enableCpuMemArena: true,
  enableMemPattern: true,
  preloadCudaDependencies: true,
  cudaLibraryDirectory: r'C:\app\runtime\cuda12',
  fallbackToDart: true,
);
```

- **CPU** is always registered.
- **CUDA** is supported by the native backend. If the loaded library does not
  contain `CUDAExecutionProvider`, a warning is written to stderr and the
  session uses CPU. Session creation can still fail if the provider is present
  but its CUDA/cuDNN dependencies cannot initialize; catch `OrtException` and
  retry with CPU options if that behavior is desired.
- **CoreML, DirectML and NNAPI** option classes exist, but their provider
  bindings are not wired yet and currently throw `UnsupportedError`.

On Windows, `preloadCudaDependencies` registers the CUDA DLL directory before
opening the runtime and loads `cudart`, `cublas`, `cublasLt` and `cudnn`
modules. `cudaLibraryDirectory` overrides discovery; the environment variables
`ONNX_RUNTIME_DART_CUDA_DIR` and `WEBPT_CUDA_DLL_DIR` are also checked.

`OrtRunner.providers` reports the providers actually registered for a session.
`OrtRunner.warnings` exposes non-fatal setup warnings, such as a requested CUDA
provider that is unavailable in the loaded library. `OrtRunner.runtimeVersion`
reports the loaded runtime version.

## Execution and threading

The ONNX Runtime C `Run` call is synchronous. `OrtRunner.runAsync` schedules it
on the calling isolate's event loop; it does not create a worker. For Flutter
UI or a server request handler, create the session and call `run` inside a
worker isolate. An ONNX Runtime session belongs to the isolate that created it
and must not be shared across isolates.

`intraOpThreads: 0` lets ONNX Runtime choose its default (usually all logical
cores). For large convolution models on a busy machine, an explicit count such
as 8 can be faster; benchmark both on the deployment machine.

## Inputs and outputs

- Inputs may be `float32`, `int64`, `int32`, `uint8`, `int8`, `bool`, `float16`
  or `float64`; values are converted to the model's declared element type.
- Outputs support `float32`, `int64`, `uint8`, `int8`, `int32`, `bool`,
  `float16`, `float64`, `bfloat16`, `uint32` and `uint64`.
- `int32`/`bool`/`uint32`/`uint64` outputs are exposed as `int64` tensors;
  `float64`/`float16`/`bfloat16` outputs are exposed as `float32` tensors.
- All requested outputs are returned; unknown or missing input names are
  rejected before the native call.
- Symbolic dimensions (`-1`) accept any positive runtime extent.

## Errors

- `OrtLibraryNotFoundException`: the shared library could not be found or
  opened.
- `OrtException`: ONNX Runtime returned a failure status, or a session was used
  after `dispose()`.
- `UnsupportedError`: an input dtype or an unwired provider was requested.
- `ArgumentError`: invalid input names, ranks or fixed dimensions.

Import `package:onnx_runtime_dart/onnx_runtime_dart_ort.dart` for the native
types (`OrtRunner`, `OrtRuntime`, `OrtException`,
`OrtLibraryNotFoundException`).

## Benchmarks

Local Windows run, LaMa manga inpainting at `1x3x1024x1024` + `1x1x1024x1024`,
deterministic synthetic inputs, ONNX Runtime 1.30, machine under ~60% external
CPU load:

| Backend | Wall time | Peak RSS |
|---|---:|---:|
| ONNX Runtime CPU, 8 intra-op threads | 6.4 s | 4.2 GB |
| ONNX Runtime CUDA, RTX 4070 Ti SUPER | 0.29 s | 1.3 GB |
| Pure Dart, single-threaded | 113.9 s | - |
| Pure Dart, 16 workers (`parallelize`) | ~22 s (previous baseline) | 3.8 GB |

CPU/ORT versus pure-Dart parity on the same feed: max absolute error
`4.9e-5`, mean `7.4e-10`, 0 of 3,145,728 elements outside the project tolerance.
Timings are load-dependent; re-run on the target machine.

Use the benchmark tool to synthesize inputs and optionally compare outputs:

```text
dart run tool/bench_native.dart model.onnx --library path/to/onnxruntime.dll
dart run tool/bench_native.dart model.onnx --library ... --cuda --iters 5
dart run tool/bench_native.dart model.onnx --library ... --compare-dart
```

The native smoke tests run only when `ORT_LIBRARY` points to a runtime:

```text
dart test test/ort_session_test.dart
```

The web path is covered by `dart test -p node test/web_session_test.dart`.
