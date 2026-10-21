/// Native ONNX Runtime backend for `onnx_runtime_dart`.
///
/// Import this library only on native targets (`dart:ffi`); it is not
/// available on the web. Most callers can use the web-safe
/// `OnnxSession`/`OnnxBackend` API from the main library instead.
library;

export 'src/ort/ort_backend.dart' show OrtRunner, OrtRuntime;
export 'src/ort/ort_exception.dart'
    show OrtException, OrtLibraryNotFoundException;
export 'src/ort/ort_options.dart';
