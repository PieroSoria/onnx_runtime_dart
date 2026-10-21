/// Errors raised by the optional native ONNX Runtime backend.
library;

/// A failure while loading or running ONNX Runtime.
class OrtException implements Exception {
  const OrtException(this.message, [this.cause]);

  final String message;
  final Object? cause;

  @override
  String toString() => cause == null
      ? 'OrtException: $message'
      : 'OrtException: $message ($cause)';
}

/// The requested ONNX Runtime shared library could not be found or loaded.
class OrtLibraryNotFoundException extends OrtException {
  const OrtLibraryNotFoundException(String message, [Object? cause])
    : super(message, cause);
}
