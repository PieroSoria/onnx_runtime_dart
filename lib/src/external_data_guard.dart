/// Validation for model-declared external-data references.
library;

/// Guards a model-declared external-data reference before it touches disk. A
/// hostile `.onnx` must not read arbitrary files or trigger a huge allocation:
/// [location] must be a plain relative path inside the model's own directory
/// (no `..`, no absolute path, no drive/volume), and `[offset, offset+length)`
/// must lie within the companion file's [fileLen] bytes. Violations reject with
/// [FormatException] — the documented reject type — never a leaked
/// `FileSystemException`/`RangeError`/OOM. (guard:extdata)
void checkExternalRef(String location, int offset, int length, int fileLen) {
  // GUARD:extdata >>>
  if (location.isEmpty ||
      location.contains('..') ||
      location.startsWith('/') ||
      location.startsWith(r'\') ||
      location.contains(':')) {
    throw FormatException('unsafe external-data location: "$location"');
  }
  if (offset < 0 ||
      length < 0 ||
      length > fileLen ||
      offset > fileLen - length) {
    throw FormatException(
      'external-data range [$offset, +$length) lies outside '
      'the companion file ($fileLen bytes)',
    );
  }
  // GUARD:extdata <<<
}
