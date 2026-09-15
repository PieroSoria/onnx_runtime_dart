/// Native (`dart:io`) helpers for loading ONNX models from disk, including
/// models whose weights live in a companion external-data file.
///
/// This library imports `dart:io`, so it is **not** available on the web —
/// keep using `package:onnx_runtime_dart/onnx_runtime_dart.dart` (and `OnnxModel.fromBytes`)
/// for web / WebAssembly targets.
library;

import 'dart:io';
import 'dart:typed_data';

import 'onnx_runtime_dart.dart';

/// Async random-access files rooted in a model directory. Handles are closed
/// after each range read, including failed/cancelled loads.
class FileOnnxDataSource implements OnnxDataSource {
  final Directory directory;
  FileOnnxDataSource(String directory) : directory = Directory(directory);
  File _file(String location) {
    checkExternalRef(location, 0, 0, 0);
    return File('${directory.path}/$location');
  }

  @override
  Future<int> length(String location) => _file(location).length();
  @override
  Future<Uint8List> read(String location, int offset, int length) async {
    final handle = await _file(location).open();
    try {
      checkExternalRef(location, offset, length, await handle.length());
      await handle.setPosition(offset);
      final bytes = await handle.read(length);
      if (bytes.length != length)
        throw FormatException('Short read: $location');
      return bytes;
    } finally {
      await handle.close();
    }
  }
}

/// Loads an ONNX model from [path], resolving any external-data weights from
/// the companion file(s) named in the model (relative to [path]'s directory).
///
/// External weights are read on demand with random access, so a model with a
/// multi-gigabyte `.onnx.data` file is not loaded into memory all at once.
OnnxModel loadOnnxModel(String path,
    {bool lastTokenLogits = false,
    OnnxExecutionControl? control,
    Set<OnnxExperiment> experiments = const {}}) {
  final file = File(path);
  final dir = file.parent.path;
  Uint8List resolve(String location, int offset, int length) {
    checkExternalRef(location, 0, 0, 0);
    final raf = File('$dir/$location').openSync();
    try {
      checkExternalRef(location, offset, length, raf.lengthSync());
      raf.setPositionSync(offset);
      return raf.readSync(length);
    } finally { raf.closeSync(); }
  }

  return OnnxModel.fromBytes(file.readAsBytesSync(),
        control: control,
        externalData: resolve,
        lastTokenLogits: lastTokenLogits,
        experiments: experiments);
}

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
        'the companion file ($fileLen bytes)');
  }
  // GUARD:extdata <<<
}
