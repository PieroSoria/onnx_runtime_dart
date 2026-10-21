/// Native (`dart:io`) helpers for loading ONNX models from disk, including
/// models whose weights live in a companion external-data file.
///
/// This library imports `dart:io`, so it is **not** available on the web —
/// keep using `package:onnx_runtime_dart/onnx_runtime_dart.dart` (and
/// `OnnxModel.fromBytes`) for web / WebAssembly targets.
library;

import 'dart:io';
import 'dart:typed_data';

import 'onnx_runtime_dart.dart';
import 'src/external_data_guard.dart';

export 'src/external_data_guard.dart' show checkExternalRef;
export 'src/onnx_file_loader_io.dart' show loadOnnxModel;

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
      if (bytes.length != length) {
        throw FormatException('Short read: $location');
      }
      return bytes;
    } finally {
      await handle.close();
    }
  }
}
