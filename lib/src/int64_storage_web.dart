/// JavaScript has no Dart Int64List. Keep shape/index integers in a Float64
/// buffer and reject values outside its exact integer range instead of silently
/// corrupting indices. Native builds retain real signed 64-bit storage.
library;

import 'dart:collection';
import 'dart:typed_data';

class Int64List extends ListBase<int> {
  final Float64List _data;
  Int64List(int length) : _data = Float64List(length);
  Int64List._(this._data);
  factory Int64List.fromList(List<int> values) {
    final out = Int64List(values.length);
    for (var i = 0; i < values.length; i++) {
      out[i] = values[i];
    }
    return out;
  }
  factory Int64List.sublistView(Int64List values, int start, [int? end]) =>
      Int64List._(Float64List.sublistView(values._data, start, end));
  ByteBuffer get buffer => _data.buffer;
  int get lengthInBytes => _data.lengthInBytes;
  @override
  int get length => _data.length;
  @override
  set length(int value) =>
      throw UnsupportedError('Fixed-length integer storage');
  @override
  int operator [](int index) => _data[index].toInt();
  @override
  void operator []=(int index, int value) {
    if (value < -9007199254740991 || value > 9007199254740991)
      throw UnsupportedError('Integer exceeds exact JavaScript range: $value');
    _data[index] = value.toDouble();
  }
}

int readOnnxInt64(ByteData data, int offset) {
  final high = data.getInt32(offset + 4, Endian.little);
  final low = data.getUint32(offset, Endian.little);
  final value = high * 4294967296 + low;
  if (value < -9007199254740991 || value > 9007199254740991)
    throw UnsupportedError('ONNX int64 exceeds exact JavaScript range');
  return value;
}
