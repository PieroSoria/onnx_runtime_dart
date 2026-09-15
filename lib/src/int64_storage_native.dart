import 'dart:typed_data';
export 'dart:typed_data' show Int64List;

int readOnnxInt64(ByteData data, int offset) =>
    data.getInt64(offset, Endian.little);
