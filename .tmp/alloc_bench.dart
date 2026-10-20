import 'dart:typed_data';
void main() {
  const n = 128 * 1024 * 1024;
  for (int r = 0; r < 3; r++) {
    final sw = Stopwatch()..start();
    final a = Float32List(n);
    sw.stop();
    print('alloc ${sw.elapsedMilliseconds}ms ${a[0]}');
  }
  final b = Float32List(n);
  for (int r = 0; r < 3; r++) {
    final sw = Stopwatch()..start();
    b.fillRange(0, n, 0.5);
    sw.stop();
    print('fill ${sw.elapsedMilliseconds}ms');
  }
}
