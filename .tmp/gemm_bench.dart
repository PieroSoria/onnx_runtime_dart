import 'dart:typed_data';
import 'package:onnx_runtime_dart/src/gemm_kernel_simd.dart' as gemm;

void main() {
  const m = 384, k = 3456, n = 32768;
  final a = Float32List(m * k);
  final b = Float32List(k * n);
  for (var i = 0; i < a.length; i++) { a[i] = (i % 17) * 0.01 - 0.08; }
  for (var i = 0; i < b.length; i++) { b[i] = (i % 13) * 0.01 - 0.06; }
  final out = Float32List(m * n);
  gemm.matmulKernel(a, 0, b, 0, out, 0, m, k, n, panelBytes: 1 << 20);
  print('warm done');
  for (final pb in [192 << 10, 512 << 10, 1 << 20, 2 << 20, 8 << 20, 64 << 20]) {
    final sw = Stopwatch()..start();
    gemm.matmulKernel(a, 0, b, 0, out, 0, m, k, n, panelBytes: pb);
    sw.stop();
    final flops = 2.0 * m * k * n;
    print('panel=${pb >> 10}KB time=${sw.elapsedMilliseconds}ms '
        'gflops=${(flops / (sw.elapsedMicroseconds * 1000)).toStringAsFixed(1)} '
        'panelCols=${pb ~/ (k * 4)}');
  }
}
