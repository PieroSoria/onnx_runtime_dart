import 'dart:math';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/src/gemm_kernel_simd.dart' as gemm;
import 'package:onnx_runtime_dart/src/onnx_nn_ops.dart' as nn;
import 'package:onnx_runtime_dart/src/parallel_pool.dart';
import 'package:onnx_runtime_dart/src/tensor.dart';

void main() async {
  final rnd = Random(3);
  const c = 384, m = 128, h = 130, w = 130;
  final wf = Float32List(c * m * 9);
  for (int i = 0; i < wf.length; i++) { wf[i] = rnd.nextDouble() * 0.1 - 0.05; }
  final xf = Float32List(c * h * w);
  for (int i = 0; i < xf.length; i++) { xf[i] = rnd.nextDouble() * 2 - 1; }
  final bias = Tensor.float(Float32List(m), [m]);
  final wt = Tensor.float(wf, [m, c, 3, 3]);
  final xt = Tensor.float(xf, [1, c, h, w]);

  for (int r = 0; r < 2; r++) {
    final sw = Stopwatch()..start();
    final y = nn.opConv(xt, wt, bias, pads: [0,0,0,0]);
    sw.stop();
    print('local ${sw.elapsedMilliseconds}ms ${y.f![0]}');
  }
  for (final workers in [1, 16]) {
    final pool = await GemmPool.spawn(workers, const {}, {'W': (wf, [m, c, 3, 3], Float32List(m))});
    for (int r = 0; r < 3; r++) {
      final sw = Stopwatch()..start();
      final out = await pool.conv('W', xf, [1, c, h, w],
          strides: null, pads: [0,0,0,0], dilations: null, group: 1,
          autoPad: 'NOTSET', kernelH: 3, n: 1, m: m, oh: 128, ow: 128);
      sw.stop();
      print('pool$workers ${sw.elapsedMilliseconds}ms ${out[0]}');
    }
    pool.dispose();
  }
}
