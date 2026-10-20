import 'dart:math';
import 'dart:typed_data';
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
  final bias = Float32List(m);
  final wt = Tensor.float(wf, [m, c, 3, 3]);
  final xt = Tensor.float(xf, [1, c, h, w]);
  final sw0 = Stopwatch()..start();
  nn.opConv(xt, wt, Tensor.float(bias, [m]), pads: [0,0,0,0], winogradPlan: nn.pretransformWinogradF2x2(wt));
  print('warm ${sw0.elapsedMilliseconds}ms');
  final plan = nn.pretransformWinogradF2x2(wt);
  for (int r = 0; r < 3; r++) {
    final sw = Stopwatch()..start();
    nn.opConv(xt, wt, Tensor.float(bias, [m]), pads: [0,0,0,0], winogradPlan: plan);
    print('local wino ${sw.elapsedMilliseconds}ms');
  }
  final pool = await GemmPool.spawn(16, const {}, {'W': (wf, [m, c, 3, 3], bias)}, {'W'});
  for (int r = 0; r < 5; r++) {
    final sw = Stopwatch()..start();
    final out = await pool.conv('W', xf, [1, c, h, w],
        strides: null, pads: [0,0,0,0], dilations: null, group: 1,
        autoPad: 'NOTSET', kernelH: 3, n: 1, m: m, oh: 128, ow: 128);
    sw.stop();
    print('pool16 wino ${sw.elapsedMilliseconds}ms ${out[0]}');
  }
  pool.dispose();
}
