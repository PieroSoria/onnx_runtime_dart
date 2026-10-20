import 'dart:math';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/src/onnx_nn_ops.dart' as nn;
import 'package:onnx_runtime_dart/src/tensor.dart';

void main() {
  final rnd = Random(3);
  const c = 384, m = 128, h = 130, w = 130, oh = 128, ow = 128;
  final wf = Float32List(c * m * 9);
  for (int i = 0; i < wf.length; i++) { wf[i] = rnd.nextDouble() * 0.1 - 0.05; }
  final xf = Float32List(c * h * w);
  for (int i = 0; i < xf.length; i++) { xf[i] = rnd.nextDouble() * 2 - 1; }
  final bias = Tensor.float(Float32List(m), [m]);
  final wt = Tensor.float(wf, [m, c, 3, 3]);
  final xt = Tensor.float(xf, [1, c, h, w]);
  final plan0 = nn.pretransformWinogradF2x2(wt);
  nn.opConv(xt, wt, bias, pads: [0,0,0,0], winogradPlan: plan0);
  double bandIm2col() {
    late Tensor y;
    final sw = Stopwatch()..start();
    for (int b = 0; b < oh; b += 8) {
      y = nn.opConv(xt, wt, bias, pads: [0,0,0,0], bandStart: b, bandEnd: b + 8);
    }
    sw.stop();
    return sw.elapsedMicroseconds / 1000.0;
  }
  double planAndBands() {
    late Tensor y;
    final sw = Stopwatch()..start();
    final p = nn.pretransformWinogradF2x2(wt);
    final t1 = sw.elapsedMicroseconds;
    for (int b = 0; b < oh; b += 8) {
      y = nn.opConv(xt, wt, bias, pads: [0,0,0,0], winogradPlan: p, bandStart: b, bandEnd: b + 8);
    }
    sw.stop();
    print('  plan ${(t1/1000).toStringAsFixed(1)}ms bands ${((sw.elapsedMicroseconds-t1)/1000).toStringAsFixed(1)}ms');
    return sw.elapsedMicroseconds / 1000.0;
  }
  for (int r = 0; r < 4; r++) {
    print('im2col ${bandIm2col().toStringAsFixed(1)}ms');
    print('wino+plan ${planAndBands().toStringAsFixed(1)}ms');
  }
}
