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
  final plan = nn.pretransformWinogradF2x2(wt);
  nn.opConv(xt, wt, bias, pads: [0,0,0,0], winogradPlan: plan);

  double time(String label, Tensor Function() fn, [int reps = 3]) {
    double best = double.infinity;
    for (int r = 0; r < reps; r++) {
      final sw = Stopwatch()..start();
      fn();
      sw.stop();
      if (sw.elapsedMicroseconds.toDouble() < best) best = sw.elapsedMicroseconds.toDouble();
    }
    print('$label: ${(best/1000).toStringAsFixed(1)}ms');
    return best;
  }

  time('im2col full', () => nn.opConv(xt, wt, bias, pads: [0,0,0,0]));
  time('winograd full', () => nn.opConv(xt, wt, bias, pads: [0,0,0,0], winogradPlan: plan));
  time('winograd 16x8-row bands', () {
    late Tensor y;
    for (int b = 0; b < oh; b += 8) {
      y = nn.opConv(xt, wt, bias, pads: [0,0,0,0], winogradPlan: plan, bandStart: b, bandEnd: b + 8);
    }
    return y;
  });
  time('im2col 16x8-row bands', () {
    late Tensor y;
    for (int b = 0; b < oh; b += 8) {
      y = nn.opConv(xt, wt, bias, pads: [0,0,0,0], bandStart: b, bandEnd: b + 8);
    }
    return y;
  }, 1);
}
