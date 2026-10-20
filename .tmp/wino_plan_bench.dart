import 'dart:math';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/src/onnx_nn_ops.dart' as nn;
import 'package:onnx_runtime_dart/src/tensor.dart';

void main() {
  final rnd = Random(1);
  for (final s in [[128,384],[64,128],[384,256],[256,128],[128,64],[384,192],[256,256]]) {
    final m = s[0], c = s[1];
    final wf = Float32List(m*c*9);
    for (int i=0;i<wf.length;i++) wf[i]=rnd.nextDouble();
    final w = Tensor.float(wf, [m,c,3,3]);
    final sw = Stopwatch()..start();
    nn.pretransformWinogradF2x2(w);
    sw.stop();
    print('${m}x${c}x3x3: ${sw.elapsedMilliseconds}ms, U=${(m*c*16*4/1048576).toStringAsFixed(1)}MB');
  }
}
