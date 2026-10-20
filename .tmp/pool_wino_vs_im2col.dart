import 'dart:math';
import 'dart:typed_data';
import 'package:onnx_runtime_dart/src/parallel_pool.dart';

void main() async {
  final rnd = Random(3);
  const c = 384, m = 128, h = 130, w = 130;
  final ws = <String, (Float32List, List<int>, Float32List?)>{};
  for (int k = 0; k < 24; k++) {
    final wf = Float32List(c * m * 9);
    for (int i = 0; i < wf.length; i++) { wf[i] = rnd.nextDouble() * 0.1 - 0.05; }
    ws['W$k'] = (wf, [m, c, 3, 3], Float32List(m));
  }
  final xf = Float32List(c * h * w);
  for (int i = 0; i < xf.length; i++) { xf[i] = rnd.nextDouble() * 2 - 1; }
  final pool = await GemmPool.spawn(16, const {}, ws);
  for (int r = 0; r < 4; r++) {
    for (int k = 0; k < 8; k++) {
      final name = 'W${r * 8 + k}';
      final swi = Stopwatch()..start();
      final a = await pool.conv(name, xf, [1, c, h, w], strides: null,
          pads: [0,0,0,0], dilations: null, group: 1, autoPad: 'NOTSET',
          kernelH: 3, n: 1, m: m, oh: 128, ow: 128);
      swi.stop();
      final sww = Stopwatch()..start();
      final b = await pool.conv(name, xf, [1, c, h, w], strides: null,
          pads: [0,0,0,0], dilations: null, group: 1, autoPad: 'NOTSET',
          kernelH: 3, n: 1, m: m, oh: 128, ow: 128, winograd: true);
      sww.stop();
      print('r$r $name im2col ${swi.elapsedMilliseconds}ms wino ${sww.elapsedMilliseconds}ms d=${a[0]-b[0]}');
    }
  }
  pool.dispose();
}
