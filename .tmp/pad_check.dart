import 'dart:typed_data';
import 'dart:math';
import 'package:onnx_runtime_dart/src/onnx_ops.dart';
import 'package:onnx_runtime_dart/src/tensor.dart';

List<int> refMap(int dim, int before, int after, String mode) {
  final out = <int>[];
  for (int j = 0; j < dim + before + after; j++) {
    int i = j - before;
    if (i < 0 || i >= dim) {
      if (mode == 'reflect') {
        while (i < 0 || i >= dim) {
          if (i < 0) i = -i;
          if (i >= dim) i = 2 * dim - 2 - i;
        }
      } else if (mode == 'edge') {
        i = i < 0 ? 0 : dim - 1;
      } else {
        i = -1;
      }
    }
    out.add(i);
  }
  return out;
}

void main() {
  print('start');
  final rnd = Random(7);
  for (final shape in [[5], [2,3,4], [2,3,4,5], [2,2,3,3,2]]) {
    print('shape $shape');
    final n = shape.fold<int>(1,(a,b)=>a*b);
    final v = Float32List.fromList(List.generate(n, (_) => rnd.nextDouble()));
    final x = Tensor.float(v, shape);
    final pads = [for (final d in shape) 1, for (final d in shape) 2];
    for (final mode in ['constant','reflect','edge']) {
      final y = opPad(x, pads, mode: mode, constantValue: -3);
      final maps = [for (int a=0;a<shape.length;a++) refMap(shape[a],1,2,mode)];
      final o = y.shape;
      final total = o.fold<int>(1,(a,b)=>a*b);
      final coords = List<int>.filled(shape.length,0);
      for (int idx=0; idx<total; idx++) {
        int so=0; var inside=true;
        for (int a=0;a<shape.length;a++) {
          final i=maps[a][coords[a]];
          if (i<0) { inside=false; break; }
          so += i*x.strides[a];
        }
        final want = inside ? v[so] : -3.0;
        if (y.f![idx] != want) {
          throw StateError('mismatch shape=$shape mode=$mode idx=$idx got=${y.f![idx]} want=$want');
        }
        for (int a=shape.length-1;a>=0;a--) { if (++coords[a]<o[a]) break; coords[a]=0; }
      }
      print('ok $shape $mode');
    }
    final yi = opPad(Tensor.int64(Int64List.fromList(List.generate(n, (k)=>k)), shape), pads, mode: 'edge');
    final wantLen = [for (int a=0;a<shape.length;a++) shape[a]+3].fold<int>(1,(a,b)=>a*b); if (yi.length != wantLen) throw StateError('int len');
  }
  print('all pad checks passed');
}





