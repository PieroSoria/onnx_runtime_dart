/// Runs the actual split Gemma export against tool/gemma_parity.py's oracles.
library;
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';
import 'package:onnx_runtime_dart/onnx_runtime_dart_io.dart';

void main(List<String> args) {
  if (args.length != 2) throw ArgumentError('Usage: gemma_parity.dart MODEL_DIR ORACLE_DIR');
  final root=args[0], oracle=args[1];
  final reference=jsonDecode(File('$oracle/reference.json').readAsStringSync());
  var lastPhase='';
  final control=OnnxExecutionControl(maxTensorBytes: 4*1024*1024*1024,
      deadline:DateTime.now().add(const Duration(minutes:20)),
      onProgress:(phase,done,total) { if(phase!=lastPhase || done==total || done%100==0) { stdout.writeln('$phase $done/$total rss=${ProcessInfo.currentRss}'); lastPhase=phase; } });
  final embedding=loadOnnxModel('$root/embedding/model.onnx',control:control);
  final decoder=loadOnnxModel('$root/decoder/model.onnx',control:control);
  if (Platform.environment['ONNX_PERSISTENT_KV'] == '1') decoder.enablePersistentKv();
  Tensor read(Map value) {
    final bytes=File('$oracle/${value['file']}').readAsBytesSync();
    return Tensor.float(Float32List.view(bytes.buffer,bytes.offsetInBytes,bytes.length~/4),(value['shape'] as List).cast<int>());
  }
  Map<String,Object> compare(Tensor got,Tensor expected) {
    if(got.length!=expected.length || got.shape.toString()!=expected.shape.toString()) throw StateError('Shape mismatch ${got.shape}/${expected.shape}');
    double maxAbs=0,sum=0,norm=0,dot=0,gn=0;
    for(var i=0;i<got.length;i++) {
      final a=got.getD(i),b=expected.getD(i);
      if(!a.isFinite || !b.isFinite) throw StateError('Non-finite output at $i');
      final delta=(a-b).abs();maxAbs=math.max(maxAbs,delta);sum+=delta*delta;norm+=b*b;gn+=a*a;dot+=a*b;
    }
    return {'maxAbs':maxAbs,'rmse':math.sqrt(sum/math.max(1,got.length)),'cosine':dot/math.sqrt(norm*gn)};
  }
  final cache=<String,Tensor>{for(final spec in decoder.inputSpecs) if(spec.name.startsWith('past_key_values.')) spec.name:Tensor.float(Float32List(0),[1,spec.shape[1],0,spec.shape[3]])};
  var past=0;
  final reports=<Map<String,Object>>[];
  for(var step=0;step<(reference['steps'] as List).length;step++) {
    final record=reference['steps'][step];
    final ids=(record['ids'][0] as List).cast<int>();
    final timer=Stopwatch()..start();
    final e=embedding.run({'input_ids':Tensor.int64(ids,[1,ids.length]),'image_features':Tensor.float(Float32List(0),[0,1536]),'audio_features':Tensor.float(Float32List(0),[0,1536])},embedding.outputNames,control:control);
    final metrics=<String,Object>{for(final name in e.keys) name:compare(e[name]!,read(record['embedding'][name]))};
    final y=decoder.run({...e,...cache,'position_ids':Tensor.int64(List.generate(ids.length,(i)=>past+i),[1,ids.length]),'attention_mask':Tensor.int64(List.filled(past+ids.length,1),[1,past+ids.length])},decoder.outputNames,control:control);
    final logits=y['logits']!;
    final width=logits.shape.last, offset=logits.length-width;
    var token=0;
    for(var i=1;i<width;i++) { if(logits.getD(offset+i)>logits.getD(offset+token)) token=i; }
    metrics['logits']=compare(logits,read(record['logits']));
    metrics['token']=token; metrics['expectedToken']=record['token']; metrics['seconds']=timer.elapsedMilliseconds/1000; metrics['rss']=ProcessInfo.currentRss;
    stdout.writeln(jsonEncode(metrics));reports.add(metrics);
    if(token!=record['token']) { exitCode=1; break; }
    for(final name in cache.keys.toList()) { cache[name]=y[name.replaceFirst('past_key_values.','present.')]!; }
    past+=ids.length;
  }
  File('$oracle/dart-report.json').writeAsStringSync(const JsonEncoder.withIndent('  ').convert(reports));
}
