/// dart run tool/audit_model.dart model.onnx [--json]
/// Does not open external weight files. Inline protobuf bytes are read in full.
library;

import 'dart:convert';
import 'dart:io';
import 'package:onnx_runtime_dart/onnx_runtime_dart.dart';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln('Usage: dart run tool/audit_model.dart model.onnx [--json]');
    exitCode = 64;
    return;
  }
  final report = auditOnnxBytes(File(args.first).readAsBytesSync());
  if (args.contains('--json')) {
    stdout.writeln(const JsonEncoder.withIndent('  ').convert(report.toJson()));
  } else {
    stdout.writeln(
        'Initializer estimate: ${report.initializerBytes} bytes; external tensors: ${report.externalTensorCount}');
    for (final entry in report.operators.entries) {
      stdout.writeln('${entry.value}\t${entry.key}');
    }
    for (final issue in report.issues) {
      stdout.writeln(issue);
    }
  }
  exitCode = report.hasErrors ? 1 : 0;
}
