/// Metadata-only compatibility inspection; never resolves external tensor data.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'onnx.pb.dart';
import 'operator_schemas.dart';
import 'external_gather.dart';

String canonicalOnnxDomain(String domain) => domain == 'ai.onnx' ? '' : domain;

final Map<String, dynamic> _schemas =
    jsonDecode(operatorSchemasJson) as Map<String, dynamic>;

/// A schema selected using the model's import for this operator's domain.
Map<String, dynamic>? resolveOnnxSchema(
  String domain,
  String name,
  int version,
) {
  final candidates = _schemas['${canonicalOnnxDomain(domain)}:$name'] as List?;
  Map<String, dynamic>? selected;
  for (final candidate in candidates ?? const []) {
    if ((candidate['version'] as int) <= version) {
      selected = candidate as Map<String, dynamic>;
    }
  }
  return selected;
}

class OnnxCompatibilityIssue {
  final String path;
  final String message;
  final bool isError;
  const OnnxCompatibilityIssue(this.path, this.message, {this.isError = true});
  Map<String, Object> toJson() => {
        'path': path,
        'message': message,
        'severity': isError ? 'error' : 'warning',
      };
  @override
  String toString() => '${isError ? 'error' : 'warning'}: $path: $message';
}

class OnnxCompatibilityReport {
  final List<OnnxCompatibilityIssue> issues;
  final Map<String, int> operators;

  /// Estimated resident initializer bytes after decoding, excluding activations,
  /// protobuf buffers, allocator overhead, folding, and kernel workspaces.
  final int initializerBytes;
  final int externalTensorCount;
  final int deferredInitializerBytes;
  const OnnxCompatibilityReport(
    this.issues,
    this.operators,
    this.initializerBytes,
    this.externalTensorCount,
    this.deferredInitializerBytes,
  );
  bool get hasErrors => issues.any((i) => i.isError);
  void throwIfIncompatible() {
    if (hasErrors) {
      throw UnsupportedError(
        'ONNX compatibility audit failed:\n${issues.where((i) => i.isError).join('\n')}',
      );
    }
  }

  Map<String, Object> toJson() => {
        'hasErrors': hasErrors,
        'issues': issues.map((i) => i.toJson()).toList(),
        'operators': operators,
        'initializerBytes': initializerBytes,
        'externalTensorCount': externalTensorCount,
        'deferredInitializerBytes': deferredInitializerBytes,
      };
}

/// Inspect a serialized model. The protobuf bytes are already resident; this
/// does not avoid reading inline weights. External weight files are untouched.
OnnxCompatibilityReport auditOnnxBytes(Uint8List bytes) =>
    auditOnnxModel(ModelProto.fromBuffer(bytes));

OnnxCompatibilityReport auditOnnxModel(ModelProto model) {
  final issues = <OnnxCompatibilityIssue>[];
  final operators = <String, int>{};
  final imports = <String, int>{};
  void issue(String path, String message, {bool warning = false}) =>
      issues.add(OnnxCompatibilityIssue(path, message, isError: !warning));
  for (final imp in model.opsetImport) {
    final domain = canonicalOnnxDomain(imp.domain),
        version = imp.version.toInt();
    if (imports.containsKey(domain)) {
      issue('opset_import', 'Duplicate domain "$domain"');
    }
    if (version <= 0 || (domain.isEmpty && version > 27)) {
      issue('opset_import', 'Unsupported domain/version "$domain":$version');
    }
    imports[domain] = version;
  }
  // Historical programmatic graphs in this package omitted imports. Keep this
  // mode explicit in the report; serialized exports should always declare them.
  final legacy = imports.isEmpty;
  if (legacy) {
    issue(
      'model',
      'No opset imports: legacy schema validation is incomplete',
      warning: true,
    );
  }
  var bytes = 0, external = 0;
  const supportedTypes = {
    1: 4,
    2: 1,
    3: 1,
    6: 8,
    7: 8,
    9: 8,
    10: 4,
    11: 4,
    21: 1,
    22: 1,
  };
  void dtype(int type, String path) {
    if (type != 0 && !supportedTypes.containsKey(type)) {
      issue(path, 'Unsupported tensor element type $type');
    }
  }

  void tensor(TensorProto t, String path) {
    if (t.dataType == 0) issue(path, 'Tensor element type is missing');
    dtype(t.dataType, path);
    var elements = 1;
    for (final dim in t.dims) {
      final d = dim.toInt();
      if (d < 0 || (d != 0 && elements > 9007199254740991 ~/ d)) {
        issue(path, 'Invalid or oversized tensor dimensions');
        return;
      }
      elements *= d;
    }
    bytes += elements * (supportedTypes[t.dataType] ?? 0);
    if (t.dataLocation == TensorProto_DataLocation.EXTERNAL) {
      external++;
      final entries = {for (final e in t.externalData) e.key: e.value};
      if ((entries['location'] ?? '').isEmpty) {
        issue(path, 'Missing external-data location');
      }
      for (final key in ['offset', 'length']) {
        if (entries.containsKey(key) &&
            (int.tryParse(entries[key]!) == null ||
                int.parse(entries[key]!) < 0)) {
          issue(path, 'Invalid external-data $key');
        }
      }
    }
  }

  void graph(GraphProto g, String path, Map<String, int> outerTypes) {
    final types = Map<String, int>.of(outerTypes);
    final shapes = <String, List<int>>{};
    for (final vi in [...g.input, ...g.output, ...g.valueInfo]) {
      if (vi.hasType() && !vi.type.hasTensorType()) {
        issue('$path/${vi.name}', 'Only tensor values are supported');
      }
      final type = vi.type.tensorType.elemType;
      if (vi.type.tensorType.hasShape()) {
        shapes[vi.name] = [
          for (final d in vi.type.tensorType.shape.dim)
            d.hasDimValue() ? d.dimValue.toInt() : -1
        ];
      }
      dtype(type, '$path/${vi.name}');
      if (type != 0) types[vi.name] = type;
    }
    for (final t in g.initializer) {
      tensor(t, '$path/${t.name}');
      types[t.name] = t.dataType;
      shapes[t.name] = t.dims.map((d) => d.toInt()).toList();
    }
    if (g.sparseInitializer.isNotEmpty) {
      issue(path, 'Sparse initializers are unsupported');
    }
    for (var ni = 0; ni < g.node.length; ni++) {
      final node = g.node[ni], domain = canonicalOnnxDomain(g.node[ni].domain);
      final version = imports[domain] ?? (legacy ? 23 : 0);
      final np =
          '$path/${node.name.isEmpty ? '#$ni' : node.name} ($domain:${node.opType}@$version)';
      operators.update(
        '$domain:${node.opType}@$version',
        (v) => v + 1,
        ifAbsent: () => 1,
      );
      final schema = resolveOnnxSchema(domain, node.opType, version);
      if (node.overload.isNotEmpty) issue(np, 'Function overloads are unsupported');
      if (version == 0) issue(np, 'Domain has no opset import');
      if (schema == null || node.opType.startsWith('_')) {
        issue(np, 'Operator/domain/opset is not implemented');
      }
      final attrs = {for (final a in node.attribute) a.name: a};
      if (attrs.length != node.attribute.length) {
        issue(np, 'Duplicate attributes');
      }
      if (schema != null && !legacy) {
        final allowed = schema['attributes'] as Map<String, dynamic>;
        for (final a in node.attribute) {
          final spec = allowed[a.name] as List?;
          if (spec == null) {
            issue(np, 'Unknown attribute "${a.name}" for selected schema');
          } else if (a.hasType() && a.type.value != spec[0]) {
            issue(np, 'Wrong type for attribute "${a.name}"');
          }
          if (spec != null &&
              !(schema['handledAttributes'] as List).contains(a.name)) {
            final Object? value = switch (spec[0]) {
              1 => a.f,
              2 => a.i.toInt(),
              3 => utf8.decode(a.s),
              6 => a.floats.toList(),
              7 => a.ints.map((i) => i.toInt()).toList(),
              8 => a.strings.map(utf8.decode).toList(),
              _ => null,
            };
            if (value == null || jsonEncode(value) != jsonEncode(spec[2])) {
              issue(np,
                  'Attribute "${a.name}" is defined by ONNX but this kernel does not implement the requested value');
            }
          }
        }
        for (final entry in allowed.entries) {
          if (entry.value[1] == true && !attrs.containsKey(entry.key)) {
            issue(np, 'Missing required attribute "${entry.key}"');
          }
        }
        for (final pair in [
          ('Inputs', node.input.length),
          ('Outputs', node.output.length),
        ]) {
          if (pair.$2 < (schema['min${pair.$1}'] as int) ||
              pair.$2 > (schema['max${pair.$1}'] as int)) {
            issue(np, 'Invalid number of ${pair.$1.toLowerCase()}');
          }
        }
        for (final index in schema['requiredInputs'] as List) {
          if (index >= node.input.length || node.input[index as int].isEmpty) {
            issue(np, 'Missing required input $index');
          }
        }
        const typeNames = {
          1: 'float',
          2: 'uint8',
          3: 'int8',
          6: 'int32',
          7: 'int64',
          9: 'bool',
          10: 'float16',
          11: 'double',
          16: 'bfloat16',
          21: 'uint4',
          22: 'int4'
        };
        final parameters = schema['inputTypes'] as List;
        for (var i = 0; i < node.input.length; i++) {
          if (node.input[i].isEmpty || parameters.isEmpty) continue;
          final p = i < parameters.length ? parameters[i] : parameters.last;
          if (i >= parameters.length && p[1] != 2) continue;
          final type = types[node.input[i]];
          if (type != null &&
              !(p[0] as List).contains('tensor(${typeNames[type]})')) {
            issue(np, 'Input $i type $type violates the selected ONNX schema');
          }
        }
      }
      int attrInt(String name, int fallback) =>
          attrs[name]?.i.toInt() ?? fallback;
      bool input(int index) =>
          node.input.length > index && node.input[index].isNotEmpty;
      bool output(int index) =>
          node.output.length > index && node.output[index].isNotEmpty;
      List<int>? shape(int index) =>
          input(index) ? shapes[node.input[index]] : null;
      if (node.opType == 'Split' &&
          attrs.containsKey('num_outputs') &&
          attrInt('num_outputs', 0) != node.output.length) {
        issue(np, 'num_outputs must match the declared output count');
      }
      if (domain.isEmpty &&
          const {
            'Attention',
            'RMSNormalization',
            'RotaryEmbedding',
          }.contains(node.opType)) {
        final x = shape(0);
        if (node.opType == 'RMSNormalization' && x != null) {
          final axis = attrInt('axis', -1);
          if (axis < -x.length || axis >= x.length)
            issue(np, 'axis is outside the input rank');
          final scale = shape(1);
          if (scale != null) {
            if (scale.length > x.length) {
              issue(np, 'Scale rank cannot broadcast to X');
            } else {
              for (var i = 0; i < scale.length; i++) {
                final d = x[x.length - scale.length + i];
                if (scale[i] >= 0 && d >= 0 && scale[i] != 1 && scale[i] != d)
                  issue(np, 'Scale shape cannot broadcast to X');
              }
            }
          }
        }
        if (node.opType != 'RMSNormalization' &&
            x != null &&
            x.length != 3 &&
            x.length != 4) issue(np, 'Input rank must be 3 or 4');
        if (node.opType == 'RotaryEmbedding') {
          final dim = attrInt('rotary_embedding_dim', 0);
          if (dim < 0 || dim.isOdd)
            issue(np, 'rotary_embedding_dim must be nonnegative and even');
          if (x?.length == 3 && attrInt('num_heads', 0) <= 0)
            issue(np, 'Rank-3 RotaryEmbedding requires positive num_heads');
          final cos = shape(1), sin = shape(2), positions = shape(3);
          final rank = input(3) ? 2 : 3;
          if ((cos != null && cos.length != rank) ||
              (sin != null && sin.length != rank))
            issue(np, 'Cosine/sine caches must have rank $rank');
          if (positions != null && positions.length != 2)
            issue(np, 'position_ids must have rank 2');
          if (cos != null &&
              sin != null &&
              jsonEncode(cos) != jsonEncode(sin) &&
              !cos.contains(-1) &&
              !sin.contains(-1))
            issue(np, 'Cosine and sine cache shapes differ');
        }
        if (node.opType == 'Attention') {
          final k = shape(1), v = shape(2);
          if (x != null &&
              ((k != null && k.length != x.length) ||
                  (v != null && v.length != x.length)))
            issue(np, 'Q/K/V ranks must match');
          if (x?.length == 3 &&
              (attrInt('q_num_heads', 0) <= 0 ||
                  attrInt('kv_num_heads', 0) <= 0))
            issue(np, 'Rank-3 Attention requires positive head counts');
          final qh = x?.length == 4 ? x![1] : attrInt('q_num_heads', 0);
          final kh = k?.length == 4 ? k![1] : attrInt('kv_num_heads', 0);
          if (qh > 0 && kh > 0 && qh % kh != 0)
            issue(np, 'Query heads must be divisible by KV heads');
          for (final name in ['scale', 'softcap']) {
            final value = attrs[name]?.f;
            if (value != null && (!value.isFinite || value < 0))
              issue(np, '$name must be finite and nonnegative');
          }
        }
        for (final name in node.output.where((s) => s.isNotEmpty)) {
          if (types[name] != null && !const {1,10}.contains(types[name]))
            issue(np,
                'Output "$name" requires unsupported precision ${types[name]}');
        }
        // Tensor storage currently widens fp16 and double. Do not silently
        // advertise precision semantics that these standard kernels cannot honor.
        for (var i = 0; i < node.input.length; i++) {
          if (!input(i)) continue;
          final type = types[node.input[i]];
          final isPosition = node.opType == 'RotaryEmbedding' && i == 3;
          final isMask = node.opType == 'Attention' && i == 3;
          if (type == null) {
            issue(
              np,
              'Input $i type unknown; must be validated at execution',
              warning: true,
            );
          } else if (!(isPosition
              ? type == 7
              : isMask
                  ? type == 1 || type == 9 || type == 10
                  : type == 1 || type == 10)) {
            issue(
              np,
              'Input $i type $type is unsupported by the standard FLOAT kernel',
            );
          }
        }
        if (node.opType == 'Attention') {
          if (version != 23 && version != 24 && version != 25) {
            issue(np, 'Attention currently implements opsets 23, 24 and 25 only');
          }
          if (input(6)) issue(np, 'nonpad_kv_seqlen is not yet implemented');
          if (![0, 1].contains(attrInt('softmax_precision', 0))) {
            issue(np, 'Only FLOAT softmax_precision is supported');
          }
          if (![0, 1].contains(attrInt('is_causal', 0))) {
            issue(np, 'is_causal must be 0 or 1');
          }
          if (attrInt('qk_matmul_output_mode', 0) < 0 ||
              attrInt('qk_matmul_output_mode', 0) > 3) {
            issue(np, 'Invalid qk_matmul_output_mode');
          }
          if (input(4) != input(5) || output(1) != output(2)) {
            issue(np, 'Past and present KV tensors must be paired');
          }
        }
        if (node.opType == 'RMSNormalization' &&
            attrInt('stash_type', 1) != 1) {
          issue(np, 'Only stash_type=1 is supported');
        }
        if (node.opType == 'RotaryEmbedding' &&
            ![0, 1].contains(attrInt('interleaved', 0))) {
          issue(np, 'interleaved must be 0 or 1');
        }
      }
      if (domain == 'com.microsoft' && node.opType == 'MatMulNBits') {
        if (attrInt('bits', 4) != 4) issue(np, 'Only bits=4 is supported');
        final block = attrInt('block_size', 0);
        if (block < 16 || (block & (block - 1)) != 0) {
          issue(np, 'block_size must be a power of two >= 16');
        }
        if (attrInt('K', 0) <= 0 || attrInt('N', 0) <= 0) {
          issue(np, 'K and N must be positive');
        }
        if (input(4)) issue(np, 'g_idx is not supported');
      }
      for (final a in node.attribute) {
        if (a.hasT()) tensor(a.t, '$np/${a.name}');
        for (final t in a.tensors) {
          tensor(t, '$np/${a.name}');
        }
        if (a.hasG()) graph(a.g, '$np/${a.name}', types);
        for (final sub in a.graphs) {
          graph(sub, '$np/${a.name}', types);
        }
      }
      // Preserve known original element types across common producer chains so
      // widening in Tensor storage cannot hide a fp16/double standard-op input.
      int? produced;
      if (node.opType == 'Constant' && attrs['value']?.hasT() == true) {
        produced = attrs['value']!.t.dataType;
      } else if (node.opType == 'Cast') {
        produced = attrInt('to', 0);
        dtype(produced, np);
      } else if (const {'Shape', 'Size', 'ArgMax', 'ArgMin', 'NonZero'}
          .contains(node.opType)) {
        produced = 7;
      } else if (const {
        'Equal',
        'Less',
        'Greater',
        'LessOrEqual',
        'GreaterOrEqual',
        'And',
        'Or',
        'Not',
        'Xor',
        'IsNaN',
        'IsInf'
      }.contains(node.opType)) {
        produced = 9;
      } else if (const {
            'Identity',
            'Add',
            'Sub',
            'Mul',
            'Div',
            'MatMul',
            'Gemm',
            'Reshape',
            'Transpose',
            'Squeeze',
            'Unsqueeze',
            'Concat',
            'Gather',
            'Slice',
            'Expand',
            'Softmax',
            'LogSoftmax',
            'RMSNormalization',
            'RotaryEmbedding',
            'Attention',
            'MatMulNBits'
          }.contains(node.opType) &&
          node.input.isNotEmpty) {
        produced = types[node.input.first];
      }
      if (produced != null &&
          node.output.isNotEmpty &&
          node.output.first.isNotEmpty) {
        types.putIfAbsent(node.output.first, () => produced!);
      }
    }
  }

  graph(model.graph, 'graph', {});
  if (model.functions.isNotEmpty) {
    issue(
      'model.functions',
      'Registered kernels take precedence over bundled function fallbacks. Unregistered function calls must be inlined before loading',
      warning: true,
    );
  }
  issue(
    'model',
    'Structural audit is not full shape inference; data-dependent constraints and legacy kernel attribute semantics still require execution tests',
    warning: true,
  );
  final deferred = externalGatherNames(model.graph);
  final deferredBytes = model.graph.initializer.where((t) => deferred.contains(t.name)).fold(0,
      (int a, t) => a + t.dims.fold(1, (int n, d) => n * d.toInt()) * 4);
  return OnnxCompatibilityReport(
    List.unmodifiable(issues),
    Map.unmodifiable(operators),
    bytes,
    external,
    deferredBytes,
  );
}
