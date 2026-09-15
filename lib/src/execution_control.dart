library;

import 'dart:async';
import 'tensor.dart';

class OnnxCancelledException implements Exception {
  const OnnxCancelledException();
  @override
  String toString() => 'ONNX execution cancelled';
}

class OnnxMemoryLimitException implements Exception {
  final int requiredBytes;
  final int limitBytes;
  const OnnxMemoryLimitException(this.requiredBytes, this.limitBytes);
  @override
  String toString() =>
      'ONNX tensor budget exceeded: $requiredBytes > $limitBytes bytes';
}

/// Cooperative control for one load or run. Cancellation is observed at node
/// boundaries and inside standard attention / quantized matmul. A synchronous
/// call cannot process UI events; use runAsync or a worker isolate for UI work.
///
/// [maxTensorBytes] limits accounted tensor buffers, NOT process RSS. Protobuf,
/// allocator overhead, isolates, and legacy kernel scratch are not included.
/// New transformer kernels check their allocations before allocating; legacy
/// kernel results are checked on return. This is not an OS-level OOM guarantee.
class OnnxExecutionControl {
  final int? maxTensorBytes;
  final DateTime? deadline;
  final void Function(String phase, int completed, int total)? onProgress;
  bool _cancelled = false;
  int _liveBytes = 0;
  int peakAccountedBytes = 0;
  OnnxExecutionControl({this.maxTensorBytes, this.deadline, this.onProgress}) {
    if (maxTensorBytes != null && maxTensorBytes! < 0) {
      throw ArgumentError.value(maxTensorBytes, 'maxTensorBytes');
    }
  }
  void cancel() => _cancelled = true;
  bool get isCancelled => _cancelled;
  void checkpoint() {
    if (_cancelled ||
        (deadline != null && !DateTime.now().isBefore(deadline!))) {
      throw const OnnxCancelledException();
    }
  }

  void checkAdditionalBytes(int bytes) {
    checkpoint();
    final required = _liveBytes + bytes;
    if (required > peakAccountedBytes) peakAccountedBytes = required;
    if (maxTensorBytes != null && required > maxTensorBytes!) {
      throw OnnxMemoryLimitException(required, maxTensorBytes!);
    }
  }

  void account(Iterable<Tensor> tensors) {
    // Deduplicate shared typed lists. Distinct views may be conservatively
    // counted twice: Wasm does not expose stable ByteBuffer wrapper identity.
    final buffers = <Object>{};
    var bytes = 0;
    for (final t in tensors) {
      final buffer = t.storageBuffer;
      if (buffers.add(t.storageIdentity)) bytes += buffer.lengthInBytes;
    }
    _liveBytes = bytes;
    checkAdditionalBytes(0);
  }

  void progress(String phase, int completed, int total) {
    checkpoint();
    onProgress?.call(phase, completed, total);
    checkpoint();
  }
}

final Object _controlKey = Object();
OnnxExecutionControl? get activeOnnxControl =>
    Zone.current[_controlKey] as OnnxExecutionControl?;
T withOnnxControl<T>(OnnxExecutionControl? control, T Function() body) =>
    control == null
        ? body()
        : runZoned(body, zoneValues: {_controlKey: control});
