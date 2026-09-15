# Transformer compatibility and execution

The runtime audits a graph before decoding its weights. Inspection never opens
external tensor files. A serialized `.onnx` with inline weights is still parsed
in full; this is not a streaming protobuf reader.

```sh
dart run tool/audit_model.dart model.onnx --json
```

The report includes domain/opset-qualified operator counts, structural errors,
unsupported attributes and declared tensor types, an initializer-memory estimate,
and warnings for constraints that cannot be established statically. It walks
subgraphs and tensor attributes. Local functions must be inlined before loading.
Unknown domains cannot fall back to a standard operator with the same name.
Models constructed without opset imports retain legacy behavior with a warning.

The schema snapshot is generated from ONNX 1.22.0 and ORT 1.29.0. Selecting a
schema is not proof of numerical support for every legacy kernel configuration:
data-dependent constraints and incomplete type information still require runtime
validation. Unsupported non-default attributes are rejected when the dispatch
does not consume them. The auditor is deliberately conservative.

## Standard operators

| Operator | Implemented contract |
| --- | --- |
| `Attention` | Standard domain, opsets 23-25, FLOAT. Rank 3/4 Q/K/V, MHA/GQA/MQA, broadcast boolean/additive masks, causal offset with past KV, sliding-window bounds (`left_window_size`/`right_window_size` from opset 25, composed with the causal mask at absolute positions), scale, softcap, optional present KV, all four debug-output modes. Fully masked rows produce zero. |
| `RMSNormalization` | Standard schema 23, FLOAT, `stash_type=1`, suffix reduction from `axis`, broadcast scale. |
| `RotaryEmbedding` | Standard schema 23, FLOAT, correct standard input order, optional positions/pre-indexed caches, partial/interleaved rotation, rank 3/4. Microsoft input order remains separate. |
| `MatMulNBits` | Microsoft schema 1, four bits, packed or floating zero points, optional bias with empty `g_idx` slot. Invalid blocks/shapes are rejected. `g_idx` and other bit widths remain unsupported. |

FP16, BF16 and double precision semantics for the new standard operators are
not implemented; known incompatible input/output types are rejected, rather than
silently treating widened storage as equivalent. Attention opsets 24/25 are also
rejected. Wiring for a real FP16 export (Gemma 4 E2B) is in place: the decoder
reuses fp32 kernels with fp16 rounding at node boundaries, the deferred Gather
tables read rows over the external data file, and the audit passes for all three
graphs (decoder, embedding, vision encoder).

MatMulNBits token decoding reads packed weights directly. Multi-row execution
dequantizes at most 32 output columns per tile, avoiding a full floating K*N
weight copy. Packed zero points respect per-column padding for odd block counts.
Attention uses one score row of scratch unless its debug output is requested.
Past KV is read directly. Requested present KV outputs are immutable snapshots
and still copy history; a persistent cache with reserved capacity is future work.

### Persistent KV cache

`OnnxModel.enablePersistentKv()` switches `Attention` present outputs to views
over a per-node capacity-managed cache instead of fresh immutable snapshots.
Keyed by node (not by name), the cache survives across `run` calls so a decode
loop passes no history on later steps:

```dart
model.enablePersistentKv();
for (final ids in steps) {
  final y = model.run(stepInputs(ids), model.outputNames);
  // stepInputs omits past_key_values.* after the first step; the executor
  // appends to the same buffers seeded on step 0. present.* outputs still
  // work and alias the cache for inspection or checkpointing.
}
```

Cache growth is geometric (double the capacity when full); growing copies only
the resident rows. The returned present tensors are offset views
(`Tensor.cacheView`, with an `offset` into the shared buffer — `getD`/`getI`
resolve positions through the offset, so view and copy callers are
interchangeable). The causal mask in cached steps is applied against the
resident row count, not the zero-length past, keeping incremental decode
equivalent to a full run.

The Gemma 3.2 text-parity harness was rerun with the flag enabled and produced
the identical token (9259) and fingerprint (`logits` cosine 0.99999986, same
per-layer metrics, ~17.4 s/step, ~1.84 GB RSS). Internal tests assert: views
alias the seed buffer, growing moves history, the appended row matches the last
row of a fresh full run, resident rows survive grows unchanged, and a graph-level
stepped decode matches the copy path bit for bit.

Views must only travel through the runtime (past inputs, present outputs, cache
grow); `_roundOutputs` and the attention dispatch skip them so allocation
accounting and mid-graph tensor values stay uncluttered.

## Portable loading and control

```dart
final source = MemoryOnnxDataSource({
  'model.onnx': modelBytes,
  'weights.bin': weightBytes,
});
final control = OnnxExecutionControl(
  maxTensorBytes: 512 * 1024 * 1024,
  deadline: DateTime.now().add(const Duration(minutes: 2)),
  onProgress: (phase, done, total) => print('$phase $done/$total'),
);
final model = await OnnxModel.fromSource(source, 'model.onnx', control: control);
final outputs = await model.runAsync(inputs, ['logits'], control: control);
// A UI cancellation handler can call control.cancel().
model.dispose();
```

`OnnxDataSource` has asynchronous `length(location)` and
`read(location, offset, length)` methods. Implement those methods to connect
browser Blob/IndexedDB/OPFS storage. `FileOnnxDataSource(directory)` in the IO
library implements the same interface for native files. External locations are
relative to the source root; place that root at the model's directory.

Async loading decodes one tensor at a time, including external Constant and
subgraph tensors, instead of prefetching all raw weights. The bytes and synchronous
file APIs remain available, and accept an execution control too.

`maxTensorBytes` is an accounted buffer budget, not a process-RSS or OS allocation
limit. It excludes protobuf storage, allocator overhead, isolate copies and
legacy-kernel scratch. Standard transformer allocations are checked before they
are made; legacy results are checked on return. Kernel cancellation is cooperative:
node boundaries plus checkpoints inside the new transformer kernels. `runAsync`
yields between nodes when control is supplied, but a long synchronous legacy
kernel still requires a worker isolate to keep the UI responsive.

Intermediate values are released after their last consumer, preserving subgraph
captures and explicitly requested outputs. Requesting `['*']` keeps every value
for debugging and disables that memory optimization.

For portable integer inputs, use `Tensor.int64(<int>[...], shape)` instead of
constructing Dart's native `Int64List`. JavaScript storage rejects integers
outside ±(2^53−1); native execution retains signed 64-bit storage.

## Precision

The Gemma 4 E2B export is fp16 throughout, so the runtime's transformer kernels
compute in fp32 and round tensor values to fp16 once at each node boundary
(CastLike/Cast entry nodes round, constant feeds are de-duplicated by exact
value). That reproduces native-ORT fp16 numerics on all three validated graphs
without fp16 storage or kernel variants.

BFLOAT16 and double input tensors are not claimed: the standard kernels validate
FLOAT (fp32) with fp16 boundary rounding, and no operator in the target model
uses bf16 or fp64. `Attention` additionally covers opsets 23-25. Opset 25 only
adds the sliding-window bounds (`left_window_size`/`right_window_size`, both
`-1` = unbounded by default); they compose with `is_causal` at absolute
positions `p = offset + i`, and the persistent-cache path applies them to the
resident rows exactly like a fresh full run. The "in-place KV cache" mentioned
in the Attention-25 notes is deliberately realized as `enablePersistentKv()`
(the exporter itself keeps past/present KV as paired named inputs), so no extra
operator inputs are needed.

## Validation and remaining work

`tool/gen_transformer_fixtures.py` produces deterministic native-ORT CPU fixtures
for the new operators and quantized bias/zero-point variants. The shared fixture
test checks shapes, mixed absolute/relative tolerance (2e-5), and non-finite values.
`test/transformer_runtime_test.dart` exercises audit failures before weight reads,
cancellation, memory accounting, portable loading, and incremental attention.

`tool/gemma_parity.py` generates a bounded ORT oracle for the Gemma 4 E2B export
(three graphs, fp16, opset 24) without expanding the 5.5 GB embedding tables:
external `Gather` data is substituted with just the selected rows via a small
static subgraph evaluator. `tool/gemma_parity.dart` runs the same token stream in
this runtime. Verified at a fixed sequence (prompt `Hello` → token 9259):
embedding output exact, per-layer inputs cosine 0.9999999, logits cosine
0.99999985, roughly 1.85 GB peak RSS and ~18 s/step within a 4 GB budget.
`tool/gemma_vision_parity.py` + `tool/gemma_vision_parity.dart` run a synthetic
patch grid through the vision encoder; every probe of the 16 blocks up to
layer-14 residuals matches ORT at fp16 rounding (cosine ≥ 0.9999), and the final
pooled `image_features` tracks at cosine 0.968 for i.i.d. random patches, which is
fp16 chaos amplification with synthetic inputs, not a structural mismatch.
`tool/gemma_multimodal.py` + `tool/gemma_multimodal.dart` run a real image
(`webpt_workspace/webpt_converter/bin/test.webp` by default) through all three
graphs end to end with the persistent cache: the vision encoder reproduces its
`image_features`, the embedding graph splices those features at the image-token
positions (258880) with `inputs_embeds` at cosine 0.996, and the decoder tracks
the oracle's token stream exactly (prompt `Describe this image in detail.` →
generated tokens `[106, 107, 1]`, `logits` cosine 0.999 at step 0, ~2.85 GB
RSS, ~11 s/step after prefill). The one-graph parity depends on binding
`OnnxModel` loads with `OnnxExecutionControl(maxTensorBytes: ...)` and the
harnesses are not part of CI.

CI runs native tests on Linux, Windows and macOS and portable tests in Chrome.
The portable ORT oracle suite runs under both JavaScript and WebAssembly.

### Emulator measurements (Android API 36, Pixel 7a AVD, 8 GB RAM, 16 GB data)

`.tmp/device_harness` is a minimal Flutter (Android) app that loads both graphs
from `/data/local/tmp/gemma` (pushed via adb) and measures wall time and VmRSS.
It reuses the multimodal oracle bins for a 10-token prompt. `largeHeap` is on.
Results (`/data/local/tmp/gemma` push, debug APK):

| graph          | load  | run (perf)     | VmRSS after |
| -------------- | ----- | -------------- | ----------- |
| vision_encoder | 10.5 s| 4.7 s (grid-4) | 1.07 GB     |
| decoder        | 8.4 s | 20.3 s (prefill 10 tok) | 2.65 GB |

20 s prefill is dominated by the fp16 `per_layer_inputs` mixing; the persistent
KV cache itself is in place, so decode-only steps reuse the resident rows. The
AVD needs `hw.ramSize=8192` (2 GB OOM-kills the decoder load) and
`disk.dataPartition.size=16G` (the 1.4 GB `.onnx.data` plus the 0.34 GB vision
graph exceed the default 6 G). Models and bins push to `/data/local/tmp/gemma`.
