# Benchmark log

Workload: `all-MiniLM-L6-v2.onnx` (BERT-6L-384d), batch 1, seq 32, deterministic
token ids, `dart run tool/bench.dart <model> --seq 32 --iters 5`, min wall time.
Machine: Apple Silicon (this repo's dev machine), Dart AOT-less `dart run` (JIT).

Native ORT reference on identical inputs (`onnxruntime` 1.27.0 CPU):
**min 16.0 ms single-thread, 6.8 ms multi-thread.**

| Date | Change | min wall | vs ORT 1-thread | Top ops |
|---|---|---|---|---|
| 2026-07-17 | B0 baseline (pre-optimization) | 775.3 ms | 48× | MatMul 81.4%, Add 5.8%, ReduceMean 3.5% |
| 2026-07-17 | B1: monomorphic+suffix-broadcast elementwise, Gemm transB prepack | 518.4 ms | 32× | MatMul 85.5%, ReduceMean 4.7%, Erf 2.7% |
| 2026-07-17 | B2: packed 4-row Float32x4 SIMD GEMM kernel | 142.5 ms | 8.9× | MatMul 46.9%, ReduceMean 16.9%, Erf 9.5% |
| 2026-07-17 | B3: constant folding; Transpose/ReduceMean fast paths; row-scalar broadcast (LayerNorm), Pow/Erf direct loops | 107.4 ms | 6.7× | MatMul 84.4%, Erf 6.0%, Add 2.4% |
| 2026-07-17 | + GELU/SDPA fusion (quiet machine, load ≈ 4) | 66.2 ms | 4.1× | MatMul 83.2%, _FusedGelu 8.0%, _FusedSDPA 3.0% |
| 2026-07-17 | + isolate pool, 4 workers (`runAsync`, single warmed run) | ≈42 ms | 2.6× | — |

| 2026-07-17 | register-blocked 4×8 GEMM microkernel (accumulators in locals) | 67.3 ms | 4.2× | — |
| 2026-07-17 | + isolate pool, 4 workers (min of 15) | **32.6 ms** | **2.0×** | — |

Isolate pool scaling on MiniLM (bitwise-identical outputs): the 4-worker
run is 2.0× off single-threaded native ORT and 4.8× off ORT's own
multi-threaded 6.8 ms. Small-m transformer GEMMs saw little from the
register-blocked kernel (sync ≈ unchanged); large conv GEMMs did —
see the vision table.

Caveat: this machine runs other dev workloads; min-of-15-iters is the robust
number, means can inflate 2–3× under contention. Rows above marked "quiet
machine" were measured at load ≈ 4; the earlier rows at load 10–30.

## MiniLM direct-GEMM kernel (Windows, 2026-10-22)

Same model and seq length on the Windows dev box (32 logical cores, `dart run`
JIT, machine load ~32%). Before/after uses `tool/bench.dart --seq 32`; the
direct path reads B through a `Float32x4List` view and skips the per-call
O(k*n) panel pack. Accumulation order is unchanged (bitwise-identical output,
confirmed against ORT: cosine 1.0, max abs diff 3.6e-6).

| Change | min wall | mean | Top ops |
|---|---|---|---|
| packed-panel GEMM (before) | 54.0 ms | 56.8 ms | MatMul 85.0% |
| direct 4x16 GEMM, no B pack copy | **42.6 ms** | 45.4 ms | MatMul 81.7% |
| + isolate pool, 4 workers | **21.0 ms** | 26.5 ms | MatMul 58.9% |

The direct kernel is ~21% lower min wall end-to-end (microbenchmarks on the
same process showed 15-32% lower GEMM time). Native ORT on this box:
6.2 ms single-thread, 1.7 ms default multi-thread. The Windows box is noisy;
min-of-20 is the robust number.

## Maia3-5M chess transformer (batch 1, min of 25, cosine-1.0 throughout)

| Change | min wall |
|---|---|
| baseline (register-blocked GEMM era) | 335 ms |
| + batch-collapse for shared-weight MatMul (`[64,1,k]@[k,n]` was re-packing the weight 64×, m=1) and SIMD row-dot einsum kernels | 135 ms |
| + isolate pool, 4 workers | 102 ms |

For engine throughput, batch positions: `tokens` accepts `[N, 64, 96]`, and
GEMM efficiency rises with N (m = 64·N rows per matmul).

## Kokoro-82M TTS (16 phonemes -> 1.5 s of 24 kHz audio, quiet machine)

| Change | wall |
|---|---|
| baseline (1-D convs on the generic N-D path) | 398 s |
| Conv1D routed through im2col+SIMD GEMM | 11.5 s |
| odometer general-broadcast path (Mul 2.65 -> 1.15 s) | 9.8 s |
| GEMM-backed 1-D ConvTranspose (overlap-add off the profile) | 9.3 s |
| + isolate pool with 1-D conv fan-out (bitwise-identical) | **7.3 s** |

Remaining profile: Conv 69% (genuine GEMM work, ~15-20 GMAC per run).

## GEMM cache-blocking (large-k matmuls / pointwise convs)

Controlled A/B (old single-panel vs new column-panel kernel, back-to-back on
the same machine so load cancels), ECAPA-TDNN (81 MB, dominated by
`3072×3072` and `1024×1024` pointwise convs):

| kernel | ECAPA min wall |
|---|---|
| single-panel (packs all of B once, re-streams it per A-row tile) | 20.5 s |
| column-panel (keeps a ~192 KB k×panel slice of B in L2) | 9.0 s — **2.3×** |

Accumulation order is unchanged, so results stay bitwise identical (pool
bitwise tests + all live models pass). Small-k matmuls (transformers) are
unaffected — one panel covers all columns, so the path is identical to
before. `Softmax`/`LayerNorm` inner loops also switched from the
dtype-branching `getD()` to direct float-buffer access.

## Vision models (224×224, batch 1, cosine-1.0 parity vs ORT throughout)

| Date | Change | MobileNetV2 | ResNet18 |
|---|---|---|---|
| 2026-07-17 | A1: naive direct conv | 4612 ms | 11520 ms |
| 2026-07-17 | A2: im2col + SIMD GEMM conv (depthwise stays direct) | 655 ms | 625 ms |
| 2026-07-17 | + padded-buffer branchless depthwise (quiet machine) | 260 ms | 422 ms |
| 2026-07-17 | + register-blocked 4×8 GEMM microkernel | 230 ms | 332 ms |

ORT reference: MobileNetV2 15.4 ms (1-thread) / 5.1 ms (default);
ResNet18 47.9 ms / 14.9 ms. Gap ≈ 15× / 6.9× single-threaded; Conv is
~88% of both profiles.

Per-op shares come from `ExecutionProfile` (`--iters` accumulate). Regenerate
the ORT reference with the snippet in `git log` for this file or ad-hoc via
`.venv/bin/python` + `onnxruntime`.

## LaMa inpainting, 1024² (`lama_manga_large_1024.onnx`, 196 MB)

Windows dev box, 32 logical cores, `dart run tool/bench.dart <model> --iters 1`
(min of one timed run after warmup). Peak RSS is the whole process as reported
by `ProcessInfo.maxRss` (all isolate workers live in it); the bench prints it
after the profile. Inputs are synthesized (`image [1,3,1024,1024]`,
`mask [1,1,1024,1024]`), so the numbers measure the full graph, not
pre/post-processing.

| Date | Mode | wall | Conv | MatMul | ConvTranspose | Pad | peak RSS |
|---|---|---|---|---|---|---|---|
| 2026-10-20 | `run()` baseline (full im2col workspace) | 177.0 s | 151.0 s | 10.3 s | 6.1 s | 5.2 s | ~15 GB |
| 2026-10-20 | `run()` + chunked im2col (32 MiB cap) | 108.7 s | 89.4 s | 9.7 s | 5.9 s | 0.4 s | ~1.3 GB |
| 2026-10-20 | `run()`, 3×3 Winograd (busy box below) | 85.2 s | 65.9 s | — | — | — | 2.6 GB |
| 2026-10-20 | `runAsync()`, 16 w, `poolConv`, replicated kernels | 27.9 s | 19.4 s | 3.8 s | 1.2 s | 0.5 s | ~7.0 GB |
| 2026-10-20 | `runAsync()`, 16 w, `poolConv` im2col A/B (busy box) | 41.8 s | 29.8 s | 5.3 s | 1.7 s | 0.8 s | 7.0 GB |
| 2026-10-20 | + Winograd, replicated kernels (busy box) | 28.4 s | 17.1 s | 4.5 s | 1.5 s | 0.8 s | 7.5 GB |
| 2026-10-20 | + on-demand kernels (busy box) | **22.2 s** | 14.1 s | 3.4 s | 1.3 s | 0.5 s | **3.8 GB** |

The last three rows are back-to-back A/B runs while another process held
~50% of the CPU, so their absolute times are conservative; they isolate the
two changes: 3×3 Winograd took the pooled wall from 41.8 s to 28.4 s, then
shipping conv kernels per request (instead of replicating all 222 of them to
16 workers) took it to 22.2 s and cut peak RSS from 7.5 GB to 3.8 GB. Run
with `-Donnx_no_winograd=true` to reproduce the im2col side. Errors
vs the im2col path are at FP32 rounding level (~1e-6), not bitwise: `run()`
and `runAsync()` make the same Winograd choice, so they remain bitwise-equal
to each other.

What moved the needle:

- **Bounded im2col.** The 1024² 384→128 3×3 convs need
  `C·kh·kw·oh·ow ≈ 14 GB` for a full column matrix. `opConv` chunks over
  output rows (32 MiB panel), and the graph only keeps a workspace when it is
  under 8 M floats.
- **Winograd F(2×2,3×3).** All 3×3 `pads=0/1` convs with ≥1024 output pixels
  run Winograd, locally and in row bands. Plans are rebuilt per conv node
  (each kernel is distinct) but the 16 transformed-kernel planes and the
  banded scratch are reused by shape, so the pool does not churn GBs through
  the GC. ≈2× fewer multiplies per output.
- **2 MiB GEMM panels + small-M kernel.** Conv `k` is large, so the default
  192 KiB B-panel re-read the weights hundreds of times per call; 2 MiB keeps
  the panel in L2. The final 3-channel 7×7 conv (`m=3`) now streams B once
  instead of packing a panel for three rows (21.7 s → 1.6 s pooled).
- **`Pad` row copy.** Reflect/edge padding on 1024² tensors now maps indices
  once per axis and copies whole rows (`setRange`); 10.7 s → 0.45 s.
- **Isolate fan-out.** Conv is split by output rows, with each worker receiving
  only its input-row band (`TransferableTypedData`); `ConvTranspose` uses the
  same scheme with the transposed-conv halo. Runtime MatMul operands are split
  along A (large) or B's leading batch (FFC twiddles) instead of copying the
  whole activation to every worker. Workers also reuse a persistent im2col
  scratch.
- **Bounded resident set.** Conv kernels travel with each request instead of
  being replicated at `parallelize()` time. A 196 MB model at 16 workers used
  to pin ~3 GB of duplicate weights; the per-call copy costs <1 s over 222
  convs. 32 workers now fit (4.2 GB) but are no faster (23.7 s).
- **Adaptive scope.** Depthwise/grouped convs and `output_shape` ConvTranspose
  stay local; MatMul weights below `minWeightElements` stay local.

At 22.2 s the main isolate is the bottleneck for the remaining non-conv work
(elementwise ≈ 2.4 s, `MatMul` 3.4 s, `ConvTranspose` 1.3 s) and slabs must
still be sliced/stitched there. `Conv` is 14.1 s (~64% of wall); the next
lever is parallelizing large elementwise ops and overlapping their bands with
conv work.



## Native ONNX Runtime backend (optional)

LaMa manga inpainting, input `1x3x1024x1024` + `1x1x1024x1024`, deterministic
synthetic feed, ONNX Runtime 1.30, Windows, machine at ~60% external CPU load.

| Backend | Wall | Peak RSS |
|---|---:|---:|
| ONNX Runtime CPU, 8 intra-op threads | 6.4 s | 4.2 GB |
| ONNX Runtime CUDA (RTX 4070 Ti SUPER) | 0.29 s | 1.3 GB |
| Pure Dart, single-threaded | 113.9 s | - |

CPU ORT vs pure-Dart parity on the same feed: max abs error 4.9e-5, mean
7.4e-10, 0/3,145,728 elements outside tolerance. The earlier 4.8 s ORT CPU
reference was measured on a quieter machine; re-benchmark on the target host.
See `docs/native_backend.md` and `tool/bench_native.dart`.

