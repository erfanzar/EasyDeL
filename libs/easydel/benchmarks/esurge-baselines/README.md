# eSurge benchmark baselines

`bench_esurge.py` JSON outputs used to gate the 2026-07 eSurge modularization
(23 commits over `16b3f2127`). Compare `profile_by_total_tokens` bucket-by-
bucket, never aggregate tokens/sec alone.

## 2026-07-10 — refactor performance parity (config A)

Host: single-host TPU v5p-8 (4 devices), model Qwen/Qwen3.5-4B (local HF
snapshot), `--max-model-len 8192 --num-prompts 32 --prompt-len 1024
--output-len 256 --max-num-seqs 32 --max-num-batched-tokens 4096
--hbm-utilization 0.8 --page-size 32 --seed 0 --warmups 1 --trials 2`,
`JAX_PLATFORMS=tpu,cpu` (the script's tpu-only default breaks spectrax's
CPU-seeded Rngs).

- `BASELINE_A_mixed_1024x256.json` — pre-refactor `16b3f2127`: 1860.1 tok/s
- `HEAD_A_mixed_1024x256.json` — post-refactor `416d3aedd`: 1878.8 tok/s (+1.0%)

Structural fields identical in every bucket (steps, bucket ladders,
greedy_argmax_fastpath count — 245/245 in the hot decode bucket). Hot-bucket
timings within tolerance (decode bucket 32: wallclock +0.6%, forward +0.8%,
prep −1.3%). Verdict: performance-neutral.

Still open on hardware: multi-host pod validation of the coordination="zmq"
plane (v5p-8 is one process; needs v5p-16+), decode-heavy config B, PP-sharded
run, and the broadcast-skip gain measurement.

## 2026-07-10 — request-plane phases P0-P5 (config A, same command/host)

- `PLANE_A_mixed_1024x256.json` — post-plane `d1995e594`: 1893.5 tok/s
  (+0.8% vs post-refactor 1878.8, +1.8% vs pre-refactor 1860.1)

Bucket ladders and step counts identical (hot decode bucket 32: 245 steps,
greedy fastpath 245/245 in both); hot-bucket wallclock 1.127s vs 1.160s
(−2.8%), prefill buckets byte-equal timings. Single-host builds no plane
object, so this run measures the added per-step checks (one getattr on the
emit path, forks on submit/abort): performance-neutral.

Unmeasured on hardware: multi-host plane throughput and the deleted
broadcast_one_to_all gain (needs v5p-16+), the flip branch's trainer path
(pod GRPO parity gate), DP-replica aggregate throughput scaling.

## 2026-07-10 — decode host-path optimization (configs A + B)

Config B (decode-heavy): `--prompt-len 128 --output-len 512`, otherwise
identical command/host to config A.

- `BASE_B_decode_128x512.json` — pre-optimization: 4685.4 tok/s
- `OPT2_B_decode_128x512.json` — post-optimization: 4967.7 tok/s (+6.0%)
- `OPT_A_mixed_1024x256.json` — config A post-optimization: 1919.0 tok/s
  (+1.3% over `PLANE_A`'s 1893.5)

Two host-dispatch caches, byte-identical outputs in every run
(`per_request_generated` equal, greedy fastpath 245/245 resp. 511/511):
greedy-argmax device transfers cached by content (724 -> 139 us/step) and
small host-payload slots content-keyed so unchanged metadata skips its
device_put (prep_put 657 -> 436 us/step). Decode-bucket wallclock
4471 -> 3678 us/step.

Finding for the next round: after these cuts the decode step is
DEVICE-bound (~3.7 ms effective; sync-mode logits wait 5.25 ms/step),
~3-4x above the weight-bandwidth roofline for a 4B on 4 v5p chips —
further host-side work buys nothing until the device step (hybrid GDR
kernels / ragged paged attention / LM head) is profiled and improved.
The remaining measured host costs per decode step: compiled-call arg
processing ~1.8 ms (scales with ~550 pytree leaves, C++ fastpath floor),
prep 0.8 ms, scheduler gap 0.4 ms.
