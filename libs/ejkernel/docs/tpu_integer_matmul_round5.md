# TPU integer matmul — wrap-up after round 5

Optimization work stopped at the user's request to wrap up. Existing WIP remains saved on n_server_spot_m; nothing was committed or pushed. This is a completed optimization pass, not a claim that every shape reaches a physical hardware ceiling.

## Latest integrated kernel improvement

Added optional output_row_scale[M,1] and output_channel_scale[E,1,N] to grouped_matmulv3's integer path. The full K contraction accumulates in INT32, then multiplies row scale followed by channel scale in FP32 before the final output cast. Existing rhs_scale/bias/fused-activation semantics are unchanged; incompatible combinations are rejected. Partial-row ownership uses selection on this path to preserve signed zero and prevent contamination between experts.

A singleton-width row-scale pipeline DMA failed TPU compilation. The implemented version broadcasts row scales to 128 columns before transfer. It still avoids the much larger raw INT32 output and expanded per-row channel-scale arrays. This is not a fully compact row-scale implementation.

The public Pallas channelwise operator selects the epilogue only on v5p for the two measured E128 matrix families, M1280..81920, A4/A8, BF16/FP32 output. Small decode stays on the previous path: measured decode candidates were slightly slower. EasyDeL's existing mode/platform dispatch remains unchanged; W8A8 models still ordinarily select XLA.

### Device-time evidence

Runtime operands, M81920 with 20480 active rows, E128, three randomized captures, 16 calls per variant per capture. Full outputs exactly matched the previous unfused Pallas implementation.

| Complete grouped operator | Previous Pallas | Fused epilogue |
|---|---:|---:|
| W4A4 gate/up K2560,N1280 | 2.417–2.419 ms | 1.835–1.840 ms |
| W4A4 down K640,N2560 | 2.359–2.360 ms | 1.099–1.103 ms |
| W8A8 gate/up | 2.462–2.472 ms | 1.914–1.915 ms |
| W8A8 down | 2.379–2.380 ms | 1.154–1.155 ms |

Compiled temporary-buffer estimates (not peak HBM): gate/up 839,630,496 -> 251,723,712 bytes; down 1,678,425,824 -> 94,437,312 bytes.

A separate current public-API capture confirmed W4A4 Pallas at 1.838–1.844 / 1.100–1.102 ms versus current XLA at 2.425–2.429 / 2.033–2.035 ms for gate/down. These complete operators include quantization and scaling; this is not a raw-dot hardware-ceiling claim.

## Verification

- Seven missing-API RED tests preceded implementation; initial compact DMA failed compilation, then nine raw/compiled-AD tests passed after lane-aligned row-scale storage.
- Two executable temporary-memory budget tests failed before public dispatch integration and passed afterward.
- Found and fixed pre-existing NaN padding behavior: an unused nonfinite expert scale contaminated output padding. Both XLA and unfused Pallas now mask after scaling, and XLA JVP masks its result. Three XLA and three Pallas tests failed before the fix; all six passed afterward, including reverse gradients.
- Integrated TPU gate: 17 passed in 25.96s.
- Final compatibility/large-output/full-range-int8/INT4-tail/compiled-AD TPU gate: 40 passed in 109.14s. Includes accumulated integers greater than 2^24 before FP32 conversion and actual public Pallas prefill JVP/VJP coverage.
- CPU invalid-option and nonfinite-padding gate: 16 passed, 3 TPU-only skips.
- Current production compileall and git diff --check passed.
- Independent review found no concrete correctness defect; its concern about production-dispatch AD coverage was addressed by expanding test_w4_prefill_ad.py to both XLA and Pallas.

Test suites overlap; counts are not summed as unique coverage.

## Serving status and limitations

The last full Qwen W4A4 benchmark is round 4, BEFORE this final epilogue: TP4 ExpertTensor, CC8, configured context262144, actual ISL1024/OSL512, 27.79 decode TPS on every stream, 10.06s mean TTFT, 143.7 aggregate end-to-end TPS, 30.0GiB/device. All 4096 tokens exactly matched the pre-prefill-fix version and warmup/timed passes.

**The full model was not rerun with the round-5 epilogue.** No additional serving TPS or TTFT gain is claimed. The user asked to stop further tuning instead of spending another model-load/benchmark cycle. Likewise, there is no cross-mode quality-equivalence claim, full-262K live-context benchmark, or universal 3x speedup claim.

Remaining optional follow-up: final-source full-model repeat, more route/shape timing coverage, compact row-scale loading, and additional dense tuning. These are not running or promised in the background.

## Evidence locations

- /dev/shm/grouped_epilogue_prefill_r5 and corresponding .log
- /dev/shm/grouped_epilogue_decode_r5 and corresponding .log
- /dev/shm/grouped_epilogue_public_r5 and corresponding .log
- /dev/shm/epilogue_integrated_gate_r5.log
- /dev/shm/epilogue_compat_tpu_r5.log
- Round-4 serving evidence and report remain unchanged.

Source files: grouped_matmulv3/_pallas_impl.py, grouped_matmul_channelwise/_interface.py, grouped_matmul_quant/_channelwise.py. Prototype-only scripts remain in /tmp/quant_ceiling_work and the local scratch directory; they are not production entry points.
