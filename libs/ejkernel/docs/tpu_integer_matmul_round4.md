# TPU integer matmul optimization — round 4

## Large grouped-prefill finding

On the v5p/JAX 0.10.0 development host, the large native INT4 **ragged** dot is not a useful hardware ceiling. It was dramatically slower than the exact INT8 alternative. This does not contradict the faster native INT4 **dense** dot measured in round 3.

Qwen4-Exp TP4 ExpertTensor retains a local static buffer of 81920 routed rows for 8192 tokens × top-10 routing. Each chip owns 128 of 512 experts. Balanced routing gives 20480 active local rows; the remaining rows are padding/nonlocal assignments. ExpertTensor does not divide the K/N dimensions by TP4.

Device-module medians across three shuffled captures, milliseconds:

| Local shape E/M/K/N | Raw BF16 | Raw INT4 | Raw INT8 | Complete W4A4 XLA | Complete W4A4 Pallas |
|---|---:|---:|---:|---:|---:|
| 128/81920/2560/1280 | 1.584–1.585 | 270.646–270.652 | 1.381–1.383 | 270.340–270.343 | 2.415–2.420 |
| 128/81920/640/2560 | 1.483–1.485 | 240.035–240.036 | 1.547–1.550 | 240.224–240.227 | 2.358–2.360 |

The raw diagnostics omit quantization/scaling and do not represent equal-quality complete operators. Runtime operands and group sizes were passed to compiled executables. The profiler parser verifies 16 invocations per variant per capture and does not sum parallel cores.

An exact XLA arithmetic-widening control retained the INT4 quantization grid but widened codes to INT8 before the dot. Its complete gate/up and down times were approximately 2.42 and 2.03 ms. Full-array comparisons established equality between the old XLA W4A4 output, widened control and Pallas output for the tested matrices. Widening is not a change to eight-bit activation quantization.

Smaller/padded and compact controls also showed the native INT4 regression:

| Static M / active M | Gate/up XLA W4A4 → Pallas | Down XLA W4A4 → Pallas |
|---|---:|---:|
| 1280 / 320 | 5.88 → 0.227 ms | 11.56 → 0.201 ms |
| 10240 / 2560 | 38.67 → 0.477 ms | 35.96 → 0.443 ms |
| 20480 / 20480 | 67.77 → 1.183 ms | 60.18 → 1.078 ms |

## Integrated policy

The existing EasyDeL `_channelwise_grouped_platform` selector now additionally selects Pallas for W4A4 with `1280 <= M <= 81920`, **only** on the same two measured expert matrix families, BF16 activations, signed INT4 weights and v5p. The existing decode range `0 < M <= 128` is unchanged. A16 remains decode-only, W8A8 is unchanged, and forced-XLA is honored. Other shapes/hardware remain unchanged.

The Pallas path keeps weights packed in INT4 in HBM and performs INT8 arithmetic with INT32 accumulation. It preserves the four-bit activation grid and existing XLA-reference surrogate autodiff. This is a workaround for the measured ragged INT4 lowering, not a native INT4 throughput claim.

The XLA implementation now also uses exact INT8 arithmetic on the same v5p W4A4 prefill families. Thus forcing XLA no longer forces the slow native INT4 primitive. Persistent codes and the activation quantization grid remain four-bit; XLA may materialize wider temporary weights. The public XLA API measured **2.425–2.429 ms gate/up** and **2.034–2.035 ms down** after this change, versus approximately 270/240 ms before. This heuristic uses the process default backend and chip identity during tracing; cross-target compilation/export is not promised to select the optimal path. Trace/rebuild on the intended device for the measured policy.

## End-to-end result

The Qwen TP4 ExpertTensor run completed at configured context262144, CC8, actual ISL1024/OSL512:

| Metric | Before prefill fix | After |
|---|---:|---:|
| Mean TTFT | 33.83 s | **10.06 s** |
| Aggregate end-to-end TPS | 78.3 | **143.7** |
| Decode TPS on every stream | 27.78 | **27.79** |
| Allocated memory/device | 30.0 GiB | **30.0 GiB** |

All 4096 timed output tokens exactly match the preserved pre-change artifact; all eight warmup/timed 512-token outputs also match. The decode difference is not claimed as a meaningful speedup. The measured improvement is prefill latency. These are short live prompts under a large configured context, not 262K-token occupancy measurements.

Evidence: `/dev/shm/qwen4_w4a4_prefill_pallas_r4.log`, `qwen4_integer_w4a4_before_prefill_r4_tokens.json`, `qwen4_integer_w4a4_tokens.json`.

## Verification

- Four failing dispatch regressions preceded the policy change; dispatch/fused-MoE CPU gate: **27 passed**.
- Two failing actual-ragged-primitive dtype checks preceded XLA widening; lowering/modes/JIT-AD CPU gate: **42 passed**.
- Additional CPU compatibility/optimizer gate: **74 passed, 4 TPU-only skips**; configuration/dispatch/expert gate: **63 passed**. These overlap and are not summed as unique coverage.
- Initial large TPU/fused/AD gate: **23 passed**.
- Actual public XLA and Pallas large-shape full-output checks, plus a selected-family compiled JVP/VJP test against analytical represented-weight derivatives: **16 passed**. Tests include signed codes, balanced/single/empty routes, NaN padding and zero tail cotangents.
- Final current-source distributed-MoE/prefill-AD/dense-Pallas/grouped-integer TPU gate: **31 passed** in 118.97 s. Current production `compileall` and `git diff --check` passed.
- Independent read-only reviews found no concrete numerical/AD defect; the default-backend policy limitation above was recorded.

## Tile sweep and remaining work

The balanced M81920 prefill sweep checked exact full outputs at M tiles32/64/128/256. Explicit M256 measured approximately **2.20 ms gate/up and 2.005 ms down**, versus M32's 2.42/2.36 ms. This is available through the existing explicit `tiling=(256,K,N//2)` option. It was **not** made an automatic default: the sweep does not establish a broadly optimal tile for different route distributions. The end-to-end result above uses the existing M32 default.

The core Pallas GMM was about 0.85 ms inside a 2.42 ms complete gate/up operator. Quantization and external scale/mask operations remain material costs. A plan for a separate optional output-scale epilogue was reviewed, but not implemented; it must preserve ordered FP32 multiplies, K accumulation, partial-row ownership, padding/nonfinite behavior and the existing custom JVP. No future speedup is assumed.

## Evidence

Remote captures: `/dev/shm/grouped_prefill_r4`, `grouped_prefill_widen_r4`, and `grouped_prefill_{1280_320,10240_2560,20480_20480}_r4`.

Downloaded trace/log archive: `.xerxes-scratch/quant-kernel-work/results/round4/quant-prefill-evidence.tgz`.
SHA256: `c7ab6141c7772f9854f0f932f2521a71ec1f12efceef80fe8cc2155c96331f0e` (matched on remote and local).

No universal hardware-limit attainment or cross-mode quality parity is claimed. The dense gap described in round 3 remains separate from this grouped-prefill improvement.
