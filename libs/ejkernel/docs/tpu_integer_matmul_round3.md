# TPU integer matmul optimization — round 3 evidence

This is an **in-progress** optimization report, not a hardware-limit claim. Measurements were made on `n_server_spot_m` (TPU v5p, JAX 0.10.0). Existing WIP is preserved; no commits or pushes were made.

## End-to-end result

Qwen4-Exp, physical TP4 with ExpertTensorMode, W4A4 routed experts, BF16 KV, configured context 262144, **actual 1024-token prompts**, CC8 with 512 generated tokens per request:

| Metric | Previous XLA W4A4 | Integrated Pallas decode |
|---|---:|---:|
| Decode tokens/s per stream | 14.20 | **27.50** |
| Aggregate end-to-end tokens/s | 58.6 | **78.1** |
| Mean TTFT | 33.81 s | 33.81 s |
| Timed output tokens | 4096 | 4096 |

All eight output token sequences are identical across these two implementations, and warmup/timed outputs are identical within the new run. This does not establish quality parity with BF16 or W8A8. The new run used 30.0 GiB/device. Its log ended with `BENCH_EXIT_CODE=0`.

Evidence: `/dev/shm/qwen4_integer_w4a4_pallas_decode.log`, `/dev/shm/qwen4_integer_w4a4_round2.log`, and the `qwen4_integer_w4a4{,_before_pallas}_tokens.json` artifacts.

EasyDeL selects this path only for W4A4, BF16 activations, TPU v5p, at most 128 dispatch rows, and local code-bank shapes `[128,2560,1280]` or `[128,640,2560]`. `moe_force_xla_gmm` still overrides it. Other shapes/modes retain XLA. The public ejkernel API still defaults to XLA; explicit `platform='pallas'` now supports A16/A4/A8.

## Device-time grouped measurements

Times below are microseconds, extracted from correlated TPU **XLA Modules** events, taking the maximum across parallel cores for each run—not adding core times or timing host dispatch. Runtime operands include group sizes. The harness uses one active row per selected expert and checks sampled numerical results independently.

| E/M/K/N | Original BF16 hint | Best BF16 measured after tile sweep | Old W4A4 XLA | New W4A4 Pallas | W8A8 XLA | W8A8 Pallas |
|---|---:|---:|---:|---:|---:|---:|
| 128/24/2560/1280 | 126.58 | **84.82** | 398.34 | **84.12** | 96.56 | 91.63 |
| 128/80/640/2560 | 287.44 | **122.03** | 648.24 | **130.75** | 134.92 | 143.73 |

**Important correction:** comparison with the original BF16 hint overstated the relative benefit. W4A4 is roughly tied with, or behind, the stronger BF16 baseline. It is not demonstrated to be 1.5–2x faster than tuned BF16. The serving improvement versus old W4A4 remains real.

A16 full-K tuning also helps, but does not beat the stronger BF16 baseline:

| E/M/K/N | W4A16 old Pallas | W4A16 chosen tiles | W8A16 old Pallas | W8A16 chosen tiles |
|---|---:|---:|---:|---:|
| 128/24/2560/1280 | 215.57 | 109.06 | 209.61 | 110.26 |
| 128/80/640/2560 | 396.47 | 156.26 | 396.58 | 152.27 |

Chosen weight-only defaults apply to those decode matrix families only. They retain M=16, use full K, and bound each RHS buffer to 2 MiB. The fastest first-shape W8A16 trial was 105.61 us with a larger RHS tile; the bounded default instead chooses 110.26 us. These A16 figures are operator measurements, **not new A16 serving results**.

Profiles: `/dev/shm/grouped_integer_public`, `/dev/shm/grouped_weight_only_tiles`. BF16 and quantized variants use the same device/compiler environment. Best observed tiles are not proof of a physical ceiling.

## Implementation and correctness

- Integer LHS/raw integer RHS now uses INT32 dot accumulation. Packed INT4 weights are widened inside VMEM when the arithmetic pair requires it; A4 values remain on their original grid. This path is not a claim of native INT4 arithmetic.
- Public grouped A4/A8 masks unused rows before quantization, pads physical M to a multiple of 32, and applies activation then channel scales after the integer dot.
- Scaled/bias-bearing raw integer-LHS calls are rejected; scales belong outside the raw dot, avoiding fractional-scale truncation.
- Grouped custom JVP keeps frozen arrays explicit and delegates derivatives to the XLA represented-weight surrogate. No backward streaming-memory improvement is claimed.
- Independent review found that the first RHS-budget rule only reduced K and could exceed the bound at very wide N. It now also reduces N; pure-shape regressions cover this.
- A separate dense JVP regression exposed XLA eliding the BF16 activation-dot rounding boundary when compiling primal and tangent together. The activation-only tangent differed from an independent arithmetic reference, while the scale-only tangent did not. An optimization barrier at that existing rounding boundary fixed the regression without changing primal arithmetic or relaxing tolerances.

Verification observed in this work:
- 47 combined TPU grouped, tail, distributed-MoE, and fused-MoE tests passed.
- 26 CPU dispatch/fused/contract tests passed.
- After the weight-only policy change: 24 CPU policy/contract tests and 21 TPU streaming tests passed.
- Dense rounding regression plus persistent-prototype primal/JVP/VJP: 8 TPU tests passed.
- Broader dense numerical/AD and four-mode scale-optimizer gate: 20 TPU tests passed.
- Device-profile parser: 4 tests passed, checking max-not-sum correlation, picosecond conversion, and rejection of missing device metadata.
- `git diff --check` was clean after the code changes; final broad all-mode verification remains outstanding.

## Rejected dense experiment

A new temporary prototype retains activation codes/scales in VMEM once per M tile and streams N tiles with `emit_pipeline`. Full signed-weight-range and half-tie forward checks passed exactly; JVP/VJP checks passed after the reference rounding fix.

However, three shuffled device-profile captures did **not** show a win:
- 2048-square W4A4: best tested persistent tile about 75–77 us versus XLA 49–50 us.
- 4096-square W4A4: about 388–389 us versus XLA 175–176 us.
- 4096-square W8A8: about 594–595 us versus XLA 235–238 us.

Some 4096/M128 tiles exceeded default scoped VMEM and were rejected. The prototype is **not integrated into production dispatch**. Reducing repeated quantization alone did not yield a complete-operator win; scheduling/data movement still needs investigation. Evidence: `/dev/shm/persistent_dense/repeat*`.

A follow-up isolated the large-M VMEM failure: a 256-row quantization tile required 30.14 MiB against the default 16 MiB scoped limit. Quantizing 64 rows at a time into retained integer scratch fixed that test without increasing the limit (8 TPU memory/numerical/AD tests passed). It enabled larger dot tiles, but three new captures still lost to XLA:
- 2048 W4A4: 55.3–56.2 us versus 48.8–48.9 us.
- 4096 W4A4: 211.5–212.2 us versus 174.1–175.1 us.
- 4096 W8A8: 289.1–290.3 us versus 235.8–236.1 us.

The chunked version also remains experimental. Evidence: `/dev/shm/persistent_dense_chunked/repeat*`.

## Scale-map follow-up (serving rerun pending)

A route-distribution sweep found substantial overhead outside the Pallas matmul. In one sampled per-core A16 invocation the module took 56.75 us while the core GMM took 9.59 us; the wrapper included scatter-offload operations. Replacing small-decode `repeat` scale-map construction with cumulative endpoints, bounded row-to-endpoint comparisons and a gather removed those scatter-offload operations. The core GMM remained around 9.77 us in the inspected new invocation.

Separate captures were required for each route distribution: identical compiled functions reused module labels across different runtime route inputs. The initial combined-route capture must **not** be used as a per-route timing table. Isolated captures were verified to contain 16 variants with 24 runs each, repeated three times.

For E=128, M=80, with only 20 valid rows assigned to one/two experts:
- Before the scale-map change: A16 approximately 59–60 us; A4/A8 approximately 61–64 us.
- After: A16 approximately 33–36 us; A4/A8 approximately 34–39 us.
- Tuned BF16 on these skewed cases remained faster, around 23–27 us.

For the 20-active-expert spread case, K2560/N1280, new W4A4 measured 58.27–59.01 us versus BF16 Pallas 71.16–73.52 us; K640/N2560 W4A4 was 50.70–52.23 us versus BF16 46.04–48.12 us. Benefits remain shape/route dependent.

Evidence: `/dev/shm/grouped_routes_isolated` and `/dev/shm/grouped_routes_noscatter`. The BF16 XLA baseline explicitly masks unwritten padding rows so it satisfies the same output contract.

The mapping is now shared by Pallas, explicit XLA modes and legacy W8A8. It preserves dtype, repeated empty endpoints and final-element padding, with the original repeat fallback beyond M=128 or E=1024. Eight CPU mapping tests and 20 TPU Pallas correctness/AD/anchor tests passed before sharing; the shared version passed 52 focused CPU tests, 39 compatibility tests (4 hardware skips), and 14 x64-enabled dtype/fallback tests. Its new XLA TPU performance/verification gate is pending until the active serving run finishes.

The follow-up W4A4 serving run completed successfully: **27.78 decode TPS on every stream**, 78.3 aggregate end-to-end TPS, TTFT 33.83 s, 30.0 GiB/device. All eight 512-token outputs are identical to both earlier W4A4 implementations. The difference from 27.50 TPS is too small to establish a meaningful separate serving benefit from the scale-map optimization. Evidence: `/dev/shm/qwen4_integer_w4a4_noscatter.log` and token artifacts.

The shared mapping's final TPU gate passed **88 tests, 2 skipped** (float64 disabled), including scale mapping, all explicit modes, compiled AD, streaming full-output tests, distributed and fused MoE. The x64 CPU run covered those dtype cases separately. Independent read-only review found no concrete defect.

New isolated XLA profiles confirm lower wrapper overhead after sharing: skewed-route W8A8 XLA is approximately 38–44 us versus 63–69 us before. Pallas is about 35–40 us on those cases; no automatic W8A8 platform change was made.

All-mode profiles in `/dev/shm/grouped_routes_allmodes` additionally compared A16 to the existing XLA implementation: across the tested distributions, W4A16/W8A16 Pallas measured roughly 33–78 us, while XLA measured roughly 274–650 us. EasyDeL's narrow v5p decode selection now also covers A16 with INT4/INT8 codes on the same two matrix families; forced-XLA still wins over the heuristic. Updated CPU dispatch/fused tests passed 18 cases and the A16/fused TPU gate passed 14 cases. The full Qwen W4A16 benchmark completed successfully: **27.62 decode TPS on every CC8 stream**, **9.98 s mean TTFT**, **143.5 aggregate end-to-end TPS**, **30.0 GiB/device**, and exact warmup/timed 8×512-token parity. Evidence: `/dev/shm/qwen4_integer_w4a16_pallas_decode.log`. There is no earlier complete full-model W4A16 baseline here, so this is a result, not a measured full-model speedup ratio. W8A16 also completed successfully: **31.05 decode TPS on every CC8 stream**, **9.96 s mean TTFT**, **154.7 aggregate end-to-end TPS**, **44.4 GiB/device**, and exact warmup/timed 8×512-token parity. Evidence: `/dev/shm/qwen4_integer_w8a16_pallas_decode.log`. The current-source W8A8 rerun completed: **31.71 decode TPS on every stream**, **9.93 s TTFT**, **156.9 aggregate end-to-end TPS**, **44.4 GiB/device**, exact warmup/timed 8×512 parity and exact token equality with its preserved previous-version outputs. Evidence: `/dev/shm/qwen4_integer_w8a8_shared_scale.log`. All four mode runs have now completed.

## Opt-in dense Pallas path

`channelwise_quantized_matmul(..., quantize_activations=True, activation_bits=4 or 8, prefill_threshold=0, platform='pallas')` now exposes a full-K fused dense implementation. Default remains `platform='xla'`. It supports positive rank-two BF16 inputs, matching signed INT4/INT8 weights, equal contracting dimensions, M divisible by 64, K/N divisible by 128 and at most 4096. It uses a 40 MiB compiler VMEM budget, not a claimed measured memory saving. Backward/JVP uses the established XLA surrogate, not a new streaming backward.

Three configuration-matched captures of the **public** API, with exact full-output equality checks:

| M/K/N | Mode | XLA us | Pallas us |
|---|---|---:|---:|
| 2048/2048/2048 | W4A4 | 48.30–48.81 | 44.89–45.10 |
| 2048/2048/2048 | W8A8 | 57.35–57.71 | 54.81–55.26 |
| 4096/4096/4096 | W4A4 | 173.98–175.07 | 167.03–167.63 |
| 4096/4096/4096 | W8A8 | 235.53–235.77 | 246.53–247.55 |
| 512/2560/1280 | W4A4 | 26.81–27.02 | 25.62–26.04 |
| 512/2560/1280 | W8A8 | 28.61–29.09 | 28.15–28.70 |

The slower 4096 W8A8 case is why this is explicit rather than globally selected. Dense weight-only mode retains XLA; grouped weight-only has the tuned Pallas path described above. Evidence: `/dev/shm/dense_public_fullk/repeat*`. Empty `XLA_FLAGS` and `LIBTPU_INIT_ARGS` were verified for these runs.

Independent review caught missing rank/contraction/zero-dimension validation; six failing rejection cases were added and fixed before block construction. CPU contract suite passed 17 tests; expanded M64/M128/M192 TPU primal/JVP/VJP plus dense AD regressions passed 14 tests. Larger prototype M256/512 sweeps gave no substantial further improvement, and one W8A8 M512 tile exceeded scoped VMEM. The production default tile was not enlarged.

Fresh diagnostic raw-dot timings at 4096-square were BF16 340.477 us, raw INT8 172.176 us and raw INT4 98.462 us. These exclude activation quantization/scaling and are **not numerically equivalent complete operators or proof of a physical ceiling**. The complete quantized operators still have a substantial gap. A sampled XLA W4A4 invocation showed approximately 83 us in the integer-dot fusion, 49 us in quantization and 15 us in the reduction/other fusion. An explicit row-reciprocal experiment preserved outputs but showed no convincing overall improvement and was not integrated.

## Final round verification and remaining work

The consolidated production TPU gate passed **56 tests** in 134.26 s: public dense Pallas, grouped integer and weight-only Pallas, full-output matrix-family checks, compiled JVP/VJP, distributed MoE and all four modes' scale-optimizer steps. `compileall` and `git diff --check` passed. All four current-source full-model serving runs completed as recorded above.

Two more temporary scheduling experiments were correctness-gated and rejected: paired M64 dots with delayed epilogues measured about 171–174 us on 4096 W4A4, versus the ordinary fused path's 167–168 us; keeping the full weight resident while streaming M tiles measured about 167–169 us, also no additional win. Neither establishes VPU/MXU overlap, and neither was integrated. Profiles: `/dev/shm/dense_paired_candidates` and `/dev/shm/dense_weight_resident`.

A separate dense A16 Pallas primal was also tested rather than presumed slower: 2048 W4A16 XLA measured 56.7–58.1 us versus the best tested Pallas tile's 58.9–60.3 us; 4096 W4A16 XLA measured 322–324 us versus 327–328 us Pallas. W8A16 was likewise close or slower with Pallas. Exact full-output checks passed for the tested matrices. The temporary primal has no new training API and was not integrated; dense A16 retains XLA. Evidence: `/dev/shm/dense_a16_candidates/repeat*`.

A BF16-only cheaper half-tie detector was explored with enumerated significand-ratio tests and normal-range full-matrix comparisons. Its gains were small, and extreme-exponent/overflow equivalence was not established; production quantization semantics remain unchanged.

The complete dense operator still trails its raw integer-dot diagnostic significantly; larger training/prefill shapes, additional hardware configurations and training performance costs remain open. No universal 3x gain or physical-limit attainment is claimed. The optimization goal remains active, with the demonstrated wins preserved and the slower experiments excluded from production dispatch.
