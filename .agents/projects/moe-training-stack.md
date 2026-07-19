# MoE training stack: layout regimes, measured failure modes, and what's built

Grounding: everything here was measured on 2026-07-16 on v5p-2048 (1024
chips) training qwen3.6-35B-A3B (256 routed experts, top-8, H=2048, moe
M=512, bf16 params) with a same-architecture teacher for distillation, or
verified on the fake 8-device CPU mesh. No projected numbers.

## The three layout regimes (fused MoE, `_sparse_moe_call`)

The fused path's per-core cost is dominated by two tensors:

1. the permuted dispatch buffer `[local_tokens * k, H]` (plus ~3 copies live
   across fwd+bwd: forward activation, backward cotangent, one transient),
2. the expert weights the shard_map needs local to the core.

Which regime wins is a trade between weight movement and token movement:

- **Weight-gather DP (ep=1, batch over dp*fsdp)** — tokens stay put
  (`local_tokens = B*S/(dp*fsdp)`), expert weights are the thing that must
  be materialized per core. Wins when tokens/step >> expert bytes, i.e.
  large-batch training. This is the production training regime today
  (dp4/fsdp64/ep1/tp4).
- **Token-a2a EP (ep>1, fsdp_is_ep_bound=False)** — weights stay put
  (sharded over ep), the combine moves ~`local_tokens*k*H/tp` bytes per
  layer through `ragged_all_to_all` (non-ring) or an ep all-gather +
  psum_scatter (ring). Wins when expert bytes >> token bytes, i.e.
  small-batch serving or very large expert counts per layer.
- **Token replication (fsdp_is_ep_bound=True, the default)** — fsdp folds
  into expert parallelism, so the token batch is replicated per core within
  each dp replica. Fine for inference (tiny batches, maximally sharded
  weights); catastrophic for training batches — the dispatch buffer scales
  with the *global* batch.

## Measured failure modes (all from tonight)

| config | result |
| --- | --- |
| dp1/fsdp64/ep4/tp4, ep-bound, B=512 S=8192 k=8 H=2048 | dispatch buffer `bf16[33554432, 2048]` = 137.4 GB/core; TPU jellyfish bounds check (`allocation_size_words` over int32) rejected the compile after ~40 pod-minutes |
| dp1/fsdp64/ep4/tp4, ep-unbound, B=256 S=131072 | 16 GiB/core buffer, ~3 alive = 48 GB; `CompileTimeHbmOom` 112.6G/95.7G |
| dp4/fsdp64/ep1/tp4, B=256 S=131072 | 4.3 GB/core buffer; compile fits |
| dp4/fsdp64/ep1/tp4, expert weights `(EP, None, TP)` at rest | ep=1 degenerates to TP-only sharding: 80x bf16 (256,2048,1024) gate_up shards (268 MB) + 80x (256,512,2048) down shards (134 MB) = 30.06G live at `init_tx` entry (student+teacher, 40 layers each, ~15 GB/model); optimizer init (f32 mu + f32 nu + bf16 z over the student) then died with `RuntimeBufferAllocationFailure` (256M ask / 221M free) |

Bounds-check calibration note: the 137.4 GB buffer tripped the int32 check;
the 16 GiB buffer's compile died at the HBM budgeter instead, so the check's
word granularity could not be confirmed as 4 bytes (which would bound at
8.6 GB). The planner hard-fails only above 2^31 x 32-byte words (68.7 GB) and
warns above 2^31 x 4 bytes.

## What this change set provides

All three features are opt-in config knobs on `EasyDeLBaseConfig`
(TypedDict + `__init__` + `add_basic_configurations` + `read_basics_from_config`),
inherited by every MoE family with zero per-model changes. Defaults preserve
prior behavior bit-for-bit (pinned by tests).

### `moe_fsdp_shard_expert_weights` (ZeRO-3-style expert weights)

- At rest: expert (leading) dim of `ParallelMoELinear` kernels/biases sharded
  `P(('ep','fsdp'), None, 'tp')`. Plumbed ambiently: the parameter-init
  sharding context (`base_module._parameter_init_sharding_context`) enters
  `moe_expert_param_layout_scope(...)` from the config, and
  `_moe_parameter_layout` reads it — no model-family changes.
- At use: `_sparse_moe_call` weight in-specs carry `(ep, fsdp)` on the expert
  dim and the shard_map body does one `jax.lax.all_gather(w, fsdp, axis=0,
  tiled=True)` per weight per layer invocation, outside the chunk scan. The
  gather's transpose is a psum_scatter over fsdp, so weight grads land
  pre-sharded (asserted in tests), and `EasyDeLState.init_tx` mirrors the
  sharding onto mu/nu (asserted in tests).
- Gating: requires `fsdp_is_ep_bound=False`, a real fsdp axis on the active
  expert mesh, and `E % (ep*fsdp) == 0`; otherwise falls back to
  fsdp-replicated in-specs with a `warn_once` (an at-rest-sharded weight is
  then gathered implicitly at the shard_map boundary — correct, just not the
  explicit path). Non-divisible at-rest specs sanitize away shape-aware.
- Expected production effect (dp4/fsdp64/ep1/tp4): expert-weight residency
  15 GB -> ~0.24 GB per model per device; optimizer state ~75 GB -> ~1.2 GB;
  new per-layer transient = one gathered weight set (268+134 MB post-tp)
  during that layer's compute, re-gathered in backward under remat.
- Checkpoint compat: tensorstore stores global arrays; rest-sharding changes
  don't alter checkpoint layout.

### `moe_chunk_size` (chunk-scanned dispatch)

- Inside the shard_map, the local flattened token stream (post ep-gather in
  ring mode) is zero-padded to `ceil(T/chunk)` chunks and processed under
  `jax.lax.scan` with `jax.checkpoint` on the chunk body: permute -> grouped
  matmuls -> activation -> down -> tp psum_scatter -> combine -> unpermute,
  stitched back in token order. Peak dispatch memory becomes
  `chunk * k * H` per live buffer regardless of bucket size.
- Exact for token-choice top-k (per-token math); padded rows carry zero gate
  scores hence zero combine weights and are sliced off. Measured 0.0 max-abs
  diff vs single-pass on the CPU fake mesh (f32), outputs and input grads.
- All three branches are chunked: ep=1, ring (chunk after the ep all-gather,
  ep psum_scatter combine once at the end), and non-ring ep>1 (per-chunk
  `local_permute` + `ragged_all_to_all` with per-chunk group sizes). The
  ep>1 branch cannot execute on XLA:CPU (ragged_all_to_all unsupported);
  its runtime numerics were validated on the v5p-2048 slice — see the TPU
  validation section below.

### Layout planner (`easydel/layers/moe/_layout_planner.py`)

- `estimate_moe_layout(...)`: pure shape math -> `MoeLayoutEstimate`
  (dispatch buffer bytes chunk-aware, x3 live heuristic, expert-weight
  residency, weight-gather and token-comm bytes/layer, int32 verdicts,
  summary line with recommendations).
- `validate_moe_layout(...)`: lru-cached; logs INFO once per distinct
  layout, warns on the 4-byte-word band and on >128 MiB/layer fsdp-shardable
  weight residency, and raises `ValueError` (actionable: names
  `moe_chunk_size`, `fsdp_is_ep_bound=False`, dp/fsdp ways) for buffers over
  the conservative bound — called from `_sparse_moe_call` at trace time via
  `BaseMoeModule._validate_fused_moe_layout`.
- Regression tests encode the four measured configs above plus the
  weight-residency case.

## TPU validation of the ep>1 non-ring branch (2026-07-16, v5p-2048)

After the M3 measurement run (dp4/fsdp16/ep4/tp4, non-ring, chunk=65536,
fsdp-shard-weights, 35B-A3B distillation) died at step 1-2 with "NaN Loss",
the ep>1 non-ring branch was bisected on the real 1024-device slice via
`scripts/tpu_probe_moe_ep_parity.py` (eray-submitted, `@execute` fan-out).
Every configuration was FINITE and consistent with its ep=1 reference:

| probe (all on-slice, 1024 devices) | result |
| --- | --- |
| GptOss block f32, ep4 non-ring, base / chunk16 / fsdpW / chunk+fsdpW / single-padded-chunk, XLA gmm | finite, max diff vs ep1 1.2e-4..1.8e-4 |
| same five configs, Pallas gmm | identical results |
| GptOss bf16 fwd+bwd (H=256 I=768) ep4 base + chunk | loss/grads finite; elementwise fwd spread vs ep1 max 5.8e-2 at mean 1e-4 — same spread as the trusted ring path at ep2, i.e. bf16 accumulation-order noise, not a defect |
| GptOss bf16 M3-scale (65536 tokens/core, chunk=65536, fsdpW, Pallas) fwd | finite, matches ep1 (6.29e-3 vs 6.26e-3 max-abs) |
| same, fwd+bwd | loss 8.1391e-07 both meshes, all grads finite |
| Qwen3NextSparseMoeBlock (the qwen3_5_moe block: FUSED gate_up, shared expert, TOP_K) bf16 M3-scale fwd+bwd | ep1 loss 5.3693e-08, ep4 loss 5.3694e-08, all grads finite |

Garbage-tail hypothesis disproven for this branch: `jax.lax.ragged_all_to_all`
sends only the leading `send_sizes` rows of each shard's buffer and the
receive buffer is fresh zeros, so the TPU grouped-matmul's unzeroed tail never
reaches another shard's combine (unlike the ring branch, whose combine sums
full static-length buffers across shards and needed the explicit tail mask).
An explicit tail-zero guard was A/B'd on the slice and was bit-identical to
the unguarded path; the branch carries a comment instead of dead masking.

Consequence: the M3 NaN is NOT reproducible in the fused-MoE ep>1 non-ring
dispatch in isolation (random weights, up to M3 scale, fwd+bwd, bf16+Pallas).
Remaining deltas to the real failure: real 35B checkpoint weights (256
experts, real routing skew), the full 48-layer GDN/full-attention hybrid
stack under the changed mesh (batch rows/shard 2 -> 8 for every layer, not
just MoE), the distillation teacher forward + KL loss, and the external
ternary optimizer step. Next-probe order: (1) tiny full-model qwen3_5_moe
distillation step on both meshes, (2) real-checkpoint student forward-only
finiteness on the M3 mesh, (3) M3 relaunch with per-component finiteness
dumps.

## Hardware-unverified (explicitly)

- Pallas grouped-matmul small-group tile occupancy/autotune performance (the
  parity probe validated numerics, not step time).
- Actual step-time cost of chunking (scan overhead + remat recompute) and of
  the per-layer fsdp weight gather (expected ~0.4 GB/layer/device of
  gather traffic; whether XLA overlaps it with compute is unmeasured).
- The int32 bounds-check word granularity (see calibration note).
- The full 35B ep>1 configuration end-to-end (see the M3 investigation above).

## Roadmap NOT built here

- **Expert-over-tp for small-expert families** (`use_expert_tensor_mode`
  currently requires tp-size-1 inside the fused path and stays
  planner-unmodeled): needed for families whose E < fsdp so the expert dim
  can't absorb fsdp.
- **Offline teacher-logit datapacks for KD**: the teacher currently doubles
  expert-weight residency and step compute; precomputing teacher logits
  would remove the second model from the mesh entirely.
- **Async weight-gather overlap verification**: confirm on TPU profiles that
  the per-layer fsdp all-gather overlaps the previous layer's compute (and
  pipeline it explicitly if not).

## M3 NaN root-cause investigation (2026-07-16/17, this session)

Primary evidence recovered from the M3 driver log
(`eray logs launch_m3-erfan-20260716-171447`): at train_step 1 the metrics
line reads `loss: 0.011, kl_loss: 0.011, max_grad_norm: nan,
mean_grad_norm: nan` — the GRADIENTS were already non-finite at step 1
while the loss was finite. The trainer's NaN gate only checks the scalar
loss, `update_state_respectfully`'s in-graph skip also only checks the
loss, and the optimizer chain starts with `optax.clip_by_global_norm(1.0)`
— a single non-finite grad leaf makes the global norm non-finite and
poisons EVERY parameter in one update. Step 2's forward then NaNs. The
optimizer is exonerated as the generator (norms are computed on raw grads
before tx.update); the optimizer update math never saw finite grads.
launch_m4.py (diff: mesh (1,4,64,1,4,1) instead of (1,4,16,4,4,1), plus
profiler) ran 281+ healthy steps on the same code/data/checkpoint.

Tiny repro REPRODUCES this (scripts/tpu_repro_ep_nan.py, job ep_nan_tiny1):
tiny qwen3_5_moe (4 layers, 3xGDN+1xfull-attn, 256 tiny experts, H=256,
V=4096), student+frozen teacher KD (real `distillation_step`, chunked KL,
T=1 alpha=1), external optimizer package fused_adamw production kwargs, B=512 S=1024,
bf16, random weights:
  ep1 (1,4,64,1,4,1): 6/6 steps finite (loss ~0.1021, max_grad_norm 6.4e-3)
  ep4 (1,1,64,4,4,1): step 0 loss FINITE 0.1021 but 40 grad leaves
    non-finite; params/opt-state poisoned after the update. lm_head grad
    finite; embed + layer-0 grads NaN -> origin mid-backward.

### Root cause (identified 2026-07-17)

Leaf-level grad-norm map on the failing ep4 mesh (job ep_nan_leafmap2):
the TOP MoE layer's six mlp WEIGHT grads are finite (1e-7..7e-5) while its
`post_attention_layernorm` grad — fed by d(x) of the expert dispatch — is
NaN, and everything upstream in backward order is NaN (40 leaves,
exactly embed + 11 x layers0-2 + 6 x layer3-attn-side). lm_head and the
final norm (computed before the poison point) are finite. So the NaN
enters through the dispatch's INPUT-cotangent path, not through dW.

Mechanism: with ep>1 the expert-sorted dispatch buffers keep full static
length but only the leading `sum(group_sizes)` rows belong to this shard's
experts. The Pallas grouped-matmul grid only visits tiles covered by
groups (grouped_matmulv2 `make_group_metadata`), so rows past the group
prefix are NEVER WRITTEN — on TPU they hold stale buffer garbage
including NaNs (already documented for the FORWARD in the ring-combine
note in `_moe_module.py`). The forward is protected (non-ring:
ragged_all_to_all sends only leading rows; ring: explicit local_mask),
but the BACKWARD was not: `back_grouped_matmul` (custom-VJP grad_lhs)
leaves the tail of d(x_rows)/d(intermediate) unwritten, and the
sort-VJP transpose then SCATTERS that garbage into real token cotangents.
At ep=1 every row belongs to a group (no tail) — hence ep-mesh-only.
The correct cotangent for tail rows is exactly zero (they influence
nothing), so masking is the mathematically exact fix.

Fix: `_mask_dispatch_tail` in `_sparse_call` (_moe_module.py) — zero rows
past `sum(group_sizes)` on the gmm operands (x_rows at `_expert_ffn`
entry, gate_up/w0/w1 outputs, post-activation intermediate). Emitted only
when `ep_size > 1`; the ep=1 graph is unchanged. Covers chunked +
single-pass and ring + non-ring branches (ring ep>1 training had the same
backward exposure).

Side observation (separate issue, NOT the NaN cause, mesh-independent):
in the tiny repro the GDN-core input grads (`in_proj_qkv/a/b`, `conv1d`,
`A_log`, `dt_bias`) are exactly 0.0 while `in_proj_z`/`norm`/`out_proj`
receive gradient — needs a follow-up check on the ep1 dump / real model.

### Post-fix leaf-norm comparison (job ep_nan_fixval3, tiny model, step 0)

ep1 vs ep4 with the tail-mask fix: all leaves finite on both meshes;
~60 leaves match within bf16 noise (<1%). Systematic exception: the fused
expert weights (`experts.gate_up_proj`, `experts.down_proj`) have grad
norms EXACTLY 1/ep of the ep1 values on every layer (measured ratios
3.999-4.001 at ep=4, e.g. layer0 gate_up 1.317e-05 -> 3.293e-06). Router
gate / shared-expert / all non-dispatch leaves match. Pre-existing ep>1
dispatch-backward property (the masks only zero rows outside every group,
which cannot scale dW; dW over valid rows is untouched). A uniform
per-leaf grad scale cancels in Adam-family preconditioners
(s*mu/sqrt(s^2*nu) = mu/sqrt(nu)), so training dynamics are essentially
unaffected; resuming ep1-trained mu/nu on the ep mesh gives a ~20-100
step transient on expert-leaf update scale. Follow-up: root-cause the
replication/transpose bookkeeping (tokens are ep-replicated inside the
fused shard_map with check_vma=False; suspect the unmapped-out /
fan-out-a2a transpose pair) and restore exact dW scale.

### Fix validation (job ep_nan_fixval3, 2026-07-17, tiny model, 6 steps/arm)

| arm | result |
| --- | --- |
| ep1 (1,4,64,1,4,1) + fix | 6/6 finite; losses BITWISE identical to pre-fix ep1 (1.020937/927/851/870/832/794e-01) — ep=1 graph unchanged |
| ep4 (1,1,64,4,4,1) + fix | 6/6 finite (was: NaN grads at step 0); losses track ep1 within ~2e-6; max_grad_norm 6.409e-03 == ep1 |
| ep4 chunk=0 (single-pass) + fix | 6/6 finite; loss series identical to the chunked arm |
| m3 mesh (1,4,16,4,4,1) + fix | 6/6 finite |

GDN side-observation resolved: on healthy meshes all GDN-core grads are
nonzero (ep1 dump: in_proj_qkv 8.5e-4, conv1d 4.5e-5, A_log 1.2e-6, ...);
the exact-zeros seen in the broken ep4 run were an artifact of the
poisoned backward, not a separate bug.

CPU: moe suite + stage-aware expert mesh = 47 passed with the fix.

### Full-scale validation on the ordered mesh (job ep_validate_full, 2026-07-17)

launch_ep_validate.py = production launch.py with mesh (1,1,64,4,4,1),
per-bucket gradient_accumulation_steps=4 on the 131k bucket (microbatch 64
= 1 row/core; TrainingBucket field, distillation static slot 3), WANDB
disabled, profiler off. Resume from run-1000 (START_STEP=1000), 35B
student + bf16 teacher, worktree libs with the tail-mask fix.

Steps 1001-1018, every step logged `[graddiag] all 966 grad leaves
finite`; no NaN gate, no skip:

- step 1001 (131k bucket + compile): 1266.2 s, kl 0.172 — EXACTLY the
  production ep1 run's step-1001 kl (0.172; job launch-erfan-20260716-215251)
- step 1002 (8k + compile): 861.3 s, kl 0.158
- steps 1003-1011 (8k steady): 22.1 s/step typical (22.06-29.9),
  kl 0.119 -> 0.096, max_grad_norm 0.3-1.3
- step 1012 (131k steady, accum=4): 1175.3 s (dispatch 878.0 + device
  sync 297.2, data wait 0.0), kl 0.145, max_grad_norm 0.902
- stopped by hand at step 1018 (kl 0.096) — measurement policy; save_steps
  500 means no checkpoint was written in the window.

ep1-mesh comparators: 8k ~14.6-17.0 s/step (production step 1027 =
14.585; M4 step 281 = 16.974); 131k first step 927.91 s compile-inclusive
(steady unknown). The ep4 mesh pays ~1.4x on the 8k bucket (non-MoE
layers are ep-replicated: 4x tokens/core there) and ~1.2-1.3x on the
observed 131k step sample.

Bucket cadence note: ModBucketRule.select(current_step) is called with the
pre-increment step, so the 131k bucket runs at DISPLAYED train_steps 1001,
1012, 1023, ... ((step-10) % 11 == 0).

## "ep carries batch" dispatch (`moe_ep_carries_batch`, 2026-07-17, this session)

MaxText-style batch-sharded EP for training: the token batch shards over the
expert axis too, and tokens travel to their expert-owning shard instead of
every ep shard replicating the batch. Removes the ~1.4x ep4 tax measured on
the 8k bucket (non-MoE layers were ep-replicated: 4x tokens/core) and cuts
the fused-MoE dispatch buffers by ep.

### MaxText reference mapping (clone at
`/home/erfan/.claude/jobs/129ae660/tmp/maxtext/src/maxtext/layers/moe.py`)

| MaxText | easydel port |
| --- | --- |
| `is_batch_sharded_by_ep` (~line 1500): training batch IS ep-sharded; `activation_batch` logical axis includes "expert" | `EasyDeLBaseConfig._maybe_extend_batch_axis_for_ep` (base_config.py): `PartitionAxis.batch_axis` -> `("fsdp","dp","ep")`, model-wide via the BATCH semantic axis; `_moe_batch_axis_names` adds ep to the shard_map specs |
| `get_all_to_all_params(..., is_batch_sharded=True, is_dispatch=...)` (~1139) with `transform_array` strategies | `_communication_utils.get_all_to_all_params` already had `is_batch_sharded=True`; added `is_dispatch` (combine = transposed traffic matrix, matching MaxText line ~1272) |
| `ra2a_and_route` batch-sharded arm (~1662-1706): all_gather group sizes -> a2a params -> `ragged_all_to_all` -> `local_permute(is_offset=False)` | `_dispatch_ep_batch` nested in `_sparse_call` (_moe_module.py): one all_gather of `group_sizes` feeds both `build_ep_traffic_matrix` and `local_permute` |
| `unsort_output_and_ra2a` batch-sharded arm (~1884-1922): local unsort + combine a2a (`is_dispatch=False`) | same flow, with `_mask_row_tail` on the gmm output and the unsorted buffer (see tail-masking note) |
| `get_ragged_buffer_size` worst case `min(ep, E/k)` (~1304-1338) | `ep_batch_receive_buffer_rows`: `ep * local_tokens * min(k, E/ep)` (integer form of the same bound) |
| `ragged_buffer_factor` capacity truncation (~95-113, ~1009-1033, `_truncate_matrix`) | NOT ported — follow-up. Unbounded worst-case buffers are acceptable at our shapes (17.2 GB at 131k, 2.1 GB at 8k); the factor becomes interesting if 131k HBM pressure bites |
| `DISPATCH`/`COMBINE` (lines 56-57) | In this MaxText clone those are *einsum names for the dense capacity path*, not remat tags. easydel instead tags the two a2a outputs `checkpoint_name(x, "moe_dispatch")` / `"moe_combine"` following its own `mlp_gate`/`mlp_up`/`mlp_down` convention; names registered in `infra/etils.py` target lists |

### Design decisions

- **Gate** (`BaseMoeModule._ep_carries_batch_active` + shape check in
  `_sparse_moe_call`): ep>1, non-ring, `fsdp_is_ep_bound=False`,
  `sp_is_ep_bound=False`, no expert-tensor mode, stage-mesh path (distinct
  fsdp axis), `E % ep == 0`, and batch divisible by `dp*fsdp*ep` (else
  warn_once + per-layer fallback to the old path).
- **Global batch**: activations resolve BATCH -> `("fsdp","dp","ep")` via the
  config's PartitionAxis (single lever; spectrax
  `sharding/manager.py:536` is where the default `(fsdp, dp)` lives). The
  trainer's `step_partition_spec` default stays `P(("dp","fsdp"), "sp")` —
  set it to `P(("dp","fsdp","ep"), "sp")` in TrainingArguments to also place
  the input batch; otherwise there is exactly one input-boundary reshard per
  step. Non-divisible batches degrade safely: spec sanitization drops the
  trailing ep entry (activations) and the MoE gate falls back.
- **Chunking composes AFTER dispatch**: the dispatch a2a runs once over all
  local rows; `moe_chunk_size` then scans `_expert_ffn` over row-chunks of
  the *received* buffer (`chunk_group_sizes` computes per-chunk group sizes
  from cumulative offsets). Consequence: chunking bounds gmm live buffers
  but NOT the a2a receive buffer.
- **Tail masking (the 8ef1ddb9 lesson, applied by construction)**: every
  ragged buffer's rows past `sum(group_sizes)` are assumed garbage on TPU in
  fwd AND bwd. `_expert_ffn` masks its operands (pre-existing);
  `_dispatch_ep_batch` additionally masks the gmm output before the local
  unsort and the unsorted buffer before the combine — their transposes zero
  the unwritten cotangent tails of the a2a transposes before the sort-VJP
  scatters can route them into real token cotangents. CPU zeroes these tails,
  so CPU tests cannot catch this class; the masks are forward no-ops.
- **Remat**: to keep the two collectives out of the backward replay under a
  NOTHING_SAVEABLE-style policy, use
  `gradient_checkpointing="save_only_these_names"` with
  `gradient_checkpointing_targets=["moe_dispatch", "moe_combine", ...]`
  (add the mlp_* names you also want saved), or
  `"save_anything_except_these_names"` excluding everything else. No default
  policy was changed.

### Expected numbers (planner-verified, `test_moe_layout_planner.py` case f)

Mesh (1,1,64,4,4,1), k=8, H=2048, bf16, E=256:

| bucket | batch ways | tokens/core | a2a per dir per layer | worst-case recv buffer |
| --- | --- | --- | --- | --- |
| 131k (B=256, S=131072) | 256 (dp*fsdp*ep) | 131,072 | 4.29 GB (combine dir = /tp) | 17.2 GB (warn band, ~x3 live fwd+bwd) |
| 8k (B=512, S=8192) | 256 | 16,384 | 537 MB | 2.1 GB |

The planner recommends the knob on any ep>1/non-ring/unbound layout where it
is off, and reports the worst-case receive buffer as the binding allocation.

### Verification state

- CPU (fake 8-device): `tests/layers/moe/test_moe_ep_carries_batch.py` —
  numpy-emulated 2-shard dispatch == dense reference (offsets/permutes/masks
  exact), hand-computed a2a params both directions, receive-bound brute
  force, chunk group-size tiling, fwd+bwd `eval_shape` traces (chunked and
  unchunked), remat tags in the jaxpr, non-divisible fallback, knob-off and
  knob-inactive bit-identity. XLA:CPU cannot execute `ragged_all_to_all`, so
  multi-shard runtime numerics are NOT CPU-provable.
- TPU (pending, next production pause): `scripts/tpu_probe_ep_carries_batch.py`
  — ep4+knob vs ep1 loss parity (fwd+bwd finite), plus step timing of
  REF ep1 / BASE ep4 knob-off / CAND ep4 knob-on for 8k- and 131k-shaped
  configs (`REPRO_SHAPE=131k`). Unknowns it resolves: ragged a2a throughput
  on the v5p torus at 537 MB-4.3 GB/dir/layer, and whether the knob's
  worst-case buffers fit alongside remat at 131k. Hot-expert imbalance under
  real routing (recv skew toward popular shards) remains unmeasured until a
  real-checkpoint run.

### Teacher ring-attention verify (2026-07-19, launch_teacher_ring_verify.py)

Config under test: mesh (1,1,64,4,2,2), SEQUENCE_AXIS_NAME="sp",
moe_ep_carries_batch=False, chunked MoE, attn=RING. Driver logs:
/tmp/vp_run1_corehalt.log (run1), /tmp/vp2.log (retry).

- ring@4096: overall_H=0.732, top1=63.7% (ppl 2.1) vs flash/vanilla ref
  0.744 — MATCH on all 256 processes. Ring numerics validated at 4k under
  the exact production candidate config.
- ring@131072 run1: E0200 RuntimeUnexpectedCoreHalt (assertion args all
  zero) reported by host 10.164.15.242; rest of pod died on the collapsed
  collective (exitcode 1). Not a numeric result — TPU runtime fatal.
  Undiagnosed: hardware flake vs reproducible ring@131k compiler/kernel bug.
  15.242 has no known-bad history in ops memory.
- Retry (L=131072 only, same config): REPRODUCED the TPU fatal. Evidence:
  second libtpu slice dump initiated ~01:25:40 with 2.4 GB disk still free
  (dumps are written only on TPU fatals); the driver died at exit 120 in the
  ensuing disk-full cascade before the E0200 line could be logged. Two
  fatals in two attempts at the exact same shape => reproducible
  compiler/kernel bug in ring attention at B=256 S=131072 under
  (1,1,64,4,2,2), NOT an infra flake, NOT host-specific evidence.
- ring@65536: never executed under this config (trimmed for disk budget) —
  unknown. Non-SPLASH 131k forward remains UNVALIDATED by any mechanism
  (flash/vanilla refs exist only for 4k/65k).
- DECISION (user, via coordinator): PATH B — fix the SPLASH head_dim=256
  kernel itself. No launch.py ring/mesh changes applied or committed. Ring
  documented here as partial fallback only (numerics good <=4k; fatal at
  131k as-is).
- Ops post-mortem: each TPU fatal writes an ~11 GB /tmp/slice_*.dump on
  this dev VM (live production ray head + worker host!). run1's dump ate
  the 15 GB headroom; the retry's dump then took / to 100%, which is what
  killed the retry driver (raylet file_system_monitor + ENOSPC), and put
  the production worker host at risk. 48 GB of accumulated dumps from this
  hunt's crashes deleted 2026-07-19; also cleared 10.2 GiB uv cache and
  truncated fat ray session logs (monitor.log/raylet.out/gcs_server.out).
  Rule of thumb: after any TPU fatal probe on this host, delete
  /tmp/slice_*.dump before the next launch.
