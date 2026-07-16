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
just MoE), the distillation teacher forward + KL loss, and the prismcore
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
