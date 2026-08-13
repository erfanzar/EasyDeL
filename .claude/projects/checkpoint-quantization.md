# Pre-quantized checkpoint support — design

Goal: load externally quantized checkpoints (fp8, compressed-tensors, AWQ/GPTQ, MXFP4, NVFP4) for serving **and**
QAT/finetune, across all model families, without per-format layer classes, per-model code, or forward-path branching.

Status: design. No code yet.

## The problem, stated precisely

EasyDeL today only *self-quantizes*: `ParallelLinear.to_quantized(config)` takes an fp weight and calls
`prepack_quantized_weights`. Nothing parses an HF
`quantization_config` — `rg 'quant_method|config_groups|weight_block_size|activation_scheme|weight_scale'`
over `libs/easydel/easydel/` returns nothing. So every pre-quantized checkpoint on HF is currently unloadable.

What we already own (do not rebuild):

| capability                                                             | location                                                                                                                                              |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| formats + packing (affine/nf4/int8/mxfp4/nvfp4/mxfp8/nvfp8/turboquant) | `ejkernel/quantization/` (~4.4k lines)                                                                                                                |
| fused dequant+matmul, weight-only                                      | `ejkernel/kernels/_pallas/tpu/quantized_matmul/`, `_tilelang/`                                                                                        |
| **grouped matmul with in-kernel dynamic activation quant**             | `ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:1159-1171` (`maybe_quantize_lhs`, fp8/int8 by hardware, `rhs_scale [G, blocks, 1, N]`) |
| one quantized linear, mode-parameterized                               | `easydel/layers/linears/_linear_quantized.py:617`                                                                                                     |
| STE / QAT per format                                                   | `easydel/layers/quantization/_straight_through.py`                                                                                                    |
| KV-cache quantization (incl. turboquant paged attention)               | `easydel/caching/`, `layers/attention/_flexible.py:1036`                                                                                              |
| fused-projection TP-portable layout                                    | `fused_param_tp`, `layers/layouts/`                                                                                                                   |
| N-tensor → 1-param conversion with inverse                             | `utils/parameters_transformation.py:341-375`                                                                                                          |

The kernels are done. **This is a loader project.**

## The seam

One contract. Everything above it is format-specific; everything below it is already written and shared.

```
CanonicalQuantizedWeight:
    mode:          str          # ejkernel mode: affine | nf4 | mxfp4 | nvfp4 | mxfp8 | nvfp8
    group_size:    int
    bits:          int
    needs_biases:  bool
    quant_kernel:  Array        # packed, ejkernel layout
    quant_scales:  Array
    quant_biases:  Array | None
    expert_dim:    bool         # MoE: leading E axis on kernel + scales
    output_scale:  Array | None # per-tensor scalar applied post-matmul
    activation:    ActivationPolicy
```

The last three fields exist for reasons established below; they are what keep MoE, NVFP4 and W8A8 from each spawning a
parallel code path.

**`expert_dim` is the one that must be right in phase 0.** Dense is `(K, N)`
with rank-2 scales; MoE is `(E, K, N)` with `(E, blocks, 1, N)` scales. ejkernel's dense
`validate_packed_quantized_matmul_layout` hardcodes rank-2 — if the canonical contract inherits that assumption, every
MoE format becomes a duplicate of its dense twin later.

**`output_scale`** carries a per-tensor scalar. Needed because NVFP4 checkpoints have two scale levels (e4m3 block +
fp32 global `weight_scale_2`) and folding the global into an e4m3 code is lossy. It does not need folding — a per-tensor
scalar commutes with the matmul:
`x @ (w_q · s_block · s_global) == s_global · (x @ (w_q · s_block))`. Exact, one multiply, no kernel change. Also serves
compressed-tensors' fused per-tensor scales.

**`activation`** is a runtime policy, not a weight property:
`none | dynamic(int8|fp8, per-token) | static(scale, per-tensor)`, resolved once from the checkpoint scheme (W8A16 /
W8A8-dynamic / W8A8-static) and consumed at kernel invocation.

This is precisely what `_linear_quantized.py:777-798` already builds and what
`_dequantize_array` / `_distributed_quantized_matmul` / `from_quantized` /
`restage` already consume. Landing every external format on this triple means zero new forward paths, zero new layer
classes, zero model edits.

**Rule: a new checkpoint format may add exactly one adapter function and one registry entry. If it needs anything else,
the seam is wrong — fix the seam.**

## Layer 1 — format adapters (the only per-format code)

```python
@checkpoint_quant_registry.register("compressed-tensors")
def adapt(tensors: dict[str, Array], meta: SchemeMeta) -> CanonicalQuantizedWeight
```

Pure function, no module state, no sharding, no mesh. ~60-150 lines each:

- **fp8 per-tensor / per-channel** — scale rank normalize → canonical.
- **fp8 blockwise (128x128, `weight_scale_inv`)** — block dequant → requantize to per-channel or subchannel (see
  "requant policy").
- **AWQ int4** — u32→u4 with the `(0,4,1,5,2,6,3,7)` reorder, zero points → affine.
- **GPTQ int4** — same shape, different packing + `g_idx`.
- **MXFP4 (gpt-oss)** — u8→e2m1, u8→e8m0 scales → requantize to wider block.
- **NVFP4 (modelopt / CT)** — e2m1 + fp8 block scale × fp32 `weight_scale_2`.

Duplication control: the *primitives* (bit unpack, e8m0↔fp32, block dequant, requantize) live once in
`ejkernel/quantization/_utils/`; several already exist (`bitpack.py`, `fp_tables.py`, `qparams.py`). Adapters are
compositions, never implementations. Any primitive written twice is a review failure.

## Layer 2 — ingestion via existing `reform_param` fusion

`reform_param` already accepts `{'sources': [...], 'fuser': fn, 'inverse_fuser': fn}`
(schema validated at `parameters_transformation.py:341-375`, applied at
`apply_reform_param_fusions`). That is *exactly* N-checkpoint-tensors → 1-EasyDeL-param with an export inverse.

So ingestion is: **auto-generate reform rules from the resolved scheme**, e.g.

```
"<prefix>.quant_kernel": {
    "sources": ["<hf>.weight", "<hf>.weight_scale_inv"],
    "fuser": partial(adapter, meta=...),
    "inverse_fuser": partial(adapter.invert, meta=...),
}
```

Consequences we get for free:

- no new loader, no second conversion path;
- `.to_torch` export works via `inverse_fuser` (vLLM-TPU has **no** export path);
- fused QKV/gate-up already flows through this machinery, so scales ride the same `fused_param_tp` permutation as the
  weight — one place, tested once.

## Layer 3 — scheme resolution (model-agnostic dispatch)

`CheckpointQuantScheme.from_hf_config(quantization_config) -> scheme`, answering
`scheme.for_path(path) -> ResolvedScheme | None`. Responsibilities:

- `quant_method` → adapter selection;
- `ignored_layers` (exact) vs `modules_to_not_convert` (prefix) — different match semantics, one implementation;
- compressed-tensors `config_groups` / `targets` regex matching;
- **fused-shard consistency**: all shards of a fused projection must share a scheme, else hard error (silent
  mixed-precision fusion is a correctness bug);
- per-layer opt-out honored identically for dense and MoE.

Plugs into `EasyQuantizer` (which already walks paths with a regex at
`_quants.py:558-583`) rather than living beside it — same traversal, richer predicate.

**Why 80+ families cost zero:** models build `ParallelLinear`/`BaseMoeModule`; conversion already routes every HF tensor
through `parameters_transformation`; the resolver works on paths. No model file is touched.

## Layer 4 — activation quantization

Current state is worse than it looks: `_quantize_runtime`
(`_linear_quantized.py:1011`) is **never called** — dead code — and
`QuantizationConfig.runtime_dtype` is unwired for linears (every `runtime_dtype`
hit in the repo is `OperationMetadata.runtime_dtype`, the attention compute dtype). ejkernel's dense op takes float `x`
(`output = x @ dequantize(w, scales, zeros)`). **EasyDeL has no activation quantization for linears today.**

The only in-kernel activation quant in the stack is `grouped_matmulv3`'s
`maybe_quantize_lhs`. So:

- **MoE gets W8A8 first, for free** — the kernel already does it.
- **Dense needs one of:** (a) route through `grouped_matmulv3` with
  `group_sizes=[m]` — zero kernel work, immediate W8A8, and exactly what vLLM-TPU does (their standalone
  quantized-matmul Pallas kernel is dead code; production goes through gmm_v2); or (b) add `maybe_quantize_lhs` to the
  dense Pallas kernel properly. Do (a) as the correctness baseline, (b) only on a measured win.

Static (calibrated per-tensor) activation scales are cheap and exact via
`output_scale`. Note vLLM-TPU implements this **incorrectly** in their fused path — `compressed_tensors_w8a8_fp8.py:248`
applies `weight_scale` but drops
`input_scale`, while their split path at line 293 applies both. Our parity test must be the kind that catches that.

**Requant policy — do not copy their default.** vLLM-TPU rewrites narrow checkpoint blocks to a wider block at load
because a quant block narrower than the MXU column forces dequantize-before-matmul (`should_dequantize_before_matmul`:
`quant_block_size < mxu_column_size`). For gpt-oss MXFP4 they dequantize to fp32 and requantize to block-512 fp4 — a
real accuracy loss. We only need that if measurement shows group-32 starves the MXU, and then it is an **opt-in**
policy, not a default. Make the threshold a property queried from the kernel, never a constant inside an adapter.

## Kernel capability matrix (verified, not assumed)

`quantized_matmul` (dense) is registered on six platforms; `grouped_matmulv3`
(MoE) on three.

| kernel           | platform                              | weight quant | sub-byte packed W              | activation quant                             | notes                                                                            |
|------------------|---------------------------------------|--------------|--------------------------------|----------------------------------------------|----------------------------------------------------------------------------------|
| grouped_matmulv3 | Pallas TPU                            | yes          | yes (`should_bitcast`, `:396`) | **yes** (`maybe_quantize_lhs`, `:1160-1171`) | `rhs_scale (G, blocks, 1, N)` validated `:1017`                                  |
| grouped_matmulv3 | TileLang GPU                          | yes          | —                              | no                                           | has VJPs for lhs/rhs/rhs_scale/rhs_bias (`_interface.py:79`)                     |
| grouped_matmulv3 | XLA                                   | yes          | —                              | no                                           | dequantize-then-matmul fallback when `rhs_scale` given (`:27`); parity reference |
| quantized_matmul | Pallas TPU                            | yes (+bwd)   | yes                            | **no**                                       | `x: Float[Array, "m k"]`, `Y = X @ dequant(W)`                                   |
| quantized_matmul | XLA / Triton / CuTe / CUDA / TileLang | yes          | yes                            | **no**                                       | same signature on all                                                            |

**But v3 is not the production MoE path, so none of that is reachable today.**
`_moe_module.py:1524-1540` builds `gmm_kws` with `platform="xla"`
(`bypass_xla_tiling=True`, i.e. XLA `ragged_dot`) and a conditional Pallas branch; `use_v3` is never set anywhere. And
the plain `grouped_matmul`
entry point **rejects** quantized weights outright:

```
grouped_matmul.py:324-325
    if self.op_id != "grouped_matmulv3" and (rhs_scale is not None or rhs_bias is not None):
        raise ValueError("rhs_scale and rhs_bias are only supported by grouped_matmulv3.")
```

So MoE expert quantization is **not** free plumbing, as an earlier draft of this document claimed. It needs one of: a
quantized `ragged_dot` path, dequantize-experts-before-`ragged_dot` (memory win only, no compute win), or moving the
production MoE path onto v3 — a separate decision with its own performance evidence. Treat MoE as its own investigation,
not a phase-1 freebie.

### Adding activation quantization to dense

Two designs:

- **(A)** extend the dense Pallas kernel with an lhs-quant path — roughly the 40 lines at
  `grouped_matmulv3/_pallas_impl.py:508-545`, but it changes accumulator dtype and per-block rescale in the core loop
  shared with the working weight-only path.
- **(B)** route dense through `grouped_matmulv3` with `group_sizes=[m]` — zero kernel work, already-exercised code. What
  vLLM-TPU actually does.

Do (B) first, (A) only on a measured win.

**The dispatch lives inside ejkernel's dense op, never in EasyDeL.** EasyDeL passes an `ActivationPolicy`; ejkernel
chooses Pallas-weight-only / v3-one-group / XLA. Anything else leaks kernel internals into the framework and breaks the
registry contract (golden rule 6).

Signature change is additive across the six impls:
`activation_quant: "none" | "dynamic"`, `activation_scale: Array | None`.

- **XLA gets a real reference impl** (~20 lines: quantize `x`, `dot_general`
  with `preferred_element_type=int32/float32`, rescale). Mandatory fallback *and* the parity baseline for every
  activation-quant test.
- **Pallas TPU** dispatches to v3-one-group when `activation != none`.
- **Triton / CuTe / CUDA / TileLang** raise `NotImplementedError` → registry falls back to XLA on GPU (correct, slow)
  until TileLang gets a real lhs-quant path mirroring the Pallas one — which also fixes GPU MoE activation quant.

## MX / NV FP4 / FP8 — already on-disk compatible

Verified, not assumed:

- `qparams.py:479-481` **enforces uint8 scales** for all of
  `{mxfp4, mxfp8, nvfp4, nvfp8}`.
- MX decode is `scale = jnp.exp2(exp.astype(float32))`
  (`quantizations.py:613`) → e8m0 semantics, spec-correct.
- NV decode is an `e4m3_table` lookup on uint8 codes (`quantizations.py:682, 1028-1040`) → e4m3 block scales,
  spec-correct.
- Group sizes already pinned to spec: mxfp4/mxfp8=32, nvfp4/nvfp8=16.

Consequences:

- **MXFP4 (gpt-oss) is a repack, not a requantize** — their e2m1-pairs-in-u8 + e8m0-in-u8 at group 32 map onto our
  packed codes + uint8 scales at group 32. Only the packing container differs (2×e2m1 per u8 vs our uint32 codes).
  **Zero numeric loss**, strictly better than vLLM-TPU's lossy block-512 rewrite.
- **NVFP4** matches on the block scale; only the fp32 global needs the
  `output_scale` slot.
- **MXFP8 / NVFP8** have no HF checkpoints today, but we support them natively, so they are available as *requantization
  targets* — e.g. blockwise-fp8 DeepSeek → mxfp8 (group 32, e8m0), an option vLLM-TPU cannot express.

## QAT / finetune

Three supported modes, all sharing the canonical triple:

1. **Frozen quantized + LoRA** — quantized params as buffers, adapters train. Works with `_lora.py` today. Expected 90%
   case.
2. **STE-QAT** — `_straight_through.py` already implements per-format STE; it needs a dequant→STE→requant VJP over the
   canonical triple. Because every format lands on one representation, **STE is implemented once, not per format.**
3. **Dequantize and full finetune** — `from_quantized()` already exists.

Optimizer-state interaction (mode 2) is the open question: quantized params must not receive dense optimizer state.
Decide before implementing mode 2.

## Non-negotiables

- No per-format layer class. vLLM-TPU's failure mode is 12 config classes × 12 method classes × 2 stacks ≈ 7.7k lines of
  near-duplicate wrapper.
- No format branching inside any model file or any forward path.
- Every format ships an XLA reference parity test against the reference dequantization. Their CI records `"unverified"`
  for all of it; that is the gap we beat them on, not feature count.
- Scales travel through `fused_param_tp` with the weight, verified at tp>1.
- Registry, not if-chains (golden rule 6).

## Phasing

| phase | content                                                                                | status   |
|-------|----------------------------------------------------------------------------------------|----------|
| 0     | canonical contract, scheme resolver, adapter registry, reform-rule generation          | **done** |
| 1     | fp8: per-tensor, per-channel, blockwise `weight_scale_inv`                             | **done** |
| 2     | AWQ + GPTQ (incl. act-order) int4                                                      | **done** |
| 3     | MXFP4 + NVFP4 with `output_scale`                                                      | **done** |
| 4     | compressed-tensors (int8 / fp8 / int4 / NVFP4, per-config-group)                       | **done** |
| 5     | **grouping-axis fix** — the dominant accuracy item, see below                          | next     |
| 6     | activation-quantization runtime (ejkernel signature + XLA reference + Pallas dispatch) |          |
| 7     | MoE expert quantization — needs a kernel decision first, not plumbing                  |          |
| 8     | STE-QAT over the canonical state + export inverses                                     |          |

## Weight formats — shipped

`_codecs.py` holds one decoder per on-disk encoding; `_formats.py` holds the adapters. Registered `quant_method`
aliases: `fp8`, `awq`, `auto_awq`, `gptq`,
`mxfp4`, `gpt_oss_mxfp4`, `modelopt_fp4`, `nvfp4`, `compressed-tensors`,
`compressed_tensors`.

Every adapter is decode-only: it ends in
`CanonicalQuantizedWeight.from_dense`, which routes through the same
`prepack_quantized_weights` the self-quantization path uses, so a checkpoint-loaded weight is layout-identical to a
natively quantized one. No adapter reimplements packing, bit manipulation or scale math.

Measured round-trip error against independent NumPy references (`test_checkpoint_quant_formats.py`, 28 tests):

| format                  | target      | rel. err |
|-------------------------|-------------|----------|
| fp8 per-channel         | affine int8 | 0.003    |
| fp8 blockwise 128×128   | affine int8 | 0.003    |
| compressed-tensors int8 | affine int8 | 0.004    |
| AWQ int4                | affine int4 | 0.061    |
| MXFP4                   | mxfp4 g32   | 0.042    |
| NVFP4                   | nvfp4 g16   | 0.154    |

### Two decisions that measurement reversed

**fp8 targets affine int8, not mxfp8.** The intuitive argument — keep 8-bit floats as 8-bit floats rather than push them
through an integer grid — is wrong for *grouped* quantization. Within a group the dynamic range is small, so int8's 256
levels with an exact float scale beat E4M3's 3-bit mantissa with a power-of-two E8M0 scale: 0.003 versus 0.049 on
gaussian weights. Exposed as
`ScaledElementsAdapter.float8_target` for re-measurement on real weights.

**MXFP4 is not the lossless repack claimed earlier.** It would be — measured at exactly 0.0 — if the repack grouped
along input features. It does not.

### The grouping-axis gap (phase 5, and the largest remaining item)

Every checkpoint format groups scales along **input** features.
`ParallelLinearQuantized` packs `[in, out]` with `transpose=False`, which groups along **output** features. A
checkpoint's calibration therefore cannot survive repacking — the groups do not correspond.

Measured on data lying exactly on an MXFP4 grid (`[256, 128]`):

| grouping                            | rel. err |
|-------------------------------------|----------|
| input axis (`axis="row"`)           | **0.0**  |
| output axis (`axis="col"`, current) | 0.042    |

AWQ-like 4-bit data: 0.031 (input) versus 0.061 (output). 8-bit absorbs the mismatch; 4-bit does not — the entire 4-bit
error budget in the table above is this one issue. Fixing it means changing the quantized linear's layout
(`_quantized_linear_layout_spec`, `_resolve_shard_specs`,
`_distributed_quantized_matmul` and the sharding that follows), which is a real change to a working kernel path and is
why it is its own phase rather than something slipped into an adapter. `TestGroupingAxisLimitation` pins the current
cost and is written to fail once it is fixed.

## Phase 0 — what shipped

`libs/easydel/easydel/layers/quantization/checkpoint/`:

* `_canonical.py` — `QuantSpec`, `CanonicalQuantizedWeight`, `SourceFormat`,
  `ActivationPolicy`. `QuantSpec` validates itself through the existing
  `resolve_ejkernel_quant_params`, so the per-mode `(group_size, bits)` table is not duplicated, and projects onto
  `QuantizationConfig` — the type the quantized layers already accept.
* `_adapter.py` — `CheckpointQuantAdapter` ABC + `register_adapter`, on top of the existing `Registry` under category
  `"checkpoint-quant"`. No new registry.
* `_scheme.py` — `CheckpointQuantScheme.from_hf_config` / `for_path` /
  `for_fused`, including the exact-vs-prefix ignore distinction and the fused-shard agreement check.
* `_reform.py` — `checkpoint_quant_reform_param`, projecting a resolved scheme onto `reform_param` rules.

One converter change, in `utils/parameters_transformation.py`:
`TensorConverter.to_jax_preserving_dtype` plus a `preserve_dtype` flag honored by `process_tensor`. Without it,
`convert_pytorch_to_jnp` casts every leaf to the model's param dtype and a packed `uint32` kernel or `uint8` scale is
destroyed silently.

Two non-obvious things the implementation forced, both now encoded in the code and its tests:

1. **The rule key must be the target parameter (`quant_kernel`), not a source suffix.** `apply_reform_param_fusions`
   skips a group when the fused key is already present in the state dict, so a self-hosted rule keyed on `weight` —
   which is also one of its own sources — would silently never fire.
2. **`SourceFormat.raw` is excluded from hashing.** A frozen dataclass holding a mapping raises on `hash()`, which
   breaks any use as a dict key or JIT static argument.

Tests: `libs/easydel/tests/layers/quantization/test_checkpoint_quant_seam.py`
(25, passing) drive the real `StateDictConverter.huggingface_to_easydel` rather than a stand-in, asserting that packed
dtypes survive while ordinary dense weights are still cast and transposed.

## TPU v5 measurements (2026-08-03, single chip, jitted, autotune off)

Op-level: `jax.jit(x @ w)` vs `jax.jit(quantized_matmul(...))`, warmup 5, median of 20. bf16 baselines: qkv (4096x6144)
147us @ t=8/32 (overhead floor — identical at both batch sizes), 359us @ t=2048; gate_up (4096x28672) 216us / 1210us;
down (14336x4096) 171us / 687us.

**XLA quantized path: uniformly SLOWER than bf16.**

| shape   | mode    | t=8       | t=2048 |
|---------|---------|-----------|--------|
| qkv     | mxfp4   | 0.16x     | 0.26x  |
| qkv     | nf4     | 0.22x     | 0.33x  |
| gate_up | mxfp4   | **0.05x** | 0.20x  |
| gate_up | affine8 | **0.05x** | 0.20x  |
| down    | mxfp4   | 0.08x     | 0.23x  |

The XLA path dequantizes in-graph (unpack + scale before the matmul); on TPU the dequant work dwarfs the bandwidth
saved. Never route serving through
`platform="xla"` qmm expecting speed — it is a correctness fallback only.

**Fused Pallas packed path: a SINGLE kernel compile exceeds 10 minutes**
(mxfp4, 4096x6144, t=8; `use_best_config=False`, so not autotune). Confirmed twice at model level — the whole-model
bench died inside
`jax::PyClient::CompileAndLoad` after 20-30 min with quantization already finished (2.38 GB packed) — and now once at op
level. RSS grows ~9 GB during the compile, pointing at pathological Mosaic IR size (likely a fully unrolled in-kernel
unpack over the K/group dimension). This blocks ANY steady-state measurement of the fused path and is the top perf item
in the subsystem.

Fixed along the way, verified: eager weight packing in
`ParallelLinearQuantized._quantize_array` — `prepack_quantized_weights` ran unjitted; on (4096,6144): mxfp4 4.80s eager
vs 8.2ms jitted (585x), affine 2.34s vs 0.7ms (3490x). Now routed through `prepack_quantized_weights_jit`
(also used by the checkpoint `from_dense` path).

Also real: mxfp4 model compression measured 7.50 GB -> 2.38 GB (3.15x); un-jitted eager model forward is ~275x slower
than `spx.jit` (982ms vs 3.6ms)
and flat across batch size — never benchmark without jit.

## The compile-bomb root cause, and the vLLM-style fix (2026-08-03, later)

Root cause of the >25-minute quantized compiles, confirmed at source level and by experiment: ejkernel's dense qmm
kernel **hand-decodes packed uint32 inside the Pallas kernel** (`_pallas_impl_core.py:459-486` e2m1 sign/exp/mant bit
math, `:565-604` shift/mask unpack) — Mosaic IR explodes. vLLM never unpacks in-kernel: weights are native sub-byte
dtypes, `pltpu.bitcast` once per tile. Our `grouped_matmulv3` already had the bitcast design but was missing two guards
its vLLM counterpart (`gmm_v2`) has. Ported them:

* `_pallas_impl.py` `_matmul` unquantized branch — added the dequantize-before-matmul path (fold per-block scales into
  rhs in registers, full-K matmul per column strip), triggered when
  `not is_matmul_supported(lhs, rhs)` or the rhs quant block is narrower than the MXU column. Upcast happens BEFORE any
  reshape: Mosaic rejects rank-3 sub-byte vectors.
* quantized-lhs branch — added the exact `is_matmul_supported` upcast guard.
* **fixed a scale-indexing bug vLLM also has latently**: the k-loop stepped by the lhs quant block (512) while indexing
  rhs scales as
  `start_k // rhs_qbs` — with rhs blocks of 128, one scale covered four blocks. Clamped the step to the rhs block;
  relerr dropped 0.02-0.03 → 0.016 uniform. (vLLM masks it by never configuring rhs blocks < 512.)
* widened `rhs` annotations (`Float | Int`) across the v3 interfaces and the module op — integer weights were rejected
  by beartype before reaching the kernel.

Measured v5 Mosaic storage support (load+upcast one tile):
`float4_e2m1fn` **UNSUPPORTED** (`vector<8x128x8xf4E2M1FN>` Mosaic error);
`int4` OK; `int8` OK; `float8_e4m3fn` OK. This mechanically explains vLLM's support matrix (MXFP4 = v7 only; int4/int8 =
v5/v6) and means:
**on v5, f4 runtime storage is impossible — MXFP4/NVFP4 checkpoints must target int4/int8 affine on this hardware.**
`target_spec` should become hardware-aware. fp8 storage works (no fp8 MXU, so W8A16 dequantize-before).

Result — v3-one-group int8 W8A8 (K-blocked g128, in-kernel int8 lhs quant), TPU v5 single chip, vs jitted bf16 `x@w`:

|                          | compile             | decode t=8/32  | prefill t=2048 |
|--------------------------|---------------------|----------------|----------------|
| before (dense qmm)       | **>25 min, killed** | —              | —              |
| v3 untuned (128,128,128) | 1.3-7 s             | 0.15-0.17x     | 0.02-0.03x     |
| v3 tiled (128,2048,1024) | 3.8-8.3 s           | **0.92-0.96x** | 0.58-0.66x     |

relerr 0.016 (double 8-bit quantization). Tiling is decisive — a 6-64x swing; vLLM ships a 700-line tuned-block-size LUT
for exactly this reason.

**W8A8 vs W8A16 A/B** (same int8 weights, same kernel, tiles m128/k2048/n1024, impl-level `maybe_quantize_lhs` toggle):
**W8A16 wins everywhere on v5** — faster at every shape/token count (qkv t=32: 0.91x vs 0.78x; gate_up t=2048:
0.43x vs 0.26x) AND 5-10x more accurate (relerr 0.002-0.003 vs 0.016). The per-token lhs quant is VPU work v5's int8 MXU
cannot pay back. fp8-W8A16 is unusable on v5 (0.03-0.20x — f8 upcast has no fast path this generation). v5 recipe:
**int8 K-blocked storage, dequantize-before, bf16 MXU, maybe_quantize_lhs=False**; revisit W8A8 on fp8-capable
generations. Caveat on the earlier sweep's 0.92-0.96x W8A8 numbers: measured BEFORE the scale-indexing fix, i.e. with
illegally coarse lhs blocking; post-fix honest W8A8 is 0.76-0.78x at those points. If W8A8 returns, rhs g512 restores
the coarser legal lhs blocking (LUT dimension).

Not yet >1x. Why, and the path there:

- decode t<=32 is at an overhead floor (~150us) for BOTH bf16 and v3 — bf16 runs 50MB at only ~343GB/s, far off
  roofline. Halved weight bytes cannot show up until the floor is lowered or the weight-per-chip is bigger; at 27B-class
  shards the bandwidth win should surface.
- prefill pays the in-kernel lhs-quant VPU cost without recouping it; fp8 W8A16 storage (supported!) with
  dequantize-before may beat int8 W8A8 there.
- a tuned tiling table per (shape-class, tokens) is mandatory, not optional.

Next: wire ejkernel's dense `quantized_matmul` TPU route through v3-one-group with native-dtype storage (kills the
uint32 in-kernel unpack path on TPU), make checkpoint `target_spec` hardware-aware (v5 -> affine int8/int4), add a tuned
tiling LUT seeded from the sweep above.

## Autoresearch: 2x-speedup loop (2026-08-03, branch `autoresearch-qmm-2x`)

8 iterations, git-as-memory on the branch, log in `autoresearch-results.tsv`. Metric: min speedup vs jitted bf16 over
{int8,int4} x {qkv,gate_up,down} x {t=8,2048}, relerr gate 0.08. **0.456 -> 1.078** (2.4x metric improvement); every
cell >1x.

The path was one demolition and one construction:

- **Pallas is the wrong tool for this op.** The v3 kernel's *pure bf16*
  ceiling measured 0.47-0.78x of XLA's matmul — quantization tuning inside it is capped below 1 before it starts. (Same
  lesson as the MoE ragged_dot history.)
- **The winner is a 30-line XLA composition** — now
  `ejkernel.kernels._xla.quantized_matmul.channelwise_quantized_matmul`
  (12 CPU parity tests): decode = `x @ w_q.astype(bf16)` (XLA fuses the upcast into the weight stream; ANY pre-dot
  arithmetic breaks the fusion)
  with the per-channel scale on the [m,n] output; prefill = per-token int8 acts + native int8xint8 dot (459 TOPS/core =
  2x bf16) + epilogue scales. Per-channel scales are load-bearing: K-blocked scales force either the fusion break
  or [blocks,m,n] int32 partials (1.9 GB at m=2048).

Where 2x stands, honestly:

- **Achieved**: W4A4 prefill 2.88x (int4 MXU is 920 TOPS/core = 4x bf16) — but relerr 0.11 unsmoothed, so it is an
  opt-in behind calibration, and the relerr gate correctly rejected it as a default.
- **Approached with size**: decode speedup grows as the dispatch floor (~125us, shared with bf16) amortizes — 27B-class
  shard: int8 1.52x, int4 1.77x, asymptote = byte ratio.
- **Structurally capped in the harness min**: qkv-7B t=8 is floor-bound for BOTH paths; no single-op change can reach 2x
  there. Reaching 2x end-to-end needs either per-layer weights large enough to bury the floor (27B+ shards)
  or multi-op fusion per dispatch — a serving-integration lever, not a kernel one.

Kernel-side facts bought along the way: Mosaic CSEs repeated lhs quant (hoisting: no-op); the 128-col strip loop
genuinely pipelines VPU/MXU (removing it: -6%); f32 epilogue is free (bf16 epilogue: slower AND less accurate);
int8xint4 is not MXU-native (upcast, int8 rate).

Next integration steps: route ejkernel's dense qmm TPU dispatch to
`channelwise_quantized_matmul` for int-target specs; adapters emit per-channel int8/int4 on v5; wire `qmm_*` layer
knobs; expose W4A4 as an explicitly-calibrated opt-in.

## Autoresearch: packed-int4 decode / 4x attempt (2026-08-04)

Six iterations + two probes on `autoresearch-qmm-2x`. Deliverables in
`ejkernel/kernels/_pallas/tpu/quantized_matmul/_packed_gemv.py` (TPU parity tests 3/3): `packed_int4_gemv` (W4A16, 2
weights/byte, split-K packing) and
`w4a4_gemv` (packed weights fed to the int4 MXU via `pltpu.bitcast` — zero per-element decode). Bench:
`benchmarks/bench_packed_int4_decode.py`.

Decode vs jitted bf16, TPU v5 single chip:

| path                               | 13B gate_up | 27B gate_up                                                |
|------------------------------------|-------------|------------------------------------------------------------|
| XLA fused-upcast int4 (W4A16)      | 1.36-1.41x  | 1.76-1.78x                                                 |
| Pallas packed W4A16                | 1.15-1.18x  | 1.29-1.32x                                                 |
| **Pallas W4A4 (bitcast MXU feed)** | **1.44x**   | **2.07x** (241us, kernel-exact vs its quantized semantics) |

**4x verdict for v5: not reachable, two measured walls.**

1. *Convert bound* — any W-A16 path converts every weight element for the MXU (~1.5-1.7T elem/s through the convert
   pipeline, XLA's fused convert being the best implementation); packing bytes doesn't help because bytes are not the
   pole. Cap ≈ 1.8x.
2. *int4-MXU ingest floor* — W4A4 removes the convert entirely (bitcast is a register reinterpret) but plateaus at ~
   223us/2.2x on 27B across every tile/split-k geometry: the MXU's weight-ingest rate at tiny m is the wall.

4x lives on v6e/v7 (native sub-byte MXU feeds / faster convert) or at per-chip weights well beyond 27B-class. On v5 the
practical ladder is:
prefill W8A8 1.77x / W4A4 2.88x (calibration-gated), decode W4A4 2.07x (calibration-gated) or W4A16 1.8x (exact).

Mosaic v5 lowering facts collected: no 8-bit vector bitwise; `arith.shrui`
does not legalize (use signed shifts in int32); `bitcast(u8->int4)` expands sublane-major matching adjacent-rows
packing; CPU backend rejects int4 dot_general even in interpret mode (W4A4 tests are TPU-only).

## Push-more session: the cost model that closes the case (2026-08-07)

Follow-up probes on `autoresearch-qmm-2x` produced a complete cost model that explains every measurement of the
packed-decode investigation:

    time_per_op = fixed + bytes / 2.51 TB/s
    fixed = ~123us at a jit boundary; ~36us (native XLA op) / ~57us
    (pallas_call) inside a compiled graph.

Verified fits: w4a4 27B microbench 123+94=217us (measured 222); bf16 27B 123+375=498 (measured 497). Everything earlier
called "MXU ingest floor" or
"dtype stream penalty" was this fixed cost polluting small-array rates — the dtype-stream sweep showed all dtypes stream
identically once overhead is subtracted, and the size sweep fit fixed=123us, BW=2.51 (Pallas marginal)
/ 1.74 (XLA reduce marginal; XLA *matmul* streams at ~2.5 like Pallas).

Consequences, all measured:

- **In-graph (serving-realistic) w4a4 decode at 27B: 2.72x** (151.2 vs 411.4us/op; jit-boundary microbench shows only
  2.08-2.23x). In-graph is the number that matters for eSurge.
- `parallel` grid semantics beat `arbitrary` by ~8% for the w4a4 grid (222 vs 241us); promoted into the kernel.
  `CORE_PARALLEL` (megacore) gave nothing — Pallas single-kernel BW is the same ~2.51 TB/s marginal.
- XLA cannot read packed nibbles: `bitcast_convert_type+reshape` into a dot does NOT fuse (materializes, 6-13x slower)
  and its nibble order differs.
- Two measurement traps for future benches: XLA dead-column-eliminates weights whose outputs are sliced (a `[:, :K]`
  chain let bf16 "read" 940MB in 90us); and any chaining operand as large as the weight doubles the traffic being
  measured.

Remaining route to ~4x on v5, now exactly quantified: amortize the fixed cost over more bytes per dispatch — a fused MLP
kernel (gate_up+act+down in one pallas_call) projects (2x411)/ (57+188) = 3.4x, four fused weights ~3.8x. That is an
eSurge/layer-level integration, not further kernel tuning; the w4a4 kernel itself sits on the DMA line.

## Fused MLP op (2026-08-07): forward + backward, modular, measured

Shipped on `autoresearch-qmm-2x`: `ejkernel.modules.fused_mlp` — one surface for `down(act(gate(x)) * up(x))` across
formats (bf16 / channelwise int8 / channelwise int4 / packed-int4), layouts (separate, fused-concat, TP-interleaved —
normalized by `split_gate_up` at the boundary so kernels carry zero layout branches), activations
(silu/gelu/gelu_tanh/relu/sigmoid), and **both directions**:

* forward: Pallas single-dispatch W4A4 kernel on TPU decode shapes (`kernels/_pallas/tpu/fused_mlp`, registered
  `Platform.PALLAS`); XLA composition otherwise (`kernels/_xla/fused_mlp`, registered fallback). The Pallas kernel keeps
  hidden activations in registers, re-quantizing per- (token, I-tile) — finer than the per-token scales that failed the
  accuracy gate — and is tested kernel-exact against a tile-wise NumPy replication of its semantics.
* backward: `custom_vjp`, flash-style recompute (no saved intermediates); dense weights get true dW (parity vs naive
  autodiff < 0.5% on all four cotangents); integer codes are frozen (float0 cotangents) with exact dx — the
  LoRA/frozen-backbone training contract.

Measured (TPU v5, single chip, `bench_fused_mlp.py`), whole MLP block:

| path                                    | 13B decode  | 27B decode                                | 27B train fwd+bwd (t=2048) |
|-----------------------------------------|-------------|-------------------------------------------|----------------------------|
| bf16 single-jit / in-graph              | 303 / 207us | 689 / 589us                               | 16.3ms                     |
| w4a4 unfused (2 dispatches)             | 185us       | (VMEM OOM at tile 1024)                   | —                          |
| **w4a4 FUSED (1 dispatch, tile_i=512)** | **185us**   | **298us = 1.98x in-graph / 2.31x single** | —                          |
| int8-frozen differentiable path         | —           | —                                         | 15.5ms                     |

Honest notes (corrected after the optimize-pallas pass): the 197us "floor"
was WRONG — it used the DMA-only marginal bandwidth (2.51 TB/s) where the int4-matmul weight-ingest rate applies (~1.06
TB/s; the same wall measured for the single gemv AND for XLA's own int4 dot — all three agree, so it is the hardware's
int4-matmul ingest path at tiny m, not a schedule bug). True floor ~ 352MB/1.06 + fixed ~= 330-370us; the kernel at
295us/tile_i=1024 is AT its floor. Raising vmem_limit_bytes and quadrupling tile size moved it
~2% — step-count overhead was a misattribution, now falsified by experiment. Dense-format training runs the plain
composition under autodiff and is bit-identical to XLA (1.00x, worst grad relerr 0.0). 13B MLPs are small enough that
fixed costs still dominate (fused == unfused). Training with quantized-frozen weights is ~5% faster than bf16 fwd+bwd
(the bwd is the same dense math; the fwd saves) — real training wins need the int8-dot prefill path in bwd's transposed
products, future work. Tests:
13 CPU + 2 TPU new, all green; layering intact.

## Open decisions

- **fp8 has no matching ejkernel mode.** Our modes are affine (integer), nf4, mxfp4/nvfp4, mxfp8/nvfp8 — none is "fp8
  with arbitrary float scales". So phase 1 must pick a requantization target (affine int8 / mxfp8 group-32 / nvfp8
  group-16) or add an fp8 mode to ejkernel. This is the first real decision of phase 1 and it wants a numerics
  comparison, not a coin flip.
- **Export.** The load direction is complete; export is stubbed (`RebuildCanonical` raises). The `inverse_spliter`
  contract for M-params-to-one-value has no consumer to validate against, and the export path also casts dtypes
  unconditionally (`parameters_transformation.py:1412`)
  — the symmetric fix to `preserve_dtype`, deliberately not made without a test vector.
- **STE-QAT (phase 5):** quantized params must not receive dense optimizer state. Settle before implementing.
- Whether requant-to-wider-block is ever worth it on our TPUs, or whether group-32 MX feeds the MXU adequately. Needs
  measurement, not inheritance from vLLM-TPU's default.

## 2026-08-08 — Sharding-aware dense bf16 fused MLP (shipped, uncommitted)

**Kernel** `fused_mlp_bf16` (PALLAS/TPU + XLA/ANY ref, signature-validated):
fused-weight two-view BlockSpecs (no split materialization — a split variant loses its whole edge at model level),
megacore `("parallel","arbitrary")`, optional `save_preacts` bf16 residual outputs for a recompute-free vjp. 94-96% of
bf16 compute roofline vs XLA's 90-92% (1.08-1.09x at m>=4096, block level); 0.8x at decode m (XLA composition sits at
the DMA roofline).

**Routing** (`resolve_dense_sharded_plan`, never-slower policy, all measured on Qwen3.5-4B v5p-8 model level, batch
8x2048):

- rows-sharded / replicated weights -> pallas: fwd 1.016x, train 1.007x WIN
- tp (interleaved cols + psum)      -> composition: kernel measured tie fwd, -1.3..-2% train (psum-dominated; bwd
  re-fusion churn). Machinery kept and tested for re-eval at larger per-shard I.
- fsdp K-sharded weights -> composition (GSPMD collective-matmul; every materialized-gather integration measured ~4%
  behind: boundary reshard, in-step pre-gather, in-shard_map all_gather, one-layer lookahead ALL land within 0.3ms of
  each other; trace shows 1127ms of exposed all-gather events vs A's 3.4ms — layer bodies are custom-call-dominated
  (GDR/attention kernels), collectives cannot overlap custom calls).
- m_local < 512 -> composition (decode roofline). Size-1 mesh axes in specs are filtered (EasyDeL specs carry dead axis
  names). Concat-layout + I-sharding rejected (locally-garbage halves; interleaved layout with segments == tp required —
  caught by an adversarial test).

**fsdp ring kernel (next)**: single-core collective kernels (in-repo all_gather_matmul style) cap at half the MXU ->
cannot win. pl.when-divergent cross-core semaphores DON'T work (probe: per-core semaphore instances, nonzero-on-exit
halt). jax 0.10.0 HAS create_tensorcore_mesh/core_barrier/ run_on_first_core -> comm-core ring is buildable. Design:
grid (j, m, s-inner), per-I-tile K-piece ring HBM->HBM, gu tiles in VMEM (~90MB at tile_m 512), hop ~60us vs ~300us
compute per step. Projected ceiling vs GSPMD: 2-6%.

**Side finding**: at tp=1 the SPLIT-weight composition beats the model's fused-dot-then-split path by ~1% model-level
(268.1 vs 274.8 fwd) — that's
`separate_mlp_gate_up_proj` — worth flipping the default after a 27B check.

**Bench traps (new)**: sequential benchmarking biases the first subject (+4-12%, DVFS) — interleave rounds;
adjacent-program thermal state shifts pallas readings; a jit cache-hit "0.3s compile" means the variant reused the other
variant's trace (monkeypatched forwards need fresh function objects).

## 2026-08-08 — Zoo-wide MLP wiring (shipped, uncommitted)

`easydel/layers/mlp.py::gated_mlp_forward` — ONE shared forward the zoo delegates to. Routing: dense bf16 + supported
layout -> ejkernel fused_mlp pallas; per-channel symmetric int8 ParallelLinearQuantized -> fused channelwise integer
path (engages on any mesh — no shard_map); everything else -> byte-identical legacy composition. Specs resolve from the
model's own machinery (RuntimeShardingResolver._partition_spec_for_axis_names for HiddenStateSharding (batch,seq,hidden=
(dp/fsdp),sp,tp) with decode/train mode; weight specs from each Parameter's declared axis_names). Kill switch:
EASYDEL_DISABLE_FUSED_MLP=1.

Sweep: 38 modeling files / 41 MLP forwards converted (llama + qwen3_next by hand; 36 files by 4 parallel agents). Legit
skips: non-gated MLPs (gpt2/gpt_j/gpt_neox/phi/falcon/mpt/clip/siglip/vision towers/gidd), nonstandard attrs (arctic
w1/w3/w2, exaone c_fc_*, internlm2 w1/w3/w2, grok_1), nonstandard math (deepseek_v4 + minimax clamped swiglu, falcon_h1
muP multipliers, kimi 2-arg activation, cohere hardcoded silu, gemma-v1 hidden_activation-vs-hidden_act hazard), MoE
expert stacks. Special branches kept verbatim: qwen3_next separate/ring, deepseek_v3 + mistral4 2D moe-infer paths,
gemma4 activations_in_float32 (defaults True — delegation engages only when disabled).

Hazard hardening: fused route requires mlp.act_fn IS ACT2FN[hidden_act]
(identity) — blocks silent wrong-activation when families derive act from other fields. Test: test_gated_mlp_forward.py
(7 tests).

TPU validation through the REAL model path (no monkeypatch): llama 8-layer 2048/8192 on dp-mesh v5p — helper-on 24.22ms
vs off 24.82ms fwd = 1.025x. Stock fsdp rules -> resolver falls back (parity; awaits ring kernel). tp serving ->
composition parity by policy. Quantized route = the checkpoint-quantization formats finally reaching model forwards.

All per-family CPU tests pass or fail only with documented pre-existing classes (ragged-all-to-all MoE, qwen3_5
segfault, olmo3 transformers-env, hy_v3 sparse-call batch divisibility — each baselined via stash).

## 2026-08-08 (later) — Parameterized two-branch combines (shipped, uncommitted)

Kernel-side: fused MLP activations generalized from unary-act-times-up to a COMBINE table
(`resolve_mlp_combine(name, act_params)` in
_xla/fused_mlp/_interface.py, public via ejkernel.modules). New names:
clamped_swiglu (limit) [deepseek_v4], swiglu_oai (alpha, limit)
[minimax/gpt-oss recipe], scaled_silu (gate_multiplier) [falcon_h1 muP], situ (beta, linear_beta; 0=no linear squash;
f32 internals) [kimi K3].
`act_params: tuple[float,...]` threaded identically through all five impls (signature-validated) + module wrapper
(custom_vjp bwd = jax.vjp (combine) — generalizes with zero extra code) + sharded plan. Registry _TUNING list gained
tile_m/tile_n/tile_i/chunk (ignored-param exemption).

Helper contract: modules DECLARE nonstandard combines via
`self.fused_act_name` / `self.fused_act_params` (bypasses the ACT2FN identity guard — the model states what it computes;
validated eagerly); declared combine runs on BOTH fused and fallback paths. `output_scale`
param for falcon_h1's down_multiplier. `_act_name` scans hidden_act AND hidden_activation (gemma-v1).

Conversions: deepseek_v4 (8 passed), minimax_m3_vl (dense MLP; MoE
_sparse_call failures file-scope-baselined pre-existing), falcon_h1 (2 passed; non-silu configs keep in-place body),
kimi_linear (SITU declared; suite skips on CPU), gemma v1 (2 passed; act->act_fn rename), cohere (3 passed) + cohere2 (2
passed) (act_fn=jax.nn.silu attr added — exact original hardcoded behavior). Zoo total now 45 files / 48 MLP forwards on
the helper. Remaining off-helper: non-gated MLPs (arch mismatch), nonstandard attr names
(arctic/exaone/internlm2/grok_1 — attr-map follow-up), MoE expert stacks (separate grouped-kernel project).

## 2026-08-08 (final) — CHANNELWISE quant mode: W4A4/A8W8 reach real models

New `QuantizationType.CHANNELWISE` (bits 4|8) + `activation_bits` config field: per-output-channel SYMMETRIC codes over
full K — the storage format the fast kernels consume directly (the pre-existing affine/int8 modes are all group-wise
WITH zero-points and can never hit the fused paths). ParallelLinearQuantized: channelwise quantize branch (int8/int4
codes +
[1,N] f32 scales, no biases), int4 gets an adjacent-packed uint8 companion Parameter (quant_kernel_packed) for the fused
W4A4 decode kernel, forward routes to channelwise_quantized_matmul (now public via ejkernel.modules, with
activation_bits threading). Helper's quantized route detects the REAL format (mode=="channelwise"), reads
activation_bits from the linear's quant config, and passes packed_weights when packed >= 48MB.

TPU v5p, 4B MLP shapes (K2560 I9216), vs bf16 dense, through the helper:
W8A16 prefill 1.03x (relerr .004) | A8W8 prefill 1.58x (.045)
W4A16 prefill 1.03x (.005)        | W8A16 decode 1.17x (.004)
W4A4-config decode 1.24x (.004)   — size-gated: packed int4 kernel only engages at >=48MB packed weights (27B-class,
where it measured ~2x); below the gate an activation_bits=4 config still gets int4 storage + int4-MXU prefill dots +
exact fused-upcast decode. Ungated 4B packed measured 0.79x AND relerr 0.40 (two compounding int4 stages) — the gate is
a correctness+perf decision.

Tests: 12 helper tests incl. REAL quantized linears end-to-end (engage + dequant parity int8/int4, packed companion
bit-exact roundtrip, linear forward parity, config roundtrip with activation_bits). Combine/act_params threaded through
the integer vjp path unchanged (frozen-weight training works for channelwise formats).

## 2026-08-09 — eSurge RPA v3 device hang: root cause + kernel fix (shipped, uncommitted)

Symptom: any RPA-v3 serving run with more requests than seats hangs the device (block_until_ready on logits never
returns; scheduler heartbeat stale) right when the first requests complete. v2 attention immune. Pre-existing on clean
tree; both qwen3_5-4B (hybrid) and qwen3-30B-A3B (pure attention) reproduce.

Root cause (confirmed by ring-dump capture, EASYDEL_DUMP_BATCH_META in batch_preparer.py): rows with q_len==0 occur
mid-batch legitimately — async scheduling keeps a request that hit max_tokens in `running` with scheduled=0 until
postprocess retires it (also: budget-skips, model-len clamps). Hang step 150 capture: sched=[1x29, 0, 1065] — an empty
row 29 plus a fresh 1065-token prefill at row 30. The v3 kernel's manually pipelined DMA chain (both _pallas_impl_fwd.py
and _h64) prefetches row j's first bq/bkv from row j-1's body iterations; an empty row runs ZERO iterations, so the next
row waits on a never-started DMA semaphore → permanent device hang. The count-based-distribution fix alone could not
cure it (empty rows land in some case range regardless).

Kernel fix (the true fault site): next_nonempty_seq () scalar while-loop skips q_len==0 rows in the prefetch chain
(get_next_bq_ids / get_next_bkv_ids / sliding-window next-start), prologue targets the first non-empty row, per-row
processing guarded on q_len>0. Empty rows get no fetches, no waits, no sends — semaphore pairing stays consistent in all
orderings (empty first/mid/last/all). dynamic_validate_inputs updated: q_len==0 rows are part of the contract now.

Verified on v5p-8: (1) live capture hung at 2/48 with old kernel; (2) captured hang-step metadata replayed standalone
with fixed kernel — completes, finite; (3) 5-scenario parity suite vs ref (empty rows in every position): max out err 1
bf16 ulp, cache writes byte-exact; (4) producer-side positional prefix classification
(compute_request_distribution_prefix + 8 CPU tests) kept — correct ordering semantics for reused slots.

Also fixed: serving-bench token deficit was the BENCH's fault — random token-id prompts decoded->re-encoded inflate past
max_model_len and the engine caps max_new_tokens (log: "Capping max_new_tokens from 128 to 108 (prompt_len=1076)").
bench now builds natural-text prompts with verified exact token counts (make_text_prompts); engine exonerated.

MoE coverage note for the 30B bench: ParallelMoELinear has no quantized path — on qwen3-30B-A3B the quant combos
quantize attention/dense linears only; expert weights stay bf16.

## 2026-08-09 — Serving bench delivered: 5 combos on qwen3-30B-A3B (v5p-8, tp=4)

128 reqs x 1024 prompt x 512 decode, cc=32, RPA v3 (fixed), eSurge defaults (max_num_batched_tokens=1152). All combos:
128x exactly 512 tokens (ignore_eos+min_tokens honored — deficit bug was the bench prompt roundtrip, fixed), no hangs
across 4 admission waves per combo.

bf16 927.2 out tok/s 2781.5 total tok/s (70.7s)
w8a16 930.7 2792.0 (70.4s)
a8w8 926.1 2778.3 (70.8s)
w4a16 930.0 2790.0 (70.5s)
w4a4 932.7 2798.1 (70.3s)

Flat within +-0.6% = noise. Expected and honest: ParallelMoELinear has no quantized path, so on this MoE only
attention/dense linears quantize — expert compute (the dominant cost at 128E/8-active) stays bf16. End-to-end serving
deltas from linear-only quantization are invisible at this scale. The measured layer-level wins (A8W8 prefill 1.58x,
W8A16 decode 1.17x on 4B dense MLP shapes; packed W4A4 ~2x at 27B-class decode) require dense-MLP models or a quantized
MoE expert path (grouped-matmul quant kernel) to show up in serving. Combos ran one-per-process (in-process model reload
OOMs HBM).

## 2026-08-09 — CORRECTED serving bench (quantization actually engaged)

The first 5-combo table was INVALID: from_pretrained on a native tensorstore checkpoint silently ignores
quantization_config unless apply_quantization=True (quantized checkpoints auto-apply; fresh quantize-at-load does not) —
all four
"quant" combos ran pure bf16. Fixes: (a) bench passes apply_quantization=True and HARD-FAILS unless >0
ParallelLinearQuantized modules exist after load; (b) bridge warns loudly on config-without-flag; (c) the load path's
streaming prepack didn't know channelwise (group_size=0 ValueError) — added a channelwise branch reusing the new shared
channelwise_quantize_array ()
(single source of truth with _quantize_array), including the int4 quant_kernel_packed companion. CPU roundtrip smoke:
save tiny qwen3 → quantized reload engages 8/8, forwards finite (a8w8/w4a16/w4a4).

Corrected numbers (qwen3-30B-A3B, 128x1024x512, cc=32, tp=4, RPA v3 fixed, 144/144 attention linears
channelwise-quantized, experts bf16 — no MoE path):

bf16 927.2 out tok/s (baseline)
w8a16 935.1 (+0.9%)
a8w8 898.7 (-3.1%)  runtime act-quant overhead on small-m projections w4a16 942.0 (+1.6%)  best combo w4a4 922.1 (-0.6%)
below 48MB packed gate at these shapes

Physics: in decode at cc=32 nearly all 128 experts stream (~58GB/step bf16)
vs ~1.8GB attention-linear weights → quantized surface is ~3% of decode weight traffic → ceiling ~1.5-3%; measurements
match. The 2x-class kernel wins (packed W4A4 fused MLP, A8W8 prefill) live on dense-MLP surfaces this MoE never routes
through. Next lever for real 30B serving gains: channelwise quantized MoE expert path (ParallelMoELinear +
grouped-matmul kernel) —
~58GB/step of expert traffic is where a genuine ~2x decode win sits.

## 2026-08-09 — Profile-driven serving fix #1: replicated expert weights (SHIPPED, +35%)

xprof of the 30B decode step (cc=32, tp=4) showed 21.9ms/step with 8.1ms (37%) in per-layer
`constant_dynamic-slice_fusion` — XLA materializing each chip's tp-local slice of REPLICATED expert weights inside every
shard_map call. Root cause: leaves synthesized after the rule-based tensorstore read (native gate/up fusions, tp
re-interleaves) bypass loader placement — their sources match no live-variable rule, load replicated, and the fused
result stays replicated (66GB/chip HBM, 4x expert copies).

Fix (bridge.py, after apply_native_reform_param_fusions): re-place any post-load synthesized leaf whose sharding
mismatches its resolve_shardings_regex rule (literal-path dict + regex fallback, device_put). 30B: "Re-placed 96
post-load synthesized leaves", gate_up local (128,2048,384), HBM 66->15GB/chip, KV pages 3734->11295 (3x).

Measured: device step 21.87->12.90ms (1.70x, slice fusion GONE, expert path at streaming roofline); standard bench
(128x1024x512, cc=32):
bf16 927.2->1250.9 out tok/s (+35%), a8w8 898.7->1211.2 (+35%).

a8w8 3x ledger (target 2696 tok/s vs original 898.7): now 1.35x. Remaining levers by size: (1) HOST GAP ~12.7ms/step
(wall 25.6ms vs device 12.9ms) — the largest single item; (2) quantized experts int8/int4 (ragged-dot 4.4ms ->
2.2/1.1ms); (3) tp collectives 2.5ms; (4) RPA decode 2.2ms.

## 2026-08-09 — Fix #2: tool-parser O (n^2) prescreen (shipped) + a8w8 ledger

Host profile showed the output pipeline re-tokenizing the FULL accumulated text on EVERY streamed delta inside
DelegatingParser._process_tool_delta (O (n^2)/request; ~165ms encode per 215ms window at cc=32) even for requests that
never emit tool syntax. Blanket-gating on _is_tools_enabled () broke 3 tests (contract: tool parsing runs whenever a
parser is configured, even with tool_request=None). Shipped fix: marker PRESCREEN — DelegatingParser caches
tool_parser.get_streaming_buffer_hints () and skips the machinery until a full hint appears in the cumulative text (or
its tail is a proper prefix of one); empty hints fail OPEN (custom parsers keep old behavior). 450/450 parser tests
pass.

Bench: a8w8 1211.2 -> 1218.1 (+0.6% only — the output thread overlaps the device; it was saturated, not binding, at
512-token generations). Still correct: real win grows with generation length (O (n^2) killed) and frees a core.

a8w8 ledger vs 898.7 baseline: 1218.1 = 1.355x. Device step 12.9ms; 3x (2696 tok/s) needs ~11.9ms step-equivalent WALL —
requires int8/int4 expert weights (ragged-dot 4.4 -> 2.2/1.1ms), residual host-gap work, and collective trims. NEXT
BUILD: quantized grouped-matmul expert path (channelwise per-expert int8/int4 codes + [E,1,N] scales; decode = Pallas
expert-streaming kernel with in-VMEM dequant; prefill = dequant+ragged_dot v1; routing on static token count in _
expert_ffn).

## 2026-08-09 — Quantized experts SHIPPED via grouped_matmulv3's native feature

Per user directive ("use correct gmm / add as feature, don't invent kernels"): deleted the redundant standalone Pallas
quant gmm; experts now route through grouped_matmulv3 (existing kernel) with int8/int4 codes + rhs_scale [G,1,1,N]
(channelwise = its num_blocks=1 case). KEY FINDINGS:
(1) v3's auto-tiler is ~10x too slow at MoE decode shapes; explicit whole-K/whole-N tiles (block_m 64 decode / 128
prefill) fix it: 113us int8 / 101us int4 at m=256 vs 212us bf16 ragged_dot; 177us vs 265us at m=9216. (2) v3 ALREADY
does a8: maybe_quantize_lhs defaults True → in-kernel int8 lhs quant (per-512-block scales) when rhs is int-quantized;
a4 pointless on TPU (no int4 MXU, acts are ~1% of traffic). (3) Mosaic requires 128-aligned slice shapes: tp-local down
k=192 can't take explicit whole-K tiles → block_k/n=0 sentinel added to the modules wrapper to defer to the kernel's own
tiler (wins prefill 198vs267, loses decode 175vs124 — net positive per layer; k%128 decode tiling is a follow-up).
grouped_matmul_w8a8 (native int8 ragged_dot composition) kept as XLA reference, not routed. ParallelMoELinearQuantized
supports bits 4|8.

Bench (cc=32, 128x1024x512, all fixes stacked):
w8a16 1418.7 | w4a16 1401.3 | a8w8 1389.2 | bf16 1250.9 | orig-a8w8 898.7 w8a16 = best = 1.58x vs original a8w8
baseline. int4-vs-int8 wall parity proves the HOST GAP (~9-10ms/step vs ~11ms device) is now the binding constraint at
cc=32; also cc=32 structurally can't amortize the 128-expert weight sweep (~2.6ms/step int8, batch-independent) —
cc=128-256 is the big multiplier (fits post-sharding-fix; user deferred to avoid recompiles).

## 2026-08-09 (late) — Host-gap hunt blocked: xprof wedges on the v3-experts build

Three capture attempts of the w8a16 quantized-experts build with engine start_profiling (host_tracer_level=2, python
1|2, 30|120 batches)
never auto-stop: no trace file, no tqdm progress, no heartbeat-stale warning, engine loop apparently stalled under
tracing. All PRE-expert builds traced fine with identical knobs; suspicion: jax profiler host tracing x grouped_matmulv3
Pallas launches (TraceMe interaction). OPEN ITEM. Next tool for the host gap: env-gated perf counters inside
loop.py/execution_manager (per-phase wall timers), not xprof.

Session ledger at cc=32 (user-fixed): 898.7 -> 1418.7 out tok/s (1.58x)
via sharding fix + v3 quantized experts; device ~11ms vs wall ~22ms — host gap remains the binding constraint; cc
scaling (128-256, now fits in HBM) is the other 2-4x, deferred to avoid bucket recompiles.

## 2026-08-09 — Decode-phase truth from runner [perf] probes (w8a16, cc=32)

Decode phase measured 10.3ms/step wall = 3,098 tok/s (NOT bench-average 1418 — prefill waves + transitions drag the
bench). Per step: device 4.6ms (drain wait) + host ~5.7ms SERIAL (prep 1.06, jit dispatch 2.50 ≈ 3.3us/leaf x ~1000+
leaves — quantization added ~240, sampler enqueue/drain/schedule rest). One-step-ahead pipeline alternates host/device
instead of overlapping; the device got fast enough tonight that the host became the wall. Prefill warm ≈
84-90ms/1024-token chunk (near device-bound); the 300-1250ms "outliers" are queue-backlog accounting in admission
bursts. Greedy argmax fastpath already used on 484/495 steady decode steps.

REMAINING FIXES (both engine-side): (1) cut host/step below 4.6ms — leaf folding (stack per-layer quant codes/scales) +
prep trim; forces one bucket recompile cycle. (2) pipeline depth-2 (scheduler runs two ahead) — no recompile, pins wall
to device 4.6ms ≈ 6.7k tok/s decode at cc=32; real scheduler surgery on the async placeholder machinery.

## 2026-08-09 (final) — CORRECTION: decode is DEVICE-bound; host already hidden

The "host serialization" model was wrong. Re-derivation: pre-experts device step 12.9ms − measured expert savings ≈
2.3ms → ~10.6ms device, matching the 10.3ms decode wall exactly. The drain's 4.6ms wait is the UNOVERLAPPED TAIL of the
device step; the loop's enqueue-before-drain overlap works. Depth-2 pipelining and host-leaf work would buy ~nothing at
cc=32. Remaining decode levers are device-side and structural: TP collectives ~2.5ms (latency-bound, 2/layer), RPA
decode 2.2ms, routing 1.2, misc fusions ~1.9, experts ~2.5 (already int8). The big multiplier stays concurrency (cc
128-256).

Also: jax profiler (even device-only) WEDGES serving on grouped_matmulv3 builds — engine loop stops progressing under an
active trace, no heartbeat-stale (3x host-level variants + 1x device-only, all reproduce; pre-v3-experts builds trace
fine). Real bug, unresolved; use runner
[perf] logs (runner_verbose) for this build instead.

FINAL cc=32 ledger: bench 898.7 -> 1418.7 out tok/s (1.58x); decode-phase 3,098 tok/s device-bound; prefill ~
84-90ms/1024-chunk warm.

## 2026-08-09 (close) — Collective-dedup campaign: audited, foundation laid, blocked on mode plumbing

HLO audit of the decode step (32/32 bucket): 435 collectives/step, ~9/layer; only ~2/layer are legitimate (o_proj
all-reduce + MoE down psum_scatter). The other ~7/layer are self-inflicted by TP-SHARDED DECODE ACTIVATIONS:
97x RMSNorm mean-square all-reduces, 48x pre-qkv all-gathers, 48x norm-mul all-gathers, 96x router softmax max/sum
all-reduces, 48x router gathers. Removing them = est +25% decode (2.5ms of the 10.3ms step).

Foundation SHIPPED (kept, all CPU-verified, 1407 inference tests pass):

- spectrax: decode_hidden_state_axis field + generation map entry + None now honored as a real decode override (was
  silently skipped — also fixes the documented-but-inert decode_query_sequence_axis=None collapse).
- easydel base_config: decode-mode hidden replicated by default (_with_replicated_decode_hidden;
  EASYDEL_DECODE_HIDDEN_REPLICATED=0 opt-out).
- eSurge model call sites pass mode=MODE_DECODE (EASYDEL_ESURGE_GENERATION_MODE=0 opt-out).

BLOCKED: compiled decode program still shows 435 collectives — per-layer constraint sites (norm/attention/router
apply_logical_sharding) re-derive mode from SHAPES, and eSurge's packed [1, N] batch always looks like TRAIN. The
top-level mode kwarg doesn't reach them. Follow-up: thread mode through decoder-layer/norm constraint sites across the
modeling stack (or make apply_logical_sharding honor an ambient generation-mode scope set by the runner). Also:
minimax_m3_vl spmd esurge parity test (8 params) fails on the CLEAN tree — pre-existing, unrelated (bisect-verified).

## 2026-08-09 (final close) — Ambient decode-spec forcing: built, measured, rejected

Built the one-change path: RuntimeShardingResolver._resolve_mode returns MODE_DECODE whenever easydel's
set_inference_mode () context is active (trace-safe, JIT-cache-keyed, already wraps eSurge model calls). Verified the
forcing works (explicit TRAIN -> __autoregressive__ inside the scope); CPU eSurge suite green (898 pass, only the
pre-existing minimax spmd fails).

TPU measurement REJECTED it: collectives 435 -> 482 (all-gather 146->194), drain 4.56 -> 4.96ms, step unchanged 10.3ms.
Root cause: the dominant collectives are anchored by the MoE shard_map OUT_SPEC (down psum_scatter emits tp-sharded
output; the following norm's all-reduce operates on that sharded dataflow regardless of constraint modes) and o_proj
partial sums. Replicated constraints just added gathers on top. The real +25% requires mode-aware MoE boundary
(all-reduce out at decode instead of psum_scatter+gather) and o_proj psum placement — scoped as the follow-up. Hook kept
but OPT-IN (EASYDEL_INFERENCE_DECODE_SPECS=1), default off with the measurement documented in-code.

## 2026-08-10 — Env flags converted to code args / context scopes (user rule)

Per user: no os.environ knobs for new surfaces. Conversions:

- EASYDEL_DECODE_HIDDEN_REPLICATED: DELETED. The default now lives in the spectrax field itself —
  PartitionAxis.decode_hidden_state_axis resolves to None (replicated) unless explicitly set; the constructor arg IS the
  knob. easydel base_config helper removed (base_config untouched again).
- EASYDEL_INFERENCE_DECODE_SPECS: replaced with the mesh-scope-style
  `easydel.infra.sharding.decode_mode_specs()` context manager (ContextVar); the measured-worse experiment stays
  available as an explicit opt-in scope.
- EASYDEL_ESURGE_GENERATION_MODE + the eSurge mode= kwarg pathway: DELETED entirely (measured inert; future mode-aware
  MoE-boundary work will reintroduce mode passing properly).
- EASYDEL_DUMP_BATCH_META: replaced with set_batch_metadata_dump_dir (path)/get_batch_metadata_dump_dir () in
  batch_preparer (debug API).
- EASYDEL_DISABLE_FUSED_MLP: replaced with config attr
  `config.use_fused_mlp = False` in gated_mlp_forward.
- EJKERNEL_AUTOTUNE_POLICY kept — pre-existing documented ejkernel convention used across operations/.

Verified: no new getenv in the diff; PartitionAxis default None + explicit
'tp' honored; decode spec P ((fsdp,dp), None, None); scope + setter APIs work; 90 focused tests pass; tiny-MoE forward
finite.

## 2026-08-10 — SparseCore gate OPEN on v5p + ejkernel collectives family scoped

Also fixed today: the pre-existing minimax_m3_vl eSurge SPMD failure (8
params) — fused-MoE shard_map batch specs now shape-sanitize COORDINATED
across x/gate-logits/output when the batch can't divide the ('dp','fsdp')
group (eSurge packs [1, N]; batch-replicated body is numerically identical).
All VLM serving tests pass; qwen3_moe spmd fails remain the documented
XLA:CPU ragged-all-to-all limit.

vLLM tpu-inference study (user-directed):
- kernels/sparse_core: MoE permute/combine on SparseCore (ragged_scatter,
  ragged_gather_reduce_v2) via jax.experimental.pallas.tpu_sc + core_map.
- kernels/collectives: all_gather_matmul (bidirectional ring, remote DMA,
  tuned blocks) + hierrs_sc (hierarchical reduce-scatter, SparseCore DMA
  pipeline).
Maps 1:1 onto our measured decode residue: routing machinery ~1.2-1.8ms +
collectives ~2.5ms of the 10.3ms step -> est +25-35% combined.

PROBE (v5p, jax 0.10.0): tpu_sc imports; VectorSubcoreMesh(num_subcores=16)
constructs; core_map kernel with per-subcore async DMA COMPILES AND RUNS
CORRECTLY. Constraints learned: vector subcores are 8-wide f32; loads only
from core-local memory (explicit pltpu.make_async_copy); 16 subcores/core.

PLANNED ejkernel collectives family (user-requested; kernels live in
ejkernel, NOT spectrax — layering + existing all_gather_matmul/ring_attention
precedent): all_gather, all_reduce/psum (incl. latency-optimized small-tensor
variant for 131KB decode messages), reduce_scatter, matmul_reduce_scatter
(missing fused twin), sparse_all_gather / sparse_reduce_scatter / sparse_psum
(SparseCore-driven), ragged_all_to_all. Plus sparse_core MoE permute/combine
ports. easydel moe/attention pick via registry; spectrax untouched.

## 2026-08-10 — Unified collectives P1 SHIPPED + the measured verdict on custom TP collectives (v5p-8)

Shipped (ejkernel, all gates green: 17 TPU + 33 CPU tests, sig-parity,
benchmark specs): new op ids `all_reduce` / `all_gather` / `reduce_scatter` —
registry-registered (PALLAS/TPU + XLA/ANY), module wrappers with
Executor/autotune + mesh/manual shard_map duality, custom_vjp transpose
pairs (AR is self-dual — under check_vma=False the replicated-output
cotangent arrives as dy/tp so bwd MUST psum, identity bwd is 4x-wrong;
AG<->RS mutual duals). One-shot direct-exchange Pallas kernels (single grid
step, all-peer double barrier, static offset-slot indexing, symmetric
send/recv sem idiom, f32 accumulation).

MEASURED VERDICT (v5p-8, tp=4, chained steady-state, min-of-5):
- one-shot AR vs lax.psum: LOSES everywhere (64KB: 13.3 vs 12.2us; 512KB:
  18.2 vs 14.8; 2MB: 65.5 vs 34.8). One-shot moves 2x the wire bytes of
  RS+AG and pays barrier+staging; XLA is at its launch/fence floor on 4
  directly-connected chips. `mode="auto"` now resolves to the XLA path.
- fused all_gather_matmul / reduce_scatter_matmul (the previously UNWIRED
  ring kernels) vs plain lax compositions inside shard_map:
    ROW bf16 m2048: 166 vs 146us; m8192: 1356 vs 1156us (rsmm bf16 is
      internally dot+psum_scatter fallback — no pipeline).
    COL bf16 m2048: 122 vs 79us (55% slower).
    ROW f32: 601 vs 237 / 6253 vs 2188 (2.5x slower).
    COL f32: 209 vs 91 / 756 vs 322 (2.3x slower).
  XLA's async collective scheduler already overlaps AG/AR with MXU work.
  THIS IS WHY THE FUSED KERNELS WERE NEVER WIRED. Do not wire them for
  speed on single-slice v5p; they may only be revisited for multi-hop
  topologies (v5p-16+, multi-slice) with fresh measurements.
- Also dead: "zero-comm backward" matmul_allreduce sweetener — row-parallel
  linear ALREADY has a collective-free backward natively (grad_x = dy @
  w_local.T, grad_w = x_local.T @ dy, both local).

Live perf levers that survive the evidence: (1) collective REMOVAL
(mode-aware MoE boundary, audited ~+25% decode), (2) SparseCore OFFLOAD
(concurrency with TC, not per-op wins), (3) ragged_all_to_all custom vjp
(functional gap for MoE training), (4) DCN/multi-slice hierarchy (unmeasured
here, needs pods). The unified op family + ParallelLinear.distributed_matmul
seam ship as ARCHITECTURE (one differentiable comm API, training+inference
one code path) with XLA engines and default "none" — no perf claim.

RS-matmul bf16 "parity failure" during benching was a tolerance artifact
(max diff 0.5 at ref scale 18 = bf16 rounding, uniform across M-quarters);
f32 is bit-exact.

Benchmark-gate debt (pre-existing, tracked): 16 op ids missing
OpBenchmarkSpecs (incl. this campaign's fused_mlp*/channelwise/gmm-quant),
3 spec entries missing scripts, 3 stray bench_*.py files.

## 2026-08-10 — P2 closed (qwen3_next ring replaced) + SC CONCURRENCY VERIFIED

P2 re-scoped on evidence and closed:
- lax.ragged_all_to_all ALREADY differentiates in jax 0.10 (trace-probe) —
  no custom-vjp op needed; the a2a "functional gap" was a phantom.
- _communication_utils / quantized-linear re-backing onto unified ops:
  SKIPPED deliberately — those utils are already the choke point; swapping
  lax→executor adds overhead for zero functional change.
- qwen3_next hand-rolled ppermute K-streaming ring REPLACED by shard_map
  AG+dot via the unified ejkernel all_gather (first real consumer).
  MEASURED (v5p-8 tp4 bf16): decode [16,1,2048]x4096 18.5 vs ring 22.8 vs
  GSPMD 24.8 us; prefill [8,1024,2048] 237 vs ring 319 us. Also better bf16
  numerics (one f32-accumulated einsum vs ring's bf16 outer adds; f32
  bit-identical). Renamed _qwen3_next_tp_ring_* → _qwen3_next_tp_sharded_*.
  qwen3_next CPU fails unchanged = documented XLA:CPU ragged-a2a limit
  (verified identical on pre-edit file via stash bisect).

Seam (P1 easydel side) shipped: ParallelLinear.forward reads
distributed_matmul (None→einsum decline protocol);
layers/linears/_distributed_matmul.py make_distributed_matmul (row→
shard_map dot + unified all_reduce; column→None on canonical layout);
EasyDeLBaseConfig.collective_matmul_impl ("none"/"xla"/"pallas_ring",
default none, both init sites + TypedDict);
EasyDeLBaseModule.wire_distributed_matmul() via spx.iter_modules.
Validated: 9 seam tests CPU+TPU, model logits parity, jitted
split_module/merge_module grad parity wired-vs-unwired BIT-EXACT on v5p.

SC CONCURRENCY PROBE (v5p, jax 0.10, the P3 premise): core_map SC kernel
(per-subcore 8-wide DMA loop) + independent TC matmul chain in ONE jit:
SC alone 368us, TC alone 533us, both 604us (sum=900) → 81% of SC work
hides behind TC compute. SparseCore genuinely executes concurrently.
Working incantation: pl.run_state(inner) over (x, out) refs;
@pl.core_map(VectorSubcoreMesh(core_axis_name=..., subcore_axis_name=...,
num_cores=1, num_subcores=16), scratch_shapes=[pltpu.VMEM((8,), f32),
pltpu.SemaphoreType.DMA]); Get/compute strictly (8,)-shaped f32;
fori_loop rows; pltpu.make_async_copy per row. plsc API surface includes
load_gather, store_scatter, addupdate_scatter, sort_key_val,
parallel_loop, fetch_and_add — the MoE permute/combine primitive set.

## 2026-08-10 — collective_matmul_impl default flipped to "auto" (user-directed) + auto-wiring

- Default now "auto" (both base_config init sites). "auto" resolves to the
  measured-best engine: shard_map local-dot + LAYOUT-MATCHED finisher —
  reduce_scatter when runtime_sharding_resolver keeps the hidden feature dim
  tp-sharded (canonical layout: P(('fsdp','dp'),'sp','tp') — RS is half the
  wire bytes of AR and matches GSPMD's own choice), all_reduce otherwise.
- Engine is layout-aware: leading (batch/seq) in_specs come from the
  resolver at CALL time, so dp/fsdp/sp-sharded training batches stay sharded
  (a naive P(None, tp) spec would have force-replicated the batch).
  StageMesh unwrap via resolve_stage_mesh (pick_array_mesh can return
  SpxMesh, which raw shard_map rejects).
- Auto-wiring: bridge.from_pretrained (native + torch loaders) and
  BaseAutoEasyModel.from_config now call model.wire_distributed_matmul();
  direct constructor calls remain manual (tests unchanged behavior).
- VALIDATED (v5p-8, llama-2B-class 24L tp4 bf16): jitted train step
  wired-vs-unwired BIT-EXACT (loss + grad-norm deltas 0.0); step 28.51 vs
  29.65 ms; compile 25.98 vs 26.48 s. Earlier fwd-only: decode 3.470 vs
  3.497 ms, prefill identical; compile +0.4-0.5s. 11 seam tests green
  (incl. new fsdp2xtp4 layout test), lint-imports clean.
- Blast radius note: every from_pretrained/from_config model now compiles
  with shard_map row-linear islands (bit-exact, timing-neutral measured);
  eSurge bucket compile may grow slightly — watch next full server spin-up.
- SparseCore: NOT used by any engine yet — "sparse_core" is the reserved
  next engine value once the P3 ragged gather/scatter kernels are built.

## 2026-08-10 — P3 SC gather verdict (measured) + vLLM comparison

Built and ran a from-scratch SC ragged row-gather on v5p (core_map, 64
workers = 4 SC cores x 16 subcores, per-row HBM->HBM DMA with traced scalar
indices extracted from loaded int32 VMEM vectors — the vLLM API pattern,
reference-only). CORRECT at all sizes. Performance:
  [8192x2048 f32]  SC 978us  vs TC x[idx] 43us   (23x slower)
  [32768x2048 f32] SC 3829us vs TC x[idx] 522us  (7x slower)
Deepening the in-flight DMA window (W=4 vs 16 chunks, rolling waits) changed
NOTHING (delta < 0.1%): the bottleneck is the SC scalar sequencer's
per-descriptor issue cost (~7.6us/row), not wait serialization. Even at
vLLM-grade ~1us/descriptor, SC cannot approach TC's vectorized gather
(~0.005us/row, near-bandwidth). vLLM's own kernels (a) fall back to TC when
input+output < 60% of TC VMEM, and (b) tune col sizes for TPU gen 6/7 —
their SC bet is for newer-gen SC hardware and very large tables. VERDICT:
SC MoE permute/combine offload does NOT pay on v5p gen-5; the SC niche here
is strictly concurrency with independent TC work (81% overlap verified),
which the MoE critical path does not have.

vLLM tpu-inference inventory vs ours (comparison for the record):
  sparse_core/: core_map_helper (pl.kernel replacement via core_map — we
    reconstructed equivalent run_state+core_map pattern), ragged_scatter
    (semantically a GATHER for MoE unpermute; TC-sort preprocessing on host
    side + packed-dtype merge logic), ragged_gather_v2, gather_reduce +
    ragged_gather_reduce_v2 (top-k combine w/ cost-model partitioning,
    emit_pipeline over (blocks, cores, cols), SMEM scalars, uint32 bitcast
    packing for bf16 under TC tiling), dense_gather_reduce.
  collectives/: all_gather_matmul (+ tuned block tables) — we HAVE this
    kernel class and measured it SLOWER than XLA on v5p-8 single slice;
    hierrs_sc (hierarchical RS with SC DMA pipeline, config/topology/
    dma_pipeline) — multi-slice/pod territory, unmeasurable here.
Key API learnings absorbed (no code ported): scalar extraction from loaded
int32 VMEM vectors works; use_tc_tiling_on_sc + disable_bounds_checks
compiler params; single-sem FIFO byte-count waits; pl.loop for register
pressure; pltpu.get_tpu_info() for SC geometry (v5p: 4 cores, 16 subcores,
8 lanes, 32B granule).

P3 shipped instead (honest scope): op ids `ragged_gather` and
`ragged_gather_reduce` in ejkernel — XLA impls (negative-index invalid
convention, f32 weighting/accumulation, natively differentiable), the
registry landing pad for future-gen SC engines. No SC registration (would
ship a measured-slower engine); no easydel MoE rewiring (neutral churn —
sort-based path stays; these ops are the extension surface).

## 2026-08-10 — P3/P4 CLOSED: full unified collectives family shipped

Final op-id inventory (all registry-registered, sig-gated, benchmark-spec'd,
tested — 48 CPU + 17 TPU tests green, lint-imports clean):
  all_reduce, all_gather, reduce_scatter  — Pallas TPU one-shot (opt-in) +
    XLA engines; custom_vjp transpose duals; NOW MULTI-AXIS (axis_name
    accepts tuple — the "hierarchical" capability in its honest XLA form:
    lax composes per mesh topology; one-shot kernels are single-axis and
    raise on tuples). This is what MoE utils psum over ('fsdp','ep') needs.
  all_to_all                              — XLA (lax.all_to_all, native AD:
    transpose = inverse exchange); mesh-mode wrapper (in: concat_axis
    sharded, out: split_axis sharded). Sequence-parallel head exchange +
    dense MoE token exchange primitive.
  ragged_gather, ragged_gather_reduce     — XLA (negative-index invalid
    convention; f32 accumulate; native AD gather<->scatter-add + weight
    row-dot grads). MoE permute/unpermute + fused top-k combine API; the
    landing pad for newer-gen SC engines (v5p SC measured 7-23x slower —
    do not register an SC engine on gen-5).
  all_gather_matmul, reduce_scatter_matmul — pre-existing ring kernels,
    measured slower than XLA on single slice; explicit opt-in only.
P4 deferred-to-pod items (unmeasurable on single slice, per discipline):
  DCN-hierarchy-specific engines and any multi-slice perf claims. Dense
  barrier op: skipped (no consumer — dead-knob rule).

## 2026-08-10 — Mode-aware MoE decode boundary: BUILT, ENGAGED, MEASURED NEUTRAL (the +25% estimate is refuted)

Built the full mode-aware decode layout (the audited follow-up):
- Fused-MoE boundary: inside decode_mode_specs() the shard_map keeps the
  cheap psum_scatter on the top-k-expanded rows and adds ONE small
  all-gather on the combined [tokens, H/tp] output; out_specs + the final
  boundary constraint flip to replicated hidden (producers now match the
  decode residual layout — the thing whose absence made constraint-only
  forcing WORSE, 435->482).
- shard_attention_prod(pre_projection=True) skips the pre-o_proj hidden
  constraint in decode scope (kills the wasted head all-gather).
- Router logits constrained replicated post-gate in decode scope (kills the
  softmax max/sum ARs + div gather).
- eSurge executor: compile_model_step / compile_pipeline_model_step /
  compile_backbone trace pure-decode buckets (num_tokens == padded_num_reqs
  — derivable from the bucket key, so the compile cache stays consistent)
  under decode_mode_specs().

CPU verification (qwen3_moe tiny, tp=4): collectives 20 -> 9 on a 2-layer
forward (9/layer -> 4/layer: embed gather + per layer o_proj AR + MoE RS +
boundary AG + tiny router weight-gather); logits parity EXACT (2e-7). MoE
60 + runners 70 + attention 10 + seam 11 tests green.

TPU A/B (v5p-8, 30B w8a16 tp4 cc32, runner [perf] medians over ~500
pure-decode windows, engagement CONFIRMED via trace-time log):
  shallow (64-prompt):  step 3.78 vs 3.75 ms, fwd 2.40 vs 2.36 — WASH
  deep (1024-prompt):   step 3.74 vs 3.74 ms, fwd 2.34 vs 2.35 — WASH
  outputs BYTE-IDENTICAL both ways (greedy token parity on the real path).
VERDICT: the audit's "+25% est" (xprof op-category time) conflated
collective op-time with critical-path time — XLA's async scheduler already
hides these small collectives behind adjacent compute on single-slice v5p.
Removal is PERF-NEUTRAL here. KEPT (default-on for pure-decode buckets):
byte-identical, ~240 fewer collectives/step, and small-collective latency
is far higher on multi-host ICI/DCN where this should matter — re-measure
on a pod. Debug breadcrumb: logger.debug "MoE decode-replicated boundary
active" at trace time.

METHODOLOGY TRAP (cost one false bisect): jax pjit LOWERING CACHE is keyed
on (callable identity, avals) — reusing one `fwd` function across
scope-on/scope-off jit.lower() calls silently serves the FIRST trace to the
second. Define the traced fn fresh per variant. Production is safe (bucket
shapes differ => distinct keys; shape implies mode).

Also this probe harness measures decode at ~3.75ms/step / ~7500 agg tok/s
(32/32 bucket, both context depths) — the campaign's 10.3ms figure came
from the 128-req bench conditions; use in-harness A/B only.

## 2026-08-10 — Async/pipelined: comm verdict closed, host-prep pipelining scoped

User asked for async+pipelined communications. Evidence chain closes it on
single-slice v5p: (a) TPU HLO carries plain all-reduce ops — asyncness
lives BELOW HLO in the TPU backend scheduler (unlike GPU -start/-done
pairs); (b) the ring fused kernels ARE the pipelined form and lose 1.1-2.5x
(they fight the backend scheduler); (c) the removal A/B is the UPPER BOUND
on any comm optimization — deleting ~240 collectives/step moved nothing, so
the decode collectives already cost ~zero critical-path time. Comm
pipelining becomes live again only on pods (hierarchical modes + ring
kernels are built and waiting).

The async/pipelined lever that DOES pay on this box (measured): decode
window = fwd 2.34ms device + 1.01ms SERIAL host prep + 0.36 gap + 0.19
post = 4.26ms @ ~7500 tok/s. The batch preparer's double-buffered
start_async_prep/get_async_prep_result API is BUILT BUT UNWIRED (zero call
sites — same pattern as the fused kernels were). Wiring prep(N+1) to
overlap fwd(N) (the overlap loop already prefetches schedule(N+1) before
wait(N)) targets ~+30% decode throughput. Full build plan in task #22 —
sized for a fresh session: it rewires the hottest state machine in the
engine (window loop, spec-decode/VLM/PP interleavings) and needs the
validity-key + invalidation design done carefully.

## 2026-08-10 — "Fix our comm kernels" campaign: decomposed, attempted, closed

User directive: make OUR collective kernels win (async/pipelined). Results:

1. One-shot AR anatomy (chained 64KB bf16, us/op): lax.psum 13.10; ours-full
   14.59; single-barrier 13.81; NO-BARRIER-AT-ALL 13.52; barrier-only floor
   11.06. The double barrier costs ~0.8-1.1us (vLLM's util.local_barrier has
   the identical double barrier for the identical collective_id-reuse race).
   Even with ZERO synchronization our data path loses to psum — deficit is
   the pallas custom-call launch floor + serialized DMA issue, not sync.
   Standalone one-shot is UNWINNABLE at decode sizes on v5p. (Chain-parity
   flags in the anatomy probe are f32-accum-vs-bf16 drift compounded over
   100 iterations, not correctness failures — single-op parity remains
   test-verified.)

2. Fused ring kernels' real defect found: SINGLE-CORE execution (no
   dimension_semantics → no megacore; XLA matmuls use both TensorCores —
   explains the f32 2.3-2.5x losses). Naive fix (parallel N axis) DEADLOCKS
   the device (DMA chain conditions split across cores; 10-min hang,
   recovered). Proper fix = direction-split redesign (left ring on core 0,
   right on core 1) but is blocked by undocumented megacore <-> remote-DMA
   semaphore semantics (which core's sem does a remote signal land on?).
   DECISIVE ECOSYSTEM EVIDENCE: vLLM tpu-inference's all_gather_matmul has
   the SAME grid (tp+2, grid_n, grid_k) and ZERO dimension_semantics —
   Google's own TPU kernel team ships it single-core too.

CONCLUSION: on single-slice v5p every comm avenue is now measured or
blocked: replacement (slower), removal (neutral), barrier optimization
(floor-bound), megacore (deadlocks/unsupported), SC offload (7-23x slower).
Our collective kernels' value is the unified differentiable API +
pod-scale readiness. Remaining REAL speedup levers on this box, in order:
(a) host enqueue cost — decode step is ALL HOST (fwd=2.34ms enqueue of
    ~700 leaves at ~3.3us/leaf; device fully pipelined behind it): fewer
    arg leaves (scan_layers loads+compiles fine on the 30B MoE — killed
    mid-probe when user noted hybrids can't scan; still valid for
    non-hybrid serving) or leaf-count reduction in the compiled signature;
(b) prep pipelining, task #22 (1.01ms serial host prep, API built+unwired);
(c) cc=128-256 (user-gated compile cycle).

## 2026-08-10 — Incremental decode batch-prep SHIPPED: +6.9% decode (first measured win of the campaign)

User picked lever (b). Threading analysis first: single-threaded reordering
buys nothing in the all-host regime, and the retired-executor history says
cross-thread dispatch is unreliable — so the landed form is a CONTENT-KEYED
steady-decode cache in BatchMetadataPreparer (no threads, no prediction):

- In pure decode the only per-step-changing metadata is (seq_lens, input
  token, position) per request. `_try_incremental_decode_prepare` content-
  compares scheduled/active/6 sampling arrays/window rows/buckets/
  page_table_version against snapshots primed by the last full prep; on a
  hit it rebuilds ONLY packed_qsl_seqlens + input_ids + positions
  (vectorised, one small batched device_put) and dataclasses.replace()s the
  primed BatchMetadata template. Any change whatsoever -> full path, which
  re-primes. Safe because metadata is never donated (only kv_pages — the
  documented invariant at the payload-put cache).
- Ineligible by construction: SPMD-DP, slot-indexed state, VLM inputs,
  spec-recurrent commit, v2 slot-mapping, dump-dir, multihost broadcast.

Standalone harness (preparer-only): full 556us -> incremental 337us; field-
by-field equality with a fresh full prep at every step; invalidation on
page-version, sampling-param, and mixed-schedule changes all verified.
In-tree tests: tests/inference/esurge/runners/test_incremental_decode_prep.py
(3 tests) + full 70-test runner suite green.

TPU ENGINE A/B (same probe harness as the day's baselines): step 3.78 ->
3.51ms median, agg_tps 7440 -> 7951 (+6.9%), prep 1.01 -> 0.82ms
(prep.host 0.21 -> 0.02ms), OUTPUTS BYTE-IDENTICAL to the pre-change run.

Remaining floors in the decode window (3.51ms, all host): exec-enqueue
2.30ms (~700-leaf arg marshaling — the packed-args design: concat param
leaves into a few mega-buffers at the executor boundary, slice back inside
the jit; model-agnostic, hybrid-safe; est. -1.5ms -> ~2.0ms window ≈ +75%)
and the 0.41ms replicated device_put floor (3 tiny arrays x 4 devices).

## 2026-08-10 (later): packed-args enqueue — StatePacker at the executor boundary

Built the packed-args design: `esurge/runners/executors/state_packing.py`
(`StatePacker`) groups the flat graphstate/graphother leaves by
(shape, dtype, sharding-spec, memory_kind, mesh) equivalence class and
stacks each class into ONE `[group, ...]` device buffer with
`P(None, *orig_spec)` sharding; unpack inside the compiled fn is static
axis-0 slices (views). Model-agnostic: hybrids/quantized/VLM families form
their own classes; leaves without >=4 identical siblings pass through.

Key memory constraint discovered during design: packed copies COEXIST with
the originals (model/runner keep graphstate references), so packing is
budgeted — groups picked greedily by leaves-eliminated-per-byte under a
1.5GB cap (`_DEFAULT_MAX_PACKED_BYTES`). This packs the many small
per-layer tensors and auto-skips stacked-MoE expert tables (small dispatch
share anyway). Budget also bounds the transient stack memory at engine
init, which happens AFTER the KV pool is sized.

Integration (model_executor.py): packers built in __init__ from the live
templates (SPMD only); `set_runtime_graph_args` packs (repack on weight
hot-swap comes free); `_graph_call_args` converts compile-time pytrees with
an identity fast-path reusing the runtime packed tuples; `_graph_arg_sharding`
maps in_shardings through `packer.packed_shardings`; `_model_step` and
`_lm_head_step` unpack before tree_unflatten. Backbone fn takes the nested
pytree (unchanged); MPMD path untouched (packers None).

CPU: 4 unit tests (roundtrip-through-jit, hot-swap repack, budget exclusion,
min-group passthrough) + full 77-test runner suite green; packing verified
ENGAGED in the end-to-end runner tests (log: "graphstate 15->11 leaves").
lint-imports + ruff green. TPU A/B: probe launched (same 64/512 cc32 w8a16
harness; baseline 7951 tok/s, 3.51ms step, enqueue 2.30ms).

TPU A/B VERDICT (same probe, 64/512 cc32 w8a16 30B tp4): graphstate
675 -> 108 leaves (919MB packed copies, in-budget); step 3.51 -> 2.44ms
median (p90 2.82), fwd-enqueue 2.34 -> 1.25ms, agg_tps 7951 -> 10829
(+36%; day cumulative 7440 -> 10829 = +46%). Sample output matches the
baseline's recorded degenerate pattern (greedy); CPU runner tests assert
byte-exact parity with packing engaged. Remaining fwd 1.25ms = kv_pages
(~48-96 donated per-layer leaves) + metadata (~40 leaves) + fixed dispatch
floor; the next (invasive) lever would be stacking the cache pool itself.
Acceptance run (stress + 500 reqs 1024x512 cc=32) launched.

ACCEPTANCE RUN (stress + 500 reqs 1024x512 cc=32, w8a16 30B tp4): stress =
120 mixed-shape reqs (prompts 64..1024, decode 32..512, greedy+sampled, 8
concurrent groups) in 27.2s, ZERO failures/truncations. Bench = 500 reqs,
gen 256000 tokens in 154.4s wall -> 1657.5 gen tok/s (vs 1418.7 baseline
= +17% on the mixed prefill+decode workload), total 4972.6 tok/s incl.
prompts, short-completions NONE. Perf windows across the whole run
(n=7953): step median 2.54ms (p90 3.20), fwd 1.25ms, decode agg median
10425 tok/s at 1024-context.

## 2026-08-10 (evening): SFT training profiling campaign

Harness: eLarge dict-config drivers + trainer's native profiler knobs
(profiler_path + profiler_stop_step; trace starts AFTER step 1 so compile is
excluded) + a chrome-trace analyzer (job tmp analyze_trace.py) that reports
device busy/idle, op-category totals (must exclude the jit_training_step
umbrella span — it double-counts), and per-step normalization.

Bring-up (0.5B dense, fsdp, 8x2048 tokens): step 88.4ms device (104ms wall,
dispatch 12ms + sync 86ms), fusion 45.2 + collective 17.1 + attention 15.8
ms/step; single biggest op = all-to-all 13.9ms/step (16% of busy) on a DENSE
model — worth root-causing in the 30B trace.

FINDING (memory): full-AdamW SFT of the 30B MoE does NOT fit on 4x95GB v5p
chips even with mu=bf16 (params 15 + grads 15 + mu 15 + nu 30 GB/chip +
load-time transients -> HBM full at optimizer-state init, measured
RESOURCE_EXHAUSTED with 689MB free / 768MB wanted). Practical floor: 8+
chips for full AdamW, or factored/8-bit state. Profile proceeds with
adafactor (optimizer update is a small epilogue; fwd/bwd/collective/data
profile unchanged).

30B SFT OOM root cause (compile, 110.47G/95.74G, byte-identical across
scan/chunk-CE attempts): HLO temp 55.62GB dominated by ~20+ CONCURRENT
768MB bf16[128,2048,1536] expert gate_up all-gather.remat buffers — expert
weights are fsdp(ep-bound)-sharded at rest (302MB/layer/dev resident) but
the latency-hiding scheduler prefetches gathers many layers ahead AND remat
re-gathers in backward; arguments 54.79GB. chunk_token_size=2048 (LossConfig)
did NOT change the program (advisor: lmhead_chunksize is the operative knob);
scan_layers=True in base_config.values DID materialize into config_kwargs
(CPU-verified) yet the program was byte-identical — scan engagement on this
path unresolved, watch run 5. Run 5 levers: total_batch_size 4,
lmhead_chunksize 2048, attn_softmax_dtype fp32.

eLarge NUMERICS BUG found while debugging: processing.py:1083
`set_maybe("attn_softmax_dtype", coerce_dtype(loader.get("dtype")))` — every
eLarge config with loader.dtype=bf16 silently runs bf16 softmax accumulation
(violates CLAUDE.md "never drop attn_softmax_dtype to bf16"; same bug class
as the negative-KL bf16 pipeline fix). Fix after profile: stop deriving
softmax dtype from loader dtype (keep model default f32). kvdtype/attn_dtype
derivation is fine.

SFT-profiling BUG #2 (the actual OOM cause, affects ALL fsdp MoE training):
base_module.py:266 enters moe_expert_param_layout_scope with
`moe_fsdp_shard_expert_weights AND not fsdp_is_ep_bound` — with the
fsdp_is_ep_bound=True DEFAULT the explicit knob is masked, on the assumption
"ep-bound configs already shard experts over fsdp". But the WEIGHT partition
specs resolve to P('ep', ...) (placement log, ep=1 ⇒ replicated): the
ep-bound fold reaches the token/dispatch compute path (planner: 16k
tokens/core ✓) but NOT the expert weight partition rules ⇒ at fsdp=4/ep=1
every device holds ALL 54.8GB of expert weights as train-step arguments.
Proof: runs 5 and 6 (knob False vs True) compiled BYTE-IDENTICAL (109.17G).
Workaround (run 7): sharding.fsdp_is_ep_bound=False +
moe_fsdp_shard_expert_weights=True → scope receives True, experts shard over
fsdp at rest, gather at use. Proper fix: fold fsdp into the EXPERT semantic
axis for weight specs when fsdp_is_ep_bound (or drop the `and not` mask and
make the scope authoritative). scan_layers byte-identity in runs 3→4 is
explained by the same dominance: program memory was pinned by replicated
expert args either way (scan engagement still to be confirmed in run 7's
trace).

SFT PROFILE CAMPAIGN CLOSED (runs 7+8, 14 steps each, traces analyzed):
run7 unbound+gather 1.624s/step (slices 518 collectives 458 ragged_dot 229
fusion 197 attn 75 ms); run8 ep-bound+sharded 1.689s/step (slices 487
ragged_a2a ~475 fusion 291 ragged_dot 163 attn 75, collectives 67) —
LAYOUTS NUMERICALLY IDENTICAL (loss to 3 decimals, same data) and
PERF-NEUTRAL on single-slice v5p-8; MoE permute machinery is the top bucket
in both (~500ms = 2-3x the expert matmul). Host is NOT a bottleneck in
training (45ms dispatch, data 0ms) — opposite of serving. Both source fixes
shipped with tests (4 passed) + elarge builders 28 passed + lint green:
processing.py softmax derivation removed,
base_module.py knob unmasked. scan_layers non-engagement left OPEN.

scan_layers ROOT-CAUSED + FIXED + MEASURED (task closed): spectrax
ModuleList.scan(trace=True) = python unroll, trace=False = lax.scan;
modeling_qwen3_moe hard-coded trace=True so config.scan_layers was a
SILENT NO-OP (41 modeling files hard-code it; 26 use _layer_scan_trace
correctly — llama is the reference pattern). Fix: gate via
_layer_scan_trace (router-logits collection forces python path — growing
tuples can't ride a scan carry), cache views through the carry with
enabled=trace_layers, self.frequencies hoisted out of the body (caching
property first-touched inside the scan trace = tracer leak). CPU: scan
engages (dots 25->13), parity EXACT (3e-8), remat composes; 2 new tests in
tests/modules/test_qwen3_moe_scan_layers.py; spmd test failures confirmed
pre-existing (XLA:CPU ragged-a2a, identical on unmodified file). TPU 30B
(run 9 vs run 8, identical config): loss trajectory EXACT (1.578@14,
max_grad_norm 3.109 bit-matched), compile 176s -> 72s (2.4x faster),
steady step 1.689 -> 2.327s (38% SLOWER — scan blocks inter-layer
pipelining/overlap). VERDICT: scan_layers now an honest tradeoff knob —
ON for compile time/compile memory, OFF (default) for step throughput.
The other 40 hard-coded modeling files left as-is (same fix pattern
applies if wanted per-family).

BATCH SWEEP VERDICT (b4/b16/b32, ep-bound, scan off): 5,043 / 5,549 / 5,532
tokens/s at 1.624 / 5.905 / 11.846 s/step — tokens/s FLAT ⇒ the step is
per-token routing-bound, batch amortization is dead. b16 trace: permute
1240ms (33%) + ragged_a2a 1206ms (32%) = 65% of device busy; expert GEMMs
305ms (8%); collectives 127ms (3% — the optimized-away war). MFU ceiling
~6% until the routing path is rebuilt. Next campaign prompt handed to user:
fused training MoE path (gmm-v3-style in-kernel routing fwd+bwd) →
quantized a2a dispatch → overlap; target ≥1.5x tokens/s at exact loss
parity.

## MoE routing campaign — LEVER 1 LANDED: layout flip = 2.17x

b16 A/B (identical config, only fsdp_is_ep_bound=False +
moe_fsdp_shard_expert_weights=True i.e. weight-gather layout instead of
token-a2a): step 5.905 -> 2.718s, 5,549 -> 12,056 tokens/s (+117%), MFU 6
-> 13%, loss/acc metrics match TO THE DIGIT. Root cause of the asymmetry at
scale: token a2a + local permutes are PER-TOKEN and data-dependent (cannot
be prefetched), weight gathers are FIXED-size and prefetch/overlap behind
compute; at b4 they tie, at b16 a2a loses 2.17x. Trace (b16ub, device busy
2123ms): permute slices 571 (was 1240), weight AG/RS 491, ragged_dot 504,
attention 159, fusion 287, a2a ~0. CAMPAIGN TARGET (>=1.5x, 8.3k) ALREADY
EXCEEDED; b32 amortization run in flight. Residual levers if pushing
further: permute passes (571), gmm tiling (504, serving-v3 experience),
weight-gather amortizes with batch.

CAMPAIGN CLOSED — GOAL EXCEEDED (target 1.5x/8.3k tok/s): weight-gather
layout sweep b4/b16/b32 = 5,043 / 12,056 / 15,762 tokens/s vs a2a layout
flat 5.5k. Final: 2.85x at b32, MFU 5.4% -> 16.9%, loss parity to the digit
(1.453@14 both layouts). Recipe (config-only, REQUIRES today's
base_module.py knob-unmask fix): sharding.fsdp_is_ep_bound=false +
base_config.values.moe_fsdp_shard_expert_weights=true, then scale
total_batch_size into the freed memory (weight-gather comm is fixed-cost —
bigger batches amortize it; a2a layout is per-token and flat). Remaining
headroom (not chased, goal met): permute passes 571ms, ragged_dot tiling
504ms (gmm-v3-style explicit tiles), batch >32.

PUSH-FOR-MORE CLOSED — config-only ceiling reached (weight-gather layout,
scan off, adafactor, seq 2048): b32/b64/b128/b256 = 15,762 / 18,829 /
21,092 / 22,093 tokens/s (MFU 16.9/20.3/22.7/23.8%); gains +19/+12/+4.7%
per doubling ⇒ knee at b128-256. CUMULATIVE: 5,532 -> 22,093 tok/s = 4.0x,
MFU 5.4 -> 23.8%, loss metrics parity throughout. Sparse logging measured
ZERO effect (async pipeline hides metric sync; steps 11-14 unlogged =
4.25s/step at b32 = same). The ~820ms/step on-device idle at b32 (all 8
lanes balanced 3269ms busy vs 4089ms sync) = gather-stall bubbles — shrink
with batch, or with kernel work. Remaining (kernel projects, not config):
permute row-moves ~25% of busy, ragged_dot explicit-tile gmm ~27%.
Practical guidance: batch >=64 captures >=85% of ceiling; batch choice
above that is a convergence decision, not throughput.

KERNEL-HEADROOM ITEM CLOSED BY MEASUREMENT (both sub-levers refuted at 30B
b64 training shapes, 262k rows x 2048, 128 experts, bf16):
- Remat-policy sweep: nothing_saveable 6.961 / mlp_notsaveable 6.867
  (+1.4%) / checkpoint_dots 6.909 s/step at b64 — policy is NOT a lever
  (MoE replay tensors are memory-unsavable at any practical batch).
- Explicit-tile gmm: grouped_matmulv3 BEST tiling (512,2048,1536)=14.40ms,
  auto-lut 14.16ms vs lax.ragged_dot 12.76ms fwd (v3 LOSES; (128^3) tiles
  lose 16x; v3 grad VMEM-OOMs at big tiles). The serving 10x was
  quant-decode-shape-specific; at training m, XLA ragged_dot ~194 TF/s
  (42% MXU) is the bar and v3 does not clear it.
- Gather fusion ceiling probe: permute overhead 9.67ms/layer-fwd (43% of
  the chain, 2.7x above the 3.58ms bandwidth roofline) — real waste, but a
  Pallas gathered-gmm must ALSO match ragged_dot matmul efficiency, which
  mature v3 cannot; new-kernel path judged not credible. XLA gather is the
  floor for this access pattern.
30B SFT on v5p-8 stands at 22,093 tok/s / 23.8% MFU (4.0x day total) as
the honest ceiling of this box without model/precision changes.

BATCH "do these aswell" — ALL CLOSED:
1. scan_layers 40-file sweep: 37/40 modeling files ported to the
   _layer_scan_trace gating (llama pattern; MoE families route
   router-logits through extra=, hybrids/encoders route cache/head-mask/
   layerdrop/cross-attn through extra=). Holdouts documented in-file:
   deepseek_v4 (heterogeneous per-layer compressors, by design), gemma4
   text decoder (KV-sharing python dicts + per-layer static indexing),
   gidd (pre-existing total breakage — attention arg bug — reverted).
   Verification: per-family parity probes all pass (1e-8..2e-6), spmd
   batches ZERO new failures (pre-existing ragged-a2a CPU + config-
   validation failures stash-verified identical), new regression test
   tests/modules/test_scan_layers_gating.py 6/6 + qwen3_moe 2/2 (re-run
   in main session: 8/8). Bonus fix: xerxes dead output_hidden_states
   crash removed. Follow-up noted: layer_idx-as-spx-static splits scan
   segments per-layer in some families (cohere2/qwen2_moe/gemma2/glm4) —
   _spx_scan_safe_static_fields would consolidate.
2. ejkernel benchmark-gate debt: 16 specs + 19 scripts added, 3 strays
   deleted; gate 3/3 green (re-verified), all scripts execute on CPU.
3. Kernel headroom: closed by refutation (v3 loses to ragged_dot at
   training shapes; gather at XLA floor) — see entry above.
4. Ring fused MLP (#4): closed superseded by the ring-kernel graveyard
   verdict.

PRE-EXISTING BUG FIXES (surfaced by the scan sweep, all CPU-verified):
- gidd (diffusion LM) was TOTALLY broken — four bugs fixed: raw bool array
  passed as the performer's mask_info (now mask_info=None + init_bias with
  full-head broadcast — ops only materialize init_bias when mask_info is
  None; noise semantics verified live via hidden-state delta), inverted
  input_ids/inputs_embeds XOR guard (rejected every valid call), nn.relu on
  spectrax namespace (→ jax.nn.relu), missing compute_lm_logits on the task
  head. NOTE: head_init_scale defaults 0.0 (zero-init head BY DESIGN) — the
  logits-based "noise has no effect" was a probe artifact, documented in
  tests/modules/test_gidd_forward.py (3/3).
- Mamba2Config lacked the layer_types property newer transformers'
  Mamba2Mixer indexes on HF-parity paths — added mirroring HF
  (["linear_attention"]*n). spmd 3/3 (was fixture-dead).
- mamba + falcon_mamba advertised layer_types=["mamba"]*n which upstream
  transformers' layer-type validator REJECTS — renamed to
  "linear_attention" (routes identically: infra/utils.py:1949 matches by
  substring, "linear" and "mamba" hit the same branch). spmd 8/8 (was
  fixture-dead). FINAL TRAINING MFU for the day's campaign: 23.8% (b256,
  22,093 tok/s; practical b64 = 20.3%).

## PP TRAINING CAMPAIGN (user-requested pp-vs-fsdp test, 30B b32, v5p-8)

MPMD memory blowup ROOT-CAUSED + FIXED by debugger agent (spectrax
runtime/mpmd, fully MPMD-preserving; paper trail
.claude/projects/mpmd-training-memory.md): (1) f32 promotion in terminal
grad scaling doubled the grad tree (probe math reproduced the 93.6GB fail
point exactly); fix = dtype-preserving _scale_grad + donated in-place
scale. (2) Unbounded host run-ahead pre-allocated grad-unit outputs at
enqueue speed; fix = per-rank grad-unit throttle (BWD_I/fwd stay async —
cross-rank overlap intact). (3) Pre-existing E0200 CoreHalt from persistent
-cache DESERIALIZED stage executables (jax 0.10/libtpu defect) masked by
the OOM; fix = stage executables bypass the persistent cache
(_stage_persistent_cache_bypass; ~46s fresh warm-compile per process).
spectrax CPU suite 1793 passed; peak HBM 102.4GB(OOM) -> 78.4GB.

SCHEDULE SWEEP RESULT (b32 = 65,536 tokens/step, all 14/14 steps):
  fsdp4 weight-gather SPMD   4.158s  15,762 tok/s   <- CHAMPION stands
  pp4 DualPipeV m=16 virt=2  6.876s   9,531 tok/s   (preflight idle 0.055)
  pp4 ZeroBubbleH1 m=8       8.161s   8,031 tok/s   (idle 0.111)
  pp4 Std1F1B m=8            9.03s    7,258 tok/s   (idle 0.489)
  fsdp4 token-a2a SPMD      11.846s   5,532 tok/s
Schedule quality ordering matches theory exactly, but ALL PP variants lose
to SPMD fsdp on a single slice: bubble + schedule-unit dispatch granularity
(24-50 units/rank) + microbatch GEMM starvation (mb b2-b4 = the measured
5%-MFU regime per-expert) stack against PP; PP's comm advantage (~3GB
boundary transfers vs ~40GB weight movement) doesn't matter because fsdp's
comm is already overlapped intra-slice. VERDICT: on single-slice v5p, fsdp
weight-gather remains the training config; PP is the pod/DCN play (boundary
transfers cross DCN cheaply; weight gathers do not) — unvalidated here, no
pod. CAVEAT flagged: PP loss metric drifts from SPMD (step10: 1.789 DPV /
1.742 ZB vs 1.664 SPMD) — microbatch mean-of-means vs token-global mean
weighting; a valid estimator but not identical math; token-weighted
accumulation would align it (follow-up). DPV also reports
z_loss/accuracy=None (fused terminal drops aux metrics).

MPMD BUG SWEEP CLOSED (jax-expert agent; verified in main session 8+4 new
tests pass, lint clean; paper trail mpmd-training-memory.md):
1. PP loss/grad weighting FIXED — scheduled runtime used uniform 1/M
   mean-of-means; now sxvalue_and_grad(microbatch_weight_fn=...) scales
   loss+cotangents+terminal const-grads by w_mb/Σw (token-weighted, host
   scalars only, zero extra comm, donation preserved). PP now matches SPMD:
   step-1 EXACT 1.664, step-10 1.661/1.664, step-14 1.458/1.453 — inside
   the ±0.023 SPMD-config-noise band. Backward-compatible (no weight fn =
   old behavior); easydel base adapter supplies ForCausalLMLoss-matched
   weights, gated off for reduction=sum/none/divide_weight_sum/mtp_only.
2. Aux metrics FIXED for ALL schedules — terminal returns (loss, aux,
   grads); z_loss/accuracy now populate under 1F1B/ZB/DPV, matching SPMD.
3. Per-step schedule replan FIXED — units+preflight cached per plan;
   preflight logs once; DPV got 9-18% FASTER (6.88 -> 5.65-6.3s/step; its
   m=16 x 8-virtual DAG rebuild was the biggest per-step host cost).
   Updated PP standings vs fsdp4 4.158s: DPV ~6.3s (1.5x gap, was 1.65x).
4. Load watermark QUANTIFIED (fix scoped, not done blind): 72GB peak =
   checkpoint STREAMING holding ~45GB unplaced leaves/device + bulk
   fusion+placement spike; init_tx exonerated (+0.1GB). Follow-up: stream
   to stage-local shardings at read + per-group eager frees (touches HF
   conversion for all models — invasive). Audit: zero TODO/FIXME in
   runtime/mpmd; spectrax 1801 passed; easydel trainers 705; all 3 TPU
   drivers 14/14 clean.

FULL PARALLELISM MATRIX CLOSED (30B qwen3_moe SFT, b32=65,536 tok/step,
v5p-8, all 14/14 with sane loss under the weighted-loss fix):
  fsdp4xtp1 weight-gather   4.158s  15,762 tok/s  1.00x  <- CHAMPION
  pp4xtp1 DualPipeV(m16,v2) ~6.3s   ~10,400       1.5x
  tp4xfsdp1                 7.706s   8,505        1.85x (loss 1.445 sane)
  pp4 ZB / Std1F1B          8.16/9.03s            ~2x
  pp2xtp2 DPV(m8,v2)        8.555s   7,661        2.06x (loss 1.654)
  fsdp4 token-a2a          11.846s   5,532        2.85x
  pp2xfsdp2 DPV            14.161s   4,628        3.41x (loss 1.661) WORST
Insights: (1) pure beats hybrid on 4 chips — hybrids halve each axis's
benefit and stack both cost classes; (2) tp4's per-layer residual
all-reduces cost ~3.5s/step over fsdp (unhidable, latency-bound — matches
the serving-side collective verdicts); (3) pp x fsdp is an ANTI-PATTERN:
each microbatch re-runs the stage executable and RE-GATHERS fsdp-sharded
weights (~8x weight traffic/step) — PP wants weights RESIDENT per stage
(tp/ep/replicated), never fsdp-sharded within a stage.

SPEEDUP LEVER LANDED — SEQUENCE PACKING (the biggest single win of the
campaign, one flag): ultrafeedback@max_length=2048 measured 78.7% PADDING
(mean row 436 tok). sequence_packing=True at b64: step 7.001s vs unpacked
6.961s (+0.6% cost) with ~100% real tokens => REAL-token throughput
~4,011 -> ~18,722 tok/s = 4.67x. All prior tokens/s numbers in this log
count padded tokens (device-work-normalized, still valid for config A/Bs);
REAL-data throughput additionally requires packing ON for short-row SFT
sets. Remaining levers: xla scheduler flags A/B (est +5-10%), int8 expert
training w8a16 via ejkernel quant gmm + STE (est +15-25%, medium risk),
8+ chips / longer seq.

Scheduler-flags A/B NEUTRAL (lever closed): xla_latency_hiding_scheduler_
rerun=5 + xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=true
(both verified present in libtpu strings) on packed b64: 7.003 vs 7.001s,
loss byte-identical — default LHS schedule already optimal at this shape.
SPEEDUP CAMPAIGN FINAL STATE: packing 4.67x real-token win stands;
remaining lever = int8 expert training (goal prompt drafted); hardware
scaling beyond that.

INT8 EXPERT TRAINING — KILLED BY KILL-SWITCH BENCH (do not re-attempt
without new facts): v3 w8a16 fwd at training shapes (262k rows) beats bf16
ragged_dot 5.62 vs 7.34ms (1.31x, quant err 2% rel max — QAT-viable) BUT
grad-x through the quant vjp = 183.68ms (26x worse; transposed bwd path
mishandles codes+scale). Even with the vjp fixed, ceiling = int8 fwd+dx at
1.31x with d/dW necessarily bf16 => ~15% of the gemm bucket ~= 4-5% of
step + a few % from int8 gathers — not worth STE integration + convergence
risk on this box. int8 fwd already pays where it belongs (serving).

SEQ-SCALING SWEEP (packed, matched 131,072 tokens/step, 14/14 each):
  seq 2048 (b64): 7.001s = 18,722 tok/s
  seq 4096 (b32): 7.801s = 16,802 tok/s (-11%; flops/token 21.9->26.8G)
  seq 8192 (b16): 9.123s = 14,367 tok/s (-23%)
Pure quadratic attention tax; per-expert GEMMs were already saturated at
this tokens/step, so longer seq adds cost without efficiency gain. VERDICT:
seq length is a DATA decision, not a throughput lever, on this model/box.
8+ chips untestable here (4-chip v5p-8) — flagged, not claimed.

MAXTEXT VS EASYDEL HEAD-TO-HEAD (same Qwen3-30B-A3B arch — param-count
gate 30.532B ✓, same v5p-8, same 131,072 tok/step, fsdp=4, bf16, full
remat, synthetic all-real tokens; MaxText@9cb34e819 jax 0.11 own venv):
  EasyDeL vnext:            7.001 s/step  18,722 tok/s
  MaxText out-of-box:       7.700 s/step  17,022 tok/s  (EasyDeL +10.0%)
  MaxText megablox=false:   7.026 s/step  18,656 tok/s  (tie, +0.4%)
The out-of-box gap is MaxText's megablox Pallas gmm default — flipping to
jax ragged_dot ties, INDEPENDENTLY CONFIRMING our kernel verdict (ragged_
dot > Pallas gmm at these MoE shapes on v5p). Notable capability edge:
MaxText scan_layers=false OOMs (102.9G HLO temps) on this workload;
EasyDeL runs it unrolled at a ~43G-class footprint. Caveats: adamw (no
adafactor in MaxText), jax 0.11 vs 0.10, scan-forced vs unrolled. Evidence:
/home/erfan/maxtext/bench-notes.md.

FULL REVIEW + FIX CYCLE CLOSED (branch corrected to vnext first — ff'd
onto the stray autoresearch-qmm-2x head, branch deleted, rule memorized):
reviewer found 1 HIGH + 8 MEDIUM + ~13 LOW across the ~273-file tree, ZERO
wrong-gradient/data-corruption in the core campaigns, true-MPMD invariant
intact. ALL findings fixed by two agents with tests:
- H1 tool-parser gate (llama/xlam/openai/granite markerless JSON restored;
  process_final ungated; 346 parsing tests green)
- M1 backbone decode_only real bucket; M2 scheduled weight CLM-only gate;
  M3 seam decline-protocol made real (declared-layout resolution, auto
  default KEPT per user); M4 MoE batch-drop inference-gated (agent caught
  that decode-scope-only gating broke eSurge prefill — corrected); M5
  fused-MLP name-based-remat fallback + quant-fused inference-only; M6
  FORCE_NATIVE_RUNTIME honored in gated_mlp_forward; M7 warm-compile
  global-config toggle hoisted; M8 env var -> setter API; all LOWs incl.
  file:// checkpointer, StatePacker treedef assert, aux 2-tuple contract.
Combined verification: spectrax 1807 passed, eSurge 829+77, parsing 346,
MoE 62, linears 51, lint-imports 2/2, ruff clean. Pre-existing debt noted
(4 ejkernel test failures from the earlier committed work: RPA-v3
slice_sizes, autotune-surface, del-shim, PersistentCache signature).

/code-review (xhigh workflow) — verifiers rate-limited mid-run, so the 26
pooled candidates were verified INLINE in the main session. 14 findings
survived (10 fixed on the spot, 3 skipped-as-backlog, 1 no-change):
FIXED: fused-MLP "gelu" was tanh-approx vs ACT2FN exact (all 3 activation
tables); quant expert kernel_view() seam swept to all 21 remaining MoE
families (43 sites — expert quantization no longer crashes them);
incremental-decode prime gate now enforces prefix-contiguous active rows
(freed-slot stale-token hazard) + full 9-input VLM gate parity; spectrax
microbatch weight fn moved off the shared plan dict onto a ContextVar
(concurrency race); kimi SITU falsy-zero betas; delegating-parser previous
_token_ids reconstructed at engagement; CLM-loss identity-first check;
ruff F841/E501/RUF100/I001/RUF059 batch (tree-wide ruff now clean);
duplicate local import. BACKLOG (skipped): weight-fn m×device_get host
stalls on weighted PP, 38-file scan-gating derivation dedup into
_layer_scan_trace, adapter-bypass + duplication cluster. All fixes
test-verified (46+42+11+5+33 green, lint-imports 2/2).

SECOND REVIEW (/code-review xhigh, full 14-angle pipeline, 62 candidates ->
39 deduped -> probe-verified) — 15 findings, ALL FIXED:
Three were regressions from THIS session's own fix cycle: (a) the ContextVar
migration orphaned sxvalue_and_grad_and_apply's weight fn (fused-apply
silently reverted to uniform 1/M — probe 1.0339 vs 1.1568); (b) the
kernel_view sweep missed 5 sites (gemma4 x2, mixtral/arctic/minimax w2,
qwen3_next dense) that still crash under quantization; (c) the new
tool-machinery gate was non-monotonic — text a parser withheld during a
false marker-prefix engagement was dropped from the client stream forever
(probe: '<tool' lost). Fixed via a LATCH (once engaged, never bypass).
Also fixed: async_manager DEFAULT deserialize path destroyed integer
quantized leaves (uint32 70000 -> bf16 70144) — the floating guard existed
only on the custom branch; compressed-tensors int4 decode was
orientation+signedness wrong (dedicated decode_compressed_int4 now);
expand_scale ignored checkpoint block_shape (DSv3 fp8: 120/576 rows took
the neighbor's scale); modules_to_not_convert prefix-matched instead of
HF substring+glob; MoE quant seam TypeError on bits=None and int4 forward
crash; gmm v3 pre-dequant branch changed numerics for SUPPORTED pairs at
group<128 (easydel default is 64!) — now restricted to unsupported dtypes;
fused_mlp W8A8 bwd linearized at wrong pre-activations (knobs now
forwarded); quantized experts dropped (EP,FSDP) at-rest sharding; dead
packed-int4 companions on attention projections (~1GB/7B) now gated; PP
backbone plan-cache first-writer-wins (decode twin added); forkify escape
hatch unreachable from Ray workers (flag captured at wrap time).
Verified: 5/5 finding-reproduction probes, quantization+linears 122,
serialization+fused_mlp 152, eSurge runners 77, spectrax pipeline 11,
eformer 4; ruff clean tree-wide; lint-imports 2/2.
