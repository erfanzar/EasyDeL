# Pre-quantized checkpoint support — design

Goal: load externally quantized checkpoints (fp8, compressed-tensors, AWQ/GPTQ,
MXFP4, NVFP4) for serving **and** QAT/finetune, across all model families,
without per-format layer classes, per-model code, or forward-path branching.

Status: design. No code yet.

## The problem, stated precisely

EasyDeL today only *self-quantizes*: `ParallelLinear.to_quantized(config)` takes
an fp weight and calls `prepack_quantized_weights`. Nothing parses an HF
`quantization_config` — `rg 'quant_method|config_groups|weight_block_size|activation_scheme|weight_scale'`
over `libs/easydel/easydel/` returns nothing. So every pre-quantized checkpoint
on HF is currently unloadable.

What we already own (do not rebuild):

| capability | location |
| --- | --- |
| formats + packing (affine/nf4/int8/mxfp4/nvfp4/mxfp8/nvfp8/turboquant) | `ejkernel/quantization/` (~4.4k lines) |
| fused dequant+matmul, weight-only | `ejkernel/kernels/_pallas/tpu/quantized_matmul/`, `_tilelang/` |
| **grouped matmul with in-kernel dynamic activation quant** | `ejkernel/kernels/_pallas/tpu/grouped_matmulv3/_pallas_impl.py:1159-1171` (`maybe_quantize_lhs`, fp8/int8 by hardware, `rhs_scale [G, blocks, 1, N]`) |
| one quantized linear, mode-parameterized | `easydel/layers/linears/_linear_quantized.py:617` |
| STE / QAT per format | `easydel/layers/quantization/_straight_through.py` |
| KV-cache quantization (incl. turboquant paged attention) | `easydel/caching/`, `layers/attention/_flexible.py:1036` |
| fused-projection TP-portable layout | `fused_param_tp`, `layers/layouts/` |
| N-tensor → 1-param conversion with inverse | `utils/parameters_transformation.py:341-375` |

The kernels are done. **This is a loader project.**

## The seam

One contract. Everything above it is format-specific; everything below it is
already written and shared.

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

The last three fields exist for reasons established below; they are what keep
MoE, NVFP4 and W8A8 from each spawning a parallel code path.

**`expert_dim` is the one that must be right in phase 0.** Dense is `(K, N)`
with rank-2 scales; MoE is `(E, K, N)` with `(E, blocks, 1, N)` scales. ejkernel's
dense `validate_packed_quantized_matmul_layout` hardcodes rank-2 — if the
canonical contract inherits that assumption, every MoE format becomes a
duplicate of its dense twin later.

**`output_scale`** carries a per-tensor scalar. Needed because NVFP4 checkpoints
have two scale levels (e4m3 block + fp32 global `weight_scale_2`) and folding the
global into an e4m3 code is lossy. It does not need folding — a per-tensor scalar
commutes with the matmul:
`x @ (w_q · s_block · s_global) == s_global · (x @ (w_q · s_block))`.
Exact, one multiply, no kernel change. Also serves compressed-tensors' fused
per-tensor scales.

**`activation`** is a runtime policy, not a weight property:
`none | dynamic(int8|fp8, per-token) | static(scale, per-tensor)`, resolved once
from the checkpoint scheme (W8A16 / W8A8-dynamic / W8A8-static) and consumed at
kernel invocation.

This is precisely what `_linear_quantized.py:777-798` already builds and what
`_dequantize_array` / `_distributed_quantized_matmul` / `from_quantized` /
`restage` already consume. Landing every external format on this triple means
zero new forward paths, zero new layer classes, zero model edits.

**Rule: a new checkpoint format may add exactly one adapter function and one
registry entry. If it needs anything else, the seam is wrong — fix the seam.**

## Layer 1 — format adapters (the only per-format code)

```python
@checkpoint_quant_registry.register("compressed-tensors")
def adapt(tensors: dict[str, Array], meta: SchemeMeta) -> CanonicalQuantizedWeight
```

Pure function, no module state, no sharding, no mesh. ~60-150 lines each:

- **fp8 per-tensor / per-channel** — scale rank normalize → canonical.
- **fp8 blockwise (128x128, `weight_scale_inv`)** — block dequant → requantize
  to per-channel or subchannel (see "requant policy").
- **AWQ int4** — u32→u4 with the `(0,4,1,5,2,6,3,7)` reorder, zero points → affine.
- **GPTQ int4** — same shape, different packing + `g_idx`.
- **MXFP4 (gpt-oss)** — u8→e2m1, u8→e8m0 scales → requantize to wider block.
- **NVFP4 (modelopt / CT)** — e2m1 + fp8 block scale × fp32 `weight_scale_2`.

Duplication control: the *primitives* (bit unpack, e8m0↔fp32, block dequant,
requantize) live once in `ejkernel/quantization/_utils/`; several already exist
(`bitpack.py`, `fp_tables.py`, `qparams.py`). Adapters are compositions, never
implementations. Any primitive written twice is a review failure.

## Layer 2 — ingestion via existing `reform_param` fusion

`reform_param` already accepts `{'sources': [...], 'fuser': fn, 'inverse_fuser': fn}`
(schema validated at `parameters_transformation.py:341-375`, applied at
`apply_reform_param_fusions`). That is *exactly* N-checkpoint-tensors →
1-EasyDeL-param with an export inverse.

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
- fused QKV/gate-up already flows through this machinery, so scales ride the
  same `fused_param_tp` permutation as the weight — one place, tested once.

## Layer 3 — scheme resolution (model-agnostic dispatch)

`CheckpointQuantScheme.from_hf_config(quantization_config) -> scheme`, answering
`scheme.for_path(path) -> ResolvedScheme | None`. Responsibilities:

- `quant_method` → adapter selection;
- `ignored_layers` (exact) vs `modules_to_not_convert` (prefix) — different
  match semantics, one implementation;
- compressed-tensors `config_groups` / `targets` regex matching;
- **fused-shard consistency**: all shards of a fused projection must share a
  scheme, else hard error (silent mixed-precision fusion is a correctness bug);
- per-layer opt-out honored identically for dense and MoE.

Plugs into `EasyQuantizer` (which already walks paths with a regex at
`_quants.py:558-583`) rather than living beside it — same traversal, richer
predicate.

**Why 80+ families cost zero:** models build `ParallelLinear`/`BaseMoeModule`;
conversion already routes every HF tensor through `parameters_transformation`;
the resolver works on paths. No model file is touched.

## Layer 4 — activation quantization

Current state is worse than it looks: `_quantize_runtime`
(`_linear_quantized.py:1011`) is **never called** — dead code — and
`QuantizationConfig.runtime_dtype` is unwired for linears (every `runtime_dtype`
hit in the repo is `OperationMetadata.runtime_dtype`, the attention compute
dtype). ejkernel's dense op takes float `x`
(`output = x @ dequantize(w, scales, zeros)`). **EasyDeL has no activation
quantization for linears today.**

The only in-kernel activation quant in the stack is `grouped_matmulv3`'s
`maybe_quantize_lhs`. So:

- **MoE gets W8A8 first, for free** — the kernel already does it.
- **Dense needs one of:** (a) route through `grouped_matmulv3` with
  `group_sizes=[m]` — zero kernel work, immediate W8A8, and exactly what
  vLLM-TPU does (their standalone quantized-matmul Pallas kernel is dead code;
  production goes through gmm_v2); or (b) add `maybe_quantize_lhs` to the dense
  Pallas kernel properly. Do (a) as the correctness baseline, (b) only on a
  measured win.

Static (calibrated per-tensor) activation scales are cheap and exact via
`output_scale`. Note vLLM-TPU implements this **incorrectly** in their fused
path — `compressed_tensors_w8a8_fp8.py:248` applies `weight_scale` but drops
`input_scale`, while their split path at line 293 applies both. Our parity test
must be the kind that catches that.

**Requant policy — do not copy their default.** vLLM-TPU rewrites narrow
checkpoint blocks to a wider block at load because a quant block narrower than
the MXU column forces dequantize-before-matmul
(`should_dequantize_before_matmul`: `quant_block_size < mxu_column_size`). For
gpt-oss MXFP4 they dequantize to fp32 and requantize to block-512 fp4 — a real
accuracy loss. We only need that if measurement shows group-32 starves the MXU,
and then it is an **opt-in** policy, not a default. Make the threshold a
property queried from the kernel, never a constant inside an adapter.

## Kernel capability matrix (verified, not assumed)

`quantized_matmul` (dense) is registered on six platforms; `grouped_matmulv3`
(MoE) on three.

| kernel | platform | weight quant | sub-byte packed W | activation quant | notes |
| --- | --- | --- | --- | --- | --- |
| grouped_matmulv3 | Pallas TPU | yes | yes (`should_bitcast`, `:396`) | **yes** (`maybe_quantize_lhs`, `:1160-1171`) | `rhs_scale (G, blocks, 1, N)` validated `:1017` |
| grouped_matmulv3 | TileLang GPU | yes | — | no | has VJPs for lhs/rhs/rhs_scale/rhs_bias (`_interface.py:79`) |
| grouped_matmulv3 | XLA | yes | — | no | dequantize-then-matmul fallback when `rhs_scale` given (`:27`); parity reference |
| quantized_matmul | Pallas TPU | yes (+bwd) | yes | **no** | `x: Float[Array, "m k"]`, `Y = X @ dequant(W)` |
| quantized_matmul | XLA / Triton / CuTe / CUDA / TileLang | yes | yes | **no** | same signature on all |

**But v3 is not the production MoE path, so none of that is reachable today.**
`_moe_module.py:1524-1540` builds `gmm_kws` with `platform="xla"`
(`bypass_xla_tiling=True`, i.e. XLA `ragged_dot`) and a conditional Pallas
branch; `use_v3` is never set anywhere. And the plain `grouped_matmul`
entry point **rejects** quantized weights outright:

```
grouped_matmul.py:324-325
    if self.op_id != "grouped_matmulv3" and (rhs_scale is not None or rhs_bias is not None):
        raise ValueError("rhs_scale and rhs_bias are only supported by grouped_matmulv3.")
```

So MoE expert quantization is **not** free plumbing, as an earlier draft of
this document claimed. It needs one of: a quantized `ragged_dot` path,
dequantize-experts-before-`ragged_dot` (memory win only, no compute win), or
moving the production MoE path onto v3 — a separate decision with its own
performance evidence. Treat MoE as its own investigation, not a phase-1
freebie.

### Adding activation quantization to dense

Two designs:

- **(A)** extend the dense Pallas kernel with an lhs-quant path — roughly the
  40 lines at `grouped_matmulv3/_pallas_impl.py:508-545`, but it changes
  accumulator dtype and per-block rescale in the core loop shared with the
  working weight-only path.
- **(B)** route dense through `grouped_matmulv3` with `group_sizes=[m]` — zero
  kernel work, already-exercised code. What vLLM-TPU actually does.

Do (B) first, (A) only on a measured win.

**The dispatch lives inside ejkernel's dense op, never in EasyDeL.** EasyDeL
passes an `ActivationPolicy`; ejkernel chooses Pallas-weight-only /
v3-one-group / XLA. Anything else leaks kernel internals into the framework and
breaks the registry contract (golden rule 6).

Signature change is additive across the six impls:
`activation_quant: "none" | "dynamic"`, `activation_scale: Array | None`.

- **XLA gets a real reference impl** (~20 lines: quantize `x`, `dot_general`
  with `preferred_element_type=int32/float32`, rescale). Mandatory fallback
  *and* the parity baseline for every activation-quant test.
- **Pallas TPU** dispatches to v3-one-group when `activation != none`.
- **Triton / CuTe / CUDA / TileLang** raise `NotImplementedError` → registry
  falls back to XLA on GPU (correct, slow) until TileLang gets a real lhs-quant
  path mirroring the Pallas one — which also fixes GPU MoE activation quant.

## MX / NV FP4 / FP8 — already on-disk compatible

Verified, not assumed:

- `qparams.py:479-481` **enforces uint8 scales** for all of
  `{mxfp4, mxfp8, nvfp4, nvfp8}`.
- MX decode is `scale = jnp.exp2(exp.astype(float32))`
  (`quantizations.py:613`) → e8m0 semantics, spec-correct.
- NV decode is an `e4m3_table` lookup on uint8 codes
  (`quantizations.py:682, 1028-1040`) → e4m3 block scales, spec-correct.
- Group sizes already pinned to spec: mxfp4/mxfp8=32, nvfp4/nvfp8=16.

Consequences:

- **MXFP4 (gpt-oss) is a repack, not a requantize** — their e2m1-pairs-in-u8 +
  e8m0-in-u8 at group 32 map onto our packed codes + uint8 scales at group 32.
  Only the packing container differs (2×e2m1 per u8 vs our uint32 codes).
  **Zero numeric loss**, strictly better than vLLM-TPU's lossy block-512 rewrite.
- **NVFP4** matches on the block scale; only the fp32 global needs the
  `output_scale` slot.
- **MXFP8 / NVFP8** have no HF checkpoints today, but we support them natively,
  so they are available as *requantization targets* — e.g. blockwise-fp8
  DeepSeek → mxfp8 (group 32, e8m0), an option vLLM-TPU cannot express.

## QAT / finetune

Three supported modes, all sharing the canonical triple:

1. **Frozen quantized + LoRA** — quantized params as buffers, adapters train.
   Works with `_lora.py` today. Expected 90% case.
2. **STE-QAT** — `_straight_through.py` already implements per-format STE; it
   needs a dequant→STE→requant VJP over the canonical triple. Because every
   format lands on one representation, **STE is implemented once, not per format.**
3. **Dequantize and full finetune** — `from_quantized()` already exists.

Optimizer-state interaction (mode 2) is the open question: quantized params must
not receive dense optimizer state. Decide before implementing mode 2.

## Non-negotiables

- No per-format layer class. vLLM-TPU's failure mode is 12 config classes ×
  12 method classes × 2 stacks ≈ 7.7k lines of near-duplicate wrapper.
- No format branching inside any model file or any forward path.
- Every format ships an XLA reference parity test against the reference
  dequantization. Their CI records `"unverified"` for all of it; that is the
  gap we beat them on, not feature count.
- Scales travel through `fused_param_tp` with the weight, verified at tp>1.
- Registry, not if-chains (golden rule 6).

## Phasing

| phase | content | status |
| --- | --- | --- |
| 0 | canonical contract, scheme resolver, adapter registry, reform-rule generation | **done** |
| 1 | fp8: per-tensor, per-channel, blockwise `weight_scale_inv` | **done** |
| 2 | AWQ + GPTQ (incl. act-order) int4 | **done** |
| 3 | MXFP4 + NVFP4 with `output_scale` | **done** |
| 4 | compressed-tensors (int8 / fp8 / int4 / NVFP4, per-config-group) | **done** |
| 5 | **grouping-axis fix** — the dominant accuracy item, see below | next |
| 6 | activation-quantization runtime (ejkernel signature + XLA reference + Pallas dispatch) | |
| 7 | MoE expert quantization — needs a kernel decision first, not plumbing | |
| 8 | STE-QAT over the canonical state + export inverses | |

## Weight formats — shipped

`_codecs.py` holds one decoder per on-disk encoding; `_formats.py` holds the
adapters. Registered `quant_method` aliases: `fp8`, `awq`, `auto_awq`, `gptq`,
`mxfp4`, `gpt_oss_mxfp4`, `modelopt_fp4`, `nvfp4`, `compressed-tensors`,
`compressed_tensors`.

Every adapter is decode-only: it ends in
`CanonicalQuantizedWeight.from_dense`, which routes through the same
`prepack_quantized_weights` the self-quantization path uses, so a
checkpoint-loaded weight is layout-identical to a natively quantized one. No
adapter reimplements packing, bit manipulation or scale math.

Measured round-trip error against independent NumPy references
(`test_checkpoint_quant_formats.py`, 28 tests):

| format | target | rel. err |
| --- | --- | --- |
| fp8 per-channel | affine int8 | 0.003 |
| fp8 blockwise 128×128 | affine int8 | 0.003 |
| compressed-tensors int8 | affine int8 | 0.004 |
| AWQ int4 | affine int4 | 0.061 |
| MXFP4 | mxfp4 g32 | 0.042 |
| NVFP4 | nvfp4 g16 | 0.154 |

### Two decisions that measurement reversed

**fp8 targets affine int8, not mxfp8.** The intuitive argument — keep 8-bit
floats as 8-bit floats rather than push them through an integer grid — is
wrong for *grouped* quantization. Within a group the dynamic range is small,
so int8's 256 levels with an exact float scale beat E4M3's 3-bit mantissa with
a power-of-two E8M0 scale: 0.003 versus 0.049 on gaussian weights. Exposed as
`ScaledElementsAdapter.float8_target` for re-measurement on real weights.

**MXFP4 is not the lossless repack claimed earlier.** It would be — measured
at exactly 0.0 — if the repack grouped along input features. It does not.

### The grouping-axis gap (phase 5, and the largest remaining item)

Every checkpoint format groups scales along **input** features.
`ParallelLinearQuantized` packs `[in, out]` with `transpose=False`, which
groups along **output** features. A checkpoint's calibration therefore cannot
survive repacking — the groups do not correspond.

Measured on data lying exactly on an MXFP4 grid (`[256, 128]`):

| grouping | rel. err |
| --- | --- |
| input axis (`axis="row"`) | **0.0** |
| output axis (`axis="col"`, current) | 0.042 |

AWQ-like 4-bit data: 0.031 (input) versus 0.061 (output). 8-bit absorbs the
mismatch; 4-bit does not — the entire 4-bit error budget in the table above is
this one issue. Fixing it means changing the quantized linear's layout
(`_quantized_linear_layout_spec`, `_resolve_shard_specs`,
`_distributed_quantized_matmul` and the sharding that follows), which is a
real change to a working kernel path and is why it is its own phase rather
than something slipped into an adapter. `TestGroupingAxisLimitation` pins the
current cost and is written to fail once it is fixed.

## Phase 0 — what shipped

`libs/easydel/easydel/layers/quantization/checkpoint/`:

* `_canonical.py` — `QuantSpec`, `CanonicalQuantizedWeight`, `SourceFormat`,
  `ActivationPolicy`. `QuantSpec` validates itself through the existing
  `resolve_ejkernel_quant_params`, so the per-mode `(group_size, bits)` table
  is not duplicated, and projects onto `QuantizationConfig` — the type the
  quantized layers already accept.
* `_adapter.py` — `CheckpointQuantAdapter` ABC + `register_adapter`, on top of
  the existing `Registry` under category `"checkpoint-quant"`. No new registry.
* `_scheme.py` — `CheckpointQuantScheme.from_hf_config` / `for_path` /
  `for_fused`, including the exact-vs-prefix ignore distinction and the
  fused-shard agreement check.
* `_reform.py` — `checkpoint_quant_reform_param`, projecting a resolved scheme
  onto `reform_param` rules.

One converter change, in `utils/parameters_transformation.py`:
`TensorConverter.to_jax_preserving_dtype` plus a `preserve_dtype` flag honored
by `process_tensor`. Without it, `convert_pytorch_to_jnp` casts every leaf to
the model's param dtype and a packed `uint32` kernel or `uint8` scale is
destroyed silently.

Two non-obvious things the implementation forced, both now encoded in the code
and its tests:

1. **The rule key must be the target parameter (`quant_kernel`), not a source
   suffix.** `apply_reform_param_fusions` skips a group when the fused key is
   already present in the state dict, so a self-hosted rule keyed on `weight` —
   which is also one of its own sources — would silently never fire.
2. **`SourceFormat.raw` is excluded from hashing.** A frozen dataclass holding
   a mapping raises on `hash()`, which breaks any use as a dict key or JIT
   static argument.

Tests: `libs/easydel/tests/layers/quantization/test_checkpoint_quant_seam.py`
(25, passing) drive the real `StateDictConverter.huggingface_to_easydel` rather
than a stand-in, asserting that packed dtypes survive while ordinary dense
weights are still cast and transposed.

## TPU v5 measurements (2026-08-03, single chip, jitted, autotune off)

Op-level: `jax.jit(x @ w)` vs `jax.jit(quantized_matmul(...))`, warmup 5,
median of 20. bf16 baselines: qkv(4096x6144) 147us @ t=8/32 (overhead floor —
identical at both batch sizes), 359us @ t=2048; gate_up(4096x28672) 216us /
1210us; down(14336x4096) 171us / 687us.

**XLA quantized path: uniformly SLOWER than bf16.**

| shape | mode | t=8 | t=2048 |
| --- | --- | --- | --- |
| qkv | mxfp4 | 0.16x | 0.26x |
| qkv | nf4 | 0.22x | 0.33x |
| gate_up | mxfp4 | **0.05x** | 0.20x |
| gate_up | affine8 | **0.05x** | 0.20x |
| down | mxfp4 | 0.08x | 0.23x |

The XLA path dequantizes in-graph (unpack + scale before the matmul); on TPU
the dequant work dwarfs the bandwidth saved. Never route serving through
`platform="xla"` qmm expecting speed — it is a correctness fallback only.

**Fused Pallas packed path: a SINGLE kernel compile exceeds 10 minutes**
(mxfp4, 4096x6144, t=8; `use_best_config=False`, so not autotune). Confirmed
twice at model level — the whole-model bench died inside
`jax::PyClient::CompileAndLoad` after 20-30 min with quantization already
finished (2.38 GB packed) — and now once at op level. RSS grows ~9 GB during
the compile, pointing at pathological Mosaic IR size (likely a fully unrolled
in-kernel unpack over the K/group dimension). This blocks ANY steady-state
measurement of the fused path and is the top perf item in the subsystem.

Fixed along the way, verified: eager weight packing in
`ParallelLinearQuantized._quantize_array` — `prepack_quantized_weights` ran
unjitted; on (4096,6144): mxfp4 4.80s eager vs 8.2ms jitted (585x), affine
2.34s vs 0.7ms (3490x). Now routed through `prepack_quantized_weights_jit`
(also used by the checkpoint `from_dense` path).

Also real: mxfp4 model compression measured 7.50 GB -> 2.38 GB (3.15x);
un-jitted eager model forward is ~275x slower than `spx.jit` (982ms vs 3.6ms)
and flat across batch size — never benchmark without jit.

## The compile-bomb root cause, and the vLLM-style fix (2026-08-03, later)

Root cause of the >25-minute quantized compiles, confirmed at source level and
by experiment: ejkernel's dense qmm kernel **hand-decodes packed uint32 inside
the Pallas kernel** (`_pallas_impl_core.py:459-486` e2m1 sign/exp/mant bit
math, `:565-604` shift/mask unpack) — Mosaic IR explodes. vLLM never unpacks
in-kernel: weights are native sub-byte dtypes, `pltpu.bitcast` once per tile.
Our `grouped_matmulv3` already had the bitcast design but was missing two
guards its vLLM counterpart (`gmm_v2`) has. Ported them:

* `_pallas_impl.py` `_matmul` unquantized branch — added the
  dequantize-before-matmul path (fold per-block scales into rhs in registers,
  full-K matmul per column strip), triggered when
  `not is_matmul_supported(lhs, rhs)` or the rhs quant block is narrower than
  the MXU column. Upcast happens BEFORE any reshape: Mosaic rejects rank-3
  sub-byte vectors.
* quantized-lhs branch — added the exact `is_matmul_supported` upcast guard.
* **fixed a scale-indexing bug vLLM also has latently**: the k-loop stepped by
  the lhs quant block (512) while indexing rhs scales as
  `start_k // rhs_qbs` — with rhs blocks of 128, one scale covered four
  blocks. Clamped the step to the rhs block; relerr dropped 0.02-0.03 → 0.016
  uniform. (vLLM masks it by never configuring rhs blocks < 512.)
* widened `rhs` annotations (`Float | Int`) across the v3 interfaces and the
  module op — integer weights were rejected by beartype before reaching the
  kernel.

Measured v5 Mosaic storage support (load+upcast one tile):
`float4_e2m1fn` **UNSUPPORTED** (`vector<8x128x8xf4E2M1FN>` Mosaic error);
`int4` OK; `int8` OK; `float8_e4m3fn` OK. This mechanically explains
vLLM's support matrix (MXFP4 = v7 only; int4/int8 = v5/v6) and means:
**on v5, f4 runtime storage is impossible — MXFP4/NVFP4 checkpoints must
target int4/int8 affine on this hardware.** `target_spec` should become
hardware-aware. fp8 storage works (no fp8 MXU, so W8A16 dequantize-before).

Result — v3-one-group int8 W8A8 (K-blocked g128, in-kernel int8 lhs quant),
TPU v5 single chip, vs jitted bf16 `x@w`:

| | compile | decode t=8/32 | prefill t=2048 |
| --- | --- | --- | --- |
| before (dense qmm) | **>25 min, killed** | — | — |
| v3 untuned (128,128,128) | 1.3-7 s | 0.15-0.17x | 0.02-0.03x |
| v3 tiled (128,2048,1024) | 3.8-8.3 s | **0.92-0.96x** | 0.58-0.66x |

relerr 0.016 (double 8-bit quantization). Tiling is decisive — a 6-64x swing;
vLLM ships a 700-line tuned-block-size LUT for exactly this reason.

**W8A8 vs W8A16 A/B** (same int8 weights, same kernel, tiles m128/k2048/n1024,
impl-level `maybe_quantize_lhs` toggle): **W8A16 wins everywhere on v5** —
faster at every shape/token count (qkv t=32: 0.91x vs 0.78x; gate_up t=2048:
0.43x vs 0.26x) AND 5-10x more accurate (relerr 0.002-0.003 vs 0.016). The
per-token lhs quant is VPU work v5's int8 MXU cannot pay back. fp8-W8A16 is
unusable on v5 (0.03-0.20x — f8 upcast has no fast path this generation).
v5 recipe: **int8 K-blocked storage, dequantize-before, bf16 MXU,
maybe_quantize_lhs=False**; revisit W8A8 on fp8-capable generations.
Caveat on the earlier sweep's 0.92-0.96x W8A8 numbers: measured BEFORE the
scale-indexing fix, i.e. with illegally coarse lhs blocking; post-fix honest
W8A8 is 0.76-0.78x at those points. If W8A8 returns, rhs g512 restores the
coarser legal lhs blocking (LUT dimension).

Not yet >1x. Why, and the path there:
- decode t<=32 is at an overhead floor (~150us) for BOTH bf16 and v3 — bf16
  runs 50MB at only ~343GB/s, far off roofline. Halved weight bytes cannot show
  up until the floor is lowered or the weight-per-chip is bigger; at 27B-class
  shards the bandwidth win should surface.
- prefill pays the in-kernel lhs-quant VPU cost without recouping it; fp8
  W8A16 storage (supported!) with dequantize-before may beat int8 W8A8 there.
- a tuned tiling table per (shape-class, tokens) is mandatory, not optional.

Next: wire ejkernel's dense `quantized_matmul` TPU route through v3-one-group
with native-dtype storage (kills the uint32 in-kernel unpack path on TPU),
make checkpoint `target_spec` hardware-aware (v5 -> affine int8/int4), add a
tuned tiling LUT seeded from the sweep above.

## Autoresearch: 2x-speedup loop (2026-08-03, branch `autoresearch-qmm-2x`)

8 iterations, git-as-memory on the branch, log in `autoresearch-results.tsv`.
Metric: min speedup vs jitted bf16 over {int8,int4} x {qkv,gate_up,down} x
{t=8,2048}, relerr gate 0.08. **0.456 -> 1.078** (2.4x metric improvement);
every cell >1x.

The path was one demolition and one construction:
- **Pallas is the wrong tool for this op.** The v3 kernel's *pure bf16*
  ceiling measured 0.47-0.78x of XLA's matmul — quantization tuning inside it
  is capped below 1 before it starts. (Same lesson as the MoE ragged_dot
  history.)
- **The winner is a 30-line XLA composition** — now
  `ejkernel.kernels._xla.quantized_matmul.channelwise_quantized_matmul`
  (12 CPU parity tests): decode = `x @ w_q.astype(bf16)` (XLA fuses the
  upcast into the weight stream; ANY pre-dot arithmetic breaks the fusion)
  with the per-channel scale on the [m,n] output; prefill = per-token int8
  acts + native int8xint8 dot (459 TOPS/core = 2x bf16) + epilogue scales.
  Per-channel scales are load-bearing: K-blocked scales force either the
  fusion break or [blocks,m,n] int32 partials (1.9 GB at m=2048).

Where 2x stands, honestly:
- **Achieved**: W4A4 prefill 2.88x (int4 MXU is 920 TOPS/core = 4x bf16) —
  but relerr 0.11 unsmoothed, so it is an opt-in behind calibration, and the
  relerr gate correctly rejected it as a default.
- **Approached with size**: decode speedup grows as the dispatch floor
  (~125us, shared with bf16) amortizes — 27B-class shard: int8 1.52x, int4
  1.77x, asymptote = byte ratio.
- **Structurally capped in the harness min**: qkv-7B t=8 is floor-bound for
  BOTH paths; no single-op change can reach 2x there. Reaching 2x end-to-end
  needs either per-layer weights large enough to bury the floor (27B+ shards)
  or multi-op fusion per dispatch — a serving-integration lever, not a kernel
  one.

Kernel-side facts bought along the way: Mosaic CSEs repeated lhs quant
(hoisting: no-op); the 128-col strip loop genuinely pipelines VPU/MXU
(removing it: -6%); f32 epilogue is free (bf16 epilogue: slower AND less
accurate); int8xint4 is not MXU-native (upcast, int8 rate).

Next integration steps: route ejkernel's dense qmm TPU dispatch to
`channelwise_quantized_matmul` for int-target specs; adapters emit
per-channel int8/int4 on v5; wire `qmm_*` layer knobs; expose W4A4 as an
explicitly-calibrated opt-in.

## Autoresearch: packed-int4 decode / 4x attempt (2026-08-04)

Six iterations + two probes on `autoresearch-qmm-2x`. Deliverables in
`ejkernel/kernels/_pallas/tpu/quantized_matmul/_packed_gemv.py` (TPU parity
tests 3/3): `packed_int4_gemv` (W4A16, 2 weights/byte, split-K packing) and
`w4a4_gemv` (packed weights fed to the int4 MXU via `pltpu.bitcast` — zero
per-element decode). Bench: `benchmarks/bench_packed_int4_decode.py`.

Decode vs jitted bf16, TPU v5 single chip:

| path | 13B gate_up | 27B gate_up |
| --- | --- | --- |
| XLA fused-upcast int4 (W4A16) | 1.36-1.41x | 1.76-1.78x |
| Pallas packed W4A16 | 1.15-1.18x | 1.29-1.32x |
| **Pallas W4A4 (bitcast MXU feed)** | **1.44x** | **2.07x** (241us, kernel-exact vs its quantized semantics) |

**4x verdict for v5: not reachable, two measured walls.**
1. *Convert bound* — any W-A16 path converts every weight element for the
   MXU (~1.5-1.7T elem/s through the convert pipeline, XLA's fused convert
   being the best implementation); packing bytes doesn't help because bytes
   are not the pole. Cap ≈ 1.8x.
2. *int4-MXU ingest floor* — W4A4 removes the convert entirely (bitcast is a
   register reinterpret) but plateaus at ~223us/2.2x on 27B across every
   tile/split-k geometry: the MXU's weight-ingest rate at tiny m is the wall.

4x lives on v6e/v7 (native sub-byte MXU feeds / faster convert) or at
per-chip weights well beyond 27B-class. On v5 the practical ladder is:
prefill W8A8 1.77x / W4A4 2.88x (calibration-gated), decode W4A4 2.07x
(calibration-gated) or W4A16 1.8x (exact).

Mosaic v5 lowering facts collected: no 8-bit vector bitwise; `arith.shrui`
does not legalize (use signed shifts in int32); `bitcast(u8->int4)` expands
sublane-major matching adjacent-rows packing; CPU backend rejects int4
dot_general even in interpret mode (W4A4 tests are TPU-only).

## Push-more session: the cost model that closes the case (2026-08-07)

Follow-up probes on `autoresearch-qmm-2x` produced a complete cost model that
explains every measurement of the packed-decode investigation:

    time_per_op = fixed + bytes / 2.51 TB/s
    fixed = ~123us at a jit boundary; ~36us (native XLA op) / ~57us
    (pallas_call) inside a compiled graph.

Verified fits: w4a4 27B microbench 123+94=217us (measured 222); bf16 27B
123+375=498 (measured 497). Everything earlier called "MXU ingest floor" or
"dtype stream penalty" was this fixed cost polluting small-array rates — the
dtype-stream sweep showed all dtypes stream identically once overhead is
subtracted, and the size sweep fit fixed=123us, BW=2.51 (Pallas marginal)
/ 1.74 (XLA reduce marginal; XLA *matmul* streams at ~2.5 like Pallas).

Consequences, all measured:
- **In-graph (serving-realistic) w4a4 decode at 27B: 2.72x** (151.2 vs
  411.4us/op; jit-boundary microbench shows only 2.08-2.23x). In-graph is the
  number that matters for eSurge.
- `parallel` grid semantics beat `arbitrary` by ~8% for the w4a4 grid
  (222 vs 241us); promoted into the kernel. `CORE_PARALLEL` (megacore) gave
  nothing — Pallas single-kernel BW is the same ~2.51 TB/s marginal.
- XLA cannot read packed nibbles: `bitcast_convert_type+reshape` into a dot
  does NOT fuse (materializes, 6-13x slower) and its nibble order differs.
- Two measurement traps for future benches: XLA dead-column-eliminates
  weights whose outputs are sliced (a `[:, :K]` chain let bf16 "read" 940MB
  in 90us); and any chaining operand as large as the weight doubles the
  traffic being measured.

Remaining route to ~4x on v5, now exactly quantified: amortize the fixed cost
over more bytes per dispatch — a fused MLP kernel (gate_up+act+down in one
pallas_call) projects (2x411)/(57+188) = 3.4x, four fused weights ~3.8x.
That is an eSurge/layer-level integration, not further kernel tuning; the
w4a4 kernel itself sits on the DMA line.
## Fused MLP op (2026-08-07): forward + backward, modular, measured

Shipped on `autoresearch-qmm-2x`: `ejkernel.modules.fused_mlp` — one surface
for `down(act(gate(x)) * up(x))` across formats (bf16 / channelwise int8 /
channelwise int4 / packed-int4), layouts (separate, fused-concat,
TP-interleaved — normalized by `split_gate_up` at the boundary so kernels
carry zero layout branches), activations (silu/gelu/gelu_tanh/relu/sigmoid),
and **both directions**:

* forward: Pallas single-dispatch W4A4 kernel on TPU decode shapes
  (`kernels/_pallas/tpu/fused_mlp`, registered `Platform.PALLAS`); XLA
  composition otherwise (`kernels/_xla/fused_mlp`, registered fallback).
  The Pallas kernel keeps hidden activations in registers, re-quantizing
  per-(token, I-tile) — finer than the per-token scales that failed the
  accuracy gate — and is tested kernel-exact against a tile-wise NumPy
  replication of its semantics.
* backward: `custom_vjp`, flash-style recompute (no saved intermediates);
  dense weights get true dW (parity vs naive autodiff < 0.5% on all four
  cotangents); integer codes are frozen (float0 cotangents) with exact dx —
  the LoRA/frozen-backbone training contract.

Measured (TPU v5, single chip, `bench_fused_mlp.py`), whole MLP block:

| path | 13B decode | 27B decode | 27B train fwd+bwd (t=2048) |
| --- | --- | --- | --- |
| bf16 single-jit / in-graph | 303 / 207us | 689 / 589us | 16.3ms |
| w4a4 unfused (2 dispatches) | 185us | (VMEM OOM at tile 1024) | — |
| **w4a4 FUSED (1 dispatch, tile_i=512)** | **185us** | **298us = 1.98x in-graph / 2.31x single** | — |
| int8-frozen differentiable path | — | — | 15.5ms |

Honest notes (corrected after the optimize-pallas pass): the 197us "floor"
was WRONG — it used the DMA-only marginal bandwidth (2.51 TB/s) where the
int4-matmul weight-ingest rate applies (~1.06 TB/s; the same wall measured
for the single gemv AND for XLA's own int4 dot — all three agree, so it is
the hardware's int4-matmul ingest path at tiny m, not a schedule bug). True
floor ~ 352MB/1.06 + fixed ~= 330-370us; the kernel at 295us/tile_i=1024 is
AT its floor. Raising vmem_limit_bytes and quadrupling tile size moved it
~2% — step-count overhead was a misattribution, now falsified by experiment.
Dense-format training runs the plain composition under autodiff and is
bit-identical to XLA (1.00x, worst grad relerr 0.0).
13B MLPs are small enough that fixed costs still dominate (fused == unfused).
Training with quantized-frozen weights is ~5% faster than bf16 fwd+bwd (the
bwd is the same dense math; the fwd saves) — real training wins need the
int8-dot prefill path in bwd's transposed products, future work. Tests:
13 CPU + 2 TPU new, all green; layering intact.
## Open decisions

- **fp8 has no matching ejkernel mode.** Our modes are affine (integer),
  nf4, mxfp4/nvfp4, mxfp8/nvfp8 — none is "fp8 with arbitrary float scales".
  So phase 1 must pick a requantization target (affine int8 / mxfp8 group-32 /
  nvfp8 group-16) or add an fp8 mode to ejkernel. This is the first real
  decision of phase 1 and it wants a numerics comparison, not a coin flip.
- **Export.** The load direction is complete; export is stubbed
  (`RebuildCanonical` raises). The `inverse_spliter` contract for
  M-params-to-one-value has no consumer to validate against, and the export
  path also casts dtypes unconditionally (`parameters_transformation.py:1412`)
  — the symmetric fix to `preserve_dtype`, deliberately not made without a
  test vector.
- **STE-QAT (phase 5):** quantized params must not receive dense optimizer
  state. Settle before implementing.
- Whether requant-to-wider-block is ever worth it on our TPUs, or whether
  group-32 MX feeds the MXU adequately. Needs measurement, not inheritance from
  vLLM-TPU's default.
