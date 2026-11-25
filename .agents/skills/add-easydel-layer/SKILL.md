---
name: add-easydel-layer
description: Add or update reusable EasyDeL neural-network layers under libs/easydel/easydel/layers. Use for attention/DSA variants, ParallelLinear, norms, RoPE, MoE routing, embeddings, linear attention, indexers, compressed window, residuals, engram, quantization-aware layers, or fused projection layouts.
---

# Skill: Add Or Update An EasyDeL Layer

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the change lives in `libs/easydel/easydel/layers` or when a
model needs a new reusable primitive.

## First Reads

Read the relevant surfaces before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/layers/__init__.py`
- `libs/easydel/easydel/layers/_sharding.py`
- `libs/easydel/easydel/layers/mlp.py`
- `libs/easydel/easydel/layers/attention/_unified.py`, `_flexible.py`, `_decoder_base.py`, `_dsa.py`, `_compressed_window.py`
- `libs/easydel/easydel/layers/linears/_linear.py`, `_linear_quantized.py`
- `libs/easydel/easydel/layers/embeddings/_embeddings.py`, `_vision.py`
- `libs/easydel/easydel/layers/norms/_norms.py`
- `libs/easydel/easydel/layers/rotary/_rotary.py`
- `libs/easydel/easydel/layers/moe/_moe_module.py`
- `libs/easydel/easydel/layers/indexer/_indexer.py`
- `libs/easydel/easydel/layers/linear_attention/` (`_conv_state.py`, `_depthwise_conv.py`, `_kda.py`, `_masking.py`)
- `libs/easydel/easydel/layers/residual/` (`_gated.py`, `_manifold.py`)
- `libs/easydel/easydel/layers/compressed_window/`, `engram/`, `sparse_attention/`
- `libs/easydel/easydel/layers/quantization/_configs.py`, `_quants.py`, `_quantized_training.py`, `checkpoint/`
- `libs/easydel/easydel/layers/layouts/_builders.py`, `_dense.py`, `_moe.py`

## Layer Contracts

- **SpectraX module model:** subclass `spx.Module` and follow `.agents/skills/spectrax-modules/SKILL.md` for mutable
  state, buffers, and transform compatibility.
- **Dtype triple & softmax precision:** honor `dtype` (activation compute), `param_dtype` (storage), and `precision`
  (matmul). Keep attention softmax accumulation in `attn_softmax_dtype` (default `float32`) — never downcast to `bf16`.
- **Logical sharding:** annotate parameters with `ColumnWise`, `RowWise`, `SRowWise`, or `Replicated` from
  `spectrax.common_types`, and reuse `resolve_safe_sharding` / `all_gather_then_dot` from
  `libs/easydel/easydel/layers/_sharding.py` (backed by `libs/easydel/easydel/infra/sharding.py`) and `PartitionAxis`.
- **Reuse shared building blocks:** extend `UnifiedAttention`, `DeepSeekSparseAttention` (with
  `dsa_segment_pool_scope`), `ParallelLinear`, `BaseMoeModule`, `GatedResidual`, `ManifoldHyperConnection`, and MLP
  helpers (`gated_mlp_forward`, `clamped_swiglu`, `moe_gated_expert_forward`) instead of duplicating sharding, RoPE, or
  gating logic inside model files.
- **HF conversion hook:** expose `@property def reform_param` on fused or custom projection layers so weight conversion
  stays declarative rather than ad hoc per model.
- **No new `os.environ` knobs:** wire new toggles through layer/config arguments, ContextVars, or explicit setters.
- **Hardware kernels belong in eJKernel:** never inline Triton/Pallas/CUDA/CuTe/TileLang kernels in `easydel/layers/`;
  put kernels in `libs/ejkernel/` (with `Platform.XLA`, `Backend.ANY` fallback) and expose them via registered
  `OperationImpl` adapters in `libs/easydel/easydel/operations/kernels/`.

## Import-Order Safety (`layers/__init__.py`)

`libs/easydel/easydel/layers/__init__.py` intentionally enforces cold-import order safety:

- **Do NOT eagerly re-export `easydel.layers.attention`** (`UnifiedAttention`, `FlexibleAttentionModule`,
  `DeepSeekSparseAttention`) from `layers/__init__.py` — attention primitives live under `easydel.layers.attention`.
- **Do NOT eagerly import `easydel.layers.indexer`** at top level of `layers/__init__.py`. Because `easydel.caching`
  imports `layers.quantization` during initialization while `layers.indexer` and `layers.attention` depend on
  `easydel.operations` -> `easydel.caching`, public indexer exports in `layers/__init__.py` must go through
  `_INDEXER_EXPORTS` + lazy `__getattr__`.
- Always verify cold-import order with `libs/easydel/tests/layers/test_import_safety.py`.

## Routing

- Quantized linears, QAT rules, pre-quantized checkpoint adapters, or fused layouts:
  `.agents/skills/quantization-layout/SKILL.md`.
- New or updated EasyDeL operation adapter for attention/GDN/linear ops:
  `.agents/skills/port-ejkernel-to-easydel-operation/SKILL.md` (and `.agents/skills/add-ejkernel-kernel/SKILL.md` for
  new backend kernels).
- SpectraX module/transform behavior: `.agents/skills/spectrax-modules/SKILL.md`.
- Sharding & mesh policies: `.agents/skills/spectrax-sharding/SKILL.md`.
- Model-level integration: `.agents/skills/add-easydel-model/SKILL.md`.

## Verification

Run the cold-import safety guard (marked `cpu_ok`; CPU trio is allowed) and import-layering check:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/layers/test_import_safety.py

uv run lint-imports
```

All layer computation tests execute JAX numerics and must run on an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as unverified on
hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/layers/<area>/
```

When touching attention, linear projections, or sharding helpers, also run:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/layers/linears libs/easydel/tests/layers/attention
```
