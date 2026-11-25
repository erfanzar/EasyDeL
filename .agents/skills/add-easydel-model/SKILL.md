---
name: add-easydel-model
description: Add or update an EasyDeL model family, task head, configuration, registry entry, HF weight conversion via reform_param, or module test under libs/easydel/easydel/modules. Use for model zoo directories, AutoEasyDeLModel loading, TaskType registration, fused projection layout integration, hybrid/sparse cache wiring, or model tests.
---

# Skill: Add Or Update An EasyDeL Model

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. This skill covers the model-zoo contracts under `libs/easydel/easydel/modules`.

## First Reads

Read these authoritative code surfaces before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/infra/base_config.py`
- `libs/easydel/easydel/infra/base_module.py`
- `libs/easydel/easydel/infra/factory.py`
- `libs/easydel/easydel/infra/mixins/bridge.py`
- `libs/easydel/easydel/modules/__init__.py`
- `libs/easydel/easydel/modules/_base/` (`causal_lm_module.py`, `sequence_classification_module.py`,
  `embedding_module.py`, `vision_language_module.py`, etc.)
- `libs/easydel/easydel/modules/auto/auto_modeling.py`
- `libs/easydel/docs/infra/elarge_model.md` (when wiring into eLarge)

Note: `libs/easydel/docs/infra/adding_models.md` still shows older 5D sharding and manual `get_partition_rules()`
examples — treat `infra/base_config.py`, `infra/base_module.py`, and modern model modules below as authoritative.

Open an adjacent exemplar matching the architecture:

- Dense causal LM: `libs/easydel/easydel/modules/qwen3/{qwen3_configuration.py,modeling_qwen3.py,__init__.py}` +
  `libs/easydel/tests/modules/spmd/test_qwen3.py`
- Hybrid / linear attention (GDN):
  `libs/easydel/easydel/modules/qwen3_next/{qwen3_next_configuration.py,modeling_qwen3_next.py}` (uses
  `RaggedGatedDeltaRule` from `libs/easydel/easydel/operations/kernels/inference_gdn.py`)
- Sparse attention (DSA):
  `libs/easydel/easydel/modules/glm5_next/{glm5_next_configuration.py,modeling_glm5_next.py}` (uses
  `DeepSeekSparseAttention` and `dsa_segment_pool_scope`)

## Required Surfaces

A new model family needs:

- `libs/easydel/easydel/modules/<family>/<family>_configuration.py` with `@register_config("<model_type>")`
- `libs/easydel/easydel/modules/<family>/modeling_<family>.py` with `@register_module(TaskType.<task>, ...)`
- `libs/easydel/easydel/modules/<family>/__init__.py`
- **Registration in `libs/easydel/easydel/modules/__init__.py`** (import and add to `__all__` — without this,
  `@register_config` and `@register_module` never execute when `AutoEasyDeLModel*` resolves model types)
- Task heads subclassing the matching base under `libs/easydel/easydel/modules/_base/` (e.g., `BaseCausalLMModule`,
  `BaseSequenceClassificationModule`, `BaseEmbeddingModule`) and declaring `_config_class`, `_model_type`, and
  `_task_type`
- Tests under `libs/easydel/tests/modules/spmd/test_<family>.py` (and `libs/easydel/tests/modules/mpmd/` when adding a
  pipelineable layer stack)

Use `TaskType`, `register_config`, and `register_module` from `libs/easydel/easydel/infra/factory.py`. Never add a side
registry or bypass `AutoEasyDeLModel*`.

## Conversion And HF Compatibility

Individual models do **not** implement `from_torch` or `to_torch`:

- PyTorch export is centralized in `EasyDeLBaseModule.to_torch` (`libs/easydel/easydel/infra/base_module.py`).
- PyTorch import lives in `EasyBridgeMixin._from_torch_pretrained` (`libs/easydel/easydel/infra/mixins/bridge.py`).
- Models hook custom weight remapping, transposes, or projection fusion into `EasyDeLBaseModule._get_reform_param` by
  defining `@property def reform_param` on the relevant submodules (e.g., `Qwen3MLP.reform_param` in
  `modeling_qwen3.py`, `Qwen3NextLinearAttentionLayout.reform_param` in `modeling_qwen3_next.py`).
- Add or update coverage in `libs/easydel/tests/modules/test_conversion_roundtrip.py` (`DENSE_MODELS`, `MOE_MODELS`,
  `SSM_MODELS`, `ROUNDTRIP_MODELS`, `KNOWN_GAPS`) and verify tensor names, shapes, transposes, fused packing, and
  dtypes on real tensors.

When the model uses fused QKV, fused gate/up, fused MoE experts, quantized linears, or pre-quantized checkpoint
formats, also load `.agents/skills/quantization-layout/SKILL.md`.

## Architecture, Pipeline & Cache Contracts

- **6-axis sharding:** use EasyDeL's 6-axis convention `("pp", "dp", "fsdp", "ep", "tp", "sp")` via `EasyDeLBaseConfig`,
  `PartitionAxis`, and `spectrax.sharding` (`.agents/skills/spectrax-sharding/SKILL.md`). Prefer reusable layers from
  `libs/easydel/easydel/layers/` over model-local helpers.
- **Layer stack, remat & MPMD PP:** wire decoder stacks through `EasyDeLLayerStackMixin` (`assign_layer_stage`,
  `scan_layers`, `auto_remat`, `checkpoint_name` in `infra/base_module.py`) so gradient checkpointing, scanned layers,
  and MPMD pipeline parallelism work cleanly.
- **Cache & mask alignment:** for sliding-window, hybrid, linear-attention, or sparse models, keep `config.layer_types`
  and `config.get_mask_details()` consistent with the cache config and attention operations (mismatches fail silently
  or surface as distant shape errors; see `.agents/skills/debug-easydel-cache/SKILL.md`).
- Derive vocabulary-sensitive dimensions from tensors when present, preserve public output containers/shapes, and keep
  attention softmax in `attn_softmax_dtype` (default `float32`).

## Verification

Model tests execute JAX numerics and must run on an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). A CPU run is not validation — without an accelerator, report the
change as unverified on hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/spmd/test_<family>.py

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/test_conversion_roundtrip.py -k <family>
```

When touching decoder layer stacks or pipeline stage assignment, also verify MPMD coverage:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/modules/test_pipeline_stage_constructor_coverage.py \
                libs/easydel/tests/modules/mpmd/test_<family>.py

uv run lint-imports
```

Do not claim training or serving readiness from constructor-only tests. For eLarge training or eSurge serving claims,
also load `.agents/skills/train-elarge/SKILL.md` or `.agents/skills/debug-esurge/SKILL.md`.
