---
name: add-easydel-model
description: Add or update an EasyDeL model family, task head, configuration, registry entry, HF conversion path, or module test under libs/easydel/easydel/modules. Use when working on model zoo directories, AutoEasyDeLModel loading, TaskType registration, from_torch/to_torch conversion, fused projection layout integration, or model tests.
---

# Skill: Add Or Update An EasyDeL Model

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. This skill adds the model-zoo routing for
`libs/easydel`.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/docs/infra/adding_models.md`
- `libs/easydel/docs/infra/base_config.md`
- `libs/easydel/docs/infra/base_module.md`
- `libs/easydel/docs/infra/elarge_model.md` when the model is meant to train
  or serve through eLarge.
- `libs/easydel/easydel/infra/factory.py`
- `libs/easydel/easydel/modules/auto/auto_modeling.py`

Then open a nearby model family with the same task shape. For decoder-only
causal LM work, start with:

- `libs/easydel/easydel/modules/qwen3/qwen3_configuration.py`
- `libs/easydel/easydel/modules/qwen3/modeling_qwen3.py`
- `libs/easydel/easydel/modules/qwen3/__init__.py`
- `libs/easydel/tests/modules/spmd/test_qwen3.py`

## Required Surfaces

A new model family usually needs:

- `libs/easydel/easydel/modules/<family>/<family>_configuration.py`
- `libs/easydel/easydel/modules/<family>/modeling_<family>.py`
- `libs/easydel/easydel/modules/<family>/__init__.py`
- config registration with `@register_config("<model_type>")`
- module registration with `@register_module(TaskType.<task>, ...)`
- `_model_type` and `_task_type` on task heads
- tests under `libs/easydel/tests/modules/spmd/`

Use `TaskType`, `register_config`, and `register_module` from
`libs/easydel/easydel/infra/factory.py`. Do not add a side registry or bypass
`AutoEasyDeLModelForCausalLM` and related auto classes.

## Conversion And HF Compatibility

If the model maps to Hugging Face weights:

- inspect `from_torch` and `to_torch` implementations in nearby models
- add or update conversion tests near
  `libs/easydel/tests/modules/test_conversion_roundtrip.py`
- verify tensor names, transposes, fused projection packing, and dtype handling
  from real tensors, not by assuming config names are enough

When the model uses fused QKV, fused gate/up projections, quantized linears, or
TP-portable checkpoint layout, load
`.claude/skills/quantization-layout/SKILL.md` before editing.

## Shape And Sharding Rules

- Prefer existing EasyDeL layers and sharding APIs over model-local helpers.
- Keep mesh and partition semantics compatible with `PartitionAxis` and the
  model's existing config fields.
- Derive vocabulary-sensitive behavior from tensor shapes when the tensor is
  already present.
- Preserve public return shapes and cache structures from adjacent model heads.

## Verification

Use this CPU environment for host-side and SPMD unit tests:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/modules/spmd/test_<family>.py
```

For conversion work, add or run a focused roundtrip check:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/modules/test_conversion_roundtrip.py
```

Do not claim training or serving readiness from constructor-only tests. If the
claim is eLarge training or eSurge serving, also load the matching skill.
