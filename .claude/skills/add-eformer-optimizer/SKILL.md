---
name: add-eformer-optimizer
description: Add or update an optimizer or scheduler in libs/eformer/eformer/optimizers. Use for OptimizerFactory/SchedulerFactory, new optimizer builders, stage-local gradient transforms, or wiring configs into training/eLarge YAMLs.
---

# Skill: Add Or Update An eFormer Optimizer

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/eformer/eformer/optimizers` or when a new optimizer/scheduler must be
available through `OptimizerFactory`.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/eformer/pyproject.toml`
- `libs/eformer/eformer/optimizers/__init__.py`
- `libs/eformer/eformer/optimizers/_base.py`
- `libs/eformer/eformer/optimizers/_config.py`
- `libs/eformer/eformer/optimizers/_factory.py`
- `libs/eformer/eformer/optimizers/_builders.py`
- `libs/eformer/eformer/optimizers/_stage_local.py`

## Required Surfaces

A new optimizer usually needs:

- a config dataclass in `_config.py` extending the optimizer/scheduler config
  pattern and `SerializationMixin`
- a builder subclass in `_builders.py` extending `OptimizerBuilder` and
  registered with `@register_optimizer`
- export from `libs/eformer/eformer/optimizers/__init__.py`
- if pipeline parallelism is required, a stage-local apply kernel in
  `_stage_local.py` and a `build_mpmd()` path in the builder
- tests under `libs/eformer/tests/optimizers/`

A new scheduler follows the same pattern with `@register_scheduler` and
`SchedulerBuilder`.

## Routing

- Training / eLarge YAML wiring: load `.claude/skills/train-elarge/SKILL.md`.
- Training OOM or memory issues: load
  `.claude/skills/debug-training-oom/SKILL.md`.
- MPMD pipeline runtime issues: load
  `.claude/skills/spectrax-pipeline-runtime/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eformer/tests/optimizers/
```

Also verify the optimizer is reachable through `OptimizerFactory.create(...)`
with a minimal config object.
