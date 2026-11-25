---
name: add-eformer-optimizer
description: Add or update an optimizer or scheduler in libs/eformer/eformer/optimizers and wire it into EasyDeL/eLarge configs. Use for OptimizerFactory, SchedulerFactory, _tx/ custom Optax transforms, stage-local MPMD optimizers (_stage_local.py), or EasyDeL fused optimizers (fused_adamw/fused_lion/fused_rmsprop).
---

# Skill: Add Or Update An eFormer Optimizer Or Scheduler

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

Use this skill for work inside `libs/eformer/eformer/optimizers/` or when exposing a new optimizer/scheduler in EasyDeL
(`EasyDeLOptimizers`, `EasyDeLSchedulers`, `TrainingArguments`, eLarge YAML configs) or registering a custom optimizer
from EasyDeL via `@register_optimizer`.

## Package Boundary

`libs/eformer` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `spectrax`, `ejkernel`, `eray`, or `surgecore`. EasyDeL-specific optimizers (such as
`libs/easydel/easydel/trainers/fused_optimizers.py`) live in `libs/easydel` and register into eFormer via
`@register_optimizer`. Verify with `uv run lint-imports`.

## First Reads

- eFormer optimizer core:
  - `libs/eformer/eformer/optimizers/__init__.py`
  - `libs/eformer/eformer/optimizers/_base.py` (`OptimizerBuilder`, `SchedulerBuilder`, `@register_optimizer`, `@register_scheduler`)
  - `libs/eformer/eformer/optimizers/_config.py` (`SerializationMixin`, `SchedulerConfig`, optimizer config dataclasses)
  - `libs/eformer/eformer/optimizers/_factory.py` (`TxConfigs`, `SchedulerFactory`, `OptimizerFactory`)
  - `libs/eformer/eformer/optimizers/_builders.py` (`_build_metadata_mpmd_optimizer`, registered builders)
  - `libs/eformer/eformer/optimizers/_stage_local.py` (`StageLocalOptimizerMetadata`, `StageLocalGradientTransformation`, stage-local kernels)
  - `libs/eformer/eformer/optimizers/_tx/{__init__.py,mars.py,white_kron.py,utils.py}`
- EasyDeL wiring (when exposing to EasyDeL / eLarge):
  - `libs/easydel/easydel/infra/etils.py` (`EasyDeLOptimizers`, `EasyDeLSchedulers`)
  - `libs/easydel/easydel/trainers/training_configurations.py` (`get_optimizer_and_scheduler`)
  - `libs/easydel/easydel/trainers/fused_optimizers.py` (in-repo external registration pattern: `fused_adamw`, `fused_lion`, `fused_rmsprop`)

## Required Surfaces (New Optimizer)

Currently registered optimizers: 8 in eFormer (`adamw`, `adafactor`, `lion`, `rmsprop`, `muon`, `quad`, `skew`, `mars`)
plus 3 registered from EasyDeL (`fused_adamw`, `fused_lion`, `fused_rmsprop`). Note that `KronConfig`, `ScionConfig`,
and `SoapConfig` exist in `_config.py` as orphan configs with no registered builders.

To add a new optimizer:

1. **Custom Optax transform (if needed):** implement in `libs/eformer/eformer/optimizers/_tx/<name>.py` and export from
   `libs/eformer/eformer/optimizers/_tx/__init__.py` (reusing `libs/eformer/eformer/optimizers/_tx/utils.py` helpers
   `optax_add_scheduled_weight_decay`, `build_path_exclusion_mask`, `optax_scale_by_path_multipliers`).
2. **Config dataclass:** add `<Name>Config(SerializationMixin)` in `_config.py` and add it to the `TxConfigs` type union
   in `_factory.py`.
3. **Builder:** add `@register_optimizer("<name>")` `@dataclasses.dataclass` `<Name>Optimizer(OptimizerBuilder)` in
   `_builders.py` with a `config: <Name>Config` field.
4. **Stage-local MPMD support (if PP-compatible):** add `_apply_<name>_stage_local` in `_stage_local.py` and implement
   `build_mpmd()` via `_build_metadata_mpmd_optimizer(...)`. Builders without `build_mpmd()` fall back to
   `make_unsupported_stage_local_gradient_transformation` (normal Optax works; stage-local PP raises `NotImplementedError`).
5. **Exports:** re-export config and builder from `libs/eformer/eformer/optimizers/__init__.py`.
6. **EasyDeL / eLarge wiring (when exposed to EasyDeL):**
   - Add enum entry to `EasyDeLOptimizers` in `libs/easydel/easydel/infra/etils.py`.
   - Ensure `TrainingArguments.get_optimizer_and_scheduler` (`training_configurations.py`) forwards any new kwargs
     cleanly (note: `OptimizerFactory.create` warns on unconsumed non-`None` kwargs).
7. **Tests:** add tests in `libs/eformer/tests/optimizers/test_optimizers.py` (factory creation, serialization round-trip,
   parity, and stage-local MPMD apply if implemented) plus EasyDeL config/trainer checks if wired into EasyDeL.

## Critical Footguns (Verified In Source)

1. **`@register_scheduler` is currently ignored by `SchedulerFactory`:**
   - `SchedulerFactory.create_scheduler` (`_factory.py`) does **NOT** consult `_SCHEDULER_BUILDER_REGISTRY`; it
     hardcodes `if config.scheduler_type == "linear" ... elif config.scheduler_type == "cosine" ... else: raise ValueError(...)`.
   - `SchedulerConfig.scheduler_type` (`_config.py`) is also hardcoded to `Literal["linear", "cosine"] | None` with
     hardcoded branches in `_validate()`.
   - Adding a new scheduler ONLY with `@register_scheduler` creates an **unreachable** scheduler. You must EITHER update
     `SchedulerConfig.scheduler_type` + `_validate()` (`_config.py`), `SchedulerFactory.create_scheduler` (`_factory.py`),
     `_builders.py`, and `EasyDeLSchedulers` (`libs/easydel/easydel/infra/etils.py`), OR refactor
     `SchedulerFactory.create_scheduler` to dispatch through `_SCHEDULER_BUILDER_REGISTRY`.
2. **NEVER add `from __future__ import annotations` to optimizer builder files:**
   - `OptimizerFactory._get_config_class` (`_factory.py`) reads `config_field.type` directly from
     `dataclasses.fields(builder_cls)` and passes it straight to `isinstance(optimizer_config, config_cls)` (and
     `config_cls()`). Enabling postponed string annotations turns `config_field.type` into a `str` and crashes
     `OptimizerFactory.create`.
3. **Double weight-decay trap & exact chain order:**
   - `OptimizerFactory._build_optimizer_chain` (`_factory.py`) assembles transforms in this exact order:
     `optax.clip_by_global_norm(clip_grad)` -> `base_optimizer` -> `optax_add_scheduled_weight_decay` (honoring
     `weight_decay_mask` and `no_weight_decay_patterns`) -> `optax_scale_by_path_multipliers` (`param_lr_multipliers`)
     -> `optax.MultiSteps(tx, gradient_accumulation_steps)` -> `builder.build_mpmd(...)`.
   - Because the factory adds `optax_add_scheduled_weight_decay` externally, base builders like `AdamWOptimizer` and
     `FusedAdamWOptimizer` MUST pass `weight_decay=0.0` to the underlying transform or weight decay will be applied
     twice. (Note: `LionOptimizer` and `fused_lion` intentionally retain `optax.lion`'s default `weight_decay=1e-3` for
     parity.)
4. **Stage-local MPMD constraints (`_stage_local.py`):**
   - `StageLocalOptimizerMetadata` explicitly rejects `gradient_accumulation_steps > 1` at runtime.
   - Multi-submesh global norm reduction uses `_mpmd_grad_global_sq_norm` (per-submesh jitted sq-norm + single
     `process_allgather` from lowest owner rank); cross-device-set resharding uses `_mpmd_safe_reshard_leaf`
     (`make_array_from_single_device_arrays`) to avoid multi-controller JAX cross-host `device_put` errors.
5. **String dtype conversion:** `OptimizerFactory._convert_dtypes` automatically converts any config field whose name
   contains `"dtype"` from `str` to `jax.numpy.<dtype>`.

## Routing

- Training / eLarge YAML configuration: `.agents/skills/train-elarge/SKILL.md`
- MPMD pipeline runtime & `sxvalue_and_grad_and_apply`: `.agents/skills/spectrax-pipeline-runtime/SKILL.md`
- Training OOM / optimizer-state memory: `.agents/skills/debug-training-oom/SKILL.md`

## Verification

Computation tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/eformer/tests/optimizers/
```

When wiring a new optimizer into EasyDeL, also verify `OptimizerFactory.create("<name>", ...)` and EasyDeL's
`TrainingArguments(optimizer="<name>").get_optimizer_and_scheduler(...)` on accelerator hardware.
