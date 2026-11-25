---
name: add-easydel-trainer
description: Add or update an EasyDeL trainer algorithm under libs/easydel/easydel/trainers. Use when creating a new SFT, preference, RL, distillation, reward, or embedding trainer, extending TrainingArguments, or adding trainer registry entries and tests.
---

# Skill: Add Or Update An EasyDeL Trainer

Specialization of `.agents/skills/run-research/SKILL.md`. Load `run-research` first when adding a trainer or changing
`TrainingArguments`, `BaseTrainer`, or compiled training steps.

## First Reads

- `WORKSPACE.md` and `libs/easydel/pyproject.toml`
- Core plumbing: `libs/easydel/easydel/trainers/{base_trainer.py,training_configurations.py,trainer_protocol.py,__init__.py}`
  and `libs/easydel/easydel/trainers/trainer/trainer.py`
- Closest exemplar package:
  - Supervised: `libs/easydel/easydel/trainers/supervised_fine_tuning_trainer/{sft_config.py,sft_trainer.py}`
  - Preference: `libs/easydel/easydel/trainers/direct_preference_optimization_trainer/{dpo_config.py,dpo_trainer.py}`
  - Online RL: `libs/easydel/easydel/trainers/group_relative_policy_optimization/{grpo_config.py,grpo_trainer.py}`
- Distributed/step infrastructure:
  - `libs/easydel/easydel/trainers/data_lanes.py` (Grain map lanes, `PackedParquetLoader` cursor checkpointing, `data_lane_local_batch`)
  - `libs/easydel/easydel/trainers/batch_placement.py` (`BatchPlacedStep`, `batch_shard_group_rows`)
  - `libs/easydel/easydel/trainers/execution_accumulation.py` (`gradient_accumulation_mode="execution"`)
  - `libs/easydel/easydel/trainers/dp_sharding.py` (`dp_spmd_vmap_scope`, `DP_REDUCE_COMPILE_OPTIONS`)
- eLarge typing surface: `libs/easydel/easydel/infra/elarge/types/training.py`

## Required Surfaces

1. **Config**: `libs/easydel/easydel/trainers/<name>_trainer/<name>_config.py` subclassing `TrainingArguments` and
   decorated with `@Registry.register("trainer-arguments", "<name>")`.
2. **Trainer**: `libs/easydel/easydel/trainers/<name>_trainer/<name>_trainer.py` subclassing `Trainer` or `BaseTrainer`
   and decorated with `@Registry.register("trainer", "<name>")`.
3. **Step / loss module (optional)**: `_fn.py` for loss math and compiled step functions. To opt into
   `gradient_accumulation_mode="execution"`, accept keyword-only `accumulate_into` and `apply_accumulated` (see
   `libs/easydel/easydel/trainers/distillation_trainer/_fn.py`).
4. **Package exports**: `libs/easydel/easydel/trainers/<name>_trainer/__init__.py` re-exporting public classes only
   (registration decorators live on the config and trainer classes).
5. **Eager top-level import**: import both config and trainer eagerly in `libs/easydel/easydel/trainers/__init__.py`
   — **never** put registered trainers in `_LAZY_EXPORTS`, or `Registry.get_or_raise(..., wakeup=True)` will fail to
   discover them.
6. **eLarge config sync**: update `libs/easydel/easydel/infra/elarge/types/training.py` (`BaseTrainerCfg`,
   `<Name>TrainerCfg`, `TrainerConfig`, `_TRAINER_TYPE_ALIASES`) when adding a trainer or `TrainingArguments` fields.
7. **Tests**:
   - Add a non-default value for every new `TrainingArguments` field in
     `libs/easydel/tests/trainers/test_training_arguments_save_load_roundtrip.py`.
   - Add loss-math parity tests (independent reference, e.g. `test_distillation_loss_math.py`,
     `test_trl_dpo_loss_parity.py`) and trainer smoke tests under `libs/easydel/tests/trainers/`.

## Invariants

- Do not add side registries or bypass `BaseTrainerProtocol`.
- **No new `os.environ` knobs**: expose new switches as `TrainingArguments` / trainer-config fields, ContextVars, or
  explicit setters.
- `TrainingArguments.data_lane_local_batch=True` requires an SPMD `BatchPlacedStep` (not MPMD or custom step shardings).
- `gradient_accumulation_mode="execution"` splits grad accumulation and optimizer apply into two programs; it is
  rejected when `mpmd_scheduler` is set.
- Remote/GCS checkpoint paths go through `spectrax.serialization` (`Checkpointer`, `AsyncCheckpointManager`,
  `libs/spectrax/spectrax/serialization/_fs.py`) or `ePath` — never raw `pathlib.Path`.

## Routing

- Dataset / packing / mixing: `.agents/skills/build-dataset-pipeline/SKILL.md`
- Compile-time HBM OOM, remat, or execution accumulation: `.agents/skills/debug-training-oom/SKILL.md`
- eLarge YAML wiring: `.agents/skills/train-elarge/SKILL.md`
- Checkpoint save/load & GCS serialization: `.agents/skills/spectrax-serialization/SKILL.md`
- Sharding & meshes: `.agents/skills/spectrax-sharding/SKILL.md`
- Quantized model training: `.agents/skills/quantization-layout/SKILL.md`

## Verification

Computation tests (trainer steps, loss math, batch placement, execution accumulation) run on the accelerator (GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation changes as **unverified on
hardware**:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/trainers/test_<name>.py \
    libs/easydel/tests/trainers/test_execution_accumulation.py \
    libs/easydel/tests/trainers/test_batch_placement.py
```

Pure-Python config tests (`test_training_arguments_save_load_roundtrip.py` carries `cpu_ok`, and
`test_training_arguments_tx_template.py` / `test_training_configuration_validation.py` do no array compute) can run
under the CPU trio when the accelerator is busy:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_training_arguments_save_load_roundtrip.py \
    libs/easydel/tests/trainers/test_training_arguments_tx_template.py \
    libs/easydel/tests/trainers/test_training_configuration_validation.py
```

Do not claim training readiness from constructor-only tests; a full training claim requires a real step on hardware or
an explicit unverified-on-hardware note.
