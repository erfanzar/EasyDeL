---
name: add-easydel-trainer
description: Add or update an EasyDeL trainer algorithm under libs/easydel/easydel/trainers. Use when creating a new SFT, preference, RL, distillation, reward, or embedding trainer, extending TrainingArguments, or adding trainer registry entries and tests.
---

# Skill: Add Or Update An EasyDeL Trainer

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the work is inside
`libs/easydel/easydel/trainers` or when `TrainingArguments` / `BaseTrainer`
behavior must change.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/trainers/base_trainer.py`
- `libs/easydel/easydel/trainers/training_configurations.py`
- `libs/easydel/easydel/trainers/trainer_protocol.py`
- `libs/easydel/easydel/trainers/trainer/trainer.py`
- `libs/easydel/easydel/trainers/supervised_fine_tuning_trainer/sft_config.py`
- `libs/easydel/easydel/trainers/supervised_fine_tuning_trainer/sft_trainer.py`
- `libs/easydel/easydel/trainers/direct_preference_optimization_trainer/dpo_config.py`
- `libs/easydel/easydel/trainers/direct_preference_optimization_trainer/dpo_trainer.py`
- `libs/easydel/easydel/trainers/group_relative_policy_optimization/grpo_config.py`
- `libs/easydel/easydel/trainers/group_relative_policy_optimization/grpo_trainer.py`
- `libs/easydel/easydel/trainers/__init__.py`

Pick the closest existing trainer (SFT for supervised, DPO for preference, GRPO for online RL) and mirror its package
shape.

## Required Surfaces

A new trainer usually needs:

- `libs/easydel/easydel/trainers/<name>_trainer/<name>_config.py` extending
  `TrainingArguments`
- `libs/easydel/easydel/trainers/<name>_trainer/<name>_trainer.py` extending
  `Trainer` or `BaseTrainer`
- optional `_fn.py` for step functions / loss helpers
- `libs/easydel/easydel/trainers/<name>_trainer/__init__.py` exporting the public classes and registering with
  `Registry`
- `@Registry.register("trainer", "<name>")` or the equivalent registration used by `train-elarge`
- tests under `libs/easydel/tests/trainers/`

Do not add a side registry or bypass `BaseTrainerProtocol`.

## Routing

- Dataset / packing / mixing trouble: load
  `.xerxes/skills/build-dataset-pipeline/SKILL.md`.
- Compile-time HBM OOM or remat issues: load
  `.xerxes/skills/debug-training-oom/SKILL.md`.
- eLarge YAML wiring: load `.xerxes/skills/train-elarge/SKILL.md`.
- Checkpoint save/load or sharding layout: load
  `.xerxes/skills/eformer-checkpoint-sharding/SKILL.md`.
- Quantized model training: load `.xerxes/skills/quantization-layout/SKILL.md`.

## Verification

Run the affected trainer's own CPU smoke tests:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_<name>.py
```

If you changed `TrainingArguments` or the trainer registry, also run:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_training_arguments.py
```

Do not claim training readiness from constructor-only tests. A full training claim needs a short real run or a clear
hardware-risk note.
