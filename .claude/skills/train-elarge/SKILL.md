---
name: train-elarge
description: Create, validate, run, or debug EasyDeL eLarge YAML training, evaluation, or serving configs. Use for python -m easydel.scripts.elarge, actions lists, SFT/reward/KTO/ORPO configs, EasyData trainer integration, quantization config wiring, or eLarge serve mode.
---

# Skill: Train Or Serve With eLarge

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the entry point is
`libs/easydel/easydel/scripts/elarge.py` or an eLarge config/type object.

## First Reads

Read these before editing configs or code:

- `WORKSPACE.md`
- `libs/easydel/docs/scripts/elarge_cli.md`
- `libs/easydel/docs/infra/elarge_model.md`
- `libs/easydel/easydel/scripts/elarge.py`
- `libs/easydel/easydel/infra/elarge/model.py`
- `libs/easydel/easydel/infra/elarge/processing.py`
- `libs/easydel/easydel/infra/elarge/builders.py`
- `libs/easydel/easydel/infra/elarge/defaults.py`
- `libs/easydel/easydel/infra/elarge/types/root.py`
- `libs/easydel/easydel/infra/elarge/types/model.py`
- `libs/easydel/easydel/infra/elarge/types/data.py`
- `libs/easydel/easydel/infra/elarge/types/training.py`
- `libs/easydel/easydel/infra/elarge/types/eval.py`
- `libs/easydel/easydel/infra/elarge/types/engine.py`
- `libs/easydel/easydel/infra/elarge/types/quantization.py`
- `libs/easydel/tests/elarge_configs/`

## CLI Contract

Run eLarge through:

```bash
uv run python -m easydel.scripts.elarge --config <config.yaml>
```

The script also accepts a positional config path and `--dry-run`. YAML must
contain an `actions:` list. Supported actions include `validate`, `print`,
`show`, `dump_config`, `print_config`, `config`, `to_json`, `to_yaml`, `train`,
`eval`, and `serve`.

Start with validation:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run python -m easydel.scripts.elarge --config <config.yaml> --dry-run
```

## Config Routing

- Model or conversion trouble: load `.claude/skills/add-easydel-model/SKILL.md`
  or `.claude/skills/convert-checkpoint/SKILL.md`.
- Dataset, packing, or mixed-source trouble: load
  `.claude/skills/build-dataset-pipeline/SKILL.md`.
- Quantized model build trouble: load
  `.claude/skills/quantization-layout/SKILL.md`.
- Compile-time HBM OOM: load `.claude/skills/debug-training-oom/SKILL.md`.
- Serve mode or OpenAI-compatible output trouble: load
  `.claude/skills/debug-esurge/SKILL.md` and, for tool/reasoning outputs,
  `.claude/skills/tool-reasoning-parser/SKILL.md`.

## Serve Mode

`serve` supports parameters such as `host`, `port`, `workers`, `log_level`,
`ssl_keyfile`, `ssl_certfile`, `tool_parser_name`, `oai_like_processor`,
`enable_function_calling`, `require_api_key`, `admin_key`, `enable_cors`, and
`cors_origins`. The current script requires `workers` to be `1` and does not
support reload mode.

## Verification

Use focused config tests and builder tests before long runs:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/infra/elarge/test_builders_quantization_qmm_kwargs.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/trainers/test_sequence_packing_flag.py
```

Do not infer training success from `--dry-run`. A train/eval claim needs the
actual affected action or a clearly named skipped-hardware risk.
