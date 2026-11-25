---
name: train-elarge
description: Create, validate, run, or debug EasyDeL eLarge YAML training, evaluation, or serving configs. Use for python -m easydel.scripts.elarge, actions lists, SFT/preference/RL/distillation configs, auxiliary model roles, EasyData trainer integration, quantization config wiring, or eLarge serve / serve.replicas mode.
---

# Skill: Train Or Serve With eLarge

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the entry point is
`libs/easydel/easydel/scripts/elarge.py` or an `eLargeModel` / `eLMConfig` object.

## First Reads

Read the authoritative code surfaces before editing configs or builders:

- `WORKSPACE.md`
- `libs/easydel/easydel/scripts/elarge.py`
- `libs/easydel/easydel/infra/elarge/{model.py,processing.py,builders.py,defaults.py,benchmarking.py}`
- `libs/easydel/easydel/infra/elarge/types/{root.py,model.py,infra.py,data.py,training.py,eval.py,engine.py,quantization.py,aliases.py}`
- `libs/easydel/docs/infra/elarge_model.md`
- `libs/easydel/docs/scripts/elarge_cli.md` (note: still contains older 5-axis sharding and deprecated
  `serve.tool_parser_name` examples — treat `libs/easydel/easydel/scripts/elarge.py` and
  `libs/easydel/easydel/infra/elarge/types/` as authoritative)
- `libs/easydel/tests/elarge_configs/`

## CLI Contract & Actions

Run eLarge through:

```bash
uv run python -m easydel.scripts.elarge --config <config.yaml>
```

- Accepts `-c` / `--config`, a positional YAML path, and `--dry-run`.
- YAML layout supports `config:`, `elarge_model:`, `elm:` wrappers, or flat top-level keys alongside the required
  `actions:` list.
- Supported actions (`_run_action` in `libs/easydel/easydel/scripts/elarge.py`):
  - Inspection/export: `validate`, `print` / `show`, `dump_config` / `print_config` / `config`, `to_json` /
    `save_json` / `write_json`, `to_yaml` / `save_yaml` / `write_yaml`
  - Execution: `train`, `eval` (`tasks` required, plus `num_fewshot`, `output_path`, `print_results`), `serve` /
    `server`, and `serve_replica` (`control_port`, `auth_token`, optional `bind_host`)

Start with `--dry-run` (pure config parse/print — CPU trio is allowed):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run python -m easydel.scripts.elarge --config <config.yaml> --dry-run
```

## Configuration Surface Highlights

- **6-axis sharding (`types/infra.py`):** `sharding.axis_names` / `sharding.axis_dims` follow
  `("pp", "dp", "fsdp", "ep", "tp", "sp")`.
- **Auxiliary model roles (`types/root.py`):** `target_model` (speculative verifier), `teacher_model` (distillation),
  `reference_model` (preference/RL), plus `<role>_loader`, `<role>_sharding`, and `<role>_base_config` for `teacher`,
  `target`, `reference`, and `reward`. Role sharding overrides may adjust `partition_axis` but must keep the primary
  mesh (`axis_dims` / `axis_names`).
- **Grain local batch lanes (`types/training.py`):** `trainer.data_lane_local_batch` enables per-group host batch
  loading (see `.agents/skills/build-dataset-pipeline/SKILL.md`).

## Serve Mode & Multi-Replica Serving

- **Single-engine `serve` parameters:** `host`, `port`, `workers` (must be `1` for in-process serving), `log_level`,
  `ssl_keyfile`, `ssl_certfile`, `oai_like_processor`, `enable_function_calling`, `require_api_key`, `admin_key`,
  `enable_cors`, `cors_origins`. `reload: true` is not supported.
- **Do NOT use `serve.tool_parser_name`:** it is deprecated in `libs/easydel/easydel/scripts/elarge.py` and superseded by
  `esurge.tool_parser` (or `esurge.parsing.tool_parser`). Non-`"auto"` values warn and migrate when unset, or raise
  `SystemExit` if they conflict with `esurge.tool_parser`.
- **Multi-process pods (`world_size > 1`):** `serve` automatically enables `esurge.distributed.coordination="zmq"` with
  a deterministic config-derived auth token so rank 0 serves HTTP while worker ranks replay leader steps.
- **DP multi-replica serving (`serve.replicas`):** runs N independent `serve_replica` engines behind one CPU-pinned
  `RoutedEngine` API server (the parent marks FDs `CLOEXEC` and re-execs with `--dp-router-child` and
  `JAX_PLATFORMS=cpu` so replicas own the accelerator):
  - **Spawn mode (default):** derives `count` from the mesh `dp` axis (or explicit `count`), collapses `dp` to `1` per
    replica, carves TPU chips via `_plan_replica_tpu_envs`, and binds request planes on `base_control_port + i`
    (options: `count`, `base_control_port`, `ready_timeout_s`, `auth_token`, `workdir`, `platform`).
  - **Attach mode (`endpoints: ["host:port", ...]`):** fronts externally launched `serve_replica` instances (e.g.,
    one per pod host via `eray`).

## Config Routing

- Model or checkpoint conversion trouble: `.agents/skills/add-easydel-model/SKILL.md`,
  `.agents/skills/convert-checkpoint/SKILL.md`.
- Dataset, packing, or Grain data-lane trouble: `.agents/skills/build-dataset-pipeline/SKILL.md`.
- Quantized model builds: `.agents/skills/quantization-layout/SKILL.md`.
- Trainer algorithms or loss math: `.agents/skills/add-easydel-trainer/SKILL.md`.
- Compile-time HBM OOM: `.agents/skills/debug-training-oom/SKILL.md`.
- Serve mode, multi-replica routing, or parser output: `.agents/skills/debug-esurge/SKILL.md` and
  `.agents/skills/tool-reasoning-parser/SKILL.md` (pod orchestration: `.agents/skills/eray-orchestration/SKILL.md`).

## Verification

Pure-Python config and builder tests (no array computation; CPU trio is allowed):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/infra/elarge/test_builders_comprehensive.py \
                libs/easydel/tests/infra/elarge/test_builders_esurge_kwargs.py \
                libs/easydel/tests/infra/elarge/test_builders_quantization_qmm_kwargs.py \
                libs/easydel/tests/infra/elarge/test_auxiliary_model_overrides.py \
                libs/easydel/tests/infra/elarge/test_save_load_roundtrip.py
```

JAX computation tests (e.g., packing) and real `train`/`eval`/`serve` runs require an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Do not infer training or serving success from `--dry-run` — without an
accelerator, report execution as unverified on hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/trainers/test_sequence_packing_flag.py
```
