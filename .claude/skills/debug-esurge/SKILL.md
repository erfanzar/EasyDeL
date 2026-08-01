---
name: debug-esurge
description: Debug or benchmark EasyDeL eSurge inference. Use for scheduler, runner, executor, KV-cache shape, PP/SPMD, DP page placement, no-MTP text serving, OpenAI API serving, benchmark, xprof, or throughput regressions under libs/easydel/easydel/inference/esurge.
---

# Skill: Debug eSurge

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For operational symptoms, also read the
`eSurge Debugging` section in `.claude/ops/OPS.md`.

## First Reads

Read these before editing or explaining an eSurge failure:

- `WORKSPACE.md`
- `.claude/ops/OPS.md`
- `libs/easydel/docs/esurge.rst`
- `libs/easydel/docs/esurge_examples.rst`
- `scripts/bench_esurge.py`
- `libs/easydel/easydel/inference/esurge/`

For runner/executor failures, open:

- `libs/easydel/easydel/inference/esurge/runners/model_runner.py`
- `libs/easydel/easydel/inference/esurge/runners/execution_manager.py`
- `libs/easydel/easydel/inference/esurge/runners/executors/model_executor.py`
- `libs/easydel/easydel/inference/esurge/scheduler/`
- `libs/easydel/easydel/inference/esurge/core/`

## Symptom Routes

- Cache-shape, PP, or prepare-cache failures: inspect
  `ModelStepExecutor._kv_prepare_signature`, `prepare_cache_key` use sites, and
  `ExecutionManager._init_operations_cache_with_retry`.
- DP/KV-page placement failures: inspect
  `libs/easydel/easydel/inference/esurge/core/dp_sharding.py` and
  `libs/easydel/tests/inference/esurge/core/test_dp_sharding_pages.py`.
- Text-only serving accidentally using speculative decode: search for
  `num_speculative_tokens`, `SpeculativeMTPDriver`, and
  `mtp_num_hidden_layers`. Keep MTP disabled unless the user explicitly asks
  for speculative decoding.
- OpenAI API tool or reasoning output mismatches: load
  `.claude/skills/tool-reasoning-parser/SKILL.md` and inspect
  `libs/easydel/easydel/inference/esurge/server/api_server.py`.
- Throughput regressions: compare `scripts/bench_esurge.py` JSON
  `profile_by_total_tokens` buckets, not just aggregate tokens/sec.

## Benchmark Harness

`scripts/bench_esurge.py` is the repo harness for eSurge. It defaults to TPU
and exits if JAX does not report a TPU backend. Useful flags include:

- `--model`
- `--prompt-len`
- `--output-len`
- `--num-prompts`
- `--max-model-len`
- `--max-num-seqs`
- `--max-num-batched-tokens`
- `--hbm-utilization`
- `--page-size`
- `--sharding-axis-dims` in `pp,dp,fsdp,ep,tp,sp` order
- `--warmups`
- `--trials`
- `--json-out`
- `--no-async`
- `--no-overlap`
- `--use-aot-forward`
- `--verbose-runner`
- `--xprof-dir`
- `--xprof-trial`
- `--xprof-host-level`
- `--xprof-python-level`

The harness constructs a no-MTP workload with `num_speculative_tokens=0` and
writes `"no_mtp": true` to the JSON output. Do not use it as a speculative
decoding benchmark.

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
  uv run python scripts/bench_esurge.py \
    --model <model-or-checkpoint> \
    --num-prompts 32 --prompt-len 1024 --output-len 256 \
    --warmups 1 --trials 1 \
    --json-out /tmp/easydel_esurge_bench.json
```

## Focused Tests

Host-side tests use the CPU environment below; these do not replace TPU runtime
validation for performance or libtpu behavior.

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge/runners/test_model_executor_prepare_signature.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge/test_engine_api_authoritative.py
```

Set `EASURGE_SYNC_INPUTS_FOR_TIMING=1` only when measuring prep-time accuracy;
it adds a device round trip and can reduce throughput.
