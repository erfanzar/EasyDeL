---
name: debug-esurge
description: Debug or benchmark EasyDeL eSurge inference. Use for scheduler, runner, executor, KV-cache shape, PP/MPMD/SPMD serving, DP page placement, speculative decoding, multi-replica routing, OpenAI API serving, xprof, or throughput regressions under libs/easydel/easydel/inference/esurge.
---

# Skill: Debug eSurge

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. For operational symptoms, also read the `eSurge Debugging` section in
`.agents/ops/OPS.md`.

## First Reads

Read the relevant surfaces before editing or diagnosing an eSurge failure:

- `WORKSPACE.md`, `.agents/ops/OPS.md`
- `libs/easydel/docs/esurge.rst`, `libs/easydel/docs/esurge_examples.rst`
- Engine & config: `libs/easydel/easydel/inference/esurge/{esurge_engine.py,config.py,page_table.py,benchmarking.py}`
- Engine subsystems:
  `libs/easydel/easydel/inference/esurge/engine/{bootstrap.py,admission.py,core_driver.py,loop.py,output_pipeline.py,idle.py,registry.py}`
- Runners & executors:
  - `libs/easydel/easydel/inference/esurge/runners/{model_runner.py,execution_manager.py,pipeline_execution_manager.py,pipeline_runtime.py,pipeline_plan.py,window_planner.py,sequence_buffer.py}`
  - `libs/easydel/easydel/inference/esurge/runners/executors/{model_executor.py,batch_preparer.py,sampler_executor.py,compile_orchestrator.py}`
- Speculative decoding:
  - `libs/easydel/easydel/inference/speculative.py` (`DrafterProtocol`, `Qwen3_5MTPDrafter`)
  - `libs/easydel/easydel/inference/esurge/runners/spec/{strategy.py,verify.py,rejection.py,device_budget.py}`
- Core KV, DP, and sampling:
  - `libs/easydel/easydel/inference/esurge/core/{manager.py,interface.py,dp_sharding.py,recurrent_snapshots.py,sampler.py}`
- Multi-replica routing & serving plane:
  - `libs/easydel/easydel/inference/esurge/distributed/{routed_engine.py,remote_engine.py,request_plane.py,zmq_coordinator.py}`
  - `libs/easydel/easydel/inference/esurge/serving/{app.py,config.py,cluster.py,host_replicas.py,ray_replicas.py,dashboard.py,keys.py}`
- API server: `libs/easydel/easydel/inference/esurge/server/api_server.py`

## Python <-> Rust (`surgecore`) Host Boundary

eSurge delegates its hot host path to the Rust PyO3 crate `libs/surgecore`:

| Native class | Rust source | EasyDeL Python consumer | Responsibility |
|---|---|---|---|
| `EngineCore` (`Plan`, `Events`) | `libs/surgecore/src/engine/` | `esurge/engine/core_driver.py` | Request table, per-DP queues, scheduler, verify budget, async spec rollback |
| `KvCore` | `libs/surgecore/src/kv/` | `esurge/core/manager.py` | Page pool, LRU free list, prefix map, transactional alloc/rollback |
| `SnapshotIndex` | `libs/surgecore/src/kv/snapshot.rs` | `esurge/core/recurrent_snapshots.py` | Recurrent-state snapshot LRU slots at prefix-cache page boundaries |
| `HostPayloadBuilder` | `libs/surgecore/src/payload.rs` | `esurge/runners/executors/batch_preparer.py` | GIL-released in-place fills of preallocated NumPy staging buffers |
| `Detokenizer` | `libs/surgecore/src/python/detok.rs` | `easydel/workers/esurge/pipeline/` + `esurge/engine/output_pipeline.py` | Batched incremental UTF-8 streaming detokenization |

- **`PageTable` lazy device mirror (`page_table.py`):** production reads `page_table_cpu` (versioned via
  `cpu_version`) inside `HostPayloadBuilder` and only refreshes derived device views when `cpu_version` advances; the
  device mirror array is allocated lazily only if accessed directly.
- For Rust edits, maturin/wheel rebuilds, stale `_native.abi3.so` mismatches, or golden parity tests, load
  `.agents/skills/develop-surgecore/SKILL.md`.

## Symptom Routes

- **Cache-shape, PP/MPMD, or prepare-cache failures:** inspect `ModelStepExecutor._kv_prepare_signature`
  (`runners/executors/model_executor.py`), `prepare_cache_key` use sites (`runners/pipeline_runtime.py`,
  `model_executor.py`), and `ExecutionManager._init_operations_cache_with_retry` (`runners/execution_manager.py`,
  `runners/pipeline_execution_manager.py`). For MPMD pipeline serving, inspect `pipeline_execution_manager.py`,
  `pipeline_runtime.py`, and `pipeline_plan.py` and load `.agents/skills/spectrax-pipeline-runtime/SKILL.md`. For
  cache layout or spec bugs, load `.agents/skills/debug-easydel-cache/SKILL.md`.
- **DP / KV-page placement failures:** inspect `esurge/core/dp_sharding.py` and
  `libs/easydel/tests/inference/esurge/core/test_dp_sharding_pages.py`.
- **Speculative decoding / unintended MTP activation:** inspect `DrafterSpeculation` (`runners/spec/strategy.py`),
  `DrafterProtocol` / `Qwen3_5MTPDrafter` (`easydel/inference/speculative.py`), `num_draft_tokens` /
  `num_draft_tokens_per_batch_size` (`esurge/config.py`), and `mtp_num_hidden_layers` on model config. Keep drafting
  disabled (`num_draft_tokens=0`) for plain text baselines unless speculative decoding is explicitly requested.
- **Multi-replica routing or multi-host serving failures:** inspect `distributed/routed_engine.py` (`RoutedEngine`,
  prefix-affinity + JSQ routing, `router_inflight.bin` mmap inflight accounting), `distributed/remote_engine.py`,
  `distributed/request_plane.py`, `distributed/zmq_coordinator.py`, and `esurge/serving/` (`app.py`, `config.py`,
  `host_replicas.py`, `ray_replicas.py`, `dashboard.py`).
- **OpenAI API tool or reasoning output mismatches:** load `.agents/skills/tool-reasoning-parser/SKILL.md` and inspect
  `engine/bootstrap.py`, `engine/admission.py`, `engine/output_pipeline.py`, and `server/api_server.py`.
- **Recompilation storms & throughput regressions:** every distinct `(num_tokens, padded_num_reqs)` bucket compiles a
  separate executable — pin buckets and compare per-bucket profiles grouped by total tokens, not aggregate tokens/sec.
  Build a focused throwaway benchmark harness using `.agents/skills/benchmark-changes/SKILL.md` and the shared helpers
  in `libs/easydel/easydel/inference/esurge/benchmarking.py` (`make_prompts`, `make_real_prompts`, `all_finished`,
  `zero_schedule_needs_update`). Set pre-existing `EASYDEL_SYNC_INPUTS_FOR_TIMING=1` only when isolating host prep-time
  accuracy (it adds a device sync round trip and reduces throughput).

## Focused Tests

Separate pure-host `cpu_ok` tests from accelerator computation tests:

- **Host-path `cpu_ok` unit suites (30+ files; CPU trio allowed):** e.g., `test_core_driver_admission.py`,
  `test_admission_scheduling_contracts.py`, `test_cache_spec_grouping.py`, `test_scheduler_runtime_config.py`,
  `test_engine_api_authoritative.py`, `core/test_manager_allocation_rollback.py`,
  `distributed/test_routed_engine_unit.py`, `serving/`, and `server/`:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/inference/esurge/test_core_driver_admission.py \
                libs/easydel/tests/inference/esurge/test_engine_api_authoritative.py \
                libs/easydel/tests/inference/esurge/distributed/test_routed_engine_unit.py \
                libs/easydel/tests/inference/esurge/serving/ \
                libs/easydel/tests/inference/esurge/server/
```

- **Accelerator computation tests (TPU shown; GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`):** all runner,
  executor, sampling, speculative verification, and DP-page tests execute JAX numerics and require an accelerator.
  Without an accelerator, report them as unverified on hardware. Unit tests do not replace hardware benchmark evidence
  for throughput or libtpu claims:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/inference/esurge/runners/test_model_executor_prepare_signature.py \
                libs/easydel/tests/inference/esurge/core/test_dp_sharding_pages.py

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/inference/esurge
```

If `libs/surgecore/` changed, also run the Rust and pytest checks in `.agents/skills/develop-surgecore/SKILL.md`.
