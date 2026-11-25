---
name: generate-tests
description: Generate repository-consistent tests for EasyDeL workspace changes — models, trainers, operations/kernels, eSurge, surgecore, data pipelines, infra sharding. Use when new behavior needs coverage or when asked to write tests; for choosing/running existing tests use test-workspace instead.
---

# Skill: Generate Tests

Specialization of `.agents/skills/run-research/SKILL.md`. Quality bar and run environments come from
`.agents/skills/test-workspace/SKILL.md` — load it first.

## Placement Map

| Behavior under test                 | Location + exemplar to imitate                                                                                                                                             |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Model forward / anatomy             | `libs/easydel/tests/modules/spmd/test_<family>.py` (see `test_qwen3.py`, using `CausalLMTester` / task `*Tester` in `tests/modules/test_utils/` + `small_model_config`)    |
| HF conversion                       | `libs/easydel/tests/modules/test_conversion_roundtrip.py`                                                                                                                  |
| PP / distributed model behavior     | `libs/easydel/tests/modules/mpmd/`                                                                                                                                         |
| Trainer loss math                   | `libs/easydel/tests/trainers/test_distillation_loss_math.py` (independent `optax` CE reference, parametrized over `attention_mask`/`loss_mask`), `test_grpo_discard_groups.py` / `test_grpo_discard_accumulation.py` (`loss_type`), or `test_trl_dpo_loss_parity.py` (external reference parity) |
| Trainer config                      | `libs/easydel/tests/trainers/test_training_arguments_save_load_roundtrip.py` (non-default value per new field; `cpu_ok`), `test_training_configuration_validation.py` (`cpu_ok`) |
| Trainer lanes / placement / accum   | `libs/easydel/tests/trainers/{test_batch_placement.py,test_grain_data_lanes.py,test_lane_batch_shard_groups.py,test_execution_accumulation.py}`                            |
| Operation adapters                  | `libs/easydel/tests/operations/` (config plumbing, dtype/layout equivalence)                                                                                               |
| eJKernel kernels                    | `libs/ejkernel/test/kernels/<backend>/` (parity vs `_xla` reference)                                                                                                       |
| eSurge serving / scheduler / cache  | `libs/easydel/tests/inference/esurge/{core,engine,runners,scheduler,server,serving,distributed}/` plus top-level `test_*.py`                                               |
| surgecore (Rust host engine + PyO3) | Rust unit tests in `libs/surgecore/src/` (`cargo test`) and golden/invariant/behavior Python tests in `libs/surgecore/tests/` (`pytestmark = pytest.mark.cpu_ok`)          |
| Data pipeline                       | `libs/easydel/tests/data/` (packing determinism, source row limits)                                                                                                        |
| Infra / sharding / checkpointing    | `libs/easydel/tests/infra/` (`test_base_config_scan_stage_config.py`, `test_gradient_checkpointing_targets.py`, `test_checkpoint_{save,load}_contract.py`)                 |
| spectrax / eformer / eray           | `libs/<pkg>/tests/` mirroring the package layout                                                                                                                           |

## Authoring Rules

1. **Imitate nearby exemplars.** For SPMD model tests, instantiate `CausalLMTester` (or the task-appropriate `*Tester`
   in `libs/easydel/tests/modules/test_utils/task_testers.py`) with the `small_model_config` fixture as in
   `libs/easydel/tests/modules/spmd/test_qwen3.py`.
2. **Keep shapes tiny.** ~2–4 layers, 4–8 heads, 2–4 KV heads, 128 hidden, seq 128, batch 2, seed 42. Mark multi-second
   tests `@pytest.mark.slow`.
3. **Independent references.** Compare numerical outputs against an independent reference (HF model, `_xla` kernel,
   `optax` reference, hand-computed constants) with explicit `atol`/`rtol` — never production code against itself.
4. **Real sharding checks.** Run sharding-sensitive tests on the accelerator's multi-device mesh; assert resolved specs
   or output equivalence across mesh shapes, not private internals.
5. **Reproduce the bug.** A bugfix test must fail on pre-fix code (verify mentally or via `git stash`).
6. **No new `os.environ` knobs.** Test explicit config, ContextVar, or setter APIs (see
   `test_pipeline_stage_layout_is_explicit_not_env` in `libs/easydel/tests/infra/test_base_config_scan_stage_config.py`).
7. **Respect the CPU guard.** Each JAX library's `conftest.py` enforces accelerator execution via collection-time
   `_ARRAY_LIBRARY_IMPORT` scanning plus the runtime `_cpu_compile_guard` (there is no filename allow-list). Mark a test
   `@pytest.mark.cpu_ok` ONLY when it does zero array computation. Distinguish optional capability skips (e.g.
   `jax.device_count() < 2` for MPMD, unavailable GPU backends in ejkernel) from computation tests that intentionally
   abort on CPU.

## Rejected Patterns

Private helper-call assertions, incidental log strings, constructor-does-not-raise checks, permanent skips, tautological
comparisons, marking computation tests `cpu_ok` to bypass the CPU guard, and multi-device/backend-specific tests that
crash instead of skipping cleanly when optional capabilities are absent.

## Verification

Run the new test file and its surrounding directory on the accelerator (GPU: `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`;
CPU trio only for `cpu_ok` non-computation, `eray`, or `surgecore` Python tests, plus `cargo test` for Rust):

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <new-test-file> <its-directory>
```
