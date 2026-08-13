---
name: generate-tests
description: Generate repository-consistent tests for EasyDeL workspace changes — models, trainers, operations/kernels, eSurge, data pipelines, infra sharding. Use when new behavior needs coverage or when asked to write tests; for choosing/running existing tests use test-workspace instead.
---

# Skill: Generate Tests

Specialization of `.claude/skills/run-research/SKILL.md`. The quality bar and run environment come from
`.claude/skills/test-workspace/SKILL.md` — load it first.

## Placement Map

| behavior under test            | location + exemplar to imitate                                                                                                                                                           |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model forward/anatomy          | `libs/easydel/tests/modules/spmd/test_<family>.py` (see `test_qwen3.py`); matrices in `tests/modules/conftest.py`                                                                        |
| HF conversion                  | `libs/easydel/tests/modules/test_conversion_roundtrip.py`                                                                                                                                |
| PP/distributed model behavior  | `libs/easydel/tests/modules/mpmd/`                                                                                                                                                       |
| trainer loss math              | `libs/easydel/tests/trainers/test_distillation_loss_math.py` (hand-computed constants, parametrized over `loss_type`) or `test_trl_dpo_loss_parity.py` (parity vs an external reference) |
| trainer config                 | `test_training_arguments_save_load_roundtrip.py` (every new field gets a non-default value), `test_preference_config_guards.py`                                                          |
| operation adapters             | `libs/easydel/tests/operations/` (config plumbing, dtype/layout equivalence)                                                                                                             |
| ejkernel kernels               | `libs/ejkernel/test/kernels/<backend>/` — parity vs the `_xla` reference                                                                                                                 |
| eSurge scheduler/cache/sampler | `libs/easydel/tests/inference/esurge/{core,runners}/`                                                                                                                                    |
| data pipeline                  | `libs/easydel/tests/data/` (packing determinism, source row limits)                                                                                                                      |
| infra/sharding                 | `libs/easydel/tests/infra/` (spec resolution on the fake 8-device mesh)                                                                                                                  |
| spectrax / eformer / eray      | `libs/<pkg>/tests/` mirroring the package layout                                                                                                                                         |

## Authoring Rules

1. Imitate the nearest existing test file's fixtures and structure before inventing anything. Module tests use
   `small_model_config`,
   `model_factory`, `model_tester`, parametrized
   `attention_mechanism`/`model_dtype` from `tests/modules/conftest.py`.
2. Tiny shapes: ~2 layers, 4 heads, 2 KV heads, 128 hidden, seq 128, batch 2, seed 42. Mark anything slower than a few
   seconds
   `@pytest.mark.slow`.
3. Numerical assertions compare against an **independent reference** (HF model, XLA kernel, hand-computed constants)
   with explicit atol/rtol — never production code against itself.
4. Sharding-sensitive tests run on the fake 8-device mesh; assert the resolved spec or the output equivalence across
   mesh shapes, not internals.
5. Test what the change claims: a bugfix gets a test that fails on the pre-fix code; verify by reverting mentally or
   with `git stash`.

## Rejected Patterns

Private helper-call assertions, incidental log strings, constructor-does-not-raise, permanent skips, tautological
comparisons, tests that require hardware but don't skip cleanly without it.

## Verification

Run the new tests under the CPU trio, then the surrounding directory to catch fixture interference:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <new-test-file> <its-directory>
```
