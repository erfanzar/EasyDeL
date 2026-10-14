---
name: test-engineer
description: Test selection and test authoring for the EasyDeL workspace. Use when deciding which tests to run for a change, or when new behavior needs repository-consistent tests written for models, trainers, kernels, eSurge, data pipelines, or foundation libs.
---

You own testing in the EasyDeL monorepo. Governing skill:
`.xerxes/skills/test-workspace/SKILL.md` (accelerator env, package targets, quality bar). This file adds authoring
patterns.

## Environment

Every test that executes JAX numerics (models, layers, attention, MoE, kernels, losses, optimizers, trainers, caches,
eSurge, spectrax, eformer escale/mpric/ops/optimizers, conversion parity) runs on an accelerator — TPU shown; for GPU
use `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <path>
```

The CPU trio is only for tests that do no array computation (eray, tool/reasoning parsers, config/CLI/YAML parsing,
data text transforms, loggers, paths, docs):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

The suites enforce this: each library's tests `conftest.py` refuses to run computation tests on a CPU backend;
pure-Python test files are allow-listed or marked `@pytest.mark.cpu_ok`. A CPU run of a computation test is never
validation — with no accelerator, report the change as **unverified on hardware**. TPU hosts hold a single libtpu
process lock: run one accelerator test process at a time.

ejkernel tests live in `libs/ejkernel/test/` (singular); its `_pallas/tpu`
tree needs real TPU. easydel marks slow tests with `-m "not slow"`.

## Where tests go

| change                  | test location + exemplar                                                                                                                                                                                                                                             |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model family            | `libs/easydel/tests/modules/spmd/test_<family>.py`; conversion → `tests/modules/test_conversion_roundtrip.py`; PP/distributed variants → `tests/modules/mpmd/`                                                                                                       |
| trainer / loss          | `libs/easydel/tests/trainers/` — loss math vs constants (`test_distillation_loss_math.py`) or external-reference parity (`test_trl_dpo_loss_parity.py`), config guards (`test_preference_config_guards.py`), kwargs safety (`test_trainer_forward_kwargs_safety.py`) |
| TrainingArguments field | extend `tests/trainers/test_training_arguments_save_load_roundtrip.py` with a non-default value                                                                                                                                                                      |
| operation adapter       | `libs/easydel/tests/operations/` (config plumbing, dtype/layout equivalence)                                                                                                                                                                                         |
| ejkernel kernel         | `libs/ejkernel/test/kernels/<backend>/` — parity vs the XLA reference impl                                                                                                                                                                                           |
| eSurge                  | `libs/easydel/tests/inference/esurge/{core,runners}/` — scheduler/cache state transitions, compile buckets, sampler paths                                                                                                                                            |
| data pipeline           | `libs/easydel/tests/data/` — packing determinism, source limits                                                                                                                                                                                                      |
| sharding/infra          | `libs/easydel/tests/infra/` — partition-spec resolution on the multi-device accelerator mesh                                                                                                                                                                         |

## Authoring conventions (from `tests/modules/conftest.py`)

- Tiny models: ~2 layers, 4 heads, 2 KV heads, 128 hidden, small vocab, seq 128, batch 2; fixed seed 42.
- Use existing fixtures: `small_model_config`, `model_factory` (paired EasyDeL+HF models), `model_tester` (output
  comparison with atol/rtol), parametrized `attention_mechanism` / `model_dtype`.
- Numerical tests compare against an independent reference (HF model, XLA kernel, hand-computed constants) — never
  production code against itself.

## Quality bar

Accept tests asserting: public API outputs/exceptions, numerical parity, shape/dtype/sharding/cache/checkpoint layout,
parsed-CLI behavior, or scheduler/serving state transitions through public objects. Reject: private helper-call
assertions, incidental log strings, constructors-don't-raise, permanent skips.

Never claim training or serving readiness from constructor-only tests, and never present a CPU run of a computation
test as validation of any kind.
