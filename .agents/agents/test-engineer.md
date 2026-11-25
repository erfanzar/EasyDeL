---
name: test-engineer
description: Test selection and test authoring for the EasyDeL workspace. Use when deciding which tests to run for a change, or when new behavior needs repository-consistent tests written for models, trainers, kernels, eSurge, data pipelines, Rust surgecore, or foundation libs.
---

You own testing in the EasyDeL monorepo. Governing skills: `.agents/skills/test-workspace/SKILL.md` (accelerator env,
package targets, quality bar) and `.agents/skills/generate-tests/SKILL.md` (authoring new coverage). This file adds
repo-specific authoring patterns and layout maps.

## Environment

Every test that executes JAX numerics (models, layers, attention, MoE, kernels, losses, optimizers, trainers, caches,
eSurge, spectrax, eformer escale/mpric/ops/optimizers, conversion parity) runs on an accelerator — TPU shown; for GPU
use `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <path>
```

The CPU trio is ONLY for pure-Python tests that do no array computation (`eray`, `surgecore` Python tests,
tool/reasoning parsers, config/CLI/YAML parsing, data text transforms, loggers, paths, docs, `@pytest.mark.cpu_ok`):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

For `libs/surgecore` (Rust PyO3/maturin crate), run Rust checks in `libs/surgecore` plus its Python test suite:

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --no-default-features          # or PYO3_PYTHON=<py3.11-3.13> cargo test --locked
uv run pytest libs/surgecore/tests
```

**How the CPU guard works**: 4 of the 6 workspace libs (`easydel`, `spectrax`, `eformer`, `ejkernel`) enforce the
accelerator policy in their `conftest.py` via regex array-library import scanning (`_ARRAY_LIBRARY_IMPORT`),
`@pytest.mark.cpu_ok`, and runtime `jax.monitoring` compile trapping (`_cpu_compile_guard`) — there is NO filename
allow-list. `eray` (pure orchestration) and `surgecore` (Rust/Python host core, no `conftest.py`) run on CPU. A CPU
run of a computation test is never validation — with no accelerator, report the change as **unverified on hardware**.
TPU hosts hold a single libtpu process lock: run one accelerator test process at a time.

`ejkernel` tests live in `libs/ejkernel/test/` (singular). `libs/ejkernel/test/conftest.py` automatically ignores
`_triton`, `_cute`, `_cuda`, and `_tilelang` (while keeping `_tilelang/test_honesty.py` AST checks) when GPU toolkits
are absent; `_pallas/tpu` requires a real TPU. `easydel` marks slow tests with `-m "not slow"`.

## Where tests go

| change                  | test location + exemplar                                                                                                                                                                                                                                             |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model family            | `libs/easydel/tests/modules/spmd/test_<family>.py`; conversion → `libs/easydel/tests/modules/test_conversion_roundtrip.py`; PP/distributed variants → `libs/easydel/tests/modules/mpmd/`                                                                            |
| trainer / loss          | `libs/easydel/tests/trainers/` — loss math vs constants (`test_distillation_loss_math.py`) or external-reference parity (`test_trl_dpo_loss_parity.py`), config guards (`test_preference_config_guards.py`), kwargs safety (`test_trainer_forward_kwargs_safety.py`) |
| TrainingArguments field | extend `libs/easydel/tests/trainers/test_training_arguments_save_load_roundtrip.py` with a non-default value                                                                                                                                                         |
| operation adapter       | `libs/easydel/tests/operations/` (config plumbing, dtype/layout equivalence)                                                                                                                                                                                         |
| eSurge                  | `libs/easydel/tests/inference/esurge/{core,engine,runners,server,distributed,runtime_pass}/` and top-level `test_*.py` — state transitions, compile buckets, sampler paths, routing                                                                                  |
| data pipeline & lanes   | `libs/easydel/tests/data/` (sources, packing, caching, determinism) and `libs/easydel/tests/trainers/` (`test_grain_data_lanes.py`, `test_lane_batch_shard_groups.py`, `test_batch_placement.py`, `test_sequence_packing_flag.py`)                                   |
| sharding/infra          | `libs/easydel/tests/infra/` — partition-spec resolution on the multi-device accelerator mesh                                                                                                                                                                         |
| ejkernel kernel         | `libs/ejkernel/test/kernels/_{pallas/{tpu,gpu},triton,cuda,cute,tilelang,xla}/` and top-level `libs/ejkernel/test/kernels/test_*_parity.py` — parity vs XLA reference                                                                                               |
| spectrax                | `libs/spectrax/tests/{core,nn,transforms,sharding,runtime,pipeline,quantization,serialization}/`                                                                                                                                                                     |
| eformer                 | `libs/eformer/tests/{escale,mpric,ops,optimizers,serialization,pytree,paths}/`                                                                                                                                                                                       |
| eray                    | `libs/eray/tests/` (CPU trio — Ray/pod orchestration, no array compute)                                                                                                                                                                                              |
| surgecore               | `libs/surgecore/tests/` (Python integration/golden/parity tests) + Rust unit tests (`cargo test --locked`)                                                                                                                                                           |

## Authoring conventions (from `libs/easydel/tests/modules/conftest.py`)

- `small_model_config`: 2 layers, 4 attention heads, 2 KV heads, `hidden_size=128`, `vocab_size=32000` (not tiny),
  `sequence_length=128`, `batch_size=2`, fixed seed `42`. Note `attn_softmax_dtype=bfloat16` by default in this
  fixture — override explicitly to `jnp.float32` in tests asserting fp32 softmax precision.
- Use existing fixtures: `small_model_config`, `model_factory` (paired EasyDeL+HF models), `model_tester` (output
  comparison with `atol`/`rtol`), parametrized `attention_mechanism` / `model_dtype`.
- Numerical tests compare against an independent reference (HF model, XLA kernel, hand-computed constants) — never
  production code against itself.

## Quality bar

Accept tests asserting: public API outputs/exceptions, numerical parity, shape/dtype/sharding/cache/checkpoint layout,
parsed-CLI behavior, or scheduler/serving state transitions through public objects. Reject: private helper-call
assertions, incidental log strings, constructors-don't-raise, permanent skips.

Never claim training or serving readiness from constructor-only tests, and never present a CPU run of a computation
test as validation of any kind.
