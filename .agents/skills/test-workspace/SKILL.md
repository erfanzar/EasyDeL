---
name: test-workspace
description: Select and run correct EasyDeL workspace checks. Use for affected-package test planning, accelerator (TPU/GPU) test env setup, the conftest guard that refuses computation tests on CPU (cpu_ok marker), import-layering checks, pre-commit behavior, hardware-bound test selection, Rust surgecore checks, and rejecting weak tests across libs/easydel, libs/spectrax, libs/ejkernel, libs/eformer, libs/eray, libs/surgecore, and libs/eyvan.
---

# Skill: Test The EasyDeL Workspace

Load this when choosing or running verification checks. For multi-step debugging, load
`.agents/skills/run-research/SKILL.md` first and use this skill as the verification layer.

## First Reads

- `WORKSPACE.md` and `.pre-commit-config.yaml`
- Touched package `pyproject.toml` (plus `libs/surgecore/Cargo.toml` for `surgecore`)
- Touched package docs under `libs/<package>/docs/` or `libs/<package>/README.md` (`eray` and `surgecore` have
  `README.md`, not `docs/`)

## Accelerator Environment (Computation Tests)

Every test executing JAX numerics — models, layers, attention, MoE, kernels, losses, optimizers, trainers, caches,
eSurge compute paths, spectrax modules/runtime/sharding/serialization, eformer escale/mpric/ops/optimizers, conversion
parity — MUST run on an accelerator (TPU, or GPU when GPU is the target hardware):

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <path>
```

For GPU use `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`. `ENABLE_DISTRIBUTED_INIT=0` prevents joining a distributed
runtime; `env -u XLA_FLAGS` drops stray host-device flags from the shell.

- **TPU `libtpu` lock**: run one accelerator process at a time. While a TPU job runs, only non-computation checks may
  use CPU; computation tests wait for the TPU.
- **No CPU substitution**: a CPU run of a computation test is never validation. Without an accelerator, report the
  change as **unverified on hardware**.
- **Inference conftest caveat**: `libs/easydel/tests/inference/conftest.py` sets `JAX_PLATFORMS=cpu` and
  `JAX_PLATFORM_NAME=cpu` when unset, which triggers the root CPU guard on inference compute tests — always pass the
  accelerator env vars explicitly when running `libs/easydel/tests/inference/`.

## CPU Trio (Non-Computation Tests Only)

Tests that perform zero array computation (`eray`, `surgecore`, `eyvan`, tool/reasoning parsers, config/CLI/YAML
parsing, data text transforms, loggers, paths, `@pytest.mark.cpu_ok`) may run on CPU:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

## Conftest Enforcement

`libs/{easydel,spectrax,eformer}/tests/conftest.py` and `libs/ejkernel/test/conftest.py` enforce accelerator execution
with a **two-layer guard** (there is **no** filename allow-list in any `conftest.py`):

1. **Collection-time import scan** (`_ARRAY_LIBRARY_IMPORT`): scans each test file for `jax`, `torch`, `spectrax`,
   `ejkernel`, `triton`, `tilelang`, or `flax` imports. Files without those imports pass automatically; files with them
   raise `pytest.UsageError` on CPU unless marked `@pytest.mark.cpu_ok`.
2. **Runtime compile guard** (`_cpu_compile_guard`): listens on `jax.monitoring` compile and cache-retrieval events and
   fails any unmarked test that compiles an XLA program on CPU (catching transitive imports).

Mark a test `@pytest.mark.cpu_ok` ONLY when it performs zero array computation (e.g. imports array libraries purely for
types or config metadata). Never add `cpu_ok` to get a computation test past the guard.

In `libs/ejkernel/test/conftest.py`, `pytest_ignore_collect` auto-skips unavailable GPU kernel trees (`_triton`,
`_cute`, `_cuda`, and `_tilelang` except `_tilelang/test_honesty.py`), and defaults `EJKERNEL_AUTOTUNE_POLICY=heuristics`.

## Workspace Gates

```bash
uv run lint-imports
uv lock --check
uv run pre-commit run --all-files
```

- `lint-imports` enforces `[tool.importlinter]` in `pyproject.toml`: only `easydel` may import `spectrax`, `ejkernel`,
  `eformer`, `eray`, `surgecore`, and `eyvan`; the 6 companion packages stay independent of `easydel` and of each
  other. Inside `libs/eyvan`, `(cd libs/eyvan && uv run --no-sync lint-imports)` checks eyvan's internal control-plane
  contracts.
- `uv lock --check` (also enforced in pre-push hooks and `.github/workflows/uv-lock-checkup.yaml`) verifies `uv.lock`
  matches all `libs/*/pyproject.toml` files.
- Pre-commit caveats:
  - Auto-fixing hooks report `Failed` when they rewrite files — inspect the diff, restage intended edits, and rerun.
    Never put `uv run` inside hook entries.
  - `check-easydel-imports` calls `jax.devices()` with `JAX_PLATFORMS=''` and can fail or block if another process
    holds the `libtpu` lock.
  - `cargo-fmt-surgecore` requires `cargo` on `PATH` when `libs/surgecore/**/*.rs` changes.
- CI (`.github/workflows/lint.yaml`) also runs report-only `basedpyright libs/easydel/easydel`.

## Package Test Targets

```bash
# EasyDeL (accelerator)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests -m "not slow"

# SpectraX (accelerator)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests

# eFormer (accelerator)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/eformer/tests

# eJKernel (accelerator; note test/, not tests/)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/ejkernel/test

# eRay (orchestration, no array compute — CPU trio)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eray/tests

# surgecore (Rust PyO3/maturin crate + CPU Python tests; PYO3_PYTHON = Python 3.11-3.13)
(cd libs/surgecore && cargo fmt --check)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo clippy --locked --all-targets -- -D warnings)
(cd libs/surgecore && PYO3_PYTHON=$(uv run which python) cargo test --locked)
(cd libs/surgecore && cargo test --locked --no-default-features)
uv build --package surgecore --wheel
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu uv run pytest libs/surgecore/tests

# eyvan (control plane, no array compute — fast CPU smoke; full suite needs dashboard build + PostgreSQL)
(cd libs/eyvan && uv run --no-sync lint-imports)
(cd libs/eyvan && uv run --package eyvan --extra dev pytest tests -m fast -q)
```

Run accelerator suites one at a time on TPU (`libtpu` lock). Pallas TPU tests under
`libs/ejkernel/test/kernels/_pallas/tpu` require TPU; GPU kernel trees require GPU
(`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`).

## Test Quality

Prefer tests that assert:

- public API outputs and exceptions
- numerical parity against independent references
- shape, dtype, sharding, cache layout, or checkpoint layout
- CLI parsed-argument behavior or produced artifacts
- scheduler/serving state transitions visible through public objects

Reject tests that only assert private helper calls, incidental log strings, constructors not raising, permanent skips,
or production logic compared with itself.
