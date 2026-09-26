---
name: test-workspace
description: Select and run correct EasyDeL workspace checks. Use for affected-package test planning, accelerator (TPU/GPU) test env setup, the conftest guard that refuses computation tests on CPU (cpu_ok marker), import-layering checks, pre-commit behavior, hardware-bound test selection, and rejecting weak tests across libs/easydel, libs/spectrax, libs/ejkernel, libs/eformer, and libs/eray.
---

# Skill: Test The EasyDeL Workspace

Load this when the task is choosing or running tests rather than designing a new feature. For multi-step debugging, load
`.xerxes/skills/run-research/SKILL.md`
first and use this as the verification layer.

## First Reads

- `WORKSPACE.md`
- `.pre-commit-config.yaml`
- touched package `pyproject.toml`
- touched package docs under `libs/<package>/docs/`

## Accelerator Environment (computation tests)

Every test that executes JAX numerics — models, layers, attention, MoE, kernels, losses, optimizers, trainers, caches,
eSurge, spectrax transforms/runtime/nn/core, eformer escale/mpric/ops/optimizers, conversion parity — runs on an
accelerator: TPU, or GPU where that is the target hardware.

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <path>
```

For GPU use `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`. `ENABLE_DISTRIBUTED_INIT=0` prevents local tests from
joining a real distributed runtime; unsetting `XLA_FLAGS` drops a stray `--xla_force_host_platform_device_count` from
the shell.

TPU hosts hold a single libtpu process lock: run one accelerator test process at a time. While a TPU job runs, only
non-computation probes may use `JAX_PLATFORMS=cpu`; computation tests wait for the TPU.

A CPU run of a computation test is never validation. With no accelerator available, do not run it on CPU as a stand-in —
report the change as **unverified on hardware**.

## CPU Trio (non-computation tests only)

Tests that do no array computation — eray, tool/reasoning parsers, config/CLI/YAML parsing, data text transforms,
loggers, paths, docs — may use the CPU trio:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

## Conftest Enforcement

Each library's tests `conftest.py` refuses to run computation tests on a CPU backend. Pure-Python test files are
allow-listed in the conftest or marked `@pytest.mark.cpu_ok`. Mark a test `cpu_ok` only when it does no array
computation; never add the marker (or an allow-list entry) to get a computation test past the guard.

## Workspace Gates

```bash
uv run lint-imports
uv run pre-commit run --all-files
```

`lint-imports` enforces package layering from `WORKSPACE.md`: only
`libs/easydel` may import the foundation packages.

Pre-commit hooks may auto-fix and report `Failed` because files changed. When that happens, inspect the diff, restage
intended edits, and rerun. Do not put
`uv run` inside hook entries.

## Package Test Targets

```bash
# EasyDeL
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests -m "not slow"

# SpectraX
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests

# eFormer
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/eformer/tests

# eJKernel (note: test/, not tests/)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/ejkernel/test

# eRay: orchestration, no array compute — CPU trio
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eray/tests
```

Run the accelerator targets one at a time (libtpu lock). eJKernel Pallas TPU tests under
`libs/ejkernel/test/kernels/_pallas/tpu` need a TPU backend; GPU kernel trees need a GPU with
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`.

## Test Quality

Prefer tests that assert:

- public API outputs and exceptions
- numerical parity against independent references
- shape, dtype, sharding, cache layout, or checkpoint layout
- CLI parsed-argument behavior or produced artifacts
- scheduler/serving state transitions visible through public objects

Reject tests that only assert private helper calls, incidental log strings, constructors not raising, permanent skips,
or production logic compared with itself.
