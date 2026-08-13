---
name: test-workspace
description: Select and run correct EasyDeL workspace checks. Use for affected-package test planning, CPU JAX env setup, import-layering checks, pre-commit behavior, hardware-bound test selection, and rejecting weak tests across libs/easydel, libs/spectrax, libs/ejkernel, and libs/eformer.
---

# Skill: Test The EasyDeL Workspace

Load this when the task is choosing or running tests rather than designing a new feature. For multi-step debugging, load
`.claude/skills/run-research/SKILL.md`
first and use this as the verification layer.

## First Reads

- `WORKSPACE.md`
- `.pre-commit-config.yaml`
- touched package `pyproject.toml`
- touched package docs under `libs/<package>/docs/`

## CPU JAX Environment

For CPU JAX tests, use the full trio:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

The fake host-device count is load-bearing for multi-device sharding tests.
`ENABLE_DISTRIBUTED_INIT=0` prevents local tests from joining a real distributed runtime.

CPU checks are not substitutes for TPU kernel correctness, Mosaic lowering, eSurge TPU runtime behavior, or benchmark
claims.

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
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests -m "not slow"

# SpectraX
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests

# eFormer
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eformer/tests

# eJKernel XLA/host-side
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/ejkernel/test/kernels/_xla
```

eJKernel Pallas TPU tests under `libs/ejkernel/test/kernels/_pallas/tpu` need a TPU backend and an available libtpu
process lock.

## Test Quality

Prefer tests that assert:

- public API outputs and exceptions
- numerical parity against independent references
- shape, dtype, sharding, cache layout, or checkpoint layout
- CLI parsed-argument behavior or produced artifacts
- scheduler/serving state transitions visible through public objects

Reject tests that only assert private helper calls, incidental log strings, constructors not raising, permanent skips,
or production logic compared with itself.
