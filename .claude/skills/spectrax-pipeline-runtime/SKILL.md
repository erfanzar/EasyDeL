---
name: spectrax-pipeline-runtime
description: Implement, debug, or benchmark SpectraX module-system and pipeline-runtime behavior. Use for pipeline_step, sxcall, sxjit, sxstage_iter, sxstage_region, MPMD/SPMD pipeline scheduling, per-rank execution, stage-local mesh rules, or libs/spectrax runtime tests and benchmarks.
---

# Skill: Work On SpectraX Pipeline Runtime

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. This skill adds SpectraX-specific package
ownership and runtime routing.

## First Reads

- `WORKSPACE.md`
- `libs/spectrax/pyproject.toml`
- `libs/spectrax/docs/design.md`
- `libs/spectrax/docs/performance.md`
- `libs/spectrax/docs/guides/pipeline.md`
- `libs/spectrax/docs/guides/sharding.md`
- `libs/spectrax/spectrax/runtime/mpmd/`
- `libs/spectrax/tests/pipeline/`
- `libs/spectrax/tests/runtime/test_mpmd_pipeline_executor.py`

Examples and benchmarks:

- `libs/spectrax/examples/07_mpmd/`
- `libs/spectrax/benchmarks/bench_mpmd_jit_vs_spmd.py`
- `libs/spectrax/benchmarks/bench_mpmd_jit_llama8b.py`
- `libs/spectrax/benchmarks/train_pipeline.py`

## Package Boundary

`libs/spectrax` is a foundation package. It must not import `easydel`,
`ejkernel`, or `eformer`. Keep examples generic or inside the EasyDeL package
when EasyDeL integration is required.

## Runtime Decision Tree

- Use `pipeline_step` for SPMD-only staged computation. It rejects MPMD-tagged
  meshes.
- Use `sxcall` or `sxjit` for true MPMD execution with per-rank executables.
- Use `sxstage_iter` and `sxstage_region` for inline stage markers.
- Choose a schedule from the existing runtime: `GPipe`, `Std1F1B`,
  `ZeroBubbleH1`, `InterleavedH1`, or the documented `DualPipeV` example path
  when the code path supports it.

Stage-local mesh rule: an MPMD full mesh such as `(pp=2, dp=4, tp=2)` becomes a
stage-local mesh such as `(dp=4, tp=2)` inside a stage.

## What To Inspect

- Schedule shape or bubble issue: inspect schedule `total_steps`,
  `peak_activations`, and `bubble_ratio` before changing executor code.
- Per-rank mismatch: inspect
  `libs/spectrax/spectrax/runtime/mpmd/per_rank.py` and compiler lowering.
- Transport or synchronization issue: inspect
  `libs/spectrax/spectrax/runtime/mpmd/transport_gate.py`.
- Marker issue: inspect
  `libs/spectrax/spectrax/runtime/mpmd/markers.py` and the stage-region tests.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests/pipeline

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/spectrax/tests/runtime/test_mpmd_pipeline_executor.py
```

For benchmark claims, run the relevant script under `libs/spectrax/benchmarks/`
and report hardware, mesh, schedule, microbatch count, compile-including timing
when relevant, and steady-state timing.
