---
name: spectrax-pipeline-runtime
description: Implement, debug, or benchmark SpectraX MPMD and SPMD pipeline-parallel runtimes and schedules under libs/spectrax/spectrax/runtime. Use for spx.run, sxjit, sxgrad, sxvalue_and_grad, sxvalue_and_grad_and_apply, sxcall, MpmdPipelineExecutor, pipeline_step, sxstage_iter, sxstage_region, sxloop, treduce, pscan_compiler, DualPipeV/1F1B/ZeroBubble/Interleaved schedules, watchdog/tracing, or MPMD OOM/E0200 issues.
---

# Skill: Work On SpectraX Pipeline Runtime

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

Use this skill for work inside `libs/spectrax/spectrax/runtime/`, `libs/spectrax/spectrax/api.py`, or MPMD/SPMD
pipeline scheduling, per-rank compilation, activation/cotangent transport, stage markers, and memory/hang debugging.

## Package Boundary

`libs/spectrax` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `ejkernel`, `eformer`, `eray`, or `surgecore`. Keep examples generic; EasyDeL trainer/eSurge integrations
live in `libs/easydel`. Verify with `uv run lint-imports`.

## First Reads

- Public exports & `spx.run`: `libs/spectrax/spectrax/api.py`, `libs/spectrax/spectrax/runtime/__init__.py`, `libs/spectrax/spectrax/runtime/mpmd/__init__.py`
- Core MPMD runtime: `libs/spectrax/spectrax/runtime/mpmd/{runtime.py,schedule_units.py,grad_core.py,stage_callables.py,training_step.py,pipeline_executor.py,pscan_compiler.py,markers.py,treduce.py,transport_gate.py}`
- Diagnostics & flat-GPipe helpers: `libs/spectrax/spectrax/runtime/mpmd/{watchdog.py,tracing.py,per_rank.py,compiler.py}`
- Schedules & SPMD: `libs/spectrax/spectrax/runtime/schedules/{__init__.py,base.py,dualpipe.py,fusion.py,gpipe.py,interleaved.py,one_f_one_b.py,zero_bubble.py}`, `libs/spectrax/spectrax/runtime/spmd/api.py`
- Docs, benchmarks & memory notes: `libs/spectrax/docs/guides/pipeline.md`, `libs/spectrax/docs/performance.md`, `.agents/projects/mpmd-training-memory.md`, `libs/spectrax/examples/07_mpmd/`, `libs/spectrax/benchmarks/{bench_mpmd_jit_vs_spmd.py,bench_mpmd_jit_llama8b.py,train_pipeline.py}`

## Runtime Decision Tree & Entry Points

- **`spx.run(model, inputs=..., mesh=..., mode=...)` (`libs/spectrax/spectrax/api.py`):** unified entry point
  dispatching to SPMD (`jax.jit`) when `mesh.is_mpmd` is `False`, or MPMD (`sxcall`) when `mesh.is_mpmd` is `True`
  (requires `mpmd_axis` dim > 1 and a single positional microbatched input in MPMD mode).
- **Canonical scheduled MPMD training:** `sxjit(..., schedule=...)` +
  `sxvalue_and_grad(fn, argnums, *, has_aux=False, microbatch_weight_fn=None)` (or `sxgrad`).
  - `microbatch_weight_fn` computes host-side per-microbatch weights (e.g., valid token counts) so the scheduled
    estimator matches SPMD global token-weighted mean instead of mean-of-means; `has_aux=True` threads terminal-stage
    auxiliary metrics (must be produced after the last `sxstage_iter`).
- **Fused grad+apply (`libs/spectrax/spectrax/runtime/mpmd/training_step.py`):** `sxvalue_and_grad_and_apply` wraps
  scheduled `sxjit` with per-rank `APPLY` units. **v1 caveats:** `apply_fn` currently runs right after the final
  gradient fold (not yet overlapping rank-0 backward tail), and global gradient clipping (`clip_grad`) is not yet
  handled inside the schedule (pre-compute clip scale, use stage-local clip, or use `sxvalue_and_grad` + post-schedule
  apply).
- **Forward-only inference wavefront (`libs/spectrax/spectrax/runtime/mpmd/pipeline_executor.py`):**
  `MpmdPipelineExecutor` consumes `sxjit._mpmd_prepare` plans to pipeline same-shaped microbatches across stages
  without backward passes.
- **`PipelineSequential` compatibility wrapper:** `sxcall` (`libs/spectrax/spectrax/runtime/mpmd/runtime.py`) lowers
  `PipelineSequential` into marker-instrumented scheduled `sxjit` in train mode (uniform loss only; rejects `has_aux`).
- **SPMD-only pipeline (`libs/spectrax/spectrax/runtime/spmd/api.py`):** `pipeline_step` compiles a single `shard_map`
  HLO for homogeneous stages and **explicitly rejects** `MpMdMesh` and MPMD-tagged `SpxMesh`.
- **Inline stage & loop markers (`markers.py`, `treduce.py`, `pscan_compiler.py`):**
  `sxstage_iter` / `sxstage_region` delimit stages; `sxloop` / `sxenter_loop` / `sxexit_loop` and `treduce` /
  `treduce_i` emit `pscan_p` equations compiled by `pscan_compiler.py` (`build_pscan_plan`, `dispatch_pscan`).

## Schedules (9 Concrete Subclasses)

All subclasses inherit `Schedule` (`libs/spectrax/spectrax/runtime/schedules/base.py`) with `microbatches`,
`lazy_bwd_batching=False`, `terminal_backward_mode="eager"|"scheduled"`, and
`schedule_dispatcher="auto"|"deterministic_nonblocking"`:

- **Flat (V=1):** `GPipe`, `Std1F1B`, `Eager1F1B`, `ZeroBubbleH1`
- **Virtual-stage (V>1):** `InterleavedH1`, `InterleavedGPipe`, `Interleaved1F1BPlusOne`, `KimiK2`
- **Bidirectional (first-class):** `DualPipeV` (plus `dualpipev_tasks`)
- **Fusion helpers:** `FusedTask`, `fuse_1f1b_steady_state`, `fuse_zerobubble_bwd_pair`

## Critical Runtime Invariants (Do Not Break)

1. **Stage-local mesh rule:** an MPMD mesh `(pp=2, dp=4, tp=2)` becomes stage-local `(dp=4, tp=2)` inside each stage.
   `SpxMesh.is_mpmd` is `True` only when `mpmd_axis` is set **and** its dim > 1.
2. **MPMD tracing constraints:** single positional microbatched input; RNG streams are not staged; stage markers only
   take effect under the MPMD trace.
3. **PJRT run-ahead & terminal grad memory invariants (`.agents/projects/mpmd-training-memory.md`):**
   - PJRT allocates executable output buffers at **enqueue time**. Without pacing, host-speed dispatch piles up multiple
     full per-microbatch grad trees per rank and hits `E0100 RESOURCE_EXHAUSTED`. Preserve per-rank grad-unit run-ahead
     throttling (`_throttle_grad_unit_runahead` waiting on prior donated merge handles in `runtime.py` before enqueueing
     the next grad-producing unit — never plain FWD or BWD_I).
   - `_scale_grad` (`libs/spectrax/spectrax/runtime/mpmd/utils/tree.py`) MUST preserve leaf dtype (never silently
     promote bf16 -> f32 and double terminal grad HBM), and scheduled terminal scaling MUST use the donating in-place
     jit (`_scale_grad_donate` in `grad_core.py` / `_scale_terminal_grads` in `runtime.py`).
4. **Persistent compilation cache TPU E0200 hazard (`pscan_compiler.py`, `runtime.py`, `schedule_units.py`):**
   - Deserializing MPMD stage executables from JAX's persistent disk compilation cache causes deterministic TPU
     `E0200 RuntimeUnexpectedCoreHalt` (`EncodeRemoteSyncFlagAddress() no HLO mapping`) on the first non-terminal BWD.
   - All scheduled warm-compile lowering/compilation (including worker-thread `lowered.compile()`) MUST execute inside
     `_stage_persistent_cache_bypass()` (`pscan_compiler.py`), and every direct stage jit (e.g.,
     `_get_schedule_direct_fused_fwd_bwd_jit` in `schedule_units.py`) MUST be wrapped in `_scope_stage_persistent_cache`.
5. **Plan caching:** `_dispatch_schedule_faithful` caches schedule units/dependencies and preflight stats on the plan
   (`_schedule_units_cache`) so per-step dispatch does not rebuild the DAG or spam preflight logs.

## Debugging Map

- **Schedule shape or bubble ratio:** inspect `schedule.total_steps`, `peak_activations`, `bubble_ratio`, and
  `format_schedule_grid(schedule, n_stages)` (`tracing.py`) before touching runtime code.
- **Scheduled execution / dispatch / grad accumulation:** `libs/spectrax/spectrax/runtime/mpmd/runtime.py` (main
  dispatcher), `schedule_units.py` (DAG + fused units), `grad_core.py` (donated grad math), `stage_callables.py`
  (per-stage jits). Note: `per_rank.py` and `compiler.py` are **flat-GPipe / non-overlapped** dispatch paths only — do
  not look there for 1F1B/ZeroBubble/DualPipeV bugs.
- **Hangs or deadlocks:** wrap the run in `with dispatch_watchdog(stall_after_s=..., on_stall=...):` (`watchdog.py`)
  to capture `StallReport` with per-thread Python stacks (identifying blocked transport gates or `device_put`s), and
  inspect `transport_gate.py`.
- **Bubbles or transfer bottlenecks:** capture Perfetto Chrome traces with
  `with trace_schedule("/tmp/step.json", block_until_ready=...) as tracer:` (`tracing.py`) or `collect_task_times_ms()`
  (`runtime.py`).
- **Inference wavefront bugs:** inspect `pipeline_executor.py` and `test_mpmd_pipeline_executor.py`.
- **Stage marker / loop clustering:** inspect `markers.py`, `treduce.py`, `pscan_compiler.py`.

## Routing

- Stage-local sharding & `audit_placement`: `.agents/skills/spectrax-sharding/SKILL.md`
- `Module` / `assign_stage` / `spx.jit`: `.agents/skills/spectrax-modules/SKILL.md`
- Stage-local optimizers (`StageLocalGradientTransformation`): `.agents/skills/add-eformer-optimizer/SKILL.md`
- Training OOM attribution: `.agents/skills/debug-training-oom/SKILL.md`

## Verification

Computation tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/pipeline/ libs/spectrax/tests/runtime/
```

Pure-Python `@pytest.mark.cpu_ok` tests (runnable on CPU): `libs/spectrax/tests/runtime/test_prepared_plan_summary.py`,
and `cpu_ok`-marked cases in `libs/spectrax/tests/pipeline/{test_schedule_body.py,test_schedule_tracing.py}`.

For benchmark claims, run the relevant script under `libs/spectrax/benchmarks/` on target hardware and report hardware,
mesh, schedule, microbatch count, compile time, and steady-state step time separately.
