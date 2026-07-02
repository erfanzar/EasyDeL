# eJKernel Profiling Reference

Use this when a kernel change makes a performance claim or when a benchmark
result needs attribution.

## Contents

- Local profiling surfaces
- Benchmark suite workflow
- Embedded JAX profiler hooks
- What to report
- Common mistakes

## Local Profiling Surfaces

Start with the local tools before writing new harnesses:

- `libs/ejkernel/benchmarks/benchmark_suite.py` runs the registry-driven matrix
  and writes JSON plus Markdown reports.
- `libs/ejkernel/benchmarks/_op_benchmark_registry.py` owns `SPECS`,
  `run_benchmark`, platform wrapping, and `EJKERNEL_BENCH_CONFIG_LIMIT`.
- `libs/ejkernel/ejkernel/benchmarks.py` owns `Benchmark`, warmup/iteration
  timing, JSON save, and plots.
- `libs/ejkernel/ejkernel/ops/execution/profiler.py` owns the `Profiler`
  class for JAX trace parsing and function/event timing.
- `libs/ejkernel/ejkernel/loggings.py` exposes `ignite_profiler`,
  `extinguish_profiler`, and `create_step_profiler`.

For EasyDeL integration paths:

- Training configs expose `profiler_path`, `profiler_host_tracer_level`, and
  `profiler_python_tracer_level` in
  `libs/easydel/easydel/trainers/training_configurations.py`.
- The base trainer starts JAX profiling after step 1 and stops it in
  `libs/easydel/easydel/trainers/base_trainer.py`.
- eSurge benchmark profiling uses `scripts/bench_esurge.py --xprof-dir`,
  `--xprof-trial`, `--xprof-host-level`, and `--xprof-python-level`.

## Benchmark Suite Workflow

For TPU Pallas work, run on TPU. Compare `xla` and `pallas` on the same target
device; CPU/XLA smoke timing is not evidence for TPU behavior.

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
EJKERNEL_BENCH_OPS=<kernel> \
EJKERNEL_BENCH_PLATFORMS=xla,pallas \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=1 \
EJKERNEL_BENCH_ITERS=1 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Useful suite env vars verified in `benchmark_suite.py`:

- `EJKERNEL_BENCH_OPS`
- `EJKERNEL_BENCH_SKIP_OPS`
- `EJKERNEL_BENCH_PLATFORMS`
- `EJKERNEL_BENCH_OUTPUT_DIR`
- `EJKERNEL_BENCH_CONFIG_LIMIT`
- `EJKERNEL_BENCH_WARMUP`
- `EJKERNEL_BENCH_ITERS`

Useful registry env var verified in `_op_benchmark_registry.py`:

- `EJKERNEL_BENCH_IGNORE_PLATFORMS`

The platform values come from `Platform` in
`libs/ejkernel/ejkernel/kernels/_registry.py`: `xla`, `pallas`, `triton`,
`cuda`, `cute`, and `tilelang`.

Single-op wrappers call `run_benchmark(<op>)`. Example:

```bash
uv run python libs/ejkernel/benchmarks/benchmark_gated_delta_rule.py
```

The single-op wrappers use fixed `warmup=5` and `iterations=30` through
`run_benchmark`; use `benchmark_suite.py` when you need env-controlled short
runs or machine-readable output.

Host-side preflight is allowed only to check that the benchmark harness,
registry entry, or XLA reference path still imports and runs:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
EJKERNEL_BENCH_OPS=fused_cross_entropy \
EJKERNEL_BENCH_PLATFORMS=xla \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=1 \
EJKERNEL_BENCH_ITERS=1 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_cpu_preflight \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Label this as preflight only. Do not use it for TPU correctness, lowering, or
performance claims.

## Embedded JAX Profiler Hooks

Use `ejkernel.loggings.create_step_profiler(...)` when profiling a custom loop
that already has a step counter. It starts at `start_step - 1`, stops at the
configured window end, and calls `barrier_sync()` after stopping.

Use `ignite_profiler(profile_path, enable_perfetto)` and
`extinguish_profiler(enable_perfetto)` for a manual region. Keep the region
steady-state: run at least one compile/warmup call before profiling.

Use EasyDeL trainer `profiler_path` for training-loop profiles. The trainer
starts after step 1, so the profile is steady-state rather than first-compile
time.

Use `scripts/bench_esurge.py --xprof-dir <dir>` only for eSurge serving
throughput/debugging, not for isolated ejkernel microbenchmarks.

## What To Report

For every headline result include:

- hardware type, backend, and device count
- exact operation, platform, and selected implementation
- shape/dtype/config grid
- warmup and timed iteration counts
- output JSON/Markdown path
- steady-state timing from `Benchmark`
- compile-including timing if measured separately
- direct baseline vs candidate numbers
- failures, skipped platforms, or narrowed scope

If only the suite timing ran, say "steady-state after warmup"; do not call it
compile-including timing.

## Common Mistakes

- Treating CPU/XLA preflight as TPU Pallas validation.
- Letting `EJKERNEL_AUTOTUNE_POLICY=autotune` run inside a deterministic unit
  test.
- Reporting a single fastest run without the JSON/Markdown benchmark artifact.
- Changing shape, dtype, platform set, warmup, or iteration count between
  baseline and candidate.
- Treating a side benchmark as proof for an EasyDeL training or eSurge path
  without exercising that path.
