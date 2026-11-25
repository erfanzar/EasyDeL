# eJKernel Profiling Reference

Canonical profiling and benchmark reference for `.agents/skills/optimize-ejkernel-kernel/SKILL.md` and
`.agents/skills/add-ejkernel-kernel/SKILL.md`. Use this whenever a kernel change makes a performance claim or a
benchmark result needs attribution.

## Local Profiling & Benchmarking Surfaces

Start with verified in-repo tools before writing throwaway harnesses:

- `libs/ejkernel/benchmarks/benchmark_suite.py` — registry-driven multi-op, multi-platform matrix writing JSON and
  Markdown reports.
- `libs/ejkernel/benchmarks/_op_benchmark_registry.py` — `SPECS`, `run_benchmark`, platform dispatch, and
  `EJKERNEL_BENCH_CONFIG_LIMIT` / `EJKERNEL_BENCH_IGNORE_PLATFORMS`.
- `libs/ejkernel/ejkernel/benchmarks.py` — `Benchmark` class (warmup, steady-state timing, JSON save, plots).
- `libs/ejkernel/ejkernel/ops/execution/profiler.py` — `Profiler` for JAX trace parsing and function/event timing.
- `libs/ejkernel/ejkernel/loggings.py` — `ignite_profiler`, `extinguish_profiler`, and `create_step_profiler`.
- `libs/ejkernel/ejkernel/ops/tuned/` — shipped `tuned_kernels.db` table and `python -m ejkernel.ops.tuned` CLI
  (`stats`, `dump`, `sweep`, `merge`).

EasyDeL end-to-end profiling surfaces:

- **Trainers (`TrainingArguments` in `libs/easydel/easydel/trainers/training_configurations.py` and
  `libs/easydel/easydel/trainers/base_trainer.py`):** `profiler_path`, `profiler_stop_step`,
  `profiler_include_compile` (`False` by default, so profiling starts after step 1; set `True` to include initial
  compile/HLO), `profiler_log_all_workers`, `profiler_host_tracer_level`, `profiler_python_tracer_level`,
  `profiler_duration_ms`, and `profiler_enable_hlo_proto`.
- **eSurge serving:** see `.agents/skills/debug-esurge/SKILL.md` and `.agents/skills/benchmark-changes/SKILL.md` (use
  `jax.profiler.start_trace` directly or `RunnerPerfTracker` in `libs/easydel/easydel/inference/esurge/runners/perf.py`).
  For speculative decoding, `libs/easydel/tests/inference/esurge/bench_specdecode.py` enables xprof via env vars
  `EASYDEL_BENCH_XPROF_DIR`, `EASYDEL_BENCH_XPROF_LABELS`, `EASYDEL_BENCH_XPROF_TRIAL`,
  `EASYDEL_BENCH_XPROF_HOST_LEVEL`, and `EASYDEL_BENCH_XPROF_PYTHON_LEVEL` (not CLI flags).

## Benchmark Suite Workflow

Run on the target accelerator (TPU or GPU). Never run `benchmark_suite.py` on CPU to substantiate correctness, lowering,
or performance claims.

```bash
# TPU (XLA vs Pallas)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_BENCH_OPS=<kernel> \
  EJKERNEL_BENCH_PLATFORMS=xla,pallas \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=5 \
  EJKERNEL_BENCH_ITERS=30 \
  EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_tpu \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py

# GPU (XLA vs GPU backends)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_BENCH_OPS=<kernel> \
  EJKERNEL_BENCH_PLATFORMS=xla,triton,cuda,cute,tilelang \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=5 \
  EJKERNEL_BENCH_ITERS=30 \
  EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_gpu \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Verified suite & registry env vars:

- `EJKERNEL_BENCH_OPS` / `EJKERNEL_BENCH_SKIP_OPS`
- `EJKERNEL_BENCH_PLATFORMS` / `EJKERNEL_BENCH_IGNORE_PLATFORMS` (values from `Platform`: `xla`, `pallas`, `triton`,
  `cuda`, `cute`, `tilelang`)
- `EJKERNEL_BENCH_OUTPUT_DIR`
- `EJKERNEL_BENCH_CONFIG_LIMIT` (`0` = all configs)
- `EJKERNEL_BENCH_WARMUP` (default `5`), `EJKERNEL_BENCH_ITERS` (default `30`)

Ops instrumentation env vars:

- `EJKERNEL_OPS_STAMP=hash|json` — injects `ejkernel_ops#...` scopes into HLO/xprof traces.
- `EJKERNEL_OPS_RECORD=1` — records invocations for offline autotuning (`autotune_recorded` / `autotune_lowered`).
- `EJKERNEL_TUNED_DB=/path/to/custom.db` — validates a `python -m ejkernel.ops.tuned sweep` table before merging.

Single-op shims (`libs/ejkernel/benchmarks/benchmark_<op>.py`) call `run_benchmark("<op>")` with fixed `warmup=5` and
`iterations=30`; prefer `benchmark_suite.py` for env-controlled runs and JSON/Markdown artifacts. Note that checked-in
files in `libs/ejkernel/benchmarks/baselines/` are historical artifacts (`baselines/README.md`) with no active perf-gate
script.

Non-compute host preflight (validates registration/spec wiring only — executes zero JAX numerics):

```bash
uv run python -c "import ejkernel.kernels; from benchmarks._op_benchmark_registry import SPECS; print('<kernel>' in SPECS)"
```

## Embedded JAX Profiler Hooks

- **Stepped loops:** `ejkernel.loggings.create_step_profiler(...)` starts at `start_step - 1`, stops at the window end,
  and calls `barrier_sync()`.
- **Manual regions:** `ignite_profiler(profile_path, enable_perfetto=False)` and
  `extinguish_profiler(enable_perfetto)` — note `enable_perfetto` is a **required** boolean argument on
  `extinguish_profiler`. Warm up (compile) at least one step before calling `ignite_profiler` and call
  `block_until_ready()` before `extinguish_profiler`.
- **Trainers:** configure `TrainingArguments(profiler_path=..., profiler_stop_step=...)` (set `profiler_duration_ms` or
  `profiler_enable_hlo_proto=False` if traces approach the 2 GB XSpace cap).

## What To Report

For every headline performance result include:

- Hardware model, backend, and device count
- Operation name, `Platform`, and selected implementation
- Shape/dtype/config grid and cache state (`EJKERNEL_AUTOTUNE_POLICY`, `EJKERNEL_PERSISTENT_CACHE_DIR`,
  `EJKERNEL_TUNED_DB`)
- Warmup and timed iteration counts
- Output JSON/Markdown artifact path (and trace/dump paths if collected)
- Steady-state timing (explicitly labeled steady-state after warmup; measure and report compile latency separately if
  relevant)
- Direct baseline-vs-candidate numbers, plus skipped platforms or narrowed scope

## Common Mistakes

- Executing `benchmark_suite.py` on CPU and reporting the numbers as indicative or preflight evidence.
- Leaving `EJKERNEL_AUTOTUNE_POLICY=autotune` active during deterministic unit tests or mixing cache states between
  baseline and candidate.
- Calling `extinguish_profiler()` without its required `enable_perfetto` boolean argument, or stopping a trace without
  `block_until_ready()`.
- Changing shape, dtype, platform set, warmup, or iterations between baseline and candidate.
- Treating an isolated kernel microbenchmark as proof of an EasyDeL training or eSurge serving speedup without
  measuring the end-to-end path.
