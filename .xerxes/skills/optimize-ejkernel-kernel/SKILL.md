---
name: optimize-ejkernel-kernel
description: Optimize, profile, regress, retune, or diagnose an existing ejkernel kernel or operation. Use for performance regressions, Pallas TPU/GPU tuning, Triton/CUDA/CuTe/TileLang tuning, XLA-vs-accelerator comparisons, benchmark-suite evidence, HLO/LLO/Mosaic dump analysis, DMA/async overlap validation, autotune/config-cache changes, or "kernel is slower than baseline" work in libs/ejkernel.
---

# Skill: Optimize An Existing eJKernel Kernel

This is a specialization of `.xerxes/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the operation already exists and the job is to make it faster,
explain a regression, or validate a performance claim. If the task adds a new operation or backend surface, load
`.xerxes/skills/add-ejkernel-kernel/SKILL.md` instead.

## First Reads

Read the real ejkernel docs and the current implementation before changing anything:

- `WORKSPACE.md`
- `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/docs/kernel_registry_system.md`
- `libs/ejkernel/docs/ops_system_architecture.md`
- `libs/ejkernel/docs/kernel_implementations.md`
- `libs/ejkernel/docs/test_suite_and_examples.md`
- `libs/ejkernel/docs/api/ops.md`
- `libs/ejkernel/benchmarks/README.md`
- `libs/ejkernel/benchmarks/benchmark_suite.py`
- `libs/ejkernel/benchmarks/_op_benchmark_registry.py`
- `libs/ejkernel/ejkernel/benchmarks.py`
- `libs/ejkernel/ejkernel/ops/execution/profiler.py`
- `libs/ejkernel/ejkernel/loggings.py`
- `docs/reference/profiling.md` before any performance claim.
- `docs/reference/llo.md` for TPU Pallas structural diagnosis.

Then open the existing operation wrapper, kernel backend, XLA/reference path, and tests for the exact operation:

- `libs/ejkernel/ejkernel/modules/operations/<kernel>.py`
- `libs/ejkernel/ejkernel/kernels/_xla/<kernel>/`
- `libs/ejkernel/ejkernel/kernels/<backend>/<kernel>/`
- `libs/ejkernel/test/kernels/<backend>/test_<kernel>.py`
- `libs/ejkernel/test/modules/operations/test_<kernel>.py` when present

## Baseline First

Before editing:

1. Record the exact operation, backend/platform, hardware, device count, shape, dtype, config, and command.
2. Run correctness or parity for the current implementation.
3. Run a baseline benchmark and save the JSON/Markdown artifact.
4. Identify whether the comparison is XLA vs accelerator, previous commit vs current, config A vs config B, or isolated
   decomposition vs full kernel.

Do not report a win from a single changed command line or a different shape. Keep warmup, iteration count, platform set,
config limit, and hardware fixed between baseline and candidate.

## Benchmark Workflow

Use the registry-driven suite first when the operation has a spec:

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

Useful env vars verified in the benchmark suite:

- `EJKERNEL_BENCH_OPS`
- `EJKERNEL_BENCH_SKIP_OPS`
- `EJKERNEL_BENCH_PLATFORMS`
- `EJKERNEL_BENCH_OUTPUT_DIR`
- `EJKERNEL_BENCH_CONFIG_LIMIT`
- `EJKERNEL_BENCH_WARMUP`
- `EJKERNEL_BENCH_ITERS`
- `EJKERNEL_BENCH_IGNORE_PLATFORMS`

Single-op wrappers under `libs/ejkernel/benchmarks/` call
`run_benchmark("<kernel>")`, for example:

```bash
uv run python libs/ejkernel/benchmarks/benchmark_gated_delta_rule.py
```

Read `docs/reference/profiling.md` for profiler hooks, output artifacts, and reporting requirements.

## Optimization Loop

Use a narrow loop:

1. Freeze one representative shape and dtype.
2. Compare current backend against XLA/reference or last-known-good baseline.
3. Attribute the bottleneck before changing code: benchmark JSON, profile trace, HLO/LLO/Mosaic dump, or decomposition
   variant.
4. Change one thing: tiling, block size, config choice, memory layout, staging, DMA, masking, or dispatch policy.
5. Rerun correctness and the same benchmark.
6. Keep the change only if it wins on the target hardware and does not break parity or supported shapes.

Do not broad-sweep block sizes before checking structure. If Pallas is slower than XLA on one fixed shape, read
`docs/reference/llo.md` and dump both paths before adding DMA or async complexity.

## TPU Pallas Rules

- Run TPU Pallas validation only when this process owns libtpu.
- CPU/XLA can be used only for host-side preflight: imports, benchmark harness syntax, registry selection, or simple
  reference math.
- CPU timing is not TPU correctness, Mosaic lowering, LLO behavior, DMA/async overlap, or performance evidence.
- DMA/async only counts when the measured target run shows overlap and a win.
- If libtpu is busy, stop target validation and say it was not run.

## Config And Autotune

For tunable operations, use the existing ops stack:

- `libs/ejkernel/ejkernel/ops/config/selection.py` for
  `ConfigSelectorChain`.
- `libs/ejkernel/ejkernel/ops/config/cache.py` for `ConfigCache`.
- `libs/ejkernel/ejkernel/ops/config/persistent.py` for `PersistentCache`.
- `libs/ejkernel/ejkernel/ops/execution/executor.py` for `Executor`.

Use `EJKERNEL_AUTOTUNE_POLICY=heuristics` for deterministic checks. Use
`autotune` only when measuring config candidates, and report cache state when it can affect the result.

## Verification

Host-side preflight only:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
EJKERNEL_BENCH_OPS=<kernel> \
EJKERNEL_BENCH_PLATFORMS=xla \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=1 \
EJKERNEL_BENCH_ITERS=1 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_cpu_preflight \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Target TPU correctness and benchmark:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
  uv run pytest libs/ejkernel/test/kernels/_pallas/tpu/test_<kernel>.py

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
EJKERNEL_BENCH_OPS=<kernel> \
EJKERNEL_BENCH_PLATFORMS=xla,pallas \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=5 \
EJKERNEL_BENCH_ITERS=30 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

## Definition Of Done

- Baseline and candidate commands are recorded.
- Correctness/parity passes for the affected backend and representative shapes.
- Benchmark artifact paths are reported.
- Direct baseline vs candidate numbers are reported.
- Hardware, shape, dtype, config, warmup, iterations, and platform are named.
- Dump/profile evidence is included when timing alone did not explain the result.
- Losing variants are removed or left behind only as clearly marked references.
