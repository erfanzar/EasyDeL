---
name: optimize-ejkernel-kernel
description: Optimize, profile, regress, retune, or diagnose an existing ejkernel kernel or operation. Use for performance regressions, Pallas TPU/GPU tuning, Triton/CUDA/CuTe/TileLang tuning, XLA-vs-accelerator comparisons, benchmark-suite evidence, HLO/LLO/Mosaic dump analysis, DMA/async overlap validation, autotune/config-cache changes, or "kernel is slower than baseline" work in libs/ejkernel.
---

# Skill: Optimize An Existing eJKernel Kernel

Specialization of `.agents/skills/run-research/SKILL.md` for making an existing `libs/ejkernel/` kernel faster,
diagnosing a regression, retuning configs, or validating performance claims. To add a new operation or backend surface,
load `.agents/skills/add-ejkernel-kernel/SKILL.md` instead.

## Backend Optimization Routing

Always pair this skill with the matching target-backend skill:

- **TPU Pallas / Mosaic:** `.agents/skills/optimize-pallas-tpu/SKILL.md`
- **Triton GPU (CUDA / ROCm):** `.agents/skills/optimize-triton-gpu/SKILL.md`
- **Native CUDA C++ / JAX FFI:** `.agents/skills/optimize-cuda-gpu/SKILL.md`
- **CUTLASS CuTe DSL (TVM-FFI):** `.agents/skills/optimize-cute-gpu/SKILL.md`
- **TileLang GPU (TVM-FFI):** `.agents/skills/optimize-tilelang-gpu/SKILL.md`
- **Ops framework / selection / cache bugs:** `.agents/skills/debug-ejkernel-ops/SKILL.md`
- **Cross-stack A/B benchmarking:** `.agents/skills/benchmark-changes/SKILL.md`

> **CI Guardrail:** `libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` **bans** `@triton.autotune`,
> `triton.autotune`, `@autotune(`, `_AUTOTUNE_CACHE`, `autotune_tilelang_ffi`, and `EJKERNEL_*AUTOTUNE*` under
> `libs/ejkernel/ejkernel/kernels/`. All config selection and autotuning belong in `modules/operations/<op>.py`
> (`Executor` / `ConfigSelectorChain`) or `ejkernel.ops.tuned`.

## First Reads

Read the authoritative code and profiling references before editing:

- `WORKSPACE.md` and `libs/ejkernel/pyproject.toml`
- `libs/ejkernel/ejkernel/kernels/_registry.py` and `libs/ejkernel/ejkernel/modules/base.py` (authoritative over prose
  docs `libs/ejkernel/docs/kernel_registry_system.md` and `libs/ejkernel/docs/api/ops.md`)
- `libs/ejkernel/benchmarks/benchmark_suite.py`, `libs/ejkernel/benchmarks/_op_benchmark_registry.py`, and
  `libs/ejkernel/ejkernel/benchmarks.py`
- `libs/ejkernel/ejkernel/ops/execution/{executor,profiler,offline}.py`, `libs/ejkernel/ejkernel/ops/config/{selection,persistent}.py`,
  and `libs/ejkernel/ejkernel/ops/tuned/`
- `.agents/skills/optimize-ejkernel-kernel/docs/reference/profiling.md` (canonical profiling/benchmark reference)
- `.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` (canonical TPU HLO/LLO/Mosaic dump reference)
- Target op files: `libs/ejkernel/ejkernel/modules/operations/<kernel>.py`, `_xla/<kernel>/`, `<backend>/<kernel>/`,
  and tests under `libs/ejkernel/test/kernels/<backend>/test_<kernel>.py` and `test/modules/operations/test_<kernel>.py`.

## Baseline First

1. Record exact op, platform/backend, hardware, device count, shape, dtype, config, cache policy, and command.
2. Run parity tests on the target accelerator before changing code.
3. Capture a steady-state baseline via `benchmark_suite.py` and save the JSON/Markdown output.
4. Name the exact comparison: XLA vs accelerator backend, last-good commit vs candidate, config A vs B, or full kernel
   vs decomposition variant.

> **NVIDIA GPU dispatch warning:** `detect_platform()` (`modules/base.py`) and `Executor._prefer_cuda_cfg()`
> (`executor.py`) automatically upgrade `platform="auto"` and `"triton"` to `"cute"` (preferred) or `"cuda"` on NVIDIA
> GPUs when available, and never auto-select `"tilelang"`. Always pass explicit `platform=` (or `EJKERNEL_BENCH_PLATFORMS`)
> so you benchmark the backend you actually modified.

## Benchmark Workflow

Use `libs/ejkernel/benchmarks/benchmark_suite.py` for machine-readable JSON/Markdown comparisons (warmup and iterations
identical between baseline and candidate). Checked-in JSONs under `libs/ejkernel/benchmarks/baselines/` are historical
artifacts (`baselines/README.md`), not an active CI perf gate.

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

# GPU (XLA vs Triton / CUDA / CuTe / TileLang)
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

Suite env vars: `EJKERNEL_BENCH_OPS`, `EJKERNEL_BENCH_SKIP_OPS`, `EJKERNEL_BENCH_PLATFORMS`,
`EJKERNEL_BENCH_IGNORE_PLATFORMS`, `EJKERNEL_BENCH_OUTPUT_DIR`, `EJKERNEL_BENCH_CONFIG_LIMIT`, `EJKERNEL_BENCH_WARMUP`,
`EJKERNEL_BENCH_ITERS`. Single-op shims (`libs/ejkernel/benchmarks/benchmark_<op>.py`) call `run_benchmark("<op>")` with
fixed `warmup=5, iterations=30`.

See `.agents/skills/optimize-ejkernel-kernel/docs/reference/profiling.md` for profiler hooks and reporting rules.

## Optimization Loop

1. Freeze one representative `(shape, dtype, hardware, device count)`.
2. Attribute the bottleneck before touching code: benchmark JSON, profiler trace, HLO/LLO/Mosaic dump (TPU), or
   ncu/nsys/IR dump (GPU).
3. Change **one** lever: tiling/block size, grid order, memory layout, staging, DMA/pipelining, masking, or config
   policy.
4. Re-run accelerator parity tests and the exact same steady-state benchmark.
5. Keep only verified hardware winners that preserve parity across supported and edge shapes.

Never broad-sweep block sizes before checking structure. When TPU Pallas is slower than XLA on a fixed shape, follow
`.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` before adding DMA or async complexity.

## Config, PersistentCache & `ejkernel.ops.tuned`

- **Runtime selection stack:** `ConfigSelectorChain`, `ConfigCache`, `PersistentCache`, and `Executor` under
  `libs/ejkernel/ejkernel/ops/`.
  - `EJKERNEL_AUTOTUNE_POLICY=heuristics` for deterministic baseline/candidate checks; `autotune` only when explicitly
    measuring candidate configs.
  - `PersistentCache` defaults to `~/ejkernel-presistent-cache/<op>.json` (exact spelling; override with
    `EJKERNEL_PERSISTENT_CACHE_DIR`). Only `PROVENANCE_AUTOTUNE` entries are trusted on read — clear or pin the cache
    directory identically across baseline and candidate so stale entries never mask changes.
  - Diagnostics: `EJKERNEL_AUTOTUNE_PROGRESS=1`, `EJKERNEL_LOG_AUTOTUNE=1`, `EJKERNEL_OPS_STAMP=hash|json`,
    `EJKERNEL_OPS_RECORD=1`.
- **Shipped tuned table (`libs/ejkernel/ejkernel/ops/tuned/`):** stores measured `(platform, config)` winners and
  runner-up timings in `tuned_kernels.db`, queried via `tuned_choice(kernel, dtypes=..., shape=...)`. Point lookups at a
  candidate DB with `EJKERNEL_TUNED_DB=/tmp/<op>.db` to validate sweeps before merging:
  ```bash
  uv run python -m ejkernel.ops.tuned stats
  uv run python -m ejkernel.ops.tuned dump --kernel <op>
  uv run python -m ejkernel.ops.tuned sweep --kernel <op> --out /tmp/<op>.db
  uv run python -m ejkernel.ops.tuned merge --into libs/ejkernel/ejkernel/ops/tuned/tuned_kernels.db /tmp/<op>.db
  ```

## Verification

CPU is strictly for non-computation host preflight (imports, registry/spec presence, test collection — NEVER executing
`benchmark_suite.py` or kernel numerics on CPU). All kernel tests and benchmarks MUST run on TPU or GPU; if hardware is
unavailable or `libtpu` is busy (`.agents/ops/OPS.md`), report the change as **unverified on hardware**.

```bash
# Non-compute preflight only (no JAX execution)
uv run python -c "import ejkernel.kernels; from benchmarks._op_benchmark_registry import SPECS; print('<kernel>' in SPECS)"
uv run pytest --collect-only libs/ejkernel/test/modules/operations/test_<kernel>.py

# Target TPU verification
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/ejkernel/test/kernels/_pallas/tpu/test_<kernel>.py libs/ejkernel/test/modules/operations/test_<kernel>.py

# Target GPU verification
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest libs/ejkernel/test/kernels/_<backend>/test_<kernel>.py libs/ejkernel/test/modules/operations/test_<kernel>.py
```

## Definition Of Done

- Baseline and candidate commands, env vars, hardware, device count, shape, dtype, config, cache policy, warmup, and
  iterations are recorded.
- Parity passes on target hardware for representative and edge shapes.
- Direct baseline-vs-candidate steady-state numbers and JSON/Markdown artifact paths are reported.
- Dump/profile evidence is included when timing alone did not explain the delta; losing variants are removed.
