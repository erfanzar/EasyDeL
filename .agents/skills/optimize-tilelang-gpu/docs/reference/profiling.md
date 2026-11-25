# TileLang GPU Profiling Cheatsheet (ejKernel / JAX TVM-FFI)

Exact commands, TVM-FFI inspection recipes, and env vars for `.agents/skills/optimize-tilelang-gpu/SKILL.md`. Lock GPU
clocks before timing and use `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` for the shared
`nvidia-smi -lgc`, `ncu`, `nsys`, and `compute-sanitizer` reference.

## 1. ejKernel TileLang Invocation, Testing & Benchmarking

In ejKernel, TileLang kernels **always** use `out_idx=None` and `execution_backend="tvm_ffi"` (enforced inside
`libs/ejkernel/ejkernel/callib/_tilelang_ffi.py`) because JAX preallocates output buffers and passes all inputs followed
by all outputs explicitly. Never pass `out_idx=[-1]` or `@tilelang.jit`; always benchmark through ejKernel's JAX
harnesses below.

Standard TileLang GPU test + static guardrail run:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest libs/ejkernel/test/kernels/_tilelang/test_<op>.py \
                libs/ejkernel/test/kernels/_tilelang/test_honesty.py \
                libs/ejkernel/test/kernels/_tilelang/test_registry_coverage.py \
                libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py
```

Benchmarking via `libs/ejkernel/benchmarks/benchmark_suite.py` (must list `tilelang` explicitly because
`detect_platform()` never auto-selects it):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
EJKERNEL_BENCH_OPS=flash_attention \
EJKERNEL_BENCH_PLATFORMS=xla,tilelang,triton,cute,cuda \
EJKERNEL_BENCH_CONFIG_LIMIT=3 \
EJKERNEL_BENCH_WARMUP=5 \
EJKERNEL_BENCH_ITERS=30 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_tilelang \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

For a custom JAX microbenchmark comparing two TileLang variants, use `ejkernel.benchmarks.Benchmark`
(`libs/ejkernel/ejkernel/benchmarks.py`) with `jax.block_until_ready` handled automatically.

## 2. Config Selection & Verified `EJKERNEL_TILELANG_*` Env Vars

Production kernels under `libs/ejkernel/ejkernel/kernels/_tilelang/` must NOT run local autotuning (banned by
`libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py`). Tune configs at the operation layer
(`libs/ejkernel/ejkernel/modules/operations/<op>.py`) through `Executor` + `ConfigSelectorChain`:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
EJKERNEL_AUTOTUNE_POLICY=autotune \
EJKERNEL_LOG_AUTOTUNE=1 \
EJKERNEL_AUTOTUNE_PROGRESS=1 \
EJKERNEL_PERSISTENT_CACHE_DIR=/tmp/ejk_tilelang_autotune_cache \
  uv run python <probe_calling_op_with_platform_tilelang.py>
```

Verified env vars read by `libs/ejkernel/ejkernel/callib/_tilelang_call.py`:

- `EJKERNEL_TILELANG_TARGET` — explicit TileLang target (e.g. `"cuda"`), overriding auto-detection.
- `EJKERNEL_TILELANG_COMPILE_FLAGS` — NVCC flags forwarded to `tilelang.compile` (overrides kernel defaults such as
  `("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)`).
- `EJKERNEL_TILELANG_AUTOTUNE_WARMUP` (default `25`), `EJKERNEL_TILELANG_AUTOTUNE_REP` (default `100`),
  `EJKERNEL_TILELANG_AUTOTUNE_TIMEOUT` (default `60` seconds; set `none`/`off` to disable) — used only by
  `build_tilelang_call(..., configs=...)` / `autotune_tilelang_ffi` in ad-hoc scratch prototypes outside
  `libs/ejkernel/ejkernel/kernels/`.

## 3. Read The Generated CUDA — Confirm Lowering Intent

Either compile a single `@T.prim_func` with ejKernel's exact TVM-FFI contract, or read already-compiled entries from
`_tilelang_ffi._COMPILE_CACHE`:

```python
import tilelang
from ejkernel.callib import _tilelang_ffi

# Option A: compile one prim_func directly with ejKernel's exact contract
compiled = tilelang.compile(
    prim_func,
    out_idx=None,
    execution_backend="tvm_ffi",
    compile_flags=["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"],
)
print(compiled.get_kernel_source())

# Option B: inspect kernels compiled during a JAX run
for key, entry in _tilelang_ffi._COMPILE_CACHE.items():
    print(entry.target_name)
    print(entry.kernel.get_kernel_source())
```

What to verify in the emitted CUDA source:

- `T.gemm` lowered to tensor-core MMA (`mma`/`wgmma`/CUTLASS fragment ops), not scalar loops.
- `T.copy` vectorized (`float4`/128-bit) or emitted `cp.async` / Hopper TMA bulk copies.
- Shared-memory allocation and swizzle layout match your `T.annotate_layout` / `T.use_swizzle` intent.
- Total SMEM across `T.alloc_shared` (multiplied by `num_stages` under `T.Pipelined`) fits the SM budget.

## 4. Hardware Counters (`ncu`) & Timeline (`nsys`) on JAX

Always set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_FLAGS='--xla_gpu_enable_command_buffer='` when profiling JAX
TileLang TVM-FFI calls under `ncu` or `nsys`:

```bash
# 1) List emitted CUDA kernel names in your JAX probe:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --list-kernels uv run python <probe.py>

# 2) Full profile of 1 steady-state launch:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --set full -k "regex:<generated_kernel_name>" --launch-skip 3 --launch-count 1 \
  -o /tmp/tilelang_ncu uv run python <probe.py>

# 3) Timeline (spot transpose/host/multi-kernel overhead):
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  nsys profile --trace=cuda,nvtx,osrt --stats=true -o /tmp/tilelang_nsys \
  uv run python <probe.py>
```

Read `ncu` sections in the exact order in `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` (Speed Of Light
-> Memory Workload Analysis with 32-byte sectors / 4-sector 128 B cache lines -> Occupancy -> Scheduler/Warp State ->
Source Counters). For JAX trace scopes, use `EJKERNEL_OPS_STAMP=hash|json` with
`ejkernel.ops.execution.profiler.Profiler` or `ejkernel.loggings.{ignite_profiler,extinguish_profiler}`.

## 5. Reporting Checklist

- GPU model (SM arch), CUDA + TileLang + JAX versions, clock-lock state
- Exact op, `platform="tilelang"`, shape, dtype, tile sizes (`BM/BN/BK`), `threads`, `num_stages` (`T.Pipelined` vs
  `T.Serial`)
- `benchmark_suite.py` JSON/Markdown artifact paths and steady-state timings vs `Platform.XLA` (and sibling GPU
  platforms) under identical clock-locked conditions
- `ncu` SOL %s, coalescing sectors/request, bank-conflict and spill readings
- `get_kernel_source()` note confirming tensor-core MMA / async copy lowering
- Passing GPU `pytest` parity (including boundary shapes), `test_honesty.py`, and
  `test_backend_autotune_guardrails.py`
