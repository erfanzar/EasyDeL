# CUTLASS CuTe DSL GPU Profiling Cheatsheet (ejKernel / JAX TVM-FFI)

Exact commands, diagnostic env vars, TVM-FFI cache inspection, and profiler invocations for
`.agents/skills/optimize-cute-gpu/SKILL.md`. Lock GPU clocks before timing and use
`.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` for the shared `nvidia-smi -lgc`, `ncu`, `nsys`, and
`compute-sanitizer` reference.

## 1. Environment & Testing

CuTe DSL kernels require the `[cuda]` extra in `libs/ejkernel/pyproject.toml` (`nvidia-cutlass-dsl`, `jax-tvm-ffi`,
`apache-tvm-ffi`). `CUTE_DSL_ENABLE_TVM_FFI=1` is set automatically on `import ejkernel`
(`libs/ejkernel/ejkernel/__init__.py`) and in each CuTe `_cute_impl*.py`.

Quick readiness check on an NVIDIA GPU host:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run python -c "import ejkernel; from ejkernel.callib._cute_ffi import has_cute_ffi_support; print(has_cute_ffi_support())"
```

Run CuTe parity tests + backend guardrails on GPU:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest libs/ejkernel/test/kernels/_cute/test_<op>.py \
                libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py
```

## 2. Benchmarking — `benchmark_suite.py` & `ejkernel.benchmarks.Benchmark`

Because `Platform.CUTE` is auto-preferred over `Platform.CUDA` and `Platform.TRITON` on NVIDIA GPUs, always benchmark
`cute` alongside `xla`, `cuda`, `triton`, and `tilelang` under identical clock-locked settings:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
EJKERNEL_BENCH_OPS=quantized_matmul \
EJKERNEL_BENCH_PLATFORMS=xla,cute,cuda,triton,tilelang \
EJKERNEL_BENCH_CONFIG_LIMIT=3 \
EJKERNEL_BENCH_WARMUP=5 \
EJKERNEL_BENCH_ITERS=30 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_cute \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

For operation-level config tuning (never inside `ejkernel/kernels/_cute/` — enforced by
`test_backend_autotune_guardrails.py`), run `Executor` with `EJKERNEL_AUTOTUNE_POLICY=autotune EJKERNEL_LOG_AUTOTUNE=1`
and an isolated `EJKERNEL_PERSISTENT_CACHE_DIR=/tmp/ejk_cute_autotune_cache`.

## 3. Verified CuTe Diagnostic Env Vars (Tier Ablation & Attribution)

Existing env vars verified in `libs/ejkernel/ejkernel/kernels/_cute/`:

- `CUTE_DSL_ENABLE_TVM_FFI=1` — enables TVM-FFI compilation in CuTe DSL (defaulted by `ejkernel`).
- **Quantized Matmul 4-tier ladder** (`libs/ejkernel/ejkernel/kernels/_cute/quantized_matmul/_cute_impl.py`):
  - Default (unset): 3-stage pipelined Tensor Core MMA (`_build_mma_pipelined_qmm_host_fns`, requires
    `tile_m % 32 == 0`, `tile_n % 32 == 0`, `tile_k % 16 == 0`; otherwise silently falls back to tiled/naive).
  - `EJKERNEL_CUTE_QMM_USE_MMA_SINGLE=1`: forces single-stage MMA (`_build_mma_qmm_host_fns`, no `cp.async` pipeline).
  - `EJKERNEL_CUTE_QMM_USE_TILED=1`: forces tiled SMEM + 4x4 register scalar kernel (`_build_tiled_qmm_host_fns`).
  - `EJKERNEL_CUTE_QMM_USE_NAIVE=1`: forces naive scalar fallback (`_build_naive_qmm_host_fns`).
- **Chunked prefill KV scatter** (`libs/ejkernel/ejkernel/kernels/_cute/chunked_prefill_paged_decode/_cute_impl_fwd.py`):
  - `EJKERNEL_CUTE_CPPD_THREADS` — CTA thread count for `_kv_update_kernel` (default `256`, min `32`).
- **Unified attention wrapper** (`libs/ejkernel/ejkernel/kernels/_cute/unified_attention/_cute_impl.py`):
  - `EJKERNEL_CUTE_UNIFIED_ATTENTION_DISABLE_TRITON=1` — disables delegation to Triton `unified_attention` (raises
    `EjkernelRuntimeError` until a native CuTe body is implemented).

Example tier-by-tier QMM ablation:

```bash
for mode in "" "EJKERNEL_CUTE_QMM_USE_MMA_SINGLE=1" "EJKERNEL_CUTE_QMM_USE_TILED=1" "EJKERNEL_CUTE_QMM_USE_NAIVE=1"; do
  echo "=== tier: ${mode:-pipelined_mma} ==="
  env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
    EJKERNEL_AUTOTUNE_POLICY=heuristics EJKERNEL_BENCH_OPS=quantized_matmul \
    EJKERNEL_BENCH_PLATFORMS=cute EJKERNEL_BENCH_CONFIG_LIMIT=1 \
    EJKERNEL_BENCH_WARMUP=5 EJKERNEL_BENCH_ITERS=30 \
    EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejk_cute_tiers \
    ${mode:+$mode} uv run python libs/ejkernel/benchmarks/benchmark_suite.py
done
```

## 4. Inspecting Compiled TVM-FFI Cache & Confirming Active Tier

Verify that `build_cute_ffi_call` compiled once and reused the cached TVM-FFI target across dynamic shapes (and check
which `@cute.jit` launcher function actually landed in `_COMPILE_CACHE`):

```python
from ejkernel.callib import _cute_ffi
from ejkernel.kernels._cute.quantized_matmul import _cute_impl as qmm_impl

for key, compiled in _cute_ffi._COMPILE_CACHE.items():
    fn, in_keys, out_keys, opts, static_kw = key
    print(compiled.target_name, fn.__qualname__, in_keys, out_keys, opts)

print("QMM call cache entries:", len(qmm_impl._QMM_CALL_CACHE))
```

Watch out: if `qmm_tiled_kernel` or `qmm_kernel` (naive) appears in `ncu --list-kernels` when you expected pipelined
MMA, check whether `tile_m`/`tile_n`/`tile_k` failed `_validate_mma_tile` or whether `cute.compile` hit an alignment
error (requiring `assumed_align=16` and stride divisibility `128 // dtype.width`) and fell through the `try/except` in
`_build_fused_qmm_host_fns`.

## 5. Hardware Counters (`ncu`), Timeline (`nsys`) & `compute-sanitizer`

Use the shared JAX GPU flags from `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` so XLA command buffers
and preallocation do not interfere with profiling:

```bash
# 1) List exact CuTe kernel names launched by your JAX probe:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --list-kernels uv run python <probe.py>

# 2) Full ncu profile of 1 steady-state launch:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --set full -k "regex:<cute_kernel_name>" --launch-skip 3 --launch-count 1 \
  -o /tmp/cute_ncu uv run python <probe.py>

# 3) Timeline (spot XLA cast/transpose/dummy-buffer overhead around cute_call):
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  nsys profile --trace=cuda,nvtx,osrt --stats=true -o /tmp/cute_nsys \
  uv run python <probe.py>

# 4) Sanitizer gate on a focused CuTe pytest case:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  compute-sanitizer --tool memcheck \
  uv run pytest libs/ejkernel/test/kernels/_cute/test_<op>.py -k "<case>"
```

In `ncu`, check **Speed Of Light** (confirm Tensor Core / HMMA pipe utilization, not just FP32 ALU), **Memory Workload
Analysis** (32-byte sectors/request — 4 sectors/req for 32-bit, 2 for 16-bit, 16 for 128-bit `CopyG2SOp`; L2 hit rate;
SMEM bank conflicts), **Occupancy**, and **Source Counters** (local spills). For JAX-level scopes (including
`cute_call(..., name=...)`), enable `EJKERNEL_OPS_STAMP=hash|json` with `ejkernel.ops.execution.profiler.Profiler` or
`ejkernel.loggings.{ignite_profiler,extinguish_profiler}`.

## 6. Reporting Checklist

- GPU model (SM arch), CUDA + `nvidia-cutlass-dsl` + `jax-tvm-ffi` + JAX versions, clock-lock state
- Exact op, `platform="cute"`, shape, dtype, confirmed kernel tier (e.g. pipelined MMA vs tiled vs naive) and tile
  config (`tile_m`, `tile_n`, `tile_k` / `q_block`)
- `benchmark_suite.py` JSON/Markdown artifact paths and steady-state timings vs `Platform.XLA` and sibling NVIDIA
  backends (`cuda`, `triton`, `tilelang`) under identical clock-locked conditions
- `ncu` SOL %s (including Tensor Core pipe %), sectors/request, SMEM bank conflicts, top stall reason, and register
  spill status
- `compute-sanitizer` clean, GPU `pytest` parity passing (including boundary shapes), and
  `test_backend_autotune_guardrails.py` passing
