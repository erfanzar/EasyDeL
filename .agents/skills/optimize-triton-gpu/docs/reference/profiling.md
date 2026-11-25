# Triton GPU Profiling Cheatsheet (ejKernel / JAX)

Exact commands, env vars, IR-dump recipes, and profiler invocations for `.agents/skills/optimize-triton-gpu/SKILL.md`.
Lock GPU clocks before timing and use `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` for the shared
`nvidia-smi -lgc`, `ncu`, `nsys`, and `compute-sanitizer` reference.

## 1. Timing & Benchmarking — `benchmark_suite.py` and `ejkernel.benchmarks.Benchmark`

Always benchmark through ejKernel's JAX harnesses against `Platform.XLA` (and sibling GPU platforms when registered).
Force `platform="triton"` or `EJKERNEL_BENCH_PLATFORMS=xla,triton` so `Executor._prefer_cuda_cfg` does not silently
substitute CuTe or CUDA on NVIDIA GPUs.

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
EJKERNEL_BENCH_OPS=flash_attention \
EJKERNEL_BENCH_PLATFORMS=xla,triton,cute,cuda,tilelang \
EJKERNEL_BENCH_CONFIG_LIMIT=3 \
EJKERNEL_BENCH_WARMUP=5 \
EJKERNEL_BENCH_ITERS=30 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_triton \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

For a focused microbenchmark script comparing two Triton variants under `jax.jit`, use `ejkernel.benchmarks.Benchmark`
(`libs/ejkernel/ejkernel/benchmarks.py`):

```python
import jax
from ejkernel.benchmarks import Benchmark
from ejkernel.kernels._xla.decode_attention import decode_attention as xla_decode
from ejkernel.kernels._triton.decode_attention import decode_attention as triton_decode

bench = Benchmark(
    algorithms={"xla": xla_decode, "triton": triton_decode},
    configs=[{"batch": 8, "seq_len": 2048, "q_heads": 32, "kv_heads": 8, "head_dim": 128}],
    input_generator=make_inputs,  # returns tuple matching kernel args
    warmup=5,
    iterations=30,
    unpack_inputs=True,
)
bench.run()
bench.save("/tmp/decode_triton.json")
```

If timing manually in a throwaway probe, always compile with `jax.jit`, run warmup iterations, and call
`jax.block_until_ready(out)` on every timed iteration; report median and min across iterations, never include compile
time in steady-state numbers.

## 2. Config Selection & Autotune Inspection (`Executor` / `ConfigSelectorChain`)

`@triton.autotune` is banned under `libs/ejkernel/ejkernel/kernels/` (`test_backend_autotune_guardrails.py`), and
`_triton_call.py` bypasses Triton's Python autotuner (`TRITON_PRINT_AUTOTUNING=1` does not fire). Inspect ejKernel's
real config selector instead:

```bash
# Deterministic heuristics (no autotune sweep):
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  uv run python <script.py>

# Autotune candidates in candidate_cfgs_gpu with per-candidate logs + isolated cache:
EJKERNEL_AUTOTUNE_POLICY=autotune \
EJKERNEL_LOG_AUTOTUNE=1 \
EJKERNEL_AUTOTUNE_PROGRESS=1 \
EJKERNEL_PERSISTENT_CACHE_DIR=/tmp/ejk_triton_autotune_cache \
  uv run python <script.py>
```

Key env vars verified in `libs/ejkernel/ejkernel/`:

- `EJKERNEL_AUTOTUNE_POLICY=heuristics|autotune` — controls `cache_miss_fallback` on operation executors.
- `EJKERNEL_LOG_AUTOTUNE=1` — logs every candidate config + measured time and the winning `best_config` in `Tuner.autotune`
  (`libs/ejkernel/ejkernel/ops/config/selection.py`).
- `EJKERNEL_AUTOTUNE_PROGRESS=1` — live tqdm bar (or `log_autotune_progress()` / `set_autotune_progress(True)`).
- `EJKERNEL_PERSISTENT_CACHE_DIR` — overrides `~/ejkernel-presistent-cache/<op>.json`. Clear or isolate it when a code
  change alters candidate timing so stale winners do not mask regressions.
- `EJKERNEL_TRITON_SMEM_LIMIT` — shared-memory byte budget (default `99 * 1024`) used to prune Triton candidates in
  `libs/ejkernel/ejkernel/modules/operations/{flash_attention,blocksparse_attention}.py`.
- `EJKERNEL_OPS_STAMP=hash|json` — stamps `Executor` scopes (`jax.named_call` / `jax.named_scope`) in JAX profiles and
  HLO.

To verify which platform/config `Executor` actually chose without running the kernel, call
`executor.choose_config(op, *args, platform="triton", **kwargs)`.

## 3. What Triton Actually Emitted — `triton_call(..., debug=True)` & Disk Cache

`libs/ejkernel/ejkernel/callib/_triton_call.py` compiles directly via `code_gen.ast_to_ttir` ->
`compile_ttir_inplace`, bypassing `triton.compile`, `~/.triton/cache`, `TRITON_KERNEL_DUMP=1`, and `TRITON_INTERPRET=1`.

To inspect TTIR, TTGIR, LLIR, and PTX for an ejKernel Triton call, pass `debug=True` to `triton_call(...)` during a
single uncached compile (disable disk cache while inspecting):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
EJKERNEL_TRITON_CACHE_COMPILES=0 \
  uv run python <repro_calling_triton_call_with_debug_true.py> > /tmp/triton_ir.txt 2>&1
```

What to verify in the dumped IR:

- **TTGIR (`#ttg` / `#mma`):** confirm `tl.dot` lowered to tensor-core MMA layout (`#ttg.nvidia_mma` / `#mma`), not
  scalar FMA; confirm software pipelining (`num_stages`) emitted `async_copy_global_to_local` / `cp.async`.
- **PTX:** check `.local` loads/stores (register spills — `.local` traffic inside the inner loop usually wipes out
  tiling gains) and verify `shared_mem_bytes` fits comfortably on the target SM.

ejKernel Triton compilation disk-cache env vars (`libs/ejkernel/ejkernel/callib/_triton_call.py`):

- `EJKERNEL_TRITON_CACHE_COMPILES=0|1` (default `0` / disabled)
- `EJKERNEL_TRITON_CACHE_VERBOSE=0|1` (logs memory/disk hits and misses)
- `EJKERNEL_TRITON_CACHE_DIR` (defaults to `<cache_dir>/triton_kernels`)
- `EJKERNEL_TRITON_CACHE_MAX_ITEMS` (default `256` in-memory LRU cap)

Warning: `_triton_kernel_source_hash` hashes only `fn.fn`, not nested `@triton.jit` helper functions. Keep
`EJKERNEL_TRITON_CACHE_COMPILES=0` (or clear the cache dir) while iterating on kernel helpers.

## 4. Hardware Counters — `ncu` (NVIDIA) and `rocprof` (AMD ROCm)

Always disable XLA GPU command buffers and preallocation when profiling JAX under `ncu` so individual Triton kernels
appear cleanly and `ncu` does not OOM:

```bash
# List kernel launches first to get the exact Triton symbol name:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --list-kernels uv run python <script.py>

# Deep profile one steady-state launch (skip warmup launches):
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --set full -k "regex:<triton_fn_name>" --launch-skip 3 --launch-count 1 \
  -o /tmp/triton_ncu uv run python <script.py>
```

Read `ncu` sections in the exact order documented in
`.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md`: **Speed Of Light** -> **Memory Workload Analysis**
(sectors/request coalescing, L2 hit rate, shared-memory bank conflicts) -> **Occupancy** -> **Scheduler / Warp State
Statistics** -> **Source Counters** (spills).

On AMD ROCm (`HIPBackend`, HSACO, wave64), use `rocprof` / `rocprofv3` against the same `JAX_PLATFORMS=rocm,cpu`
script to capture kernel duration, VALU/MFMA utilization, LDS bank conflicts, and VGPR/SGPR spill counts.

## 5. JAX Timeline & Step Profiling

For launch-overhead, multi-kernel gaps, or XLA-vs-Triton attribution inside a larger JAX graph:

- Use `nsys` with the JAX flags in `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` (look for `triton_call`
  `name="ejkernel::triton::..."` / `triton_kernel_call` ranges).
- Use `ejkernel.ops.execution.profiler.Profiler` (`libs/ejkernel/ejkernel/ops/execution/profiler.py`) or
  `ejkernel.loggings.{ignite_profiler,extinguish_profiler,create_step_profiler}` (`libs/ejkernel/ejkernel/loggings.py`)
  with `EJKERNEL_OPS_STAMP=hash` (after at least one warmup call). See
  `.agents/skills/optimize-ejkernel-kernel/docs/reference/profiling.md` for details.

## 6. Reporting Checklist

- GPU model, CUDA/ROCm + Triton + JAX versions, clock-lock state
- Exact op, `platform="triton"`, shape, dtype, and chosen config (`block_*`, `num_warps`, `num_stages`, cache state)
- `benchmark_suite.py` JSON/Markdown artifact paths and steady-state median/mean/min vs `Platform.XLA` (and sibling GPU
  platforms if registered)
- `ncu`/`rocprof` SOL %s, top stall reason, coalescing/bank-conflict/spill readings
- `triton_call(debug=True)` TTGIR/PTX note when verifying `#mma` tensor-core lowering or `.local` spills
- Passing GPU `pytest` output under `jax.jit` (including boundary shapes) and `test_backend_autotune_guardrails.py`
