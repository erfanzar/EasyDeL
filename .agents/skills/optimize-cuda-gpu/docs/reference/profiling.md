# CUDA & Shared JAX GPU Profiling Cheatsheet (ejKernel)

Exact commands, build inspection, `ncu` sections/metrics, `nsys`, and `compute-sanitizer` recipes for
`.agents/skills/optimize-cuda-gpu/SKILL.md`, and the shared NVIDIA GPU profiling reference linked by
`.agents/skills/optimize-triton-gpu/SKILL.md`, `.agents/skills/optimize-cute-gpu/SKILL.md`, and
`.agents/skills/optimize-tilelang-gpu/SKILL.md`.

Two rules govern every GPU measurement: **lock clocks before timing**, and **`ncu` is for attribution, while
`benchmark_suite.py` / `ejkernel.benchmarks.Benchmark` are for timing** (`ncu` replays kernels and inflates elapsed
time).

## 0. Stabilize GPU Clocks & Standard JAX Profiling Environment

```bash
sudo nvidia-smi -pm 1                       # persistence mode on
nvidia-smi -q -d SUPPORTED_CLOCKS           # list lockable graphics/memory clocks
sudo nvidia-smi -lgc <gfx_MHz>              # lock graphics clock
sudo nvidia-smi -lmc <mem_MHz>              # lock memory clock (where supported)
# ... run baseline & candidate ...
sudo nvidia-smi -rgc && sudo nvidia-smi -rmc  # reset clocks when done
```

Standard JAX GPU env for **testing & benchmarking** (matches `AGENTS.md` Golden Rule #4):

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest libs/ejkernel/test/kernels/_cuda/test_<op>.py
```

Standard JAX GPU env for **`ncu`, `nsys`, and `compute-sanitizer`** (mirrors `libs/ejkernel/test/conftest.py` so XLA
does not preallocate all VRAM or hide individual launches inside CUDA command buffers):

```bash
ENABLE_DISTRIBUTED_INIT=0 \
JAX_PLATFORMS=cuda,cpu \
JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics
```

## 1. Build & Inspect Registers / Shared Memory / Spills (ejKernel CMake + FFI)

All 5 `libs/ejkernel/csrc/*/CMakeLists.txt` already compile CUDA with `-lineinfo` and `--use_fast_math` (plus
`--expt-relaxed-constexpr` in `flash_attention`). Build all CUDA shared libraries or one op:

```bash
# Regenerate template instantiation stubs if you edited template tables or target SMs:
uv run python libs/ejkernel/csrc/code_gen.py --sms 80,90,100,110,120

# Build all 5 CUDA .so libraries (env overrides: EJKERNEL_CUDA_ARCH, EJKERNEL_CUDA_ARCHS,
# EJKERNEL_CUDA_BUILD_JOBS, EJKERNEL_CUTLASS_INCLUDE):
EJKERNEL_CUDA_ARCH=90 EJKERNEL_CUDA_BUILD_JOBS=8 \
  uv run python -m ejkernel.build_cudalib
```

Built `.so` paths live under `libs/ejkernel/ejkernel/kernels/_cuda/<op>/_build/`:
- `libejkernel_flash_attention_cuda_sm<arch>.so`
- `libejkernel_blocksparse_attention_cuda_sm<arch>.so`
- `libejkernel_qmm_cuda_sm<arch>.so`
- `libejkernel_ragged_page_attention_v3_cuda_sm<arch>.so`
- `libejkernel_unified_attention_cuda_sm<arch>.so`

Because `_build.py` runs CMake with `capture_output=True`, inspect per-kernel registers, shared memory, and local
spills directly on the compiled `.so` with `cuobjdump`:

```bash
cuobjdump --dump-resource-usage \
  libs/ejkernel/ejkernel/kernels/_cuda/quantized_matmul/_build/libejkernel_qmm_cuda_sm90.so
```

To see live `ptxas` messages (`-Xptxas=-v`) during a single-op rebuild, invoke CMake directly on the op's `_build`
directory with verbose output:

```bash
cmake --build libs/ejkernel/ejkernel/kernels/_cuda/quantized_matmul/_build --verbose
```

Watch `stack frame` / `spill stores` / `spill loads` after any `__launch_bounds__` or tile change.

## 2. Timing — `benchmark_suite.py` & `ejkernel.benchmarks.Benchmark`

Always force `platform="cuda"` (or `EJKERNEL_BENCH_PLATFORMS=xla,cuda,...`) so `Executor._prefer_cuda_cfg` /
`detect_platform` does not route `auto` to `Platform.CUTE`:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
EJKERNEL_BENCH_OPS=quantized_matmul \
EJKERNEL_BENCH_PLATFORMS=xla,cuda,cute,triton,tilelang \
EJKERNEL_BENCH_CONFIG_LIMIT=3 \
EJKERNEL_BENCH_WARMUP=5 \
EJKERNEL_BENCH_ITERS=30 \
EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_cuda \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

`Benchmark` (`libs/ejkernel/ejkernel/benchmarks.py`) JIT-compiles, warms up, synchronizes with `jax.block_until_ready`,
and reports `mean_ms`, `median_ms`, `min_ms`, `max_ms`, and `std_ms` (plus `*_bwd` when `bench_bwd=True`). Never
divide total elapsed time by `ITERS` and mislabel the arithmetic mean as median or min; collect per-iteration samples
with `jax.block_until_ready()` if writing a custom probe.

## 3. Nsight Compute (`ncu`) — Per-Kernel Attribution on JAX

```bash
# 1) Discover exact mangled kernel names in your JAX script:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --list-kernels uv run python <probe.py>

# 2) Full profile of 1 steady-state launch (skip warmup launches):
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --set full -k "regex:<kernel_regex>" --launch-skip 3 --launch-count 1 \
  -o /tmp/ejk_ncu uv run python <probe.py>

# 3) Quick targeted SOL + Memory sections:
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
  -k "regex:<kernel_regex>" --launch-skip 3 --launch-count 1 \
  uv run python <probe.py>
```

Read `ncu` sections in this order:

1. **GPU Speed Of Light Throughput (`SpeedOfLight`)** — compare **Compute (SM) %** vs **Memory %** of peak and check the
   roofline chart. The higher percentage names the primary bound.
2. **Memory Workload Analysis (`MemoryWorkloadAnalysis`)** — check **sectors/request**, L1/L2 hit rates, and
   **shared-memory bank conflicts**.
   - Hardware rule: one NVIDIA GPU memory sector is **32 bytes**, and one cache line is **128 bytes = 4 sectors**.
   - Perfect coalescing for a 32-thread warp:
     - 16-bit (`fp16`/`bf16`, 64 B/warp) = **2 sectors/request**
     - 32-bit (`float`/`int32`, 128 B/warp) = **4 sectors/request**
     - 64-bit (`double`/`int64`, 256 B/warp) = **8 sectors/request**
     - 128-bit (`float4`/`int4`, 512 B/warp) = **16 sectors/request**
   - **32 sectors/request** on 32-bit loads means every thread touched a separate 32-byte sector (1024 B fetched for
     128 B used = **8x overfetch**, worst-case scattered access).
3. **Occupancy (`Occupancy`)** — achieved vs theoretical occupancy and the limiting resource (registers, shared memory,
   or block size).
4. **Scheduler / Warp State Statistics (`SchedulerStats`, `WarpStateStats`)** — top stall reasons (`Long Scoreboard` =
   waiting on global/L2 loads, `Short Scoreboard` = SMEM/local/SFU, `MIO Throttle`, `Barrier`, `Math Pipe Throttle`).
5. **Source Counters (`SourceCounters`)** — per-line stalls, branch divergence, and local-memory traffic (spills),
   correlated to `.cu` lines via `-lineinfo`.

Useful targeted metrics (`ncu --metrics ...`):

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed              # compute SOL %
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed        # DRAM SOL %
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                # global load sectors
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum               # global load requests (ratio = sectors/req)
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum      # shared-memory load bank conflicts
launch__registers_per_thread                                  # registers/thread
smsp__sass_average_branch_targets_threads_uniform.pct         # branch uniformity %
```

## 4. Nsight Systems (`nsys`) & JAX Profiler — Timeline / FFI Overhead

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
EJKERNEL_AUTOTUNE_POLICY=heuristics \
  nsys profile --trace=cuda,nvtx,osrt --stats=true -o /tmp/ejk_nsys \
  uv run python <probe.py>
```

Check for: gaps between launches (Python/host dispatch overhead), `cudaMalloc`/`cudaFree`/`cudaStreamSynchronize`
inside FFI handlers, unexpected H2D/D2H copies, and whether the CUDA kernel itself dominates total step time.

For JAX-level trace inspection (XLA scopes + `EJKERNEL_OPS_STAMP=hash|json`), use
`ejkernel.ops.execution.profiler.Profiler` (`libs/ejkernel/ejkernel/ops/execution/profiler.py`) or
`ejkernel.loggings.{ignite_profiler,extinguish_profiler,create_step_profiler}` (`libs/ejkernel/ejkernel/loggings.py`).

## 5. Correctness — `compute-sanitizer` on JAX GPU Tests

Run `compute-sanitizer` on a focused JAX GPU test **before** trusting any speedup (disable preallocation and command
buffers):

```bash
for tool in memcheck racecheck synccheck initcheck; do
  ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
    compute-sanitizer --tool "$tool" \
    uv run pytest libs/ejkernel/test/kernels/_cuda/test_<op>.py -k "<case>"
done
```

- `memcheck`: out-of-bounds / misaligned global accesses and invalid device pointers.
- `racecheck`: shared-memory read/write hazards (missing `__syncthreads()`).
- `synccheck`: illegal divergent `__syncthreads()` / barrier usage.
- `initcheck`: reads of uninitialized global device memory.

## 6. Reporting Checklist

- GPU model (SM arch), CUDA toolkit/driver + JAX versions, clock-lock state, build env vars
- Exact op, `platform="cuda"`, shape, dtype, launch config (grid/block/SMEM)
- Registers/thread, shared mem/block, and spill bytes from `cuobjdump --dump-resource-usage`
- `ncu` SOL Compute % vs Memory %, sectors/request, bank conflicts, and top stall reason
- `benchmark_suite.py` JSON/Markdown artifact paths with steady-state median/mean/min (baseline vs candidate and vs
  `Platform.XLA` / sibling GPU backends) under identical clock-locked conditions
- `compute-sanitizer` clean and GPU `pytest` parity passing (including boundary shapes)
- Re-profile showing the targeted limiter improved
