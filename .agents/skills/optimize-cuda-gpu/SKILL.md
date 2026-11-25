---
name: optimize-cuda-gpu
description: Optimize, profile, or diagnose a native CUDA C++ GPU kernel in ejKernel (JAX FFI). Use for libs/ejkernel/csrc/ and libs/ejkernel/ejkernel/kernels/_cuda/ performance work — CMake/code_gen.py/build_cudalib builds, JAX FFI bindings (xla/ffi/api/ffi.h), memory coalescing, occupancy and register/shared-memory budgets, bank conflicts, warp divergence, CUTLASS C++ tensor cores (mma/wmma), cp.async/TMA pipelining, cuobjdump/ncu/nsys/compute-sanitizer profiling on JAX GPU processes, or comparing Platform.CUDA against Platform.XLA / CuTe / Triton / TileLang.
---

# Skill: Optimize A CUDA C++ GPU Kernel (ejKernel / JAX FFI)

This is a GPU-backend specialization of `.agents/skills/optimize-ejkernel-kernel/SKILL.md` (which layers on
`.agents/skills/run-research/SKILL.md`). Use it when optimizing, profiling, or fixing regressions in ejKernel's native
CUDA C++ / CUTLASS C++ kernels (`libs/ejkernel/csrc/` + `libs/ejkernel/ejkernel/kernels/_cuda/`).

Choose the right backend skill:
- Use **this skill** (`optimize-cuda-gpu`) for the 5 native C++/CMake/JAX-FFI libraries under `libs/ejkernel/csrc/`
  (`flash_attention`, `blocksparse_attention`, `quantized_matmul`, `ragged_page_attention_v3`, `unified_attention`),
  or whenever you need the shared JAX GPU `nvidia-smi`/`ncu`/`nsys`/`compute-sanitizer` profiling guide.
- Use `.agents/skills/optimize-cute-gpu/SKILL.md` for Python-authored CUTLASS CuTe DSL kernels (`_cute/`).
- Use `.agents/skills/add-ejkernel-kernel/SKILL.md` when adding a brand-new operation or backend surface.

## Mental Model: Hardware + ejKernel JAX FFI Pipeline

### GPU Hardware Fundamentals

- **Execution.** Threads run in **warps of 32** lockstep inside thread blocks on **SMs**. Each SM partitions a fixed
  register file and shared-memory/L1 pool across resident blocks.
- **Memory hierarchy & coalescing.** Global HBM -> chip-wide L2 -> per-SM L1/shared -> per-thread registers. Each GPU
  memory sector is **32 bytes** (a 128-byte cache line = **4 sectors**). Consecutive warp threads touching consecutive
  aligned addresses minimize sectors/request (e.g. **4 sectors/request** for 32-bit loads, **2** for 16-bit, **16** for
  128-bit `float4`/`int4`); scattered access up to **32 sectors/request** causes up to 8x overfetch.
- **Shared memory banks.** Shared memory has **32 banks** (4-byte stride). Threads in a warp hitting different addresses
  in the same bank serialize; pad or swizzle SMEM layouts to remove conflicts.
- **Occupancy vs ILP.** Occupancy buys latency hiding, not speed by itself. A register-rich, high-ILP kernel can beat a
  high-occupancy one (Volkov). Never cap registers into local-memory spills.
- **Tensor cores & async copies.** Use `mma`/`wgmma` / CUTLASS C++ (`libs/ejkernel/csrc/cutlass/`) for GEMM-shaped
  loops, and `cp.async` (SM80+) / TMA (SM90+) with 2-4 stage pipelines to overlap global->shared transfers with compute.

### How ejKernel Builds & Calls CUDA C++ Kernels

- **Sources & codegen:**
  - `libs/ejkernel/csrc/{flash_attention,blocksparse_attention,quantized_matmul,ragged_page_attention_v3,unified_attention}/`
    each contain a `CMakeLists.txt` and `src/` (plus `hopper/` and `include/` in `flash_attention`).
  - `libs/ejkernel/csrc/cutlass/` is the CUTLASS C++ header submodule (overridable via `EJKERNEL_CUTLASS_INCLUDE`, used
    by `flash_attention` and `quantized_matmul`).
  - `libs/ejkernel/csrc/code_gen.py` (default `--sms 80,90,100,110,120`) regenerates per-kernel `.cu` template
    instantiation stubs. Re-run it after changing template dispatch tables or target SMs.
- **CMake build & shared libraries:**
  - Build all 5 libraries with `uv run python -m ejkernel.build_cudalib` (`libs/ejkernel/ejkernel/build_cudalib.py` ->
    `ejkernel.kernels._cuda.build_cuda_libs()`), or build one op via its `libs/ejkernel/ejkernel/kernels/_cuda/<op>/_build.py:build_cuda_lib()`.
  - Output lands in `libs/ejkernel/ejkernel/kernels/_cuda/<op>/_build/libejkernel_<name>_cuda_sm<arch>.so`
    (`libejkernel_qmm_cuda_sm<arch>.so` for `quantized_matmul`; `libejkernel_<op>_cuda_sm<arch>.so` for the other 4).
  - Verified build env vars: `EJKERNEL_CUDA_ARCH`, `EJKERNEL_CUDA_ARCHS`, `EJKERNEL_CUDA_BUILD_JOBS`,
    `EJKERNEL_CUTLASS_INCLUDE`.
  - **Build flags already set:** all 5 `CMakeLists.txt` pass `-lineinfo` and `--use_fast_math` (plus
    `--expt-relaxed-constexpr` in `flash_attention`). Note two implications: (1) `--use_fast_math` lowers accuracy of
    transcendentals (`exp`, `tanh`, `rsqrt`) vs `Platform.XLA`, so check numerical tolerances carefully; (2) `_build.py`
    runs CMake with `capture_output=True`, hiding compiler output — use `cuobjdump --dump-resource-usage` on the built
    `.so` or run `cmake --build <build_dir> --verbose` manually to inspect register/SMEM/spill counts.
- **JAX FFI contract:**
  - C++ entry points include `"xla/ffi/api/ffi.h"` and bind `AnyBuffer`, `Result`/`Ret<AnyBuffer>`, `.Attr<...>(...)`,
    and `.Ctx<PlatformStream<cudaStream_t>>()` via `xla::ffi::Ffi::Bind()`, returning `xla::ffi::Error`.
  - Python wrappers (`_cuda_impl*.py`) load the `.so` via `ctypes.CDLL`, register handlers with
    `jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(sym), platform="gpu", api_version=1)`, and call them with
    `jax.ffi.ffi_call` (inside `@ejit` / `jax.custom_vjp`).
  - **Hot-path FFI pitfall:** watch for `cudaMalloc`, `cudaFree`, `cudaMemcpyAsync(..., D2H)`, or
    `cudaStreamSynchronize` inside FFI handlers (e.g. scratch allocations or RNG state syncing) — these break CUDA-graph
    capture and stall the GPU pipeline.

## Critical ejKernel Contracts & Traps

1. **`Platform.CUTE` shadows `Platform.CUDA` by default.** In `detect_platform()` (`libs/ejkernel/ejkernel/modules/base.py`)
   and `Executor._prefer_cuda_cfg` (`libs/ejkernel/ejkernel/ops/execution/executor.py`), NVIDIA GPU priority is
   `cute > cuda > triton > xla`. Whenever a CuTe impl is registered for the op (`flash_attention`, `quantized_matmul`,
   `unified_attention`), calling the operation wrapper with `platform="auto"` runs CuTe, **not** CUDA C++. Always pass
   `platform="cuda"` (or `EJKERNEL_BENCH_PLATFORMS=xla,cuda`) when profiling or benchmarking CUDA C++.
2. **No backend-local autotuning.** `libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` bans runtime
   autotuning under `libs/ejkernel/ejkernel/kernels/`. Tune CUDA launch parameters/configs via
   `libs/ejkernel/ejkernel/modules/operations/<op>.py` (`heuristic_cfg_gpu`, `candidate_cfgs_gpu`, `Executor`).
3. **CMake cache invalidation.** `_build.py` auto-invalidates `CMakeCache.txt` when `nvcc` or `EJKERNEL_CUDA_ARCH(S)`
   changes and rebuilds when `csrc/<op>/` mtimes advance, but if you edit CMake flags directly or test `-Xptxas=-v`
   builds, delete `_build/CMakeCache.txt` or run `cmake --build` explicitly.

## Orient Before You Touch Code

Read the op wrapper, CUDA Python FFI wrapper, CMakeLists, and C++/CUDA kernels before editing:

- `libs/ejkernel/ejkernel/modules/operations/<op>.py`
- `libs/ejkernel/ejkernel/kernels/_cuda/<op>/{_interface.py,_cuda_impl*.py,_build.py}`
- `libs/ejkernel/csrc/<op>/{CMakeLists.txt,src/}` (and `hopper/` for `flash_attention`)
- `libs/ejkernel/ejkernel/kernels/_xla/<op>/` (reference)
- `libs/ejkernel/test/kernels/_cuda/test_<op>.py`

Inspect registers/SMEM/spills on the current `.so` up front with `cuobjdump --dump-resource-usage <lib.so>`, and check
the FFI handler for any hidden `cudaMalloc` or `cudaStreamSynchronize` calls. Freeze one shape and dtype.

## Establish A Baseline

1. Record GPU model (SM arch), CUDA toolkit/driver + JAX versions, shape, dtype, clock state, and build flags.
2. Lock GPU clocks (`nvidia-smi -lgc`; see `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md`).
3. Build and verify correctness vs `Platform.XLA` on the target GPU:
   ```bash
   uv run python -m ejkernel.build_cudalib
   env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
     uv run pytest libs/ejkernel/test/kernels/_cuda/test_<op>.py
   ```
4. Capture a steady-state baseline with `libs/ejkernel/benchmarks/benchmark_suite.py` comparing `xla,cuda` (plus
   `cute,triton,tilelang` when registered):
   ```bash
   ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
   EJKERNEL_AUTOTUNE_POLICY=heuristics \
   EJKERNEL_BENCH_OPS=<op> \
   EJKERNEL_BENCH_PLATFORMS=xla,cuda,cute,triton \
   EJKERNEL_BENCH_WARMUP=5 \
   EJKERNEL_BENCH_ITERS=30 \
   EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_cuda \
     uv run python libs/ejkernel/benchmarks/benchmark_suite.py
   ```

## Measure, Don't Guess

- **`cuobjdump --dump-resource-usage <lib.so>`** — immediate per-kernel register count, static SMEM, and local spill
  bytes on the compiled shared library without changing CMake.
- **Nsight Compute (`ncu`)** — per-kernel attribution on a JAX GPU process (with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  and `XLA_FLAGS='--xla_gpu_enable_command_buffer='`). Read **Speed Of Light** first, then **Memory Workload Analysis**
  (sectors/request, L1/L2 hit rate, bank conflicts), **Occupancy**, **Scheduler/Warp State Stats**, and **Source
  Counters** (`-lineinfo` is already enabled in every `CMakeLists.txt`).
- **Nsight Systems (`nsys`)** — timeline across JAX FFI calls, stream launches, `cudaMalloc`/D2H sync stalls, and
  multi-kernel launch overhead.
- **`compute-sanitizer`** (`memcheck`, `racecheck`, `synccheck`, `initcheck`) — correctness gate on the JAX GPU test
  before trusting any speedup.

`ncu` replays kernels; its absolute times are for **attribution**, never your benchmark numbers. Use `benchmark_suite.py`
/ `ejkernel.benchmarks.Benchmark` for timing. See `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` for
exact commands.

## Attribute The Bottleneck

- **Memory-bound** (memory % >> compute %, `Long Scoreboard` stalls, high sectors/request) → fix coalescing (align
  innermost thread index to contiguous dim), vectorize loads/stores (`int4`/`float4`/128-bit), stage reused tiles in
  SMEM, improve L2 tile ordering, cut redundant type casts or workspace round-trips.
- **Compute-bound** (compute % >> memory %) → engage tensor cores (`mma`/`wgmma`/CUTLASS), raise ILP, eliminate
  redundant math and SFU bottlenecks (`__exp2f` with pre-scaled `log2(e)`).
- **Bank-conflict-bound** (`ncu` shared-memory bank conflicts high) → swizzle or pad SMEM tiles, use `ldmatrix`.
- **Latency / occupancy / spill-bound** (neither unit saturated, "no eligible warp", or `.local` spills) → adjust
  `__launch_bounds__` / tile size / pipeline stages and re-check `cuobjdump --dump-resource-usage` so reducing registers
  does not introduce spills; reduce `__syncthreads()` and warp divergence.
- **Overhead / FFI-bound** (`nsys` shows gaps, `cudaMalloc`/`cudaStreamSynchronize`, or tiny kernels) → pass scratch
  buffers from JAX FFI instead of `cudaMalloc`, remove host syncs, fuse kernels, keep `XLA` command-buffer compatibility.

## Optimization Levers, Roughly By Impact

1. **Remove FFI host syncs & per-call `cudaMalloc`** — pass temporary buffers as JAX FFI outputs/scratch instead of
   allocating/freeing per call.
2. **Fix global access coalescing & vectorization** — target 4 sectors/request (32-bit) / 2 (16-bit) / 16 (`float4`
   128-bit) with aligned vectorized loads (`__restrict__`, `__ldg`).
3. **Tile through swizzled shared memory** — stage reused data once, eliminate 32-bank conflicts.
4. **Engage CUTLASS / tensor cores (`mma` / `wgmma`)** — avoid scalar inner loops on matmul-shaped work.
5. **Overlap with `cp.async` / TMA multi-stage pipelines** — 2-4 stages hiding HBM latency behind MMA, verified with
   `ncu` stall reduction.
6. **Tune register/occupancy balance** — `__launch_bounds__` / tile sizes verified via `cuobjdump` (zero new spills)
   and `ncu` SOL.
7. **Eliminate warp divergence & instruction overhead** — hoist uniform branches, `#pragma unroll` bounded loops,
   warp-level shuffles (`__shfl_xor_sync`).

## The Optimization Loop

1. Freeze one shape, dtype, SM arch, locked clock, and `platform="cuda"` baseline.
2. Attribute the bottleneck from `cuobjdump` + `ncu` SOL/stalls (and `nsys` for FFI overhead).
3. Change **one** lever (regenerate stubs via `libs/ejkernel/csrc/code_gen.py` if templates changed).
4. Rebuild (`uv run python -m ejkernel.build_cudalib` or per-op `_build.py`), run `compute-sanitizer` on the JAX GPU
   test, and run `pytest libs/ejkernel/test/kernels/_cuda/test_<op>.py` (including boundary/unaligned shapes and
   `jax.jit` / backward).
5. Re-run `benchmark_suite.py` under identical clock-locked conditions and re-profile to confirm the targeted limiter
   shrank.
6. Keep only if it wins cleanly and stays correct; otherwise revert and record why.

## Correctness & Hardware Verification Gate

Per `AGENTS.md` Golden Rule #4 and `libs/ejkernel/test/conftest.py`, all CUDA kernel tests and benchmarks must run on an
NVIDIA GPU (`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an NVIDIA GPU, report changes as **unverified on
hardware**.

Every kept change must pass:

- `compute-sanitizer` clean (`memcheck`, `racecheck`, `synccheck`, `initcheck`) on a JAX GPU run.
- Parity vs `Platform.XLA` (accounting for `--use_fast_math` tolerances, never masking accuracy regressions) across
  aligned and boundary/remainder shapes, under `jax.jit` (and backward when applicable).
- `uv run pytest libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` and `uv run lint-imports`.

## Definition Of Done

- GPU model (SM arch), CUDA/driver + JAX versions, build env vars, clock-lock state, shape, dtype, and `platform="cuda"`
  config are recorded.
- `compute-sanitizer` is clean and GPU parity tests vs `Platform.XLA` (including boundary shapes) pass.
- Resource usage (`cuobjdump --dump-resource-usage`) and `ncu` SOL + top stall reason name the bound.
- `benchmark_suite.py` JSON/Markdown artifact paths and steady-state timings (baseline vs candidate and vs `xla`/sibling
  GPU platforms) are reported under identical clock-locked conditions.
- A re-profile confirms the targeted stall/limiter improved; only the measured winning variant is kept.
