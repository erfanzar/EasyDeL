---
name: optimize-cute-gpu
description: Optimize, profile, tune, or diagnose a CUTLASS CuTe DSL GPU kernel in ejKernel (JAX TVM-FFI). Use for libs/ejkernel/ejkernel/kernels/_cute/ and ejkernel.callib.{_cute_call,_cute_ffi} performance work — CUTE_DSL_ENABLE_TVM_FFI=1, build_cute_ffi_call/cute_call, assumed_align=16 128-bit cp.async CopyG2SOp, SmemAllocator and swizzled staged SMEM layouts, MmaF16BF16Op/wgmma tensor-core atoms, LdMatrix s2r copies, QMM tier flags (EJKERNEL_CUTE_QMM_USE_NAIVE/TILED/MMA_SINGLE), and profiling Platform.CUTE against Platform.XLA / CUDA / Triton / TileLang.
---

# Skill: Optimize A CUTLASS CuTe DSL GPU Kernel (ejKernel / JAX TVM-FFI)

This is a GPU-backend specialization of `.agents/skills/optimize-ejkernel-kernel/SKILL.md` (which layers on
`.agents/skills/run-research/SKILL.md`). Use it when optimizing, profiling, or upgrading ejKernel's CUTLASS CuTe DSL
kernels (`libs/ejkernel/ejkernel/kernels/_cute/`) on NVIDIA GPUs.

`Platform.CUTE` is ejKernel's **highest-priority NVIDIA GPU backend** (`detect_platform` in
`libs/ejkernel/ejkernel/modules/base.py` and `Executor._prefer_cuda_cfg` in
`libs/ejkernel/ejkernel/ops/execution/executor.py` auto-select `cute > cuda > triton > xla`). Because any registered
CuTe kernel automatically shadows CUDA and Triton on NVIDIA GPUs, performance regressions or unoptimized scalar paths
in `_cute/` directly affect default GPU performance.

For adding a new operation or CuTe backend surface, see `.agents/skills/add-ejkernel-kernel/SKILL.md`. For native C++
CUDA shared libraries under `libs/ejkernel/csrc/`, see `.agents/skills/optimize-cuda-gpu/SKILL.md`.

## Mental Model: How ejKernel Lowers CuTe DSL To JAX

- **TVM-FFI bridge (`_cute_call.py` + `_cute_ffi.py`).** CuTe DSL kernels (`nvidia-cutlass-dsl`, `jax-tvm-ffi`,
  `apache-tvm-ffi` in `libs/ejkernel/pyproject.toml` `[cuda]` extra) are authored with `@cute.kernel` and `@cute.jit`
  host launchers and dispatched through `build_cute_ffi_call` (`libs/ejkernel/ejkernel/callib/_cute_ffi.py`) +
  `cute_call` (`libs/ejkernel/ejkernel/callib/_cute_call.py`).
- **TVM-FFI enablement.** `CUTE_DSL_ENABLE_TVM_FFI=1` is defaulted in `libs/ejkernel/ejkernel/__init__.py` and each
  CuTe `_cute_impl*.py`; `build_cute_ffi_call` defaults to `compile_options="--enable-tvm-ffi"`.
- **Launcher signature & stream auto-detection.** `_fn_expects_stream` inspects the `@cute.jit` launcher: when the
  first positional parameter is named `stream` (`stream: cuda.CUstream`), `_cute_ffi.py` compiles with
  `cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)` and forwards XLA's CUDA stream at runtime. Remaining
  positional args are **all input tensors followed by all output tensors** (preallocated by JAX). Use
  `input_output_aliases={in_idx: out_idx}` for in-place updates (e.g. KV cache scatter).
- **Symbolic compilation & 128-bit alignment contract.** `_fake_tensor_from_shaped` (`_cute_ffi.py`) builds fake
  tensors with `cute.runtime.make_fake_compact_tensor(cutlass_dtype, sym_shape, stride_order=..., assumed_align=16)`
  (16 bytes = 128 bits, required by MLIR verification for 128-bit `cp.async` `CopyG2SOp`; safe because JAX/XLA
  guarantees >=128-byte device buffer alignment). `_COMPILE_CACHE` keys on
  `(fn, input (rank, dtype), output (rank, dtype), compile_options, static_kwargs)`, so one compiled TVM-FFI target
  (`ejkernel_cute_tvm_ffi_<sha256[:24]>`) reuses across dynamic sizes when rank, dtype, and static kwargs stay fixed. Supported `_DTYPE_TO_CUTLASS` dtypes:
  `float16`, `bfloat16`, `float32`, `int8`, `uint8`, `int32`, `uint32`.
- **CuTe layout & hardware hierarchy.** Performance comes from expressing tiled hardware mappings explicitly:
  - **Tensor Cores:** `cute.nvgpu.warp.MmaF16BF16Op` (SM80+) / `wgmma` (SM90+) + `cute.make_tiled_mma` + `cute.gemm`
    accumulating in `cutlass.Float32`.
  - **Async GMEM->SMEM:** `cute.nvgpu.cpasync.CopyG2SOp(cache_mode=...)` (128-bit `cp.async`) with
    `cute.make_tiled_copy_tv` and multi-stage circular SMEM layouts (`cute.tile_to_shape` to `(row, col, num_stages)`),
    allocated via `cutlass.utils.SmemAllocator()`.
  - **Bank-conflict-free SMEM->Reg:** swizzled SMEM layouts + `LdMatrix` tiled copies (`tiled_copy_s2r_*`).
  - **Coalesced SMEM epilogue:** reuse SMEM (`cute.recast_ptr(sA.iterator, dtype=out_dtype)`) + `cute.autovec_copy`
    from accumulator fragments before coalesced GMEM stores.

## Current `_cute/` Kernel Landscape (Verified In-Repo)

Inspect `libs/ejkernel/ejkernel/kernels/_cute/` before assuming a kernel's structure — the 5 directories have very
different maturity levels:

1. **`quantized_matmul`** (registered; native CuTe DSL fwd + `jax.custom_vjp` bwd + `custom_vmap`) — primary CuTe
   optimization exemplar (`_cute_impl.py`). Implements a 4-tier kernel ladder in `_build_fused_qmm_host_fns`:
   - Default (when `_validate_mma_tile`: `tile_m % 32 == 0`, `tile_n % 32 == 0`, `tile_k % 16 == 0`): 3-stage
     pipelined Tensor Core MMA (`_build_mma_pipelined_qmm_host_fns`, `_MMA_PIPELINE_STAGES=3`, `MmaF16BF16Op`, 128-bit
     `CopyG2SOp`, `SmemAllocator`, swizzled 3D SMEM, `LdMatrix`).
   - Single-stage MMA (`_build_mma_qmm_host_fns`, forced with `EJKERNEL_CUTE_QMM_USE_MMA_SINGLE=1`).
   - Tiled SMEM + 4x4 register scalar kernel (`_build_tiled_qmm_host_fns`, forced with `EJKERNEL_CUTE_QMM_USE_TILED=1`).
   - Naive scalar fallback (`_build_naive_qmm_host_fns`, forced with `EJKERNEL_CUTE_QMM_USE_NAIVE=1`).
   - Supports `affine`, `nf4` (SMEM LUT), `mxfp4`, `mxfp8`, `nvfp4`, `nvfp8`.
2. **`flash_attention`** (registered; native CuTe DSL fwd + `jax.custom_vjp` bwd) — currently scalar thread-per-query
   online softmax (`_build_flash_host_fns`, `q_block` up to 256) and 3 recomputing backward kernels
   (`_flash_dq_kernel`, `_flash_dk_kernel`, `_flash_dv_kernel`, no saved LSE). High optimization headroom (SMEM tiling,
   `MmaF16BF16Op`/`wgmma`, `CopyG2SOp`/TMA, saved LSE).
3. **`chunked_prefill_paged_decode`** (registered; hybrid) — native CuTe DSL KV-cache scatter kernel
   (`_kv_update_kernel` using `input_output_aliases` and `EJKERNEL_CUTE_CPPD_THREADS`, default `256`) + delegates
   attention to Triton `unified_attention`.
4. **`unified_attention`** (registered; delegating wrapper) — currently delegates to Triton `unified_attention`
   (disableable via `EJKERNEL_CUTE_UNIFIED_ATTENTION_DISABLE_TRITON=1`); no native CuTe DSL attention body yet.
5. **`ragged_page_attention_v3`** (unregistered stub) — docstring-only placeholders in `_interface.py` /
   `_cute_impl_fwd.py`.

## Critical ejKernel Contracts & Footguns

1. **Silent fallback out of Tensor Core MMA.** In `quantized_matmul/_cute_impl.py`, `_build_fused_qmm_host_fns` silently
   falls back from 3-stage pipelined MMA -> single-stage MMA -> tiled scalar -> naive scalar if `_validate_mma_tile`
   fails (`tile_m` or `tile_n` not a multiple of 32, or `tile_k` not a multiple of 16) or if compilation throws an
   exception inside `try/except`. Always confirm which tier actually launched via `ncu --list-kernels` /
   `ncu --set full`!
2. **MLIR alignment verification on `CopyG2SOp`.** 128-bit `cp.async` requires `assumed_align=16` and leading-dimension
   divisibility `128 // dtype.width` on fake tensors (`_fake_tensor_from_shaped` in `_cute_ffi.py` sets
   `assumed_align=16`; custom `_fake_tensor` helpers must match or use `mark_compact_shape_dynamic(..., divisibility=...)`).
3. **No backend-local autotuning.** `libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` bans runtime
   autotuning under `libs/ejkernel/ejkernel/kernels/`. Wire tile configs (`block_m`, `block_n`, `block_k`, `q_blocksize`)
   through `libs/ejkernel/ejkernel/modules/operations/<op>.py` (`heuristic_cfg_gpu`, `candidate_cfgs_gpu`, `Executor`).
4. **JAX transform rules.** `cute_kernel_call_p` has no native JVP or `vmap` (`_cute_ffi.py` raises
   `NotImplementedError`). Wire `jax.custom_vjp` (never put JAX arrays in `nondiff_argnums`) and
   `jax.custom_batching.custom_vmap` (see `_wrap_vmap_compatible_jax_call` in `quantized_matmul/_cute_impl.py`).
5. **Multi-GPU & placement.** `cute_call` enforces single-device inputs and requires `jax.shard_map` on multi-GPU hosts
   (`_has_multi_accelerators()` asserts `_in_shard_map_context()`).
6. **Do not ship a CuTe kernel that is slower than sibling CUDA/Triton.** Because `Platform.CUTE` wins auto-selection on
   NVIDIA GPUs, always benchmark `xla,cute,cuda,triton,tilelang` side by side on the target shapes.
7. **No new `os.environ` knobs.** Use the existing diagnostic flags in `quantized_matmul` / `chunked_prefill_paged_decode`
   only for attribution; wire any new tuning knobs through config dataclasses or explicit args.

## Orient Before You Touch Code

Read the op wrapper, CuTe impl, XLA reference, and tests:

- `libs/ejkernel/ejkernel/modules/operations/<op>.py`
- `libs/ejkernel/ejkernel/kernels/_cute/<op>/{_interface.py,_cute_impl*.py}`
- `libs/ejkernel/ejkernel/callib/{_cute_call.py,_cute_ffi.py}`
- `libs/ejkernel/ejkernel/kernels/_xla/<op>/` (mandatory reference)
- `libs/ejkernel/test/kernels/_cute/test_<op>.py`

Check: whether the kernel is actually hitting Tensor Core MMA vs a scalar/tiled fallback, SMEM allocation & stage count,
`assumed_align=16` / `CopyG2SOp` usage, cache key granularity (`_QMM_CALL_CACHE` / `_COMPILE_CACHE`), and any dummy
buffer allocations or dtype casts wrapping `cute_call`. Freeze one representative shape and dtype.

## Establish A Baseline

1. Record GPU model (SM arch), CUDA + `nvidia-cutlass-dsl` + `jax-tvm-ffi` + JAX versions, clock state, shape, dtype,
   and tile/tier config.
2. Lock GPU clocks (`nvidia-smi -lgc`; see `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md`).
3. Verify correctness vs `Platform.XLA` on GPU:
   ```bash
   env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
     uv run pytest libs/ejkernel/test/kernels/_cute/test_<op>.py
   ```
4. Capture a steady-state baseline via `libs/ejkernel/benchmarks/benchmark_suite.py` comparing `cute` against `xla` and
   sibling NVIDIA backends (`cuda`, `triton`, `tilelang`):
   ```bash
   ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
   EJKERNEL_AUTOTUNE_POLICY=heuristics \
   EJKERNEL_BENCH_OPS=quantized_matmul \
   EJKERNEL_BENCH_PLATFORMS=xla,cute,cuda,triton,tilelang \
   EJKERNEL_BENCH_WARMUP=5 \
   EJKERNEL_BENCH_ITERS=30 \
   EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_cute \
     uv run python libs/ejkernel/benchmarks/benchmark_suite.py
   ```

## Measure, Don't Guess

- **`benchmark_suite.py` / `ejkernel.benchmarks.Benchmark`** — headline steady-state forward/backward timings vs `xla`,
  `cuda`, `triton`, and `tilelang`.
- **QMM tier A/B ablation** — compare `EJKERNEL_CUTE_QMM_USE_NAIVE=1`, `EJKERNEL_CUTE_QMM_USE_TILED=1`,
  `EJKERNEL_CUTE_QMM_USE_MMA_SINGLE=1`, and default 3-stage pipelined MMA under identical locked clocks to isolate
  exact gains per structural tier.
- **Nsight Compute (`ncu`)** on the JAX GPU process — read Speed Of Light (verify Tensor Core pipe utilization, not just
  FP32 FMA), `MemoryWorkloadAnalysis` (coalescing sectors/request, L2 hit rate, SMEM bank conflicts), `Occupancy`, and
  `WarpStateStats` / `SourceCounters` (register spills). Always pass `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_FLAGS='--xla_gpu_enable_command_buffer='`.
- **Nsight Systems (`nsys`) / JAX `Profiler`** — check whether `cute_call` cache misses, TVM-FFI dispatch, or surrounding
  XLA casts/dummy buffers add overhead.

See `.agents/skills/optimize-cute-gpu/docs/reference/profiling.md` for exact CuTe TVM-FFI inspection and tier-ablation
commands, and `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` for the shared `ncu`/`nsys`/`compute-sanitizer`
reference.

## Attribute The Bottleneck

- **Scalar / non-MMA execution** (`ncu` shows high FP32 FMA and ~0% Tensor Core pipe, or kernel name shows tiled/naive
  fallback) → align tile sizes to MMA constraints (`tile_m % 32 == 0`, `tile_n % 32 == 0`, `tile_k % 16 == 0`), adopt
  `MmaF16BF16Op`/`wgmma` + `cute.gemm`.
- **Memory-bound** (`ncu` memory % high, `Long Scoreboard` stalls) → vectorize global loads with 128-bit `CopyG2SOp`
  (`cp.async`) / TMA, widen tiles for higher arithmetic intensity, eliminate redundant global re-reads (e.g. dequantize
  into SMEM/registers once per tile; save LSE in flash attention instead of recomputing in backward).
- **Bank-conflict-bound** (`ncu` SMEM bank conflicts high) → use swizzled SMEM layouts (`_make_staged_smem_layout` /
  `cute.make_swizzle`) and `LdMatrix` SMEM->register copies.
- **Pipeline-starved** (`Long Scoreboard` / `Barrier` stalls between loads and MMA) → increase staged circular SMEM
  pipeline depth (`num_stages` = 2-4, watching SMEM budget) and overlap `cp.async` commit/wait groups with MMA +
  on-the-fly dequant.
- **Occupancy / spill-bound** (`ncu` shows `.local` spills or SMEM limit) → shrink tile sizes or stage count, reuse
  SMEM for the epilogue (`cute.recast_ptr`), trim live registers in dequant/unpack loops.

## Optimization Levers, Roughly By Impact

1. **Tensor-Core MMA atoms (`MmaF16BF16Op` / `wgmma`)** — move scalar thread loops (`_flash_kernel`, naive/tiled QMM)
   to warp/warpgroup MMA accumulating in `cutlass.Float32`.
2. **128-bit async `CopyG2SOp` + multi-stage SMEM pipeline** — overlap global loads with compute using 3-D staged
   swizzled SMEM layouts (`SmemAllocator`, `cute.tile_to_shape`).
3. **Swizzled SMEM + `LdMatrix` + SMEM-reusing epilogue** — eliminate shared-memory bank conflicts on both MMA operand
   loads and coalesced `cute.autovec_copy` output writeback.
4. **Tile shapes & operation-level tuning (`candidate_cfgs_gpu`)** — ensure every candidate satisfies MMA divisibility
   constraints and fits SMEM; expose tuned tiles via `libs/ejkernel/ejkernel/modules/operations/<op>.py`.
5. **Symbolic shape reuse & aliasing** — keep shapes symbolic via `cute.sym_int()` / `mark_layout_dynamic` so
   `_COMPILE_CACHE` hits across sequence lengths/batches; use `input_output_aliases` for in-place KV/state updates.

## The Optimization Loop

1. Freeze one shape, dtype, SM arch, locked clock, and `platform="cute"` baseline (plus `xla`/`cuda`/`triton` targets).
2. Attribute the bottleneck from `benchmark_suite.py` + `ncu` SOL/stalls (confirming which CuTe kernel tier actually
   launched).
3. Change **one** lever (kernel structure, MMA/pipeline atom, or one config axis).
4. Verify GPU parity vs `Platform.XLA` (including boundary/non-tile-aligned shapes, all quantization modes/bits or
   attention feature flags, `jax.jit`, `jax.vmap`, and `jax.grad` when applicable), run `compute-sanitizer`, and run
   `test_backend_autotune_guardrails.py`.
5. Re-run the identical benchmark command and re-profile to confirm the targeted limiter improved.
6. Keep only if it wins cleanly and stays correct; otherwise revert and record why.

## Correctness & Hardware Verification Gate

Per `AGENTS.md` Golden Rule #4 and `libs/ejkernel/test/conftest.py`, all CuTe kernel tests and benchmarks must run on an
NVIDIA GPU (`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without GPU hardware, report changes as **unverified on
hardware**.

Every kept change must pass:

- GPU parity vs `Platform.XLA` under `jax.jit` (and `jax.grad` / `jax.vmap` where supported), covering boundary shapes
  and every supported quantization mode / attention modifier.
- `compute-sanitizer` clean (`memcheck`, `racecheck`, `synccheck`, `initcheck`).
- `uv run pytest libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` and `uv run lint-imports`.

## Definition Of Done

- GPU model (SM arch), CUDA + `nvidia-cutlass-dsl` + `jax-tvm-ffi` + JAX versions, clock-lock state, shape, dtype,
  `platform="cute"`, and confirmed kernel tier/tile config are recorded.
- GPU parity vs `Platform.XLA` (including boundary shapes), `compute-sanitizer`, and
  `test_backend_autotune_guardrails.py` pass.
- Bottleneck class is named and backed by `ncu` SOL + Tensor Core / memory / stall readings.
- `benchmark_suite.py` JSON/Markdown artifact paths and steady-state timings (baseline vs candidate and vs `xla`,
  `cuda`, `triton`, `tilelang`) are reported under identical clock-locked conditions.
- A re-profile confirms the targeted limiter improved; only the measured winning variant is kept.
