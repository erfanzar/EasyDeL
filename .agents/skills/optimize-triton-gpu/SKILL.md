---
name: optimize-triton-gpu
description: Optimize, profile, tune, or diagnose an OpenAI Triton GPU kernel in ejKernel (JAX). Use for ejkernel.callib.triton_call performance work — block sizes (BLOCK_M/N/K), num_warps and num_stages pipelining, Executor/ConfigSelectorChain candidate_cfgs_gpu tuning, tl.dot tensor-core use, L2 program swizzling, masking and contiguity hints, triton_call(debug=True) TTIR/TTGIR/LLIR/PTX inspection, NVIDIA CUDA and AMD ROCm profiling, or diagnosing Triton regressions vs Platform.XLA / CuTe / CUDA / TileLang.
---

# Skill: Optimize A Triton GPU Kernel (ejKernel / JAX)

This is a GPU-backend specialization of `.agents/skills/optimize-ejkernel-kernel/SKILL.md` (which layers on
`.agents/skills/run-research/SKILL.md`). Use it when a Triton kernel under `libs/ejkernel/ejkernel/kernels/_triton/`
already produces correct results and the job is to make it faster, tune its `candidate_cfgs_gpu` / `heuristic_cfg_gpu`
space, explain a regression, or back a performance claim with profiler evidence.

For adding a new op or backend surface, use `.agents/skills/add-ejkernel-kernel/SKILL.md`. For `Executor` /
`ConfigSelectorChain` / cache debugging, use `.agents/skills/debug-ejkernel-ops/SKILL.md`.

## Mental Model: How ejKernel Lowers Triton To The GPU

- **Block programming via `triton_call`.** Each Triton program (`tl.program_id`) owns an output tile. In ejKernel,
  Triton kernels are invoked from JAX through `ejkernel.callib.triton_call`
  (`libs/ejkernel/ejkernel/callib/_triton_call.py`), which compiles `@triton.jit` functions via
  `code_gen.ast_to_ttir` -> `compile_ttir_inplace` (`CUDABackend` PTX on NVIDIA; `HIPBackend` HSACO, wave64, on AMD
  ROCm) and lowers to XLA custom call `"triton_kernel_call"` (`jax.ffi.ffi_lowering(..., api_version=2)`).
- **`num_warps` and `num_stages` dominate with block shape.** `num_warps` sets warps per CTA (default `4` in
  `triton_call`); `num_stages` sets software-pipeline depth over reduction loops (default `3`), overlapping global
  loads with `tl.dot` at the cost of shared memory. `num_ctas > 1` requires Hopper SM90+ (`compute_capability >= 90`).
- **`tl.dot` uses tensor cores** when inputs are `fp16`/`bf16`/`fp8` (never cast `tl.dot` operands to `tl.float32`;
  accumulate into `tl.float32` via `acc` / `out_dtype=tl.float32`).
- **Memory & L2 locality.** Coalescing along the innermost contiguous axis, `tl.make_block_ptr`, `tl.max_contiguous` /
  `tl.multiple_of`, `eviction_policy`, int64 pointer offsets for large KV caches (>4 GiB), and `GROUP_SIZE_M` L2
  program swizzling for GEMM-like grids directly reduce DRAM traffic and stalls.
- **No Hopper TMA descriptors yet in `triton_call`.** `libs/ejkernel/ejkernel/callib/_triton_call.py` does not yet wire
  `create_tma_descriptor_parameter`, so do not design optimizations around Triton TMA tensor descriptors until the
  callib bridge adds them.

If a change does not improve pipeline overlap, tensor-core (`tl.dot`) utilization, coalescing/L2 reuse, or remove
masking/register-spill/launch overhead, it will not help. Name the saturated resource first.

## Critical ejKernel Contracts & Footguns

Check these before touching kernel or operation code:

1. **No `@triton.autotune` inside `ejkernel/kernels/`.** Enforced by
   `libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py`: backend kernels must consume concrete configs
   chosen by operation executors (`libs/ejkernel/ejkernel/modules/operations/<op>.py`). Furthermore, `_triton_call.py`
   bypasses Triton's Python `Autotuner.run()` (ignoring `key=[...]`, rejecting `pre_hook`, and never firing
   `TRITON_PRINT_AUTOTUNING=1`). All tuning lives in `heuristic_cfg_gpu` / `candidate_cfgs_gpu` via `Executor` +
   `ConfigSelectorChain` + `PersistentCache`.
2. **NVIDIA GPU platform-override trap (`Executor._prefer_cuda_cfg`).** On NVIDIA GPUs, `detect_platform()`
   (`libs/ejkernel/ejkernel/modules/base.py`) resolves `cute > cuda > triton > xla`, and `Executor._prefer_cuda_cfg`
   (`libs/ejkernel/ejkernel/ops/execution/executor.py`) rewrites `cfg.platform` from `"auto"` or `"triton"` to
   `"cute"` or `"cuda"` after config selection whenever a CuTe/CUDA impl is registered (e.g. `flash_attention`,
   `quantized_matmul`, `ragged_page_attention_v3`, `unified_attention`, `blocksparse_attention`,
   `chunked_prefill_paged_decode`). **Always pass `platform="triton"`** (or an explicit `override_cfg` /
   `EJKERNEL_BENCH_PLATFORMS=xla,triton`) when profiling or benchmarking Triton through an operation wrapper.
3. **IR dumps and disk caching bypass `~/.triton/cache`.** Because `_triton_call.py` calls `code_gen.ast_to_ttir` and
   `make_ttir`/`make_ttgir`/`make_llir`/`make_ptx` directly, `TRITON_KERNEL_DUMP=1`, `TRITON_CACHE_DIR`, and
   `TRITON_INTERPRET=1` do not apply. Use `triton_call(..., debug=True)` to print TTIR/TTGIR/LLIR/PTX, and
   `EJKERNEL_TRITON_CACHE_{COMPILES,VERBOSE,DIR,MAX_ITEMS}` for ejKernel's disk cache (note: `_triton_kernel_source_hash`
   hashes only `fn.fn`, not helper `@triton.jit` functions — disable or clear the cache when editing helpers).
4. **`triton_call` scalar & heuristic traps (`_triton_call.py`):**
   - `None` is not accepted as a scalar arg (it falls into `array_args` and crashes lowering); use constexpr flags or
     explicit sentinel values.
   - `v == 1` specializes Python `True` AND float `1.0` (e.g. `softmax_scale=1.0`, `logits_soft_cap=1.0`) as integer
     constexpr `1`, and `get_triton_type` checks `int` before `bool`. Guard scalar parameters carefully.
   - `@triton.heuristics` receives `jax.core.ShapedArray` (abstract values), **not** `jax.Array` — never check
     `isinstance(args[...], jax.Array)` inside a heuristic lambda.
   - No `tl.tanh` in Triton — use overflow-safe `exp(-2|x|)` tanh math (portable across CUDA and ROCm).
   - Multi-GPU hosts require `jax.shard_map` (`_has_multi_accelerators()` asserts `_in_shard_map_context()`), and
     single-GPU non-zero device placement requires a matching `device=` argument.
   - No native JVP or `vmap`: wrap with `jax.custom_vjp` (never place JAX arrays in `nondiff_argnums` — that breaks
     `jax.jit(jax.grad(...))`) or `jax.custom_batching.custom_vmap`.
   - Pass `zeroed_outputs` only for buffers that actually require zero-init (e.g. atomic-add or `SPLIT_K > 1`), never
     unconditionally (avoids a wasted memset kernel launch); use `input_output_aliases` for in-place state/KV updates.
5. **Never sweep semantic or dead knobs in `candidate_cfgs_gpu`.** Only sweep performance-only knobs (`block_q`,
   `block_k`, `num_warps`, `num_stages`, split-K) that the Triton kernel actually consumes — never parameters that alter
   mathematical semantics (e.g. sparse block index granularity or pooling chunk windows). Respect
   `EJKERNEL_TRITON_SMEM_LIMIT` (default `99 * 1024` in `flash_attention.py` / `blocksparse_attention.py`) when pruning
   high-SMEM candidates.
6. **No new `os.environ` knobs.** Wire new tuning/options through config dataclasses or explicit parameters.

## Orient Before You Touch Code

Read the op wrapper, Triton kernel, XLA reference, and tests first:

- `libs/ejkernel/ejkernel/modules/operations/<op>.py` (`heuristic_cfg_gpu`, `candidate_cfgs_gpu`, config dataclass)
- `libs/ejkernel/ejkernel/kernels/_triton/<op>/{_interface.py,_triton_impl_fwd.py,_triton_impl_bwd.py}`
- `libs/ejkernel/ejkernel/kernels/_xla/<op>/` (mandatory reference)
- `libs/ejkernel/test/kernels/_triton/test_<op>.py`
- `libs/ejkernel/ejkernel/callib/_triton_call.py` (when diagnosing lowering, aliasing, or cache behavior)

Check the launch grid, `triton_call(..., name=...)`, `tl.dot` operand dtypes, any `tl.debug_barrier()` left in hot
loops, Python/XLA loops wrapping `triton_call` (e.g. per-token `lax.fori_loop` scatter or per-head Python loops), and
whether `heuristic_cfg_gpu` exists for the op. Freeze one representative shape and dtype for diagnosis.

## Establish A Baseline

1. Record GPU model (NVIDIA SM or AMD GFX arch), driver/CUDA/ROCm + Triton + JAX versions, shape, dtype, clock state,
   and `EJKERNEL_AUTOTUNE_POLICY` + persistent cache state (`~/ejkernel-presistent-cache/`).
2. Lock GPU clocks before timing (see `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md`).
3. Verify correctness against `Platform.XLA` on the target GPU, including under `jax.jit` (and `jax.jit(jax.grad(...))`
   for backward):
   ```bash
   env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
     uv run pytest libs/ejkernel/test/kernels/_triton/test_<op>.py
   ```
4. Capture a baseline via `libs/ejkernel/benchmarks/benchmark_suite.py` comparing `xla,triton` (and sibling GPU
   backends `cute,cuda,tilelang` when registered for the same op):
   ```bash
   ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
   EJKERNEL_AUTOTUNE_POLICY=heuristics \
   EJKERNEL_BENCH_OPS=<op> \
   EJKERNEL_BENCH_PLATFORMS=xla,triton \
   EJKERNEL_BENCH_WARMUP=5 \
   EJKERNEL_BENCH_ITERS=30 \
   EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_triton \
     uv run python libs/ejkernel/benchmarks/benchmark_suite.py
   ```

## Measure, Don't Guess

- **`benchmark_suite.py` / `ejkernel.benchmarks.Benchmark`** — headline steady-state forward (and backward) timings in
  JSON + Markdown against `Platform.XLA` and sibling GPU platforms.
- **Autotune & config inspection** — set `EJKERNEL_AUTOTUNE_POLICY=autotune EJKERNEL_LOG_AUTOTUNE=1` (and
  `EJKERNEL_AUTOTUNE_PROGRESS=1`) with a clean `EJKERNEL_PERSISTENT_CACHE_DIR` to verify which candidate won, or
  `EJKERNEL_OPS_STAMP=hash|json` to stamp XLA/profiler scopes.
- **`triton_call(..., debug=True)`** — prints TTIR, TTGIR (check `#mma` tensor-core layouts and pipelined async
  copies), LLIR, and PTX (check `.local` spill loads/stores).
- **Nsight Compute (`ncu`) / `nsys` (NVIDIA) or `rocprof` (AMD)** — read Speed Of Light (compute % vs memory %),
  `MemoryWorkloadAnalysis` (coalescing sectors/request, L2 hit rate, shared-memory bank conflicts), `Occupancy`, and
  `SourceCounters` (register spills). Run JAX with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and
  `XLA_FLAGS='--xla_gpu_enable_command_buffer='` so XLA command buffers do not obscure individual Triton launches.

See `.agents/skills/optimize-triton-gpu/docs/reference/profiling.md` for exact `triton_call(debug=True)`,
`benchmark_suite.py`, `Profiler`, `ncu`, and `rocprof` recipes, and
`.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md` for the shared clock-locking and `ncu`/`nsys` reading
guide.

## Attribute The Bottleneck

- **Memory-bound** (`ncu` memory % high, low arithmetic intensity) → improve coalescing (`tl.max_contiguous` /
  `tl.multiple_of` / `tl.make_block_ptr`), add `GROUP_SIZE_M` L2 swizzling, share KV across GQA query heads, tune
  `eviction_policy`, eliminate redundant XLA pre/post transposes or `lax.fori_loop` scatters.
- **Compute-bound** (compute % high, `tl.dot` saturated) → verify `#mma` tensor-core lowering in TTGIR (fix any
  accidental `tl.float32` operand cast before `tl.dot`), use `tl.math.exp2` with pre-scaled `scale * log2(e)`, skip
  masked/causal/sliding-window tiles, raise `num_stages` to keep tensor cores fed.
- **Pipeline-starved** (long-scoreboard stalls, loads not overlapping `tl.dot`) → raise `num_stages` (within
  `EJKERNEL_TRITON_SMEM_LIMIT` and register budget), remove any `tl.debug_barrier()`, restructure the K-loop.
- **Occupancy/spill-bound** (`ncu` / PTX shows `.local` spills or low occupancy) → shrink `BLOCK_M/N/K`, `num_stages`,
  or `num_warps`; register spills usually erase any tiling gain.
- **Overhead-bound** (`nsys` shows host/launch gaps or many tiny kernels) → fuse Python/XLA loops into the Triton grid,
  remove unnecessary `zeroed_outputs` memsets, avoid host syncs (`jax.device_get` / `.item()`).

## Optimization Levers, Roughly By Impact

1. **Eliminate host/XLA wrapper overhead & algorithmic waste** — fuse per-token/per-head loops into the grid, skip
   out-of-window/causal KV blocks, share KV tiles across GQA heads, reduce GQA gradients in-kernel instead of expanding
   to `num_q_heads` in fp32.
2. **Engage FP16/BF16 Tensor Cores (`tl.dot`)** — keep `tl.dot` operands in fp16/bf16 with fp32 accumulation; verify
   `#mma` in TTGIR.
3. **Block sizes (`BLOCK_M/N/K`), `num_stages`, and `num_warps`** — wire concrete, SMEM-checked candidates into
   `candidate_cfgs_gpu` and sane defaults into `heuristic_cfg_gpu` in `libs/ejkernel/ejkernel/modules/operations/<op>.py`.
4. **L2 program swizzling (`GROUP_SIZE_M`) & memory hints** — reorder GEMM/QMM CTA grids for L2 operand reuse; add
   `tl.make_block_ptr`, `tl.max_contiguous`, `tl.multiple_of`, and `eviction_policy`.
5. **Aliasing & zero-init hygiene** — use `input_output_aliases` for in-place KV/state updates and `zeroed_outputs` only
   when actually needed.

## The Optimization Loop

1. Freeze one shape, dtype, GPU, locked clock, and `platform="triton"` baseline.
2. Attribute the bottleneck via `benchmark_suite.py` + `ncu` SOL + `triton_call(..., debug=True)` IR dumps.
3. Change **one** lever (kernel structure or one axis of `candidate_cfgs_gpu` / `heuristic_cfg_gpu`).
4. Verify correctness on GPU under `jax.jit` (and `jax.grad` when differentiable) vs `Platform.XLA`, including
   non-power-of-2 / boundary shapes and feature flags (`causal`, `sliding_window`, `logits_soft_cap`, `softmax_aux`,
   GQA, varlen), plus run `test_backend_autotune_guardrails.py`.
5. Re-run the identical benchmark command (with fresh cache when re-tuning) and re-profile to confirm the targeted
   limiter improved.
6. Keep only if it wins cleanly and stays correct; otherwise revert and record why.

## Correctness & Hardware Verification Gate

Per `AGENTS.md` Golden Rule #4 and `libs/ejkernel/test/conftest.py`, all JAX/Triton computation tests and benchmarks
must run on a real GPU (`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without GPU hardware, run static checks only
(`libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` with `JAX_PLATFORMS=cpu`) and report kernel behavior
and performance as **unverified on hardware**.

Every kept change must pass:

- Forward (and full `dq`/`dk`/`dv` backward when applicable) parity vs `Platform.XLA` under `jax.jit`.
- Boundary / non-power-of-2 shapes (`seq_len`, `head_dim`, GQA ratio) exercising masking paths.
- `uv run pytest libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py` and `uv run lint-imports`.

## Definition Of Done

- GPU model, CUDA/ROCm + Triton + JAX versions, clock-lock state, shape, dtype, and `platform="triton"` config are
  recorded.
- Parity vs `Platform.XLA` passes on GPU under `jax.jit` (including boundary shapes and backward if supported), and
  `test_backend_autotune_guardrails.py` passes.
- Bottleneck class is named and backed by `ncu`/`rocprof` and/or `triton_call(debug=True)` IR evidence.
- Baseline vs candidate `benchmark_suite.py` JSON/Markdown paths and steady-state timings (and comparison vs `xla` /
  sibling GPU platforms) are reported under identical clock-locked conditions.
- A re-profile confirms the targeted limiter improved; only the measured winning change is kept.
