---
name: optimize-tilelang-gpu
description: Optimize, profile, tune, or diagnose a TileLang GPU kernel in ejKernel (JAX TVM-FFI). Use for libs/ejkernel/ejkernel/kernels/_tilelang/ performance work — @T.prim_func out_idx=None TVM-FFI contracts, build_tilelang_call/tilelang_call, tile sizes (block_M/N/K), T.Pipelined num_stages vs T.Serial, thread counts, shared/fragment memory scopes, T.annotate_layout and T.use_swizzle, T.gemm tensor-core use, generated CUDA inspection, test_honesty.py AST guardrails, and benchmark_suite.py/ncu profiling vs Platform.XLA.
---

# Skill: Optimize A TileLang GPU Kernel (ejKernel / JAX TVM-FFI)

This is a GPU-backend specialization of `.agents/skills/optimize-ejkernel-kernel/SKILL.md` (which layers on
`.agents/skills/run-research/SKILL.md`). Use it when a TileLang kernel under
`libs/ejkernel/ejkernel/kernels/_tilelang/` already produces correct results and the job is to make it faster, tune
operation-level `candidate_cfgs_gpu` / `heuristic_cfg_gpu`, explain a regression, or back a performance claim with
profiler evidence.

For adding a new operation or TileLang kernel, see `.agents/skills/add-ejkernel-kernel/SKILL.md`. For `Executor` /
cache debugging, see `.agents/skills/debug-ejkernel-ops/SKILL.md`.

## Mental Model: How ejKernel Lowers TileLang To The GPU

- **TVM-FFI bridge (`_tilelang_call.py` + `_tilelang_ffi.py`).** In ejKernel, TileLang kernels are authored as
  `@T.prim_func` factories and lowered through `build_tilelang_call` / `tilelang_call`
  (`libs/ejkernel/ejkernel/callib/_tilelang_call.py`) and `build_tilelang_ffi_call`
  (`libs/ejkernel/ejkernel/callib/_tilelang_ffi.py`). Internally, `_tilelang_ffi.py` compiles with
  `tilelang.compile(prim_func, out_idx=None, execution_backend="tvm_ffi", ...)`, registers `compiled.adapter.func` via
  `jax_tvm_ffi.register_ffi_target`, and dispatches through `jax.ffi.ffi_call`.
- **Mandatory buffer contract (`out_idx=None`).** Every `@T.prim_func` must list **all** buffers explicitly — **inputs
  first, then outputs** — because JAX preallocates output buffers and passes them by reference. Never pass
  `out_idx=[-1]` or use `@tilelang.jit`. Use `input_output_aliases={in_idx: out_idx}` for in-place updates (e.g. KV
  cache writes).
- **Explicit memory scopes.** Inside `with T.Kernel(grid..., threads=N):`, you place tiles explicitly in
  `T.alloc_shared` (SMEM), `T.alloc_fragment` (register/MMA fragments), and `T.alloc_local` (per-thread registers).
  Keeping hot operands and accumulators in shared/fragment instead of re-reading global memory is the primary structural
  lever.
- **Compute & movement primitives.** `T.copy` moves tiles between scopes (auto-vectorized, `cp.async` / Hopper TMA);
  `T.gemm` issues tensor-core MMA accumulating into a fragment (typically `float32`); `T.Parallel` distributes loops
  across CTA threads; `T.reduce_max` / `T.reduce_sum` perform cross-thread reductions; `T.exp2` with pre-scaled
  `scale * log2(e)` hits fast SFU exp2.
- **Pipelining (`T.Pipelined` vs `T.Serial`).** `T.Pipelined(count, num_stages=S)` double-/multi-buffers SMEM to
  overlap async `T.copy` with `T.gemm` (e.g. `flash_attention/_kernel.py`, `ragged_page_attention_v3/_kernel.py`).
  However, for streaming reduction loops where IR-level pipeline unrolling blows up compile-time RAM, `T.Serial` is
  preferred — TileLang still vectorizes `T.copy` via `cp.async` (see `fused_cross_entropy/_kernel.py`).
- **Layout & L2 swizzling.** `T.annotate_layout` sets swizzled SMEM layouts to kill 32-bank conflicts; `T.use_swizzle`
  reorders CTA rasterization to raise L2 hit rate on GEMM-like workloads.

If a change does not improve scope residency, pipeline overlap, tensor-core (`T.gemm`) utilization, bank-conflict / L2
behavior, or occupancy, it will not help. Name the saturated resource first.

## Critical ejKernel Contracts & Guardrails

Check every one of these before editing `_tilelang/` kernels or operation configs:

1. **`detect_platform()` NEVER auto-selects TileLang.** In `libs/ejkernel/ejkernel/modules/base.py`, NVIDIA GPU auto
   priority is `cute > cuda > triton > xla`. TileLang is selected only when the caller passes `platform="tilelang"` or
   `prefer_tilelang=True` (or when `candidate_cfgs_gpu` benchmarks `"tilelang"` under `EJKERNEL_AUTOTUNE_POLICY=autotune`).
   Always pass `platform="tilelang"` or `EJKERNEL_BENCH_PLATFORMS=xla,tilelang` when profiling or benchmarking.
2. **No backend-local autotuning inside `ejkernel/kernels/`.** Enforced by
   `libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py`: backend kernels under
   `libs/ejkernel/ejkernel/kernels/_tilelang/` MUST NOT call `autotune_tilelang_ffi`, `@tilelang.autotune`,
   `@autotune(`, `_AUTOTUNE_CACHE`, or read `EJKERNEL_*AUTOTUNE*`. Production kernels consume concrete tile/stage/thread
   configs supplied by `libs/ejkernel/ejkernel/modules/operations/<op>.py` (`Executor` + `ConfigSelectorChain` +
   `candidate_cfgs_gpu` / `heuristic_cfg_gpu`). (While `build_tilelang_call(..., configs=...)` / `autotune_tilelang_ffi`
   exist in `callib/` for ad-hoc scratch prototypes outside `ejkernel/kernels/`, never wire them into kernel files.)
3. **AST honesty guardrail (`test_honesty.py`).** `libs/ejkernel/test/kernels/_tilelang/test_honesty.py` (collected and
   enforced even without GPU!) fails CI if:
   - Any `_tilelang/**/_interface.py` uses `del <arg>` (must honor all arguments or raise explicitly).
   - Any `_tilelang/**/{_interface,_impl}.py` introduces new JAX-side compute/orchestration calls from
     `_BANNED_JAX_CALLS` (`jnp.einsum`, `jnp.exp`, `jnp.sum`, `jnp.where`, `jax.lax.scan`, `.at[]`, etc.) outside the
     shrinking `_KNOWN_JAX_COMPUTE` backlog. Do computation inside `@T.prim_func`, not in JAX glue!
4. **Registry coverage guardrail (`test_registry_coverage.py`).**
   `libs/ejkernel/test/kernels/_tilelang/test_registry_coverage.py` verifies every `Platform.XLA` op has a
   `Platform.TILELANG` registration and standalone test calling `_tl("<algorithm>")`.
5. **Compile flags & JAX transform rules:**
   - Modern CUDA wheels generally require `compile_flags=("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)` (already
     `_DEFAULT_COMPILE_FLAGS` across `_tilelang/` impls; overridable via `EJKERNEL_TILELANG_COMPILE_FLAGS`, and target
     via `EJKERNEL_TILELANG_TARGET`).
   - Multi-GPU hosts require `jax.shard_map` (`_tilelang_call.py` asserts `_in_shard_map_context()`).
   - No native JVP or `vmap` (`_tilelang_ffi.py` raises `NotImplementedError`): wire `jax.custom_vjp` (with separate
     backward `@T.prim_func`s, e.g. `flash_attention/_impl.py`) or `jax.custom_batching.custom_vmap`.
   - `_tilelang_ffi.py` caches compiled kernels on `(id(prim_func), in_shapes_dtypes, out_shapes_dtypes, target, ...)`,
     so keep a process-local cache keyed on static shape/tile/feature tuples in `_impl.py` so identical shapes reuse the
     same `prim_func` and compiled FFI target instead of recompiling on every call.

## Orient Before You Touch Code

Read the op wrapper, TileLang kernel/impl, XLA reference, and tests:

- `libs/ejkernel/ejkernel/modules/operations/<op>.py` (`heuristic_cfg_gpu`, `candidate_cfgs_gpu`)
- `libs/ejkernel/ejkernel/kernels/_tilelang/<op>/{_interface.py,_impl.py,_kernel.py}`
- `libs/ejkernel/ejkernel/kernels/_xla/<op>/` (mandatory reference)
- `libs/ejkernel/test/kernels/_tilelang/test_<op>.py`

Check: `T.Kernel` grid and `threads`, tile constants (`BM/BN/BK`), total `T.alloc_shared` bytes (including `num_stages`
multi-buffering), whether the inner loop uses `T.Pipelined` or `T.Serial`, `T.gemm` vs scalar loops, layout
transposes in `_impl.py` (e.g. `(B, N, H, D)` -> `(B, H, N, D)`), and whether any feature path falls back to an
unoptimized path. Freeze one representative shape and dtype.

## Establish A Baseline

1. Record GPU model (SM arch), CUDA + TileLang (`tilelang==0.1.9` in `libs/ejkernel/pyproject.toml`) + JAX versions,
   clock state, shape, dtype, tile sizes, `threads`, and `num_stages`.
2. Lock GPU clocks (`nvidia-smi -lgc`; see `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md`).
3. Verify correctness vs `Platform.XLA` on GPU plus static AST guardrails:
   ```bash
   env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
     uv run pytest libs/ejkernel/test/kernels/_tilelang/test_<op>.py \
                   libs/ejkernel/test/kernels/_tilelang/test_honesty.py \
                   libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py
   ```
4. Capture a steady-state baseline via `libs/ejkernel/benchmarks/benchmark_suite.py` comparing `xla,tilelang` (and
   sibling GPU platforms `cute,cuda,triton` when registered):
   ```bash
   ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
   EJKERNEL_AUTOTUNE_POLICY=heuristics \
   EJKERNEL_BENCH_OPS=<op> \
   EJKERNEL_BENCH_PLATFORMS=xla,tilelang,triton,cute,cuda \
   EJKERNEL_BENCH_WARMUP=5 \
   EJKERNEL_BENCH_ITERS=30 \
   EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_tilelang \
     uv run python libs/ejkernel/benchmarks/benchmark_suite.py
   ```

## Measure, Don't Guess

- **`benchmark_suite.py` / `ejkernel.benchmarks.Benchmark`** — headline steady-state forward/backward timings vs
  `Platform.XLA` and sibling GPU platforms.
- **Generated CUDA source** — call `entry.kernel.get_kernel_source()` from `ejkernel.callib._tilelang_ffi._COMPILE_CACHE`
  (or `tilelang.compile(prim_func, out_idx=None, execution_backend="tvm_ffi", compile_flags=["-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"]).get_kernel_source()`)
  to verify `T.gemm` lowered to tensor-core MMA, `T.copy` vectorized / used `cp.async` or TMA, and SMEM swizzles
  lowered cleanly.
- **Nsight Compute (`ncu`)** on the JAX GPU process — ground truth for Speed Of Light, coalescing sectors/request,
  shared-memory bank conflicts, occupancy limiters, and register spills (disable XLA command buffers and preallocation;
  see `.agents/skills/optimize-cuda-gpu/docs/reference/profiling.md`).
- **Nsight Systems (`nsys`) / JAX `Profiler`** — check whether `_impl.py` layout transposes, multiple launches, or host
  dispatch overhead dominate.

See `.agents/skills/optimize-tilelang-gpu/docs/reference/profiling.md` for exact `get_kernel_source()`,
`benchmark_suite.py`, `EJKERNEL_TILELANG_*`, and `ncu` recipes.

## Attribute The Bottleneck

- **Memory-bound** (`ncu` memory % high, low arithmetic intensity) → increase tile reuse (`BM/BN/BK`), move hot buffers
  from global into `T.alloc_shared` / `T.alloc_fragment`, add `T.use_swizzle` for L2 locality, ensure contiguous
  `T.copy` accesses, eliminate redundant layout transposes in `_impl.py`.
- **Compute-bound** (compute % high, `T.gemm` tensor cores saturated) → confirm tensor-core MMA in generated CUDA,
  use `T.exp2` with pre-scaled `log2(e)`, skip causal/window-masked blocks, increase `num_stages` to keep MMA fed.
- **Bank-conflict-bound** (`ncu` shared-memory bank conflicts high) → add or adjust `T.annotate_layout` / swizzled SMEM
  layouts.
- **Pipeline-starved** (`Long Scoreboard` stalls, `T.copy` not overlapping `T.gemm`) → raise `num_stages` in
  `T.Pipelined` (watching SMEM budget and compile-time RAM; use `T.Serial` on streaming reductions where `T.Pipelined`
  bloats IR).
- **Occupancy / spill-bound** (`ncu` shows `.local` spills or SMEM/register occupancy wall) → shrink tiles or
  `num_stages`, tune `threads` (typically 128 or 256).

## Optimization Levers, Roughly By Impact

1. **Memory scope placement** — keep reused inputs and accumulators in `T.alloc_shared` / `T.alloc_fragment` and fuse
   elementwise/mask/online-softmax work inside the `@T.prim_func` (also keeps `test_honesty.py` green).
2. **Tile sizes (`BM/BN/BK`), `num_stages`, and `threads`** — wire concrete, SMEM-safe configs into
   `libs/ejkernel/ejkernel/modules/operations/<op>.py` (`heuristic_cfg_gpu` / `candidate_cfgs_gpu`).
3. **Tensor-core `T.gemm` lowering** — route matmul-shaped work through `T.gemm` with `float32` accumulators and confirm
   MMA in `get_kernel_source()`.
4. **Layout & swizzling** — `T.annotate_layout` for bank-conflict-free SMEM and `T.use_swizzle` for L2 tile locality.
5. **Vectorized / async `T.copy` & `T.Pipelined` vs `T.Serial`** — align inner dimensions for 128-bit `cp.async` /
   Hopper TMA; choose `T.Serial` when `T.Pipelined` causes IR bloat on streaming reductions.
6. **In-place aliasing & cache hygiene** — use `input_output_aliases` for KV/state updates and ensure `_impl.py` caches
   by static shape/tile key so `_COMPILE_CACHE` hits cleanly.

## The Optimization Loop

1. Freeze one shape, dtype, GPU, locked clock, and `platform="tilelang"` baseline.
2. Attribute the bottleneck from `benchmark_suite.py` + `ncu` SOL + `get_kernel_source()`.
3. Change **one** lever (kernel structure or one axis of `candidate_cfgs_gpu` / `heuristic_cfg_gpu`).
4. Verify GPU parity vs `Platform.XLA` (including boundary/non-tile-divisible shapes, `jax.jit`, and backward when
   supported) plus `test_honesty.py` and `test_backend_autotune_guardrails.py`.
5. Re-run the identical benchmark command and re-profile to confirm the targeted limiter improved.
6. Keep only if it wins cleanly and stays correct; otherwise revert and record why.

## Correctness & Hardware Verification Gate

Per `AGENTS.md` Golden Rule #4 and `libs/ejkernel/test/conftest.py`, all TileLang numerical tests and benchmarks must
run on an NVIDIA GPU (`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without GPU hardware, you may run the static
AST checks (`test_honesty.py` and `test_backend_autotune_guardrails.py` with `JAX_PLATFORMS=cpu`) but MUST report kernel
correctness and performance as **unverified on hardware**.

Every kept change must pass:

- Parity vs `Platform.XLA` under `jax.jit` (and `jax.grad` when differentiable), including non-tile-aligned remainder
  shapes and all supported feature flags.
- `libs/ejkernel/test/kernels/_tilelang/test_honesty.py`, `test_registry_coverage.py`,
  `libs/ejkernel/test/kernels/test_backend_autotune_guardrails.py`, and `uv run lint-imports`.

## Definition Of Done

- GPU model (SM arch), CUDA + TileLang + JAX versions, clock-lock state, shape, dtype, `platform="tilelang"`, and tile /
  `threads` / `num_stages` config are recorded.
- GPU parity vs `Platform.XLA` (including boundary shapes), `test_honesty.py`, and `test_backend_autotune_guardrails.py`
  pass.
- Bottleneck class is named and backed by `ncu` SOL + `get_kernel_source()` evidence.
- `benchmark_suite.py` JSON/Markdown artifact paths and steady-state timings (baseline vs candidate and vs `xla` /
  sibling GPU platforms) are reported under identical clock-locked conditions.
- A re-profile confirms the targeted limiter improved; only the measured winning variant is kept.
