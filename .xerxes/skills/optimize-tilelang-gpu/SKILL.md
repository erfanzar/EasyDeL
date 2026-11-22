---
name: optimize-tilelang-gpu
description: Optimize, profile, autotune, or diagnose a TileLang GPU kernel. Use for TileLang performance work — tile sizes (block_M/N/K), T.Pipelined num_stages and async copy, thread counts, shared/fragment memory scopes, layout annotation and swizzling for bank conflicts and L2, T.gemm/tensor-core use, do_bench measurement against cuBLAS/torch, generated-CUDA inspection, or "this TileLang kernel is slower than the library / slower than it should be."
---

# Skill: Optimize A TileLang GPU Kernel

Use this when a TileLang kernel already produces correct results and the job is to make it faster, autotune it, explain
a regression, or back a performance claim. TileLang sits between Triton-level tiles and CUDA-level control: you declare
memory **scopes** (global/shared/fragment/local), tiled copies, and pipelines explicitly, and the compiler handles the
index math. That explicitness is exactly what you tune.

The loop is unchanged: measure, attribute the saturated resource, change one thing, re-measure on a locked device.
TileLang's autotuner makes sweeping cheap, so the discipline is to attribute *before* sweeping, not instead of it.

## Mental Model: How TileLang Maps To The GPU

- **Tiles and scopes are explicit.** Inside a `T.Kernel(grid..., threads=N)`
  block you allocate `T.alloc_shared` (per-block shared memory),
  `T.alloc_fragment` (register/tensor-core fragments), and `T.alloc_local`
  (per-thread registers). Where a buffer lives is your decision and the central performance lever — the same algorithm
  is fast or slow depending on what stays in shared/fragment vs what re-reads global.
- **`T.copy` moves tiles** between scopes and is auto-vectorized (and can lower to TMA on Hopper). `T.gemm` issues
  tensor-core MMA accumulating in a fragment.
  `T.Parallel` distributes a loop across the block's threads; `T.reduce_*` does cross-thread reductions.
- **`T.Pipelined(range, num_stages=S)`** software-pipelines a loop, double- buffering shared memory and overlapping
  `T.copy` (async global→shared) with
  `T.gemm`. `num_stages` is the depth/overlap-vs-shared-memory knob, exactly analogous to a multi-stage `cp.async`
  pipeline.
- **Layout and swizzle.** `T.annotate_layout` sets shared-memory layout; swizzled layouts remove **bank conflicts**.
  `T.use_swizzle` (rasterization / program reordering) raises **L2 hit rate** for GEMM, the same idea as Triton's group
  ordering.
- **`threads=`** sets threads per block — the parallelism the compiler spreads a tile across, interacting with register
  pressure and occupancy.

If a change doesn't improve scope residency (less global re-reading), pipeline overlap, tensor-core/`T.gemm` use,
bank-conflict/L2 behavior, or occupancy, it won't help. Name the resource first.

## Orient Before You Touch Code

Read the kernel and its caller before editing:

- The `T.Kernel` grid and `threads`, and the tile constants (`block_M`,
  `block_N`, `block_K`).
- Every `alloc_shared` / `alloc_fragment` / `alloc_local` — total shared-memory footprint must fit the SM budget once
  `num_stages` double-buffering is counted.
- The main loop: `T.Pipelined` range and `num_stages`, the `T.copy`s feeding it, and the `T.gemm`/accumulator.
- Any `T.annotate_layout` / `T.use_swizzle` already present.
- The `@tilelang.autotune` config space and `@tilelang.jit` wiring, if any.
- The reference (cuBLAS/torch) the profiler compares against.

Pick one representative shape/dtype and freeze it for diagnosis.

## Establish A Baseline

1. Record GPU, CUDA/TileLang version, shape, dtype, `threads`, tile sizes, and
   `num_stages`.
2. Verify correctness: build the kernel, get its profiler (`kernel.get_profiler()`), and run `profiler.assert_allclose(ref_program,
   ...)` (or compare against torch directly) at dtype-appropriate tolerance.
3. Benchmark with **`profiler.do_bench(...)`** — it warms up and times on CUDA events; pass the reference program to get
   the comparison latency in the same run.
4. **Lock GPU clocks** (`nvidia-smi -lgc`) for stable numbers. Reset after.

Keep shape, dtype, clocks, and the comparison fixed between baseline and candidate. If autotune is on, record the
winning config — a "regression" is often just a different selected config.

## Measure, Don't Guess

- **`profiler.do_bench`** — your timing harness; reports latency and, with a reference program, the
  baseline-vs-candidate comparison. Headline numbers come from here.
- **Autotune logs** — when autotuning, record which config won and its latency, so a later "slowdown" can be traced to
  config selection rather than the kernel.
- **Generated CUDA source** — `kernel.get_kernel_source()` prints the emitted CUDA. Read it to confirm `T.gemm` lowered
  to tensor-core MMA, that `T.copy`
  vectorized / used async copy, and that shared-memory layout matches your swizzle intent.
- **Nsight Compute (`ncu`)** on the generated kernel — the ground truth. **Speed Of Light** classifies compute- vs
  memory-bound;
  `MemoryWorkloadAnalysis` exposes coalescing, **bank conflicts**, and L2 hit rate; `Occupancy` and **register spills**
  show whether tiles/`num_stages` are too large. This is how you see *why* a config wins, which `do_bench` alone can't
  tell you.
- **Nsight Systems (`nsys`)** — when you suspect launch/host overhead rather than the kernel itself.

See `docs/reference/profiling.md` for the `compile`/`get_profiler`/`do_bench`/
`assert_allclose` recipe, reading `get_kernel_source()`, the autotune-config logging pattern, and running `ncu` on the
generated kernel.

## Attribute The Bottleneck

- **Memory-bound** (`ncu` memory % high, low arithmetic intensity). → raise reuse with bigger tiles, keep operands in
  shared/fragment instead of re-reading global, add `T.use_swizzle` for L2, fix coalescing in `T.copy`, cut bytes with a
  smaller dtype.
- **Compute-bound** (`tl.gemm`/tensor cores saturated). → confirm tensor-core lowering in the generated source, raise
  `num_stages` to keep MMA fed, remove redundant work; near the roofline, tiling won't move it.
- **Bank-conflict-bound** (`ncu` shows shared-memory conflicts / serialized access). → fix `T.annotate_layout` / add
  swizzle so warps hit distinct banks.
- **Pipeline-starved** (copies not overlapping `T.gemm`). → raise `num_stages`
  (watch the shared-memory ceiling), confirm async copy is engaged.
- **Occupancy/spill-bound** (low achieved occupancy or spills). → shrink tiles or
  `num_stages`, or adjust `threads`; spills usually erase tiling gains.

## Optimization Levers, Roughly By Impact

1. **Tile sizes (`block_M/N/K`)** — primary lever; larger tiles raise reuse and tensor-core efficiency but cost shared
   memory, registers, and occupancy. Drive with the autotuner.
2. **`num_stages` in `T.Pipelined`** — pipeline depth / async-copy overlap vs shared-memory budget. Sweep it; too high →
   shared-mem-over-budget or spills.
3. **Memory scope placement** — keep hot operands and accumulators in
   `alloc_fragment`/`alloc_shared`; don't let the loop re-read global. The biggest structural wins come from moving a
   buffer to the right scope.
4. **Layout + swizzle** — `T.annotate_layout` to remove bank conflicts;
   `T.use_swizzle` for L2 reuse on GEMM. Both are often large, cheap wins.
5. **`threads=`** — threads per block; tune against register pressure and occupancy.
6. **`T.gemm` / fragment use** — ensure matmul-shaped work goes through tensor cores, not scalar loops.
7. **Vectorized / TMA `T.copy`** — let the compiler emit wide or hardware- accelerated transfers (TMA on Hopper).

Wire these into **`@tilelang.autotune`** with a config space, but keep the space no wider than your attribution
justifies.

## The Optimization Loop

1. Freeze one shape/dtype/GPU with locked clocks.
2. Attribute the bottleneck from `do_bench` + `ncu` SOL + generated source.
3. Change one lever — or expand the autotune space along **one** axis.
4. Rebuild; rerun `profiler.assert_allclose` / reference parity, including a boundary shape.
5. Re-run `do_bench` under identical conditions; note the winning config.
6. Keep only if it wins and stays correct; otherwise revert and record why. Re-profile to confirm the targeted limiter
   improved.

## Correctness Is A Gate, Not A Step

Every kept change must still pass:

- `profiler.assert_allclose` (or independent torch/cuBLAS parity) at dtype- appropriate tolerance — `fp16`/`bf16`
  accumulate-in-`f32` is not bit-exact.
- **Boundary shapes** not divisible by the tile, exercising the remainder/ predication path.
- The full shape sweep used in the report, not just the tuned shape.

## Common Mistakes

- Reporting `do_bench` numbers without locked clocks (boost/thermal noise).
- Shared-memory over budget once `num_stages` double-buffering is counted → build failure or silent fallback.
- Bank conflicts left in place because the kernel "still works" — a large hidden tax `ncu` would show immediately.
- Assuming `T.gemm` used tensor cores without checking the generated source.
- Autotune space so large it overfits one shape and is slow to tune, used as a substitute for attribution.
- Comparing only against an unoptimized baseline instead of cuBLAS/torch.
- Validating only the tile-aligned shape and shipping a remainder bug.

## Definition Of Done

- GPU, CUDA/TileLang version, shape, dtype, tile sizes, `threads`, and
  `num_stages` are recorded.
- `assert_allclose` / reference parity passes, including boundary shapes.
- The bottleneck class is named and backed by `ncu` SOL and generated-source evidence.
- `do_bench` baseline-vs-candidate numbers are reported under identical, clock-locked conditions, with the winning
  config named and compared to a real library baseline (cuBLAS/torch) where one exists.
- A re-profile confirms the targeted limiter improved.
- Only the measured winning configuration is kept.
