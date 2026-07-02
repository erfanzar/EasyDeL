---
name: optimize-triton-gpu
description: Optimize, profile, autotune, or diagnose an OpenAI Triton GPU kernel. Use for Triton performance work — block sizes (BLOCK_M/N/K), num_warps and num_stages pipelining, autotune config spaces and cache keys, tl.dot/tensor-core use, L2 program swizzling, masking and contiguity hints, do_bench measurement, MLIR/PTX dump inspection, register-spill diagnosis, or "this Triton kernel is slower than torch / slower than it should be."
---

# Skill: Optimize A Triton GPU Kernel

Use this when a Triton kernel already produces correct results and the job is to
make it faster, tune it, explain a regression, or back a performance claim.
Triton hides the thread/register layout, so optimization is mostly about
**block shape, pipeline depth, parallelism, and memory access structure** — plus
reading what the compiler actually generated when those don't explain the time.

The loop is the same as any kernel: measure, attribute the saturated resource,
change one thing, re-measure on a locked device. Triton makes step three cheap
(constexprs + autotune), which makes it tempting to skip steps one and two —
don't.

## Mental Model: How Triton Maps To The GPU

- **You program a block, not a thread.** Each program instance
  (`tl.program_id`) owns a tile of the output. The grid is over tiles. Inside,
  you express math on `BLOCK_M × BLOCK_N`(× `BLOCK_K`) arrays and the compiler
  lowers it to warps, registers, and shared memory.
- **`num_warps`** sets how many warps execute one program instance — the
  parallelism Triton has to cover your block. **`num_stages`** sets the depth of
  the **software pipeline** over a reduction loop (typically the K loop): more
  stages overlap more `tl.load`s with `tl.dot` via async copy, at the cost of
  more shared memory. These two knobs, with the block sizes, dominate.
- **`tl.dot` uses tensor cores** and accumulates in `f32`. Matmul-shaped inner
  loops should go through it.
- **Memory.** `tl.load`/`tl.store` move global↔on-chip. Coalescing still
  matters: the block's memory layout should make consecutive elements
  contiguous in global memory. `tl.make_block_ptr` (and TMA on Hopper) give the
  compiler the structure to emit good transfers. `eviction_policy` and cache
  hints control reuse.
- **L2 reuse via program ordering.** For GEMM, the order in which program ids map
  to output tiles changes L2 hit rate. The classic **group/swizzle ordering**
  (the `GROUP_SIZE_M` trick) reorders programs so co-scheduled blocks share
  operands in L2 — often a double-digit-percent win for free.
- **Masking has a cost.** Boundary masks on `tl.load`/`tl.store` add predication.
  When dims are known multiples of the block, contiguity/divisibility hints let
  the compiler drop masking overhead.

If a change doesn't improve the pipeline overlap, tensor-core/`tl.dot` use,
coalescing/L2 reuse, or remove masking/spill overhead, it won't help.

## Orient Before You Touch Code

Read the kernel and its launch before editing:

- The grid lambda and how `program_id`s map to output tiles (is there L2
  swizzling, or naive row-major?).
- The constexpr block sizes and any `@triton.autotune` config list and its
  `key=[...]` — the key must include every shape dim that should re-trigger
  tuning, or you will silently reuse a config tuned for a different shape.
- The K-loop: `tl.load`s, `tl.dot`, accumulator dtype, `num_stages` assumption.
- Masking, `tl.multiple_of` / `tl.max_contiguous` hints, `eviction_policy`.
- The torch/reference op you compare against — and make sure it's the *fused*
  equivalent, not an unfused baseline that flatters you.

Pick one representative shape/dtype and freeze it for diagnosis.

## Establish A Baseline

1. Record GPU, Triton/CUDA version, shape, dtype, and the autotune state.
2. Verify correctness vs an independent torch reference at dtype-appropriate
   tolerance.
3. Benchmark with **`triton.testing.do_bench`** — it warms up, flushes the L2
   cache between runs, and returns timing (with quantiles). Do not hand-roll
   timing; `do_bench` exists to avoid the usual mistakes.
4. **Lock GPU clocks** (`nvidia-smi -lgc`) so boost/thermal drift isn't read as
   a result. Reset after.

Keep shape, dtype, clocks, and the comparison fixed across baseline and
candidate. If autotune is on, record which config won — a "regression" is often
just a different cached config.

## Measure, Don't Guess

- **`do_bench` / `triton.testing.Benchmark` + `perf_report`** — your timing and
  sweep harness across shapes; produces the headline numbers and plots.
- **`TRITON_PRINT_AUTOTUNING=1`** — prints which config autotune selected and
  the timings it compared. First thing to check when "the same kernel got
  slower."
- **Nsight Compute (`ncu`)** — works on Triton kernels. Read **Speed Of Light**
  (compute vs memory %) to classify, `MemoryWorkloadAnalysis` for coalescing/L2,
  `Occupancy` and **register spills** (Triton can spill silently when blocks /
  `num_stages` are too large). This is how you see *why* a config wins.
- **Compiler dumps** — when block/stage tuning doesn't explain the time, read
  what was generated: `TRITON_KERNEL_DUMP=1` (dumps IR stages to the cache),
  `MLIR_ENABLE_DUMP=1` (per-pass MLIR), and the cached `ttir`/`ttgir`/`llir`/
  `ptx`/`cubin` under `TRITON_CACHE_DIR`. `TRITON_ALWAYS_COMPILE=1` forces a
  rebuild so you see fresh IR. Check the generated layouts, whether `tl.dot`
  lowered to tensor-core MMA, and shared-memory allocation.
- **proton** — Triton's lightweight intra-kernel profiler for hot-region
  attribution without a full `ncu` run.

See `docs/reference/profiling.md` for the `do_bench` recipe, every Triton dump
env var, how to read the `.ttgir`/`.ptx` artifacts, and running `ncu` on a
mangled Triton kernel name.

## Attribute The Bottleneck

- **Memory-bound** (`ncu` memory % high, low arithmetic intensity). → improve
  coalescing/contiguity, add L2 swizzling, raise reuse with larger blocks, cut
  bytes with a smaller dtype, tune `eviction_policy`.
- **Compute-bound** (compute % high, `tl.dot` saturated). → confirm tensor cores
  are actually used (dump), raise `num_stages` to keep them fed, remove
  redundant math; near the roofline, tiling won't move it.
- **Pipeline-starved** (loads not overlapping compute; scheduler stalls on
  memory). → raise `num_stages` (watch shared-mem/spill ceiling), ensure async
  copy is engaged, restructure the K-loop.
- **Occupancy/spill-bound** (`ncu` shows spills or low achieved occupancy). →
  shrink block sizes or `num_stages`, or lower `num_warps`; spills usually
  erase any tiling gain.
- **Overhead-bound** (tiny kernel, launch/Python dominates per `nsys`). → fuse,
  use CUDA graphs, or batch.

## Optimization Levers, Roughly By Impact

1. **Block sizes (`BLOCK_M/N/K`)** — the primary lever; bigger blocks raise reuse
   and tensor-core efficiency but cost shared memory, registers, and occupancy.
   Drive this with autotune, not by hand.
2. **`num_stages`** — pipeline depth over the reduction loop; more overlap vs
   more shared memory. Sweep it; too high → shared-mem-over-budget compile
   failure or spills.
3. **`num_warps`** — parallelism per program; interacts with block size and
   register pressure.
4. **L2 program swizzling** (`GROUP_SIZE_M`-style reordering) for GEMM — often a
   large, cheap win by raising L2 hit rate.
5. **Contiguity / divisibility hints** — `tl.multiple_of`, `tl.max_contiguous`,
   and block pointers let the compiler vectorize and drop masking when shapes
   are known-aligned.
6. **`eviction_policy` and cache modifiers** on `tl.load` — keep reused tiles in
   cache, evict streamed-once data.
7. **TMA / `tl.make_block_ptr`** on Hopper for hardware-accelerated tiled
   transfers.

Wire these into **`@triton.autotune`** with a config list and a `key` covering
every shape dim that should re-tune. Set the key wrong and you ship a config
tuned for the wrong shape.

## The Optimization Loop

1. Freeze one shape/dtype/GPU with locked clocks.
2. Attribute the bottleneck from `do_bench` + `ncu` SOL + (if needed) IR dumps.
3. Change one lever — or expand the autotune space along **one** axis.
4. Rerun correctness vs the torch reference (including a boundary shape).
5. Re-run `do_bench` under identical conditions; with autotune on, note the
   winning config.
6. Keep only if it wins and stays correct; otherwise revert and record why.
   Re-profile to confirm the targeted limiter improved.

Do not expand a giant autotune grid as a substitute for attribution — it is slow
to tune, easy to overfit to one shape, and hides *why* the win happened.

## Correctness Is A Gate, Not A Step

Every kept change must still pass:

- Value parity vs an independent torch reference at dtype-appropriate tolerance.
- **Boundary / non-power-of-2 shapes** that exercise the masking path — the most
  common place a "faster" Triton kernel is silently wrong.
- The full sweep used in the report, not just the one tuned shape.

## Common Mistakes

- Hand-rolled timing instead of `do_bench` (no warmup, no L2 flush).
- Not locking clocks; reading boost/thermal noise as a speedup.
- Autotune `key` missing a shape dim → reusing a config tuned for another shape
  and calling the difference a regression.
- `num_stages` too high → shared-memory-over-budget compile failure, or silent
  register spills that erase the gain.
- Comparing against an **unfused** torch baseline to manufacture a speedup.
- Testing only the aligned tuned shape and shipping a masking bug.
- Assuming `tl.dot` used tensor cores without confirming in the dumped IR.

## Definition Of Done

- GPU, Triton/CUDA version, shape, dtype, clock state, and autotune config are
  recorded.
- Correctness vs an independent reference passes, including boundary shapes.
- The bottleneck class is named and backed by `ncu` and/or IR-dump evidence.
- `do_bench` baseline-vs-candidate numbers are reported under identical,
  clock-locked conditions, with the winning autotune config named.
- A re-profile confirms the targeted limiter improved.
- Only the measured winning configuration is kept; the autotune space is no
  wider than the evidence justifies.
