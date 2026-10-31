---
name: optimize-cuda-gpu
description: Optimize, profile, or diagnose a CUDA C++ GPU kernel on NVIDIA hardware. Use for CUDA kernel performance work — memory coalescing, occupancy and register/shared-memory budgets, bank conflicts, warp divergence, tensor cores (mma/wmma), cp.async/TMA pipelining, Nsight Compute/Systems profiling, roofline/speed-of-light attribution, compute-sanitizer correctness, or "this CUDA kernel is slower than it should be / slower than cuBLAS."
---

# Skill: Optimize A CUDA GPU Kernel

Use this when a CUDA kernel already produces correct results and the job is to make it faster, explain a regression, or
back a performance claim with profiler evidence. Correctness comes first; a fast wrong kernel is a bug.

The discipline is fixed: profile to find the saturated resource, relieve that one resource, re-measure under locked
conditions. Block-size roulette without a profiler is how people spend a week to lose 3%.

## Mental Model: What A GPU Actually Is

- **Execution.** Threads run in **warps of 32**, lockstep within a warp. Warps are grouped into thread blocks, blocks
  run on **SMs**. Each SM has a fixed pool of registers and shared memory split across its resident warps. Latency is
  hidden by having many warps ready: when one warp stalls on memory, the scheduler runs another.
- **Memory hierarchy.** Global (HBM/GDDR, the bandwidth wall) → **L2** (chip- wide) → **L1/shared** (per-SM, a
  configurable split) → **registers** (per- thread, fastest). Most kernels live or die on how they move data through
  this, not on arithmetic.
- **Coalescing.** When the 32 threads of a warp touch consecutive, aligned addresses, the hardware services them in the
  fewest transactions. Strided or scattered access multiplies transactions and wastes bandwidth.
- **Shared memory banks.** Shared memory has **32 banks**; if threads in a warp hit the same bank with different
  addresses, accesses serialize (a *bank conflict*). Padding or swizzling the layout removes it.
- **Occupancy.** Active warps per SM ÷ the architectural max, capped by registers/thread, shared mem/block, and
  blocks/SM. Occupancy buys latency hiding — but it is a means, not the goal. A register-rich, high-ILP kernel can beat
  a high-occupancy one (Volkov). Optimize the bottleneck, not the occupancy number.
- **Tensor cores.** `mma`/`wmma` (and CUTLASS/CuTe) do GEMM-shaped work in
  `fp16`/`bf16`/`tf32`/`fp8` far faster than CUDA cores. Any matmul-heavy kernel not using them is leaving most of the
  FLOPs on the table.
- **Async movement.** `cp.async` (Ampere+) and **TMA** (Hopper, via
  `cuda::memcpy_async` / bulk copies) overlap global→shared transfer with compute, enabling multi-stage software
  pipelines and double buffering.

If a change does not improve coalescing, cut bytes moved, raise useful occupancy/ILP, remove a conflict/divergence
stall, or engage tensor cores, it will not help. Name the resource first.

## Orient Before You Touch Code

Read the kernel and its launch before editing:

- The launch config: grid/block dims, dynamic shared memory size, stream.
- Per-thread work: global loads/stores and their index expressions (is the innermost-varying thread index on the
  contiguous dimension?), shared-memory tiles, register-heavy locals, branches.
- `__restrict__`, `__launch_bounds__`, `#pragma unroll`, and any intrinsics already present.
- The reference/library result (e.g. cuBLAS/cuDNN/torch) you compare against.

Compile with `nvcc --ptxas-options=-v` (or `-Xptxas -v`) once up front to see **registers/thread, shared mem/block, and
spill stores/loads**. Spills to local memory are silent in source and often the whole story.

Pick one representative problem size and dtype. Freeze it.

## Establish A Baseline

1. Record GPU (e.g. A100/H100/L4), driver/CUDA version, problem size, dtype, launch config, and the build flags.
2. Verify correctness against an independent reference.
3. Time the kernel **with CUDA events** around the launch, after warmup, over many iterations — never wall-clock that
   includes H2D/D2H copies or the first compile. Report median/min over the run.
4. **Lock clocks** for stable numbers: `nvidia-smi -lgc <freq>` (and
   `-lmc`) so boost/thermal drift does not masquerade as a result. Reset after.

Keep problem size, dtype, launch config, and clocks fixed between baseline and candidate.

## Measure, Don't Guess

- **Nsight Compute (`ncu`)** — per-kernel deep profile. Start with
  `ncu --set full` (or targeted `--section`s). Read the **GPU Speed Of Light**
  section first: it tells you compute throughput % vs memory throughput % — that single comparison classifies the
  kernel. Then
  `MemoryWorkloadAnalysis` (coalescing, sectors/request, bank conflicts),
  `Occupancy` (achieved vs theoretical and the limiter), `SchedulerStats` /
  `WarpStateStats` (top stall reasons), and the built-in roofline chart.
- **Nsight Systems (`nsys`)** — timeline across kernels, streams, and H2D/D2H copies. Use it to see whether the kernel
  is even the bottleneck, or whether you are serialized on copies / launch overhead / missing overlap.
- **`--ptxas-options=-v`** — registers, shared mem, and spills at compile time.
- **`compute-sanitizer`** — correctness, not speed: `memcheck` (OOB/misaligned),
  `racecheck` (shared-memory races), `synccheck` (bad `__syncthreads`),
  `initcheck` (uninitialized global reads). Run it before trusting any result.

`ncu` serializes and replays kernels; its absolute times are not your benchmark numbers — use it for *attribution*, use
CUDA events for *timing*.

See `docs/reference/profiling.md` for the clock-locking recipe, the exact `ncu`
sections/metrics to read in order, the CUDA-event timing pattern, and the
`compute-sanitizer` invocations.

## Attribute The Bottleneck

From the SOL section and stall stats, classify:

- **Memory-bound** (memory % ≫ compute %, low arithmetic intensity, long-scoreboard stalls). → improve coalescing,
  vectorize loads, raise reuse via shared-memory tiling, cut bytes (smaller dtype), improve L2 hit rate.
- **Compute-bound** (compute % ≫ memory %). → engage tensor cores, raise ILP, remove redundant math, reduce
  special-function pressure; you are near the roofline so memory tricks won't move it.
- **Latency / occupancy-bound** (neither unit saturated, schedulers starved, many "no eligible warp" cycles). → raise
  occupancy (cut registers/shared mem)
  *or* raise ILP per thread; reduce sync, reduce divergence.
- **Overhead-bound** (nsys shows tiny kernels dwarfed by launch/copy time). → fuse kernels, use CUDA graphs, overlap
  copies on streams.

## Optimization Levers, Roughly By Impact

1. **Fix the global access pattern.** Make consecutive threads read consecutive aligned addresses. Use **vectorized
   loads** (`float4`/`int4`) for aligned contiguous data — fewer, wider transactions. This is usually the single biggest
   lever.
2. **Tile through shared memory.** Stage reused global data into shared memory once, compute from there. **Pad/swizzle**
   the shared layout to kill bank conflicts (`ncu` reports them directly).
3. **Engage tensor cores** for GEMM-shaped inner loops (`wmma`/`mma`, or lean on CUTLASS/CuTe rather than hand-rolling).
4. **Tune the register/occupancy trade-off.** `__launch_bounds__` and
   `-maxrregcount` cap registers to raise occupancy — but watch for **new spills** (check `-v`); a spill usually costs
   more than the occupancy gains. Test both directions; do not assume more occupancy is better.
5. **Overlap with async copy.** `cp.async` + a 2–4 stage pipeline (or TMA on Hopper) to hide global latency behind
   compute. Counts only with a measured overlap and win.
6. **Remove warp divergence.** Restructure branches so a warp takes one path; hoist uniform conditions; use predication
   for short divergent regions.
7. **Instruction-level cleanups.** `__restrict__` (enables reordering/reuse),
   `#pragma unroll` on bounded loops, fast intrinsics where precision allows,
   `__ldg`/read-only path for reused read-only data.

## The Optimization Loop

1. Freeze one problem size/dtype/GPU with locked clocks.
2. Attribute the bottleneck from `ncu` SOL + stalls (and `nsys` if overhead is suspected).
3. Change exactly **one** lever.
4. Recompile, rerun `compute-sanitizer`, rerun correctness vs reference.
5. Re-time with CUDA events under identical conditions.
6. Keep only if it wins and stays correct; otherwise revert and record why. Re-profile to confirm the stall you targeted
   actually shrank.

## Correctness Is A Gate, Not A Step

Every kept change must still pass:

- `compute-sanitizer` clean (memcheck + racecheck + synccheck + initcheck).
- Value parity vs an independent reference at the right tolerance for the dtype (`fp16`/`bf16`/`tf32` are not bit-exact
  with `f32`).
- Boundary shapes that aren't multiples of the tile/block (the masking and remainder paths), not just the nice aligned
  size.

## Common Mistakes

- Timing with wall-clock or including H2D/D2H copies and the first compile.
- Not locking clocks; reporting thermal/boost noise as a speedup.
- Chasing occupancy while the kernel is memory-bound, or raising occupancy into register spills.
- Ignoring uncoalesced access — the most common real bottleneck.
- Shared-memory bank conflicts left in place because the kernel "still works."
- Using `ncu` replay times as benchmark numbers.
- Claiming async/overlap wins without an `nsys`/`ncu` measurement showing it.
- Validating only on the aligned size and shipping a remainder-path bug.

## Definition Of Done

- GPU, CUDA/driver version, problem size, dtype, launch config, and clock state are recorded.
- `compute-sanitizer` is clean and correctness vs reference passes, including boundary shapes.
- The bottleneck class is named and backed by an `ncu` SOL/stall reading.
- Baseline-vs-candidate event-timed numbers are reported under identical, clock-locked conditions.
- A re-profile confirms the targeted stall/limiter actually improved.
- Only the measured winning variant is kept.
