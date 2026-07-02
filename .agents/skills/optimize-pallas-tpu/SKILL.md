---
name: optimize-pallas-tpu
description: Optimize, profile, or diagnose a JAX Pallas kernel on TPU. Use for Pallas/Mosaic TPU performance work — block/tiling choice, VMEM budget, grid ordering and pipelining, megacore (dimension_semantics), MXU/VPU utilization, DMA overlap, bf16 layout, HLO/LLO/Mosaic dump analysis, roofline attribution, or "this Pallas kernel is slower than XLA / slower than it should be" on a fixed TPU shape.
---

# Skill: Optimize A Pallas TPU Kernel

Use this when a Pallas TPU kernel already runs correctly and the job is to make
it faster, explain a regression, or justify a performance claim with evidence.
If you are adding a brand-new kernel or porting an algorithm, get it correct
first, then come back here.

Optimization on TPU is not guesswork over block sizes. It is: build a roofline,
find which resource is saturated, change the one thing that relieves it, and
re-measure on the same device. Everything below serves that loop.

## Mental Model: What A TPU Actually Is

You cannot optimize a TPU kernel without holding its execution model in your
head. The relevant machine:

- **MXU** — the matrix unit, a 128x128 systolic array. It eats `bf16` operands
  and accumulates in `f32`. Matmul-shaped work wants the MXU. A matmul whose
  contracting or output dims are not multiples of 128 wastes MXU columns to
  padding.
- **VPU** — the vector unit, laid out as `(8 sublanes, 128 lanes)`. All
  elementwise, masking, reduction, and transcendental work runs here. The
  native tile is `(8, 128)` for `f32`; the **last dimension wants to be a
  multiple of 128 and the second-to-last a multiple of 8**, or Mosaic pads.
- **Memory hierarchy** — `HBM` (large, the bandwidth wall) → `VMEM` (a few MB
  of fast scratch, where blocks live during compute) → vector registers.
  `SMEM` holds scalars/indices. Pallas's job is to stage HBM tiles into VMEM
  and keep them resident across grid steps.
- **The grid is a sequential loop with prefetch.** `pl.pallas_call(grid=...)`
  walks the grid in row-major order (the **last grid axis varies fastest**).
  Mosaic double-buffers DMAs: while the kernel computes step `i`, it is already
  copying the inputs for step `i+1`. An input whose `BlockSpec` index map does
  **not** change between steps stays resident and is not re-fetched.
- **Megacore.** v4 / v5p chips have two TensorCores sharing HBM. Marking a grid
  axis as parallel (via the Mosaic/TPU compiler params,
  `dimension_semantics=("parallel", "arbitrary", ...)`) lets Mosaic split that
  axis across the two cores. `"arbitrary"` axes are serialized (needed when
  steps carry a dependency, e.g. an accumulator).

If a change does not improve MXU occupancy, cut HBM bytes, fit VMEM better, or
remove a VPU bottleneck, it will not help. Name the resource first.

## Orient Before You Touch Code

Read the kernel and its caller before changing anything:

- The `pl.pallas_call`: `grid`, every input/output `BlockSpec` and its index
  map, `scratch_shapes`, `dimension_semantics`, and `interpret`.
- The kernel body: where `pl.dot`/MXU work happens, where masking/`pltpu`
  primitives run, what stays in VMEM scratch vs what is re-read from HBM.
- The input shapes, dtypes, and which dims are contracted vs mapped over.
- The reference/XLA path that produces the same result — your correctness and
  performance baseline.

Write down the one representative `(shape, dtype, device)` you will optimize.
Freeze it. Diagnosing on a moving shape produces noise, not signal.

## Establish A Baseline

Before editing:

1. Record op, TPU generation (v4/v5e/v5p/v6e), device count, shape, dtype,
   and the exact launch command and env.
2. Run correctness/parity against the reference and save the result.
3. Run a baseline benchmark — **steady state, after warmup** — and save the raw
   numbers. One warmup call compiles; time the calls after it.
4. Decide what the comparison is: XLA vs Pallas, last-good commit vs current,
   config A vs config B, or full kernel vs an isolated decomposition.

Keep warmup count, iteration count, shape, dtype, and device fixed between
baseline and candidate. A "win" from a changed command line or a different
shape is not a win. Lock TPU state by owning the device for the whole run.

CPU/XLA may only be used for host-side preflight — imports, shape checks,
reference math, harness syntax. **CPU timing is never evidence for TPU
behavior, Mosaic lowering, or DMA overlap.** If you cannot get the TPU, say the
target run was not performed rather than substituting CPU numbers.

## Measure, Don't Guess

Pick the tool that answers the question you actually have:

- **Static cost model** — `jax.jit(f).lower(*args).compile().cost_analysis()`
  returns FLOPs and bytes-accessed. Combine into arithmetic intensity
  (FLOPs/byte) to predict whether the op *should* be compute- or memory-bound
  before you run anything.
- **Trace / timeline** — `jax.profiler.start_trace(dir)` … `stop_trace()` (or
  `with jax.profiler.trace(dir):`) then open the trace in TensorBoard/xprof.
  Shows op durations, fusion boundaries, and gaps/overlap on the device
  timeline. Use it to see whether you are even spending time in the kernel vs
  in surrounding copies/transposes.
- **HLO dump** — `XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text"`.
  Confirms the expected custom-call/fusion is emitted and the kernel is on the
  path you think it is.
- **LLO / Mosaic dump** — for structural TPU diagnosis when wall-clock alone
  does not explain the result. Use a per-variant dump root and the
  `LIBTPU_INIT_ARGS` LLO/Mosaic flags; inspect
  `*schedule-analysis_final_bundles.txt` for total/non-empty bundle counts, and
  compare lane-rotation (`vrot`), select/mask (`vsel`), and transcendental
  (`vpow2`) op counts plus spills and schedule gaps against the XLA path. This
  is the highest-signal tool when Pallas is mysteriously slower than XLA on one
  shape. Run one tiny TPU smoke compile with the exact flags first to confirm
  the runtime accepts them; CPU dumps only validate flag spelling.

Keep dump directories separate per variant and keep `LIBTPU_INIT_ARGS`
identical between compared variants.

See `docs/reference/profiling.md` for exact commands, env vars, the LLO/Mosaic
flag block, and how to read schedule bundles and pressure counters.

## Attribute The Bottleneck

Classify before you optimize. The class dictates the fix:

- **Memory-bound** (HBM bandwidth is the wall; low arithmetic intensity, MXU
  idle). → cut HBM bytes: larger blocks for reuse, keep operands resident
  across grid steps, fuse adjacent ops, use `bf16` to halve transfer width.
- **Compute-bound** (MXU saturated). → you are near the roofline; remaining wins
  come from removing padding (align dims to 128/8) or reducing redundant FLOPs,
  not from tiling tricks.
- **VPU-bound** (elementwise/masking/transcendental dominates; LLO shows heavy
  `vrot`/`vsel`/`vpow2`). → reduce mask work, precompute, restructure to push
  work onto the MXU, or cut transcendentals.
- **Latency / overhead-bound** (kernel is fast but the timeline shows it dwarfed
  by surrounding transposes, reshapes, or poor DMA overlap). → fix the
  surrounding graph or the pipeline, not the inner loop.

If LLO shows bundle inflation versus XLA, that is a **structural** problem —
fix the decomposition before sweeping block sizes.

## Optimization Levers, Roughly By Impact

1. **Cut HBM traffic.** Order the grid so the largest operand's block index
   changes slowest, so it stays resident. Reuse VMEM scratch for accumulators
   instead of re-reading from HBM. Fuse producer/consumer ops into one kernel.
2. **Choose block shapes deliberately.** Contracting and feature dims as
   multiples of 128; sublane dims as multiples of 8. Blocks large enough to
   fill the MXU and amortize DMA, small enough that **all resident buffers plus
   double-buffering (≈2× each streamed input) fit in VMEM**. VMEM OOM or
   silent spill kills the win.
3. **Enable megacore.** Mark independent grid axes `"parallel"` so Mosaic uses
   both cores on v4/v5p. Keep accumulator-carrying axes `"arbitrary"`.
4. **Use `bf16` for MXU inputs, accumulate in `f32`.** Halves HBM/VMEM bytes and
   feeds the MXU its native type. Validate tolerances after.
5. **Pipeline manually only when the grid cannot express the reuse.**
   `pltpu.emit_pipeline` / explicit `pltpu.make_async_copy` give an inner
   software pipeline. Add this only after the automatic grid pipeline is shown
   insufficient — it is more code and more ways to be wrong.
6. **Kill VPU hotspots.** Replace per-element masking with precomputed masks or
   structural changes; avoid lane rotations and selects the LLO dump flags.

Async/DMA overlap **only counts when the measured target run shows both the
overlap and a wall-clock win.** Correct-looking async with no measured win is
not an optimization.

## The Optimization Loop

1. Freeze one shape/dtype/device.
2. Attribute the bottleneck from cost model + trace + (if needed) LLO dump.
3. Change exactly **one** lever: block size, grid order, `dimension_semantics`,
   dtype, scratch reuse, or pipeline structure.
4. Rerun correctness/parity.
5. Rerun the **same** benchmark, steady state.
6. Keep the change only if it wins on the target TPU and breaks no supported
   shape. Otherwise revert and record why.

Never broad-sweep block sizes before checking structure. If Pallas is slower
than XLA on a fixed shape, dump both paths and compare schedule bundles before
adding any DMA or async complexity.

## Correctness Is A Gate, Not A Step

Every kept change must still pass:

- Value parity against an **independent** reference (XLA/numpy), not another
  wrapper that dispatches to the same code.
- Gradient parity for differentiable kernels.
- Shape, dtype, and finite-output checks across a small shape grid, including
  non-aligned shapes that exercise masking/padding.
- `bf16` runs compared at appropriate tolerance, not `f32` exactness.

A faster kernel that is wrong on an edge shape is a regression, not a win.

## Common Mistakes

- Reporting CPU/XLA timing as TPU evidence.
- Calling steady-state suite timing "compile-including," or vice versa.
- Sweeping block sizes before checking the LLO schedule for structural
  regressions.
- VMEM OOM / silent spill from blocks too large once double-buffering is
  counted.
- Last dim not a multiple of 128 (or sublane not a multiple of 8), paying
  padding on every tile.
- Forgetting `"parallel"` semantics and leaving the second core idle.
- Changing shape, dtype, warmup, iterations, or `LIBTPU_INIT_ARGS` between
  baseline and candidate.
- Claiming DMA/async overlap without a measured win.

## Definition Of Done

- Baseline and candidate commands, env, and TPU generation are recorded.
- Correctness/parity passes for the affected kernel and a representative shape
  grid.
- The bottleneck class is named and supported by cost model, trace, or dump
  evidence.
- Direct baseline-vs-candidate steady-state numbers are reported with shape,
  dtype, device count, warmup, and iterations.
- Dump/LLO evidence is included whenever timing alone did not explain the
  result.
- Only the measured winning path is kept; losing variants are removed or
  clearly marked as references.
