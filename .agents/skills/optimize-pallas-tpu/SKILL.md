---
name: optimize-pallas-tpu
description: Optimize, profile, or diagnose a JAX Pallas kernel on TPU. Use for Pallas/Mosaic TPU performance work — block/tiling choice, VMEM budget, grid ordering and pipelining, megacore (dimension_semantics), MXU/VPU utilization, DMA overlap, bf16 layout, HLO/LLO/Mosaic dump analysis, roofline attribution, or "this Pallas kernel is slower than XLA / slower than it should be" on a fixed TPU shape.
---

# Skill: Optimize A Pallas TPU Kernel

Specialization of `.agents/skills/optimize-ejkernel-kernel/SKILL.md` (and `.agents/skills/run-research/SKILL.md`) for
JAX Pallas/Mosaic kernels under `libs/ejkernel/ejkernel/kernels/_pallas/tpu/`. Use this once the kernel is functionally
correct and the goal is to make it faster, explain a regression, or justify a performance claim on TPU hardware. For new
kernels or backend wiring, start with `.agents/skills/add-ejkernel-kernel/SKILL.md`.

Optimization on TPU is not guesswork over block sizes: build a roofline from the XLA reference, identify which hardware
unit is saturated, change the one lever that relieves it, and re-measure on the same TPU.

## Mental Model: TPU Architecture

- **MXU (matrix unit):** systolic array — **128×128 on TPU v4/v5e/v5p** and **256×256 on TPU v6e (Trillium)**. Consumes
  `bf16` (or supported low-bit operands) and accumulates in `f32`. Contracting and output dimensions that are not
  multiples of the MXU tile width (128 on v4/v5e/v5p, 256 on v6e) waste MXU lanes to padding.
- **VPU (vector unit):** `(8 sublanes, 128 lanes)`. Handles elementwise ops, masking, reductions, and transcendentals.
  The native `f32` tile is `(8, 128)`: keep the last dim a multiple of 128 and the second-to-last a multiple of 8, or
  Mosaic pads every tile.
- **Memory hierarchy:** `HBM` (bandwidth wall) → `VMEM` (fast on-chip scratch: ~16 MB/core on v4, ~64 MB/core on v5p,
  ~128 MB on v5e, ~32 MB on v6e) → vector registers; `SMEM` holds scalars and indices. Control the VMEM ceiling via
  `pltpu.CompilerParams(dimension_semantics=..., vmem_limit_bytes=...)`.
- **Grid loop & DMA buffering:** `pl.pallas_call(grid=...)` walks the grid in row-major order (the **last grid axis
  varies fastest**). Mosaic double-buffers DMAs by default: while computing step `i`, it prefetches step `i+1`. Inputs
  whose `BlockSpec` index map does not change across inner steps stay resident in VMEM. In ejkernel, explicit
  multi-buffering and scalar prefetch use `ejkernel.callib.buffered_pallas_call`
  (`libs/ejkernel/ejkernel/callib/_pallas_call.py`), `pltpu.PrefetchScalarGridSpec`, and
  `pl.Buffered(buffer_count=..., use_lookahead=True)` (or `input_buffer_count=...`).
- **Megacore:** TPU **v4 and v5p** are dual-core chips sharing HBM; marking independent grid axes `"parallel"` in
  `dimension_semantics` splits work across both cores, while `"arbitrary"` serializes dependent/accumulator axes. TPU
  **v5e and v6e** are single-core per chip (no dual-core Megacore split, though `"parallel"` still guides Mosaic
  scheduling).

If a change does not raise MXU utilization, cut HBM bytes, fit VMEM cleanly, or remove a VPU bottleneck, it will not
win. Name the resource first.

## Orient Before Touching Code

Read the kernel, its operation wrapper, and its XLA reference before editing:

- Backend kernel: `libs/ejkernel/ejkernel/kernels/_pallas/tpu/<kernel>/` (`_interface.py`, `_pallas_impl_fwd.py`,
  `_pallas_impl_bwd.py`).
- XLA reference: `libs/ejkernel/ejkernel/kernels/_xla/<kernel>/`.
- Public wrapper & configs: `libs/ejkernel/ejkernel/modules/operations/<kernel>.py` and `configs.py`.
- Inspect the `pl.pallas_call` / `buffered_pallas_call`: `grid`, `BlockSpec` index maps, `scratch_shapes`,
  `pltpu.CompilerParams` (`dimension_semantics`, `vmem_limit_bytes`), and buffer counts.
- Inspect the kernel body: where `jax.lax.dot_general(..., preferred_element_type=jnp.float32, precision=...)` or
  `jnp.dot` feeds the MXU (note: TPU Mosaic kernels do **not** use `pl.dot` — `pl.dot` is Pallas GPU/Triton only), where
  masking/`pltpu` primitives run, and what stays resident in VMEM scratch vs re-read from HBM.

Freeze one representative `(shape, dtype, TPU generation, device count)` before tuning.

## Baseline & Hardware Policy

Before editing:

1. Own the TPU exclusively (libtpu holds a single-process lock; check `.agents/ops/OPS.md` if busy).
2. Record op, TPU generation (`v4`/`v5e`/`v5p`/`v6e`), device count, shape, dtype, config, and exact command/env.
3. Run parity against `Platform.XLA` on TPU:
   ```bash
   env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
     EJKERNEL_AUTOTUNE_POLICY=heuristics \
     uv run pytest libs/ejkernel/test/kernels/_pallas/tpu/test_<kernel>.py
   ```
4. Capture a steady-state baseline via `libs/ejkernel/benchmarks/benchmark_suite.py` (with `EJKERNEL_AUTOTUNE_POLICY`
   and `~/ejkernel-presistent-cache/` / `EJKERNEL_PERSISTENT_CACHE_DIR` / `EJKERNEL_TUNED_DB` controlled identically on
   both sides):
   ```bash
   env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
     EJKERNEL_AUTOTUNE_POLICY=heuristics \
     EJKERNEL_BENCH_OPS=<kernel> \
     EJKERNEL_BENCH_PLATFORMS=xla,pallas \
     EJKERNEL_BENCH_CONFIG_LIMIT=1 \
     EJKERNEL_BENCH_WARMUP=5 \
     EJKERNEL_BENCH_ITERS=30 \
     EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench_baseline \
     uv run python libs/ejkernel/benchmarks/benchmark_suite.py
   ```

CPU is strictly for non-computation host preflight (imports, syntax, registry shape). Parity tests (reference math
included) and benchmarks MUST run on TPU. If the TPU is unavailable or busy, stop, route through `.agents/ops/OPS.md`,
and report the change as **unverified on hardware**.

## Measure, Don't Guess

Pick the tool that answers your hypothesis:

- **Static roofline:** run `jax.jit(xla_ref).lower(*args).compile().cost_analysis()` on the **XLA reference** to obtain
  theoretical FLOPs, bytes accessed, and arithmetic intensity. XLA `cost_analysis()` treats Pallas `tpu_custom_call` as
  opaque (missing inner FLOPs/bytes), so compute the roofline from the XLA reference and compare measured Pallas latency
  against it.
- **Timeline / xprof:** `jax.profiler.start_trace(dir)` … `out.block_until_ready()` … `stop_trace()` (or
  `ejkernel.loggings.{ignite_profiler,extinguish_profiler,create_step_profiler}` /
  `ejkernel.ops.execution.profiler.Profiler`) to verify time is spent inside the kernel rather than surrounding
  transposes, copies, or DMA gaps.
- **HLO dump:** `XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_as_text"` (with `EJKERNEL_OPS_STAMP=hash|json`) to confirm
  the expected `tpu_custom_call` and fusion boundaries.
- **LLO / Mosaic dump:** when Pallas is slower than XLA on a fixed shape or sweeps are noisy, follow the canonical dump
  workflow in `.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` to compare
  `*schedule-analysis_final_bundles.txt` and `vrot`/`vsel`/`vpow2`/spill/gap counters between XLA and Pallas.

See `.agents/skills/optimize-pallas-tpu/docs/reference/profiling.md` for Pallas TPU profiling commands and
`.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` for the canonical LLO/Mosaic flag block.

## Attribute Before Optimizing

- **Memory-bound (HBM wall, low arithmetic intensity, MXU idle):** increase reuse via larger blocks, keep largest
  operand stationary in VMEM across inner grid steps, fuse adjacent ops, stream `bf16` instead of `f32`.
- **Compute-bound (MXU saturated):** eliminate padding by aligning contracting/output dims to 128 (256 on v6e) and
  sublanes to 8; cut redundant FLOPs.
- **VPU-bound (heavy `vrot`/`vsel`/`vpow2` in LLO):** eliminate per-element masking/rotations, precompute masks, move
  work onto the MXU, or reduce transcendentals.
- **Latency / overhead-bound (timeline shows surrounding transposes, copies, or serial DMA gaps):** fix layout around
  the call or DMA pipelining rather than inner-loop arithmetic.

If LLO shows large schedule-bundle inflation versus XLA, fix the **structural decomposition** before sweeping block
sizes.

## Optimization Levers (Roughly By Impact)

1. **Cut HBM traffic:** order the grid so the largest operand's `BlockSpec` index changes slowest; keep accumulators in
   `pltpu.VMEM` scratch instead of round-tripping to HBM; fuse producer/consumer stages.
2. **Tile deliberately for MXU + VMEM:** align feature/contracting dims to 128 (256 on v6e) and sublane dims to 8. Ensure
   all resident buffers plus double/triple buffering (`~2-3×` streamed inputs) fit within `vmem_limit_bytes` without
   VMEM OOM or silent spills.
3. **Exploit Megacore on v4/v5p:** mark independent grid axes `"parallel"` and dependent/accumulator axes `"arbitrary"`
   in `pltpu.CompilerParams(dimension_semantics=...)`.
4. **Use `bf16` MXU inputs with `f32` accumulation:** `jax.lax.dot_general(..., preferred_element_type=jnp.float32)`
   halves HBM/VMEM bandwidth while preserving accumulation accuracy (verify tolerances).
5. **Explicit buffering / pipelining only when the default grid pipeline falls short:** use `buffered_pallas_call`
   (`input_buffer_count`), `pl.Buffered(buffer_count=..., use_lookahead=True)`, or `pltpu.emit_pipeline` /
   `pltpu.make_async_copy`. Async/DMA complexity only counts when the target TPU run proves both overlap and a
   wall-clock win.
6. **Eliminate VPU hotspots:** remove unnecessary `vrot`/`vsel`/`vpow2` sequences identified in LLO dumps.

## Optimization Loop & Correctness Gate

1. Freeze `(shape, dtype, TPU generation, device count)`.
2. Attribute the bottleneck from roofline + trace + (when needed) LLO dump.
3. Change **one** lever (block size, grid order, `dimension_semantics`, `vmem_limit_bytes`, buffering, dtype, or
   decomposition).
4. Re-run TPU parity against `Platform.XLA` (values, gradients for differentiable ops, finite checks, non-aligned edge
   shapes at dtype-appropriate tolerances).
5. Re-run the exact same steady-state benchmark on TPU.
6. Keep only changes that win on hardware without breaking parity; revert losers and record why.

## Common Mistakes

- Writing `pl.dot` in TPU Mosaic kernels (use `jax.lax.dot_general` or `jnp.dot`).
- Assuming 128×128 MXU tiles are optimal on TPU v6e (v6e MXU is 256×256) or assuming Megacore splits work on single-core
  v5e/v6e.
- Trusting `compiled.cost_analysis()` FLOPs directly on an opaque Pallas `tpu_custom_call` instead of the XLA reference.
- Sweeping block sizes before comparing LLO schedule bundles against XLA.
- Exceeding VMEM once double/triple-buffered inputs are counted, causing VMEM OOM or silent spills.
- Leaving `~/ejkernel-presistent-cache/` or `EJKERNEL_AUTOTUNE_POLICY` uncontrolled between baseline and candidate.
- Claiming DMA/async wins without measured steady-state TPU numbers, or substituting CPU runs when TPU is busy.

## Definition Of Done

- Baseline and candidate commands, env vars, TPU generation, device count, shape, dtype, warmup, and iters are recorded.
- Parity passes on TPU across representative and non-aligned edge shapes.
- Bottleneck class is named and backed by roofline, trace, or HLO/LLO evidence.
- Direct baseline-vs-candidate steady-state numbers (and artifact paths) are reported; losing variants are removed.
