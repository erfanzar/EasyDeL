# TPU Dump Diagnostics Reference

Use this when TPU Pallas performance is unclear and wall-clock timing alone
does not explain the result.

## Contents

- When to use this reference
- Verification status
- HLO dump setup
- TPU LLO/Mosaic setup
- Variant matrix
- What to inspect
- XLA to Pallas replication loop
- Reporting template
- Common mistakes

## When To Use This Reference

Use dump-driven diagnosis when:

- Pallas is slower than XLA on one fixed shape.
- Block-size or config sweeps are noisy or do not improve throughput.
- A compiler/runtime failure is shape-dependent.
- A decomposition variant behaves very differently from the full kernel.
- DMA/async changes appear correct but do not produce a measured win.

Keep one fixed shape, dtype, backend, and device count while diagnosing.

## Verification Status

`XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_as_text` can be checked on CPU for
flag spelling, but CPU dumps are not TPU Pallas diagnostics.

This workspace does not ship a wrapper for TPU LLO/Mosaic dumps. Treat
`LIBTPU_INIT_ARGS` dump flags as target-TPU diagnostics: before a large run,
run one tiny TPU compile with the exact flags and record whether that TPU
runtime accepts them.

## HLO Dump Setup

Create separate dump roots per variant:

```bash
ROOT=/tmp/ejkernel_dumps/gated_delta_rule
VARIANT=xla
mkdir -p "${ROOT}/${VARIANT}/hlo"

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
XLA_FLAGS="--xla_dump_to=${ROOT}/${VARIANT}/hlo --xla_dump_hlo_as_text" \
EJKERNEL_BENCH_OPS=gated_delta_rule \
EJKERNEL_BENCH_PLATFORMS=xla \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=1 \
EJKERNEL_BENCH_ITERS=1 \
EJKERNEL_BENCH_OUTPUT_DIR="${ROOT}/${VARIANT}/bench" \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Run this only when this process owns the TPU. If you only need to verify HLO
dump flag spelling on a non-TPU host, you may run a separate CPU preflight, but
do not include those dumps in TPU diagnosis.

## TPU LLO/Mosaic Setup

Use this only after a target TPU smoke compile accepts the flags:

```bash
ROOT=/tmp/ejkernel_dumps/gated_delta_rule
VARIANT=pallas
mkdir -p "${ROOT}/${VARIANT}/hlo" "${ROOT}/${VARIANT}/llo" "${ROOT}/${VARIANT}/mosaic"

export XLA_FLAGS="--xla_dump_to=${ROOT}/${VARIANT}/hlo --xla_dump_hlo_as_text"
export LIBTPU_INIT_ARGS="\
  --xla_jf_dump_to=${ROOT}/${VARIANT}/llo \
  --xla_jf_dump_hlo_text=true \
  --xla_jf_dump_llo_text=true \
  --xla_jf_dump_llo_html=false \
  --xla_jf_dump_llo_static_gaps=true \
  --xla_jf_emit_annotations=true \
  --xla_jf_debug_level=2 \
  --xla_mosaic_dump_to=${ROOT}/${VARIANT}/mosaic \
  --xla_mosaic_enable_dump_debug_info=true \
  --xla_mosaic_enable_llo_source_annotations=true"

EJKERNEL_BENCH_OPS=gated_delta_rule \
EJKERNEL_BENCH_PLATFORMS=pallas \
EJKERNEL_BENCH_CONFIG_LIMIT=1 \
EJKERNEL_BENCH_WARMUP=1 \
EJKERNEL_BENCH_ITERS=1 \
EJKERNEL_BENCH_OUTPUT_DIR="${ROOT}/${VARIANT}/bench" \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Keep `LIBTPU_INIT_ARGS` identical between compared TPU variants. If scoped
VMEM flags are part of the experiment, record them and keep them fixed for the
matrix.

## Variant Matrix

For a high-signal comparison, run at least:

1. XLA/reference implementation with `EJKERNEL_BENCH_PLATFORMS=xla`.
2. Current Pallas implementation with `EJKERNEL_BENCH_PLATFORMS=pallas`.
3. One decomposition variant that removes or isolates a suspected stage.

Use the same benchmark spec, config limit, shape, dtype, warmup, iterations,
and target TPU type. Do not compare CPU XLA against TPU Pallas for correctness,
lowering, or performance.

## What To Inspect

HLO:

- Confirm the expected implementation path is used.
- Check custom-call/fusion placement.
- Check aliasing and update patterns.
- Confirm a decomposition variant actually removed the intended stage.

LLO schedule summaries:

- Search dump outputs for `*schedule-analysis_final_bundles.txt`.
- Compare total bundle count and non-empty bundle count between variants.
- Large bundle inflation versus XLA points to structural issues before tiling.

LLO pressure signals:

- Compare lane-rotation-heavy ops such as `vrot`.
- Compare select/mask-heavy ops such as `vsel`.
- Compare transcendental-heavy ops such as `vpow2`.
- Look for spills, register pressure, and schedule gaps.

Mosaic/source annotations:

- Map expensive generated kernels back to source locations.
- Check whether errors or pressure align with the suspected Pallas block.

## XLA To Pallas Replication Loop

1. Freeze one representative shape.
2. Dump XLA and current Pallas first.
3. Infer stage boundaries from XLA fusion structure.
4. Replicate the dominant stage first.
5. Add missing stages incrementally.
6. After each addition, run correctness, throughput, and dumps.
7. Use pressure counters to find the first regressing step.
8. Retune block sizes only after the structure is close.

## Reporting Template

Include:

- operation, platform, backend, hardware, and device count
- shape/dtype/config
- exact command and env vars
- benchmark JSON/Markdown paths
- HLO/LLO/Mosaic dump paths
- throughput table for XLA, Pallas, and decomposition variants
- schedule bundle comparison
- pressure-counter deltas
- next hypothesis

## Common Mistakes

- Mixing dump directories between variants.
- Comparing different effective shapes or device counts.
- Changing `LIBTPU_INIT_ARGS` between variants.
- Running broad block-size sweeps before checking structural regressions.
- Claiming a TPU result when libtpu was busy, the command fell back to CPU, or
  `JAX_PLATFORMS=tpu` was not used for the target run.
