# TPU HLO / LLO / Mosaic Dump Diagnostics Reference

Canonical dump-driven diagnosis guide for TPU Pallas kernels across `.agents/skills/optimize-ejkernel-kernel/SKILL.md`,
`.agents/skills/add-ejkernel-kernel/SKILL.md`, and `.agents/skills/optimize-pallas-tpu/SKILL.md`. Pair with
`.agents/skills/optimize-pallas-tpu/SKILL.md` for TPU MXU (128×128 on v4/v5e/v5p, 256×256 on v6e), VPU (8×128), VMEM,
and Megacore roofline modeling.

## When To Use This Reference

Use dump-driven diagnosis when wall-clock timing alone does not explain TPU behavior:

- Pallas is slower than XLA on one fixed shape.
- Block-size or config sweeps are noisy or fail to improve throughput.
- A compiler or runtime failure is shape-dependent.
- A decomposition variant behaves sharply differently from the full kernel.
- DMA/async changes look correct but do not produce a measured speedup.

Freeze one shape, dtype, backend, TPU generation, and device count while diagnosing.

## Hardware & Runtime Notes

- Own the TPU exclusively for every dump run (`libtpu` holds a single-process lock; check `.agents/ops/OPS.md` if busy).
- Never run `benchmark_suite.py` or kernel numerics on CPU as a proxy for TPU lowering or HLO/LLO diagnosis.
- This workspace does not ship a wrapper for TPU LLO/Mosaic dumps: run one small TPU smoke compile with the exact
  `LIBTPU_INIT_ARGS` flags first and record whether that TPU runtime version accepts every flag.
- **Separate dump capture from throughput measurement:** capture dumps with `EJKERNEL_BENCH_WARMUP=1 EJKERNEL_BENCH_ITERS=1`,
  then run steady-state timing with `EJKERNEL_BENCH_WARMUP=5 EJKERNEL_BENCH_ITERS=30` and dump flags unset so dump I/O
  and debug annotations do not skew latency numbers.

## HLO Dump Setup

Use separate dump roots per variant. Add `EJKERNEL_OPS_STAMP=hash` (or `json`) to tag ejkernel op scopes inside large
HLO graphs:

```bash
ROOT=/tmp/ejkernel_dumps/gated_delta_rule
VARIANT=xla
mkdir -p "${ROOT}/${VARIANT}/hlo"

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu JAX_PLATFORM_NAME=tpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_OPS_STAMP=hash \
  XLA_FLAGS="--xla_dump_to=${ROOT}/${VARIANT}/hlo --xla_dump_hlo_as_text" \
  EJKERNEL_BENCH_OPS=gated_delta_rule \
  EJKERNEL_BENCH_PLATFORMS=xla \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=1 \
  EJKERNEL_BENCH_ITERS=1 \
  EJKERNEL_BENCH_OUTPUT_DIR="${ROOT}/${VARIANT}/bench_dump" \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

## TPU LLO / Mosaic Setup

Run after confirming the TPU runtime accepts these `LIBTPU_INIT_ARGS` flags. Keep `LIBTPU_INIT_ARGS` identical across
all compared variants (including any scoped VMEM flags):

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

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu JAX_PLATFORM_NAME=tpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_OPS_STAMP=hash \
  EJKERNEL_BENCH_OPS=gated_delta_rule \
  EJKERNEL_BENCH_PLATFORMS=pallas \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=1 \
  EJKERNEL_BENCH_ITERS=1 \
  EJKERNEL_BENCH_OUTPUT_DIR="${ROOT}/${VARIANT}/bench_dump" \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

After capturing dumps, unset `XLA_FLAGS` and `LIBTPU_INIT_ARGS` and run the steady-state throughput matrix
(`EJKERNEL_BENCH_WARMUP=5 EJKERNEL_BENCH_ITERS=30`) on the exact same frozen shape/config.

## Variant Matrix

Run at least three variants on the identical shape, dtype, config limit, and TPU device:

1. XLA reference (`EJKERNEL_BENCH_PLATFORMS=xla`).
2. Current Pallas implementation (`EJKERNEL_BENCH_PLATFORMS=pallas`).
3. One decomposition variant that removes or isolates a suspected bottleneck stage.

## What To Inspect

- **HLO (`*after_optimizations.txt`):** confirm the expected dispatch path (`tpu_custom_call` vs XLA fusion), check
  `ejkernel_ops#...` labels (`EJKERNEL_OPS_STAMP`), verify buffer aliasing/updates, and verify a decomposition variant
  actually removed the intended stage.
- **LLO schedule summaries (`*schedule-analysis_final_bundles.txt`):** compare **total bundle count** and **non-empty
  bundle count** between XLA and Pallas. Large bundle inflation vs XLA signals a structural issue — fix decomposition
  before sweeping block sizes.
- **LLO pressure counters:** compare counts across variants for:
  - `vrot` — lane rotations (layout/transpose overhead)
  - `vsel` — selects/masking (boundary/mask overhead)
  - `vpow2` and friends — transcendentals
  - spills/fills and static schedule gaps — register/VMEM pressure and stalls
- **Mosaic / source annotations:** map expensive LLO blocks back to kernel lines to verify whether pressure aligns with
  the suspected stage.

## XLA-To-Pallas Replication Loop

1. Freeze one representative shape and dump both XLA and current Pallas.
2. Infer stage boundaries from XLA's fusion structure.
3. Replicate the dominant stage first in Pallas.
4. Add remaining stages incrementally; after each addition, check parity, steady-state throughput, and dumps.
5. Use bundle and pressure-counter deltas to pinpoint the first regressing stage.
6. Retune block sizes only after the structure matches or beats XLA.

## Reporting Template

Include:
- Operation, platform, backend, TPU generation, and device count
- Frozen shape, dtype, and config
- Exact commands and env vars (dump run and clean steady-state run separately)
- Benchmark JSON/Markdown paths and HLO/LLO/Mosaic dump paths
- Steady-state throughput table (XLA, Pallas, decomposition variants — from clean `WARMUP=5 / ITERS=30` runs)
- Schedule bundle counts and pressure-counter deltas (`vrot`, `vsel`, `vpow2`, spills, gaps)
- Next hypothesis

## Common Mistakes

- Populating the throughput table from `WARMUP=1 ITERS=1` dump runs instead of clean steady-state runs.
- Omitting `ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu JAX_PLATFORM_NAME=tpu` and silently falling back to CPU.
- Mixing dump roots or changing `LIBTPU_INIT_ARGS` between compared variants.
- Sweeping block sizes before checking schedule-bundle inflation against XLA.
- Claiming TPU results when `libtpu` was busy or unavailable (report as **unverified on hardware**).
