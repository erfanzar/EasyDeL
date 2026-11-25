# Pallas TPU Profiling Cheatsheet

Exact commands, env vars, and ejkernel profiling hooks for `.agents/skills/optimize-pallas-tpu/SKILL.md`. Everything
below assumes this process owns the TPU exclusively (see `.agents/ops/OPS.md` if `libtpu` is busy). CPU is strictly for
non-computation host preflight (imports, syntax, static config inspection) — never for JAX reference math, parity, or
timing (`libs/ejkernel/test/conftest.py` enforces this).

## 1. Roofline Before Timing: Static Cost Model

Run `cost_analysis()` on the **XLA reference** (`Platform.XLA`), not only on the Pallas call: XLA treats Pallas
`tpu_custom_call` as opaque and reports zero/incomplete FLOPs and bytes for the kernel body.

```python
compiled_ref = jax.jit(xla_ref_fn).lower(*args).compile()
print(compiled_ref.cost_analysis())     # theoretical flops, bytes accessed, transcendentals
print(compiled_ref.memory_analysis())   # argument/output/temp/alias sizes

compiled_pallas = jax.jit(pallas_fn).lower(*args).compile()
print(compiled_pallas.memory_analysis())  # catches HBM temp blowups around the custom call
print(jax.jit(pallas_fn).lower(*args).as_text())  # StableHLO/HLO inline
```

- **Arithmetic intensity** = `flops / bytes accessed` (from `compiled_ref`). Compare against your TPU generation's
  FLOP:byte ridge point to classify memory-bound vs compute-bound before benchmarking.
- Evaluate measured Pallas wall-clock latency against that theoretical roofline.

## 2. Benchmarking & Profiling In eJKernel

Use the registry-driven suite with deterministic config selection (`EJKERNEL_AUTOTUNE_POLICY=heuristics`) and the
standard accelerator env:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  EJKERNEL_AUTOTUNE_POLICY=heuristics \
  EJKERNEL_BENCH_OPS=<kernel> \
  EJKERNEL_BENCH_PLATFORMS=xla,pallas \
  EJKERNEL_BENCH_CONFIG_LIMIT=1 \
  EJKERNEL_BENCH_WARMUP=5 \
  EJKERNEL_BENCH_ITERS=30 \
  EJKERNEL_BENCH_OUTPUT_DIR=/tmp/ejkernel_bench \
  uv run python libs/ejkernel/benchmarks/benchmark_suite.py
```

Key ejkernel profiling & benchmarking surfaces:

- `libs/ejkernel/ejkernel/benchmarks.py` (`Benchmark`) — warmup + timed steady-state iterations, JSON export.
- `libs/ejkernel/ejkernel/ops/execution/profiler.py` (`Profiler`) — JAX trace parsing and event timing.
- `libs/ejkernel/ejkernel/loggings.py` — `ignite_profiler(profile_path, enable_perfetto=False)`,
  `extinguish_profiler(enable_perfetto)` (note: `enable_perfetto` is a **required** boolean arg), and
  `create_step_profiler(...)` for stepped loops.
- `EJKERNEL_OPS_STAMP=hash|json` — stamps ejkernel operation labels into HLO/xprof scopes for attribution.

See `.agents/skills/optimize-ejkernel-kernel/docs/reference/profiling.md` for full benchmark suite env vars and
EasyDeL trainer profiling controls.

## 3. Device Timeline / xprof Trace

Always discard the first call (compile) and call `block_until_ready()` before stopping the trace:

```python
out = f(*args)
out.block_until_ready()  # warmup / compile

jax.profiler.start_trace("/tmp/tpu_trace")
for _ in range(20):
    out = f(*args)
out.block_until_ready()
jax.profiler.stop_trace()
```

Open in TensorBoard / xprof (`tensorboard --logdir /tmp/tpu_trace`) and inspect in order:

1. **trace_viewer** — confirm time is inside the Pallas custom call (not surrounding transposes/copies) and check for
   DMA stall gaps.
2. **op_profile** — per-op time ranking.
3. **memory_viewer** — HBM high-water mark and buffer fragmentation.

## 4. HLO Dump — Verify Dispatch Path

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu JAX_PLATFORM_NAME=tpu \
  EJKERNEL_OPS_STAMP=hash \
  XLA_FLAGS="--xla_dump_to=/tmp/dumps/pallas/hlo --xla_dump_hlo_as_text" \
  uv run python <script.py>
```

Inspect `*after_optimizations.txt` to confirm the expected `tpu_custom_call`, input/output aliasing, and fusion
boundaries. Add `--xla_dump_hlo_as_html` for an interactive fusion graph.

## 5. TPU LLO / Mosaic Dump — Structural Diagnosis

When Pallas is unexpectedly slower than XLA on a fixed shape or block sweeps are noisy, capture HLO + LLO + Mosaic dumps
and compare `*schedule-analysis_final_bundles.txt` and VPU pressure counters (`vrot`, `vsel`, `vpow2`, spills, gaps)
against XLA.

Use the single canonical flag block, variant matrix, and bundle-reading guide in
`.agents/skills/optimize-ejkernel-kernel/docs/reference/llo.md` (do not duplicate or drift the `LIBTPU_INIT_ARGS` flags
here). Keep dump capture (`WARMUP=1`, `ITERS=1`) separate from steady-state timing runs (`WARMUP=5`, `ITERS=30` without
dump flags).

## 6. Correctness Debugging vs Profiling

- `pl.pallas_call(..., interpret=True)` (or `interpret=pltpu.InterpretParams(...)`) executes the kernel body via pure
  JAX for **correctness** debugging — never for performance.
- `pl.debug_print` / `jax.debug.print` inspect tile values inside the kernel.
- Under pytest, `libs/ejkernel/test/conftest.py` blocks CPU execution for any test file importing JAX/ejkernel unless
  marked `@pytest.mark.cpu_ok`; run parity and `interpret=True` debugging on TPU.

## Reporting Checklist & Common Mistakes

Include TPU generation, device count, shape, dtype, config, steady-state warmup/iters (distinct from compile time),
baseline-vs-candidate numbers, and artifact/dump paths.

- Forgetting `block_until_ready()` before `stop_trace()` → truncated trace.
- Reading `cost_analysis()` FLOPs off an opaque Pallas custom call instead of the XLA reference.
- Reporting 1-iteration dump runs as steady-state throughput.
- Mixing dump roots, changing `LIBTPU_INIT_ARGS`, or leaving `~/ejkernel-presistent-cache/` inconsistent between
  variants.
