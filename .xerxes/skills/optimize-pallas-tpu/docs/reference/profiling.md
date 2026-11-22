# Pallas TPU Profiling Cheatsheet

Exact commands, env vars, and dump-reading recipes for the workflow in
`SKILL.md`. Everything here assumes you own the TPU for the run. CPU is for host-side preflight only (imports, shapes,
reference math, flag spelling) and is never evidence for TPU behavior.

## 1. Predict before you run: static cost model

```python
compiled = jax.jit(f).lower(*args).compile()
print(compiled.cost_analysis())     # flops, bytes accessed, transcendentals
print(compiled.memory_analysis())   # argument/output/temp/alias sizes
print(jax.jit(f).lower(*args).as_text())  # StableHLO/HLO inline
```

- **Arithmetic intensity** = `flops / bytes accessed`. Compare to the TPU's FLOP:byte ratio (hundreds:1 for modern
  parts). Below it → memory-bound; above → compute-bound. This tells you which lever class to reach for *before* timing.
- `memory_analysis()` flags when temporaries blow up — a sign of missing fusion.

## 2. Timeline / trace (xprof)

```python
jax.profiler.start_trace("/tmp/tpu_trace")
for _ in range(20):
    out = f(*args)
out.block_until_ready()             # MUST block before stopping
jax.profiler.stop_trace()
# or: with jax.profiler.trace("/tmp/tpu_trace"): ...
```

```bash
tensorboard --logdir /tmp/tpu_trace        # open the "Profile" tab
```

Read, in order:

- **trace_viewer** — the device timeline. Is time in your kernel, or in surrounding transposes/reshapes/copies? Are DMAs
  overlapping compute, or is there a serial gap before each step (failed pipeline)?
- **op_profile** — per-op time ranking; find the real hot op.
- **memory_viewer** — HBM high-water mark and fragmentation.

Always profile **steady state**: discard the first call (it compiles).

## 3. HLO dump — confirm the path

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu \
XLA_FLAGS="--xla_dump_to=/tmp/dumps/xla/hlo --xla_dump_hlo_as_text" \
  python run.py
```

Inspect `*after_optimizations.txt`. Confirm the expected custom-call / fusion is emitted and your kernel is actually on
the path. Add `--xla_dump_hlo_as_html`
for a navigable fusion graph. CPU may be used only to check flag *spelling*.

## 4. TPU LLO / Mosaic dump — structural diagnosis

Use when Pallas is mysteriously slower than XLA on one fixed shape, or sweeps are noisy. **Run one tiny TPU smoke
compile with these exact flags first** to confirm the runtime accepts them. Keep a separate dump root per variant and
keep
`LIBTPU_INIT_ARGS` identical between compared variants.

```bash
ROOT=/tmp/dumps/mykernel; VARIANT=pallas
mkdir -p "$ROOT/$VARIANT"/{hlo,llo,mosaic}

export XLA_FLAGS="--xla_dump_to=$ROOT/$VARIANT/hlo --xla_dump_hlo_as_text"
export LIBTPU_INIT_ARGS="\
  --xla_jf_dump_to=$ROOT/$VARIANT/llo \
  --xla_jf_dump_hlo_text=true \
  --xla_jf_dump_llo_text=true \
  --xla_jf_dump_llo_static_gaps=true \
  --xla_jf_emit_annotations=true \
  --xla_jf_debug_level=2 \
  --xla_mosaic_dump_to=$ROOT/$VARIANT/mosaic \
  --xla_mosaic_enable_dump_debug_info=true \
  --xla_mosaic_enable_llo_source_annotations=true"

ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu python run.py
```

### What to read in the dump

- **Schedule bundles** — find `*schedule-analysis_final_bundles.txt`. Compare **total bundle count** and **non-empty
  bundle count** between Pallas and XLA. Large inflation vs XLA = a structural problem; fix the decomposition before
  touching block sizes.
- **VPU pressure counters** — grep the LLO for op mnemonics and compare counts between variants:
    - `vrot` — lane rotations (layout/transpose cost)
    - `vsel` — selects/masking (boundary/mask cost)
    - `vpow2` and friends — transcendentals
    - spill / fill and **static schedule gaps** — stalls and register pressure.
- **Mosaic source annotations** — map the expensive generated block back to the line of your kernel; check whether the
  pressure aligns with the block you suspected.

## 5. Debugging vs profiling

- `pl.pallas_call(..., interpret=True)` runs the kernel in pure-JAX on the host for **correctness** debugging — not a
  performance tool.
- `jax.debug.print` / `pl.debug_print` inside the kernel for value inspection.

## 6. Reporting checklist

- TPU generation (v4/v5e/v5p/v6e) and device count
- shape, dtype, and the frozen config
- steady-state timing (label it as such — not compile-including)
- baseline vs candidate, same warmup/iters/shape/device
- trace path, HLO dump path, and (if used) LLO/Mosaic dump paths
- bundle-count and pressure-counter deltas when timing alone didn't explain it

## Common mistakes

- Forgetting `block_until_ready()` before `stop_trace()` → truncated trace.
- Profiling the first (compiling) call and calling it runtime.
- Mixing dump roots or changing `LIBTPU_INIT_ARGS` between variants.
- Treating a CPU HLO dump as TPU diagnosis.
- Sweeping block sizes before checking the schedule-bundle structure.
