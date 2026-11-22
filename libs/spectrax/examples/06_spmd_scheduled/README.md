# 06 — Pipeline Parallelism with Schedules

Pipeline-parallel training driven by `spectrax.pipeline.mpmd_call`
with a `spectrax.pipeline.Schedule`. The MPMD runtime compiles one
jit per stage, placed on its sub-mesh, and orchestrates the
schedule in Python with cross-stage transfers via `jax.device_put`.

For the decorator-based marker path (model-agnostic function with
inline `sxstage_iter`), see `../07_mpmd/11_mpmd_jit_generation.py`.

## Examples

| File                       | Schedule             | Point                                                         |
| -------------------------- | -------------------- | ------------------------------------------------------------- |
| `01_bare_spmd_pipeline.py` | `GPipe`              | Minimal setup: `PipelineSequential` + `mpmd_call`.            |
| `02_gpipe.py`              | `GPipe`              | Train one step on a small transformer block stack.            |
| `03_1f1b.py`               | `GPipe` vs `Std1F1B` | Side-by-side step-time comparison.                            |
| `04_zerobubble.py`         | `ZeroBubbleH1`       | BWD_I / BWD_W split; bubble shrinkage.                        |
| `05_virtual_stages.py`     | `InterleavedH1`      | `V > 1`: each rank holds `V` logical stages; prints the grid. |

## Device handling

Every example sizes the pp mesh axis to whatever devices are
visible (`axis_dims=(-1,)`). When `pp` matches `num_stages`, SPMD
runs for real; otherwise the example prints a note and falls back
to eager `model(x)` so the file still exits 0 on a single device.

Simulate more CPU devices locally with::

    XLA_FLAGS="--xla_force_host_platform_device_count=4" \
      JAX_PLATFORMS=cpu python -m examples.06_spmd_scheduled.02_gpipe
