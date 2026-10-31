# spectrax transforms — runnable examples

Five small CPU-friendly files demonstrating spectrax's module-aware
JAX transforms. Each transform is a drop-in for its `jax.*` counterpart
that understands `spx.Module` arguments and their `State`.

Run any file with `python -m examples.03_transformations.<name>`
from the repo root.

## Index

| File                                         | Transforms                                              |
| -------------------------------------------- | ------------------------------------------------------- |
| [`01_jit.py`](01_jit.py)                     | `spx.jit` — compile, cache, donate                      |
| [`02_grad.py`](02_grad.py)                   | `spx.grad`, `spx.value_and_grad` — scalar + vector      |
| [`03_vmap.py`](03_vmap.py)                   | `spx.vmap` — batch a Module with `in_axes` / `out_axes` |
| [`04_remat.py`](04_remat.py)                 | `spx.remat` — function-style and class-style checkpoint |
| [`05_scan_and_fori.py`](05_scan_and_fori.py) | `spx.scan` over stacked modules + `spx.fori_loop`       |
