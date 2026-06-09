# 01_basics — spectrax fundamentals

Minimal, runnable introductions to the spectrax core API. Every file
is self-contained (`python -m examples.01_basics.<name>`) and stays
well under 100 LOC.

| File                       | Concepts                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------- |
| `01_module_and_forward.py` | Defining a `spx.Module` subclass, PyTorch-style `__init__` + `__call__`, and running a forward pass.      |
| `02_training_loop.py`      | Hand-rolled SGD training loop with `spx.value_and_grad`, `spx.jit(mutable="parameters")`, and `spx.update`.   |
| `03_export_bind.py`        | Round-tripping a module via `spx.export` -> `(GraphDef, State)` and `spx.bind` back to a live module.     |
| `04_state_pop_update.py`   | State-pytree surgery: `spx.tree_state`, `spx.clone`, `spx.update`, and `spx.pop` on `Parameter`/`Buffer`. |
| `05_optimizer.py`          | Training with `spectrax.contrib.Optimizer` wrapping `optax.adam`, inside a jitted step.                   |
| `06_multi_optimizer_lora.py` | Separate optimizer policies for base `"parameters"` and LoRA `"lora"` collections with `MultiOptimizer`. |

All examples run on CPU with small sizes (`d=32`, `hidden=64`, `bs=8`)
and finish in a few seconds.
