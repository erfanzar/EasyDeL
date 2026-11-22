# 04 — Model Surgery

Hands-on recipes for editing a live spectrax :class:`Module`:
selectors, LoRA injection, FP8 casts, gradient freezing, and
attribute / state-level module swaps.

| File                   | Topic                                                              |
| ---------------------- | ------------------------------------------------------------------ |
| `01_selectors.py`      | `spx.find`, `iter_variables`, `iter_modules`, selector combinators |
| `02_lora_injection.py` | `spx.nn.wrap_lora` over pretrained linears                         |
| `03_fp8_cast.py`       | FP32 -> `spx.nn.Fp8Linear`, meta inspection                        |
| `04_freeze_params.py`  | Grad masking via selector-derived frozen paths                     |
| `05_module_swap.py`    | Attribute rebinding + `spx.update` / `spx.pop` / `spx.clone`       |

Run any of them from the repo root, e.g.::

    python -m examples.04_surgery.01_selectors
