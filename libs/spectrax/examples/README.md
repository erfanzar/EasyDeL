# spectrax examples

Seven topic folders, 5+ runnable scripts each, progressing from
single-Module forward passes to multi-device MPMD pipeline training.

| folder                                                 | topic                                                                                                               |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| [`01_basics/`](01_basics/)                             | Defining `spx.Module`, training loops, `export`/`bind`, state manipulation, single and multi optimizers.            |
| [`02_implementation_guide/`](02_implementation_guide/) | Full model implementations — Llama 3, Qwen 2, GPT-2, ViT, custom transformer block.                                 |
| [`03_transformations/`](03_transformations/)           | `spx.jit`, `spx.grad`/`value_and_grad`, `spx.vmap`, `spx.remat` (function + class-aware), `spx.scan` / `fori_loop`. |
| [`04_surgery/`](04_surgery/)                           | Selectors (`find`, `iter_*`, `of_type`, `path_*`), LoRA injection, FP8 cast, parameter freezing, module swapping.   |
| [`05_shardings/`](05_shardings/)                       | FSDP, tensor-parallel, FSDP+TP hybrid, logical axis rules, `with_sharding_constraint_by_name`.                      |
| [`06_spmd_scheduled/`](06_spmd_scheduled/)             | SPMD pipeline runtime with GPipe, Std1F1B, ZeroBubbleH1, InterleavedH1 (virtual stages).                            |
| [`07_mpmd/`](07_mpmd/)                                 | Real MPMD pipeline via `spx.run` / `sxjit` — train, forward, decode, stage regions, stage-local meshes.             |

Shared model implementations live in [`models/`](models/):

- [`models/llama.py`](models/llama.py) — Llama 3 (GQA + RoPE + SwiGLU + RMSNorm) with role-specific FSDP+TP annotations.
- [`models/qwen.py`](models/qwen.py) — Qwen 2 (QKV bias, `rope_theta=1_000_000`).

## Running

```bash
python -m examples.01_basics.02_training_loop
python -m examples.01_basics.06_multi_optimizer_lora
python -m examples.02_implementation_guide.01_llama3
python -m examples.07_mpmd.01_train_homogeneous
```

Most examples run on CPU (small configs); sharding and pipeline
examples benefit from multi-device TPU / GPU but fall back cleanly
to 1 device.

## The 30-second tour

```python
import spectrax as spx
from spectrax.sharding import logical_axis_rules
from examples.models.llama import Llama3, Llama3Config, FSDP_TP_RULES

cfg  = Llama3Config(d_model=256, n_heads=4, n_kv_heads=2, ffn=512, n_layers=4)
mesh = spx.create_mesh(axis_dims=(2, 1, -1, 1, 1, 1), mpmd_axis="pp")

with logical_axis_rules(FSDP_TP_RULES), mesh:
    model = Llama3(cfg, rngs=spx.Rngs(0))
    loss, grads = spx.run(model, inputs=ids, targets=labels,
                          mesh=mesh, mode="train", loss_fn=ce,
                          microbatches=4)
```

Drop `mpmd_axis="pp"` and the same model runs under pure SPMD —
`spx.run` dispatches on the mesh.

## `spx.run` signature

```python
spx.run(
    model,                      # any spx.Module
    *,
    inputs,                     # array | tuple | dict
    targets=None,               # array | tuple | dict — passed to loss_fn
    mesh,                       # SpxMesh — mesh decides SPMD vs MPMD
    mode="forward",             # "train" | "forward"
    loss_fn=None,               # required for mode="train"
    microbatches=1,             # ignored under SPMD
)
```

- `inputs=ids`            → `model.forward(ids)`
- `inputs=(ids, mask)`    → `model.forward(ids, mask)`
- `inputs={'ids': ids,'mask': m}` → `model.forward(ids=ids, mask=m)`

Same rules for `targets` against `loss_fn`.
