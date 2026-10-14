# Optimizers

SpectraX keeps optimizers in `spectrax.contrib` so the core package
stays small while training loops still get a first-class optax bridge.

Install the development/test extra or the contrib extra:

```bash
pip install "spectrax-lib[contrib]"
```

From source, `test` and `dev` already include optax so the optimizer
examples and tests run by default:

```bash
uv sync --extra test
```

## `Optimizer`

`Optimizer` wraps an optax `GradientTransformation` and makes it a
JAX pytree. Its dynamic leaves are the optax state and step counter;
the transform and SpectraX selector are static metadata.

```python
import optax
import spectrax as spx
from spectrax.contrib import Optimizer

model = MyModel(rngs=spx.Rngs(0))
optimizer = Optimizer.create(model, optax.adamw(3e-4))


@spx.jit(mutable="parameters")
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        return ((model(x) - y) ** 2).mean()

    loss, grads = spx.value_and_grad(loss_fn)(model)
    parameters = spx.tree_state(model).filter("parameters")
    new_parameters, optimizer = optimizer.update(parameters, grads)
    spx.update(model, new_parameters)
    return loss, optimizer
```

Use the same `update` method when you want a fully functional step
that threads `State` explicitly and does not write back to a live
module:

```python
parameters = spx.tree_state(model).filter("parameters")
new_parameters, optimizer = optimizer.update(parameters, grads)
```

## Selector-Scoped State

The `wrt=` argument controls which variables allocate optimizer state.
For LoRA fine-tuning, this prevents Adam moments from being created
for frozen base weights:

```python
optimizer = Optimizer.create(model, optax.adamw(1e-3), wrt="lora")
```

Any selector accepted by `spx.grad(wrt=...)` works here too:

```python
optimizer = Optimizer.create(
    model,
    optax.adamw(1e-3),
    wrt=spx.select().variables("parameters").at_path("decoder.*"),
)
```

## `MultiOptimizer`

`MultiOptimizer` composes several `Optimizer` objects over disjoint
variable slices. This is useful for adapter training, per-collection
learning rates, or separate optimizer transforms for heads and trunks.

```python
from spectrax.contrib import MultiOptimizer

optimizer = MultiOptimizer.create(
    model,
    {
        "parameters": optax.sgd(1e-3),
        "lora": optax.adamw(2e-2),
    },
)


@spx.jit(mutable=("parameters", "lora"))
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        return ((model(x) - y) ** 2).mean()

    loss, grads = spx.value_and_grad(loss_fn, wrt=("parameters", "lora"))(model)
    parameters = spx.tree_state(model).filter("parameters", "lora")
    new_parameters, optimizer = optimizer.update(parameters, grads)
    spx.update(model, new_parameters)
    return loss, optimizer
```

See [`examples/01_basics/06_multi_optimizer_lora.py`](../../examples/01_basics/06_multi_optimizer_lora.py)
for a runnable LoRA example with separate base/adapter optimizer
policies.

## Choosing The Form

| API | Mutates live module | JIT-friendly | Best for |
| --- | --- | --- | --- |
| `Optimizer.update` | No | Yes | Pure functional training steps |
| `Optimizer.apply_eager` | Yes | No | Familiar eager loops |
| `MultiOptimizer.update` | No | Yes | Explicit state threading with multiple slices |
| `MultiOptimizer.apply_eager` | Yes | No | Adapter or multi-policy eager loops |

Inside `spx.jit`, prefer `update` plus `spx.update(...)` and declare
every collection that may be written in `spx.jit(mutable=...)`.
Reserve `apply_eager` for ordinary Python loops outside transforms.
