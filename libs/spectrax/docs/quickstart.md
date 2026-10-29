# Quickstart

This page walks from `pip install spectrax-lib` to a complete training
loop with checkpointing in about 100 lines of code. Each section
introduces one concept at a time; if you already know one, skip it.

## Install

```bash
pip install spectrax-lib                    # core (CPU)
pip install "spectrax-lib[contrib]"         # adds optax integration
pip install "spectrax-lib[cuda]"            # CUDA jaxlib (H100, A100, ...)
pip install "spectrax-lib[tpu]"             # TPU jaxlib
pip install "spectrax-lib[docs]"            # build the docs locally
```

From source with [uv](https://docs.astral.sh/uv/) (recommended for
development):

```bash
git clone https://github.com/erfanzar/spectrax
cd spectrax
uv sync --extra dev --extra test
uv run pytest -q                        # 1700+ tests, about 3 min on CPU
```

### Requirements

| Item     | Version             |
|----------|---------------------|
| Python   | ≥ 3.11, ≤ 3.13      |
| JAX      | ≥ 0.10.0            |
| jaxlib   | ≥ 0.10.0            |
| numpy    | ≥ 1.26              |
| treescope| ≥ 0.1.7             |
| optax    | ≥ 0.2.8 (optional)  |

AGPL-3.0-or-later. Project is `v0.1.0` (alpha) — pin the version if you
depend on behavioral stability.

## A first model

```python
import jax.numpy as jnp
import spectrax as spx
from spectrax import nn
from spectrax import functional as F


class MLP(spx.Module):
    """Two-layer MLP with GELU."""

    def __init__(self, d_in, d_hidden, d_out, *, rngs):
        super().__init__()                          # MUST come before any attr assign
        self.fc1 = nn.Linear(d_in, d_hidden, rngs=rngs)
        self.fc2 = nn.Linear(d_hidden, d_out, rngs=rngs)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


rngs = spx.Rngs(0)
model = MLP(16, 64, 4, rngs=rngs)

x = jnp.ones((8, 16))
y = model(x)                                        # eager call, no compile
print(y.shape)                                      # (8, 4)
```

What's happening line by line:

- `spx.Rngs(0)` seeds an RNG source. Layers pull fresh keys from
  `rngs` during `__init__` (e.g. for parameter initialization). The
  RNG state lives in the `"rng"` collection.
- `spx.Module` subclasses record their submodule / variable
  attribute order, making the module a JAX pytree whose leaves are
  the raw weight / bias arrays.
- `model(x)` runs `forward` eagerly — no compile, no tracing — great
  for stepping through with `pdb`.

## Inspecting the model

```python
spx.inspect.summary(model, jnp.zeros((1, 16)))
# ┌────────┬────────────┬─────────────┬───────┐
# │ path   │ module     │ output      │ #parameters │
# ├────────┼────────────┼─────────────┼───────┤
# │ fc1    │ Linear     │ (1, 64)     │ 1088  │
# │ fc2    │ Linear     │ (1, 4)      │ 260   │
# └────────┴────────────┴─────────────┴───────┘
print("total parameters:", spx.inspect.count_parameters(model))
print("total bytes :", spx.inspect.count_bytes(model))
```

## A train step

```python
@spx.jit
def train_step(m, x, y):
    """Forward + MSE + grad against parameters."""
    def loss(m):
        return ((m(x) - y) ** 2).mean()

    return spx.value_and_grad(loss)(m)


loss_val, grads = train_step(model, jnp.ones((8, 16)), jnp.zeros((8, 4)))
print(float(loss_val))
print(type(grads))                                  # spx.State
print(grads["parameters"]["fc1.weight"].shape)
```

`spx.value_and_grad` differentiates the loss against the `"parameters"`
collection by default. The result `grads` is a `State` shaped like
the parameters slice — `{collection: {dotted_path: array}}`. `spx.jit`
caches the compile by the model's graph shape, so subsequent calls
reuse the cached XLA executable.

## Adding an optimizer

The contrib package wraps optax in a pytree-friendly object that
threads through `spx.jit` as a normal arg:

```python
from spectrax.contrib import Optimizer
import optax

opt = Optimizer.create(model, optax.adamw(3e-4))


@spx.jit(mutable="parameters")
def step(m, o, x, y):
    def loss(m):
        return ((m(x) - y) ** 2).mean()

    loss_val, grads = spx.value_and_grad(loss)(m)
    new_opt = o.apply_eager(m, grads)               # mutates m['parameters']; returns new opt
    return loss_val, new_opt


for i in range(100):
    x = jnp.ones((8, 16))
    y = jnp.zeros((8, 4))
    loss_val, opt = step(model, opt, x, y)
    if i % 10 == 0:
        print(f"step {i}: loss = {float(loss_val):.4f}")
```

Two new things:

- `mutable="parameters"` declares which variable collections may be
  written back. The optimizer writes new parameters via
  `apply_eager(m, grads)`; without `mutable=`, SpectraX raises
  `IllegalMutationError`.
- `Optimizer.create(model, tx, wrt="parameters")` allocates Adam state
  only for the `"parameters"` collection. Pass `wrt="lora"`, `wrt=
  nn.LoraParameter`, or any selector to scope the allocation
  precisely — see [LoRA fine-tuning](guides/lora.md) for a worked
  example with zero base-weight optimizer memory.

## Save / load

Modules are JAX pytrees, so any pytree-aware checkpoint format works.
The simplest: pickle the `(GraphDef, State)` pair via JAX's standard
serialization helpers.

```python
import pickle

# Save
gdef, state = spx.export(model)
with open("model.pkl", "wb") as f:
    pickle.dump({"gdef": gdef, "state": state}, f)

# Load
with open("model.pkl", "rb") as f:
    data = pickle.load(f)
loaded = spx.bind(data["gdef"], data["state"])

# Verify
y_orig = model(x)
y_loaded = loaded(x)
assert jnp.allclose(y_orig, y_loaded)
```

For production, use [`safetensors`](https://huggingface.co/docs/safetensors)
or [`orbax`](https://github.com/google/orbax) — both work directly on
the leaf arrays from `spx.export(model)[1]`.

## Eval mode

```python
model.eval()                            # propagates training=False to all submodules
y = model(x)                            # Dropout / BatchNorm now in eval mode
model.train()                           # back to training mode
```

## A complete training loop

Full skeleton with logging, periodic eval, and checkpointing:

```python
import time
import pickle
import jax
import jax.numpy as jnp
import spectrax as spx
from spectrax import nn, functional as F
from spectrax.contrib import Optimizer
import optax


# Toy data — replace with your loader
def batch(seed, n):
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, (n, 16))
    y = (x[:, :4] > 0).astype(jnp.float32)          # arbitrary target
    return x, y


class Net(spx.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.fc1 = nn.Linear(16, 64, rngs=rngs)
        self.drop = nn.Dropout(0.1)
        self.fc2 = nn.Linear(64, 4, rngs=rngs)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


rngs = spx.Rngs(0, dropout=1)
model = Net(rngs=rngs)
opt = Optimizer.create(model, optax.adamw(3e-4))


@spx.jit(mutable="parameters")
def train_step(m, o, x, y):
    def loss(m):
        return jnp.mean((m(x) - y) ** 2)

    loss_val, grads = spx.value_and_grad(loss)(m)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt


@spx.jit
def eval_step(m, x, y):
    return jnp.mean((m(x) - y) ** 2)


N_EPOCHS = 5
ITERS_PER_EPOCH = 100

for epoch in range(N_EPOCHS):
    model.train()
    t0 = time.perf_counter()
    losses = []
    for i in range(ITERS_PER_EPOCH):
        x, y = batch(epoch * ITERS_PER_EPOCH + i, n=32)
        loss_val, opt = train_step(model, opt, x, y)
        losses.append(float(loss_val))
    epoch_time = time.perf_counter() - t0
    print(
        f"epoch {epoch + 1}: train_loss={sum(losses)/len(losses):.4f} "
        f"time={epoch_time:.2f}s"
    )

    # Eval
    model.eval()
    x_val, y_val = batch(seed=999, n=128)
    val_loss = float(eval_step(model, x_val, y_val))
    print(f"           val_loss={val_loss:.4f}")

    # Checkpoint
    gdef, state = spx.export(model)
    with open(f"checkpoint_epoch_{epoch + 1}.pkl", "wb") as f:
        pickle.dump({"gdef": gdef, "state": state, "epoch": epoch + 1}, f)
```

## What to read next

- **[Modules](guides/modules.md)** — the full eager-surface API:
  containers, custom variable types, hooks, sow/perturb,
  `train()` / `eval()`, the graph/state seam.
- **[Transforms](guides/transforms.md)** — `spx.jit` /
  `spx.grad` / `spx.vmap` / `spx.scan` / `spx.remat` with worked
  examples and composition patterns.
- **[Selectors](guides/selectors.md)** — the predicate DSL shared by
  every "subset of the model" API: `wrt=`, `mutable=`,
  `partition_state(...)`, `iter_variables(...)`, `freeze(...)`.
- **[Dynamic scope](guides/scope.md)** — `spx.scope(**values)` for
  threading attention masks, mode flags, and per-batch context
  through deep call stacks without per-layer arg plumbing.
- **[LoRA fine-tuning](guides/lora.md)** — parameter-efficient
  fine-tuning with **zero base-weight optimizer memory**.
- **[FP8 training](guides/fp8.md)** — delayed-scaling FP8 with
  rolling amax history.
- **[Sharding](guides/sharding.md)** — SPMD over `jax.sharding.Mesh`
  with logical axis names; full DP and TP walkthroughs.
- **[Design](design.md)** — why SpectraX is shaped the way it is.
- **[Performance](performance.md)** — benchmark methodology, the 15
  dispatch-path optimizations, profiling recipes.
