# LoRA fine-tuning

[Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA) freezes
the pretrained base weights of a model and trains only small "adapter"
matrices added in parallel:

```md
y  =  W₀·x  +  (α/r) · B·A·x
       ↑              ↑
   frozen base     trainable adapter (B is rxout, A is inxr)
```

The adapter has `r·(in + out)` parameters instead of `in·out` — for
typical r=8 and d=768, that's ~12 600 vs ~590 000, a ~50x reduction.
Combined with the fact that you only allocate optimizer state for
the adapter, you can fine-tune a multi-billion-parameter model on a
consumer GPU.

spectrax implements LoRA as a **first-class collection** — not as
a tree-walker that rewrites your model at import time, not as a
metaclass hack, not as a special `OVERWRITE_WITH_GRADIENT`
mechanism. It's just a `Variable` subclass + a small
reference-holding `Module`.

## Three ways to use it

### 1. Wrap an existing layer

```python
import spectrax as spx
from spectrax import nn

base = nn.Linear(768, 768, rngs=spx.Rngs(0))
model = nn.wrap_lora(base, rank=8, alpha=16, rngs=spx.Rngs(1))

print(type(model))           # spectrax.nn.LoRALinear
print(type(model.base))      # spectrax.nn.Linear
print(type(model.lora))      # spectrax.nn.LoRA
```

`wrap_lora(base, ...)` returns a `LoRALinear` whose forward computes
`base(x) + (α/r)·B·A·x`. The base weights stay where they were —
`model.base` is the same `Linear` you passed in.

### 2. Build `LoRALinear` from scratch

```python
ll = nn.LoRALinear(in_features=768, out_features=768, rank=8, alpha=16, rngs=spx.Rngs(0))
```

Equivalent to `wrap_lora(nn.Linear(768, 768, rngs=...), rank=8, alpha=16, rngs=...)`.

### 3. Add LoRA inside a custom module

```python
class MyAttention(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.q = nn.LoRALinear(d, d, rank=8, rngs=rngs)        # adapt q
        self.k = nn.LoRALinear(d, d, rank=8, rngs=rngs)        # adapt k
        self.v = nn.Linear(d, d, rngs=rngs)                    # don't adapt v
        self.out = nn.LoRALinear(d, d, rank=8, rngs=rngs)      # adapt out

    def forward(self, x):
        ...
```

You decide which layers to adapt; spectrax doesn't impose a default.

## The zero-init invariant

`lora_b` is **zero-initialized**, so `B·A = 0` and the adapter is a
no-op at step 0:

```python
ll = nn.LoRALinear(64, 64, rank=4, rngs=spx.Rngs(0))
x = jnp.ones((1, 64))

# Output of the wrapped module equals the base alone
y_wrapped = ll(x)
y_base = ll.base(x)
assert jnp.allclose(y_wrapped, y_base)
```

This means **loading a pretrained base weight into a wrapped
module preserves the base model's behavior exactly until you start
training**. No drift, no warmup needed.

## Train only the adapter

LoRA factors live in the `"lora"` collection and use the
`LoraParameter` Variable subclass. Both selectors work identically:

```python
@spx.jit(mutable="lora")
def step(m, o, x, y):
    def loss(m):
        return ((m(x) - y) ** 2).mean()

    loss_val, grads = spx.value_and_grad(loss, wrt="lora")(m)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt


# equivalently:
spx.value_and_grad(loss, wrt=nn.LoraParameter)(m)
```

`spx.grad`/`spx.value_and_grad` only differentiate the adapter leaves;
base weights never see a gradient. This both saves compute (smaller
HLO) and saves memory (smaller cotangents).

## Optimizer with zero base-weight state

`Optimizer(wrt=...)` allocates optax moments **only for the selected
leaves**, so LoRA fine-tuning uses ~1% of the Adam memory of a full
fine-tune:

```python
from spectrax.contrib import Optimizer
import optax

opt = Optimizer.create(model, optax.adamw(1e-3), wrt="lora")

# Compare:
# Full fine-tune of a 7B model with Adam at fp32:
#   2 x 7B x 4 bytes = 56 GB optimizer state alone.
# LoRA r=8 of the same model:
#   ~10 M trainable parameters x 8 bytes = 80 MB optimizer state.
```

For mixed schedules — e.g. base at low LR, adapter at high LR — use
`MultiOptimizer`:

```python
from spectrax.contrib import MultiOptimizer

opt = MultiOptimizer(
    {
        "parameters": optax.adamw(1e-4),       # base stays trainable, slow
        "lora":   optax.adamw(1e-3),       # adapter trains fast
    },
    module=model,
)
opt.update(grads)                          # dispatches per collection
```

## Folding the adapter into the base (inference)

When training is done, bake `B·A` into the base weight and zero the
adapter for **zero inference-time overhead**:

```python
def fold_lora(layer: nn.LoRALinear) -> None:
    """In-place fold layer.lora into layer.base.weight."""
    a = layer.lora.lora_a.value
    b = layer.lora.lora_b.value
    scale = layer.lora.alpha / layer.lora.rank
    layer.base.weight.value = layer.base.weight.value + scale * (a @ b)
    layer.lora.lora_a.value = jnp.zeros_like(a)
    layer.lora.lora_b.value = jnp.zeros_like(b)


# Apply to every LoRALinear in the model
for path, layer in spx.iter_modules(model, select=nn.LoRALinear):
    fold_lora(layer)


# Now the adapters are no-ops; you can swap them out entirely:
def strip_lora(model):
    """Replace each LoRALinear with its underlying nn.Linear."""
    for path, layer in spx.iter_modules(model, select=nn.LoRALinear):
        # Walk to the parent and replace the attribute
        ...                                # see `Save / load just adapters` below
```

## Save / load just the adapter weights

Adapters are tiny — save and ship them on their own:

```python
import pickle

# Save adapter only
gdef, state = spx.export(model)
adapter_state = state.filter("lora")
with open("adapter.pkl", "wb") as f:
    pickle.dump({"state": adapter_state}, f)

# Load adapter into a freshly-built model with same base
model = build_my_model()                    # fresh: base loaded from pretrained, adapter zero-init
with open("adapter.pkl", "rb") as f:
    adapter = pickle.load(f)
spx.update(model, adapter["state"])         # in-place patch only the lora keys
```

Because `update` walks the model and writes only the leaves whose
`(collection, path)` matches, you can load an adapter into any
structurally-equivalent model.

## Multi-task / multi-adapter

Train one base + multiple task adapters by giving each its own
`Variable` subclass:

```python
class TaskAAdapter(spx.Variable): default_kind = "lora_a_task"
class TaskBAdapter(spx.Variable): default_kind = "lora_b_task"


# Build a model with two parallel adapter sets per layer (custom LoRALinear-like class)...

# Train task A's adapter
grads_a = spx.grad(loss_a, wrt="lora_a_task")(model, x, y)
opt_a = spx.contrib.Optimizer.create(model, optax.adamw(1e-3), wrt="lora_a_task")

# Train task B's adapter
grads_b = spx.grad(loss_b, wrt="lora_b_task")(model, x, y)
opt_b = spx.contrib.Optimizer.create(model, optax.adamw(1e-3), wrt="lora_b_task")

# At inference, swap which adapter is active by zeroing the others
```

## Full fine-tune script

End-to-end LoRA fine-tune of a transformer encoder, with
checkpointing of just the adapter weights:

```python
import pickle
import jax
import jax.numpy as jnp
import optax

import spectrax as spx
from spectrax import nn, functional as F
from spectrax.contrib import Optimizer


# ---- Model ----
class Encoder(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.attn_q = nn.LoRALinear(d, d, rank=8, alpha=16, rngs=rngs)
        self.attn_k = nn.LoRALinear(d, d, rank=8, alpha=16, rngs=rngs)
        self.attn_v = nn.Linear(d, d, rngs=rngs)
        self.attn_o = nn.LoRALinear(d, d, rank=8, alpha=16, rngs=rngs)
        self.ln = nn.LayerNorm(d, rngs=rngs)
        self.fc1 = nn.Linear(d, d * 4, rngs=rngs)
        self.fc2 = nn.Linear(d * 4, d, rngs=rngs)

    def forward(self, x):
        q = self.attn_q(x); k = self.attn_k(x); v = self.attn_v(x)
        a = jax.nn.softmax(q @ k.transpose(0, 2, 1) / jnp.sqrt(q.shape[-1]), axis=-1)
        h = self.attn_o(a @ v) + x
        h = self.ln(h)
        return self.fc2(F.gelu(self.fc1(h))) + h


class Model(spx.Module):
    def __init__(self, n, d, vocab, *, rngs):
        super().__init__()
        self.embed = nn.Embed(vocab, d, rngs=rngs)
        self.layers = nn.Sequential(*[Encoder(d, rngs=rngs) for _ in range(n)])
        self.head = nn.Linear(d, vocab, rngs=rngs)

    def forward(self, ids):
        return self.head(self.layers(self.embed(ids)))


# ---- Setup ----
model = Model(n=4, d=128, vocab=1000, rngs=spx.Rngs(0))

# (Pretend we loaded pretrained weights here)
# ... load_pretrained(model, "checkpoint.pkl") ...

opt = Optimizer.create(model, optax.adamw(1e-3), wrt="lora")
print("trainable parameters:", spx.inspect.count_parameters(model))        # all of them
print("optimizer state size:")
for col, d in opt.opt_state[0].mu.items():                              # Adam μ slots
    print(f"  {col}: {sum(jnp.size(v) for v in d.values()):,}")         # only "lora"


# ---- Training step ----
@spx.jit(mutable="lora")
def step(m, o, ids, targets):
    def loss(m):
        logits = m(ids)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

    loss_val, grads = spx.value_and_grad(loss, wrt="lora")(m)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt


# ---- Loop ----
for epoch in range(5):
    for batch_i in range(100):
        ids = jax.random.randint(jax.random.PRNGKey(epoch * 100 + batch_i), (16, 128), 0, 1000)
        targets = jax.random.randint(jax.random.PRNGKey(epoch * 100 + batch_i + 999), (16, 128), 0, 1000)
        loss_val, opt = step(model, opt, ids, targets)
    print(f"epoch {epoch + 1}: loss = {float(loss_val):.4f}")

    # Checkpoint just the adapter
    _, state = spx.export(model)
    adapter = state.filter("lora")
    with open(f"adapter_epoch_{epoch + 1}.pkl", "wb") as f:
        pickle.dump(adapter, f)
```

## Selector cookbook for LoRA

```python
# Train adapters only (default)
spx.grad(loss, wrt="lora")

# Train adapters + biases
sel = spx.as_selector("lora") | spx.path_endswith(".bias")
spx.grad(loss, wrt=sel)

# Train adapters in attention layers only
sel = spx.as_selector("lora") & spx.path_contains("attn")
spx.grad(loss, wrt=sel)

# Train adapters except in the final block
sel = spx.as_selector("lora") - spx.path_startswith("layers.11.")
spx.grad(loss, wrt=sel)

# Multi-rate optimizer: adapters + last-layer head
opt = spx.contrib.MultiOptimizer({
    "lora": optax.adamw(1e-3),
    spx.path_startswith("head.").variables("parameters"): optax.adamw(1e-4),
}, module=model)
```

## Under the hood

```python
class LoraParameter(spx.Variable):
    default_kind = "lora"


class LoRA(spx.Module):
    def __init__(self, in_features, out_features, rank, alpha, *, rngs):
        super().__init__()
        self.rank = rank                         # static
        self.alpha = alpha                       # static
        self.lora_a = LoraParameter(
            spx.init.kaiming_uniform("linear")(rngs.parameters, (in_features, rank), jnp.float32)
        )
        self.lora_b = LoraParameter(jnp.zeros((rank, out_features), jnp.float32))

    def forward(self, x):
        return (self.alpha / self.rank) * x @ self.lora_a.value @ self.lora_b.value


class LoRALinear(spx.Module):
    def __init__(self, in_features, out_features, rank, alpha, *, rngs):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, rngs=rngs)
        self.lora = LoRA(in_features, out_features, rank, alpha, rngs=rngs)

    def forward(self, x):
        return self.base(x) + self.lora(x)
```

That's it — the entire LoRA mechanism is selector + Variable subclass +
reference-holding module. No magic, fully introspectable.

## API reference

- [`spectrax.nn.lora`](../api_docs/nn/lora.rst) — `LoRA`,
  `LoRALinear`, `LoraParameter`, `wrap_lora`.
- [`spectrax.contrib.optimizer`](../api_docs/contrib/optimizer.rst) —
  `Optimizer`, `MultiOptimizer`.
