# Modules

`spx.Module` is the PyTorch-shaped base class. Subclass it, declare
submodules and parameters in `__init__`, override `forward`, call
`model(x)`. Underneath, every `Module` is a registered **JAX pytree**
whose leaves are the raw arrays held by descendant `Variable` cells.

The two halves — eager Python surface and graph/state seam — are
designed to coexist: you write everyday code as if it were PyTorch and
reach for `spx.export` / `spx.bind` only when you need to compute,
serialize, or transform the underlying state explicitly.

## Anatomy of a Module

```python
import jax.numpy as jnp
import spectrax as spx
from spectrax import nn, functional as F


class Encoder(spx.Module):
    """Self-attention block: q/k/v projections + scaled-dot-product attn + out."""

    def __init__(self, d, num_heads, *, rngs):
        super().__init__()                        # MUST be first
        self.d = d                                # static — goes into GraphDef
        self.num_heads = num_heads                # static
        self.q = nn.Linear(d, d, rngs=rngs)       # submodule
        self.k = nn.Linear(d, d, rngs=rngs)
        self.v = nn.Linear(d, d, rngs=rngs)
        self.out = nn.Linear(d, d, rngs=rngs)

    def forward(self, x, mask=None):
        # x: (batch, seq, d); mask: optional (batch, seq, seq) bool
        b, s, d = x.shape
        h = self.num_heads
        head_dim = d // h
        q = self.q(x).reshape(b, s, h, head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, h, head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, h, head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        if mask is not None:
            scores = jnp.where(mask[:, None], scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        y = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.out(y)
```

### Three rules

1. **Call `super().__init__()` before any attribute assignment.** It
   sets up the private slots: attribute order, static-field dict,
   training flag, hook lists. Skipping this raises the moment you
   assign a child.
2. **Allowed attribute types**: another `Module`, a `Variable`
   subclass (`Parameter`, `Buffer`, `LoraParameter`, custom subclasses),
   a static hashable scalar (Python `int` / `float` / `str` / `bool`
   / `tuple` of statics — folded into `GraphDef`), an `Opaque(...)`
   escape hatch, or a name starting with `_` (implementation detail,
   not graph-visible).
3. **Override `forward`**. `__call__` invokes pre-hooks -> `forward`
   -> post-hooks and applies any active dtype policy.

### What goes where

| Attribute kind        | Where it ends up             | Affects compile cache key? |
| --------------------- | ---------------------------- | -------------------------- |
| `Module` subclass     | Child node in `GraphDef`     | yes (structure)            |
| `Variable` subclass   | Leaf in `GraphDef` + `State` | yes (path/kind)            |
| Static hashable       | `GraphDef.static_fields`     | yes (value)                |
| `Opaque(value)`       | Hidden side dict             | no (invisible to graph)    |
| `_underscore` private | Plain Python attr            | no (invisible to graph)    |

Notes:

- A `tuple` of statics is itself static; a `list` is not (lists are
  mutable). Use a `tuple` for things like kernel sizes.
- Anything that can't go in `GraphDef` should be wrapped in `Opaque`
  (e.g. a callable, a dict of strings) so spectrax doesn't try to
  introspect it.

## Container types

Containers wrap collections of submodules so they're individually
addressable and serializable.

### `nn.Sequential`

Forward calls each child in declaration order:

```python
classifier = nn.Sequential(
    nn.Linear(784, 256, rngs=rngs),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10, rngs=rngs),
)
y = classifier(x)
```

### `nn.ModuleList` and `nn.ModuleDict`

Indexable / keyable, with normal Python `len()` / iteration:

```python
class Stack(spx.Module):
    def __init__(self, n, d, *, rngs):
        super().__init__()
        self.blocks = nn.ModuleList([
            Encoder(d, 8, rngs=rngs) for _ in range(n)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class GatedMix(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.heads = nn.ModuleDict({
            "fast": nn.Linear(d, d, rngs=rngs),
            "slow": nn.Linear(d, d, rngs=rngs),
        })

    def forward(self, x, route="fast"):
        return self.heads[route](x)
```

You can append/extend at construction time (the structure becomes part
of `GraphDef`); avoid mutating containers after the model has been
used — that bumps the graph epoch and invalidates all cached
compiles.

### `nn.ParameterList`

A list of `Variable` cells with a shared kind. Useful when you have a
variable-length list of weight tensors:

```python
class FactoredEmbedding(spx.Module):
    def __init__(self, vocab, dims, *, rngs):
        super().__init__()
        self.factors = nn.ParameterList([
            spx.Parameter(jax.random.normal(rngs.parameters, (vocab, d)))
            for d in dims
        ])

    def forward(self, ids):
        return jnp.concatenate(
            [f.value[ids] for f in self.factors],
            axis=-1,
        )
```

## Variables

`Variable` is the mutable reference cell. Subclasses tag their
**collection** via `default_kind`:

| Subclass        | Collection (`kind`) | Typical use                      |
| --------------- | ------------------- | -------------------------------- |
| `Parameter`     | `parameters`            | Trainable weights                |
| `Buffer`        | `buffers`           | Non-trainable state (e.g. masks) |
| `LoraParameter` | `lora`              | LoRA adapters                    |
| `Fp8Meta`       | `fp8_meta`          | FP8 scales / amax history        |
| `DeferredParameter` | `parameters` (deferred) | Lazy-shape trainable weight  |
| `DeferredBuffer`    | `buffers` (deferred)| Lazy-shape non-trainable state |

### Deferred initialization

Built-in layers accept `None` for input dimensions and infer the shape
from the first forward call:

```python
model = nn.Sequential(
    nn.Linear(None, 256, rngs=rngs),      # in_features resolved on first call
    nn.ReLU(),
    nn.Linear(256, 10, rngs=rngs),
)
y = model(jnp.zeros((8, 128)))            # weight (128, 256) materialized here
```

This works for:

| Layer family | Deferred argument | Resolved from |
| ------------ | ----------------- | ------------- |
| `Linear`     | `in_features=None`| `x.shape[-1]` |
| `Conv*d`     | `in_channels=None`| `x.shape[-1]` |
| `ConvTranspose*d` | `in_channels=None` | `x.shape[-1]` |
| `Embed`      | `num_embeddings=None` | `int(ids.max()) + 1` |

Under the hood the layer stores a `DeferredParameter` that holds the
initializer and a shape spec with `None` placeholders.  The forward
pass resolves the concrete shape and calls the initializer.  After
materialization the parameter behaves exactly like a normal
`Parameter`.

**Explicit materialization** before compiling:

```python
# Option 1: eager forward pass
_ = model(jnp.zeros((1, 128)))

# Option 2: sequential_init helper
model = spx.sequential_init(model, jnp.zeros((1, 128)))

# Option 3: manual walk
model.materialize()
```

**Transform safety.** Deferred parameters refuse to materialize
inside `jax.jit`, `jax.vmap`, or `jax.scan` and raise
`LazyInitUnderTransformError`.  Resolve shapes eagerly, then compile.

**Custom layers** can use `DeferredParameter` directly:

```python
class MyLayer(spx.Module):
    def __init__(self, out, *, rngs):
        super().__init__()
        self.out_features = out
        self.weight = spx.DeferredParameter(
            (None, out),
            spx.init.kaiming_uniform("linear"),
            rngs.parameters, jnp.float32,
        )

    def forward(self, x):
        self._resolve_deferred(self.weight, (x.shape[-1], self.out_features))
        return x @ self.weight.value
```

### Reading and writing

```python
fc = nn.Linear(10, 20, rngs=rngs)

w = fc.weight.value                   # read
fc.weight.value = jnp.zeros_like(w)   # write — mutates the cell

print(fc.weight.kind)                 # 'parameters'
print(fc.weight.metadata)             # {} — opt sharding info, etc.
print(fc.weight.ref_id)               # int — identity for shared weights
```

The `.value` setter is the single mutation surface — every
`spx.update`, optimizer write, and transform write-back goes through
it (or the lower-level `_raw_set`).

### Custom variable types

Add a new `Variable` subclass with its own `default_kind` and use it
inside any `Module`. spectrax auto-registers the new collection — no
manifest, no metaclass:

```python
class AdapterParam(spx.Variable):
    """Trainable adapter weight, separate from base 'parameters'."""
    default_kind = "adapter"


class Bottleneck(spx.Module):
    def __init__(self, d, rank, *, rngs):
        super().__init__()
        self.down = AdapterParam(
            spx.init.kaiming_uniform("linear")(rngs.parameters, (d, rank), jnp.float32)
        )
        self.up = AdapterParam(jnp.zeros((rank, d), jnp.float32))

    def forward(self, x):
        return x + jnp.maximum(0.0, x @ self.down.value) @ self.up.value


# Train ONLY the adapter — base 'parameters' stays frozen
grads = spx.grad(loss, wrt="adapter")(model, x, y)
# equivalent: wrt=AdapterParam
```

### Shared weights

Assign the same `Variable` to multiple attributes — `export` detects
identity and records the canonical path + alias:

```python
class TiedEmbedDecoder(spx.Module):
    def __init__(self, vocab, d, *, rngs):
        super().__init__()
        self.embed = nn.Embed(vocab, d, rngs=rngs)
        self.decoder = nn.Linear(d, vocab, rngs=rngs)
        # Tie: share the embedding matrix as the decoder weight
        self.decoder.weight = self.embed.weight


m = TiedEmbedDecoder(50_000, 256, rngs=rngs)
gdef, state = spx.export(m)
print(state["parameters"].keys())              # one weight, not two
print(gdef.shared_paths)                   # [("decoder.weight", "embed.weight")]
```

After tying, gradient updates flow through the single underlying
`Variable` automatically.

## Worked example: a transformer block

A complete transformer encoder block, idiomatic to spectrax:

```python
class TransformerBlock(spx.Module):
    def __init__(self, d, num_heads, ffn, dropout=0.1, *, rngs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, rngs=rngs)
        self.attn = nn.MultiheadAttention(d, num_heads, rngs=rngs)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d, rngs=rngs)
        self.fc1 = nn.Linear(d, ffn, rngs=rngs)
        self.fc2 = nn.Linear(ffn, d, rngs=rngs)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h = self.attn(h, h, h, mask=mask)
        x = x + self.drop1(h)
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + self.drop2(h)


class Transformer(spx.Module):
    def __init__(self, n_layers, d, num_heads, ffn, *, rngs):
        super().__init__()
        self.blocks = nn.Sequential(*[
            TransformerBlock(d, num_heads, ffn, rngs=rngs)
            for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        # Sequential doesn't pass extra kwargs — iterate manually if you need mask
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return x


model = Transformer(n_layers=6, d=512, num_heads=8, ffn=2048, rngs=spx.Rngs(0))
spx.inspect.summary(model, jnp.zeros((1, 128, 512)))
```

## Worked example: a ResNet block

```python
class BasicResBlock(spx.Module):
    def __init__(self, channels, *, rngs):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, rngs=rngs)
        self.bn1 = nn.BatchNorm2d(channels, rngs=rngs)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, rngs=rngs)
        self.bn2 = nn.BatchNorm2d(channels, rngs=rngs)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


# BatchNorm running stats live in the "batch_stats" collection.
# Declare it mutable on the train step so updates propagate:
@spx.jit(mutable="batch_stats" | "parameters")
def step(m, o, x, y): ...
```

## Train / eval mode

```python
model.train()                        # training=True recursively
print(model.training)                # True
print(model.blocks[0].attn.training) # True

model.eval()                         # training=False recursively
y = model(x)                         # Dropout is identity; BatchNorm uses running stats
```

`Dropout` and `BatchNorm*d` consult `self.training`; your own modules
can too.

## Hooks

Forward hooks fire on every `model(x)` call **in eager mode**. They
are skipped under spectrax transforms (with a one-shot warning per
module), because they can have arbitrary side effects that aren't
trace-safe. Use `self.sow(...)` (next section) for transform-safe
capture.

```python
def watch(m, args, kwargs, out):
    print(f"{type(m).__name__} -> {out.shape}")


handle = model.fc1.register_forward_hook(watch)
y = model(jnp.ones((4, 16)))         # prints "Linear -> (4, 64)"
handle.remove()                      # detach later
```

Pre-hooks return `(args, kwargs)` to override the inputs:

```python
def normalize_input(m, args, kwargs):
    x, *rest = args
    x = (x - x.mean()) / (x.std() + 1e-6)
    return (x, *rest), kwargs


h = model.register_forward_pre_hook(normalize_input)
```

## Intermediates and perturbations

Use `self.sow("intermediates", "name", value)` inside `forward` to
capture activations into a `Variable`, declared mutable on the
transform:

```python
class Net(spx.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.encoder = ...
        self.decoder = ...

    def forward(self, x):
        h = self.encoder(x)
        self.sow("intermediates", "encoder_out", h)   # captured
        return self.decoder(h)


@spx.jit(mutable="intermediates")
def run(m, x):
    return m(x)


_ = run(model, x)
captured = spx.pop(model, "intermediates")            # {path: array}
```

`self.perturb("name", x)` is similar but adds a zero-initialized
additive cell — useful for sensitivity analysis (the gradient w.r.t.
the perturbation cell tells you how much each activation contributed
to the loss).

## The graph / state seam

Modules ARE pytrees, so `jax.tree.map(fn, model)` and
`jax.tree.leaves(model)` work. When you need to introspect or
serialize the structure explicitly, drop one level:

```python
gdef, state = spx.export(model)             # snapshot
model2 = spx.bind(gdef, state)              # reconstruct (skips __init__)
spx.update(model, state)                    # in-place state patch
clone = spx.clone(model)                    # deep copy via export+bind
intermediates = spx.pop(model, "intermediates")   # remove & return matches
```

### `GraphDef`

Immutable, hashable DAG description:

```python
print(gdef)
# GraphDef(nodes=(...), root=0, var_refs=..., var_canonical=..., shared_paths=())
print(hash(gdef))                           # cached hash — stable across calls
print(gdef.canonical_path(0))               # 'fc1.weight'
```

Two structurally-equal modules produce **equal `GraphDef` values**:

```python
a = MLP(16, 32, 4, rngs=spx.Rngs(0))
b = MLP(16, 32, 4, rngs=spx.Rngs(1))        # different seed, same structure
ga, _ = spx.export(a)
gb, _ = spx.export(b)
assert ga == gb
assert hash(ga) == hash(gb)
```

This is what makes the `spx.jit` compile cache effective across
freshly-built modules with the same shape.

### `State`

Two-level dict, `{collection: {dotted_path: leaf}}`:

```python
print(state["parameters"].keys())
# dict_keys(['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'])
print(state["parameters"]["fc1.weight"].shape)

# Filter
parameters_only = state.filter("parameters")
no_buffers = state.exclude("buffers")

# Map in place
state.map(lambda path, v: jnp.zeros_like(v), "parameters")

# Detached copy
zeroed = state.map(lambda path, v: jnp.zeros_like(v), "parameters", copy=True)

# Merge two States (right wins on collision)
merged = state.merge(other_state)

# Iterate
for collection, path, leaf in state.items():
    print(collection, path, leaf.shape)
```

### Save / load

`State` is a normal pytree of arrays. Use any JAX-compatible
serializer:

```python
# Pickle (development)
import pickle
gdef, state = spx.export(model)
with open("ckpt.pkl", "wb") as f:
    pickle.dump({"gdef": gdef, "state": state}, f)

with open("ckpt.pkl", "rb") as f:
    data = pickle.load(f)
restored = spx.bind(data["gdef"], data["state"])

# safetensors (production, only saves arrays)
from safetensors.numpy import save_file, load_file
flat = state.flatten()                      # {"parameters/fc1.weight": array, ...}
save_file(flat, "ckpt.safetensors")
flat = load_file("ckpt.safetensors")
state = spx.State.from_flat(flat)
restored = spx.bind(gdef, state)            # gdef must be saved separately
```

## Iteration and search

```python
# Every module in canonical-path order
for path, m in spx.iter_modules(model):
    ...

# Filtered by class
for path, m in spx.iter_modules(model, select=nn.Linear):
    print(path, m.in_features, m.out_features)

# Filtered by callable
for path, m in spx.iter_modules(model, select=lambda m, p: isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3)):
    ...

# Variables
for path, v in spx.iter_variables(model, select="parameters"):
    print(path, v.value.shape)

# Variables filtered by Variable subclass
for path, v in spx.iter_variables(model, select=nn.LoraParameter):
    ...

# First match
first = spx.find(model, nn.Conv2d)          # (path, module) or None
first_lora = spx.find(model, nn.LoraParameter)
```

## Built-in layers

Full list: see [`spectrax.nn`](../api_docs/nn/index.rst).

| Family         | Layers                                                                                              |
| -------------- | --------------------------------------------------------------------------------------------------- |
| Linear         | `Linear`, `Bilinear`, `DenseGeneral`, `Einsum`, `Embed` — all accept `None` for input dims          |
| Convolution    | `Conv1d`/`2d`/`3d`, `ConvTranspose1d`/`2d`/`3d` — `in_channels=None` for deferred init              |
| Attention      | `MultiheadAttention`, `CausalSelfAttention`                                                         |
| Normalization  | `LayerNorm`, `RMSNorm`, `BatchNorm1d`/`2d`, `InstanceNorm`, `GroupNorm`                             |
| Pooling        | `MaxPool1d`/`2d`/`3d`, `AvgPool1d`/`2d`/`3d`, `AdaptiveAvgPool1d`/`2d`/`3d`                         |
| Recurrent      | `SimpleRNNCell`, `GRUCell`, `LSTMCell`, `OptimizedLSTMCell`, `ConvLSTMCell`, `RNN`, `Bidirectional` |
| Containers     | `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`                                           |
| Activation     | `ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`                                                           |
| Regularization | `Dropout`                                                                                           |
| Composite      | `MLPBlock`, `Identity`                                                                              |
| PEFT           | `LoRA`, `LoRALinear`, `LoraParameter`, `wrap_lora` — see [LoRA](lora.md)                            |
| Low-precision  | `Fp8DotGeneral`, `Fp8Linear`, `Fp8Einsum`, `Fp8Meta` — see [FP8](fp8.md)                            |

Pointwise functional counterparts: [`spectrax.functional`](../api_docs/functional/index.rst)
(`F.gelu`, `F.layer_norm`, `F.attention`, `F.conv`, `F.dropout`, …).

## Inspection

```python
spx.inspect.summary(model, jnp.zeros((1, 128, 512)))    # PyTorch-style table
spx.inspect.tabulate(model, jnp.zeros((1, 128, 512)))   # finer-grained
print("parameters:", spx.inspect.count_parameters(model))   # int
print("bytes:",  spx.inspect.count_bytes(model))        # int
print("tree:",   spx.inspect.tree(model))               # nested string
```

## Common pitfalls

### Forgetting `super().__init__()`

```python
class Bad(spx.Module):
    def __init__(self, d, *, rngs):
        # super().__init__() missing!
        self.fc = nn.Linear(d, d, rngs=rngs)        # AttributeError on _spx_attr_order
```

**Always call `super().__init__()` first.**

### Assigning a list of modules

```python
class Bad(spx.Module):
    def __init__(self, n, d, *, rngs):
        super().__init__()
        self.layers = [nn.Linear(d, d, rngs=rngs) for _ in range(n)]   # ❌
```

A plain Python list is not a recognized container and won't be
discovered by graph traversal. Use `nn.ModuleList`, `nn.ModuleDict`,
or `nn.Sequential` instead.

### Mutating after first use

```python
model = MyNet()
y = model(x)
spx.export(model)                               # caches export structure

model.new_layer = nn.Linear(...)                # mutates structure
                                                # -> next jit call invalidates cache
```

`Module.__setattr__` bumps a global graph epoch which invalidates
caches lazily. Build the model fully before training.

### Using a non-hashable static value

```python
class Bad(spx.Module):
    def __init__(self):
        super().__init__()
        self.config = {"lr": 0.001}            # ❌ dict is not hashable
```

Wrap in `Opaque(...)` to hide it from graph traversal:

```python
self.config = spx.Opaque({"lr": 0.001})        # invisible to GraphDef
print(self.config.value["lr"])
```

### Mutating a `Variable` outside the right transform

```python
@jax.jit                                        # plain jax.jit — silent drop!
def step(m, x):
    m.fc.weight.value = m.fc.weight.value - 0.01 * grad
    return m(x)
```

Inside plain `jax.jit`, the model is rebuilt from leaves on entry;
`.value =` writes hit the rebuilt copy, not the original. Use
`spx.jit(mutable="parameters")` for safe in-trace mutation.

## API reference

- [`spectrax.core.module`](../api_docs/core/module.rst) — `Module`,
  `Opaque`.
- [`spectrax.core.variable`](../api_docs/core/variable.rst) —
  `Variable`, `Parameter`, `Buffer`.
- [`spectrax.core.graph`](../api_docs/core/graph.rst) — `export`,
  `bind`, `clone`, `update`, `pop`, `iter_modules`,
  `iter_variables`, `find`.
- [`spectrax.core.state`](../api_docs/core/state.rst) — `State`.
- [`spectrax.core.containers`](../api_docs/core/containers.rst) —
  base container plumbing.
- [`spectrax.nn`](../api_docs/nn/index.rst) — built-in layers.
