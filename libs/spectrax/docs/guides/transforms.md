# Transforms

Every spectrax transform is a drop-in for its JAX counterpart that
**understands `Module` arguments and their state**. `spx.eval_shape`,
`spx.jit`, `spx.grad`, `spx.value_and_grad`, `spx.jvp`, `spx.vjp`,
`spx.vmap`, `spx.scan`, `spx.associative_scan`, `spx.remat`,
`spx.cond`, `spx.switch`, `spx.fori_loop`, `spx.while_loop` — all of
them accept `Module` instances directly without you having to manually
extract `(GraphDef, State)` pairs.

This guide walks each transform with worked examples, covers the
shared mechanics, and ends with composition patterns and pitfalls.

## How transforms work (the shared shim)

Every spectrax transform follows the same five-step recipe:

1. **Locate** `Module` instances in `args` / `kwargs`
   (`locate_and_strip`).
2. **Export** each to `(GraphDef, State)`. The state values are the
   current `Variable._value`s; the graph-def encodes the structure.
3. **Build a pure function** (`make_pure`) that, given the list of
   states, rebinds fresh `Module` instances from them, calls the
   user's function, and re-exports to capture any mutations.
4. **Apply the underlying JAX transform** (e.g. `jax.jit`,
   `jax.value_and_grad`) to that pure function.
5. **On exit, most transforms write detected mutations back** to the
   live module — guarded by the `mutable=` selector
   (`apply_mutations`). `spx.eval_shape` is the read-only exception:
   it keeps abstract updates local to the trace.

The shim lives in [`spectrax.transforms.split_merge`](../api_docs/transforms/split_merge.rst)
and is shared across every transform — extending spectrax with a new
module-aware transform is mostly "wire your transform to the shim."

The cost of each step is heavily optimized — see
[Performance](../performance.md) for the breakdown.

---

## `spx.eval_shape`

Module-aware `jax.eval_shape`. Use it when you want abstract
shape/dtype inference for a function that accepts or returns
`Module` objects.

```python
x_spec = jax.ShapeDtypeStruct((8, 128), jnp.float32)
out_spec = spx.eval_shape(lambda m, x: m(x), model, x_spec)

abs_model = spx.eval_shape(lambda: MyBlock(128, rngs=spx.Rngs(0)))
gdef, abs_state = spx.export(abs_model)
```

Important difference from mutating transforms like `spx.jit`:
`spx.eval_shape` never writes abstract updates back to the live input
modules. Any `var.value = ...` assignment that happens during abstract
evaluation stays local to the traced copy, which keeps shape inference
safe while still allowing abstract module outputs.

---

## `spx.jit`

Module-aware `jax.jit`. Forwards every upstream kwarg
(`static_argnums`, `static_argnames`, `donate_argnums`, `in_shardings`,
`out_shardings`, `keep_unused`, `device`, `backend`, `inline`,
`compiler_options`); adds one new one: `mutable=`.

### Pure forward

```python
@spx.jit
def predict(m, x):
    return m(x)


y = predict(model, x)                    # first call: trace + compile
y = predict(model, x)                    # subsequent: cache hit
y = predict(other_model, x2)             # different graph -> new compile
```

### With mutation

The plain `jax.jit(f)(model)` pattern with a mutating `f` **silently
drops** `.value = ...` writes — the trace runs on a fresh unflattened
copy inside the trace boundary, and Python attribute writes don't
propagate back. spectrax fixes this, but you must **explicitly
declare which collections are mutable**:

```python
@spx.jit(mutable="batch_stats")          # BatchNorm running stats survive
def forward(m, x):
    return m(x)


@spx.jit(mutable="parameters" | "batch_stats")
def step(m, o, x, y):
    def loss(m):
        return ((m(x) - y) ** 2).mean()
    loss_val, grads = spx.value_and_grad(loss)(m)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt
```

Anything written to a non-mutable collection raises
`spx.IllegalMutationError`. This is by design — silent drop is a
correctness bug.

### `mutable=` selector

`mutable=` accepts the full [Selector DSL](selectors.md):

| `mutable=`                  | What it allows                          |
| --------------------------- | --------------------------------------- |
| `()` (default)              | Pure forward — no mutations             |
| `"parameters"`                  | Weight updates                          |
| `"batch_stats"`             | BatchNorm running mean/var              |
| `"cache"`                   | KV-cache writes during generation       |
| `"fp8_meta"`                | FP8 scale / amax updates                |
| `"intermediates"`           | `self.sow(...)` captures                |
| `"rng"`                     | `Rngs` advancement (rare; usually pure) |
| `nn.LoraParameter`          | LoRA adapter writes                     |
| `("parameters", "lora")`        | Union of two collections                |
| `spx.path_contains("attn")` | Only paths containing "attn"            |
| Custom `Selector`           | Anything composable                     |

### Compile cache

`spx.jit` maintains a two-level cache:

- **Identity cache** keyed by `(id(module1), id(module2), …)` plus a
  global graph epoch. Fast path: same model instance, no structural
  change -> O(1) dict lookup, no graph-def hashing.
- **Structural cache** keyed by the full `GraphDef` tuple. Triggered
  when you swap models or pass a freshly-built one with the same
  structure as a cached one.

You normally don't think about either — but if you're writing a
training loop where the model is reconstructed every iteration (rare),
the identity cache misses every time and you fall back to the
structural cache. Keep the live module instance.

### Inspecting the cache

```python
@spx.jit(mutable="parameters")
def step(m, o, x, y): ...

print(len(step._spx_compile_cache))      # how many structural compiles
print(len(step._spx_id_cache))           # how many identity-cache entries
```

If structural-cache size is growing unexpectedly, that's a sign of
re-tracing — usually because of a changing static value (a Python
scalar in `spx.scope`, a `static_argnames` value, etc.).

---

## `spx.grad` / `spx.value_and_grad`

Module-aware `jax.grad` and `jax.value_and_grad`. The differentiation
target is selected by `wrt=` — same selector DSL as `mutable=`. By
default, `wrt="parameters"` differentiates the `parameters` collection of the
first `Module` argument.

### Default: train all parameters

```python
def loss_fn(m, x, y):
    return jnp.mean((m(x) - y) ** 2)


grads = spx.grad(loss_fn)(model, x, y)
loss_val, grads = spx.value_and_grad(loss_fn)(model, x, y)
```

`grads` is a `State` with the same shape as the `"parameters"` slice of
the model — `{"parameters": {"fc1.weight": grad_array, ...}}`.

### LoRA: train only adapters

```python
grads = spx.grad(loss_fn, wrt="lora")(model, x, y)
# equivalent:
grads = spx.grad(loss_fn, wrt=nn.LoraParameter)(model, x, y)
```

Base weights never see a gradient — saves both compute (smaller HLO)
and memory (smaller cotangents).

### Selector composition

```python
# Train every parameter except the classifier head
sel = spx.as_selector(nn.Parameter) - spx.path_endswith("head.weight")
grads = spx.grad(loss_fn, wrt=sel)(model, x, y)

# Train only attention sub-layers' parameters
sel = spx.path_contains("attn") & spx.as_selector("parameters")
grads = spx.grad(loss_fn, wrt=sel)(model, x, y)
```

### `argnum` and `has_aux`

```python
# `argnum=` — differentiate against arg index N (default: first Module)
grads = spx.grad(fn, argnum=2)(rngs, x, model, y)

# `has_aux=True` — loss returns (loss, aux)
def loss_with_aux(m, x, y):
    pred = m(x)
    loss = jnp.mean((pred - y) ** 2)
    return loss, {"pred": pred, "norm": jnp.linalg.norm(pred)}


(loss_val, aux), grads = spx.value_and_grad(loss_with_aux, has_aux=True)(model, x, y)
print(aux["norm"])
```

### Combining with optax

```python
from spectrax.contrib import Optimizer
import optax

opt = Optimizer.create(model, optax.adamw(3e-4))


@spx.jit(mutable="parameters")
def step(m, o, x, y):
    loss_val, grads = spx.value_and_grad(loss_fn)(m, x, y)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt


for x, y in batches:
    loss_val, opt = step(model, opt, x, y)
```

`Optimizer.create(model, tx, wrt=...)` allocates optax state only for
the matched leaves — pass the same selector you'd use for `wrt=` in
`grad`.

---

## `spx.jvp` / `spx.vjp`

Forward- and reverse-mode autodiff wrappers for cases where you want
more direct access than `grad`.

```python
m_tangent = jax.tree.map(jnp.zeros_like, model)
x_tangent = jnp.ones_like(x)

out, tangent_out = spx.jvp(
    lambda m, x: m(x).sum(),
    (model, x),
    (m_tangent, x_tangent),
)

out, pullback = spx.vjp(lambda m, x: m(x).sum(), model, x)
grads_model, grads_x = pullback(jnp.array(1.0))
```

For module primals, `spx.vjp` returns `State` cotangents, mirroring the
shape conventions of `spx.grad`. `spx.jvp` accepts module tangents as a
matching `Module`, a `State`, or any pytree matching the exported state.

Both wrappers also accept `mutable=`. Just like `spx.jit`, those writes
apply to the **primal** forward pass only; the returned pullback is pure.

---

## `spx.vmap`

Module-aware `jax.vmap`. By default, `Module` arguments are mapped
with `in_axes=None` (broadcast across replicas), and the leaves of
their `State` are mapped per-collection according to the supplied
`in_axes`.

### Vectorize over data

```python
# Same model, batched input
batched = spx.vmap(lambda m, x: m(x), in_axes=(None, 0))
ys = batched(model, xs)                  # xs: (B, ...) -> ys: (B, ...)
```

### Per-collection axes via `StateAxes`

```python
from spectrax.transforms import StateAxes

# Replicate parameters across batch but split RNG state per-replica
axes = StateAxes({"parameters": None, "rng": 0})
ys = spx.vmap(
    lambda m, x: m(x),
    in_axes=(axes, 0),
)(model_with_dropout, xs)
```

### Ensemble: vectorize over models

Train an ensemble of identical-shape models in parallel by stacking
their `State`s along axis 0 and broadcasting `Module` structure:

```python
# Build N replicas with different seeds
replicas = [MLP(16, 64, 4, rngs=spx.Rngs(i)) for i in range(8)]

# Stack their states
stacked_state = jax.tree.map(
    lambda *xs: jnp.stack(xs, axis=0),
    *[spx.export(r)[1] for r in replicas],
)

@spx.vmap
def predict(state, x):
    m = spx.bind(spx.export(replicas[0])[0], state)   # rebind per-replica
    return m(x)


# Each replica predicts on the same input — output shape (8, ...)
ys = predict(stacked_state, x)
```

---

## `spx.scan`

Module-aware `jax.lax.scan`. Carries a module (or any pytree) through
a per-step body fn.

### Iterate over a sequence

```python
def body(m, h, xt):
    new_h = m.cell(h, xt)                # m is a "carrier" module
    return new_h, new_h                  # (next_carry, output)


h0 = jnp.zeros((batch, hidden_dim))
xs = jnp.ones((seq_len, batch, in_dim))
final_h, hs = spx.scan(body)(rnn, h0, xs)
```

### Scan over layers (memory-efficient)

```python
class LayerScan(spx.Module):
    """Apply the same Block N times via lax.scan — saves compile time."""

    def __init__(self, n_layers, d, *, rngs):
        super().__init__()
        # Stack parameters with leading axis = n_layers
        sub = Block(d, rngs=rngs)
        gdef, state = spx.export(sub)
        stacked = jax.tree.map(
            lambda v: jnp.broadcast_to(v, (n_layers, *v.shape)).copy(),
            state,
        )
        self.gdef = spx.Opaque(gdef)
        self.layer_parameters = spx.Parameter(stacked)

    def forward(self, x):
        def body(x, layer_state):
            block = spx.bind(self.gdef.value, layer_state)
            return block(x), None

        return spx.scan(body)(x, self.layer_parameters.value)[0]
```

This compiles **one** `Block` and runs it N times via `scan` — much
faster compile + smaller HLO than `nn.Sequential` of distinct blocks.

---

## `spx.associative_scan`

Module-aware `jax.lax.associative_scan`. Use it for parallel prefix
computations whose combine is genuinely associative.

```python
class PrefixAdder(spx.Module):
    def __init__(self):
        super().__init__()
        self.scale = spx.Buffer(jnp.array(1.0), kind="batch_stats")


def combine(m, a, b):
    return m.scale.value * (a + b)


xs = jnp.arange(8.0)
ys = spx.associative_scan(combine, PrefixAdder(), xs)
```

Important difference from `spx.scan`: there is **no module-state carry**.
`jax.lax.associative_scan` builds a tree-structured parallel prefix, so
there is no single "next module state" to thread through the combine.
For that reason `spx.associative_scan` is **pure-only**: any
`var.value = ...` write inside the combine raises
`spx.IllegalMutationError`.

---

## `spx.remat`

Gradient checkpointing — saves activation memory at the cost of a
backward re-compute pass.

```python
@spx.remat
def expensive_block(m, x):
    return m(x)


# Or as a wrapper:
remat_forward = spx.remat(model.forward)
```

`spx.remat_scan` combines `remat` and `scan` for layer-stack training:
the per-layer activations are recomputed in backward, so peak memory
is `O(activation_per_layer)` instead of `O(n_layers · activation)`:

```python
spx.remat_scan(body)(carry, xs)          # checkpointed scan
```

---

## Control flow

JAX control-flow primitives, made module-aware so state flows through
branches correctly.

### `spx.cond` / `spx.switch`

```python
def true_branch(m, x):
    return m.heavy_path(x)


def false_branch(m, x):
    return m.cheap_path(x)


@spx.jit
def predict(m, x, use_heavy):
    return spx.cond(use_heavy, true_branch, false_branch, m, x)


# n-way branch
def b0(m, x): return m.head_a(x)
def b1(m, x): return m.head_b(x)
def b2(m, x): return m.head_c(x)
y = spx.switch(idx, [b0, b1, b2], model, x)
```

Both branches must agree on the **set of collections written** —
spectrax can't reconcile a branch that writes `"parameters"` with one
that doesn't.

### `spx.fori_loop` / `spx.while_loop`

```python
# Fixed-trip loop
def body(i, carry):
    m, x = carry
    return (m, x + m(x))


_, x_final = spx.fori_loop(0, 10, body, (model, x_init))


# Data-dependent loop
def cond_fn(carry):
    m, x = carry
    return jnp.linalg.norm(x) > 1e-3


def body_fn(carry):
    m, x = carry
    return (m, x - m(x))


_, x_final = spx.while_loop(cond_fn, body_fn, (model, x_init))
```

---

## RNG inside transforms

`Rngs` lives in the `"rng"` collection and survives transforms
naturally. For per-replica or per-iteration randomness, use the
splitter helpers:

```python
from spectrax.transforms import split_rngs, split_stream_keys

# Fresh per-iteration keys for an scan body
@spx.scan
def body(m, carry, t):
    keys = split_stream_keys(rngs, n_iters=10)[t]    # or shape match
    return body_step(m, carry, t, keys)


# Per-replica RNG state for vmap
rngs_per_replica = split_rngs(rngs, n=8)
spx.vmap(forward)(rngs_per_replica, x)
```

See [`rng`](../api_docs/rng/index.rst) for the full RNG API.

---

## When to use which transform

| You want to...                       | Use                               |
| ------------------------------------ | --------------------------------- |
| Infer shapes / dtypes abstractly     | `spx.eval_shape`                  |
| Compile a forward / training step    | `spx.jit`                         |
| Compute gradients w.r.t. parameters | `spx.grad` / `spx.value_and_grad` |
| Run forward-mode autodiff            | `spx.jvp`                         |
| Build a pullback explicitly          | `spx.vjp`                         |
| Batch over a data dim                | `spx.vmap`                        |
| Train a recurrent / layer-stack loop | `spx.scan`                        |
| Parallel prefix over an associative op | `spx.associative_scan`          |
| Trade activation memory for compute  | `spx.remat` / `spx.remat_scan`    |
| Dynamic branching on scalar data     | `spx.cond` / `spx.switch`         |
| Fixed-trip integer loop              | `spx.fori_loop`                   |
| Data-dependent loop                  | `spx.while_loop`                  |
| Differ randomness across replicas    | `split_rngs`                      |
| Differ randomness across iterations  | `split_stream_keys`               |

---

## Composition patterns

### `vmap` inside `jit`

The common case — compile once, vectorize across the batch dim.
spectrax handles the module split correctly:

```python
@spx.jit
def batched_loss(m, xs, ys):
    losses = spx.vmap(lambda m, x, y: ((m(x) - y) ** 2).mean(), in_axes=(None, 0, 0))(m, xs, ys)
    return losses.mean()
```

### `grad` of a `vmap`'d loss

Gradients flow back through `vmap` automatically:

```python
@spx.jit
def step(m, xs, ys):
    def loss(m):
        per_sample = spx.vmap(lambda m, x, y: ((m(x) - y) ** 2).sum(), in_axes=(None, 0, 0))(m, xs, ys)
        return per_sample.mean()
    return spx.value_and_grad(loss)(m)
```

### `scan` inside `grad`

```python
@spx.jit
def step(m, xs, y):
    def loss(m):
        h0 = jnp.zeros((batch, hidden))
        _, hs = spx.scan(lambda m, h, xt: (m.cell(h, xt), m.cell(h, xt)))(m, h0, xs)
        return ((m.head(hs[-1]) - y) ** 2).mean()
    return spx.value_and_grad(loss)(m)
```

Backprop-through-time is automatic; spectrax just ensures `m` is
visible inside the scan body.

### `remat` for a single layer

Apply checkpointing only to the most memory-hungry block:

```python
class HybridStack(spx.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.cheap = nn.Linear(...)
        self.expensive = HugeAttention(...)        # check-point this one

    def forward(self, x):
        x = self.cheap(x)
        return spx.remat(lambda m, x: m(x))(self.expensive, x)
```

---

## Pitfalls

### 1. Silent drop in plain `jax.jit`

```python
@jax.jit                                   # PLAIN jax.jit — silent drop!
def step(m, x):
    m.fc.weight.value -= 0.01 * grad       # this write is LOST inside the trace
    return m(x)
```

Use `spx.jit(mutable="parameters")` for in-trace mutation.

### 2. Mismatched compile cache from changing static values

```python
def step(m, x, alpha):
    return m(x) * alpha


spx.jit(step, static_argnames="alpha")(m, x, 0.1)    # compile #1
spx.jit(step, static_argnames="alpha")(m, x, 0.2)    # compile #2
spx.jit(step, static_argnames="alpha")(m, x, 0.3)    # compile #3
# ... eats your compile cache ...
```

If `alpha` changes per step, demote it to a JAX array (a tracer):

```python
spx.jit(step)(m, x, jnp.asarray(0.1))     # one compile, traces over alpha
```

### 3. Wrong `wrt=` for the loss

```python
def loss(m):
    return ((m.lora(x) - y) ** 2).mean()   # only uses adapter outputs


grads = spx.grad(loss)(m)                  # wrt='parameters' — gets ZERO gradients
grads = spx.grad(loss, wrt="lora")(m)      # ✓
```

### 4. Forgetting to declare `mutable="batch_stats"`

```python
@spx.jit                                   # mutable=() default
def step(m, x):
    return m(x)                            # BatchNorm tries to write running stats -> IllegalMutationError
```

`BatchNorm*d` updates `batch_stats` on every forward pass in training
mode. Declare it mutable explicitly.

### 5. Re-tracing on every call due to `spx.scope` static churn

If a `spx.scope(...)` value changes every call (e.g. a per-step
counter as a Python int), `spx.jit` re-compiles every time. See
[the scope guide](scope.md#footguns) for the mitigation (use a
`jnp.asarray` so it becomes a tracer).

---

## API reference

- [`spectrax.transforms.jit`](../api_docs/transforms/jit.rst)
- [`spectrax.transforms.grad`](../api_docs/transforms/grad.rst)
- [`spectrax.transforms.vmap`](../api_docs/transforms/vmap.rst)
- [`spectrax.transforms.scan`](../api_docs/transforms/scan.rst)
- [`spectrax.transforms.eval_shape`](../api_docs/transforms/eval_shape.rst)
- [`spectrax.transforms.remat`](../api_docs/transforms/remat.rst)
- [`spectrax.transforms.control_flow`](../api_docs/transforms/control_flow.rst)
- [`spectrax.transforms.split_merge`](../api_docs/transforms/split_merge.rst)
- [`spectrax.transforms.rng_axes`](../api_docs/transforms/rng_axes.rst)
