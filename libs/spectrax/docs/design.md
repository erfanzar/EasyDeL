# Design

spectrax is the result of picking a handful of ideas from
[PyTorch](https://pytorch.org/) and
[Equinox](https://docs.kidger.site/equinox/), and composing them so the
boundaries are explicit and the hot paths are fast.

## Principles

### 1. PyTorch-shaped eager surface

Subclass `Module`, override `forward`, call `model(x)`. No implicit
functional rewrites, no metaclass magic, no compile-time indirection.
You can step through a forward pass in `pdb` and see an honest Python
stack.

### 2. Modules are JAX pytrees

Modules are **registered as pytrees** via `jax.tree_util`. Flatten
and unflatten go through the existing `spx.export` / `spx.bind` pair:

```python
# Flatten
gdef, state = spx.export(model)
leaves = jax.tree_util.tree_leaves(state)

# Unflatten
model2 = spx.bind(gdef, state)
```

That means `jax.jit`, `jax.tree.map`, `jax.value_and_grad` accept
modules **directly**. The `(GraphDef, State)` aux carries structural
and runtime state (training flag, hooks, dtype policy) so round-trips
preserve more than just weights.

### 3. State lives in `Variable` cells

`Variable` is a reference cell holding an array. Subclasses tag their
`default_kind` (the "collection name") — `Parameter` -> `"parameters"`,
`Buffer` -> `"buffers"`, `LoraParameter` -> `"lora"`, `Fp8Meta` ->
`"fp8_meta"`, etc. `State` is `{collection: {path: leaf}}` — flat,
keyed, transparent.

User-defined collections are a one-line subclass:

```python
class AdapterParam(spx.Variable):
    default_kind = "adapter"
```

No registration, no metaclass, no manifest file.

### 4. One filter DSL

Every API that takes "a subset of the model" takes a `Selector` or one
of its sugar forms: a collection name string, a `Variable` subclass, a
path glob, a predicate, a composite. The same selector serves
`grad(wrt=...)`, `jit(mutable=...)`, `Optimizer(wrt=...)`,
`partition_state(...)`, `iter_variables(select=...)`,
`freeze(...)`. See [selectors](guides/selectors.md).

### 5. Transforms all do the same thing

Every spectrax transform (`jit`, `grad`, `vmap`, `scan`, `remat`,
`cond`, `switch`, `fori_loop`, `while_loop`) implements the same
five-step split/merge recipe:

1. Locate `Module` instances in `args` / `kwargs`.
2. `export` each to `(GraphDef, State)`.
3. Build a pure function that rebinds fresh modules from the states,
   calls the user fn, and re-exports to capture mutations.
4. Apply the underlying JAX transform.
5. Write detected mutations back to the live modules (guarded by the
   `mutable=` selector).

See [`spectrax.transforms.split_merge`](api_docs/transforms/split_merge.rst)
for the shared implementation.

## Comparisons

### vs. Equinox

Equinox makes modules **immutable frozen dataclasses** and treats them
purely functionally — every update returns a new module. spectrax
keeps the PyTorch-shaped mutating eager surface (you write
`var.value = new_val`), and only crosses into functional territory when
a transform activates. The trade-off: Equinox gets purity by fiat;
spectrax gets it by discipline at the transform boundary.

## Performance

The hot dispatch path has received heavy optimization:

- **Per-module `export()` cache** keyed on a global graph epoch;
  invalidated only when `Module.__setattr__` changes shape.
- **Fused `locate_and_strip`** walks args and exports in one pass.
- **`GraphDef.__hash__` memoization** skips the recursive tuple walk.
- **Two-level jit cache**: module identity + graph-def structural.
- **Kinds-only fast paths** in `apply_mutations` for common `mutable=`
  selectors.
- **`__slots__`** on `_ModuleRef` and `Optimizer`.
- **Fast path-agnostic flatten** for `State` (skips `GetAttrKey`
  allocation per leaf).

On the tiny-CPU dispatch-bound benchmark (2-layer / d=64 / batch=2
transformer), spectrax reaches **1.83x** the throughput of
`flax.nnx`; on smaller d=48 it hits **2.0x**. See
[performance](performance.md) for details and the raw numbers.

## Non-goals

- **Not a full PyTorch port.** We don't implement autograd hooks,
  `register_parameter`-style manual registration, or the full
  optim.X class hierarchy. Autograd is JAX.
- **Not a full NNX port.** We don't implement reference-tracking
  across arbitrary cycles, `NodeRef` with identity semantics, or
  `UpdateContext` staging. The use-cases those solve are rare enough
  that explicit `export` / `bind` handles them with less machinery.
- **No ecosystem lock-in.** No custom optimizer API (use optax via
  `spectrax.contrib.Optimizer`), no custom serialization (use
  `jax.tree_util` + `safetensors` / `msgpack`), no custom
  distributed runtime (use `jax.sharding.Mesh`).
