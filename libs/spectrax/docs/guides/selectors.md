# Selectors

A **Selector** is spectrax's composable predicate over a module graph.
It is the single DSL shared by **every** API that takes "a subset of
the model": `grad(wrt=...)`, `value_and_grad(wrt=...)`,
`jit(mutable=...)`, `Optimizer(wrt=...)`, `partition_state(...)`,
`iter_variables(select=...)`, `iter_modules(select=...)`,
`freeze(...)`, `pop(...)`. Learn the DSL once, use it everywhere.

## The mental model

A selector is a **specification**, not a result. It carries no
references to a specific module; you apply it to a model when you
need the matching variables:

```python
sel = spx.select().of_type(spx.nn.LoraParameter)        # spec
matches = sel.apply(model)                              # result: [(path, var), ...]
```

Selectors compose with `|` (union), `&` (intersection), `-` (set
difference), `~` (invert) — they form a small algebra over
`(Variable, path) -> bool`.

## Sugar forms

Anywhere a selector is accepted, these are auto-coerced via
`spx.as_selector` (see [`spectrax.core.selector`](../api_docs/core/selector.rst)):

```python
"parameters"                          # collection-name filter
spx.nn.LoraParameter              # Variable subclass -> instance-of filter
spx.Parameter                     # likewise — any Variable subclass
("parameters", "adapter")             # union of collection names
(spx.Parameter, spx.Buffer)       # union of types
(spx.nn.LoraParameter, "buffers") # mixed iterable
lambda v, path: v.size > 1000     # arbitrary (variable, path) predicate
None                              # match nothing — safe default
```

This means you rarely need to write `spx.select().variables("parameters")`
explicitly — just pass `"parameters"` and spectrax does the conversion.

## Builder chain

For anything more complex than the sugar forms, start with
`spx.select()` and chain methods:

```python
sel = (
    spx.select()
    .at_instances_of(nn.Linear)              # ancestor must be Linear
    .not_instances_of(nn.LayerNorm)          # ... but never under LayerNorm
    .variables("parameters")                     # variable.kind == "parameters"
    .exclude_variables("cache")              # never "cache"
    .of_type(spx.nn.LoraParameter)           # variable's class
    .at_path("backbone.**.weight")           # path-glob filter
    .where_variable(lambda v, p: v.size > 1000)
)
```

Every method returns a **new** selector — selectors are frozen
dataclasses.

| Method                      | Filters by                         |
| --------------------------- | ---------------------------------- |
| `at_instances_of(*types)`   | Module ancestor instance-of        |
| `not_instances_of(*types)`  | Module ancestor NOT instance-of    |
| `variables(*kinds)`         | Variable kind                      |
| `exclude_variables(*kinds)` | NOT variable kind                  |
| `of_type(*types)`           | Variable subclass                  |
| `not_of_type(*types)`       | NOT Variable subclass              |
| `at_path(*globs)`           | Dotted path glob (`*`, `**`)       |
| `where_module(pred)`        | Custom `(module, path) -> bool`    |
| `where_variable(pred)`      | Custom `(variable, path) -> bool`  |
| `where(pred)`               | Both module and variable predicate |

## Operators

```python
a | b              # union — match if either matches
a & b              # intersection — match only if both match
a - b              # set difference — equivalent to a & ~b
~a                 # invert — match if a does NOT match (variables only)
spx.all_of(*sels)  # intersection of many
spx.any_of(*sels)  # union of many
spx.not_(sel)      # equivalent to ~sel
```

The intersection of two selectors with **module filters** matches a
variable only if some ancestor satisfies BOTH module filters
simultaneously — useful when, e.g., you want "Linear weights inside
the encoder":

```python
(spx.select().at_instances_of(nn.Linear) & spx.path_startswith("encoder."))
```

## Path globs

Paths are dotted strings: `"blocks.0.attn.q.weight"`. Glob syntax:

| Glob | Matches                             |
| ---- | ----------------------------------- |
| `*`  | exactly one path segment            |
| `**` | any number of segments (incl. zero) |

Examples:

```python
spx.select().at_path("blocks.0.attn.q.weight")    # one specific path
spx.select().at_path("blocks.*.attn.q.weight")    # all blocks, q only
spx.select().at_path("blocks.**.weight")          # all weights under blocks
spx.select().at_path("encoder.**")                # everything under encoder
spx.select().at_path("**.bias")                   # any bias anywhere
```

Module-level helpers for common cases:

```python
spx.path_contains("attn")                # substring match
spx.path_startswith("encoder.")          # prefix
spx.path_endswith(".bias")               # suffix
spx.of_type(spx.nn.LoraParameter)        # by Variable class
```

## Sentinel selectors

| Sentinel         | Meaning              |
| ---------------- | -------------------- |
| `spx.Everything` | Match every variable |
| `spx.Nothing`    | Match no variable    |

Useful as defaults or "nothing" markers in conditional logic:

```python
sel = spx.Nothing
if include_lora:
    sel |= spx.as_selector("lora")
if include_buffers:
    sel |= spx.as_selector("buffers")

opt = spx.contrib.Optimizer.create(model, tx, wrt=sel)
```

---

## Cookbook

### 1. Train everything (default)

```python
grads = spx.grad(loss)(model, x, y)             # wrt="parameters" implicit
```

### 2. LoRA-only fine-tune

```python
grads = spx.grad(loss, wrt="lora")(model, x, y)
opt = spx.contrib.Optimizer.create(model, tx, wrt="lora")
```

### 3. Train all parameters except the classifier head

```python
sel = spx.as_selector("parameters") - spx.path_contains("head")
grads = spx.grad(loss, wrt=sel)(model, x, y)
```

### 4. Freeze the encoder, train only the decoder

```python
sel = spx.as_selector("parameters") & spx.path_startswith("decoder.")
grads = spx.grad(loss, wrt=sel)(model, x, y)
```

### 5. Train base + adapter at different LRs

Use `MultiOptimizer`:

```python
from spectrax.contrib import MultiOptimizer
import optax

opt = MultiOptimizer(
    {"parameters": optax.adamw(1e-4),                # slow base
     "lora":   optax.adamw(1e-3)},               # fast adapter
    module=model,
)
opt.update(grads)
```

### 6. Apply weight decay only to weights, not biases

```python
weights = spx.path_endswith(".weight") & spx.as_selector("parameters")
biases = spx.path_endswith(".bias") & spx.as_selector("parameters")

opt = MultiOptimizer({
    weights: optax.adamw(3e-4, weight_decay=0.01),
    biases:  optax.adamw(3e-4, weight_decay=0.0),
}, module=model)
```

### 7. List all Linear weights

```python
sel = spx.select().at_instances_of(nn.Linear).variables("parameters").at_path("**.weight")
for path, var in sel.apply(model):
    print(path, var.value.shape)
```

### 8. Find every very-large parameter

```python
big = spx.select().where_variable(lambda v, p: v.size > 10_000_000)
for path, var in big.apply(model):
    print(path, f"{var.size / 1e6:.1f}M")
```

### 9. Zero out a sub-tree's weights

```python
sel = spx.path_startswith("ablation_branch.").variables("parameters")
sel.set(model, lambda v: jnp.zeros_like(v.value))
```

### 10. Mutable cache during decoding only

```python
@spx.jit(mutable="cache")
def decode_step(model, kv_cache, x):
    return model(x, cache=kv_cache)


@spx.jit                                        # no mutable= — pure forward only
def encode(model, x):
    return model(x)
```

### 11. Capture intermediates from one specific layer

```python
class Net(spx.Module):
    def forward(self, x):
        h = self.encoder(x)
        if spx.scope.get("capture_encoder", default=False):
            self.sow("intermediates", "encoder_out", h)
        return self.decoder(h)


@spx.jit(mutable="intermediates")
def run_with_capture(m, x):
    return m(x)


with spx.scope(capture_encoder=True):
    _ = run_with_capture(model, x)

inters = spx.pop(model, "intermediates")
print(inters)              # {"sow_intermediates_encoder_out": array(...)}
```

### 12. Custom Variable class as a selector key

```python
class AdapterParam(spx.Variable):
    default_kind = "adapter"


# Later, anywhere:
grads = spx.grad(loss, wrt=AdapterParam)(model, x, y)
opt = spx.contrib.Optimizer.create(model, tx, wrt=AdapterParam)
adapters_only = AdapterParam   # used as a Selector-coercible value

# Or by string:
grads = spx.grad(loss, wrt="adapter")(model, x, y)
```

### 13. Two adapter sets, both trainable

```python
class LoraA(spx.Variable): default_kind = "lora_a"
class LoraB(spx.Variable): default_kind = "lora_b"


sel = spx.as_selector(LoraA) | spx.as_selector(LoraB)
grads = spx.grad(loss, wrt=sel)(model, x, y)
opt = spx.contrib.Optimizer.create(model, tx, wrt=sel)
```

### 14. Path glob with int positions

`nn.ModuleList` / `nn.Sequential` use integer indices — paths
contain bare digits:

```python
spx.select().at_path("blocks.0.**")           # only block 0
spx.select().at_path("blocks.*.attn.**")      # any block, attention only
spx.select().at_path("blocks.[02468].**")     # ❌ no character classes (use predicate)
spx.select().where(lambda obj, p: p.startswith("blocks.") and int(p.split(".")[1]) % 2 == 0)
```

### 15. Compose with other selectors at the call site

```python
base = spx.as_selector("parameters")
no_head = base - spx.path_contains("head")
no_norm = base - spx.path_endswith(".weight")

# Use one or the other depending on flag
sel = no_head if exclude_head else no_norm
opt = spx.contrib.Optimizer.create(model, tx, wrt=sel)
```

---

## Selector internals

When you call `sel.apply(module)`, spectrax:

1. Walks the module tree once via `_spx_graph_children()`.
2. Tracks an "ancestor matched" flag for module-level filters.
3. Deduplicates by `Variable.ref_id` (so shared weights are reported
   once at the canonical path).
4. Returns a list of `(path, variable)` tuples in canonical-path
   order.

For the dispatch hot path (`spx.jit`, `apply_mutations`) spectrax has
a fast path that uses the module's cached `(collection, path,
Variable)` triples instead of re-walking the tree on every call —
selectors with no module filter benefit from this automatically.

See [Performance](../performance.md) for the optimization details.

## Comparison: why one DSL?

Other frameworks all have multiple ad-hoc APIs for this:

- `freeze(path_patterns)`
- `grad(wrt="parameters")` / `grad(model.named_parameters())`
- `Optimizer(label_fn=...)` / `optax.partition(label_fn, ...)`
- `partition_by_regex(...)`
- `params_filter=callable`

Each has its own conventions, edge cases, and bugs. spectrax has
**one** DSL — `Selector` — and every API that takes a subset takes
one. You learn it once.

## API reference

- [`spectrax.core.selector`](../api_docs/core/selector.rst) —
  `Selector`, `select`, `as_selector`, `all_of`, `any_of`, `not_`,
  `of_type`, `path_contains`, `path_startswith`, `path_endswith`,
  `Everything`, `Nothing`, `SelectorSugar`.
