# Dynamic scope

Some values get used **deep inside** a model but would otherwise need
to be plumbed through every intermediate `forward(self, ...)`
signature: attention masks, dropout RNG keys, position ids, segment
ids, `is_training` flags, decoder-time KV caches, decode-step counters.

`spx.scope` is a dynamic-scope mechanism that lets you bind those
values once at the call site and read them anywhere in the call
stack:

```python
with spx.scope(mask=attn_mask, is_training=False):
    out = model(x)              # deep inside model.forward: spx.scope.get("mask") works
```

Underneath, `spx.scope` is a thin layer over Python's
[`contextvars.ContextVar`](https://docs.python.org/3/library/contextvars.html),
with the extra plumbing required to compose correctly with
`spx.jit`'s tracing.

## API at a glance

```python
import spectrax as spx

# Push a frame
with spx.scope(mask=mask, is_training=False, beta=0.5):
    ...

# Read anywhere in the enclosed call stack
mask = spx.scope.get("mask")                    # KeyError if no scope
training = spx.scope.get("is_training", True)   # with default

# Inspect the current stack (debug helper)
snap = spx.scope.snapshot()                     # {"mask": ..., "is_training": False, ...}

# Split a snapshot into traced (arrays) vs static (Python) halves
traced, static = spx.scope.partition(snap)
```

Nested scopes shadow:

```python
with spx.scope(a=1, b=2):
    with spx.scope(b=99):
        spx.scope.get("a")     # -> 1 (from outer)
        spx.scope.get("b")     # -> 99 (inner shadows outer)
    spx.scope.get("b")         # -> 2 (inner gone)
```

## Why not just pass the value as a function arg?

Plumbing an attention mask through every transformer block, every
`MultiheadAttention`, every `forward` is mechanical and error-prone.
You add a `mask=None` arg to `Block.forward`, then to
`TransformerStack.forward`, then to `Model.forward` — and propagate
it. Every time you add a new "context-y" thing, the same
plumbing.

Scope lets you skip it for **escape-hatch context values**. Use it
deliberately:

| Best for                            | Not for                      |
| ----------------------------------- | ---------------------------- |
| Things only used in 1-2 deep layers | Top-level signal-flow values |
| Read-mostly values                  | Outputs / writes             |
| Per-call values that don't fit init | Persistent model state       |
| Modes / flags                       | Functional return values     |

If two equally valid layers in the tree both want to read X, scope is
fine. If X is part of every layer's input contract, just put it on
the call signature.

## Composition with `spx.jit`

`spx.jit` is **scope-aware**. When dispatching, it inspects the
active scope stack and:

- **Static values** (Python scalars, strings, tuples of statics) get
  folded into the **compile cache key**. Different static snapshots
  trigger distinct compiles, so `if scope.get("training"):` inside a
  jitted forward specializes correctly per value.
- **Array values** (`jax.Array`, NumPy arrays — anything with a
  `.shape` attribute) get **lifted into the jit input pytree** as
  tracers. `spx.scope.get("mask")` inside the trace returns a tracer,
  not a baked-in concrete. Changing the value between calls does
  **not** trigger re-compile (same shape / dtype).

```python
@spx.jit
def forward(m, x):
    mask = spx.scope.get("mask")              # tracer (lifted)
    flag = spx.scope.get("double", False)     # static (baked in)
    y = m(x) * mask
    return y * 2.0 if flag else y


with spx.scope(mask=mask_a, double=False):
    forward(m, x)               # compile #1: double=False

with spx.scope(mask=mask_b, double=False):    # different mask, same shape
    forward(m, x)               # cache hit on #1 — no recompile

with spx.scope(mask=mask_c, double=True):     # different static
    forward(m, x)               # compile #2: double=True
```

## Worked examples

### 1. Attention mask threaded through a transformer

```python
class Attention(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, rngs=rngs)
        self.out = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        scores = q @ k.transpose(0, 2, 1) / jnp.sqrt(q.shape[-1])
        mask = spx.scope.get("attn_mask", None)        # ← reads scope
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        attn = jax.nn.softmax(scores, axis=-1)
        return self.out(attn @ v)


class Block(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.attn = Attention(d, rngs=rngs)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4, rngs=rngs), nn.GELU(), nn.Linear(d * 4, d, rngs=rngs))

    def forward(self, x):
        return self.ffn(self.attn(x) + x) + x          # no mask plumbing needed


class Model(spx.Module):
    def __init__(self, n, d, *, rngs):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(d, rngs=rngs) for _ in range(n)])

    def forward(self, x):
        return self.blocks(x)


@spx.jit
def predict(m, x):
    return m(x)


# Without mask
y = predict(model, x)

# With mask — same compile (mask is a tracer)
causal = jnp.tri(seq_len, dtype=bool)[None]            # (1, S, S)
with spx.scope(attn_mask=causal):
    y_causal = predict(model, x)
```

The mask reaches every `Attention` instance without any block,
container, or model-level plumbing.

### 2. Dropout RNG via scope

Dropout normally takes an `Rngs` instance via `__init__`. Sometimes
you want per-call randomness:

```python
class StochasticDropout(spx.Module):
    """Dropout that pulls its key from the active scope."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        key = spx.scope.get("dropout_key")
        keep = jax.random.bernoulli(key, 1 - self.p, x.shape)
        return jnp.where(keep, x / (1 - self.p), 0.0)


key = jax.random.PRNGKey(42)
with spx.scope(dropout_key=key):
    y = model(x)
```

### 3. Mode flag for inference vs training-time differences

```python
class TopK(spx.Module):
    def forward(self, logits):
        if spx.scope.get("inference", False):          # static — re-compiles per mode
            return jax.lax.top_k(logits, k=8)[1]       # only indices
        return jax.nn.softmax(logits, axis=-1)         # full distribution


@spx.jit
def call(m, x):
    return m(x)


probs = call(model, x)                                 # one compile, full distribution

with spx.scope(inference=True):
    indices = call(model, x)                           # second compile, top-k indices
```

### 4. Decode-step counter for KV caching

```python
class CachedAttention(spx.Module):
    def __init__(self, d, max_seq, *, rngs):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d, rngs=rngs)
        self.out = nn.Linear(d, d, rngs=rngs)
        self.k_cache = spx.Buffer(jnp.zeros((1, max_seq, d)))
        self.v_cache = spx.Buffer(jnp.zeros((1, max_seq, d)))

    def forward(self, x):
        q, k_new, v_new = jnp.split(self.qkv(x), 3, axis=-1)
        t = spx.scope.get("decode_step")               # current decode position
        self.k_cache.value = self.k_cache.value.at[:, t].set(k_new[:, 0])
        self.v_cache.value = self.v_cache.value.at[:, t].set(v_new[:, 0])
        # Attend over [0:t+1]
        ...


@spx.jit(mutable=("cache", "buffers"))
def decode_step(m, token):
    return m(token)


for t in range(max_seq):
    with spx.scope(decode_step=jnp.asarray(t)):        # traced int — no per-step recompile
        token = decode_step(model, prev_token)
```

Note: `decode_step` is wrapped in `jnp.asarray` so it's a tracer
(lifted into jit args), not a static value (which would re-compile
every step).

### 5. Per-batch hyperparameters

Schedule a temperature, beta, or interpolation factor without plumbing:

```python
class Mixup(spx.Module):
    def forward(self, x, y):
        beta = spx.scope.get("mixup_beta", 0.0)
        if beta == 0.0:                                # static check at trace time
            return x, y
        ...                                            # actual mixup logic


for batch_i, (x, y) in enumerate(loader):
    beta = scheduler(batch_i)                          # Python float
    with spx.scope(mixup_beta=beta):
        loss, opt = step(model, opt, x, y)
```

If `beta` is a Python float that takes only a few discrete values, this
is cheap (one compile per value, cached). If it's continuous, demote
to a `jnp.asarray`:

```python
beta = jnp.asarray(scheduler(batch_i))                 # tracer — no recompile
with spx.scope(mixup_beta=beta):
    ...
```

### 6. Conditional intermediates capture

```python
class Encoder(spx.Module):
    def forward(self, x):
        h = self.layers(x)
        if spx.scope.get("capture_attn", False):
            self.sow("intermediates", "encoder_h", h)
        return h


@spx.jit(mutable="intermediates")
def run(m, x):
    return m(x)


# Normal path: no capture, fast
y = run(model, x)
print(spx.pop(model, "intermediates"))                 # {} — empty

# Debug path: re-compile to enable capture
with spx.scope(capture_attn=True):
    y = run(model, x)

print(spx.pop(model, "intermediates"))                 # {"encoder.sow_intermediates_encoder_h": ...}
```

### 7. Replacing global flags

```python
# Before — module-level globals (bad practice, hard to test):
GLOBAL_DETERMINISTIC = False

class Layer(spx.Module):
    def forward(self, x):
        if GLOBAL_DETERMINISTIC:
            return x
        ...


# After — scoped flag (composable, testable):
class Layer(spx.Module):
    def forward(self, x):
        if spx.scope.get("deterministic", False):
            return x
        ...


with spx.scope(deterministic=True):
    eval_outputs = run(model, x)
```

---

## Footguns

### 1. Misclassified values

A value with `.shape` but that you wanted to be static becomes a
tracer:

```python
shape_hint = jnp.array([4, 8])                # ❌ has .shape — gets lifted
with spx.scope(out_shape=shape_hint):
    @spx.jit
    def f(m, x):
        out = jnp.zeros(spx.scope.get("out_shape"))   # tracer used as shape -> error
        ...
```

Fix: use a Python tuple (static) or call `.tolist()` before scoping.

### 2. Re-compile storms from changing static values

If a per-step Python int is used as a static scope value, every step
triggers a new compile:

```python
for step_i in range(1000):
    with spx.scope(step=step_i):                       # ❌ step_i changes every iter
        train_step(...)                                # 1000 compiles!
```

Two fixes:

```python
# (a) demote to traced
for step_i in range(1000):
    with spx.scope(step=jnp.asarray(step_i)):         # ✓ tracer — one compile
        train_step(...)


# (b) drop it from scope; pass as a regular arg
@spx.jit
def train_step(m, o, step, x, y):
    ...
```

### 3. Threading

`contextvars` is asyncio-aware (per-`Task`) but **not inherited** by
threads spawned via `threading.Thread`. JAX dispatch is
single-threaded at the Python layer, so this doesn't bite in practice
— but if you wrap your training step in a `ThreadPoolExecutor`, scope
won't propagate automatically. Manually `contextvars.copy_context()`
if you need this.

### 4. No round-trip through `export` / `bind`

Scope values are **not** captured by `spx.export`. If your model
forward depends on a scope value, the caller of the bound module must
re-enter the scope.

### 5. `KeyError` is a real exception

Unlike `dict.get`, `spx.scope.get(key)` without `default=` raises
`KeyError` — that's intentional, to flag silent misconfigurations.
Always pass `default=` if the key may legitimately be absent.

---

## Hot-path cost

When **no scope is active**, `spx.jit` checks for it once via
`ContextVar.get()` (~50 ns) and falls through to its existing fast
path. Training loops that never touch scope see no measurable
overhead — verified on the dispatch-bound benchmark
(see [Performance](../performance.md)).

When **a scope is active**, the slow path adds:

- One pass over the stack to flatten it (O(total keys))
- A `partition()` call that splits into traced + static (O(total keys))
- One extra pytree-lifted dict arg into the jitted function
- A `__enter__` / `__exit__` pair at trace time (one-time per compile)

In practice, ~1 μs of extra dispatch per call for a typical
`mask + is_training` scope. Negligible compared to the compute.

---

## Comparison with related tools

| Need                             | Use                              |
| -------------------------------- | -------------------------------- |
| Persistent state between calls   | `spx.Variable` / `spx.Buffer`    |
| Per-call value read deep in tree | `spx.scope`                      |
| Randomness with PRNG state       | `spx.Rngs` / `spx.RngStream`     |
| Captured intermediates for debug | `self.sow("intermediates", ...)` |
| Dtype policy for a subtree       | `spx.Policy`                     |
| Select subsets of variables      | `spx.Selector`                   |
| Static config baked at init      | Module attribute (auto-static)   |

## API reference

- [`spectrax.core.context`](../api_docs/core/context.rst) — `scope`,
  `get`, `snapshot`, `partition`.
