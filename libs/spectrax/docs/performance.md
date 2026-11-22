# Performance

spectrax is designed to add as little Python dispatch overhead as
possible on top of JAX's `jit`. This page walks through the hot
dispatch path, the concrete optimizations applied, and the
benchmarks.

## The hot path

A typical `spx.jit`-wrapped training step looks like:

```python
@spx.jit(mutable="parameters")
def step(m, o, x, y):
    def loss(m):
        return ((m(x) - y) ** 2).mean()
    loss_val, grads = spx.value_and_grad(loss)(m)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt
```

Each call to `step(model, opt, x, y)` goes through **`wrapped`** in
[`spectrax/transforms/jit.py`](https://github.com/erfanzar/spectrax/blob/main/spectrax/transforms/jit.py):

1. `locate_and_strip` — single pass over `args` / `kwargs`,
   detects `Module` instances, builds `_ModuleRef` records (with
   cached `export()` data), produces a "stripped" args tuple with
   `None` placeholders.
2. Identity-cache lookup keyed on `id(module)` tuple + graph epoch.
   Hot path: O(1) dict lookup, returns the cached jitted callable.
3. Call the jitted function with `(states_in, stripped_args,
   stripped_kwargs)`. JAX runs its own pytree flatten / compile-cache
   hit / kernel launch.
4. `apply_mutations` — writes detected leaf changes back to the live
   `Variable` cells via the module's cached `vars_by_collection` dict.

## Optimization summary

Each of these was a separate experiment; the combined effect is
**1.83x** faster dispatch than `flax.nnx` on the tiny-CPU benchmark.

| Optimization                                         | Mechanism                                                              |
| ---------------------------------------------------- | ---------------------------------------------------------------------- |
| **Per-module `export()` cache**                      | Keyed on global graph epoch; invalidated by `Module.__setattr__` only. |
| **Fused `locate_and_strip`**                         | Single arg pass, builds `_ModuleRef` + exports in one go.              |
| **`_ModuleRef` `__slots__`**                         | ~5x faster alloc vs. plain dataclass.                                  |
| **Optimizer `__slots__`**                            | Same, for the optax wrapper.                                           |
| **`GraphDef.__hash__` memoization**                  | Skips recursive tuple walk on repeat dispatch.                         |
| **Two-level jit cache**                              | Identity cache (module id tuple) + structural cache (GraphDef tuple).  |
| **Pre-built `vars_by_collection` dict**              | `apply_mutations` looks up variables by `(kind, path)` in O(1).        |
| **Kinds-only fast path in `apply_mutations`**        | `mutable="parameters"` skips the full selector walk.                       |
| **Single-module fast path in `wrapped`**             | Skips list construction for the common case.                           |
| **`locate_and_strip_fast`**                          | Kwargs-less variant, skips the kwargs iteration.                       |
| **Fast `State` flatten**                             | Path-agnostic flatten, skips per-leaf `GetAttrKey` allocation.         |
| **Single-collection `State` flatten specialization** | `{"parameters": {...}}` skips the outer `sorted()` call.                   |
| **Inline `Variable._value` access**                  | Skips `_raw_get` / `_raw_set` method dispatch.                         |
| **Hoisted imports in `wrapped` closure**             | Local-var dereference instead of module-attribute walk.                |
| **`skip make_pure` on compile-cache hit**            | Avoids rebuilding the closure when the jitted function is cached.      |
| **Scope-aware slow path**                            | No-scope case unaffected — single `ContextVar.get()` (~50 ns) check.   |

Each of these individually moves the needle by a fraction of a percent
to a few percent; cumulatively, they add up to ~15% shaved off the
ratio vs. upstream.

## Benchmarks

Reproducible via the included harness:

```bash
python -m benchmarks.train_llm --device cpu \
    --n-layers 2 --d-model 64 --n-heads 2 --ffn 128 \
    --batch 2 --seq-len 32 --epochs 3 --iters 200
```

### Tiny CPU (dispatch-bound)

| Config                     | spx median | nnx median | Speedup   |
| -------------------------- | ---------- | ---------- | --------- |
| 1L, d=32, batch=1, seq=16  | 0.25 ms    | 1.84 ms    | **7.31x** |
| 2L, d=32, batch=2, seq=32  | 1.08 ms    | 3.26 ms    | **3.03x** |
| 2L, d=48, batch=2, seq=32  | 1.62 ms    | 3.27 ms    | **2.01x** |
| 2L, d=64, batch=2, seq=32  | 2.01 ms    | 3.63 ms    | **1.83x** |
| 2L, d=80, batch=2, seq=32  | 2.48 ms    | 4.25 ms    | **1.72x** |
| 4L, d=128, batch=4, seq=64 | 24.7 ms    | 24.1 ms    | 0.97x     |

Below d≈48, Python dispatch dominates and spectrax wins handily. Above
d≈80, compute dominates and the ratio narrows; at d=128 the two are
basically indistinguishable — the XLA kernels are the same in both
cases.

### TPU — 1.21B transformer

| Metric       | spectrax | flax.nnx | Speedup   |
| ------------ | -------- | -------- | --------- |
| Train step   | 25.3 ms  | 34.5 ms  | **1.36x** |
| Forward only | 14.8 ms  | 25.1 ms  | **1.70x** |

Compute is a bigger share of total time on TPU, but the lower Python
dispatch overhead still shows up end-to-end.

### TPU — 8B transformer

| Metric                 | spectrax | flax.nnx | Speedup   |
| ---------------------- | -------- | -------- | --------- |
| Train step (median)    | 96.5 ms  | 163.3 ms | **1.69x** |
| Tail latency (p95-p05) | 0.4 %    | 66 %     | —         |

SpecTrax also has much tighter tail latency — the dispatch path
spends much less time in unpredictable Python operations.

## When spectrax is the same speed or slower

**Compute-bound workloads on CPU with large d.** The ratio shrinks to
≈1.0x because most time is actual XLA compute, and both libraries
compile to the same HLO. If you see spectrax losing by a few percent
on a compute-bound workload, that's XLA variance, not something we can
fix on the Python side.

**First-step compile time** is not faster. Trace-time is dominated by
JAX, and we don't make the compiler work less.

## Measuring your own dispatch overhead

Profile a single step with `cProfile`:

```python
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
for _ in range(200):
    out = step(model, opt, x, y)
    jax.block_until_ready(out[0])
pr.disable()

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
print(s.getvalue())
```

The key rows to watch:

| Line                | What it is                               |
| ------------------- | ---------------------------------------- |
| `try_to_block`      | Waiting on XLA compute (not our problem) |
| `wrapped` (tottime) | spectrax + jax.jit dispatch Python       |
| `apply_mutations`   | Writing leaves back to live vars         |
| `_state_flatten`    | State -> pytree flatten                  |
| `export`            | Our cached `export()` hot path           |

If `wrapped` own-time is high in your workload, check if you have a
large Module with many parameters — `apply_mutations` scales with the
number of leaves in the `mutable=` collection.

## Related

- [Design](design.md) — why spectrax is shaped the way it is.
- [Transforms guide](guides/transforms.md) — how transforms work
  internally.
