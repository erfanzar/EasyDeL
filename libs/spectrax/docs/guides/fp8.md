# FP8 training

FP8 is a family of 8-bit floating-point formats designed for training
deep networks at half the memory bandwidth (and on capable hardware,
~2x compute throughput) of FP16/BF16. SpecTrax supports
**delayed-scaling per-tensor FP8** with rolling amax history,
matching the algorithm used by
[NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/).

This guide explains how the algorithm works, how to use the drop-in
layers, how to roll your own FP8 op, and what to watch out for.

## When to use FP8

| Trade-off          | BF16            | FP8 (delayed scaling)         |
| ------------------ | --------------- | ----------------------------- |
| Memory             | 2 bytes / param | 1 byte / param                |
| Bandwidth          | 1x              | 2x                            |
| Compute throughput | 1x (on H100)    | 2x (on H100), 4x (Blackwell)  |
| Numerical accuracy | Excellent       | Very good with delayed scale  |
| Setup overhead     | None            | Per-tensor scale + amax cells |

Use FP8 when you're memory-bandwidth-bound on H100/Blackwell or TPU
v5+ and your model is large enough that the per-tensor scaling
overhead is amortized. For tiny models on smaller hardware, BF16 is
simpler and just as fast.

## The two formats

| Format          | Range             | Used for                      |
| --------------- | ----------------- | ----------------------------- |
| `float8_e4m3fn` | ~ ±448, no inf    | Forward activations + weights |
| `float8_e5m2`   | ~ ±57344, has inf | Backward gradients            |

E4M3 has more mantissa bits -> finer precision, smaller range. E5M2
has more exponent bits -> wider range, coarser precision. Gradients
need wider range; activations need precision.

## The algorithm

For each tensor (input, weight, gradient) the model carries an
**`Fp8Meta` cell** in the `"fp8_meta"` collection containing:

- `scale` — current per-tensor multiplicative scale to apply before
  quantization
- `amax_history` — ring buffer of recent absolute maxes
- `history_idx` — current write position in the ring

On every forward pass:

```md
1. quantize:       x_fp8 = round_to_fp8(x * scale)        # E4M3
2. compute:        y = matmul(x_fp8, w_fp8)               # XLA dispatches FP8 matmul
3. observe:        new_amax = jnp.abs(x).max()
4. update history: amax_history[history_idx] = new_amax
                   history_idx = (history_idx + 1) % len
5. compute next scale from rolling history (with margin)
```

The "delayed" part is in step 5: the scale used in step 1 is from
the **previous** step's history, not the current observation. This
keeps the algorithm numerically stable (no in-step recursion) and
lets XLA fuse the scaling with the matmul.

## Drop-in layers

```python
import spectrax as spx
from spectrax import nn, functional as F


class MlpFp8(spx.Module):
    def __init__(self, d_in, d_hidden, *, rngs):
        super().__init__()
        self.fc1 = nn.Fp8Linear(d_in, d_hidden, rngs=rngs)
        self.fc2 = nn.Fp8Linear(d_hidden, d_in, rngs=rngs)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

Each `Fp8Linear` owns three `Fp8Meta` cells (input, kernel, gradient).
You can mix FP8 layers with regular ones — the scaling is per-tensor.

Available FP8 layers:

| Layer              | Equivalent                 |
| ------------------ | -------------------------- |
| `nn.Fp8Linear`     | `nn.Linear`                |
| `nn.Fp8DotGeneral` | `jax.lax.dot_general`      |
| `nn.Fp8Einsum`     | `nn.Einsum` / `jnp.einsum` |

## The training step

Declare `"fp8_meta"` as **mutable** so scale/amax updates propagate
back to the live module:

```python
@spx.jit(mutable=("parameters", "fp8_meta"))
def step(m, o, x, y):
    def loss(m):
        return ((m(x) - y) ** 2).mean()

    loss_val, grads = spx.value_and_grad(loss)(m)
    new_opt = o.apply_eager(m, grads)
    return loss_val, new_opt
```

Without `"fp8_meta"` in `mutable=`, spectrax raises
`IllegalMutationError` — silent drop would be a correctness bug
because the scales need to update every step.

`Optimizer` does **not** allocate state for the `fp8_meta` cells by
default (it's `wrt="parameters"` only). The meta cells are updated
directly inside the FP8 ops via the mutation-detection mechanism.

## Custom FP8 layer

Roll your own with the primitives in `spectrax.nn.fp8`:

```python
from spectrax.nn.fp8 import (
    Fp8Meta,
    in_qdq,        # E4M3 fwd, straight-through bwd
    out_qdq,       # identity fwd, E5M2 qdq on cotangent
    update_fp8_meta,
)


class MyFp8MatMul(spx.Module):
    """Custom: y = quant(x, x_meta) @ quant(w, w_meta), with backward in E5M2."""

    def __init__(self, in_features, out_features, history_len=16, *, rngs):
        super().__init__()
        self.weight = spx.Parameter(
            spx.init.kaiming_uniform("linear")(rngs.parameters, (in_features, out_features), jnp.float32)
        )
        self.x_meta = Fp8Meta.create(history_len=history_len)
        self.w_meta = Fp8Meta.create(history_len=history_len)
        self.g_meta = Fp8Meta.create(history_len=history_len)

    def forward(self, x):
        x_q = in_qdq(x, self.x_meta.scale)              # quant + dequant on fwd
        w_q = in_qdq(self.weight.value, self.w_meta.scale)
        y = x_q @ w_q                                    # XLA lowers to FP8 matmul on H100+
        y = out_qdq(y, self.g_meta.scale)                # bwd-only quant on cotangent
        # Update rolling amax + next-step scale
        update_fp8_meta(self.x_meta, x)
        update_fp8_meta(self.w_meta, self.weight.value)
        update_fp8_meta(self.g_meta, y)
        return y
```

Don't forget to declare `mutable="fp8_meta"` on the transform that
calls this.

## Hardware support

FP8 ops compile to **real FP8 matmuls** only on hardware that
supports them:

- **NVIDIA H100, H200**: full E4M3/E5M2 support via Tensor Cores
- **NVIDIA Blackwell (B100/B200)**: same plus FP4
- **Google TPU v5+**: hardware FP8 path
- **Older / CPU**: ops run in **simulated FP8** — numerically
  equivalent via fake-quant, but no compute speedup

Simulated FP8 on a non-supporting device is useful for **development
and testing**: the numerics match (within rounding), so a model
debugged on a CPU laptop will train identically on H100.

## Saving and loading

FP8 meta cells round-trip through `spx.export` / `spx.bind` like any
other state:

```python
gdef, state = spx.export(model)
print(state["fp8_meta"].keys())
# dict_keys(['fc1.x_meta.scale', 'fc1.x_meta.amax_history', ...])

# Save scales separately if you want to load them as initialization for fine-tune
import pickle
with open("fp8_state.pkl", "wb") as f:
    pickle.dump({"parameters": state.filter("parameters"),
                 "fp8_meta": state.filter("fp8_meta")}, f)
```

Loading:

```python
with open("fp8_state.pkl", "rb") as f:
    saved = pickle.load(f)

# Build a fresh model
model = MlpFp8(d_in=512, d_hidden=2048, rngs=rngs)

# Patch parameters + scales
spx.update(model, saved["parameters"])
spx.update(model, saved["fp8_meta"])
```

## Comparing FP8 vs BF16 numerically

```python
import jax

x = jax.random.normal(jax.random.PRNGKey(0), (32, 512))
y = jax.random.normal(jax.random.PRNGKey(1), (32, 512))


# BF16 reference
class MlpBf16(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.fc1 = nn.Linear(d, d, dtype=jnp.bfloat16, rngs=rngs)
        self.fc2 = nn.Linear(d, d, dtype=jnp.bfloat16, rngs=rngs)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.astype(jnp.bfloat16))))


bf16 = MlpBf16(512, rngs=spx.Rngs(0))
fp8 = MlpFp8(512, 512, rngs=spx.Rngs(0))


# After warm-up (the FP8 model needs a few forward passes for scales to stabilize)
@spx.jit(mutable="fp8_meta")
def warmup_fp8(m, x):
    return m(x)


for _ in range(20):
    _ = warmup_fp8(fp8, x)


# Compare
y_bf16 = bf16(x)
y_fp8 = fp8(x)
print("rel error:", jnp.abs((y_fp8 - y_bf16) / y_bf16).max())
# Should be on the order of 1e-2 to 1e-3 for typical activations
```

## Pitfalls

### 1. Forgetting `mutable="fp8_meta"`

```python
@spx.jit                                   # mutable=() default
def step(m, x):
    return m(x)                            # IllegalMutationError on first call
```

Always include `"fp8_meta"` in your `mutable=` selector when calling
FP8 layers.

### 2. Cold-start scales

The first 1-2 forward passes use the initial scale (typically `1.0`
or determined by the first observation). Output may be noisy until
the rolling history fills. Warm up before measuring numerics.

### 3. Mixing dtypes

FP8 layers expect their input in higher precision (BF16 or FP32) so
they can run the quantize-then-matmul pipeline. If you cast manually
to `float8_e4m3fn` first, you'll skip the scaling step and likely get
poor numerics. Let the FP8 layers handle quantization.

### 4. Per-call vs per-batch scales

The default delayed-scaling is per-call. For very batched workloads
(e.g. enormous activation tensors), consider longer history lengths
or batched scale updates — the constructors accept `history_len=`.

### 5. Backward type mismatch

`out_qdq` quantizes the **cotangent** to E5M2 in backward. If you
write a custom FP8 op, make sure the gradient pathway uses E5M2 —
otherwise gradient saturation can wipe out small updates.

## API reference

- [`spectrax.nn.fp8`](../api_docs/nn/fp8.rst) — `Fp8Linear`,
  `Fp8DotGeneral`, `Fp8Einsum`, `Fp8Meta`, primitives (`quantize`,
  `dequantize`, `qdq`, `compute_scale`, `compute_amax_history`,
  `update_fp8_meta`, `in_qdq`, `out_qdq`).
