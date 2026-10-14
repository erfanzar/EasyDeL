# Sharding (SPMD)

Single-program multiple-data (SPMD) parallelism in spectrax goes
through [`jax.sharding.Mesh`](https://jax.readthedocs.io/en/latest/jax.sharding.html)
with **logical axis names** annotated on each `Variable`. You never
write raw `jax.sharding.NamedSharding` objects unless you want to —
spectrax derives them from your annotations.

This guide covers:

- Annotating models with logical axes
- Building a mesh
- Materializing shardings
- End-to-end **data parallelism** (DP)
- End-to-end **tensor parallelism** (TP, Megatron-style)
- Mixed DP+TP
- Pitfalls

## Logical axis names

Annotate each weight axis with a **name**:

```python
import spectrax as spx

w = spx.Parameter(
    jnp.zeros((256, 256)),
    sharding=spx.Sharding(("data", "model")),     # 2 names — one per axis
)

w_replicated = spx.Parameter(
    jnp.zeros((256, 256)),
    sharding=spx.Sharding((None, None)),          # None = replicated on this axis
)
```

`Sharding` is a tuple of axis names (or `None` for replication). The
**name** is a logical handle; the **mesh** binds names to physical
devices. This indirection is what makes the same model annotated
once run under DP, TP, or DP+TP just by changing the mesh.

## Annotating built-in layers

Most `spectrax.nn` layers accept `weight_sharding=` / `bias_sharding=`
keyword args:

```python
from spectrax import nn, Sharding

fc = nn.Linear(
    256, 256,
    weight_sharding=Sharding(("data", "model")),
    bias_sharding=Sharding(("model",)),
    rngs=rngs,
)
```

For layers without an explicit sharding kwarg, set it on the
`Variable` directly:

```python
class MyLayer(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.weight = spx.Parameter(
            jnp.zeros((d, d)),
            sharding=Sharding(("data", "model")),
        )
```

## Building a mesh

### With spectrax's helpers (recommended)

SpecTrax wraps
[`jax.experimental.mesh_utils`](https://jax.readthedocs.io/en/latest/jax.experimental.mesh_utils.html)
with smart defaults and auto-detection of multi-slice TPU pods and
multi-process setups:

```python
import spectrax as spx

# Default: (1, 1, -1, 1, 1, 1) axis_dims + ("pp", "dp", "fsdp", "ep", "tp", "sp").
# The -1 fills with every remaining device, giving you FSDP out of the box.
mesh = spx.sharding.create_mesh()

# Custom shape and names:
mesh = spx.sharding.create_mesh(
    axis_dims=(2, 4),
    axis_names=("data", "model"),
)

# Pure data-parallel across every device:
mesh = spx.sharding.create_mesh(axis_dims=(-1,), axis_names=("data",))
```

`create_mesh` auto-selects the right mesh-creation path:

- **Single-process** -> plain `jax.experimental.mesh_utils.create_device_mesh`.
- **Multi-process (non-TPU)** -> derives a per-host submesh via
  `jax.experimental.mesh_utils.create_hybrid_device_mesh`.
- **Multi-slice TPU pods** — detected via `device.slice_index` or the
  `MEGASCALE_NUM_SLICES` env var — splits one mesh axis across
  slices and sets up the DCN (data-center-network) map.

Results are cached by argument tuple: repeated calls with the same
args return the same `Mesh` object.

### Parse a mesh from a string

For CLI / config-file driven setups:

```python
# Named form — explicit axis:dim map:
mesh = spx.sharding.parse_mesh_from_string("dp:2,tp:4", ("dp", "tp"))

# Positional form — dims mapped to names by position:
mesh = spx.sharding.parse_mesh_from_string("2,4", ("data", "model"))
```

### CPU-only contexts (debug / unit tests)

Simulate a multi-device mesh on a single CPU host:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

```python
# (a) CPU-backed mesh only:
mesh = spx.sharding.create_cpu_mesh()

# (b) Force default device to CPU inside a block:
with spx.sharding.force_cpu() as cpu:
    y = some_jax_op(x)

# (c) Combined — CPU mesh + forced-CPU execution. The common
#     "debug my SPMD code locally" idiom:
with spx.sharding.cpu_context() as mesh:
    # `mesh` is active AND JAX default device is CPU
    ...
```

### Raw JAX (still works)

```python
import jax
from jax.sharding import Mesh

devices = jax.devices()
mesh = Mesh(
    jax.numpy.array(devices).reshape(2, 4),
    axis_names=("data", "model"),
)
```

Mix freely — every spectrax API just takes a `jax.sharding.Mesh`,
whoever built it.

Now whenever you reference `"data"` or `"model"` in a `Sharding`
annotation, it resolves to the corresponding mesh axis.

For data parallel only:

```python
mesh = Mesh(jax.devices(), axis_names=("data",))
```

For 1D tensor parallel only:

```python
mesh = Mesh(jax.devices(), axis_names=("model",))
```

For 2D combined:

```python
mesh = Mesh(jax.numpy.array(jax.devices()).reshape(2, 4), axis_names=("data", "model"))
```

## Materializing shardings from annotations

`spectrax.sharding` walks the model and produces partition specs /
named-shardings for every annotated `Variable`:

```python
from spectrax.sharding import get_partition_spec, get_named_sharding

# Per-leaf PartitionSpec, derived from your sharding= annotations
specs = get_partition_spec(model)
# {"parameters": {"layer0.weight": PartitionSpec("data", "model"), ...},
#  "parameters": {"layer0.bias":   PartitionSpec("model",),         ...}}

# Wrap each in a NamedSharding bound to the mesh
shards = get_named_sharding(model, mesh)
```

Variables without a `sharding=` annotation are fully **replicated**
(matched against `PartitionSpec()`).

## Applying shardings

Three patterns, increasingly explicit:

### (a) Via `device_put` after init

Build the model on a single device, then redistribute:

```python
gdef, state = spx.export(model)
sharded_state = jax.tree.map(
    lambda leaf, shard: jax.device_put(leaf, shard),
    state,
    shards,
)
model = spx.bind(gdef, sharded_state)
```

### (b) Via `spx.jit` shardings kwargs

Constrain inputs and outputs at the jit boundary:

```python
@spx.jit(
    in_shardings=(shards, jax.sharding.PartitionSpec("data", None)),
    out_shardings=jax.sharding.PartitionSpec("data", None),
)
def step(m, x):
    return m(x)
```

### (c) Inside `forward` via `with_sharding_constraint`

Pin a tensor's sharding between ops (gives XLA a hint to insert
collective ops where needed):

```python
from spectrax.sharding import with_sharding_constraint_by_name

class Block(spx.Module):
    def forward(self, x):
        h = self.fc1(x)
        h = with_sharding_constraint_by_name(h, ("data", "model"))   # pin shard
        h = self.fc2(h)
        return h
```

---

## End-to-end: data parallelism

Replicate the full model on every device; shard the data batch.

```python
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P

import spectrax as spx
from spectrax import nn
from spectrax.contrib import Optimizer


N_DEVICES = len(jax.devices())
mesh = Mesh(jax.devices(), axis_names=("data",))


# Model — fully replicated (no sharding= annotations)
class MLP(spx.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.fc1 = nn.Linear(256, 1024, rngs=rngs)
        self.fc2 = nn.Linear(1024, 256, rngs=rngs)

    def forward(self, x):
        return self.fc2(jax.nn.relu(self.fc1(x)))


with mesh:
    model = MLP(rngs=spx.Rngs(0))
    opt = Optimizer.create(model, optax.adamw(3e-4))


    @spx.jit(
        mutable="parameters",
        in_shardings=(None, None, P("data"), P("data")),    # m, opt replicated; x, y sharded
        out_shardings=(None, None),
    )
    def step(m, o, x, y):
        def loss(m):
            return jnp.mean((m(x) - y) ** 2)
        loss_val, grads = spx.value_and_grad(loss)(m)
        new_opt = o.apply_eager(m, grads)
        return loss_val, new_opt


    for batch_i in range(100):
        x = jax.random.normal(jax.random.PRNGKey(batch_i), (32 * N_DEVICES, 256))
        y = jax.random.normal(jax.random.PRNGKey(batch_i + 999), (32 * N_DEVICES, 256))
        loss_val, opt = step(model, opt, x, y)
```

The model and optimizer state are replicated on every device. Each
device sees its slice of the batch via `P("data")`. Gradients
all-reduce across the mesh automatically (via jit's own sharding
propagation) before the optimizer step.

---

## End-to-end: tensor parallelism (Megatron-style)

Shard the model's weights across the `model` axis. Each device holds
a slice of the parameter; activation tensors are partitioned during
forward and concatenated/reduced as needed.

The Megatron MLP block:

- `fc1` weight is `(in, out)` -> split on `out` axis
- `fc2` weight is `(in, out)` -> split on `in` axis
- Output of `fc1` is partitioned across `model`
- Input to `fc2` matches -> matmul on each shard, all-reduce on output

```python
from spectrax import Sharding


class MlpTP(spx.Module):
    def __init__(self, d, h, *, rngs):
        super().__init__()
        # Column-parallel: split out
        self.fc1 = nn.Linear(
            d, h,
            weight_sharding=Sharding((None, "model")),    # (in, out)
            bias_sharding=Sharding(("model",)),
            rngs=rngs,
        )
        # Row-parallel: split in
        self.fc2 = nn.Linear(
            h, d,
            weight_sharding=Sharding(("model", None)),
            bias_sharding=Sharding((None,)),
            rngs=rngs,
        )

    def forward(self, x):
        h = jax.nn.gelu(self.fc1(x))                      # h is sharded on model axis
        return self.fc2(h)                                # all-reduce on output
```

To run:

```python
mesh = Mesh(jax.devices(), axis_names=("model",))

with mesh:
    model = MlpTP(d=256, h=1024, rngs=spx.Rngs(0))

    # Move model to mesh shardings
    shards = get_named_sharding(model, mesh)
    gdef, state = spx.export(model)
    state = jax.tree.map(jax.device_put, state, shards)
    model = spx.bind(gdef, state)


    @spx.jit(
        mutable="parameters",
        in_shardings=(shards, P()),                        # x replicated
        out_shardings=P(),
    )
    def step(m, o, x, y):
        ...
```

For attention, the standard pattern is column-parallel for Q/K/V
projections and row-parallel for the output projection:

```python
class AttentionTP(spx.Module):
    def __init__(self, d, n_heads, *, rngs):
        super().__init__()
        self.q = nn.Linear(d, d, weight_sharding=Sharding((None, "model")), rngs=rngs)
        self.k = nn.Linear(d, d, weight_sharding=Sharding((None, "model")), rngs=rngs)
        self.v = nn.Linear(d, d, weight_sharding=Sharding((None, "model")), rngs=rngs)
        self.o = nn.Linear(d, d, weight_sharding=Sharding(("model", None)), rngs=rngs)
        self.n_heads = n_heads

    def forward(self, x):
        # Each device holds a subset of heads
        q = self.q(x); k = self.k(x); v = self.v(x)
        ...
```

---

## End-to-end: combined DP + TP

A 2D mesh with both axes. Model is replicated on the `data` axis and
sharded on the `model` axis; data is sharded on `data`.

```python
mesh = Mesh(
    jax.numpy.array(jax.devices()).reshape(2, 4),         # 8 devices
    axis_names=("data", "model"),
)


class MlpDPTP(spx.Module):
    def __init__(self, d, h, *, rngs):
        super().__init__()
        self.fc1 = nn.Linear(
            d, h,
            weight_sharding=Sharding((None, "model")),     # TP on out
            rngs=rngs,
        )
        self.fc2 = nn.Linear(
            h, d,
            weight_sharding=Sharding(("model", None)),
            rngs=rngs,
        )


with mesh:
    model = MlpDPTP(d=256, h=1024, rngs=spx.Rngs(0))
    shards = get_named_sharding(model, mesh)
    gdef, state = spx.export(model)
    state = jax.tree.map(jax.device_put, state, shards)
    model = spx.bind(gdef, state)

    @spx.jit(
        mutable="parameters",
        in_shardings=(shards, P("data", None)),             # batch on data
    )
    def step(m, o, x, y):
        ...
```

Each (data, model) tile of devices holds a different batch slice +
different parameter slice. Forward and backward use all-reduces on
`model`; gradient all-reduce uses `data`.

---

## Pitfalls

### 1. Forgetting to bind names to a mesh

```python
sharding = Sharding(("data", "model"))                     # logical only
# ... no mesh bound ...
model = MyLayer(weight_sharding=sharding, ...)
y = model(x)                                                # ❌ unhelpful error
```

You must call inside a `with mesh:` block (or pass `mesh=` explicitly
to the materializing helpers) so the names resolve.

### 2. Mismatched shapes vs sharding

```python
# Weight is (in, out) but sharding has 3 entries
spx.Parameter(jnp.zeros((256, 256)),
              sharding=Sharding(("data", "model", "extra")))   # ❌
```

`Sharding` length must match the array rank, or be a single name for
1-D arrays.

### 3. Sharding RNG state

The `"rng"` collection is rarely sharded — usually replicated. If you
shard it accidentally, generated keys diverge across replicas in
unintended ways. Default to no annotation on `RngStream` cells.

### 4. Cross-mesh resharding

Moving a model from a 1D DP mesh to a 2D DP+TP mesh requires
re-deriving and re-applying the shardings:

```python
# Old 1D mesh
mesh1 = Mesh(jax.devices(), axis_names=("data",))
# ... train ...

# New 2D mesh
mesh2 = Mesh(jax.numpy.array(jax.devices()).reshape(2, 4), axis_names=("data", "model"))

# Re-derive
shards2 = get_named_sharding(model, mesh2)
gdef, state = spx.export(model)
state2 = jax.tree.map(jax.device_put, state, shards2)
model2 = spx.bind(gdef, state2)
```

### 5. Donating sharded buffers

`donate_argnums=` works with sharded inputs but you must ensure the
output sharding matches — otherwise XLA can't reuse the buffer in
place and silently allocates a copy:

```python
@spx.jit(
    in_shardings=(shards, P("data")),
    out_shardings=(shards, P("data")),                      # MUST MATCH for donation
    donate_argnums=(0, 1),
)
def step(m, x):
    ...
```

---

## Comparison with PyTorch FSDP / Megatron

| Feature                          | spectrax       | PyTorch FSDP        | Megatron-LM               |
| -------------------------------- | -------------- | ------------------- | ------------------------- |
| Annotation site                  | per-`Variable` | per-module via wrap | per-layer manual sharding |
| Logical/physical axis separation | yes (named)    | no                  | yes (manual)              |
| DP / TP / both                   | unified API    | DP only (FSDP)      | TP-only library           |
| Backend                          | XLA SPMD       | NCCL                | NCCL                      |
| Compile-time vs runtime          | fully traced   | runtime hooks       | hand-coded collectives    |

Because everything goes through XLA's SPMD compiler, spectrax's
sharding "just works" through every transform — `vmap`, `scan`,
`remat`, `cond` all see consistent sharded values.

## API reference

- [`spectrax.sharding`](../api_docs/sharding/index.rst) — top-level
  helpers (`get_partition_spec`, `get_named_sharding`,
  `with_sharding_constraint_by_name`).
- [`spectrax.core.sharding`](../api_docs/core/sharding.rst) —
  `Sharding`, `AxisNames`.
- JAX docs:
  [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).
