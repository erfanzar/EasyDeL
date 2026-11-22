# Pipeline parallelism

Train models whose per-device activation footprint exceeds a single
device's memory by **partitioning the model along the depth axis**:
each device owns a subset of layers (a "stage"), activations flow
forward through the pipeline, and cotangents flow backward. With
careful scheduling, stages overlap forward and backward work across
different microbatches so the devices stay busy instead of idling
during the pipeline bubble.

SpectraX exposes pipeline pieces from `spectrax.nn`,
`spectrax.runtime`, and `spectrax.runtime.spmd`:

- **[`PipelineSequential`](#pipelinesequential)** — the stage
  container (primary API).
- **Pipeline schedules** — [`GPipe`](#gpipe), [`Std1F1B`](#std1f1b),
  [`ZeroBubbleH1`](#zerobubbleh1), [`InterleavedH1`](#interleavedh1),
  plus the other schedules exported by `spectrax.runtime`.
- **[`pipeline_step`](#pipeline_step)** — one-call SPMD training
  step (same-shape stages, high-throughput).
- **[`sxcall`](#heterogeneous-stages-sxcall)** — MPMD runtime for
  **heterogeneous stages** (embed -> blocks -> head).
- **[`sxstage_iter` / `sxstage_region`](#inline-stage-markers)** —
  true MPMD stage markers for `sxjit`.

There are two explicit runtimes:

- `pipeline_step` is **SPMD-only**: one jaxpr is compiled across the
  whole pipeline axis through `shard_map`. It rejects MPMD-tagged
  meshes.
- `sxcall` / `sxjit` / `spx.run(..., mesh=<MPMD>)` are **true MPMD**:
  each physical pipeline rank gets its own scheduled program.

## When to use pipeline parallelism

Pipeline parallelism (PP) is complementary to data parallelism (DP)
and tensor parallelism (TP):

| Strategy    | Splits      | When to use                                      |
| ----------- | ----------- | ------------------------------------------------ |
| **DP**      | batch       | Small-medium models, plenty of memory per device |
| **TP**      | model width | Large MLP / attention blocks, fast interconnect  |
| **PP**      | model depth | Very deep models, cheaper slower interconnect    |
| **DP + PP** | both        | Production LLM training (Megatron-style 3D)      |

PP trades extra memory (per-stage activation saves for backward) for
reduced per-device parameter memory. Compared to TP, PP's
inter-device communication is *lower bandwidth but more frequent*
(one ppermute per microbatch per stage).

## `PipelineSequential`

Primary API for declaring stages. Build it like `nn.Sequential` but
each child is a stage:

```python
import spectrax as spx
from spectrax import nn
from spectrax.nn import PipelineSequential


class Block(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.fc = nn.Linear(d, d, rngs=rngs)

    def forward(self, x):
        return jax.nn.gelu(self.fc(x))


model = PipelineSequential(
    Block(256, rngs=spx.Rngs(0)),
    Block(256, rngs=spx.Rngs(1)),
    Block(256, rngs=spx.Rngs(2)),
    Block(256, rngs=spx.Rngs(3)),
)
```

### Same-shape stages (SPMD path)

For the **SPMD** backend ([`pipeline_step`](#pipeline_step)), every
stage in the container must share a structurally identical
`GraphDef` — same class, same static fields, same input and output
shapes. One compiled jaxpr runs on all pipeline devices, distinguished
only by which slice of parameters it holds. Transformer blocks,
ResNet blocks, MLP layers — anything where every layer has the same
signature — fit natively.

### Heterogeneous stages (MPMD path)

For models where stages have **different shapes, classes, or
parameter structures** — e.g. a transformer with a bulky embedding
stage, uniform middle blocks, and an lm-head — use the
[`sxcall`](#heterogeneous-stages-sxcall) runtime instead. It
compiles each stage's forward / backward separately and orchestrates
them through the true MPMD scheduler. It is fully flexible and keeps
the MPMD boundary explicit.

### Eager mode

In eager mode (no mesh, no schedule), `PipelineSequential` behaves
exactly like `nn.Sequential` — calls each stage in order on a single
device. This means you can develop your model without any pipeline
infrastructure and only turn it on when you need the parallelism.

```python
y = model(x)         # runs sequentially on one device; debug as normal
```

## Schedules

### GPipe

```python
from spectrax.runtime import GPipe

schedule = GPipe(microbatches=8)
```

All forwards first, then all backwards. Simplest to reason about and
debug. **Highest peak activation memory** — each stage holds every
microbatch's activation until the backward phase begins
(`peak = microbatches`).

Use for: prototyping, small models where activation memory isn't the
bottleneck, workloads where backward compute is much shorter than
forward (the all-forward phase stays busy).

### Std1F1B

```python
from spectrax.runtime import Std1F1B

schedule = Std1F1B(microbatches=8)
```

After a warmup that fills the pipeline, each stage alternates between
one forward and one backward. **Peak memory is bounded by `n_stages`,
independent of `microbatches`** — for M=64 and N=4, that's a 16x
memory reduction vs GPipe.

Use for: production training. This is what Megatron-LM uses by
default.

### ZeroBubbleH1

```python
from spectrax.runtime import ZeroBubbleH1

schedule = ZeroBubbleH1(microbatches=8)
```

The H1 variant of
[Qi et al. 2023's zero-bubble pipeline](https://arxiv.org/abs/2401.10241).
Splits each stage's backward into **input gradient** (`BWD_I`, on the
critical path) and **weight gradient** (`BWD_W`, off the critical
path). The W-grad fills what would otherwise be bubble slots in 1F1B,
driving the pipeline bubble toward zero.

Peak activation memory is the same as `Std1F1B`. The current runtime
performs the full VJP during the `BWD_I` step and uses `BWD_W` as a
scheduling no-op (correctness preserved; the scheduler achieves the
bubble reduction at the plan level, not via true split-backward
kernels). A future release will add true split VJP computation for
compute-level bubble elimination.

Use for: throughput-critical training where the pipeline bubble is
the main overhead.

### InterleavedH1

```python
from spectrax.runtime import InterleavedH1

schedule = InterleavedH1(microbatches=8, virtual_stages=2)
```

Each physical device owns **multiple non-contiguous virtual stages**
— for a 4-device pipeline with `virtual_stages=2`, each device owns
two of the 8 logical stages. Reduces the pipeline bubble by the
virtual-stage factor at the cost of extra `ppermute` hops per
microbatch.

Use for: high-bandwidth interconnect (NVLink, TPU v5+ ICI) where the
extra hops are cheap. Less beneficial on PCIe.

(pipeline_step)=

## `pipeline_step`

The one-call training API. Given a model, a batch, a mesh, and a
schedule, runs one full forward + backward + gradient-accumulation
step and returns the mean loss plus per-stage gradients:

```python
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from spectrax.runtime import Std1F1B
from spectrax.runtime.spmd import pipeline_step


def loss_fn(out, y):
    return ((out - y) ** 2).mean()


mesh = Mesh(jax.devices(), axis_names=("pp",))
schedule = Std1F1B(microbatches=8)

with mesh:
    loss, per_stage_grads = pipeline_step(
        model, (x, y),
        mesh=mesh, axis="pp",
        schedule=schedule, loss_fn=loss_fn,
    )

# Apply per-stage optimizer updates
for i, (stage, grad) in enumerate(zip(model.stages, per_stage_grads)):
    opts[i].update(grad)
```

### Signature

```python
def pipeline_step(
    model: PipelineSequential,
    batch: tuple[Any, ...],
    *,
    mesh: jax.sharding.Mesh | SpxMesh,
    axis: str = "pp",
    schedule: Schedule,
    loss_fn: Callable[..., jax.Array],
) -> tuple[jax.Array, tuple[State, ...]]: ...
```

- `model` — must be a `PipelineSequential` with
  `num_stages == mesh.shape[axis]`.
- `batch` — tuple of positional inputs. The **first** element is the
  pipeline input (flows through stages); the rest are targets / aux
  args passed to `loss_fn` at the final stage. Every element's
  leading axis is microbatched by `schedule.microbatches`.
- `loss_fn` — scalar loss: `loss_fn(final_stage_output, *batch[1:])`.
- `mesh` — raw JAX `Mesh` or a pure-SPMD `SpxMesh`. `MpMdMesh` and
  `SpxMesh(..., mpmd_axis=...)` are rejected so MPMD-marked calls
  cannot accidentally run through the SPMD runtime.

### Return value

`(loss, per_stage_grads)` where:

- `loss` — scalar mean loss across microbatches, broadcast identical
  on every pipeline device (useful for logging from any rank).
- `per_stage_grads` — tuple of `State` objects, one per stage, each
  containing the `"parameters"` collection gradient. Pass these to your
  per-stage optimizer.

### Microbatching constraint

The batch size must be divisible by `schedule.microbatches`. Pick
`microbatches` so that:

- `B / microbatches` is a reasonable per-stage mini-batch size
  (typically ≥ 8 to keep matmul kernels efficient).
- `microbatches >= n_stages` so 1F1B / ZB schedules can reach steady
  state.

Common choices: `microbatches = 2 x n_stages` or `4 x n_stages`.

## Full example: transformer pretraining

```python
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh

import spectrax as spx
from spectrax import nn, functional as F
from spectrax.nn import PipelineSequential
from spectrax.runtime import Std1F1B
from spectrax.runtime.spmd import pipeline_step


# --- Stages (identical shape) ---
class TransformerBlock(spx.Module):
    def __init__(self, d, n_heads, ffn, *, rngs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, rngs=rngs)
        self.attn = nn.MultiheadAttention(d, n_heads, rngs=rngs)
        self.ln2 = nn.LayerNorm(d, rngs=rngs)
        self.fc1 = nn.Linear(d, ffn, rngs=rngs)
        self.fc2 = nn.Linear(ffn, d, rngs=rngs)

    def forward(self, x):
        h = self.ln1(x)
        h = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


# --- Model ---
N_STAGES = 4
D = 512
rngs = spx.Rngs(0)
model = PipelineSequential(
    *[TransformerBlock(D, 8, 4 * D, rngs=rngs) for _ in range(N_STAGES)]
)


# --- Per-stage optimizer (one Optimizer per stage) ---
def build_opt():
    # Build a dummy stage to size the optimizer state.
    stage = TransformerBlock(D, 8, 4 * D, rngs=spx.Rngs(0))
    return spx.contrib.Optimizer.create(stage, optax.adamw(3e-4))


opts = [build_opt() for _ in range(N_STAGES)]


# --- Loss ---
def loss_fn(out, target):
    return ((out - target) ** 2).mean()


# --- Mesh ---
mesh = Mesh(jax.devices()[:N_STAGES], axis_names=("pp",))


# --- Training step ---
def step(model, opts, x, y):
    loss, per_stage_grads = pipeline_step(
        model, (x, y),
        mesh=mesh, axis="pp",
        schedule=Std1F1B(microbatches=8),
        loss_fn=loss_fn,
    )
    # Apply per-stage updates
    new_opts = []
    for i, (stage, grad) in enumerate(zip(model.stages, per_stage_grads)):
        new_opt = opts[i].apply_eager(stage, grad)
        new_opts.append(new_opt)
    return loss, new_opts


# --- Loop ---
with mesh:
    for step_i in range(100):
        x = jax.random.normal(jax.random.PRNGKey(step_i), (32, 64, D))
        y = jax.random.normal(jax.random.PRNGKey(step_i + 999), (32, 64, D))
        loss, opts = step(model, opts, x, y)
        if step_i % 10 == 0:
            print(f"step {step_i}: loss = {float(loss):.4f}")
```

## Heterogeneous stages: `sxcall`

When the SPMD same-shape constraint doesn't fit your model,
`sxcall` provides the true MPMD path — **each stage can have a
different class, different parameter structure, and different input /
output shapes**.

```python
from jax.sharding import Mesh

from spectrax.runtime import GPipe, sxcall
from spectrax.runtime.types import MpMdMesh
from spectrax.nn import PipelineSequential


# Heterogeneous: embed + blocks + head
class EmbedStage(spx.Module):
    def __init__(self, vocab, d, *, rngs):
        super().__init__()
        self.embed = nn.Embed(vocab, d, rngs=rngs)

    def forward(self, ids):
        return self.embed(ids)


class BlockStage(spx.Module):
    def __init__(self, d, *, rngs):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 8, rngs=rngs)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d, rngs=rngs),
            nn.GELU(),
            nn.Linear(4 * d, d, rngs=rngs),
        )

    def forward(self, x):
        return self.ffn(self.attn(x, x, x) + x) + x


class HeadStage(spx.Module):
    def __init__(self, d, vocab, *, rngs):
        super().__init__()
        self.head = nn.Linear(d, vocab, rngs=rngs)

    def forward(self, x):
        return self.head(x)


rngs = spx.Rngs(0)
model = PipelineSequential(
    EmbedStage(50_000, 512, rngs=rngs),     # (B, S) int -> (B, S, 512)
    BlockStage(512, rngs=rngs),              # (B, S, 512) -> (B, S, 512)
    BlockStage(512, rngs=rngs),              # (B, S, 512) -> (B, S, 512)
    HeadStage(512, 50_000, rngs=rngs),      # (B, S, 512) -> (B, S, 50_000)
)


def loss_fn(logits, targets):
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


devices = jax.devices()[:4]
mpmd_mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
loss, per_stage_grads = sxcall(
    model,
    (ids, targets),
    mesh=mpmd_mesh,
    mode="train",
    schedule=GPipe(microbatches=8),
    loss_fn=loss_fn,
)
```

### When to use MPMD vs SPMD

| Dimension            | SPMD (`pipeline_step`)             | MPMD (`sxcall`)              |
| -------------------- | ---------------------------------- | ---------------------------- |
| Stage constraint     | Same-`GraphDef`                    | None                         |
| Compilation          | One fused jaxpr                    | One true scheduled MPMD plan |
| Dispatch overhead    | Single `shard_map` call            | Scheduled per-rank dispatch  |
| Max throughput       | Best on homogeneous same-shape     | Best for heterogeneous stages |
| Heterogeneous layers | Wrap in uniform `Block`            | Native                       |
| Debug ease           | Harder (one big compile)           | Easier (per-stage jit)       |
| Schedule support     | GPipe / 1F1B / ZB-H1 / Interleaved | Same                         |

Rule of thumb: **SPMD for transformer / ResNet stacks; MPMD
everywhere else**.

## Combining with DP / TP

Pipeline parallelism stacks with data parallelism and tensor
parallelism along extra mesh axes:

```python
# 2D mesh: data-parallel x pipeline-parallel
mesh = Mesh(
    jax.numpy.array(jax.devices()).reshape(2, 4),     # 8 devices
    axis_names=("dp", "pp"),
)

# The pipeline uses "pp"; DP axis handles batch splits automatically
# (batch leading dim becomes sharded across "dp" via the caller's
# data pipeline).
with mesh:
    loss, grads = pipeline_step(
        model, (x, y),
        mesh=mesh, axis="pp",
        schedule=schedule, loss_fn=loss_fn,
    )
```

For 3D (DP + TP + PP), add a TP axis and annotate the inside of each
stage's weights — see the [sharding guide](sharding.md) for TP
conventions. The pipeline layer is orthogonal.

## Inline stage markers

For marker-based true MPMD, use `sxstage_iter` inside an `sxjit`
function. It is an identity at runtime but becomes a stage boundary
for the MPMD compiler:

```python
from spectrax.runtime import sxstage_iter, sxstage_region


class Net(spx.Module):
    def forward(self, x):
        x = self.embed(x)
        x = sxstage_iter(x)    # split — stage 0 / stage 1
        x = self.blocks[0](x)
        x = sxstage_iter(x)    # split — stage 1 / stage 2
        x = self.blocks[1](x)
        return self.head(x)
```

For multimodal or branched pipelines, `sxstage_region` creates
independent logical stage sequences before the scheduler maps them to
physical ranks.

```python
vision = spx.sxstage_region("vision", schedule=spx.GPipe(microbatches=2))
text = spx.sxstage_region("text", schedule=spx.GPipe(microbatches=2))


def multimodal_loss(image_features, token_features):
    def vision_path(x):
        x = vision_block_0(x)
        x = spx.sxstage_iter(x, stage=0)   # V0 -> V1
        return vision_block_1(x)

    def text_path(x):
        x = text_block_0(x)
        x = spx.sxstage_iter(x, stage=0)   # T0 -> T1
        return text_block_1(x)

    image = vision(vision_path)(image_features)
    text_hidden = text(text_path)(token_features)
    return contrastive_loss(image, text_hidden)
```

The parent scheduler sees this as two serial two-stage paths:

```text
logical stage 0: V0
logical stage 1: V1
logical stage 2: T0
logical stage 3: T1
```

Region-local `sxstage_iter` markers do not leak into the parent
pipeline. This matters for multimodal models: the vision tower and
text tower can both use local stage numbers (`0`, `1`, ...) without
accidentally forming one mixed V/T stage sequence. See
[`examples/07_mpmd/12_stage_region_multimodal.py`](../../examples/07_mpmd/12_stage_region_multimodal.py)
for a runnable example.

### Stage-local mesh rule

In true MPMD, the pipeline axis selects the program/rank. It is not
available as an intra-stage sharding axis. A full mesh like
`(pp=2, dp=4, tp=2)` becomes a stage-local mesh `(dp=4, tp=2)` inside
each stage executable.

```python
from jax.sharding import Mesh, PartitionSpec
from spectrax.runtime.types import MpMdMesh

full = Mesh(devices.reshape(2, 4, 2), ("pp", "dp", "tp"))
mpmd = MpMdMesh(full, "pp")

stage0 = mpmd.submesh(0)
assert stage0.axis_names == ("dp", "tp")

mpmd.sub_sharding(0, PartitionSpec("tp"))  # ok
mpmd.sub_sharding(0, PartitionSpec("pp"))  # ValueError
```

This is why MPMD boundary specs should describe only stage-local SPMD
axes. See
[`examples/07_mpmd/13_stage_local_mesh_layout.py`](../../examples/07_mpmd/13_stage_local_mesh_layout.py)
for a concrete inspection script.

## Debugging tips

### 1. Run eager first

```python
# Develop / debug without any pipeline infrastructure
y = model(x)                       # single-device sequential execution
```

Only add `pipeline_step` once the model trains correctly locally.

### 2. Check schedule correctness

Schedules expose diagnostic properties:

```python
s = Std1F1B(microbatches=8)
print(s.total_steps(n_stages=4))    # total number of time steps
print(s.peak_activations(n_stages=4))  # per-stage max live activations
print(s.bubble_ratio(n_stages=4))   # fraction of idle slots
```

### 3. Verify bit-exactness vs single-device

For correctness regression tests, assert that pipeline loss and
per-stage gradients match a single-device run on the same inputs.
spectrax's own test suite does exactly this for GPipe — see
[tests/pipeline/test_e2e.py](../../tests/pipeline/test_e2e.py).

### 4. Simulate multiple devices locally

On CPU:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
```

(set **before** any JAX import). Now `jax.devices()` returns 4 CPU
devices, enough for a 4-stage pipeline simulation.

## Pitfalls

### 1. Stages with different shapes under SPMD

```python
PipelineSequential(
    nn.Embed(vocab, d, rngs=rngs),       # different class + shape
    TransformerBlock(d, rngs=rngs),
    TransformerBlock(d, rngs=rngs),
    nn.Linear(d, vocab, rngs=rngs),      # different output shape
)

pipeline_step(model, ..., schedule=...)  # ❌ ValueError
```

Switch to MPMD:

```python
sxcall(model, ..., mesh=mpmd_mesh, schedule=...)  # ✓
```

Or keep SPMD by wrapping heterogeneous layers in a uniform `Block`.

### 2. Mismatched mesh

```python
PipelineSequential(stage_a, stage_b, stage_c)     # 3 stages
mesh = Mesh(jax.devices()[:4], axis_names=("pp",))  # 4 devices
pipeline_step(model, ..., mesh=mesh, axis="pp")    # ❌ ValueError
```

`num_stages` must equal `mesh.shape[axis]`.

### 3. `microbatches` too small for 1F1B

If `microbatches < n_stages`, 1F1B can't fill the pipeline and
silently falls back to a GPipe-like schedule (all forwards then all
backwards). Check your config:

```python
assert microbatches >= n_stages
```

### 4. Forgetting to enter the mesh context

```python
pipeline_step(..., mesh=mesh, ...)  # works, but safer inside `with mesh:`
```

`pipeline_step` invokes `shard_map`, which binds to the supplied mesh
explicitly, so an enclosing `with mesh:` isn't strictly required —
but other spectrax operations in the same scope (e.g., reading
sharded weights) do need it.

## API reference

- [`spectrax.nn.PipelineSequential`](../api_docs/nn/pipeline_sequential.rst)
  — stage container.
- [`spectrax.runtime.schedules`](../api_docs/runtime/schedules/index.rst)
  — schedules, actions, phases, and fusion helpers.
- [`spectrax.runtime.mpmd`](../api_docs/runtime/mpmd/index.rst)
  — true MPMD `sxcall`, `sxjit`, `sxgrad`, `sxvalue_and_grad`, markers.
- [`spectrax.runtime.spmd`](../api_docs/runtime/spmd/index.rst)
  — SPMD-only `pipeline_step`, `spmd_run`, and `make_scheduled_body`.

## References

- Huang et al. 2019,
  [*GPipe: Efficient Training of Giant Neural Networks using Pipeline
  Parallelism*](https://arxiv.org/abs/1811.06965).
- Narayanan et al. 2021,
  [*Efficient Large-Scale Language Model Training on GPU Clusters
  Using Megatron-LM*](https://arxiv.org/abs/2104.04473) — Std1F1B +
  Interleaved1F1B.
- Qi et al. 2023,
  [*Zero Bubble Pipeline Parallelism*](https://arxiv.org/abs/2401.10241)
  — ZB-H1 / ZB-H2.
- PyTorch Pipelining —
  [`torch.distributed.pipelining`](https://pytorch.org/docs/stable/distributed.pipelining.html).
