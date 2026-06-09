# eformer 🔮 (EasyDel Former)

[![PyPI version](https://img.shields.io/pypi/v/eformer?logo=pypi&color=3776ab)](https://pypi.org/project/eformer/)
[![Python](https://img.shields.io/badge/python-3.11--3.13-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.8%2B-informational)](https://github.com/google/jax)
[![Docs](https://img.shields.io/badge/docs-readthedocs-success)](https://eformer.readthedocs.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](CHANGES.txt)

> EasyDel Former is a batteries-included JAX toolkit for building, quantizing, scaling, and deploying modern transformer-style workloads on GPUs and TPUs.

## Table of Contents

- [Why eformer?](#why-eformer)
- [Feature Highlights](#feature-highlights)
- [Module Map](#module-map)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Examples & Guides](#examples--guides)
- [Documentation](#documentation)
- [Testing & Quality](#testing--quality)
- [Contributing](#contributing)
- [License](#license)

## Why eformer?

eformer packages the infrastructure the EasyDel project uses to run large JAX models in production:

- **Single import for system glue** – argument parsing, logging, filesystem helpers, PyTree utilities, sharding, and TensorStore checkpoints live in one coherent namespace.
- **Hardware-aware building blocks** – Ray + TPU/GPU executors, mesh utilities, quantized kernels, and loss scaling are battle-tested in multi-slice pods.
- **Productivity without boilerplate** – dataclass-driven CLIs, optimizer factories, progress loggers, and serialization APIs keep research prototypes tidy.
- **Deep integration with JAX** – everything is PyTree-friendly, `jax.jit`/`vmap` compatible, and aware of sharding semantics so you can stay inside pure JAX programs.

## Feature Highlights

### Precision & Quantization

- `eformer.mpric` exposes `Policy`, `PrecisionHandler`, and dynamic `LossScaler` utilities so you can express policies like `p=f32,c=f8_e4m3,o=f32` and automatically wrap training/inference steps with casting and loss-scaling logic.
- Unified quantization interface (`QuantizationConfig`, `QuantizationType`, `quantize`, `straight_through`) supports NF4, INT8, binary, and ternary formats with actual bit packing, TPU-optimized NF4 kernels via Pallas, and STE support for QAT.
- `eformer.jaximus` supplies the implicit-array runtime (`ImplicitArray`, `register`, `ste`, `implicit` decorator) that lets Array8B, ArrayNF4, and 1-bit tensors participate in JAX primitives without materializing unless needed.

### Distributed Scaling & Executors

- `eformer.escale` provides semantic sharding via `PartitionAxis`, `PartitionManager`, `auto_namedsharding`, and helpers to convert per-layer rules into `PartitionSpec`s that respect DP/FSDP/TP/EP/SP axes.
- Mesh tooling (`create_mesh`, `MeshPartitionHelper`) inspects pytree shapes and suggests sharding plans, while constraint utilities (`with_sharding_constraint`, `get_corrected_named_sharding`) fix up specs for real device meshes.
- `eformer.executor` builds on Ray to launch pods or multi-slice TPU jobs with automatic retries (`RayExecutor.execute_resumable`, `autoscale_execute_resumable`), Docker orchestration, and SLURM-friendly cluster discovery (`eSlurmCluster`, `auto_ray_cluster`).

### PyTree, Serialization & Storage

- `eformer.pytree` ships >50 helpers for diffing, stacking, filtering, flattening, and serializing PyTrees plus MsgPack-based `to_bytes`/`from_bytes` and type registration hooks.
- High-level checkpointing (`serialization.Checkpointer`, `AsyncCheckpointManager`, `TensorStore` backends) supports time/step policies, async cleanup, and sharded array saves without all-gathers.
- `eformer.paths.ePath` abstracts local paths and Google Cloud Storage with identical APIs, including JAX array saves/loads and recursive globbing.

### Optimizers & Training Ergonomics

- `OptimizerFactory` + `_builders` turn concise config dataclasses (AdamW, Adafactor, Muon, Lion, RMSProp, WhiteKron, Mars, Soap, Kron) into Optax transforms with scheduler composition.
- `SchedulerFactory` generates cosine/linear/warmup schedules or plugs in custom callables for experiments.
- `aparser.Argu` + `DataClassArgumentParser` transform dataclasses into CLIs with YAML/JSON loading, alias handling, and bool toggles.
- `loggings.get_logger` offers colorized, process-aware loggers and progress tracking, while `common_types` centralizes semantic axis constants (BATCH, VOCAB, DP, TP, etc.) to keep sharding specs consistent.

## Module Map

| Module | Purpose | Key entry points |
| --- | --- | --- |
| `eformer.aparser` | Dataclass-first argument parsing & config loading | `Argu`, `DataClassArgumentParser.parse_args_into_dataclasses`, `parse_yaml_file` |
| `eformer.escale` | Mesh + sharding orchestration across DP/FSDP/TP/EP/SP | `PartitionAxis`, `PartitionManager`, `auto_partition_spec`, `MeshPartitionHelper` |
| `eformer.executor` | Ray-powered TPU/GPU executors, Docker helpers, SLURM glue | `RayExecutor`, `autoscale_execute_resumable`, `auto_ray_cluster`, `TpuAcceleratorConfig` |
| `eformer.jaximus` | Implicit arrays and custom PyTree runtime for quantized tensors | `ImplicitArray`, `register`, `implicit`, `ste` |
| `eformer.mpric` | Mixed precision policies, dtype registries, dynamic loss scaling | `Policy`, `PrecisionHandler`, `LossScaleConfig`, `DynamicLossScale` |
| `eformer.ops.quantization` | NF4/INT8/1-bit quantization kernels and STE wrappers | `QuantizationConfig`, `QuantizationType`, `ArrayNF4`, `Array8B`, `quantize`, `straight_through` |
| `eformer.optimizers` | Configurable optimizer factory & scheduler utilities | `OptimizerFactory`, `SchedulerFactory`, `optax_add_scheduled_weight_decay` |
| `eformer.pytree` | Extensive PyTree manipulation and MsgPack serialization | `tree_*` helpers, `PyTree`, `to_bytes`, `save_to_file` |
| `eformer.serialization` | TensorStore checkpointing and async save managers | `Checkpointer`, `CheckpointInterval`, `AsyncCheckpointManager`, `fsspec_utils` |
| `eformer.paths` | Unified local/GCS path abstraction with ML utilities | `ePath`, `LocalPath`, `GCSPath`, `save_jax_array`, `load_jax_array` |
| `eformer.loggings` | Color logs, once-only warnings, progress meters | `get_logger`, `LazyLogger`, `ProgressLogger` |
| `eformer.common_types` | Shared axis constants & sharding-friendly aliases | `BATCH`, `EMBED`, `DP`, `TP`, `PartitionAxis`, `DynamicShardingAxes` |

## Installation

eformer targets Python 3.11–3.13 with `jax>=0.8.0`. Install the TPU/GPU-specific JAX build that matches your platform before using hardware accelerators.

### PyPI release

```bash
pip install eformer
```

### From source (development)

```bash
git clone https://github.com/erfanzar/eformer.git
cd eformer
pip install -e '.[dev]'

# optional: keep dependencies in sync with uv
uv sync --dev
```

For documentation builds:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

## Quickstart

### 1. Dataclass-driven configuration

```python
from dataclasses import dataclass
from eformer.aparser import Argu, DataClassArgumentParser

@dataclass
class RuntimeConfig:
    steps: int = Argu(help="Number of training steps", default=10_000)
    mesh: str = Argu(help="Mesh spec such as 'dp:2,tp:4'", default="dp:1,tp:1")
    policy: str = Argu(help="Precision policy string", default="p=f32,c=f8_e4m3,o=f32")

parser = DataClassArgumentParser(RuntimeConfig, description="Train a transformer with eformer.")
config, = parser.parse_args_into_dataclasses()

# Load overrides from a YAML file if desired
config, = parser.parse_yaml_file("configs/train.yaml")
print(config)
```

`Argu` stores CLI metadata (aliases/help/defaults), and the parser can read dictionaries/JSON/YAML while validating against your dataclass schema.

### 2. Mixed-precision training with `mpric`

```python
import jax
import jax.numpy as jnp
from eformer.mpric import PrecisionHandler

handler = PrecisionHandler(policy="p=f32,c=f8_e4m3,o=f32", use_dynamic_scale=True)

@jax.jit
def train_step(params, batch):
    def loss_fn(p):
        logits = model_apply(p, batch["inputs"])
        labels = batch["labels"]
        return jnp.mean(cross_entropy(logits, labels))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

train_step = handler.training_step_wrapper(train_step)
loss, grads, grads_finite = train_step(params, batch)
```

`PrecisionHandler` jit-wraps casting, loss scaling, underflow detection, and gradient unscaling so the wrapped function stays focused on model math.

### 3. Work with quantized weights (NF4/INT8/Binary)

```python
import jax
import jax.numpy as jnp
from eformer.jaximus import implicit
from eformer.ops.quantization import (
    QuantizationConfig,
    QuantizationType,
    quantize,
    straight_through,
)

@implicit
def nf4_linear(x, w):
    return x @ w  # dot_general dispatches to implicit handlers when possible

config = QuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
nf4_weights = quantize(weight_fp32, config=config)

# Inference uses compressed tensors directly
logits = nf4_linear(inputs, nf4_weights)

# Training keeps float32 master weights but injects STE quantization on the fly
def loss_fn(master_weight):
    q_weight = straight_through(master_weight, config=config)
    preds = nf4_linear(inputs, q_weight)
    return jnp.mean((preds - targets) ** 2)

loss, grads = jax.value_and_grad(loss_fn)(weight_fp32)
```

`quantize` returns implicit arrays (NF4, INT8, Binary), and the `implicit` decorator routes JAX primitives (dot, pow, matmul, etc.) through registered handlers that load custom Triton/Pallas kernels when available.

### 4. Partition tensors and launch Ray jobs

```python
import jax
from eformer.common_types import BATCH, EMBED
from eformer.escale import MeshPartitionHelper, PartitionAxis, PartitionManager, create_mesh
from eformer.executor.ray import autoscale_execute_resumable, TpuAcceleratorConfig

mesh = create_mesh(axis_dims=(2, 2), axis_names=("dp", "tp"))
helper = MeshPartitionHelper(mesh)
manager = PartitionManager(paxis=PartitionAxis(batch_axis="dp", hidden_state_axis="tp"))

with mesh:
    sharded_state = helper.auto_shard_pytree(train_state)
    hidden = manager.shard(hidden_states, axes=(BATCH, EMBED))

job_status = autoscale_execute_resumable(
    remote_fn=train_slice_remote,  # decorated with @ray.remote
    accelerator_config=TpuAcceleratorConfig(type="v4-8", pod_count=2),
    max_retries_preemption=5,
    max_retries_failure=2,
)
```

`MeshPartitionHelper` inspects trees to produce sensible `PartitionSpec`s; `PartitionManager` gives semantic sharding (batch/hidden/etc.), and `RayExecutor` manages multi-slice TPU or GPU execution with resumable jobs.

## Examples & Guides

- `examples/quantization_training.py` – end-to-end training loop demonstrating NF4/INT8/Binary quantization with the unified API.
- `env.py` – short script showing NF4 straight-through training and inference using implicit arrays.
- `QUANTIZATION.txt` – quick-reference sheet for supported quantization modes.
- `docs/pytree_utils.md` – catalog of every PyTree helper with explanations.
- `docs/api_docs/*.rst` – per-module API descriptions used by Sphinx.

Run the example locally:

```bash
python examples/quantization_training.py
```

## Documentation

Hosted docs: [https://eformer.readthedocs.org](https://eformer.readthedocs.org/)

Build the Sphinx site locally:

```bash
pip install -r docs/requirements.txt
make -C docs html
# open docs/_build/html/index.html
```

`docs/index.rst` is the landing page, and the `api_docs/` folder mirrors the Python package layout so you can quickly locate functions/classes.

## Testing & Quality

Unit tests cover key areas such as PyTree utilities, optimizer factory logic, and quantization kernels (`tests/test_*.py`). To run them:

```bash
pip install -e '.[dev]'
pytest
```

The repository also contains formatter/linter configurations:

```bash
ruff check .
black --check .
```

Feel free to wire these commands into pre-commit hooks or your CI. `uv run pytest` works out of the box if you prefer uv's virtual environments.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and follow the Apache Code of Conduct. If you plan to work on distributed/TPU features, include repro steps or environment notes in the PR so we can validate them.

- Report bugs / feature requests via GitHub issues.
- Keep PRs focused, include tests where possible, and respect existing formatting rules (Black line length 121, Ruff config in `pyproject.toml`).
- See `CHANGES.txt` for release notes and `QUANTIZATION.txt` for design background.

## License

Licensed under the [Apache License 2.0](LICENSE). Portions of the executor/cluster utilities build upon the excellent work in the Stanford CRFM Levanter project; see file headers for details.
