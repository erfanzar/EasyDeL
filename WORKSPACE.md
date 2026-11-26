# EasyDeL Stack — workspace guide

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
monorepo containing seven packages that are developed together and released
independently (plus the standalone `libs/esurge-dashboard` Rust binary crate,
which is not published to PyPI):

| package   | path             | import      | PyPI           | mirror repo           |
| --------- | ---------------- | ----------- | -------------- | --------------------- |
| EasyDeL   | `libs/easydel`   | `easydel`   | `easydel`      | (this repo)           |
| Spectrax  | `libs/spectrax`  | `spectrax`  | `spectrax-lib` | `erfanzar/Spectrax`   |
| eJKernel  | `libs/ejkernel`  | `ejkernel`  | `ejkernel`     | `erfanzar/ejkernel`   |
| eFormer   | `libs/eformer`   | `eformer`   | `eformer`      | `erfanzar/eformer`    |
| eRay      | `libs/eray`      | `eray`      | `eray`         | `erfanzar/eray`       |
| surgecore | `libs/surgecore` | `surgecore` | `surgecore`    | `erfanzar/esurgecore` |
| Eyvan     | `libs/eyvan`     | `eyvan`     | `eyvan`        | `erfanzar/eyvan`      |

**This monorepo is the source of truth.** The standalone repositories are
read-only mirrors kept in sync automatically — do not merge changes there
directly (a direct push makes the sync fail until reconciled with
`scripts/subtree-sync.sh pull <lib>`).

## What each package owns

The stack splits along one line: the six companion libraries (five Python
libraries plus the Rust `surgecore` crate) know nothing about language models;
`easydel` knows nothing about how to write a kernel, shard an array, split a
pipeline, manage Ray pods, or run a fleet capacity plane — it composes the
companion packages.

### spectrax — the module system *and* the pipeline runtime

Two things in one package. First, a JAX-native module API — PyTorch-shaped
`spx.Module`/`spx.Parameter` over an explicit graph/state split (filling
the role flax nnx plays elsewhere) plus EasyDeL's primary live sharding,
TensorStore/OCDBT checkpointing, and path-rule QT/QAT quantization stack.
Second, and what sets it apart: an **execution runtime** that compiles a model
into true MPMD pipeline parallelism — per-rank executables (`sxjit`/`sxcall`),
automatic stage splitting, and 9+ schedules — a capability JAX's single-program
SPMD model does not natively provide (an SPMD pipeline path exists alongside
it). Every EasyDeL model is an `spx.Module` executed through this runtime.

| area                                           | owns                                                                                                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `core`                                         | `Module`, `Parameter`/`Buffer`, GraphDef/State split, `Selector` state filtering                                                          |
| `nn`, `functional`, `init`                     | 70+ layer modules (Linear/Conv/Attention/LoRA/FP8), stateless ops, initializers                                                           |
| `transforms`                                   | module-aware `jit`/`grad`/`vmap`/`scan`/`remat` with mutable state                                                                        |
| `runtime`                                      | pipeline parallelism: true-MPMD (per-rank executables via `sxjit`) and SPMD paths, 9+ schedules (GPipe, 1F1B, ZeroBubble, DualPipeV, ...) |
| `sharding`, `rng`                              | `SpxMesh`, logical axis naming / `PartitionSpec` derivation, `strict`/`audit`; named-stream deterministic RNG                             |
| `quantization`                                 | module-path QT/QAT (`QuantProvider`, `QuantRule`, `quantize_model`, `qdot_general`, `qeinsum`, `qragged_dot`, `QArray`)                   |
| `serialization`, `hooks`, `inspect`, `loggers` | TensorStore/OCDBT checkpointing (`Checkpointer`, `AsyncCheckpointManager`), hooks, `hlo_cost`/summaries, TB/W&B logging                   |

### ejkernel — the kernel library

Hardware-optimized implementations of every expensive op, behind a
priority-based registry that dispatches per platform (Triton / Pallas-TPU /
Pallas-GPU / CUDA-FFI / CUTLASS CuTe / TileLang) with a mandatory XLA fallback
for everything, plus config autotuning, a shipped SQLite tuned DB
(`ops/tuned/tuned_kernels.db`), and persistent config caching. `easydel` never
writes hardware kernels — it calls these through registered `OperationImpl`
adapters.

| family                       | examples                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| dense attention              | flash, block-sparse, ring (distributed long-context), standard MHA                                                       |
| paged / inference attention  | page, decode, unified prefill+decode, ragged_page_attention v2/v3 (chunked prefill, attention sinks; `csrc/` C++ for v3) |
| linear attention / recurrent | gated delta rule (GDR), gated linear attention, lightning, KDA, RWKV 4/6/7                                               |
| state space                  | Mamba-1 (`state_space_v1`), Mamba-2 SSD (`state_space_v2`)                                                               |
| losses, matmul & quant       | fused/chunked cross-entropy and KL, quantized + grouped matmul, `quantization/` (affine/NF4/MXFP/NVFP, TurboQuant)       |

### eformer — the infrastructure library

Scale infrastructure that is neither a model nor a kernel:

| area                                            | owns                                                                                         |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `escale`                                        | standalone mesh creation, `PartitionAxis`/`PartitionManager`, auto partition specs           |
| `executor`                                      | legacy execution compatibility and SLURM integration                                         |
| `optimizers`                                    | optimizer + scheduler factories (AdamW, Adafactor, Lion, Muon, `_stage_local.py`, fused tx)  |
| `serialization`, `paths`                        | SafeTensors/msgpack `CheckpointManager`, legacy TensorStore; `ePath` local/GCS abstraction   |
| `mpric`                                         | mixed-precision policies and dynamic loss scaling                                            |
| `jaximus`, `ops`                                | implicit/lazy arrays for quantization (NF4/INT8/binary) via JAX primitive registration       |
| `aparser`, `pytree`, `loggings`, `common_types` | dataclass CLI/YAML parsing, pytree utilities, logging/profiling, semantic axis constants     |

### eray — Ray orchestration and remote execution

Ray-based distributed execution, resource pools, TPU/GPU pod orchestration,
GCP queued-resource/fleet provisioning, chip-split swarms, Docker helpers, and
CLI workflows.

| area                    | owns                                                                             |
| ----------------------- | -------------------------------------------------------------------------------- |
| `cli`                   | `eray` CLI (`tpu`, `qr`, `fleet`, `autoscale`, `dashboard`, `jobs`, `hostjob`)   |
| `execution`             | `RayExecutor`, `execute`, `execute_resumable`, autoscale execution               |
| `pool`                  | Ray actor pools for slices and device hosts (`SlicePoolManager`, `SliceActor`)   |
| `provision`             | GCP provisioning: `fleet`, `qr` (Queued Resources), `launcher`, `tunnel`, `watcher` |
| `resources`, `swarm.py` | CPU/GPU/TPU accelerator configs, `SwarmConfig`, `GpuSwarmConfig`, host splitting |
| `docker`, `core`        | container execution, cluster models, exception helpers, sentinels                |

### surgecore — native Rust host core for eSurge

EasyDeL's native host library for the eSurge inference engine (`libs/surgecore`,
Rust crate compiled as PyO3 `abi3-py311` extension `surgecore._native` via
maturin). Only eSurge imports it; `import easydel` and training do not need it.

| component            | owns                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `EngineCore`         | host state machine: request table, per-DP queues, token budget, scheduler (FCFS/priority, chunked prefill, preemption), speculative verification & stop checks (`schedule()` → `Plan`, `commit()` → `Events`, GIL-released) |
| `KvCore`             | paged KV pool with LRU free list, chained prefix-cache map, full/sliding/chunked-local group rules, transactional allocation/rollback (`CacheManager` binding) |
| `SnapshotIndex`      | recurrent-state snapshot slots at prefix-cache page boundaries for linear-attention/SSM prefix reuse                               |
| `HostPayloadBuilder` | GIL-released filling of preallocated step-input staging arrays (token/position packing, DP packing, page-table rows, sampling vectors) |
| `Detokenizer`        | in-process batched incremental detokenization with incomplete UTF-8 buffering and suffix reconciliation                            |

### esurge-dashboard — standalone monitoring binary

`libs/esurge-dashboard` is a standalone Rust/Axum server-rendered monitoring
dashboard (`publish = false`, not a uv workspace member) that polls an eSurge
OpenAI API server's `GET /metrics` endpoint and serves sixteen inline-SVG live
charts (throughput, load, latency, prefix cache, requests, per-key access) with
no Grafana/Prometheus dependency. Build and check in `libs/esurge-dashboard`:
`cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`.

### eyvan — capacity plane for research compute

Policy-aware global fleet control plane (`libs/eyvan`, `AGPL-3.0-or-later`) for
scheduling, provisioning, and executing digest-verified Python callables across
TPU, GPU, and CPU pools. Pinned by `easydel` (so installing `easydel` installs
`eyvan` and its CLI entry points automatically) and resolved editable from
`libs/eyvan` during `uv sync`. Ships a React 19 / TypeScript / Vite dashboard
(`libs/eyvan/dashboard`) staged into `eyvan/_dashboard` at wheel build time.

| area                    | owns                                                                                                                               |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `domain/`               | pure deterministic scheduler (hierarchical DRF, gang preemption, TPU chip sharing), pricing/costing/spend, memory sizing, carbon   |
| `server/`               | FastAPI control plane (`eyvan-server`), `ControlPlaneService` (33 isolated mixins), 24-phase controller + outbox, SQLAlchemy/Alembic persistence, team mode, MCP bridge, fleet assistant |
| `capacity/`             | sourced hardware catalog (`accelerator_shapes.yaml`), pool planner, autoscale compiler, conformance suite, GCP/AWS/Azure/K8s/SSH/local drivers |
| `worker/`               | `eyvan-agent` + `eyvan-native-worker`, systemd transient units, multi-host TPU slice/gang execution, OCI stager, global-rank logs  |
| `track/`, `artifacts/`  | zero-dependency experiment tracking client + `eyvan-tracker` service; content-addressed artifacts and scoped capability tokens     |
| `sdk/`, `dashboard/`    | `@execute` decorator, `TpuConfig`/`GpuConfig`/`CpuConfig`, `EasyDeLSetup`, `eyvan` CLI (`proof`, `race`, `soak`, `head`, `team`), React SPA |

### easydel — the LLM framework

The only package allowed to import the six companion packages. By subsystem:

| subsystem                        | owns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `infra/`                         | the contracts everything implements: `EasyDeLBaseConfig` (HF-compatible config + sharding/mesh), `EasyDeLBaseModule` (model lifecycle, HF interop via the bridge mixin: `from_pretrained`/`from_torch`/`to_torch`), `EasyDeLState` (params + optimizer + step), `layer_scan.py`, loss utils, the task/model registry (`TaskType`, `register_module`)                                                                                                                                                               |
| `infra/elarge/`                  | eLarge: the dict-literal/YAML config system that builds an entire run (model + data + trainer + eval + serving engine) from one typed config; driven by `easydel/scripts/elarge.py`                                                                                                                                                                                                                                                                                                                               |
| `layers/`                        | reusable NN building blocks: `UnifiedAttention` + `FlexibleAttentionModule` (kernel-dispatching attention base), Column/Row-parallel & MoE linears, fused-projection **layouts** (`layers/layouts`: QKV/gate-up fusion, TP interleaving, checkpoint reform rules, `fused_param_tp` mesh portability), MoE routing/dispatch, compressed_window, embeddings, engram, indexer, linear_attention, norms, residual, rotary, sparse_attention, quantization                                                                    |
| `operations/`                    | the registry that routes attention and kernel calls to concrete ejkernel-backed `OperationImpl` adapters (FlashAttn, RaggedPageV3, Ring, GDR, ...) by hardware and runtime mode                                                                                                                                                                                                                                                                                                                                   |
| `caching/`                       | 10 inference cache families + budgeting (`budget.py`): `Transformer`, `RaggedPages`, `MLARaggedPages`, `TurboQuantRaggedPages`, `UnifiedAttention`, `Recurrent`, `Hybrid`, `KDA`, `CompressedWindow`, and `SharedPrefix`                                                                                                                                                                                                                                                                                          |
| `modules/`                       | the model zoo: 89 architecture families (dense, MoE, SSM/Mamba, linear-attention hybrids, VLMs, speech, embeddings, diffusion-LM, plus EasyDeL-native models like xerxes and the gemma4_assistant speculative drafter). Convention: one directory per family holding `modeling_<x>.py` + `<x>_configuration.py`; task wrappers come from `_base/`; `auto/` exposes `AutoEasyDeLModelForCausalLM` and friends over the registry                                                                                    |
| `trainers/`                      | 52 subdirs / 42 registered trainers over one compiled-step foundation (`base_trainer`, `trainer_protocol`): SFT; preference family (DPO/ORPO/CPO/KTO/BCO/Nash-MD/online-DPO); policy-gradient & online-RL family (GRPO/GSPO/GFPO/RLOO/DPPO/GDPO/GMPO/PPO/SAO/async-GRPO over eSurge rollouts + `_online_rl/` disaggregated weight transport); reward + PRM; distillation (logit KD, GKD, seq-KD, on-policy, SDFT, SDPO); embeddings; RLVR; agentic multi-turn tool-use (`agent_flow_trainer`, `agentic_moshpit`, `environments/`); `data_lanes.py` + `batch_placement.py`; `RewardProtocol` |
| `inference/`                     | **eSurge**: continuous-batching paged-attention serving engine (`esurge_engine.py`, `engine/`, `runners/`, `core/` cache+sampler backed by Rust `surgecore` `EngineCore`/`KvCore`, speculative decoding, MPMD pipeline serving, multimodal, ZMQ leader/worker multi-host mode, OpenAI-compatible FastAPI `server/`, multi-replica `serving/` cluster/dashboard; `scheduler/` holds `output.py` structs only); plus `vwhisper/` speech serving, 34 tool-call parser files (37 registered names), 16 reasoning parser files (25 registered names), `parsing/`, `evaluations/`, sampling |
| `data/`                          | sharded dataset sources (Parquet/random-access/rowgroup-shuffle/HF), deterministic mixtures + shuffling, transform DSL (`>>`), sequence packing, `teacher_cache.py`, Ray-distributed preprocessing                                                                                                                                                                                                                                                                                                               |
| `utils/`, `workers/`, `scripts/` | HF↔EasyDeL weight conversion (`parameters_transformation`), `ejit` compilation caching, traversals, memory analysis; ZMQ inference workers + auth + response stores; CLI entry points                                                                                                                                                                                                                                                                                                                             |

**How a forward pass flows:** `AutoEasyDeLModelForCausalLM.from_pretrained`
resolves a config + module from the registry, converts HF weights
(`utils/parameters_transformation`, fused via `layers/layouts`), builds
`spx.Module`s sharded by `spectrax.sharding` meshes; attention goes
module → `UnifiedAttention` → `FlexibleAttentionModule` → `operations` registry
→ ejkernel kernel for the current platform. Training wraps it in `EasyDeLState`
with eformer optimizers inside a trainer's compiled step; serving hands it to
eSurge (with Rust `surgecore` on the host path); checkpoints flow through
`spectrax.serialization` (TensorStore/OCDBT).

## Development

```bash
uv sync                              # all workspace packages editable in .venv
uv sync --extra tpu                  # + TPU extras        (jax[tpu], ejkernel[tpu])
uv sync --extra cuda                 # + CUDA extras
uv sync --extra torch --extra profile
```

During development the packages resolve against each other from `libs/`
(`{ workspace = true }` sources); published wheels keep the pinned PyPI
versions declared in `libs/easydel/pyproject.toml`.

`libs/surgecore` is a Rust extension (PyO3, abi3-py311, built by maturin): `uv sync` compiles it from
source, so it needs a Rust toolchain with `cargo` on PATH (a minimal stable rustup toolchain is enough). Its
crate checks run in `libs/surgecore`: `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`
and `cargo test --locked` (clippy and the tests link libpython: set `PYO3_PYTHON` to a Python 3.11-3.13
interpreter); `cargo test --locked --no-default-features` tests the pure-Rust core without Python. Workspace CI
runs these, a wheel build and the package's CPU tests whenever `libs/surgecore`, the root `pyproject.toml` or
`uv.lock` changes.

## Layering contract (CI-enforced)

```md
spectrax   ejkernel   eformer   eray   surgecore   eyvan  <- independent of each other and of easydel
       \       |         |       /        /        /
                         easydel                          <- the only package that may import the others
```

`uv run lint-imports` checks this locally; Workspace CI runs it on every PR.
`libs/eyvan` also enforces internal control-plane subsystem boundaries via
`(cd libs/eyvan && uv run --no-sync lint-imports)`.

## Testing

Workspace CI runs affected-only *smoke* checks (imports + a tiny easydel
forward on CPU, which only catches import/crash regressions and is not
validation) plus the layering contract. The deep suites are hardware-bound and
run locally. Every test that executes JAX numerics runs on an accelerator (TPU
shown; for GPU use `JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`), one test
process at a time on a TPU host (single libtpu lock):

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests -m "not slow"
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/eformer/tests
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/ejkernel/test
```

The CPU trio is only for tests that do no array computation (`eray`, `surgecore`,
`eyvan`, tool/reasoning parsers, config/CLI/YAML parsing, data text transforms,
loggers, paths, docs):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/eray/tests libs/surgecore/tests
(cd libs/eyvan && uv run --package eyvan --extra dev pytest tests -m fast -q)   # full eyvan suite needs dashboard build + PostgreSQL
```

The four JAX-library test suites (`easydel`, `spectrax`, `eformer`, `ejkernel`)
enforce this in `conftest.py` via an array-library import regex
(`_ARRAY_LIBRARY_IMPORT`), runtime XLA compile trapping (`_cpu_compile_guard`),
and `@pytest.mark.cpu_ok` (`eray`, `surgecore`, and `eyvan` do no JAX
computation and run on CPU). A CPU run of a computation test is never
validation — with no accelerator available, report the change as unverified on
hardware.

## Releasing

Releasing is split into two explicit steps — nothing leaves the machine
until you run the second one:

```bash
scripts/release.sh ejkernel 0.0.82   # bump + sync easydel pins + lock + COMMIT (local only)
scripts/publish.sh ejkernel          # tag ejkernel-v0.0.82 + push the tag -> CI publishes to PyPI
```

`.github/workflows/publish.yaml` builds and publishes exactly the tagged
package (PyPI trusted publishing, or the `PYPI_API_TOKEN` secret).
`publish.sh` pushes the tag only — push the branch yourself with `git push`.

`surgecore` releases with the same two scripts. Because `libs/surgecore` has a `Cargo.toml`, `release.sh`
also bumps the crate's `[package] version` and its own entry in `libs/surgecore/Cargo.lock` (no cargo needed),
and it syncs easydel's `surgecore==` pin like the other pins. A `surgecore-v*` tag skips the `uv build` job and
runs the `surgecore-*` jobs of `publish.yaml` instead: maturin builds manylinux_2_28 wheels for x86_64 and
aarch64 and a macOS arm64 wheel (abi3, so one wheel per platform covers Python 3.11-3.13) plus an sdist, and one
step publishes them. Because `easydel` pins exact sibling versions (`==`), any bumped companion library (`spectrax-lib`,
`ejkernel`, `eformer`, `eray`, `surgecore`, or `eyvan`) must be on PyPI before an `easydel` release that requires it is
published.

`eyvan` also releases with `scripts/release.sh eyvan <ver>` and `scripts/publish.sh eyvan`: `release.sh`
bumps `libs/eyvan/pyproject.toml`, `libs/eyvan/dashboard/package.json` (and `package-lock.json`), and
`libs/eyvan/deploy/helm/eyvan/Chart.yaml`, syncs `eyvan==` in `libs/easydel/pyproject.toml`, and an `eyvan-v*` tag runs
the `eyvan-publish` job in `.github/workflows/publish.yaml` to build and stage the React dashboard
(`libs/eyvan/scripts/stage_dashboard.py`) into the wheel/sdist before publishing.

## Mirror sync

Mirrors update automatically when you `git push`: the pre-push hook
(`scripts/subtree-sync.sh auto`) detects which of
spectrax/ejkernel/eformer/eray/surgecore/eyvan changed and subtree-pushes just
those to their standalone repos (`SUBTREE_SYNC_SKIP=1 git push` to bypass;
mirror outages never block the push, and a library missing from HEAD is
skipped). `.github/workflows/sync-subtrees.yaml` is the server-side backstop
(requires the `SUBTREE_SYNC_TOKEN` secret: fine-grained PAT, Contents
read/write on the six mirrors). Manual fallback:

```bash
scripts/subtree-sync.sh push            # all mirrors
scripts/subtree-sync.sh pull ejkernel   # reconcile a diverged mirror
```

## Dev tooling

`uv sync` installs the `dev` group by default: `pytest`, `pre-commit`,
`ruff`, `import-linter`, `basedpyright`. Activate the git hooks once per
clone:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```
