# EasyDeL Stack — workspace guide

This repository is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
monorepo containing four packages that are developed together and released
independently:

| package  | path            | import     | PyPI           | mirror repo         |
| -------- | --------------- | ---------- | -------------- | ------------------- |
| EasyDeL  | `libs/easydel`  | `easydel`  | `easydel`      | (this repo)         |
| Spectrax | `libs/spectrax` | `spectrax` | `spectrax-lib` | `erfanzar/Spectrax` |
| eJKernel | `libs/ejkernel` | `ejkernel` | `ejkernel`     | `erfanzar/ejkernel` |
| eFormer  | `libs/eformer`  | `eformer`  | `eformer`      | `erfanzar/eformer`  |

**This monorepo is the source of truth.** The standalone repositories are
read-only mirrors kept in sync automatically — do not merge changes there
directly (a direct push makes the sync fail until reconciled with
`scripts/subtree-sync.sh pull <lib>`).

## What each package owns

The stack splits along one line: the three foundation libs know nothing
about language models; easydel knows nothing about how to write a kernel,
shard an array, or split a pipeline — it composes the other three.

### spectrax — the module system *and* the pipeline runtime

Two things in one package. First, a JAX-native module API — PyTorch-shaped
`spx.Module`/`spx.Parameter` over an explicit graph/state split (filling
the role flax nnx plays elsewhere). Second, and what sets it apart: an
**execution runtime** that compiles a model into true MPMD pipeline
parallelism — per-rank executables (`sxjit`/`sxcall`), automatic stage
splitting, and 9+ schedules — a capability JAX's single-program SPMD model
does not natively provide (an SPMD pipeline path exists alongside it).
Every EasyDeL model is an `spx.Module` executed through this runtime.

| area                                           | owns                                                                                                                                      |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `core`                                         | `Module`, `Parameter`/`Buffer`, GraphDef/State split, `Selector` state filtering                                                          |
| `nn`, `functional`, `init`                     | 70+ layer modules (Linear/Conv/Attention/LoRA/FP8), stateless ops, initializers                                                           |
| `transforms`                                   | module-aware `jit`/`grad`/`vmap`/`scan`/`remat` with mutable state                                                                        |
| `runtime`                                      | pipeline parallelism: true-MPMD (per-rank executables via `sxjit`) and SPMD paths, 9+ schedules (GPipe, 1F1B, ZeroBubble, DualPipeV, ...) |
| `sharding`, `rng`                              | logical axis naming / `PartitionSpec` derivation; named-stream deterministic RNG                                                          |
| `serialization`, `hooks`, `inspect`, `loggers` | TensorStore checkpointing, forward/variable hooks, model summaries, TB/W&B logging                                                        |

### ejkernel — the kernel library

Hardware-optimized implementations of every expensive op, behind a
priority-based registry that dispatches per platform (Triton / Pallas-TPU /
Pallas-GPU / CUDA-FFI / CUTLASS CuTe / TileLang) with an XLA fallback for
everything, plus config autotuning and caching. easydel never writes
kernels — it calls these.

| family                       | examples                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| dense attention              | flash, block-sparse, ring (distributed long-context), standard MHA                                                       |
| paged / inference attention  | page, decode, unified prefill+decode, ragged_page_attention v2/v3 (chunked prefill, attention sinks; `csrc/` C++ for v3) |
| linear attention / recurrent | gated delta rule (GDR), gated linear attention, lightning, KDA, RWKV 4/6/7                                               |
| state space                  | Mamba-1 (`state_space_v1`), Mamba-2 SSD (`state_space_v2`)                                                               |
| losses & matmul              | fused/chunked cross-entropy and KL, quantized + grouped matmul                                                           |

### eformer — the infrastructure library

Everything about *running* JAX at scale that is neither a model nor a kernel:

| area                                            | owns                                                                                         |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `escale`                                        | mesh creation, `PartitionAxis`/`PartitionManager`, auto partition specs for DP/FSDP/TP/EP/SP |
| `executor`                                      | Ray-based TPU/GPU pod orchestration, SLURM integration                                       |
| `optimizers`                                    | optimizer + scheduler factories (AdamW, Adafactor, Lion, Muon, fused variants)               |
| `serialization`, `paths`                        | TensorStore sharded checkpointing without all-gathers; `ePath` local/GCS path abstraction    |
| `mpric`                                         | mixed-precision policies and dynamic loss scaling                                            |
| `jaximus`, `ops`                                | implicit/lazy arrays for quantization (NF4/INT8/binary) via JAX primitive registration       |
| `aparser`, `pytree`, `loggings`, `common_types` | dataclass CLI/YAML parsing, pytree utilities, logging/profiling, semantic axis constants     |

### easydel — the LLM framework

The only package allowed to import the other three. By subsystem:

| subsystem                        | owns                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `infra/`                         | the contracts everything implements: `EasyDeLBaseConfig` (HF-compatible config + sharding/mesh), `EasyDeLBaseModule` (model lifecycle, HF interop via the bridge mixin: `from_pretrained`/`from_torch`/`to_torch`), `EasyDeLState` (params + optimizer + step), loss utils, the task/model registry (`TaskType`, `register_module`)                                                                                                                                                                               |
| `infra/elarge/`                  | eLarge: the dict-literal/YAML config system that builds an entire run (model + data + trainer + eval + serving engine) from one typed config; driven by `scripts/elarge.py`                                                                                                                                                                                                                                                                                                                                       |
| `layers/`                        | reusable NN building blocks: `UnifiedAttention` + `FlexibleAttentionModule` (kernel-dispatching attention base), Column/Row-parallel linears, fused-projection **layouts** (`layers/layouts`: QKV/gate-up fusion, TP interleaving, checkpoint reform rules, `fused_param_tp` mesh portability), MoE routing/dispatch, norms, rotary variants, quantization                                                                                                                                                        |
| `operations/`                    | the registry that routes attention calls to concrete ejkernel-backed implementations (FlashAttn, RaggedPageV3, Ring, ...) by hardware and runtime mode                                                                                                                                                                                                                                                                                                                                                            |
| `caching/`                       | KV-cache state types per architecture class: transformer pages, recurrent/linear-attention states, MLA, hybrid combinations                                                                                                                                                                                                                                                                                                                                                                                       |
| `modules/`                       | the model zoo: ~74 architecture families (dense, MoE, SSM/Mamba, linear-attention hybrids, VLMs, speech, embeddings, diffusion-LM, plus EasyDeL-native models like xerxes and the gemma4_assistant speculative drafter). Convention: one directory per family holding `modeling_<x>.py` + `<x>_configuration.py`; task wrappers come from `_base/`; `auto/` exposes `AutoEasyDeLModelForCausalLM` and friends over the registry                                                                                   |
| `trainers/`                      | ~40 trainers over one compiled-step foundation (`base_trainer`): SFT; the preference family (DPO/ORPO/CPO/KTO/BCO/Nash-MD/online-DPO); the policy-gradient family (GRPO/GSPO/GFPO/RLOO/PPO + async-GRPO over eSurge rollouts); reward + process-reward models; the distillation family (logit KD, GKD, sequence-KD, on-policy, self-distillation); embeddings (contrastive/Matryoshka); RLVR verifiable rewards; agentic multi-turn tool-use training (`agentic_moshpit`); `RewardProtocol` for pluggable rewards |
| `inference/`                     | **eSurge**: the continuous-batching paged-attention serving engine (scheduler + runners/executors, page-pool KV management, speculative decoding, multimodal preprocessing, Prometheus metrics, ZMQ leader/worker multi-host mode) with an OpenAI-compatible FastAPI server (auth/RBAC/quotas); plus vWhisper speech serving, 25+ model-specific tool-call parsers, reasoning parsers, sampling/logits processing                                                                                                 |
| `data/`                          | sharded dataset sources (parquet/arrow/json/HF), deterministic mixtures + shuffling, transform DSL (chat templates, tokenization, packing), Ray-distributed preprocessing                                                                                                                                                                                                                                                                                                                                         |
| `utils/`, `workers/`, `scripts/` | HF↔EasyDeL weight conversion (`parameters_transformation`), `ejit` compilation caching, traversals, memory analysis; ZMQ inference workers + response stores; CLI entry points                                                                                                                                                                                                                                                                                                                                    |

**How a forward pass flows:** `AutoEasyDeLModelForCausalLM.from_pretrained`
resolves a config + module from the registry, converts HF weights
(`utils/parameters_transformation`, fused via `layers/layouts`), builds
`spx.Module`s sharded by eformer/spectrax meshes; attention goes
module → `FlexibleAttentionModule` → `operations` registry → ejkernel
kernel for the current platform. Training wraps it in `EasyDeLState` with
eformer optimizers inside a trainer's compiled step; serving hands it to
eSurge; checkpoints flow through the TensorStore serialization stack.

## Development

```bash
uv sync                              # all four packages editable in .venv
uv sync --group tpu                  # + TPU extras        (jax[tpu], ejkernel[tpu])
uv sync --group cuda                 # + CUDA extras
uv sync --group torch --group profile
```

During development the packages resolve against each other from `libs/`
(`{ workspace = true }` sources); published wheels keep the pinned PyPI
versions declared in `libs/easydel/pyproject.toml`.

## Layering contract (CI-enforced)

```md
spectrax    ejkernel    eformer      <- independent of each other and of easydel
       \        |        /
              easydel                <- the only package that may import the others
```

`uv run lint-imports` checks this locally; Workspace CI runs it on every PR.

## Testing

Workspace CI runs affected-only *smoke* checks (imports + a tiny easydel
forward) plus the layering contract. The deep suites are hardware-bound and
run locally:

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests -m "not slow"
uv run pytest libs/spectrax/tests
uv run pytest libs/eformer/tests
uv run pytest libs/ejkernel/test        # kernels: most need GPU/TPU
```

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

## Mirror sync

Mirrors update automatically when you `git push`: the pre-push hook
(`scripts/subtree-sync.sh auto`) detects which of spectrax/ejkernel/eformer
changed and subtree-pushes just those to their standalone repos
(`SUBTREE_SYNC_SKIP=1 git push` to bypass; mirror outages never block the
push). `.github/workflows/sync-subtrees.yaml` is the server-side backstop
(requires the `SUBTREE_SYNC_TOKEN` secret: fine-grained PAT, Contents
read/write on the three mirrors). Manual fallback:

```bash
scripts/subtree-sync.sh push            # all three
scripts/subtree-sync.sh pull ejkernel   # reconcile a diverged mirror
```

## Dev tooling

`uv sync` installs the `dev` group by default: `pytest`, `pre-commit`,
`ruff`, `import-linter`, `basedpyright`. Activate the git hooks once per
clone:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```
