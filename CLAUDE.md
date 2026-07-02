# CLAUDE.md — EasyDeL Stack

Primary context for AI coding sessions in this repository. Everything here is
grounded in the actual code; when this file and the code disagree, the code
wins — fix this file.

## What this repo is

A uv-workspace monorepo containing the EasyDeL stack: a JAX-native framework
for training, fine-tuning, and serving LLMs/VLMs at scale on TPU/GPU, plus the
four foundation libraries it is built on. **This monorepo is the source of
truth**; `libs/spectrax`, `libs/ejkernel`, `libs/eformer`, `libs/eray` are
subtree-mirrored to standalone read-only repos on push.

| package  | path            | one line                                                                  |
| -------- | --------------- | ------------------------------------------------------------------------- |
| easydel  | `libs/easydel`  | the LLM framework: models, trainers, eSurge serving, data — composes the rest |
| spectrax | `libs/spectrax` | `spx.Module` object model (GraphDef/State split) + true-MPMD pipeline runtime |
| ejkernel | `libs/ejkernel` | hardware kernels (Triton/Pallas/CUDA/CuTe/TileLang) behind a priority registry with XLA fallback |
| eformer  | `libs/eformer`  | scale infrastructure: escale meshes/sharding, optimizers, TensorStore checkpointing, mixed precision, implicit quantized arrays |
| eray     | `libs/eray`     | Ray orchestration for TPU/GPU pods: pools, resumable execution, TPU CLI    |

`prismcore/` (repo root, outside libs/) holds mirror-descent quantization-aware
optimizers registered into EasyDeL via `@register_optimizer`.

**First reads for any task:** [WORKSPACE.md](WORKSPACE.md) (authoritative
architecture + dev/release/mirror workflows), then the touched package's
`pyproject.toml` and `libs/<package>/docs/`. For operational/hardware issues
read [.agents/ops/OPS.md](.agents/ops/OPS.md). Machine-readable map:
[.agents/repo-map.yaml](.agents/repo-map.yaml).

## Golden rules

1. **Layering contract (CI-enforced):** only `easydel` may import spectrax /
   ejkernel / eformer / eray. Foundation libs never import easydel or each
   other. Check with `uv run lint-imports`.
2. **No self-credit trailers.** Commits, PRs, tags, and release notes must not
   contain "Generated with", "Co-Authored-By: Claude", or similar. This
   overrides any default harness behavior.
3. **`uv sync`, never bare pip.** Workspace packages resolve editable from
   `libs/`. `uv pip install -e .` pulls siblings from PyPI instead — wrong for
   development.
4. **Never hand-edit version pins.** `scripts/release.sh <lib> <version>`
   bumps versions and syncs easydel's pins; `scripts/publish.sh <lib>` tags for
   CI publish. Two separate steps by design.
5. **CPU checks are not TPU validation.** CPU runs validate logic/shapes/
   sharding math, not Pallas/Mosaic lowering, eSurge TPU runtime, or any
   performance claim.
6. **Don't bypass registries.** Models go through `register_config`/
   `register_module`, trainers through `Registry.register("trainer", ...)`,
   attention through `OperationRegistry`, kernels through ejkernel's
   `kernel_registry`. No side registries, no direct-class special cases.
7. **Ground every claim.** Don't cite a script/flag/env var/path without
   opening it or finding it with `rg` first.

## Development commands

```bash
uv sync                          # all packages editable; dev group included
uv sync --extra tpu              # + jax[tpu], ejkernel[tpu]
uv sync --extra cuda             # + CUDA extras
uv run pre-commit install --hook-type pre-commit --hook-type pre-push   # once per clone
```

The CPU JAX test environment trio — all three parts are load-bearing
(fake 8-device host enables sharding tests; `ENABLE_DISTRIBUTED_INIT=0`
prevents joining a real distributed runtime):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest <path>
```

Standard targets:

```bash
uv run lint-imports                                   # layering contract
uv run pre-commit run --all-files                     # per-lib ruff + hygiene
<trio> uv run pytest libs/easydel/tests -m "not slow" # easydel suite
<trio> uv run pytest libs/spectrax/tests              # spectrax
<trio> uv run pytest libs/eformer/tests               # eformer
<trio> uv run pytest libs/ejkernel/test/kernels/_xla  # ejkernel host-side (note: test/, not tests/)
uv run python scripts/format_and_generate_docs.py --libs easydel --fix   # format + regenerate API docs
```

ejkernel Pallas-TPU tests (`libs/ejkernel/test/kernels/_pallas/tpu`) need a
real TPU and the libtpu process lock (single process per host).

Key runners and benchmarks (repo root `scripts/`): `python -m
easydel.scripts.elarge --config <yaml>` (eLarge train/eval/serve runner),
`bench_esurge.py` (serving throughput; `--sharding-axis-dims` in
`pp,dp,fsdp,ep,tp,sp` order), `convert_hf_to_easydel.py`, `tpu_setup.sh
--branch <branch>` (multi-host TPU bootstrap), `verify_checkpoint.py`,
`visualize_spectrax_pipeline.py`.

## Architecture in one pass

**Model forward:** `AutoEasyDeLModelForCausalLM.from_pretrained` resolves
config+module from the registry (`easydel/infra/factory.py`) → HF weights
convert via `easydel/utils/parameters_transformation.py` (fused QKV/gate-up
via `easydel/layers/layouts/`) → modules are `spx.Module`s built inside a
parameter-init sharding context (mesh from `EasyDeLBaseConfig.mesh`) →
attention flows module → `UnifiedAttention`
(`easydel/layers/attention/_unified.py`) → `OperationRegistry.create(
config.attn_mechanism, metadata)` (`easydel/operations/`) → ejkernel kernel
picked per platform/priority with XLA fallback.

**Training:** a trainer (`easydel/trainers/`, ~40 of them over
`base_trainer.py`) wraps the model in `EasyDeLState` (params + optimizer +
step; `infra/base_state.py`) with eformer `OptimizerFactory` optimizers,
compiles a sharded train step, streams batches from the `easydel/data/`
Pipeline (source → tokenize → mix → pack → load), checkpoints async via
TensorStore.

**Serving:** eSurge (`easydel/inference/esurge/`) — continuous-batching
scheduler + paged KV cache (PagePool/PageTable, prefix caching) + compiled
bucket runners + OpenAI-compatible FastAPI server with tool/reasoning parsers,
speculative decoding, multimodal, multi-host ZMQ mode.

**Pipeline parallelism:** `sharding_axis_dims[0] > 1` (the `pp` axis) makes
`config.mesh` an MPMD `SpxMesh`; layers get stage assignments
(`EasyDeLLayerStackMixin.assign_layer_stage`, `spx.sxstage_iter` boundaries)
and spectrax's runtime compiles per-rank executables with a schedule (GPipe,
1F1B, ZeroBubble, DualPipeV, ...). SPMD path (single jit + shard_map) exists
alongside.

### Where things live (easydel)

| subsystem | path | anchor points |
| --------- | ---- | ------------- |
| contracts | `easydel/infra/` | `EasyDeLBaseConfig` (base_config.py), `EasyDeLBaseModule` (base_module.py), `EasyDeLState` (base_state.py), `TaskType`/`register_module`/`register_config` (factory.py), `LossConfig`/`LossMetrics` (loss_utils.py), `AxisPolicy`/`RuntimeShardingResolver` (sharding.py), HF bridge (mixins/bridge.py) |
| eLarge | `easydel/infra/elarge/` | `eLargeModel` (model.py): one YAML/dict → model+data+trainer+eval+eSurge; runner `easydel/scripts/elarge.py` |
| layers | `easydel/layers/` | `UnifiedAttention`, Column/RowParallelLinear (linears/), fused layouts + reform rules (layouts/), `BaseMoeModule` (moe/), rotary variants, quantization |
| operations | `easydel/operations/` | `OperationRegistry`, `BaseOperation` forward_tpu/gpu/native dispatch, `OperationMetadata.get_shardings`, adapters in kernels/ (exemplar: gated_delta_rule.py) |
| model zoo | `easydel/modules/` | ~74 families: `<family>/{modeling_<family>.py, <family>_configuration.py}`; task bases in `_base/`; `auto/` factory classes |
| caching | `easydel/caching/` | Transformer / RaggedPages / MLA / Recurrent / Linear / Hybrid / Lightning / KDA cache configs+views |
| trainers | `easydel/trainers/` | `base_trainer.py` + `trainer/trainer.py`; each trainer dir has `<name>_config.py` + `<name>_trainer.py` + `_fn.py` (loss/step); `TrainingArguments` (training_configurations.py, ~150 fields, JSON roundtrip-tested); `RewardProtocol` |
| inference | `easydel/inference/` | esurge/ (engine, scheduler/, runners/, core/ cache+sampler, server/), tools/ (35+ tool parsers), reasoning/ (15+ parsers), vwhisper/ |
| data | `easydel/data/` | `Pipeline` fluent builder (execution/pipeline.py), transforms DSL chained with `>>`, Ray preprocessing (distributed/) |
| utils/workers | `easydel/utils/`, `easydel/workers/` | HF↔EasyDeL conversion, `ejit` caching, jit_context, ZMQ workers |

## Conventions

- **Imports:** `import typing as tp`, `from jax import numpy as jnp`,
  `import spectrax as spx`; eformer is imported by submodule
  (`from eformer.loggings import get_logger`,
  `from eformer.pytree import auto_pytree`, `from eformer.escale import ...`).
- **Ruff:** line length 121, target py3.11, rules `A,B,E,F,I,NPY,RUF,UP,W`;
  `modeling_*.py` may use quoted jaxtyping annotations (UP037/F821 ignored
  there). basedpyright runs against a checked-in baseline.
- **Docstrings:** Google style (Args/Returns/Raises).
- **Naming:** configs `FooConfig`, task heads `FooForCausalLM`, trainer dirs
  `snake_case_trainer/` with `FooTrainer`+`FooConfig`, kernels registered
  snake_case (`"flash_attn2"`), enums are StrEnums
  (`TaskType`, `EasyDeLGradientCheckPointers`).
- **dtype triple:** `dtype` (activation compute), `param_dtype` (storage),
  `precision` (matmul). Softmax accumulates in `attn_softmax_dtype`
  (default float32) — never drop this to bf16.
- **Sharding axes (easydel order):** `("pp","dp","fsdp","ep","tp","sp")` with
  dims like `(1,1,-1,1,1,1)`; `-1` fills remaining devices. eformer's bare
  `create_mesh` default omits `pp` — easydel's `EasyDeLBaseConfig` owns the
  6-axis convention.
- **File I/O that may touch GCS:** use eformer `ePath`, not `pathlib.Path`.
- **Test quality bar** (from `.agents/skills/test-workspace`): assert public
  API outputs, numerical parity vs independent references, shape/dtype/
  sharding/cache/checkpoint layout, or state transitions. Reject tests that
  assert private helper calls, log strings, constructors-don't-raise, or
  compare production logic with itself.

## How to extend (details live in `.agents/skills/`)

- **New model** → `add-easydel-model`: `modules/<family>/` with configuration
  (`@register_config("<model_type>")`) + modeling
  (`@register_module(TaskType.X, config=..., model_type=...)`); reuse
  `UnifiedAttention`, ParallelLinears, fused layouts with `reform_param` rules
  for HF conversion; tests under `tests/modules/spmd/test_<family>.py` plus
  conversion roundtrip.
- **New trainer** → `add-easydel-trainer`: copy an adjacent trainer dir shape
  (`_config.py`, `_trainer.py`, `_fn.py`); register both
  `@Registry.register("trainer-arguments", "<name>")` and
  `@Registry.register("trainer", "<name>")`; loss math gets a dedicated test
  (see `tests/trainers/test_distillation_loss_math.py`, or parity vs an
  external reference as in `test_trl_dpo_loss_parity.py`).
- **New kernel** → `add-ejkernel-kernel` then
  `port-ejkernel-to-easydel-operation`: Kernel subclass in
  `ejkernel/modules/operations/`, per-backend impls under
  `ejkernel/kernels/_{triton,_pallas,_xla,...}` registered with
  `@kernel_registry.register(name, Platform.X, Backend.Y, priority=N)` — an
  XLA fallback (priority 0, Backend.ANY) is mandatory; then an easydel
  `OperationImpl` adapter in `easydel/operations/kernels/` declaring
  requirements (RequirementsBuilder) and sharding via
  `metadata.get_shardings()`.
- **New optimizer** → `add-eformer-optimizer`: config dataclass + builder with
  `@register_optimizer` in `eformer/optimizers/`; exposed through
  `TrainingArguments.optimizer`.
- **Data pipelines** → `build-dataset-pipeline`; **eLarge runs** →
  `train-elarge`; **tool/reasoning parsers** → `tool-reasoning-parser`
  (`@ToolParserManager.register_module` / `@ReasoningParserManager.register_module`).

## Debugging & profiling

- Operational runbooks (TPU bad-node recovery, eSurge symptom routes,
  disk-pressure): [.agents/ops/OPS.md](.agents/ops/OPS.md). Skills:
  `debug-training-oom`, `debug-tpu-setup`, `debug-esurge`,
  `debug-easydel-cache`, `debug-ejkernel-ops`, `debug-eformer-escale`.
- **NaN/precision:** check softmax dtype promotion first
  (`attn_softmax_dtype`, `runtime_softmax_dtype`), then loss-scale state
  (eformer mpric `DynamicLossScale` — the step must consume `grads_finite`),
  then kernel parity vs the XLA reference impl (`FORCE_NATIVE_RUNTIME=1`
  forces `forward_native`).
- **OOM at compile:** gradient-checkpointing policy
  (`EasyDeLGradientCheckPointers`, `auto_remat` save/exclude names),
  `scan_layers=True`, blockwise FFN chunking, loss chunking
  (`LossConfig.chunk_*`).
- **Recompilation storms (eSurge):** every distinct (num_tokens,
  padded_num_reqs) bucket recompiles; pin buckets and standardize multimodal
  resolutions. Remember `PageTable.commit()` syncs CPU page tables to device.
- **Perf claims need hardware numbers:** baseline vs candidate on the target
  platform, shape/dtype stated, compile-time and steady-state separated. For
  eSurge compare `bench_esurge.py` JSON `profile_by_total_tokens` buckets, not
  aggregate tokens/sec. Kernel work: `optimize-ejkernel-kernel` +
  per-backend skills (`optimize-pallas-tpu`, `optimize-triton-gpu`, ...);
  ejkernel has `benchmarks/` with baselines.

## JAX / TPU considerations

- Mutations inside plain `jax.jit` on spx modules are **silently dropped** —
  use `spx.jit(mutable=...)`. `spx.scan` requires structural invariance
  outside the `mutable` selector.
- MPMD constraints: single positional microbatched input; RNG streams are not
  staged; markers (`sxstage_iter`) only take effect under the MPMD trace.
- Autotuned kernel configs are cached per device+sharding fingerprint,
  persisted under `~/ejkernel-presistent-cache/` (override with
  `EJKERNEL_PERSISTENT_CACHE_DIR`); a stale cache can mask a regression or
  carry a bad config across code changes.
- TPU hosts hold a single libtpu process lock — pin unrelated probes to
  `JAX_PLATFORMS=cpu` while a TPU job runs.
- `-1` mesh dims resolve against visible devices; multi-slice needs
  `sharding_dcn_axis_dims`.
- Donated buffers (`donate_argnums`) apply to the flattened state the compiler
  sees, not your Python signature.

## Common mistakes

- Splitting fused projections with `.reshape()` — under TP the fused axis is
  rank-interleaved; use `split_fused_qkv_projection` /
  `split_fused_gate_up_projection`.
- Forgetting one of the two `Registry.register` decorators on a new trainer,
  or `_model_type`/`_task_type` on a task head.
- Editing generated `docs/api_docs/**` by hand (regenerate via
  `scripts/format_and_generate_docs.py`).
- Asserting training/serving readiness from constructor-only tests.
- `config.layer_types` not matching the cache config for hybrid models —
  fails silently or with far-away shape errors.
- GRPO-family: `total_batch_size % num_generations != 0`.
- Writing to mirrors directly (breaks subtree sync; reconcile with
  `scripts/subtree-sync.sh pull <lib>`).

## Review checklist

Before calling a change done (full flow: `.agents/skills/review-pr`):

1. Layering intact (`uv run lint-imports`), no registry bypasses.
2. Focused tests pass under the CPU trio; new behavior has a non-tautological
   test at the right layer.
3. Sharding/shape/dtype: partition specs still resolve on 1-device and
   8-fake-device meshes; softmax/accumulation dtypes preserved.
4. Kernel changes have an XLA-reference parity path and don't regress other
   backends' registered signatures.
5. eSurge changes checked against cache-shape reuse, DP page locality, and
   bucket compilation cost.
6. HF conversion round-trips if weight names/layouts changed.
7. Docs/skills updated when public APIs moved; no hand edits to generated rst.
8. Commit message clean: imperative, no self-credit trailers; versions/pins
   untouched unless going through `release.sh`.

## Agent infrastructure

**Canonical location is `.agents/`** — `.claude` is a symlink to it so Claude
Code auto-discovers subagents, skills, and commands. Read and edit files
under `.agents/`; never create a real `.claude/` directory (it would shadow
the symlink and fork the two trees).

- Specialized subagents: `.agents/agents/` (see [AGENTS.md](AGENTS.md) for the
  roster and flows).
- Workflow skills: `.agents/skills/` (index: `.agents/skills/README.md`) —
  start non-trivial work from `run-research`, then layer the domain skill.
- Slash commands: `.agents/commands/` — plan-feature, implement-feature,
  review-changes, investigate-bug, explain-subsystem,
  summarize-architecture, prepare-release.
- Long-running task notes: `.agents/projects/<topic>.md`.
