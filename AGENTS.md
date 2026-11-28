# AGENTS.md — EasyDeL Stack

Primary context for AI coding sessions in this repository — conventions, commands, architecture, and the agent playbook
in one place. Everything here is grounded in the actual code; when this file and the code disagree, the code wins — fix
this file.

## What this repo is

A uv-workspace monorepo containing the EasyDeL stack: a JAX-native framework for training, fine-tuning, and serving
LLMs/VLMs at scale on TPU/GPU, plus the six companion libraries alongside it (five Python libraries and the native Rust
`surgecore` crate) and the standalone `libs/esurge-dashboard` Rust binary crate. **This monorepo is the source of
truth**; `libs/spectrax`, `libs/ejkernel`, `libs/eformer`, `libs/eray`, `libs/surgecore`, and `libs/eyvan` are
subtree-mirrored to standalone read-only repos on push.

| package   | path             | one line                                                                                                                        |
|-----------|------------------|---------------------------------------------------------------------------------------------------------------------------------|
| easydel   | `libs/easydel`   | the LLM framework: models, trainers, eSurge serving, data — composes the rest                                                   |
| spectrax  | `libs/spectrax`  | `spx.Module` object model (GraphDef/State split), primary EasyDeL sharding/serialization/QT-QAT stack, true-MPMD runtime        |
| ejkernel  | `libs/ejkernel`  | hardware kernels (Triton/Pallas/CUDA/CuTe/TileLang) behind a priority registry with XLA fallback                                |
| eformer   | `libs/eformer`   | scale infrastructure: optimizers, mixed precision, implicit quantized arrays, `ePath`, standalone escale/serialization          |
| eray      | `libs/eray`      | Ray orchestration and GCP provisioning for TPU/GPU pods: pools, resumable execution, swarms, CLI                                |
| surgecore | `libs/surgecore` | EasyDeL's native Rust library (PyO3, maturin) for eSurge's host path; imported only by easydel; mirror erfanzar/esurgecore      |
| eyvan     | `libs/eyvan`     | policy-aware fleet capacity plane (scheduler, worker runtime, experiment tracking, React dashboard); mirror erfanzar/eyvan      |

`libs/esurge-dashboard` is a standalone Rust/Axum server-rendered monitoring binary (`publish = false`, not a uv
workspace member) that reads eSurge's `GET /metrics`.

**First reads for any task:** [WORKSPACE.md](WORKSPACE.md) (authoritative architecture + dev/release/mirror workflows),
then the touched package's `pyproject.toml` (plus `Cargo.toml` for Rust crates) and `libs/<package>/docs/` (or
`libs/<package>/README.md` for `eray`, `surgecore`, and `esurge-dashboard`). For operational/hardware issues read
[.agents/ops/OPS.md](.agents/ops/OPS.md). Machine-readable map: [.agents/repo-map.yaml](.agents/repo-map.yaml).
Non-trivial work starts from `.agents/skills/run-research/SKILL.md` and layers a domain skill on top.

## Golden rules

These apply to every agent and every session, without exception.

1. **Layering contract (CI-enforced):** only `easydel` may import spectrax / ejkernel / eformer / eray / surgecore /
   eyvan. Companion libs never import easydel or each other. Check with `uv run lint-imports` (and
   `(cd libs/eyvan && uv run --no-sync lint-imports)` for eyvan's internal control-plane boundaries).
2. **`uv sync`, never bare pip.** Workspace packages resolve editable from
   `libs/`. `uv pip install -e .` pulls siblings from PyPI instead — wrong for development.
3. **Never hand-edit version pins.** `scripts/release.sh <lib> <version>`
   bumps versions and syncs easydel's pins; `scripts/publish.sh <lib>` tags for CI publish. Two separate steps by
   design. Exact-pin siblings must be on PyPI before publishing an `easydel` release that depends on them.
4. **Computation tests run on an accelerator — always.** Any test that executes JAX numerics (models, layers,
   attention, MoE, kernels, losses, optimizers, trainers, caches, eSurge, spectrax transforms/runtime, eformer escale /
   mpric / ops) must run on TPU (or GPU where that is the target hardware). CPU is allowed only for tests that do no
   array computation (config/CLI parsing, tool/reasoning parsers, data text transforms, loggers, paths, `eray`,
   `surgecore`, `eyvan`). A CPU run of a computation test is not validation and must never be reported as such; with no
   accelerator available, report the change as **unverified on hardware**. The four JAX-library test suites (`easydel`,
   `spectrax`, `eformer`, `ejkernel`) enforce this in `conftest.py` via an array-library import regex
   (`_ARRAY_LIBRARY_IMPORT`), runtime XLA compile trapping (`_cpu_compile_guard`), and `@pytest.mark.cpu_ok` (`eray`,
   `surgecore`, and `eyvan` do no JAX computation and run on CPU).
5. **Don't bypass registries.** Models go through `register_config`/
   `register_module`, trainers through `Registry.register("trainer", ...)`, attention through `OperationRegistry`,
   kernels through ejkernel's `kernel_registry`. No side registries, no direct-class special cases.
6. **Ground every claim.** Don't cite a script/flag/env var/path without opening it or finding it with `rg` (or
   `git grep`) first.
7. **Kernel layering & code quality bar.** Hardware kernels (Pallas, Triton, CUDA, CuTe, TileLang, etc.) always go in
   `libs/ejkernel/` following every module/registry guide — with a mandatory `Platform.XLA` (`Backend.ANY`) fallback —
   and are then ported into `libs/easydel/easydel/operations/kernels/` as registered `OperationImpl` adapters consumed
   by EasyDeL modules (never inline hardware kernels in `libs/easydel/easydel/modules/`). Every change must be
   (1) modular, (2) hackable, (3) production-ready, (4) verified with tests, (5) free of dead code, (6) free of
   duplicated code, and (7) aligned with the project mindset.

## Development commands

```bash
uv sync                          # all packages editable; dev group included
uv sync --extra tpu              # + jax[tpu], ejkernel[tpu]
uv sync --extra cuda             # + CUDA extras
uv run pre-commit install --hook-type pre-commit --hook-type pre-push   # once per clone
```

`libs/surgecore` is a Rust extension (PyO3, built by maturin), so `uv sync` compiles it from source and needs a
Rust toolchain with `cargo` on PATH (a minimal stable rustup toolchain is enough). Its crate checks run in
`libs/surgecore`: `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`, `cargo test --locked`
(clippy and tests link libpython, so set `PYO3_PYTHON` to a Python 3.11-3.13 interpreter), and
`cargo test --locked --no-default-features` (pure Rust without Python). `libs/esurge-dashboard` runs standard
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, and `cargo test`.

The accelerator test environment — use it for every computation test (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). `ENABLE_DISTRIBUTED_INIT=0` prevents joining a real distributed
runtime; unsetting `XLA_FLAGS` drops a stray `--xla_force_host_platform_device_count` from the shell:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest <path>
```

The CPU trio (`ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`) is only
for tests that do no array computation, e.g. `libs/eray/tests`, `libs/surgecore/tests`, `libs/eyvan/tests`, parser and
config tests, and tests marked `@pytest.mark.cpu_ok`.

Standard targets:

```bash
uv run lint-imports                                   # layering contract
uv run pre-commit run --all-files                     # per-lib ruff + hygiene
uv run python scripts/check_agents_integrity.py       # verify .agents/ + AGENTS.md + WORKSPACE.md
<accel> uv run pytest libs/easydel/tests -m "not slow" # easydel suite
<accel> uv run pytest libs/spectrax/tests              # spectrax
<accel> uv run pytest libs/eformer/tests               # eformer
<accel> uv run pytest libs/ejkernel/test               # ejkernel (note: test/, not tests/)
<trio>  uv run pytest libs/eray/tests                  # eray: orchestration, no array compute
<trio>  uv run pytest libs/surgecore/tests             # surgecore: Rust/Python host core, cpu_ok
(cd libs/eyvan && uv run --package eyvan --extra dev pytest tests -m fast -q)  # eyvan: control plane, no array compute
uv run python scripts/format_and_generate_docs.py --libs easydel --fix   # format + regenerate API docs
```

`<accel>` is the accelerator environment above, `<trio>` the CPU trio. TPU hosts hold a single libtpu process lock: run
one accelerator test process at a time.

Key runners (repo root `scripts/`): `python -m easydel.scripts.elarge --config <yaml>` (eLarge train/eval/serve runner),
`scripts/convert_hf_to_easydel.py` (plus `scripts/convert_hf_to_easydel_batch.py`),
`scripts/tpu_setup.sh --branch <branch>` (multi-host TPU bootstrap), `scripts/format_and_generate_docs.py`,
`scripts/subtree-sync.sh`, `scripts/release.sh` / `scripts/publish.sh`, and `scripts/check_agents_integrity.py`. The
old root benchmark and probe scripts were removed; write a throwaway harness for one-off measurements instead.

## Architecture in one pass

**Model forward:** `AutoEasyDeLModelForCausalLM.from_pretrained` resolves config+module from the registry
(`easydel/infra/factory.py`) → HF weights convert via `easydel/utils/parameters_transformation.py` (fused QKV/gate-up
via `easydel/layers/layouts/`) → modules are `spx.Module`s built inside a parameter-init sharding context (mesh from
`EasyDeLBaseConfig.mesh`) → attention flows module → `UnifiedAttention`
(`easydel/layers/attention/_unified.py`) → `FlexibleAttentionModule` (`easydel/layers/attention/_flexible.py`) →
`OperationRegistry.create(config.attn_mechanism, metadata)` (`easydel/operations/`) → ejkernel kernel picked per
platform/priority with XLA fallback.

**Training:** a trainer (`easydel/trainers/`, 52 subdirs / 42 registered trainers over `base_trainer.py`) wraps the
model in `EasyDeLState` (params + optimizer + step; `infra/base_state.py`) with eformer `OptimizerFactory` optimizers,
compiles a sharded train step, streams batches from `easydel/data/` `Pipeline` and Grain `data_lanes.py` /
`batch_placement.py`, and checkpoints async via `spectrax.serialization` (TensorStore/OCDBT).

**Serving:** eSurge (`easydel/inference/esurge/`) — continuous-batching scheduler in `surgecore`'s Rust `EngineCore`
(driven by `engine/core_driver.py`) + paged KV cache (page pool and prefix cache in `surgecore`'s `KvCore`, versioned
`page_table_cpu` mirror in `BatchMetadataPreparer`) + compiled bucket runners + OpenAI-compatible FastAPI server +
multi-replica `serving/` cluster/dashboard (`libs/esurge-dashboard`) with tool/reasoning parsers, speculative decoding,
multimodal, and multi-host ZMQ mode.

**Pipeline parallelism:** `sharding_axis_dims[0] > 1` (the `pp` axis) makes `config.mesh` an MPMD `SpxMesh`; layers get
stage assignments (`EasyDeLLayerStackMixin.assign_layer_stage`, `spx.sxstage_iter` boundaries) and spectrax's runtime
compiles per-rank executables with a schedule (GPipe, 1F1B, ZeroBubble, DualPipeV, ...). SPMD path (single jit +
shard_map) exists alongside.

### Where things live (easydel)

| subsystem     | path                                 | anchor points                                                                                                                                                                                                                                                                                           |
|---------------|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| contracts     | `easydel/infra/`                     | `EasyDeLBaseConfig` (base_config.py), `EasyDeLBaseModule` (base_module.py), `EasyDeLState` (base_state.py), `TaskType`/`register_module`/`register_config` (factory.py), `LossConfig`/`LossMetrics` (loss_utils.py), `AxisPolicy`/`RuntimeShardingResolver` (sharding.py), HF bridge (mixins/bridge.py) |
| eLarge        | `easydel/infra/elarge/`              | `eLargeModel` (model.py): one YAML/dict → model+data+trainer+eval+eSurge; runner `easydel/scripts/elarge.py`                                                                                                                                                                                            |
| layers        | `easydel/layers/`                    | `UnifiedAttention`, `FlexibleAttentionModule`, Column/RowParallelLinear + ParallelMoELinear (linears/), fused layouts + reform rules (layouts/), `BaseMoeModule` (moe/), compressed_window, embeddings, engram, indexer, linear_attention, norms, rotary, residual, sparse_attention, quantization          |
| operations    | `easydel/operations/`                | `OperationRegistry`, `BaseOperation` forward_tpu/gpu/native dispatch, `OperationMetadata.get_shardings`, adapters in kernels/ (exemplar: gated_delta_rule.py)                                                                                                                                           |
| model zoo     | `easydel/modules/`                   | 89 families: `<family>/{modeling_<family>.py, <family>_configuration.py}`; task bases in `_base/`; `auto/` factory classes                                                                                                                                                                              |
| caching       | `easydel/caching/`                   | Transformer / RaggedPages / MLARaggedPages / TurboQuantRaggedPages / UnifiedAttention / Recurrent / Hybrid / KDA / CompressedWindow / SharedPrefix caches + `budget.py`                                                                                                                                 |
| trainers      | `easydel/trainers/`                  | `base_trainer.py` + `trainer_protocol.py` + `trainer/trainer.py`; 52 subdirs / 42 registered trainers (`<name>_config.py` + `<name>_trainer.py` + `_fn.py`); `TrainingArguments` (training_configurations.py, ~190+ dataclass fields, JSON roundtrip-tested); `RewardProtocol`; `_online_rl/`; `data_lanes.py` |
| environments  | `easydel/environments/`              | agentic RL sandboxes, harnesses, `async_collector`, `code_judge`, `python_tir`, `swe_repo`, `terminal`, `rewards`                                                                                                                                                                                       |
| inference     | `easydel/inference/`                 | `esurge/` (`esurge_engine.py`, `engine/`, `runners/`, `core/` cache+sampler, `server/`, `serving/`, `distributed/`, `multimodal/`; `scheduler/` holds `output.py` structs only), `tools/` (34 parser files / 37 names), `reasoning/` (16 parser files / 25 names), `parsing/`, `evaluations/`, `vwhisper/` |
| data          | `easydel/data/`                      | `Pipeline` fluent builder (`execution/pipeline.py`), `sources/` (`ParquetShardedSource`, random-access, rowgroup-shuffle), transforms DSL chained with `>>`, `teacher_cache.py`, Ray preprocessing (`distributed/`)                                                                                     |
| utils/workers | `easydel/utils/`, `easydel/workers/` | HF↔EasyDeL conversion, `ejit` caching, `jit_context`, ZMQ workers + auth + response stores                                                                                                                                                                                                              |

## Conventions

- **Imports:** `import typing as tp`, `from jax import numpy as jnp`,
  `import spectrax as spx`; eformer is imported by submodule (`from eformer.loggings import get_logger`,
  `from eformer.pytree import auto_pytree`, `from eformer.escale import ...`).
- **Ruff:** line length 121, target py3.11, rules `A,B,E,F,I,NPY,RUF,UP,W`;
  `modeling_*.py` may use quoted jaxtyping annotations (UP037/F821 ignored there). basedpyright runs report-only
  (`continue-on-error`) scoped to `libs/easydel/easydel/`.
- **Docstrings:** Google style (Args/Returns/Raises).
- **Naming:** configs `FooConfig`, task heads `FooForCausalLM`, trainer dirs
  `snake_case_trainer/` with `FooTrainer`+`FooConfig`, kernels registered snake_case (`"flash_attn2"`), enums are
  StrEnums (`TaskType`, `EasyDeLGradientCheckPointers`).
- **dtype triple:** `dtype` (activation compute), `param_dtype` (storage),
  `precision` (matmul). Softmax accumulates in `attn_softmax_dtype`
  (default float32) — never drop this to bf16.
- **Sharding axes (easydel order):** `("pp","dp","fsdp","ep","tp","sp")` with dims like `(1,1,-1,1,1,1)`; `-1` fills
  remaining devices. eformer's bare `create_mesh` default omits `pp` — easydel's `EasyDeLBaseConfig` owns the 6-axis
  convention (backed by `spectrax.sharding`).
- **File I/O that may touch GCS:** use eformer `ePath`, not `pathlib.Path`.
- **No new `os.environ` knobs.** New toggles/settings are wired through code:
  constructor or config args (a `PartitionAxis` field, a config attribute like `use_fused_mlp`), mesh-scope-style
  context managers backed by ContextVars (`decode_mode_specs()`, `set_inference_mode()`), or explicit setter APIs
  (`set_batch_metadata_dump_dir()`). Delete a knob outright rather than shipping an inert one. Pre-existing documented
  env conventions (e.g. ejkernel's `EJKERNEL_AUTOTUNE_POLICY`, `EJKERNEL_PERSISTENT_CACHE_DIR`)
  may be *followed* where the surrounding code already uses them, but never add new env-var names.
- **Test quality bar** (from `.agents/skills/test-workspace/SKILL.md`): assert public API outputs, numerical parity vs
  independent references, shape/dtype/sharding/cache/checkpoint layout, or state transitions. Reject tests that assert
  private helper calls, log strings, constructors-don't-raise, or compare production logic with itself.

## How to extend (details live in `.agents/skills/`)

- **New model** → `add-easydel-model`: `modules/<family>/` with configuration (`@register_config("<model_type>")`) +
  modeling (`@register_module(TaskType.X, config=..., model_type=...)`); reuse `UnifiedAttention`, ParallelLinears,
  fused layouts with `reform_param` rules for HF conversion; tests under `tests/modules/spmd/test_<family>.py` plus
  conversion roundtrip.
- **New trainer** → `add-easydel-trainer`: copy an adjacent trainer dir shape (`_config.py`, `_trainer.py`, `_fn.py`);
  register both `@Registry.register("trainer-arguments", "<name>")` and `@Registry.register("trainer", "<name>")`;
  loss math gets a dedicated test (see `libs/easydel/tests/trainers/test_distillation_loss_math.py`, or parity vs an
  external reference as in `libs/easydel/tests/trainers/test_trl_dpo_loss_parity.py`).
- **New kernel** → `add-ejkernel-kernel` then `port-ejkernel-to-easydel-operation`: Kernel subclass in
  `ejkernel/modules/operations/`, per-backend impls under `ejkernel/kernels/_{triton,pallas,cuda,cute,tilelang,xla}`
  registered with `@kernel_registry.register(name, Platform.X, Backend.Y, priority=N)` — an XLA fallback (priority 0,
  `Backend.ANY`) is mandatory; then an easydel `OperationImpl` adapter in `easydel/operations/kernels/` declaring
  requirements (`RequirementsBuilder`) and sharding via `metadata.get_shardings()`.
- **New optimizer** → `add-eformer-optimizer`: config dataclass + builder with `@register_optimizer` in
  `eformer/optimizers/`; exposed through `TrainingArguments.optimizer`.
- **SpectraX / eFormer / eRay / SurgeCore** → `spectrax-modules`, `spectrax-sharding`, `spectrax-pipeline-runtime`,
  `spectrax-serialization`, `spectrax-quantization`, `eformer-serialization`, `eray-orchestration`,
  `develop-surgecore`.
- **Data pipelines** → `build-dataset-pipeline`; **eLarge runs** → `train-elarge`; **tool/reasoning parsers** →
  `tool-reasoning-parser` (`@ToolParserManager.register_module` / `@ReasoningParserManager.register_module`).

## Debugging & profiling

- Operational runbooks (TPU bad-node recovery, `eray` CLI, GPU triage, stale ejkernel cache, eSurge symptom routes,
  disk-pressure): [.agents/ops/OPS.md](.agents/ops/OPS.md). Skills: `debug-training-oom`, `debug-tpu-setup`,
  `debug-esurge`, `debug-easydel-cache`, `debug-ejkernel-ops`, `debug-eformer-escale`.
- **NaN/precision:** check softmax dtype promotion first (`attn_softmax_dtype`, `runtime_softmax_dtype`), then
  loss-scale state (eformer mpric `DynamicLossScale` — the step must consume `grads_finite`), then kernel parity vs the
  XLA reference impl by passing `platform="xla"` in the op config (or calling
  `kernel_registry.get(name, Platform.XLA, Backend.ANY)`). Note that `FORCE_NATIVE_RUNTIME=1` only routes
  `BaseOperation.__call__` to `forward_native()`, which on ejkernel-backed adapters like `flash_attention.py` and
  `gated_delta_rule.py` still dispatches to Pallas/Triton/CuTe.
- **OOM at compile:** gradient-checkpointing policy (`EasyDeLGradientCheckPointers`, `auto_remat` save/exclude names),
  `scan_layers=True`, blockwise FFN chunking, loss chunking (`LossConfig.chunk_*`).
- **Recompilation storms & stale page tables (eSurge):** every distinct `(num_tokens, padded_num_reqs)` bucket
  recompiles; pin buckets and standardize multimodal resolutions. Any CPU-side mutation to `page_table_cpu` must bump
  `cpu_version` so `BatchMetadataPreparer` refreshes its version-keyed device mirror (scheduling and commit live in
  `EngineCoreDriver` <-> Rust `surgecore`).
- **Perf claims need hardware numbers:** baseline vs candidate on the target platform, shape/dtype stated, compile-time
  and steady-state separated. For eSurge compare per-bucket profiles grouped by `(num_tokens, padded_num_reqs)`, not
  aggregate tokens/sec. Kernel work: `optimize-ejkernel-kernel` + per-backend skills (`optimize-pallas-tpu`,
  `optimize-triton-gpu`, `optimize-cuda-gpu`, `optimize-cute-gpu`, `optimize-tilelang-gpu`); ejkernel has
  `libs/ejkernel/benchmarks/` with baselines.

## JAX / TPU considerations

- Mutations inside plain `jax.jit` on spx modules are **silently dropped** — use `spx.jit(mutable=...)`. `spx.scan`
  requires structural invariance outside the `mutable` selector.
- MPMD constraints: single positional microbatched input; RNG streams are not staged; markers (`sxstage_iter`) only take
  effect under the MPMD trace.
- Autotuned kernel configs are cached per device+sharding fingerprint, persisted under `~/ejkernel-presistent-cache/`
  (override with `EJKERNEL_PERSISTENT_CACHE_DIR`); a stale cache can mask a regression or carry a bad config across
  code changes.
- TPU hosts hold a single libtpu process lock — while a TPU job runs, only non-computation probes may use
  `JAX_PLATFORMS=cpu`; computation tests wait for the TPU.
- `-1` mesh dims resolve against visible devices; multi-slice needs `sharding_dcn_axis_dims`.
- Donated buffers (`donate_argnums`) in `spx.jit` index the flattened `(states, stripped_args, stripped_kwargs)`
  3-tuple when `mutable` is non-empty, and the user signature on the default readonly path (`donate_argnames` is safest
  when `mutable` is set).

## Common mistakes

- Splitting fused projections with `.reshape()` — under TP the fused axis is rank-interleaved; use
  `split_fused_qkv_projection` / `split_fused_gate_up_projection`.
- Calling `apply_quantization_rules` on `state.model` instead of `model` before `to_state()` — `EasyDeLState.model`
  rebuilds a fresh module from `GraphDef` on every access, silently dropping the rules.
- Forgetting one of the two `Registry.register` decorators on a new trainer, or `_model_type`/`_task_type` on a task
  head.
- Editing generated `docs/api_docs/**` by hand (regenerate via `scripts/format_and_generate_docs.py`).
- Asserting training/serving readiness from constructor-only tests.
- `config.layer_types` not matching the cache config for hybrid models — fails silently or with far-away shape errors.
- GRPO-family divisibility: `total_batch_size` is prompts per step (completions = `total_batch_size * num_generations`);
  enforce EasyDeL's real guards (`generation_batch_size % total_batch_size == 0`, and `dynamic_sampling=True` requires
  `num_generations >= 2`).
- Writing to mirrors directly (breaks subtree sync; reconcile with `scripts/subtree-sync.sh pull <lib>`).

## Review checklist

Before calling a change done (full flow: `.agents/skills/review-pr/SKILL.md`):

1. Layering intact (`uv run lint-imports`), no registry bypasses, no new `os.environ` knobs.
2. Focused tests pass on TPU/GPU (the CPU trio only for non-computation tests); `libs/surgecore` /
   `libs/esurge-dashboard` changes pass cargo fmt/clippy/test; new behavior has a non-tautological test at the right
   layer.
3. Sharding/shape/dtype: partition specs still resolve on 1-device and 8-fake-device meshes; softmax/accumulation dtypes
   preserved.
4. Kernel changes have an XLA-reference parity path (`platform="xla"`) and don't regress other backends' registered
   signatures.
5. eSurge changes checked against cache-shape reuse, `page_table_cpu`/`cpu_version` invalidation, DP page locality, and
   bucket compilation cost.
6. HF conversion round-trips if weight names/layouts changed.
7. Docs/skills updated when public APIs moved (`uv run python scripts/check_agents_integrity.py`); no hand edits to
   generated rst.
8. Commit message clean: imperative, no self-credit trailers; versions/pins untouched unless going through
   `scripts/release.sh`.

## Agent infrastructure

**Canonical location is `.agents/`** for this repository's agent instructions, skills, and workflow documents.
Read and edit files there directly. Automatic loading depends on the active agent harness; these documents do
not themselves register agents, commands, or tools.

- Specialized subagents: `.agents/agents/` (17 agents; roster and flows below).
- Workflow skills: `.agents/skills/` (44 skills; index: `.agents/skills/README.md`) — start non-trivial work from
  `run-research`, then layer the domain skill.
- Task playbooks: `.agents/commands/` — `plan-feature`, `implement-feature`, `review-changes`, `investigate-bug`,
  `explain-subsystem`, `summarize-architecture`, `prepare-release`.
- Long-running task notes: `.agents/projects/<topic>.md`.

### Available agents (`.agents/agents/`)

| agent                 | use when                                                                      | scope                                                                                          |
|-----------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| `architect`           | designing a feature or refactor that spans more than one subsystem or package | decomposes work, names extension points, flags layering/API impact; does not implement         |
| `reviewer`            | a diff/PR needs a correctness review before merge                             | bugs, broken contracts, boundary violations, testing-policy violations; no style nits          |
| `test-engineer`       | choosing what to run, or writing tests for new behavior                       | test selection per package, repo-consistent test authoring, quality bar enforcement            |
| `debugger`            | a failure with an unknown cause                                               | hypothesis → smallest falsifying probe → fix; keeps a paper trail                              |
| `perf-engineer`       | a change might affect speed/memory, or a perf claim needs proof               | compile cost, memory, communication, throughput, MFU; baseline-vs-candidate discipline         |
| `docs-engineer`       | public APIs moved, docs/docstrings need generation or repair                  | Sphinx api_docs regeneration, package docs, README/skill/agent doc sync                        |
| `jax-expert`          | tracing/transform semantics, spx state, donation, jit cache questions         | JAX + spectrax transform correctness                                                           |
| `tpu-expert`          | TPU-specific behavior: Pallas/Mosaic, pods, libtpu locks, multi-host          | TPU execution + eray/gcloud operations                                                         |
| `gpu-expert`          | GPU-specific behavior: CUDA/CuTe/Triton/TileLang/Pallas-GPU/ROCm/MPS          | GPU kernel builds, profiling (`ncu`/`nsys`/sanitizer), NCCL/NVLink, eray GPU swarms            |
| `sharding-expert`     | anything touching Mesh/PartitionSpec/axis policies/PP stages                  | spectrax.sharding + eformer.escale + infra sharding + spectrax MPMD/SPMD pipelines             |
| `kernel-expert`       | writing/porting/fixing a compute kernel                                       | ejkernel modules/backends/registry + easydel operation adapters                                |
| `trainer-expert`      | trainer plumbing, SFT/preference/offline distillation/reward/embedding work   | trainers/ foundation, TrainingArguments, offline loss math, spectrax checkpoint/resume         |
| `rlhf-expert`         | online RL: GRPO family, PPO, SAO, agentic/env trainers, rollouts, rewards     | policy-gradient/agentic/self-distill trainers, RewardProtocol, eSurge rollouts, environments/  |
| `data-expert`         | EasyData pipelines, sources, packing, Grain data lanes, batch placement       | easydel/data/, prompt_transforms, data_lanes.py, batch_placement.py, Ray preprocessing         |
| `inference-expert`    | eSurge engine, Rust surgecore, API server, cluster/dashboard, parsers, speech | inference/ subsystem, libs/surgecore, libs/esurge-dashboard end to end                         |
| `quantization-expert` | quantized weights/caches/matmuls, fused layouts, SpectraX QT/QAT, STE         | layers/quantization, layouts, spectrax.quantization, TurboQuant, eformer jaximus/mpric, qmm    |
| `model-expert`        | adding/porting a model family or fixing HF conversion                         | modules/, caching/, parameters_transformation, auto classes                                    |

### Choosing and combining

- Prefer the **narrowest agent that owns the failing surface**. A sharding error inside a trainer step is
  `sharding-expert` first, `trainer-expert` second.
- Domain experts are consulted **before** implementation when the change touches their invariants (`sharding-expert` for
  anything with a `PartitionSpec`; `quantization-expert` for anything with a fused/quantized layout; `tpu-expert` /
  `gpu-expert` before interpreting accelerator symptoms as code bugs).
- Fan out only when surfaces are independent (e.g., `kernel-expert` on the ejkernel impl while `test-engineer` drafts
  the parity test). Never two agents editing the same files.

### Implementation flow

1. **Scope** — `architect` (or the main session) states the goal, the touched packages, and the extension point; loads
   the matching skill from `.agents/skills/` (`add-easydel-model`, `add-easydel-trainer`, `add-ejkernel-kernel` +
   `port-ejkernel-to-easydel-operation`, `build-dataset-pipeline`, `develop-surgecore`, ...).
2. **Consult** — pull in the domain expert(s) whose invariants the plan touches; adjust the plan, don't discover
   violations after coding.
3. **Implement** — smallest change that tests the hypothesis; follow the skill's "required surfaces" list
   (registrations, tests, docs, exports).
4. **Verify** — `test-engineer` selects the focused tests and runs them on TPU/GPU (CPU only for non-computation
   tests), plus `cargo fmt/clippy/test` for Rust crates, `uv run lint-imports`, and pre-commit. Without an accelerator
   the change is reported as unverified — never validated on CPU.
5. **Review** — `reviewer` on the final diff (flow below).

### Review flow

`reviewer` follows `.agents/skills/review-pr/SKILL.md`: gate (skip closed/draft/trivial) → context (`WORKSPACE.md` +
touched package docs) → parallel compliance and bug passes → independent validation of every finding before it is
reported. Findings must cite file/line and a concrete rule or failure path; style preferences and speculative concerns
are dropped. Escalate a finding to a domain expert instead of guessing: numerics → `kernel-expert` or `jax-expert`;
partition specs → `sharding-expert`; cache shapes or bucket compilation → `inference-expert`; GPU kernels →
`gpu-expert`.

### Escalation rules

- **No accelerator available**: computation tests cannot be run, so stop at "unverified on hardware" or hand to
  `tpu-expert` / `gpu-expert` / `perf-engineer` with a run plan. Never substitute a CPU run or extrapolate CPU timings.
- **Infrastructure symptoms** (bad TPU node, libtpu lock, disk pressure): route to `.agents/ops/OPS.md` before treating
  as a code bug; destructive recovery (deleting TPU VMs) requires explicit user approval.
- **Cross-package API changes** (easydel needs a new spectrax/eformer/ejkernel/surgecore surface): `architect` decides
  which side owns the change; the companion lib gets the general mechanism, easydel gets the composition. Never solve
  it by violating the import contract.
- **Performance regressions found during review**: block only with a measured or clearly-reasoned regression path
  (`perf-engineer`), otherwise note-and-pass.
- **Anything touching release/publish/mirrors**: follow `release-workspace` and `prepare-commit-pr` skills; tags and
  pushes are user-approved actions.
