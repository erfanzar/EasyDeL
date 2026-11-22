# Changelog

All notable changes to spectrax are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-05

### Added

- `sxstage_region` for multimodal and branched MPMD pipelines, allowing
  independent logical stage sequences inside one scheduled function.
- Region-aware MPMD schedule planning for GPipe, 1F1B, ZeroBubble,
  interleaved, KimiK2, and DualPipeV schedules.
- Runnable examples for `sxstage_region`, stage-local MPMD mesh
  layout, and `MultiOptimizer` LoRA training.
- Optimizer guide covering `Optimizer`, `MultiOptimizer`, selector
  scoped optimizer state, and mutable collection rules.

### Changed

- `sxcall`, `sxjit`, `sxgrad`, `sxvalue_and_grad`, and MPMD `spx.run`
  now route through the true scheduled MPMD path instead of legacy
  pipeline shortcuts.
- Stage-local mesh and sharding handling now drops the pipeline axis
  at stage boundaries and preserves only the local SPMD sub-mesh.
- MPMD stage JITs now use cache-visible stage/rank names and guarded
  first-compile behavior so persistent executable caches cannot be
  reused across incompatible stage meshes.

### Fixed

- Rebased nested `shard_map` / `pjit` mesh metadata inside split stage
  JAXPRs so per-stage executables do not close over full-pipeline meshes.
- Fixed stale persistent-cache reuse for MPMD MoE / expert-parallel
  stage executables.
- Fixed scalar zero-axis mesh rebasing during scheduled training plan
  construction.
- Fixed boundary sharding/layout handling that could trigger large
  unintended KV-cache copies in pipeline inference.
- Fixed `MultiOptimizer` slicing for nested dotted paths such as LoRA
  adapters under child modules.

## [0.0.1] — 2026-04-20

### Added

- **True MPMD pipeline parallelism.** `sxcall`, `sxjit`, `sxgrad`, `treduce`,
  `pscan_compiler` — each physical rank compiles and executes its own distinct
  JAX program. Heterogeneous stages (different class/shape per rank) are native.
- **9 pipeline schedules:** GPipe, Std1F1B, Eager1F1B, ZeroBubbleH1,
  InterleavedH1, InterleavedGPipe, Interleaved1F1BPlusOne, KimiK2, DualPipeV.
- **Unified runtime** — `spx.run(mesh)` dispatches to SPMD (`pjit`) or MPMD
  (`sxcall`) based on the mesh. Same model, same script; change the mesh,
  change the parallelism strategy.
- **Dynamic deferred parameter initialization.** Built-in layers (`Linear`,
  `Conv1d`/`2d`/`3d`, `ConvTranspose1d`/`2d`/`3d`, `Embed`) accept `None` for
  input dimensions. Shape is inferred from the first forward call and the
  initializer runs eagerly. Replaces static `LazyLinear` / `LazyEmbed` /
  `LazyConv*` classes.
- `DeferredParameter` and `DeferredBuffer` — core `Variable` subclasses that
  store an initializer + shape spec. Usable directly in custom layers.
- `Module._resolve_deferred()`, `Module.materialize()`,
  `Module.sequential_init(*examples)` for explicit lazy-init control.
- `LazyInitUnderTransformError` — raised when deferred parameters try to
  materialize inside `jax.jit`, `jax.vmap`, or `jax.scan`.
- Eager `Module` API — subclass `Module`, override `forward`, call `model(x)`.
  Modules are JAX pytrees; `jax.jit`, `jax.grad`, `jax.tree.map` work directly.
- Module-aware transforms — `spx.jit`, `spx.grad`, `spx.vmap`, `spx.scan`,
  `spx.remat`, `spx.cond`, `spx.switch`, `spx.while_loop`, `spx.fori_loop`.
- Selector DSL — one composable predicate for `grad(wrt=...)`,
  `jit(mutable=...)`, `Optimizer(wrt=...)`, `freeze(...)`, etc.
- Dynamic scope — `spx.scope(**values)` threads context through call stacks.
- Built-in layers — Linear, Conv*d, ConvTranspose*d, Attention, LayerNorm,
  RMSNorm, BatchNorm, InstanceNorm, GroupNorm, Dropout, Embed, MLPBlock,
  RNN/GRU/LSTM cells, containers.
- LoRA fine-tuning — `wrap_lora`, `LoRALinear`, `LoraParameter`.
- FP8 training — delayed-scaling per-tensor FP8 with rolling amax history.
- Sharding — logical axis names, `jax.sharding.Mesh`-based SPMD.
- Inspection — `spx.inspect.summary`, `tabulate`, `count_params`, `count_bytes`.
- optax integration — `spectrax.contrib.Optimizer`, `MultiOptimizer`.
