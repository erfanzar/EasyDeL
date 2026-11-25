---
name: spectrax-modules
description: Work on SpectraX's core object model, neural-network layers, functional primitives, and module-aware JAX transforms under libs/spectrax/spectrax/{core,hooks,nn,functional,transforms}. Use for Module, Variable, Parameter, Buffer, GraphDef, State, StateCallABI, Selector, Policy, containers, scan_prefetch, spx.scope, hooks, nn layers, functional ops, or spx.{jit,grad,vmap,scan,remat,cond,fori_loop,eval_shape}.
---

# Skill: Work On SpectraX Modules, Layers, And Transforms

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

Use this skill for work inside `libs/spectrax/spectrax/{core,hooks,nn,functional,transforms,init,rng}`: module/state
semantics, graph export/bind round-trips, containers and weight-prefetched scans, layers/functional primitives, or
module-aware JAX transforms and control flow.

## Package Boundary

`libs/spectrax` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `ejkernel`, `eformer`, `eray`, or `surgecore`. Verify with `uv run lint-imports`.

## First Reads

- Core object model: `libs/spectrax/spectrax/core/{module.py,variable.py,graph.py,state.py,selector.py,policy.py,lazy_init.py,stage_assignment.py,sharding.py,errors.py}`
- Containers, scope, prefetch, hooks: `libs/spectrax/spectrax/core/{containers.py,context.py,scan_prefetch.py}`, `libs/spectrax/spectrax/hooks/{__init__.py,forward.py,variable.py}`
- Layers & functional ops: `libs/spectrax/spectrax/nn/{__init__.py,_quant.py,linear.py,dense.py,attention.py,norm.py,pipeline_sequential.py}`, `libs/spectrax/spectrax/functional/__init__.py`
- Transforms & control flow: `libs/spectrax/spectrax/transforms/{__init__.py,split_merge.py,jit.py,grad.py,vmap.py,scan.py,remat.py,rng_axes.py,control_flow.py,eval_shape.py}`
- Reference docs: `libs/spectrax/docs/design.md`, `libs/spectrax/pyproject.toml`

## Core Object Model Invariants

1. **Module attribute discipline (`libs/spectrax/spectrax/core/module.py`):** every assigned attribute must be a child
   `Module` (including containers), a `Variable` (`Parameter`/`Buffer`), a `Static` marker or immutable hashable scalar
   (enters `GraphDef`), an `Opaque` escape hatch, or a `_`-prefixed private attribute. Single-underscore attributes land
   in `_spx_opaque` and **survive** `export`/`bind` (e.g., `_quant_plan`); `_spx_`-prefixed attributes are private
   instance slots and are **not** exported.
2. **PyTree registration:** BOTH `Module` (`libs/spectrax/spectrax/core/module.py`, `_register_module_pytree`) and
   `Variable` + every subclass (`libs/spectrax/spectrax/core/variable.py`, `_auto_register_init_subclass`) ARE
   registered via `jax.tree_util.register_pytree_with_keys` (despite a stale comment at the top of `module.py`).
   - Do **not** pass a live `Module` to plain `jax.jit` expecting mutations to stick — mutations inside plain `jax.jit`
     are silently dropped. Always use `spx.jit(mutable=...)`.
3. **Graph epochs (`libs/spectrax/spectrax/core/module.py`):**
   - `_GRAPH_EPOCH` (global counter) invalidates cached `(epoch, gdef, var_paths)` snapshots on `export()`; writing
     `var.value = ...` does **not** bump it, keeping training-loop hot paths cached.
   - `_THREAD_GRAPH_EPOCH` (thread-local counter) is what transforms check for structural mutations so concurrent
     threads constructing modules never trigger false `IllegalMutationError`.
4. **Variables, sharing, and stage hints (`libs/spectrax/spectrax/core/{variable.py,stage_assignment.py}`):**
   - `KINDS_BUILTIN = ("parameters", "buffers", "batch_stats", "cache", "intermediates", "rng")`.
   - Shared/tied weights are represented by the **same `Variable` instance (`ref_id`)** at multiple paths; `GraphDef`
     normalizes `ref_id`s to `0..N-1` and records `shared_paths`, and `bind()` allocates fresh `ref_id`s while
     preserving sharing topology.
   - `DeferredParameter` / `DeferredBuffer` infer shape on first eager forward (unless under `lazy_init()`, which
     requires explicit `module.init(...)` / `sequential_init(...)`); materializing inside any transform raises
     `LazyInitUnderTransformError`.
   - `variable_init_placement(hook)` intercepts new variable initialization after dtype coercion and before default mesh
     placement.
   - `assign_stage(total=..., current=...)` stamps `Variable.metadata["pipeline_stage"] = (current, total)` for
     subclasses with `inherit_stage_assignment = True`. Use `strip_pipeline_stage_metadata(gdef)`
     (`libs/spectrax/spectrax/core/graph.py`) when building layer-stack templates spanning multiple stages.
5. **Flat call ABI (`libs/spectrax/spectrax/core/state.py`):** `StateCallABI` (`state.call_abi()` /
   `state_call_abi(state)`) caches `treedef` so latency-sensitive loops (e.g., decode) pass flat leaf tuples through
   `jax.jit`; use `abi.flatten_sharding(...)` (which preserves `None` leaves) for `in_shardings`.
6. **Containers & weight-prefetched scans (`libs/spectrax/spectrax/core/{containers.py,scan_prefetch.py}`):**
   - `Sequential`, `ModuleList`, `StackedModuleList`, `ModuleDict`, `ParameterList` live in
     `libs/spectrax/spectrax/core/containers.py` (only re-exported in `libs/spectrax/spectrax/nn/__init__.py`).
     `PipelineSequential` lives in `libs/spectrax/spectrax/nn/pipeline_sequential.py`.
   - `ModuleList.scan(prefetch_axes=...)` lowers to `scan_with_weight_prefetch`
     (`libs/spectrax/spectrax/core/scan_prefetch.py`): a single `jax.custom_vjp` (reverse-mode only) overlapping
     next-slot all-gathers (2-D floating leaves mentioning `prefetch_axes` and not in `exclude`) with layer compute and
     constraining weight cotangents back to sharded specs.
7. **Scope & hooks (`libs/spectrax/spectrax/core/context.py`, `libs/spectrax/spectrax/hooks/`):**
   - `spx.scope(**values)` / `spx.scope.get(key)` uses `contextvars`; under `spx.jit`, array-like values are lifted as
     traced inputs while static scalars fold into the compile key.
   - Forward hooks (`register_forward_pre_hook`, `register_forward_hook`) and `register_variable_hook` fire only in
     eager mode and are suppressed under transforms — use `self.sow("intermediates", ...)` inside transformed code.

## Layers, Functional Ops, And Quantization Integration

When adding or editing layers under `libs/spectrax/spectrax/nn/` or primitives under `libs/spectrax/spectrax/functional/`:

1. **Exports:** add pure JAX primitives in `libs/spectrax/spectrax/functional/<mod>.py` AND export them from
   `libs/spectrax/spectrax/functional/__init__.py`; export new `Module` classes from
   `libs/spectrax/spectrax/nn/__init__.py`.
2. **Layouts:** channels-last `(N, *spatial, C)` for conv/pool; sequence-second `(N, T, ...)` for attention/RNN.
3. **Logical `axis_names` & sharding kwargs:** accept `sharding` / `bias_sharding` and attach canonical `axis_names`:
   - `Linear`: `("in", "out")` weight, `("out",)` bias; `Bilinear`: `("in1", "in2", "out")`
   - `Conv*`: `(*"k" * rank, "in", "out")`
   - `Embed`: `("vocab", "embed")`
   - Norms (`LayerNorm`, `RMSNorm`, `GroupNorm`, `InstanceNorm`, `BatchNorm*`): `("features",)` or `("channels",)`
4. **Dtype & `Policy` contract (`libs/spectrax/spectrax/core/policy.py`):**
   - Storage dtype precedence in constructors: `param_dtype or dtype or jnp.float32`.
   - In `forward`, consult `current_policy()` and downcast inputs/weights to `policy.compute_dtype` before contractions;
     `policy.output_dtype` is applied automatically by `Module.__call__`.
5. **Quantization bridge (`libs/spectrax/spectrax/nn/_quant.py`):** matmul/contraction layers
   (`libs/spectrax/spectrax/nn/{linear.py,dense.py}`) MUST obtain their contraction via `quantized_dot_general_for(self)`
   / `quantized_einsum_for(self)` so `spx.quantization.quantize_model` intercepts them with zero overhead when
   unquantized.
6. **Stateful & specialized layers:** `MultiheadAttention(decode=True)` stores KV caches in `Buffer(..., kind="cache")`
   requiring `mutable="cache"` under transforms. Also inspect existing `libs/spectrax/spectrax/nn/lora.py`
   (`LoraParameter`, `wrap_lora`), `libs/spectrax/spectrax/nn/fp8.py` (`Fp8Linear`, `Fp8Meta`), and
   `libs/spectrax/spectrax/nn/recurrent.py` before inventing parallel mechanisms.

## Module-Aware Transforms And Control Flow

All transforms in `libs/spectrax/spectrax/transforms/` build on the 5-step `split_merge.py` pipeline
(`locate_and_strip` -> `export` -> `make_pure` -> JAX transform -> `apply_mutations`):

- **`spx.jit` (`libs/spectrax/spectrax/transforms/jit.py`):**
  - **Signature rewrite trap:** when `mutable=` is non-empty, the compiled function's argument tuple becomes
    `(states, stripped_args, stripped_kwargs)`, shifting positional `static_argnums`, `donate_argnums`, `in_shardings`,
    and `out_shardings`. Prefer `*_argnames` whenever possible.
  - Read-only fast path (`mutable=()` with no kwargs) caches on `(layout_key, gdef_key, id_key)` using `_graph_epoch()`.
  - Passing an MPMD `SpxMesh` (`mesh.is_mpmd == True`) routes automatically to `spectrax.runtime.mpmd.sxjit`.
  - Integrates `spx.scope`: traced scope arrays become inputs; static scope scalars enter the compile key.
- **`spx.grad` / `value_and_grad` / `jvp` / `vjp` (`libs/spectrax/spectrax/transforms/grad.py`):** differentiate over
  `State` leaves; undeclared writes raise `IllegalMutationError`.
- **`spx.vmap` / `spx.scan` / `spx.associative_scan` (`libs/spectrax/spectrax/transforms/{vmap.py,scan.py,rng_axes.py}`):**
  - `spx.vmap` always broadcasts modules with `in_axes=None` and extracts mutations with `out_axes=None`.
  - **RNG splitting caveat:** use `split_rngs` / `split_stream_keys` (`rng_axes.py`) to split PRNG streams across mapped
    axes. `StateAxes` is a declarative dataclass in `rng_axes.py` that is **NOT yet wired** into `spx.vmap` or
    `spx.scan` parameters — do not pass it to them or assume automatic `"split"` handling.
  - `spx.scan` partitions state via `mutable` into `carry` + `invariant` and rejects any invariant structural/value
    change. `spx.associative_scan` has no carry and raises `ValueError` on any non-empty `mutable=`.
- **`spx.remat` (`libs/spectrax/spectrax/transforms/remat.py`):** supports both function wrapping and cached
  `Module`-subclass wrapping (`_REMAT_CLASS_CACHE`, registered in the parent class's module namespace for
  `export`/`bind` round-trips). Non-traceable kwargs (`str`, `bool`, and nested containers thereof) are automatically
  closed over statically.
- **Control flow & abstract eval (`libs/spectrax/spectrax/transforms/{control_flow.py,eval_shape.py}`):**
  - `cond` / `switch` require matching output pytrees AND identical `(collection, path)` mutation footprints across
    branches.
  - `fori_loop` pre-computes and injects `_spx_traced_cache` on nested `ModuleList` instances to avoid per-iteration
    re-stacking.
  - `eval_shape` runs `make_pure` under `jax.eval_shape` and intentionally **discards** `new_states` (read-only).

## Routing

- Sharding, meshes, `strict_sharding`, `audit_placement`: `.agents/skills/spectrax-sharding/SKILL.md`
- MPMD/SPMD pipeline schedules & `sxjit`/`sxvalue_and_grad`: `.agents/skills/spectrax-pipeline-runtime/SKILL.md`
- Quantization rules & QAT/PTQ numerics: `.agents/skills/spectrax-quantization/SKILL.md`
- TensorStore/OCDBT checkpointing: `.agents/skills/spectrax-serialization/SKILL.md`
- JAX transform semantics deep dive: `jax-expert` agent

## Verification

Computation tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware** (never CPU-validated).

```bash
uv run lint-imports

# Core, hooks, and API conformance
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/core/ libs/spectrax/tests/hooks/ \
    libs/spectrax/tests/test_conformance.py libs/spectrax/tests/test_api_regressions.py libs/spectrax/tests/test_lint.py

# Layers, functional primitives, and quantization integration (catches bypassed nn/_quant.py)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/nn/ libs/spectrax/tests/functional/ libs/spectrax/tests/quantization/

# Transforms (fast suite; add test_dynamic_matrix_slow.py when changing split_merge/jit/scan semantics)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/transforms/ --ignore=libs/spectrax/tests/transforms/test_dynamic_matrix_slow.py
```

Key targeted transform tests: `test_split_merge.py`, `test_rng_axes.py`, `test_thread_graph_epoch.py`,
`test_control_flow.py`, `test_eval_shape.py`, `test_dynamic_matrix.py`.
Pure-Python `@pytest.mark.cpu_ok` subsets exist in `libs/spectrax/tests/core/test_containers.py` and
`libs/spectrax/tests/core/test_module_list_scan_lowerings.py` (enforced by `libs/spectrax/tests/conftest.py`).
