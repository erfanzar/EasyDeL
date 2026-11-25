---
name: port-ejkernel-to-easydel-operation
description: Add, update, or wire EasyDeL operations under libs/easydel/easydel/operations as registered OperationImpl adapters over ejkernel.modules public operations, or extend EasyDeL's operation framework (OperationImpl, OperationRegistry, OperationMetadata, OperationExecutor, RequirementsBuilder). If the hardware kernel core still lives in EasyDeL or does not exist in ejkernel yet, use add-ejkernel-kernel first, then use this skill for the EasyDeL adapter.
---

# Skill: EasyDeL Operations & eJKernel-To-EasyDeL Adapters

Specialization of `.agents/skills/run-research/SKILL.md` for `libs/easydel/easydel/operations/` — both **wiring
ejkernel public operations into EasyDeL `OperationImpl` adapters** and **maintaining EasyDeL's operation framework**
(`OperationImpl`, `OperationRegistry`, `OperationMetadata`, `OperationExecutor`, `RequirementsBuilder`,
`ModeSpecificBuilder`, `RequirementsValidator`).

Hardware kernels (Pallas, Triton, CUDA, CuTe, TileLang, and mandatory XLA fallbacks) ALWAYS live in `libs/ejkernel/`
(`.agents/skills/add-ejkernel-kernel/SKILL.md`). Never inline hardware kernels or import private `ejkernel.kernels.*`
paths inside `libs/easydel/easydel/operations/kernels/`.

## Required Mental Model

- `Platform` (`libs/ejkernel/ejkernel/kernels/_registry.py`): `XLA`, `PALLAS`, `TRITON`, `CUDA`, `CUTE`, `TILELANG`.
- `Backend`: `CPU`, `GPU`, `MPS`, `TPU`, `ANY`.
- EasyDeL adapters adapt EasyDeL runtime state (dtype, mesh, sharding, masks, attention sinks, cache views, metadata,
  and output containers) to public `ejkernel.modules` / `ejkernel.modules.operations` entry points.

Before editing, state the EasyDeL operation class, `get_impl_name()` key(s), public `ejkernel.modules` op, required
`MetadataField`s / `CacheType`s, and target `(Platform, Backend)` pairs.

## First Reads

Framework files:
- `WORKSPACE.md`, `libs/easydel/pyproject.toml`, `libs/ejkernel/pyproject.toml`
- `libs/easydel/easydel/operations/_base_operation.py` (`BaseOperation.__call__`, `OperationRegistry`)
- `libs/easydel/easydel/operations/_operation_impl.py` (`OperationImpl`, `get_instance_requirements`, `dp_page_axis`,
  `expand_kv_heads_for_shard`, `create_stable_sharding`, `run_mosaic_kernel`)
- `libs/easydel/easydel/operations/_operation_meta.py` (`OperationMetadata`, `AttnShardingRules`, `normalize_head_sinks`)
- `libs/easydel/easydel/operations/executor.py` (`OperationExecutor`)
- `libs/easydel/easydel/operations/requirements/{types,requirements,builder,validation}.py`
- `libs/easydel/easydel/operations/kernels/__init__.py`

Diverse multi-backend adapter exemplars (pick the closest to your cache/output shape):
- `libs/easydel/easydel/operations/kernels/fused_mlp.py` — cache-free (`requires_cache(False)`),
  `FORCE_NATIVE_RUNTIME=1 -> platform="xla"`, plain array output.
- `libs/easydel/easydel/operations/kernels/gated_delta_rule.py` — multi-backend linear attention with `RecurrentCacheView`
  and custom `AttentionOutput` subclass.
- `libs/easydel/easydel/operations/kernels/inference_gdn.py` — continuous-batching ragged adapter with
  `FORCE_NATIVE_RUNTIME=1` propagation and `get_shardings()`.
- `libs/easydel/easydel/operations/kernels/ragged_page_attention.py` — multi-backend paged attention + specialized
  config lookup (e.g. `"ragged_page_attention_v3_turboquant"`).
- `libs/easydel/easydel/operations/kernels/flash_attention.py` / `ring_attention.py` — standard attention adapters
  using `normalize_head_sinks()`. (Note: `paged_flash_attention.py` is GPU-only — do not copy its TPU `NotImplementedError`
  for multi-backend ops.)

## Operation Framework & Adapter Contracts

1. **Always subclass `OperationImpl` (never `BaseOperation` directly):** every concrete operation under
   `operations/kernels/` subclasses `OperationImpl`, which supplies `__init__`, `get_impl_metadata()`,
   `get_instance_requirements()` (required for instance-level `requires_cache` overrides via `OperationRegistry.create`),
   `get_mode()`, `create_stable_sharding()`, `expand_kv_heads_for_shard()`, and mask helpers.
2. **Mandatory import-time registration:** decorate the class with `@OperationRegistry.register` AND import + export it
   in `libs/easydel/easydel/operations/kernels/__init__.py` (and `operations/__init__.py` if top-level public). Without
   the import in `kernels/__init__.py`, `@OperationRegistry.register` never executes.
3. **`BaseOperation.__call__` dispatch (`_base_operation.py`):**
   - If `check_bool_flag("FORCE_NATIVE_RUNTIME", False)` is true → `forward_native`.
   - Else matches `self.metadata.backend`: `TPU -> forward_tpu`, `GPU -> forward_gpu`, `TT -> forward_tt`,
     `CPU -> forward_native`.
   - Default `forward_tpu`, `forward_gpu`, and `forward_tt` already delegate to `forward_native`.
   - **`forward_cuda`, `forward_rocm`, and `forward_cpu` are NEVER called by `BaseOperation.__call__`** — do not add
     dead stub overrides for them. Only override `forward_tpu`/`forward_gpu` when EasyDeL-side handling genuinely differs.
4. **`FORCE_NATIVE_RUNTIME=1 -> platform="xla"` contract:** because ejkernel-backed adapters delegate to
   `ejkernel.modules` inside `forward_native`, `FORCE_NATIVE_RUNTIME=1` will silently keep dispatching to
   Pallas/CuTe/CUDA/Triton unless `forward_native` checks `check_bool_flag("FORCE_NATIVE_RUNTIME", False)` and passes
   `platform="xla"` (see `fused_mlp.py`, `inference_gdn.py`, `dual_stream_attention.py`).
5. **Requirements & `OperationExecutor`:**
   - Declare `get_requirements(mode)` with `RequirementsBuilder` or `ModeSpecificBuilder` (`requirements/builder.py`),
     validated via `RequirementsValidator` (`requirements/validation.py`).
   - Cache-free ops use `.require_metadata(MetadataField.NONE).support_cache(CacheType.any()).requires_cache(False)`.
   - `OperationExecutor` (`executor.py`) resolves `PREFILL -> prefill_impl or mixin_impl`,
     `DECODE -> decode_impl or prefill_impl or mixin_impl`, and aggregates requirements via
     `get_instance_requirements()`.
6. **Sharding, Sinks & Paged Helpers:** use `self.metadata.get_shardings()`, `create_stable_sharding()`,
   `normalize_head_sinks()` (`_operation_meta.py`), `expand_kv_heads_for_shard()`, `dp_page_axis()`, and
   `spectrax.with_sharding_constraint`.
7. **`get_impl_name()` and operation configs:** `get_impl_name()` returns a `str` or a `tuple[str, ...]` of aliases
   (e.g., `MultiLatentRaggedPageAttn`). Read configs via `self.metadata.get_operation_config("<key>")` — adapters may
   look up specialized keys (e.g., `ragged_page_attention.py` looking up `"ragged_page_attention_v3_turboquant"`).

## What Is Forbidden (And The Helper Exception)

Do NOT put any of these in an EasyDeL `OperationImpl` adapter:
- Bare function re-export stubs pretending to be an operation (`from ejkernel.modules.operations.foo import foo`).
- Private backend or direct-registry imports: `ejkernel.kernels` (`from ejkernel.kernels import ...`,
  `ejkernel.kernels._pallas`, `_xla`, `_triton`, `_cuda`, `_cute`, `_tilelang`). Legacy direct-registry calls (e.g.,
  `kda.py`) are not allowed for new ports.
- `jax.experimental.pallas`, `pallas_call`, `pl.`, or algorithm bodies that belong in `libs/ejkernel/`.

**Helper exception:** a non-operation compatibility helper (like `gdn_compute_schedule_v2.py`) may re-export a public
`ejkernel.modules.operations` helper if it is named as a helper, kept out of `OperationRegistry`, and verified free of
`ejkernel.kernels` imports.

## Port Workflow & Required Structure Checks

1. Inventory call sites across `libs/easydel` and `libs/ejkernel`. If the ejkernel public op is missing, stop and run
   `.agents/skills/add-ejkernel-kernel/SKILL.md` first.
2. Implement the `OperationImpl` subclass in `libs/easydel/easydel/operations/kernels/<op>.py`, wire
   `FORCE_NATIVE_RUNTIME=1 -> platform="xla"`, and register it via `kernels/__init__.py`.
3. Update model/layer/runner call sites to use the registered op; delete obsolete in-EasyDeL algorithm bodies once tests
   pass.
4. Run structure checks (first grep MUST return zero hits; second and third must show the adapter & public import):

```bash
rg -n "ejkernel\.kernels\b|_pallas|_xla|_triton|_cuda|_cute|_tilelang|pallas_call|jax\.experimental\.pallas|\bpl\." \
  libs/easydel/easydel/operations/kernels/<op>.py

rg -n "OperationRegistry\.register|class .*\(OperationImpl\)|get_impl_name|get_requirements|forward_native" \
  libs/easydel/easydel/operations/kernels/<op>.py

rg -n "from ejkernel\.modules|import ejkernel\.modules" \
  libs/easydel/easydel/operations/kernels/<op>.py
```

Also run `uv run lint-imports` to confirm package-layering compliance.

## Routing

- Missing or new ejkernel backend kernel: `.agents/skills/add-ejkernel-kernel/SKILL.md`
- Kernel tuning / profiling: `.agents/skills/optimize-ejkernel-kernel/SKILL.md`
- Cache view / metadata integration: `.agents/skills/debug-easydel-cache/SKILL.md`
- eSurge serving integration: `.agents/skills/debug-esurge/SKILL.md`

## Verification

Distinguish pure-Python contract tests (`@pytest.mark.cpu_ok`) from accelerator numeric tests:

```bash
# Pure-Python contract tests only (registry, requirements, metadata — zero JAX numerics)
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest \
    libs/easydel/tests/operations/test_requirements.py \
    libs/easydel/tests/operations/kernels/test_operation_contracts.py \
    libs/easydel/tests/operations/test_paged_common_cpu.py

# Accelerator numeric/parity tests (TPU; for GPU use JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest \
    libs/easydel/tests/operations/<focused-test>.py \
    libs/ejkernel/test/modules/operations/test_<op>.py \
    libs/ejkernel/test/kernels/_pallas/tpu/test_<op>.py

# GPU backend parity (swap _<backend> for _triton, _cuda, _cute, _tilelang)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu \
  uv run pytest \
    libs/easydel/tests/operations/<focused-test>.py \
    libs/ejkernel/test/kernels/_<backend>/test_<op>.py
```

All numeric/parity tests MUST run on TPU or GPU. Never present a CPU run as correctness, lowering, or performance
evidence; without an accelerator, report numeric verification as **unverified on hardware**.

## Final Report

Include: adapter path, class name, `get_impl_name()` key(s), public `ejkernel.modules` op, metadata/cache/dtype/config/
sharding mapping, `FORCE_NATIVE_RUNTIME=1` behavior, private-import grep result, `lint-imports` result, and tests run
with exact hardware/backend (or **unverified on hardware** for any unrun accelerator checks).
