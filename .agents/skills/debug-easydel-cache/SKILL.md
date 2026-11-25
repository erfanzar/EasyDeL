---
name: debug-easydel-cache
description: Debug or extend EasyDeL inference caches under libs/easydel/easydel/caching. Use for KV cache, paged caches, MLA, recurrent, KDA, compressed window, hybrid, shared prefix, unified attention, TurboQuant, cache budgeting, metadata builders, or cache-shape failures in eSurge.
---

# Skill: Debug Or Extend EasyDeL Caching

This is a specialization of `.agents/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the failure or feature involves
`libs/easydel/easydel/caching`, cache budgeting/metadata, or an eSurge cache-shape error.

## First Reads

Read the relevant surfaces before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- Core abstractions, metadata builder, and budgeting:
  - `libs/easydel/easydel/caching/_abstracts.py`
  - `libs/easydel/easydel/caching/_metadatabuilder.py`
  - `libs/easydel/easydel/caching/budget.py`
- eSurge KV-cache specifications (specs live in eSurge, not in `easydel/caching/`):
  - `libs/easydel/easydel/inference/esurge/core/interface.py`
- Cache families:
  - `libs/easydel/easydel/caching/transformer/cache.py`
  - `libs/easydel/easydel/caching/ragged_page/cache.py` and `utils.py`
  - `libs/easydel/easydel/caching/mla_ragged_page/cache.py`
  - `libs/easydel/easydel/caching/turboquant_ragged_page/cache.py`
  - `libs/easydel/easydel/caching/compressed_window/cache.py` (DeepSeek-V4 CSA/HCA + Lightning Indexer)
  - `libs/easydel/easydel/caching/recurrent/cache.py`
  - `libs/easydel/easydel/caching/kda/cache.py`
  - `libs/easydel/easydel/caching/hybrid/cache.py`
  - `libs/easydel/easydel/caching/shared_prefix/recurrent.py` (`PrefixRecurrentCaptureView`, `PrefixRecurrentForkView`)
  - `libs/easydel/easydel/caching/unified_attention/cache.py`

## Architecture & Invariants

- **Cache views vs. runtime metadata (`_abstracts.py`):** `BaseCache`, `BaseCacheView`, and `BaseCacheConfig` own the
  state tensors and per-layer cache updates. `BaseRunTimeMetadata` subclasses (`TransformerMetadata`, `HybridMetadata`,
  `RaggedPagesMetadata`, `RecurrentMetadata`, `CompressedWindowMetadata`, `KDAMetadata`) hold per-step runtime metadata.
  `OperationsMetadata` is a fixed `@auto_pytree` wrapper around runtime metadata (`transformer`, `hybrid`, `ragged`,
  `recurrent`) with `isinstance` dispatch in `OperationsMetadata.wrap()` and `unwrap_metadata()` — it is **not** a view
  registry.
- **Cache specs live in eSurge (`esurge/core/interface.py`):** `CacheSpec`, `AttentionSpec`, `FullAttentionSpec`,
  `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`, `CacheGroupSpec`, and `create_kv_cache_specs_from_config` define
  cache grouping and page-pool requirements for eSurge.
- **`AttentionMetadataBuilder` is CPU-first NumPy prep (`_metadatabuilder.py`):** computes both v2 (`slot_mapping` +
  `num_kv_update_slices`) and v3 (`request_distribution`) layouts on host NumPy arrays. Never call its raw-batch
  compute path inside `jax.jit`.
- **Hybrid cache / `config.layer_types` alignment:** `HybridCache` and `ParallelHybridCacheView` depend on
  `config.layer_types` and `config.get_mask_details()` matching the model's layers; mismatches fail silently or surface
  as distant shape errors.
- **DP-aligned page budgeting (`budget.py`):** use `compute_paged_cache_budget`, `cap_paged_cache_pages`, and
  `PagedCacheBudget` so page totals stay aligned with DP ranks and HBM utilization caps.
- **Cold-import order safety:** `easydel.caching` imports `easydel.layers.quantization`, while `easydel.layers.indexer`
  and `easydel.operations` import `easydel.caching`. Keep cross-package imports lazy where cycles can form and verify
  with `libs/easydel/tests/layers/test_import_safety.py`.

## Routing

- eSurge serving failure, runner prepare-signature mismatch, or DP page placement:
  `.agents/skills/debug-esurge/SKILL.md`.
- Operation adapter consuming the cache (paged/flash/GDN/MLA):
  `.agents/skills/port-ejkernel-to-easydel-operation/SKILL.md` (hardware kernels:
  `.agents/skills/add-ejkernel-kernel/SKILL.md`).
- TurboQuant compressed cache: `.agents/skills/ejkernel-quantization/SKILL.md` (bitpacking, formats, kernels) and
  `.agents/skills/quantization-layout/SKILL.md` (`TurboQuantConfig` / `TurboQuantConstants` EasyDeL wiring).

## Verification

Host-only contract/budget/metadata tests (marked `cpu_ok`; CPU trio is allowed):

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/caching/test_operations_metadata.py \
                libs/easydel/tests/caching/test_ragged_cache_budget_and_init.py \
                libs/easydel/tests/caching/test_transformer_mask_shapes.py \
                libs/easydel/tests/caching/test_metadatabuilder_slot_mapping.py \
                libs/easydel/tests/caching/test_group_metadata_and_budget.py \
                libs/easydel/tests/caching/test_cache_host_contracts.py \
                libs/easydel/tests/inference/esurge/test_cache_spec_grouping.py \
                libs/easydel/tests/layers/test_import_safety.py
```

Cache layout and numerical computation tests must run on an accelerator (TPU shown; GPU:
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as unverified on
hardware:

```bash
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/caching/test_ragged_page_kv_layout.py \
                libs/easydel/tests/caching/test_transformer_metadata_compute.py \
                libs/easydel/tests/caching/test_cache_backend_numerics.py \
                libs/easydel/tests/caching/test_compressed_window_cache.py

env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/easydel/tests/caching/
```
