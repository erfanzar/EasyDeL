---
name: debug-easydel-cache
description: Debug or extend EasyDeL inference caches under libs/easydel/easydel/caching. Use for KV cache, paged caches, MLA, recurrent, hybrid, turboquant, cache specs, metadata builders, or cache-shape failures in eSurge.
---

# Skill: Debug Or Extend EasyDeL Caching

This is a specialization of `.claude/skills/run-research/SKILL.md`.

Load and follow `run-research` first. Use this skill when the failure or feature involves
`libs/easydel/easydel/caching`, cache metadata, or an eSurge cache-shape error.

## First Reads

Read these before editing:

- `WORKSPACE.md`
- `libs/easydel/pyproject.toml`
- `libs/easydel/easydel/caching/_abstracts.py`
- `libs/easydel/easydel/caching/_specs.py`
- `libs/easydel/easydel/caching/_metadatabuilder.py`
- `libs/easydel/easydel/caching/transformer/cache.py`
- `libs/easydel/easydel/caching/ragged_page/cache.py`
- `libs/easydel/easydel/caching/mla_ragged_page/cache.py`
- `libs/easydel/easydel/caching/recurrent/cache.py`
- `libs/easydel/easydel/caching/hybrid/cache.py`
- `libs/easydel/easydel/caching/turboquant_ragged_page/cache.py`

## Typical Tasks

1. Add a new cache backend by subclassing `BaseCache` / `BaseCacheView` /
   `BaseCacheConfig` and registering the view in `OperationsMetadata`.
2. Add a new cache specification in `_specs.py` for a novel attention pattern.
3. Fix cache update or view logic for paged, recurrent, or compressed caches.
4. Ensure `AttentionMetadataBuilder` produces the fields the attention operation requires.

## Routing

- eSurge serving failure with cache shape: load
  `.claude/skills/debug-esurge/SKILL.md`.
- New attention kernel consuming the cache: load
  `.claude/skills/add-easydel-operation/SKILL.md` or
  `.claude/skills/port-ejkernel-to-easydel-operation/SKILL.md`.
- Quantized compressed cache (TurboQuant): load
  `.claude/skills/quantization-layout/SKILL.md`.

## Verification

```bash
ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  uv run pytest libs/easydel/tests/caching/
```

Run the specific cache-family tests first (e.g., `test_ragged_page_cache.py`,
`test_transformer_cache.py`) before the full suite.
