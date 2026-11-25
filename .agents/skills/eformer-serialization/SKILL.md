---
name: eformer-serialization
description: Work on standalone eFormer checkpointing, serialization, fsspec/ePath I/O, SafeTensors/msgpack CheckpointManager, and legacy TensorStore serialization under libs/eformer/eformer/serialization and libs/eformer/eformer/paths.py. For EasyDeL live training/state checkpointing, use spectrax-serialization instead.
---

# Skill: Work On eFormer Serialization And Checkpointing

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

> **IMPORTANT — Ownership Notice:** EasyDeL's live training, `EasyDeLState`, and bridge checkpointing use
> `libs/spectrax/spectrax/serialization/` (OCDBT + v1 commit protocol; EasyDeL has **zero** imports of
> `eformer.serialization`). Route EasyDeL checkpoint bugs to `.agents/skills/spectrax-serialization/SKILL.md` and HF
> conversion bugs to `.agents/skills/convert-checkpoint/SKILL.md`. Use this skill ONLY for standalone `libs/eformer`
> serialization, legacy eFormer checkpoints, SafeTensors/msgpack `CheckpointManager`, or `eformer.paths.ePath`.

## Package Boundary

`libs/eformer` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `spectrax`, `ejkernel`, `eray`, or `surgecore`. Verify with `uv run lint-imports`.

## First Reads

- `libs/eformer/pyproject.toml`
- `libs/eformer/eformer/serialization/__init__.py`
- `libs/eformer/eformer/serialization/checkpointer.py` (`Checkpointer`, `CheckpointInterval`, process-0 cleanup thread)
- `libs/eformer/eformer/serialization/async_manager.py` (`AsyncCheckpointManager`, `pytree_structure.json`, `_cast_floating_leaves`, `_sync_remote_checkpoint_visibility`)
- `libs/eformer/eformer/serialization/base_manager.py` (`CheckpointManager` — SafeTensors / msgpack / GCS client)
- `libs/eformer/eformer/serialization/serialization.py` (`tree_serialize_leaves`, `tree_deserialize_leaves`, `_local_index_lock`, `leaf_key_paths`)
- `libs/eformer/eformer/serialization/fsspec_utils.py` (`should_write_shared_checkpoint_files`, `exists`, `mkdirs`, `expand_glob`, `remove`, `async_remove`, `join_path`)
- `libs/eformer/eformer/serialization/sharding_utils.py` (`create_sharding_tree_from_index`, `apply_sharding_tree`)
- `libs/eformer/eformer/serialization/utils.py` (`reshard`, `to_host`, `estimate_array_nbytes`, shard/index filename helpers)
- `libs/eformer/eformer/paths.py` (`ePath`, `LocalPath`, `GCSPath`, `is_remote_path`, `is_local_path`, `path_protocol`)

## Core Architecture & Invariants

1. **Public APIs & formats:**
   - **TensorStore async stack:** `Checkpointer` (`save_checkpoint`, `load_checkpoint`, `save_pytree`, `load_pytree`),
     `AsyncCheckpointManager`, `tree_serialize_leaves`, `tree_deserialize_leaves`, `create_sharding_tree_from_index`,
     `apply_sharding_tree`. Artifacts include `tensorstore_index.json` (v1.0 & v2.0 `prefixes`),
     `{prefix or 'pytree'}_structure.json`, `metadata.json`, and `checkpoint_metadata.json`.
   - **SafeTensors / msgpack stack (`base_manager.py`):** `CheckpointManager` handles SafeTensors (`safetensors.flax`),
     msgpack, and direct `google.cloud.storage` client uploads/downloads.
2. **PyTree structure & floating-only load casts (`async_manager.py`):**
   - `{prefix or 'pytree'}_structure.json` stores base64-pickled `PyTreeDef` (`_treedef_to_b64` / `_treedef_from_b64`)
     so arbitrary PyTrees restore with exact container types.
   - `_cast_floating_leaves(arrays, dtype)` casts **only floating-point leaves** on load; non-float leaves (quantized
     `uint32` codes, `uint8` block scales, token/index tensors) pass through untouched so quantized checkpoints are not
     corrupted.
   - Pre-existing timeout env var: `GLOBAL_CHECKPOINT_TIMEOUT` (default `400`s, read in both `async_manager.py` and
     `serialization.py`). Do not add new `os.environ` knobs.
3. **Concurrency, locking, and multi-host safety (`serialization.py`, `async_manager.py`, `fsspec_utils.py`):**
   - `_local_index_lock(index_path)` serializes local `tensorstore_index.json` read-modify-write across BOTH threads
     (`_LOCAL_INDEX_THREAD_LOCK`) and processes (`fcntl.flock` on a persistent sidecar `{index_path}.lock`, because
     locking the index inode breaks across atomic rename). `os.register_at_fork(after_in_child=...)` resets the thread
     lock in forked children to prevent inherited-lock deadlocks.
   - `should_write_shared_checkpoint_files(path)` (`fsspec_utils.py`) restricts shared remote metadata writes to
     process 0 so multi-host workers never race on GCS/S3 objects.
   - `_sync_remote_checkpoint_visibility(path, scope=...)` (`async_manager.py`) issues a global barrier
     (`multihost_utils.sync_global_devices`) on multi-process remote writes before other hosts read shared metadata.
4. **Policies & paths (`checkpointer.py`, `paths.py`):**
   - `CheckpointInterval` sequences must be strictly monotonic in `until`, with only the final interval allowed
     `until=None`; process 0 runs a background daemon thread to clean up superseded temporary checkpoints.
   - Use `eformer.paths.ePath` (not `pathlib.Path`) anywhere paths may target `gs://` or `s3://`; use `is_remote_path`
     / `path_protocol` to branch on local vs remote backends.

## Routing

- EasyDeL live checkpointing (OCDBT + v1 commit protocol): `.agents/skills/spectrax-serialization/SKILL.md`
- EasyDeL HF <-> EasyDeL checkpoint conversion: `.agents/skills/convert-checkpoint/SKILL.md`
- Standalone eFormer mesh & sharding: `.agents/skills/debug-eformer-escale/SKILL.md`
- EasyDeL live mesh & sharding: `.agents/skills/spectrax-sharding/SKILL.md`

## Verification

Computation/TensorStore array tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

# Full eFormer serialization suite (accelerator required for TensorStore array tests)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/eformer/tests/serialization/ libs/eformer/tests/paths/
```

Pure-Python `@pytest.mark.cpu_ok` tests (runnable on CPU via `JAX_PLATFORMS=cpu`):
- `libs/eformer/tests/serialization/test_local_index_lock.py`
- `cpu_ok`-marked cases in `libs/eformer/tests/paths/test_paths.py`
