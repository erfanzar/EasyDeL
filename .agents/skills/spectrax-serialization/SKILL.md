---
name: spectrax-serialization
description: Work on SpectraX's TensorStore/OCDBT checkpointing and serialization stack under libs/spectrax/spectrax/serialization — EasyDeL's primary live checkpointing backend. Use for Checkpointer, AsyncCheckpointManager, OcdbtSaveOptions, v1 commit protocol (commit_success.json, data_state.p{i}.json), _fs.py GCS/fsspec helpers, per-key load dtypes, or EasyDeL checkpoint save/resume bugs.
---

# Skill: Work On SpectraX Serialization And Checkpointing

This is a specialization of `.agents/skills/run-research/SKILL.md`. Load and follow `run-research` first.

**Ownership:** `libs/spectrax/spectrax/serialization/` is **EasyDeL's actual live checkpointing backend** (imported
across `libs/easydel/easydel/infra/base_state.py`, `libs/easydel/easydel/infra/mixins/bridge.py`,
`libs/easydel/easydel/trainers/base_trainer.py`, `libs/easydel/easydel/trainers/training_configurations.py`,
`libs/easydel/easydel/trainers/_online_rl/trainer.py`, `libs/easydel/easydel/trainers/bema_trainer/bema_trainer.py`,
and `libs/easydel/easydel/trainers/trainer_protocol.py` — EasyDeL has **zero** imports of `eformer.serialization`). It
is TensorStore-only and adds coalesced OCDBT stores (`_ocdbt.py`) and an atomic v1 commit protocol not present in
eFormer.

## Package Boundary

`libs/spectrax` is an independent foundation package (`pyproject.toml` `[tool.importlinter]`). It MUST NOT import
`easydel`, `ejkernel`, `eformer`, `eray`, or `surgecore` (hence `_fs.py` replaces `eformer.paths.ePath` internally).
Verify with `uv run lint-imports`.

## First Reads

- `libs/spectrax/spectrax/serialization/__init__.py`
- `libs/spectrax/spectrax/serialization/checkpointer.py` (`Checkpointer`, `CheckpointInfo`, `CheckpointInterval`, v1 commit protocol, `data_state.p{i}.json`)
- `libs/spectrax/spectrax/serialization/async_manager.py` (`AsyncCheckpointManager`, treedef preservation, per-key load dtypes, `_cast_floating_leaves`, `_leaf_target_dtype`)
- `libs/spectrax/spectrax/serialization/_ocdbt.py` (`OcdbtSaveOptions`, per-process staging + process-0 merge, `save_context`, `run_sync`)
- `libs/spectrax/spectrax/serialization/_fs.py` (fsspec local/GCS helpers, dot-segment normalization, `invalidate_cache`, `exists(..., refresh=True)`)
- `libs/spectrax/spectrax/serialization/serialization.py` (`tree_serialize_leaves`, `tree_deserialize_leaves`, `leaf_key_paths`)
- `libs/spectrax/spectrax/serialization/fsspec_utils.py` (`should_write_shared_checkpoint_files`)
- `libs/spectrax/spectrax/serialization/_compat.py`

## Core Architecture & Invariants

1. **Commit protocol v1 (`checkpointer.py`):**
   Checkpoints use a marker file (not directory rename, which is non-atomic on GCS/S3) written in three stages:
   1. *Prepare*: process 0 removes any stale `commit_success.json` and writes `metadata.json` with `commit_protocol: 1`
      (directory is now visibly in-progress / uncommitted).
   2. *Write*: every process writes its sharded arrays (preserving TP/FSDP sharding with no all-gather) and optional
      per-process `data_state.p{i}.json` via `write_data_state` / `read_data_state`.
   3. *Commit*: after a global barrier, process 0 rewrites `metadata.json` and writes `commit_success.json` **last**.
   - `list_checkpoints` / `find_latest_checkpoint` / `is_checkpoint_committed` return only committed directories (or
     legacy directories whose `metadata.json` predates `commit_protocol`).
   - Deletion removes `commit_success.json` **first**, so a crash mid-delete leaves an invisible directory cleaned up on
     subsequent retention passes (`max_to_keep`, `cleanup_uncommitted`; legacy checkpoints are never auto-deleted).
   - `CheckpointInterval` sequences must be strictly sorted by `until`; only the final policy may set `until=None`.
2. **OCDBT coalesced layout (`_ocdbt.py`):**
   - Enabled via `Checkpointer(..., ocdbt=True, ocdbt_options=OcdbtSaveOptions(...))`.
   - Instead of one Zarr store per array shard, writes one B+tree store per prefix:
     `{root}/{prefix}.ocdbt/manifest.ocdbt`, `{root}/{prefix}.ocdbt/d/...`, and multi-process staging dirs
     `{root}/{prefix}.ocdbt/ocdbt.process_{i}/...`.
   - Each process commits its arrays in one atomic `ts.Transaction` (keeps that process's payload in host RAM until
     commit). After a barrier, process 0 merges staging manifests into the prefix manifest; **do not delete**
     `ocdbt.process_{i}` independently because the merged manifest references their data files.
   - `save_context(options)` layers concurrency overrides (`io_concurrency`, `copy_concurrency`, `gcs_concurrency`) onto
     JAX's `TS_CONTEXT` rather than constructing a bare `ts.Context` (which would drop JAX's default cache pools).
   - `run_sync` transparently offloads coroutines to a worker thread when called inside an already-running event loop.
3. **GCS / fsspec correctness (`_fs.py`, `fsspec_utils.py`):**
   - `_fs.joinpath` strips leading `./` and `.` segments so object stores never see literal `/.` prefixes (`gs://bucket/run/.` -> `gs://bucket/run`).
   - `gcsfs` caches directory listings aggressively across calls; use `_fs.exists(path, refresh=True)` / `_fs.invalidate_cache(path)` whenever another process may have just written a shared metadata or marker file.
   - `should_write_shared_checkpoint_files(path)` restricts shared metadata writes to process 0 on remote URLs while
     preserving multi-process local behavior.
4. **PyTree structure & per-key load dtypes (`async_manager.py`):**
   - `{prefix or 'pytree'}_structure.json` stores the base64-pickled `PyTreeDef` (`_treedef_to_b64`) for exact round-trips.
   - `dtype` on load accepts `None`, a single dtype, **or a callable `key -> dtype | None`** (`_resolve_target_dtypes`)
     for per-key load precision (e.g., keeping router/norm weights fp32 while loading linear weights bf16).
   - `_leaf_target_dtype` and `_cast_floating_leaves` cast **only floating-point leaves**, protecting integer/quantized
     tensors (`uint32` packed codes, `uint8` scales, token/index tensors) from silent corruption.
   - Pre-existing timeout env var: `GLOBAL_CHECKPOINT_TIMEOUT` (default `400`s). Do not add new `os.environ` knobs.

## Routing

- EasyDeL HF <-> EasyDeL conversion & publishing: `.agents/skills/convert-checkpoint/SKILL.md`
- Legacy standalone eFormer serialization (SafeTensors/msgpack/legacy TensorStore): `.agents/skills/eformer-serialization/SKILL.md`
- Sharding specs passed to checkpoint load/save: `.agents/skills/spectrax-sharding/SKILL.md`

## Verification

Computation/TensorStore array tests MUST run on an accelerator (TPU shown; for GPU use
`JAX_PLATFORMS=cuda,cpu JAX_PLATFORM_NAME=gpu`). Without an accelerator, report computation checks as **unverified on
hardware**.

```bash
uv run lint-imports

# Full SpectraX serialization suite (accelerator required)
env -u XLA_FLAGS ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=tpu,cpu JAX_PLATFORM_NAME=tpu \
  uv run pytest libs/spectrax/tests/serialization/
```

Pure-Python `@pytest.mark.cpu_ok` tests (runnable on CPU via `JAX_PLATFORMS=cpu`):
- `libs/spectrax/tests/serialization/test_checkpoint_commit.py`
- `cpu_ok`-marked cases in `libs/spectrax/tests/serialization/test_fs.py` (`TestJoinpath.test_dot_segments_dropped`,
  `TestExistsRefresh`)
