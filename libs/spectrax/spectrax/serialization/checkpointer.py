# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""High-level checkpoint management with time- and run-based policies.

Provides :class:`Checkpointer` which wraps :class:`AsyncCheckpointManager` with
discovery, temporary-checkpoint cleanup, and trainer-loop integration.

Commit protocol (version 1)
---------------------------
A checkpoint directory is written in three stages:

1. *prepare*: process 0 removes any stale ``commit_success.json`` and writes a
   ``metadata.json`` carrying ``commit_protocol``. From this point the
   directory is identifiable as an in-progress (uncommitted) save.
2. *write*: every process writes its array shards (and optional per-process
   ``data_state.p{i}.json``).
3. *commit*: after a global barrier, process 0 rewrites ``metadata.json`` and
   writes ``commit_success.json`` **last**.

Discovery only returns directories that are committed, or legacy directories
(``metadata.json`` without ``commit_protocol``) written before the protocol
existed. A marker file is used instead of a directory rename so the protocol
also holds on object stores where rename is not atomic. Deletion removes the
marker first, so a crash mid-delete leaves an invisible directory that later
retention passes clean up.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import queue
import threading
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils as mh
from jax.sharding import Mesh, NamedSharding

from spectrax._internal.logging import LazyLogger, get_logger

from . import _fs
from ._compat import PyTree, flatten_dict
from .async_manager import AsyncCheckpointManager

logger: LazyLogger = get_logger("Checkpointer")

MetadataDict: tp.TypeAlias = dict[str, object]

COMMIT_PROTOCOL_VERSION = 1
"""Version stamped into ``metadata.json`` under ``commit_protocol``."""

COMMIT_MARKER = "commit_success.json"
"""File written last by process 0; its presence makes a checkpoint visible."""

METADATA_FILE = "metadata.json"

_DATA_STATE_TEMPLATE = "data_state.p{process}.json"
_BYTES_TAG = "__spx_bytes__"


@dataclass(frozen=True)
class CheckpointInfo:
    """Discovery record for one checkpoint directory.

    Attributes:
        path: Full checkpoint directory path.
        step: Step recorded in ``metadata.json`` (``-1`` if absent).
        timestamp: Timestamp recorded in ``metadata.json``.
        is_temporary: Whether the checkpoint was saved as temporary.
        committed: Whether the checkpoint is visible (committed or legacy).
        legacy: ``True`` for directories written before the commit protocol.
    """

    path: str
    step: int
    timestamp: dt.datetime
    is_temporary: bool
    committed: bool
    legacy: bool


@dataclass(frozen=True)
class CheckpointInterval:
    """Configuration for run-based checkpoint saving policy.

    Attributes:
        every: Save checkpoint every N steps within this interval.
        until: Apply this policy until this step (inclusive). If ``None``,
            the policy applies indefinitely. Only the last policy in a
            sequence may have ``until=None``.
    """

    every: int
    until: int | None = None


class Checkpointer:
    """High-level checkpoint manager with time- and run-based policies.

    Integrates with JAX distributed training and TensorStore for efficient
    storage. Preserves existing array shardings (TP/FSDP) without all-gather.

    Attributes:
        base_path: Root directory for all checkpoints.
        save_interval: Optional time interval for temporary checkpoint saves.
        step_policies: Sequence of run-based checkpoint policies.
    """

    def __init__(
        self,
        base_path: str,
        save_interval: timedelta | None,
        step_policies: tp.Sequence[CheckpointInterval],
        *,
        manager: AsyncCheckpointManager | None = None,
        delete_old_temp_checkpoints: bool = True,
        max_to_keep: int | None = None,
        cleanup_uncommitted: bool = True,
    ) -> None:
        """Create a :class:`Checkpointer`.

        Args:
            base_path: Root directory for all checkpoints.
            save_interval: Time interval for temporary saves. ``None`` disables
                time-based temporary checkpoints.
            step_policies: Ordered sequence of :class:`CheckpointInterval`
                policies. Must be sorted by ``until``; only the last may have
                ``until=None``.
            manager: Optional :class:`AsyncCheckpointManager`. A new one is
                created if ``None``.
            delete_old_temp_checkpoints: Whether to queue deletion of prior
                temporary checkpoints found at startup.
            max_to_keep: Keep at most this many committed *permanent*
                checkpoints written under the commit protocol; older ones are
                deleted after each successful commit. ``None`` keeps all.
                Legacy checkpoints are never deleted by retention.
            cleanup_uncommitted: After each commit, delete uncommitted
                protocol directories with a lower step (crashed saves).

        Raises:
            ValueError: If *step_policies* is not sorted by ``until`` or if a
                non-final policy has ``until=None``.
        """
        self.base_path = str(base_path)
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self._last_save_time = dt.datetime.now()
        self._last_save_step = 0
        if max_to_keep is not None and int(max_to_keep) < 1:
            raise ValueError(f"max_to_keep must be >= 1 or None; got {max_to_keep!r}")
        self.max_to_keep = None if max_to_keep is None else int(max_to_keep)
        self.cleanup_uncommitted = bool(cleanup_uncommitted)
        self._queued_removals: set[str] = set()
        self._queued_removals_lock = threading.Lock()

        for i in range(1, len(step_policies)):
            prev_until = step_policies[i - 1].until
            until = step_policies[i].until
            if prev_until is None:
                raise ValueError("Only the last step policy can have an 'until' of None")
            if until is None:
                continue
            if prev_until >= until:
                raise ValueError("Step policies must be sorted by 'until' value")

        self._manager = manager or AsyncCheckpointManager()

        if jax.process_index() == 0:
            self._checkpoint_cleanup_worker_queue: queue.Queue[str] = queue.Queue(maxsize=-1)
            self._checkpoint_cleanup_worker_thread = threading.Thread(
                target=self._checkpoint_cleanup_worker, daemon=True
            )
            self._checkpoint_cleanup_worker_thread.start()
            self._checkpoint_being_removed: str | None = None

        self._last_temporary_checkpoint: str | None = None
        if jax.process_index() == 0:
            latest = find_latest_checkpoint(self.base_path)
            if latest is not None and delete_old_temp_checkpoints:
                try:
                    meta = read_checkpoint_metadata(latest)
                    if meta.get("is_temporary", False):
                        logger.info(
                            f"Found prior temporary checkpoint {latest}. Will delete it after saving a new checkpoint."
                        )
                        self._last_temporary_checkpoint = latest
                except FileNotFoundError:
                    pass

    def on_step(
        self,
        mesh: Mesh,
        pytree: PyTree | None = None,
        force: bool = False,
        *,
        step: int,
        true_callbacks: list[tp.Callable[[str, Mesh, dict], None]] | None = None,
        extras: dict | None = None,
        prefix: str = "checkpoint",
        data_state: Mapping[str, object] | None = None,
    ) -> None:
        """Process a training step and save checkpoint if policies dictate.

        Decides whether to save (and whether the save is permanent or
        temporary) based on *step_policies* and *save_interval*. Broadcasts
        the decision across all JAX processes so every rank agrees.

        Args:
            mesh: JAX mesh for distributed arrays.
            pytree: Training state PyTree to save. Can be ``None`` if callbacks
                handle saving externally.
            force: If ``True``, force a permanent checkpoint save regardless of policies.
            step: Current training step number.
            true_callbacks: Optional callbacks executed when a save is triggered.
                Each receives ``(checkpoint_path, mesh, metadata_dict)``.
            extras: Optional extra metadata written to the checkpoint.
            prefix: Prefix used when saving via ``save_pytree``. Ignored when
                ``pytree is None``.
            data_state: Optional per-process JSON-serializable data-loader
                state (``bytes`` values are allowed). Written into the same
                commit as the model so resume restores a consistent position.

        The checkpoint is committed exactly once, after the pytree and every
        ``true_callbacks`` entry have written their data.
        """
        if step == 0 and not force:
            self._last_save_time = dt.datetime.now()
            return

        if step == self._last_save_step and not force:
            return

        my_should_save = bool(force)
        my_save_permanent = bool(force)

        current_every = self._get_save_interval_for_step(step)
        elapsed = dt.datetime.now() - self._last_save_time

        if current_every is not None and step % current_every == 0:
            my_should_save = True
            my_save_permanent = True
        elif self.save_interval and elapsed >= self.save_interval:
            my_should_save = True
            my_save_permanent = False

        flags = jnp.array([my_should_save, my_save_permanent], dtype=jnp.bool_)
        flags = mh.broadcast_one_to_all(flags)
        should_save, save_permanent = bool(flags[0].item()), bool(flags[1].item())

        if not should_save:
            return

        if save_permanent:
            logger.info(f"Saving checkpoint at step {step}.")
        else:
            logger.info(f"Saving temporary checkpoint at step {step}.")

        destination = f"run-{step}"
        full_path = _fs.joinpath(self.base_path, destination)
        temporary = not save_permanent
        merged_extras = dict(extras or {})

        if pytree is None and true_callbacks is None and data_state is None:
            # Nothing to write; still advance the timers so the interval policy
            # does not fire again on the very next step.
            self._last_save_time = dt.datetime.now()
            self._last_save_step = int(step)
            return

        self._prepare_destination(full_path, step=int(step), temporary=temporary)
        if pytree is not None:
            self._write_trees(
                full_path,
                {prefix: pytree},
                step=int(step),
                mesh=mesh,
                dtype=None,
                extras=merged_extras,
            )
        if true_callbacks is not None:
            callback_meta = {"step": step, "is_temporary": temporary, **merged_extras}
            for save_callback in true_callbacks:
                save_callback(full_path, mesh, dict(callback_meta))
        self._finalize(full_path, step=int(step), temporary=temporary, data_state=data_state)

    def save_pytree(
        self,
        tree: PyTree,
        prefix: str,
        *,
        step: int | None = None,
        destination: str | None = None,
        mesh: Mesh | None = None,
        dtype: jnp.dtype | None = None,
        extras: dict | None = None,
        temporary: bool = False,
        write_index: bool = True,
        commit: bool = True,
    ) -> str:
        """Save a PyTree under a specific prefix with treedef preserved.

        Args:
            tree: PyTree to save.
            prefix: Namespace/prefix (e.g., ``"model"``, ``"tx"``).
            step: Training step number for metadata.
            destination: Optional subdirectory under ``base_path``.
            mesh: Optional compatibility argument accepted by older call sites.
                Save preserves each array's existing sharding, so the mesh is
                only needed on load and is intentionally ignored here.
            dtype: Optional dtype to cast floating point arrays to.
            extras: Optional extra metadata.
            temporary: If ``True``, mark as temporary in metadata.
            write_index: Whether to write the TensorStore index file.
            commit: When ``step`` is given, commit the checkpoint after the
                write (default). Pass ``False`` to write several prefixes and
                call :func:`commit_checkpoint` yourself, or use :meth:`save`.

        Returns:
            Full checkpoint directory path.

        Raises:
            ValueError: If *prefix* is empty or not a string.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        path = destination or self.base_path
        if step is not None:
            dest = destination or f"run-{int(step)}"
            path = _fs.joinpath(self.base_path, str(dest))

        if step is not None:
            self._prepare_destination(path, step=int(step), temporary=temporary)
        elif _should_write_shared(path):
            _fs.mkdir(path, exist_ok=True)

        merged_extras = dict(extras or {})
        if step is not None:
            merged_extras["step"] = int(step)

        self._manager.save_pytree(
            pytree=tree,
            path=path,
            prefix=prefix,
            mesh=mesh,
            dtype=dtype,
            extras=merged_extras,
            write_index=write_index,
        )

        if step is not None and commit:
            self._finalize(path, step=int(step), temporary=temporary, data_state=None)

        return path

    def load_pytree(
        self,
        mesh: Mesh,
        *,
        prefix: str,
        path: str | None = None,
        discover_latest: bool = True,
        discover_raise: bool = True,
        sharding_rules: tp.Sequence[tuple[str, NamedSharding]] | None = None,
        dtype: jnp.dtype | None = None,
        load_treedef: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        template: PyTree | None = None,
        strict_shapes: bool = True,
        key_aliases: tp.Callable[[str], tp.Iterable[str]] | None = None,
        chunk_size: int | None = None,
        can_skip_structure: bool = False,
        concurrent_gb: int = 32,
        tensorstore_io_concurrency: int | None = None,
        tensorstore_copy_concurrency: int | None = None,
        tensorstore_cache_gb: int | None = None,
        tensorstore_assume_metadata: bool = False,
        tensorstore_metadata_workers: int | None = None,
        show_progress: bool = False,
        progress_every: int = 10,
    ) -> tuple[PyTree, MetadataDict]:
        """Load a treedef-preserving PyTree saved under a specific prefix.

        Args:
            mesh: JAX Mesh for array sharding on load.
            prefix: Namespace/prefix used at save time.
            path: Optional exact checkpoint directory. If ``None`` and
                *discover_latest* is ``True``, the most recent checkpoint
                under ``base_path`` is used.
            discover_latest: If ``True``, find the most recent checkpoint.
            discover_raise: If ``True``, raise when no checkpoint is found.
            sharding_rules: Optional sequence of ``(regex, NamedSharding)``
                pairs for sharding inference.
            dtype: Optional dtype to cast loaded arrays to.
            load_treedef: If ``False``, flatten the loaded tree to a
                dotted-string dict.
            callback: Optional per-array callback ``fn(array, key) -> array``.
            template: Optional template PyTree for shape coercion.
            strict_shapes: Whether to enforce exact shape matches.
            key_aliases: Optional function that receives a template key and
                returns alternate checkpoint keys to try when the exact key is
                absent.
            chunk_size: Optional batch size for array loading.
            can_skip_structure: If ``True``, allow loading array-only
                TensorStore checkpoints from ``tensorstore_index.json`` when
                the exact ``{prefix}_structure.json`` sidecar is absent.
            concurrent_gb: In-flight TensorStore read budget in GiB.
            tensorstore_io_concurrency: Optional TensorStore file I/O
                concurrency limit.
            tensorstore_copy_concurrency: Optional TensorStore data-copy
                concurrency limit.
            tensorstore_cache_gb: Optional TensorStore cache-pool size in GiB.
            tensorstore_assume_metadata: If ``True``, use index shape/dtype
                metadata to speed TensorStore opens.
            tensorstore_metadata_workers: Worker count for parallel zarr
                metadata checks when metadata is not assumed.
            show_progress: Whether process 0 should render load progress.
            progress_every: Refresh progress every N loaded tensors.

        Returns:
            Tuple of ``(pytree, metadata)``.

        Raises:
            FileNotFoundError: If no checkpoint is found and *discover_raise*
                is ``True``.
        """
        root = path or self.base_path
        if discover_latest:
            if jax.process_index() == 0:
                discovered = find_latest_checkpoint(root)
            else:
                discovered = None

            MAX_PATH = 4096
            if jax.process_index() == 0 and discovered is not None:
                buf = discovered.encode("utf-8")[:MAX_PATH]
                buf = buf.ljust(MAX_PATH, b"\x00")
            else:
                buf = b"\x00" * MAX_PATH

            buf_arr = jnp.array([b for b in buf], dtype=jnp.uint8)
            buf_arr = mh.broadcast_one_to_all(buf_arr)
            discovered_str = bytes(buf_arr).rstrip(b"\x00").decode("utf-8")

            if not discovered_str:
                if discover_raise:
                    raise FileNotFoundError(f"No checkpoint found under {root}")
                return None, {}
            root = discovered_str

        logger.info(f"Loading checkpoint from {root}")

        pytree, extras = self._manager.load_pytree(
            path=root,
            mesh=mesh,
            prefix=prefix,
            sharding_rules=sharding_rules,
            dtype=dtype,
            strict_shapes=strict_shapes,
            template=template,
            key_aliases=key_aliases,
            callback=callback,
            chunk_size=chunk_size,
            can_skip_structure=can_skip_structure,
            concurrent_gb=concurrent_gb,
            tensorstore_io_concurrency=tensorstore_io_concurrency,
            tensorstore_copy_concurrency=tensorstore_copy_concurrency,
            tensorstore_cache_gb=tensorstore_cache_gb,
            tensorstore_assume_metadata=tensorstore_assume_metadata,
            tensorstore_metadata_workers=tensorstore_metadata_workers,
            show_progress=show_progress,
            progress_every=progress_every,
        )

        if not load_treedef:
            if isinstance(pytree, dict):
                pytree = flatten_dict(pytree, sep=".")
            else:
                logger.warning("load_treedef=False but loaded tree is not a dict; returning original structure.")

        if jax.process_index() == 0:
            try:
                meta = read_checkpoint_metadata(root)
                for k, v in meta.items():
                    if k not in extras:
                        extras[k] = v
            except Exception:
                pass

        return pytree, extras

    def save(
        self,
        step: int,
        trees: Mapping[str, PyTree],
        *,
        mesh: Mesh | None = None,
        data_state: Mapping[str, object] | None = None,
        extras: dict | None = None,
        temporary: bool = False,
        dtype: jnp.dtype | None = None,
        destination: str | None = None,
    ) -> str:
        """Save several prefixed pytrees (and data state) as ONE committed checkpoint.

        Example::

            ckpt.save(step, {"model": state, "opt": opt_state}, data_state=loader.state())

        Args:
            step: Training step; also names the directory ``run-{step}``.
            trees: Mapping ``prefix -> pytree``.
            mesh: Optional compatibility argument (ignored on save).
            data_state: Optional per-process data-loader state.
            extras: Optional extra metadata stored with each prefix.
            temporary: Mark the checkpoint as temporary.
            dtype: Optional floating cast applied on save.
            destination: Optional subdirectory name instead of ``run-{step}``.

        Returns:
            The committed checkpoint directory.

        Raises:
            ValueError: If there is nothing to save or a prefix is invalid.
        """
        if not trees and data_state is None:
            raise ValueError("Checkpointer.save requires at least one tree or a data_state")
        for prefix in trees:
            if not prefix or not isinstance(prefix, str):
                raise ValueError(f"Tree prefixes must be non-empty strings; got {prefix!r}")
        path = _fs.joinpath(self.base_path, str(destination or f"run-{int(step)}"))
        self._prepare_destination(path, step=int(step), temporary=temporary)
        self._write_trees(path, trees, step=int(step), mesh=mesh, dtype=dtype, extras=dict(extras or {}))
        self._finalize(path, step=int(step), temporary=temporary, data_state=data_state)
        return path

    def latest_checkpoint(self) -> CheckpointInfo | None:
        """Return the newest committed checkpoint under ``base_path``, if any."""
        infos = list_checkpoints(self.base_path)
        return infos[-1] if infos else None

    def load_data_state(self, path: str | None = None) -> dict | None:
        """Read this process's data state from ``path`` (default: latest committed).

        Args:
            path: Checkpoint directory. ``None`` uses :meth:`latest_checkpoint`.

        Returns:
            The decoded data-state dict, or ``None`` when absent.
        """
        if path is None:
            info = self.latest_checkpoint()
            if info is None:
                return None
            path = info.path
        return read_data_state(path)

    def _prepare_destination(self, path: str, *, step: int, temporary: bool) -> None:
        """Invalidate any previous commit of ``path`` and mark it in progress."""
        if jax.process_index() == 0:
            marker = _fs.joinpath(path, COMMIT_MARKER)
            if _fs.exists(marker):
                logger.warning(f"Overwriting committed checkpoint {path}; it is invisible until the new commit.")
                _fs.rm(marker)
            _write_checkpoint_metadata(path, step=step, is_temporary=temporary)
        if _should_write_shared(path):
            _fs.mkdir(path, exist_ok=True)
        _barrier(f"spectrax_ckpt_prepare_{int(step)}")

    def _write_trees(
        self,
        path: str,
        trees: Mapping[str, PyTree],
        *,
        step: int,
        mesh: Mesh | None,
        dtype: jnp.dtype | None,
        extras: dict,
    ) -> None:
        """Write every prefixed tree into ``path`` without committing."""
        for prefix, tree in trees.items():
            self._manager.save_pytree(
                pytree=tree,
                path=path,
                prefix=prefix,
                mesh=mesh,
                dtype=dtype,
                extras={**extras, "step": int(step)},
                write_index=True,
            )

    def _finalize(
        self,
        path: str,
        *,
        step: int,
        temporary: bool,
        data_state: Mapping[str, object] | None,
    ) -> None:
        """Write data state, commit, advance save timers, and apply retention."""
        if data_state is not None:
            write_data_state(path, data_state)
        commit_checkpoint(path, step=step, is_temporary=temporary)
        self._last_save_time = dt.datetime.now()
        self._last_save_step = int(step)

        last_tmp = self._last_temporary_checkpoint
        self._last_temporary_checkpoint = path if temporary else None
        if jax.process_index() != 0:
            return
        if last_tmp is not None and last_tmp != path:
            try:
                meta = read_checkpoint_metadata(last_tmp)
                if meta.get("is_temporary", False):
                    logger.info(f"Deleting old temporary checkpoint {last_tmp}")
                    self._queue_checkpoint_removal(last_tmp)
                else:
                    logger.info(f"Not deleting old temporary checkpoint {last_tmp} because it is no longer temporary.")
            except FileNotFoundError:
                logger.warning(f"Could not load metadata for last temporary checkpoint {last_tmp}.")
        self._apply_retention(current_step=int(step), current_path=path)

    def _apply_retention(self, *, current_step: int, current_path: str) -> None:
        """Queue deletion of crashed saves and permanent checkpoints beyond ``max_to_keep``."""
        if jax.process_index() != 0:
            return
        if not self.cleanup_uncommitted and self.max_to_keep is None:
            return
        infos = list_checkpoints(self.base_path, include_uncommitted=True)
        doomed: list[str] = []
        if self.cleanup_uncommitted:
            doomed.extend(i.path for i in infos if not i.committed and not i.legacy and i.step < current_step)
        if self.max_to_keep is not None:
            permanent = sorted(
                (i for i in infos if i.committed and not i.legacy and not i.is_temporary),
                key=lambda i: (i.step, i.timestamp),
            )
            if len(permanent) > self.max_to_keep:
                doomed.extend(i.path for i in permanent[: len(permanent) - self.max_to_keep])
        for path in dict.fromkeys(doomed):
            if path in (current_path, self.base_path):
                continue
            self._queue_checkpoint_removal(path)

    def wait_until_finished(self) -> None:
        """Block until all checkpoint operations complete.

        Includes both the async TensorStore writes and any background
        checkpoint deletion jobs queued on process 0.
        """
        self._manager.global_manager.wait_until_finished()
        if jax.process_index() == 0:
            # ``join`` waits for ``task_done`` on every queued path, closing the
            # window where the worker has dequeued a path but not yet marked it
            # as being removed (the previous empty()/flag poll could return early).
            self._checkpoint_cleanup_worker_queue.join()

    def _queue_checkpoint_removal(self, checkpoint_dir: str) -> None:
        """Add *checkpoint_dir* to the background deletion queue.

        Args:
            checkpoint_dir: Full path to the checkpoint directory to delete.
        """
        if jax.process_index() == 0:
            with self._queued_removals_lock:
                if checkpoint_dir in self._queued_removals:
                    return
                self._queued_removals.add(checkpoint_dir)
            logger.info(f"Queueing deletion of checkpoint {checkpoint_dir}")
            self._checkpoint_cleanup_worker_queue.put(checkpoint_dir)

    def _checkpoint_cleanup_worker(self) -> None:
        """Daemon thread that processes the deletion queue asynchronously.

        Runs indefinitely in the background on process 0, blocking on the
        queue until a checkpoint path is available, then removes it via
        :func:`_rm_checkpoint`.
        """
        while True:
            checkpoint = self._checkpoint_cleanup_worker_queue.get(block=True)
            self._checkpoint_being_removed = checkpoint
            try:
                _rm_checkpoint(checkpoint)
                logger.info(f"Deleted checkpoint {checkpoint}")
            except Exception:
                logger.exception(f"Failed to delete checkpoint {checkpoint}")
            finally:
                with self._queued_removals_lock:
                    self._queued_removals.discard(checkpoint)
                self._checkpoint_being_removed = None
                self._checkpoint_cleanup_worker_queue.task_done()

    def _get_save_interval_for_step(self, step: int) -> int | None:
        """Determine the checkpoint save interval that applies at *step*.

        Walks *step_policies* in order and returns the ``every`` value of the
        first policy whose ``until`` is ``None`` or ``>= step``.

        Args:
            step: Current training step number.

        Returns:
            The save interval (every N steps) or ``None`` if no policy
            applies.
        """
        current_policy = next(
            (p for p in self.step_policies if p.until is None or p.until >= step),
            None,
        )
        return None if current_policy is None else current_policy.every


def _write_checkpoint_metadata(
    checkpoint_path: str,
    step: int,
    is_temporary: bool,
) -> None:
    """Write ``metadata.json`` inside a checkpoint directory.

    Only executes on process 0 to avoid cross-host write conflicts.

    Args:
        checkpoint_path: Directory where the checkpoint was saved.
        step: Training step number to record.
        is_temporary: Whether this checkpoint is temporary.
    """
    meta = {
        "step": int(step),
        "timestamp": dt.datetime.now().isoformat(),
        "is_temporary": bool(is_temporary),
        "commit_protocol": COMMIT_PROTOCOL_VERSION,
    }
    if jax.process_index() == 0:
        _fs.mkdir(checkpoint_path, exist_ok=True)
        _fs.write_text(_fs.joinpath(checkpoint_path, METADATA_FILE), json.dumps(meta))


def _barrier(name: str) -> None:
    """Synchronize all JAX processes; a no-op in single-process runs."""
    if jax.process_count() > 1:
        mh.sync_global_devices(name)


def commit_checkpoint(path: str, *, step: int, is_temporary: bool = False) -> None:
    """Make ``path`` visible to discovery once every process has finished writing.

    Waits for in-flight array writes on all processes, then process 0 rewrites
    ``metadata.json`` (fresh timestamp) and writes ``commit_success.json``
    last. A second barrier makes the commit visible to every process before
    any of them proceeds.

    Args:
        path: Checkpoint directory.
        step: Step recorded in the metadata.
        is_temporary: Whether the checkpoint is temporary.
    """
    _barrier(f"spectrax_ckpt_written_{int(step)}")
    if jax.process_index() == 0:
        _write_checkpoint_metadata(path, step=int(step), is_temporary=is_temporary)
        marker = {
            "step": int(step),
            "timestamp": dt.datetime.now().isoformat(),
            "commit_protocol": COMMIT_PROTOCOL_VERSION,
            "process_count": int(jax.process_count()),
        }
        _fs.write_text(_fs.joinpath(path, COMMIT_MARKER), json.dumps(marker))
    _barrier(f"spectrax_ckpt_committed_{int(step)}")


def is_checkpoint_committed(path: str) -> bool:
    """Return whether ``path`` is visible: committed, or a legacy checkpoint.

    Args:
        path: Checkpoint directory.

    Returns:
        ``False`` for missing metadata or an uncommitted protocol directory.
    """
    try:
        meta = read_checkpoint_metadata(path)
    except Exception:
        return False
    if "commit_protocol" not in meta:
        return True
    return _fs.exists(_fs.joinpath(path, COMMIT_MARKER))


def _encode_data_state(value: object) -> object:
    """Convert a data-state value into JSON-safe form (bytes are base64-tagged)."""
    if isinstance(value, bytes | bytearray):
        return {_BYTES_TAG: base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"data_state keys must be strings; got {k!r}")
            out[k] = _encode_data_state(v)
        return out
    if isinstance(value, list | tuple):
        return [_encode_data_state(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray | jax.Array):
        return np.asarray(value).tolist()
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"Unsupported data_state value of type {type(value).__name__}")


def _decode_data_state(value: object) -> object:
    """Inverse of :func:`_encode_data_state`."""
    if isinstance(value, dict):
        if set(value) == {_BYTES_TAG}:
            return base64.b64decode(value[_BYTES_TAG])
        return {k: _decode_data_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_data_state(v) for v in value]
    return value


def write_data_state(path: str, state: Mapping[str, object]) -> None:
    """Write this process's data-loader state into checkpoint ``path``.

    Every process writes its own ``data_state.p{process}.json`` so per-host
    loader positions round-trip. Call before :func:`commit_checkpoint`.

    Args:
        path: Checkpoint directory.
        state: JSON-serializable mapping; ``bytes`` and numpy values are allowed.
    """
    if not isinstance(state, Mapping):
        raise TypeError(f"data_state must be a mapping; got {type(state).__name__}")
    payload = json.dumps(_encode_data_state(state))
    if not _fs.is_remote_path(path):
        _fs.mkdir(path, exist_ok=True)
    name = _DATA_STATE_TEMPLATE.format(process=int(jax.process_index()))
    _fs.write_text(_fs.joinpath(path, name), payload)


def read_data_state(path: str) -> dict | None:
    """Read this process's data-loader state from ``path``; ``None`` if absent."""
    name = _DATA_STATE_TEMPLATE.format(process=int(jax.process_index()))
    file_path = _fs.joinpath(path, name)
    if not _fs.exists(file_path):
        return None
    return tp.cast(dict, _decode_data_state(json.loads(_fs.read_text(file_path))))


def read_checkpoint_metadata(checkpoint_path: str) -> MetadataDict:
    """Read and parse ``metadata.json`` from a checkpoint directory.

    Args:
        checkpoint_path: Directory containing ``metadata.json``.

    Returns:
        Parsed metadata dictionary with keys such as ``"step"``,
        ``"timestamp"``, and ``"is_temporary"``.
    """
    text = _fs.read_text(_fs.joinpath(checkpoint_path, "metadata.json"))
    return json.loads(text)


# Backward-compatible alias
_read_checkpoint_metadata = read_checkpoint_metadata


def _checkpoint_info(path: str) -> CheckpointInfo | None:
    """Build a :class:`CheckpointInfo` for ``path``; ``None`` without metadata."""
    if not _fs.exists(_fs.joinpath(path, METADATA_FILE)):
        return None
    try:
        meta = read_checkpoint_metadata(path)
    except Exception as e:
        logger.debug(f"Could not read metadata for {path}: {e}")
        return None
    try:
        ts = dt.datetime.fromisoformat(str(meta.get("timestamp", "1970-01-01T00:00:00")))
    except ValueError:
        ts = dt.datetime.min
    legacy = "commit_protocol" not in meta
    committed = legacy or _fs.exists(_fs.joinpath(path, COMMIT_MARKER))
    try:
        step = int(meta.get("step", -1))
    except (TypeError, ValueError):
        step = -1
    return CheckpointInfo(
        path=path,
        step=step,
        timestamp=ts,
        is_temporary=bool(meta.get("is_temporary", False)),
        committed=committed,
        legacy=legacy,
    )


def _reattach_scheme(scheme: str, path: str) -> str:
    """Re-attach a URI *scheme* (e.g. ``"gs://"``) to a scheme-stripped *path*.

    ``_fs.iterdir()`` yields paths with the URI scheme stripped: absolute for
    local backends (``file:///a/b`` -> ``/a/b/child``) and bucket-relative for
    object stores (``gs://bucket/run`` -> ``bucket/run/child``). The scheme is
    prepended verbatim — no slash stripping — so ``file://`` URIs keep their
    absolute-path root (``file://`` + ``/a/b`` == ``file:///a/b``).

    Args:
        scheme: URI scheme including the separator (``"file://"``), or ``""``
            for plain local paths.
        path: Child path as returned by ``_fs.iterdir``.

    Returns:
        The path with the scheme attached (unchanged when *scheme* is empty
        or *path* already carries it).
    """
    if not scheme or path.startswith(scheme):
        return path
    return scheme + path


def list_checkpoints(base_path: str, *, include_uncommitted: bool = False) -> list[CheckpointInfo]:
    """List checkpoints under ``base_path`` (and ``base_path`` itself), oldest first.

    Ordering is ``(timestamp, step)``; the commit rewrites the timestamp, so
    order reflects commit time.

    Args:
        base_path: Root directory to search.
        include_uncommitted: Also return in-progress / crashed protocol saves.

    Returns:
        A list of :class:`CheckpointInfo`, oldest to newest.
    """
    if not _fs.is_dir(base_path):
        return []
    # ``_fs.iterdir()`` strips the URI scheme ("gs://bucket/run" -> "bucket/run"),
    # so re-attach it before probing children; otherwise every probe resolves
    # against the local filesystem and remote discovery finds nothing.
    scheme = base_path.split("://", 1)[0] + "://" if "://" in base_path else ""
    candidates = [q for q in (_reattach_scheme(scheme, p) for p in _fs.iterdir(base_path)) if _fs.is_dir(q)]
    candidates.append(base_path)
    infos = [info for info in (_checkpoint_info(p) for p in candidates) if info is not None]
    if not include_uncommitted:
        infos = [info for info in infos if info.committed]
    infos.sort(key=lambda i: (i.timestamp, i.step))
    return infos


def find_latest_checkpoint(base_path: str, *, include_uncommitted: bool = False) -> str | None:
    """Find the most recent *committed* checkpoint under *base_path*.

    Protocol checkpoints are visible only once ``commit_success.json`` exists;
    legacy checkpoints (``metadata.json`` without ``commit_protocol``) are
    always visible. Sorting is by ``timestamp`` then ``step``.

    Args:
        base_path: Root directory to search.
        include_uncommitted: Also consider uncommitted (crashed) saves.

    Returns:
        Full path to the latest checkpoint directory, or ``None``.
    """
    infos = list_checkpoints(base_path, include_uncommitted=include_uncommitted)
    if not infos:
        logger.debug(f"No committed checkpoints found under {base_path}")
        return None
    return infos[-1].path


def _rm_checkpoint(path: str) -> None:
    """Remove a checkpoint directory, deleting its commit marker first.

    Removing the marker first means a crash mid-delete leaves an invisible
    (uncommitted) directory rather than a visible, partially deleted one.

    Args:
        path: Directory to delete.
    """
    marker = _fs.joinpath(path, COMMIT_MARKER)
    if _fs.exists(marker):
        _fs.rm(marker)
    if _fs.exists(path):
        _fs.rm(path, recursive=True)


def _should_write_shared(path: str) -> bool:
    """Return whether the current process should write shared metadata files.

    For local paths every process may write. For remote paths only process 0
    writes to avoid contention.

    Args:
        path: Checkpoint path (local or remote).

    Returns:
        ``True`` if this process should perform shared writes.
    """
    from . import fsspec_utils

    return fsspec_utils.should_write_shared_checkpoint_files(path)
