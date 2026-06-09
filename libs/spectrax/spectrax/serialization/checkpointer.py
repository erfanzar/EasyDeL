# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""High-level checkpoint management with time- and run-based policies.

Provides :class:`Checkpointer` which wraps :class:`AsyncCheckpointManager` with
discovery, temporary-checkpoint cleanup, and trainer-loop integration.
"""

from __future__ import annotations

import datetime as dt
import json
import queue
import threading
import time
import typing as tp
from dataclasses import dataclass
from datetime import timedelta

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils as mh
from jax.sharding import Mesh, NamedSharding

from spectrax._internal.logging import LazyLogger, get_logger

from . import _fs
from ._compat import PyTree, flatten_dict
from .async_manager import AsyncCheckpointManager

logger: LazyLogger = get_logger("Checkpointer")

MetadataDict: tp.TypeAlias = dict[str, object]


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

        Raises:
            ValueError: If *step_policies* is not sorted by ``until`` or if a
                non-final policy has ``until=None``.
        """
        self.base_path = str(base_path)
        self.save_interval = save_interval
        self.step_policies = list(step_policies)
        self._last_save_time = dt.datetime.now()
        self._last_save_step = 0

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

        last_tmp = self._last_temporary_checkpoint
        destination = f"run-{step}"
        full_path = _fs.joinpath(self.base_path, destination)

        self._last_temporary_checkpoint = full_path if not save_permanent else None

        def callback() -> None:
            """Write metadata and clean up previous temporary checkpoint."""
            try:
                _write_checkpoint_metadata(
                    full_path,
                    step=step,
                    is_temporary=not save_permanent,
                )
            except Exception as e:
                logger.warning(f"Failed to write metadata.json: {e}")
            if last_tmp is not None and jax.process_index() == 0:
                try:
                    meta = read_checkpoint_metadata(last_tmp)
                    if meta.get("is_temporary", False):
                        logger.info(f"Deleting old temporary checkpoint {last_tmp}")
                        self._queue_checkpoint_removal(last_tmp)
                    else:
                        logger.info(
                            f"Not deleting old temporary checkpoint {last_tmp} because it is no longer temporary."
                        )
                except FileNotFoundError:
                    logger.warning(f"Could not load metadata for last temporary checkpoint {last_tmp}.")

        if pytree is not None:
            self.save_pytree(
                tree=pytree,
                prefix=prefix,
                step=step,
                destination=destination,
                mesh=mesh,
                extras=extras,
                temporary=not save_permanent,
            )
            callback()

        if true_callbacks is not None:
            for save_callback in true_callbacks:
                save_callback(
                    full_path,
                    mesh,
                    {"step": step, "is_temporary": not save_permanent, **(extras or {})},
                )
            callback()

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

        if _should_write_shared(path):
            _fs.mkdir(path, exist_ok=True)

        merged_extras = extras or {}
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

        if step is not None:
            _write_checkpoint_metadata(path, step=int(step), is_temporary=temporary)

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

    def wait_until_finished(self) -> None:
        """Block until all checkpoint operations complete.

        Includes both the async TensorStore writes and any background
        checkpoint deletion jobs queued on process 0.
        """
        self._manager.global_manager.wait_until_finished()
        if jax.process_index() == 0:
            while (
                getattr(self, "_checkpoint_being_removed", None) is not None
                or not self._checkpoint_cleanup_worker_queue.empty()
            ):
                time.sleep(0.2)

    def _queue_checkpoint_removal(self, checkpoint_dir: str) -> None:
        """Add *checkpoint_dir* to the background deletion queue.

        Args:
            checkpoint_dir: Full path to the checkpoint directory to delete.
        """
        if jax.process_index() == 0:
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
                self._checkpoint_being_removed = None

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
    }
    if jax.process_index() == 0:
        _fs.mkdir(checkpoint_path, exist_ok=True)
        _fs.write_text(_fs.joinpath(checkpoint_path, "metadata.json"), json.dumps(meta))


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


def find_latest_checkpoint(base_path: str) -> str | None:
    """Find the most recent checkpoint under *base_path*.

    Checkpoints are identified by the presence of ``metadata.json``.
    Sorting is based on ``timestamp`` (primary) and ``step`` (secondary).

    Args:
        base_path: Root directory to search.

    Returns:
        Full path to the latest checkpoint directory, or ``None`` if no
        valid checkpoints are found.
    """
    if not _fs.is_dir(base_path):
        return None

    candidates = [p for p in _fs.iterdir(base_path) if _fs.is_dir(p)]
    candidates.append(base_path)

    ckpts = [p for p in candidates if _fs.exists(_fs.joinpath(p, "metadata.json"))]
    if not ckpts:
        logger.debug(f"No checkpoints found under {base_path}")
        return None

    def sort_key(path: str):
        """Return a sort key for a checkpoint path based on timestamp and step.

        Args:
            path: Checkpoint directory path.

        Returns:
            A tuple of ``(timestamp, step)`` for sorting.
        """
        try:
            meta = read_checkpoint_metadata(path)
            ts = dt.datetime.fromisoformat(meta.get("timestamp", "1970-01-01T00:00:00"))
            step = int(meta.get("step", -1))
            return (ts, step)
        except Exception as e:
            logger.debug(f"Could not read metadata for {path}: {e}")
            return (dt.datetime.min, -1)

    return max(ckpts, key=sort_key)


def _rm_checkpoint(path: str) -> None:
    """Remove a checkpoint directory recursively.

    Args:
        path: Directory to delete.
    """
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
