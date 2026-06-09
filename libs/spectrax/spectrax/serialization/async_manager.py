# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Asynchronous checkpoint manager built on JAX GlobalAsyncCheckpointManager.

Provides :class:`AsyncCheckpointManager` which wraps JAX's TensorStore-based
:class:`GlobalAsyncCheckpointManager` for high-performance distributed
checkpointing. Supports:

* Async array serialization / deserialization
* Treedef preservation (exact PyTree structure round-trips)
* Sharding preservation without all-gather
* Structured saves with per-prefix namespaces
* Chunked loading for reduced peak memory
"""

import asyncio
import base64
import concurrent.futures
import importlib
import json
import os
import pickle
import re
import typing as tp
from dataclasses import dataclass
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts
from jax.distributed import is_initialized
from jax.experimental.array_serialization import serialization as array_serialization
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from spectrax._internal.logging import LazyLogger, get_logger
from spectrax._version import __version__
from spectrax.serialization.serialization import leaf_key_paths

from . import _fs, fsspec_utils
from ._compat import PyTree
from .serialization import tree_serialize_leaves

logger: LazyLogger = get_logger("AsyncCheckpointManager")
GLOBAL_CHECKPOINT_TIMEOUT: int = int(os.getenv("GLOBAL_CHECKPOINT_TIMEOUT", "400"))
KeyAliases: tp.TypeAlias = tp.Callable[[str], tp.Iterable[str]]


def _is_array_like(x):
    """Check if an object is array-like (has ``shape`` and ``dtype`` attributes).

    Args:
        x: Object to inspect.

    Returns:
        ``True`` if *x* looks like an array, ``False`` otherwise.
    """
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _treedef_to_b64(treedef) -> str:
    """Serialize a JAX tree definition to a base64 string.

    Args:
        treedef: A :class:`jax.tree_util.PyTreeDef`.

    Returns:
        Base64-encoded pickled treedef.
    """
    return base64.b64encode(pickle.dumps(treedef)).decode("utf-8")


def _treedef_from_b64(s: str):
    """Deserialize a JAX tree definition from a base64 string.

    Args:
        s: Base64-encoded pickled treedef.

    Returns:
        The reconstructed :class:`jax.tree_util.PyTreeDef`.
    """
    return pickle.loads(base64.b64decode(s.encode("utf-8")))


def _structure_path(path: str, prefix: str | None) -> str:
    """Return the JSON structure-file path for a given checkpoint dir and prefix.

    Args:
        path: Checkpoint directory (local or remote).
        prefix: Logical namespace/prefix (e.g. ``"model"``). If ``None``,
            defaults to ``"pytree"``.

    Returns:
        The joined path string ending in ``"{prefix}_structure.json"``.
    """
    name = f"{prefix or 'pytree'}_structure.json"
    return _fs.joinpath(path, name)


def _is_none(x):
    """Check if a value is ``None``.

    Args:
        x: Value to check.

    Returns:
        ``True`` if *x* is ``None``, ``False`` otherwise.
    """
    return x is None


def _checkpoint_metadata_extras(root: str) -> dict:
    """Read best-effort metadata extras for checkpoints without structure JSON.

    Args:
        root: Checkpoint directory that may contain
            ``checkpoint_metadata.json``.

    Returns:
        The ``custom_metadata`` dictionary from the metadata file. Returns an
        empty dictionary when the file is absent, malformed, or does not carry
        a dictionary-valued ``custom_metadata`` field.
    """
    path = _fs.joinpath(root, "checkpoint_metadata.json")
    if not _fs.exists(path):
        return {}
    try:
        metadata = json.loads(_fs.read_text(path))
    except Exception:
        return {}
    custom = metadata.get("custom_metadata", {})
    return custom if isinstance(custom, dict) else {}


def _key_from_relpath(rel_path: str) -> str:
    """Convert a TensorStore relative path to a dotted checkpoint key.

    Args:
        rel_path: Relative TensorStore array path from ``tensorstore_index``,
            for example ``"model/layer/weight"``.

    Returns:
        Dotted key form used by SpectraX checkpoint trees, for example
        ``"model.layer.weight"``.
    """
    return rel_path.replace("/", ".").replace("\\", ".")


def _strip_prefix_from_key(key: str, prefix: str) -> str:
    """Remove the checkpoint prefix from a dotted key when present.

    Args:
        key: Dotted checkpoint key, possibly prefixed by ``"{prefix}."``.
        prefix: Logical checkpoint namespace to remove.

    Returns:
        ``key`` without the leading ``"{prefix}."`` segment when it exists;
        otherwise returns ``key`` unchanged.
    """
    prefix_dot = f"{prefix}."
    return key[len(prefix_dot) :] if key.startswith(prefix_dot) else key


def _insert_nested(result: dict, key: str, value: object) -> None:
    """Insert ``value`` into ``result`` at a dotted key path.

    Args:
        result: Mutable nested dictionary being reconstructed.
        key: Dotted key path such as ``"layers.0.weight"``.
        value: Leaf value to store at the destination path.

    Raises:
        ValueError: If ``key`` is empty or if inserting would overwrite an
            existing non-dictionary intermediate node.
    """
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Cannot insert a checkpoint leaf with an empty key.")
    current = result
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Checkpoint key collision while reconstructing {key!r}.")
        current = child
    current[parts[-1]] = value


def _shape_from_index(value: object) -> tuple[int, ...] | None:
    """Return a shape tuple from TensorStore index metadata.

    Args:
        value: Raw ``shape`` field read from ``tensorstore_index.json``.

    Returns:
        Tuple of integer dimensions when ``value`` is a valid sequence;
        otherwise ``None``.
    """
    if not isinstance(value, list | tuple):
        return None
    try:
        return tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return None


def _dtype_from_index(value: object) -> object | None:
    """Return a JAX dtype from TensorStore index metadata.

    Args:
        value: Raw ``dtype`` field read from ``tensorstore_index.json``.

    Returns:
        A JAX dtype object when the metadata can be parsed; otherwise
        ``None``.
    """
    if value is None:
        return None
    try:
        return jnp.dtype(value)
    except TypeError:
        return None


def _array_nbytes(shape: tuple[int, ...] | None, dtype: object | None) -> int:
    """Estimate array byte size for progress reporting.

    Args:
        shape: Global array shape, or ``None`` when unknown.
        dtype: Array dtype, or ``None`` when unknown.

    Returns:
        Estimated byte count. Returns ``0`` when either input is unavailable
        or cannot be interpreted as a dtype/shape.
    """
    if shape is None or dtype is None:
        return 0
    try:
        itemsize = np.dtype(jnp.dtype(dtype)).itemsize
    except TypeError:
        return 0
    total = itemsize
    for dim in shape:
        total *= int(dim)
    return int(total)


def _format_gib(num_bytes: int) -> str:
    """Format bytes as GiB for compact progress postfixes.

    Args:
        num_bytes: Byte count to display.

    Returns:
        String formatted as ``"{value:.1f}GiB"``.
    """
    return f"{num_bytes / (1024**3):.1f}GiB"


def _tensorstore_context(
    *,
    io_concurrency: int | None,
    copy_concurrency: int | None,
    cache_gb: int | None,
) -> ts.Context | None:
    """Build a TensorStore context for high-throughput checkpoint reads.

    Args:
        io_concurrency: Optional file I/O concurrency limit.
        copy_concurrency: Optional data-copy concurrency limit.
        cache_gb: Optional cache-pool size in GiB.

    Returns:
        A configured :class:`tensorstore.Context` when any knob is provided;
        otherwise ``None`` so callers can use JAX/TensorStore defaults.
    """
    cfg: dict[str, dict[str, int]] = {}
    if io_concurrency is not None:
        cfg["file_io_concurrency"] = {"limit": max(1, int(io_concurrency))}
    if copy_concurrency is not None:
        cfg["data_copy_concurrency"] = {"limit": max(1, int(copy_concurrency))}
    if cache_gb is not None:
        cfg["cache_pool"] = {"total_bytes_limit": max(1, int(cache_gb)) * 1024**3}
    return ts.Context(cfg) if cfg else None


def _is_remote_path(path: str) -> bool:
    """Return whether a path points at a remote object store.

    Args:
        path: Checkpoint or TensorStore path.

    Returns:
        ``True`` for ``gs://`` and ``s3://`` paths, ``False`` otherwise.
    """
    return path.startswith(("gs://", "s3://"))


def _tensorstore_spec_for_load(
    path: str,
    *,
    sharding: object,
    shape: tuple[int, ...] | None,
    storage_dtype: object | None,
    assume_metadata: bool,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a TensorStore spec, optionally embedding metadata for fast open.

    Args:
        path: Absolute TensorStore array path.
        sharding: Destination sharding used to compute local shard shape for
            TensorStore metadata.
        shape: Global array shape from ``tensorstore_index.json``.
        storage_dtype: Stored array dtype from ``tensorstore_index.json``.
        assume_metadata: Whether exact sidecar metadata may be embedded and
            passed to TensorStore with ``assume_metadata=True``.
        metadata: Exact TensorStore metadata read from the array's ``.zarray``
            sidecar. When provided, it is embedded verbatim and takes
            precedence over generated metadata.

    Returns:
        TensorStore spec dictionary. When metadata is available and requested,
        the spec includes a ``metadata`` entry that lets TensorStore avoid
        fetching zarr metadata before opening the array.
    """
    spec = dict(array_serialization.get_tensorstore_spec(path))
    if assume_metadata and metadata is not None:
        spec["metadata"] = metadata
        return spec
    return spec


def _missing_zarr_metadata(paths: list[str], workers: int | None) -> list[str]:
    """Check TensorStore zarr metadata paths, optionally in parallel.

    Args:
        paths: Absolute TensorStore array paths.
        workers: Optional number of threads for metadata existence checks.

    Returns:
        Array paths whose ``.zarray`` metadata file is missing.
    """
    metadata_paths = [_fs.joinpath(path, ".zarray") for path in paths]
    if workers is None or int(workers) <= 1 or len(metadata_paths) <= 1:
        return [path for path, meta_path in zip(paths, metadata_paths, strict=False) if not _fs.exists(meta_path)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        exists = list(executor.map(_fs.exists, metadata_paths))
    return [path for path, ok in zip(paths, exists, strict=False) if not ok]


def _zarr_metadata_for_paths(paths: list[str], workers: int | None) -> list[dict[str, object] | None]:
    """Read exact zarr metadata sidecars for TensorStore array paths.

    Args:
        paths: Absolute TensorStore array paths.
        workers: Optional number of threads used to read ``.zarray`` files.

    Returns:
        Per-path metadata dictionaries in the same order as ``paths``. Entries
        are ``None`` when the sidecar is missing, malformed, or not a mapping.
    """

    def read_one(path: str) -> dict[str, object] | None:
        """Read one ``.zarray`` metadata file.

        Args:
            path: Absolute TensorStore array path.

        Returns:
            Parsed metadata dictionary, or ``None`` if it cannot be read.
        """
        try:
            data = json.loads(_fs.read_text(_fs.joinpath(path, ".zarray")))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    if workers is None or int(workers) <= 1 or len(paths) <= 1:
        return [read_one(path) for path in paths]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        return list(executor.map(read_one, paths))


def _index_arrays_for_prefix(root: str, prefix: str) -> list[dict[str, object]]:
    """Read TensorStore index entries for one prefix when available.

    Args:
        root: Checkpoint directory containing ``tensorstore_index.json``.
        prefix: Logical checkpoint namespace to read from the index.

    Returns:
        List of array metadata dictionaries for ``prefix``. Returns an empty
        list if the index is absent, malformed, unsupported, or has no matching
        prefix.
    """
    index_path = _fs.joinpath(root, "tensorstore_index.json")
    if not _fs.exists(index_path):
        return []
    try:
        index_data = json.loads(_fs.read_text(index_path))
    except Exception:
        return []
    if index_data.get("format") != "tensorstore":
        return []
    if "prefixes" in index_data:
        arrays = index_data.get("prefixes", {}).get(prefix, [])
    else:
        arrays = index_data.get("arrays", [])
    return [entry for entry in arrays if isinstance(entry, dict)]


def _compile_sharding_rules(
    sharding_rules: tp.Sequence[tuple[str, NamedSharding]] | None,
) -> list[tuple[tp.Any, NamedSharding]] | None:
    """Compile regex sharding rules once before matching checkpoint keys.

    Args:
        sharding_rules: Optional sequence of ``(pattern, sharding)`` pairs.

    Returns:
        A list of ``(compiled_pattern, sharding)`` pairs, or ``None`` when no
        rules were provided.
    """
    if sharding_rules is None:
        return None
    return [(re.compile(pattern), sharding) for pattern, sharding in sharding_rules]


def _resolve_apply_shardings(
    array_keys: list[str],
    *,
    mesh: Mesh,
    shardings: dict[str, tp.Callable] | None,
    sharding_rules: tp.Sequence[tuple[str, NamedSharding]] | None,
) -> list[object]:
    """Resolve destination shardings for checkpoint array keys.

    Args:
        array_keys: Dotted checkpoint array keys in load order.
        mesh: Mesh used for the fully replicated fallback sharding.
        shardings: Optional exact key-to-sharding mapping.
        sharding_rules: Optional regex rules used when an exact mapping is not
            supplied.

    Returns:
        Destination sharding objects aligned with ``array_keys``.
    """
    fallback = NamedSharding(mesh=mesh, spec=PartitionSpec())
    compiled_rules = _compile_sharding_rules(sharding_rules)
    if compiled_rules is not None:
        resolved = []
        for key in array_keys:
            slash_key = key.replace(".", "/")
            found = None
            for pattern, sharding in compiled_rules:
                if pattern.search(slash_key):
                    found = sharding
                    break
            resolved.append(found if found is not None else fallback)
        return resolved
    if shardings is None:
        return [fallback] * len(array_keys)
    return [shardings.get(key, fallback) for key in array_keys]


def _template_sharding_lookup_keys(
    array_keys: list[str],
    *,
    template: PyTree | None,
    prefix: str | None,
    key_aliases: KeyAliases | None,
) -> list[str]:
    """Choose the logical keys used to resolve shardings for saved arrays.

    Args:
        array_keys: Saved checkpoint array keys, including *prefix* when the
            checkpoint structure records one.
        template: Optional destination PyTree. When provided, arrays may be
            restored into template keys that differ from the saved checkpoint
            keys.
        prefix: Prefix passed to :func:`leaf_key_paths` for template keys.
        key_aliases: Optional alias function used by template restore.

    Returns:
        Keys aligned with ``array_keys``. Exact template matches and alias
        matches are rewritten to the current template key so sharding regexes
        are resolved against the live layout rather than a legacy checkpoint
        layout.
    """
    if template is None:
        return array_keys

    tpl_leaves, _tpl_treedef = jax.tree_util.tree_flatten(template, is_leaf=_is_none)
    tpl_leaf_keys_tree = leaf_key_paths(template, prefix=prefix, is_leaf=_is_none)
    tpl_leaf_keys_full: list[str] = jax.tree_util.tree_leaves(tpl_leaf_keys_tree, is_leaf=_is_none)
    tpl_arr_mask = [_is_array_like(x) for x in tpl_leaves]

    saved_to_template: dict[str, str] = {}
    for key, is_array in zip(tpl_leaf_keys_full, tpl_arr_mask, strict=True):
        if not is_array:
            continue
        saved_to_template.setdefault(key, key)
        if key_aliases is None:
            continue
        for alias in key_aliases(key):
            saved_to_template.setdefault(alias, key)

    return [saved_to_template.get(key, key) for key in array_keys]


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for checkpoints with versioning and validation.

    Attributes:
        version: Spectrax version string recorded at save time.
        timestamp: ISO-format timestamp. ``None`` means "use current time
            when :meth:`to_dict` is called".
        custom_metadata: Arbitrary user-supplied metadata dict.
    """

    version: str = __version__
    timestamp: str = None
    custom_metadata: dict = None

    def to_dict(self) -> dict:
        """Serialize the metadata dataclass to a plain dictionary.

        Returns:
            Dictionary with keys ``"version"``, ``"timestamp"``,
            and ``"custom_metadata"``.
        """
        return {
            "version": self.version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "custom_metadata": self.custom_metadata or {},
        }


class AsyncCheckpointManager:
    """Checkpoint manager built on top of JAX GlobalAsyncCheckpointManager (TensorStore).

    Provides checkpoint saving and loading with support for parallel operations
    and TensorStore backend. Preserves existing array shardings (TP/FSDP)
    without performing all-gather operations.
    """

    def __init__(self, float_dtype: jnp.dtype = jnp.bfloat16):
        """Initialize the async checkpoint manager.

        Args:
            float_dtype: Default dtype used when ``dtype`` is not explicitly
                passed to :meth:`save_pytree`. Defaults to ``jnp.bfloat16``.

        Raises:
            RuntimeError: If running on multiple processes and JAX distributed
                has not been initialized.
        """
        if jax.process_count() > 1:
            if not is_initialized():
                raise RuntimeError("you should call jax distribution init before running process.")

        self.float_dtype = float_dtype
        self._global_manager = None

    @property
    def global_manager(self) -> GlobalAsyncCheckpointManager:
        """Get or create the global async checkpoint manager lazily.

        Returns:
            The :class:`GlobalAsyncCheckpointManager` instance.
        """
        if self._global_manager is None:
            self._global_manager = GlobalAsyncCheckpointManager(timeout_secs=GLOBAL_CHECKPOINT_TIMEOUT)
        return self._global_manager

    def _deserialize_with_paths(
        self,
        *,
        shardings: list[object],
        paths: list[str],
        shapes: list[tuple[int, ...] | None] | None,
        storage_dtypes: list[object | None] | None,
        target_dtype: object | None,
        concurrent_gb: int,
        tensorstore_io_concurrency: int | None,
        tensorstore_copy_concurrency: int | None,
        tensorstore_cache_gb: int | None,
        tensorstore_assume_metadata: bool,
        show_progress: bool,
        progress_every: int,
        tensorstore_metadata: list[dict[str, object] | None] | None = None,
    ) -> list[jax.Array]:
        """Deserialize arrays from TensorStore paths with optional fast-load controls.

        Args:
            shardings: Destination shardings, one per TensorStore array path.
            paths: Absolute TensorStore array paths to read.
            shapes: Optional global shapes from ``tensorstore_index.json``.
                When every shape is present, the default JAX manager path can
                validate shapes and the custom path can skip metadata reads.
            storage_dtypes: Optional stored dtypes from
                ``tensorstore_index.json``. Used for byte accounting and for
                TensorStore ``assume_metadata`` specs.
            target_dtype: Optional dtype to cast arrays to while reading.
            concurrent_gb: In-flight read budget in decimal GiB units, matching
                JAX's ``GlobalAsyncCheckpointManager`` API.
            tensorstore_io_concurrency: Optional TensorStore file I/O
                concurrency limit.
            tensorstore_copy_concurrency: Optional TensorStore data-copy
                concurrency limit.
            tensorstore_cache_gb: Optional TensorStore cache-pool size in GiB.
            tensorstore_assume_metadata: Whether to embed index metadata into
                TensorStore specs and open with ``assume_metadata=True``.
            tensorstore_metadata: Optional exact ``.zarray`` metadata entries,
                one per path. Entries set to ``None`` fall back to normal
                TensorStore metadata reads for that path.
            show_progress: Whether process 0 should render a progress bar.
            progress_every: Refresh progress every N completed tensors.

        Returns:
            Loaded JAX arrays in the same order as ``paths``.
        """
        if not paths:
            return []

        use_custom_path = (
            tensorstore_io_concurrency is not None
            or tensorstore_copy_concurrency is not None
            or tensorstore_cache_gb is not None
            or tensorstore_assume_metadata
            or show_progress
        )
        target_dtypes = None if target_dtype is None else [target_dtype] * len(paths)
        if not use_custom_path:
            return list(
                self.global_manager.deserialize_with_paths(
                    shardings=shardings,  # pyright: ignore[reportArgumentType]
                    paths=paths,
                    global_shapes=shapes if shapes and all(shape is not None for shape in shapes) else None,
                    dtypes=target_dtypes,
                    concurrent_gb=int(concurrent_gb),
                )
            )

        context = _tensorstore_context(
            io_concurrency=tensorstore_io_concurrency,
            copy_concurrency=tensorstore_copy_concurrency,
            cache_gb=tensorstore_cache_gb,
        )
        shapes = shapes or [None] * len(paths)
        storage_dtypes = storage_dtypes or [None] * len(paths)
        tensorstore_metadata = tensorstore_metadata or [None] * len(paths)
        shapes = tp.cast(list[tuple[int, ...] | None], shapes)
        storage_dtypes = tp.cast(list[np.dtype | jnp.dtype | type | str | None], storage_dtypes)
        tensorstore_metadata = tp.cast(list[dict[str, tp.Any] | None], tensorstore_metadata)
        requested_bytes = [
            _array_nbytes(shape, storage_dtype) for shape, storage_dtype in zip(shapes, storage_dtypes, strict=True)
        ]
        total_bytes = int(sum(requested_bytes))
        show_bar = bool(show_progress and jax.process_index() == 0)
        every = max(1, int(progress_every))

        async def _run() -> list[jax.Array]:
            limiter_cls = array_serialization._LimitInFlightBytes  # pyright: ignore[reportPrivateUsage]
            byte_limiter = limiter_cls(max(1, int(concurrent_gb)) * 10**9)
            results: list[jax.Array | None] = [None] * len(paths)

            progress_bar = None
            if show_bar:
                try:
                    tqdm = importlib.import_module("tqdm.auto").tqdm
                    progress_bar = tqdm(
                        total=len(paths),
                        desc="Loading",
                        unit="tensor",
                        dynamic_ncols=True,
                        miniters=every,
                        mininterval=0.5,
                        leave=True,
                    )
                except Exception:
                    progress_bar = None

            completed = 0
            reported = 0
            loaded_bytes = 0

            async def _one(index: int) -> tuple[int, jax.Array]:
                shape = shapes[index]
                storage_dtype = storage_dtypes[index]
                metadata = tensorstore_metadata[index]
                assume_metadata = bool(tensorstore_assume_metadata and metadata is not None)
                spec = _tensorstore_spec_for_load(
                    paths[index],
                    sharding=shardings[index],
                    shape=shape,
                    storage_dtype=storage_dtype,
                    assume_metadata=assume_metadata,
                    metadata=metadata,
                )
                array = await array_serialization.async_deserialize(
                    shardings[index],  # pyright: ignore[reportArgumentType]
                    spec,
                    global_shape=shape,
                    dtype=target_dtype,
                    byte_limiter=byte_limiter,
                    context=context if context is not None else array_serialization.TS_CONTEXT,
                    assume_metadata=assume_metadata,
                )
                return index, array

            try:
                tasks = [asyncio.create_task(_one(index)) for index in range(len(paths))]
                for future in asyncio.as_completed(tasks):
                    index, array = await future
                    results[index] = array
                    completed += 1
                    loaded_bytes += requested_bytes[index]
                    if progress_bar is not None and (completed % every == 0 or completed == len(paths)):
                        if total_bytes:
                            progress_bar.set_postfix_str(
                                f"{_format_gib(loaded_bytes)}/{_format_gib(total_bytes)}",
                                refresh=False,
                            )
                        progress_bar.update(completed - reported)
                        reported = completed
            finally:
                if progress_bar is not None:
                    progress_bar.close()

            return [array for array in results if array is not None]

        return asyncio.run(_run())

    def save_pytree(
        self,
        pytree: PyTree,
        path: str | os.PathLike,
        *,
        prefix: str,
        mesh: Mesh | None = None,
        dtype: jnp.dtype | None = None,
        extras: dict | None = None,
        write_index: bool = True,
    ) -> str:
        """Save a PyTree with exact structure and prefix via TensorStore.

        This method preserves the original JAX PyTree definition (treedef) so
        that :meth:`load_pytree` can reconstruct the exact same structure
        without requiring a template. Array shardings (TP/FSDP) are preserved
        without performing any all-gather.

        Args:
            pytree: Arbitrary nested structure containing JAX arrays, NumPy
                arrays, and other serializable Python objects.
            path: Destination directory (local path or remote URL such as
                ``"gs://bucket/path"``).
            prefix: Logical namespace for the saved tree (e.g. ``"model"``,
                ``"tx"``). Must be a non-empty string.
            mesh: Optional compatibility argument accepted by older call sites.
                Save preserves each array's existing sharding, so the mesh is
                only needed on load and is intentionally ignored here.
            dtype: Optional dtype to cast floating-point arrays to before
                saving. If ``None``, the original dtypes are preserved.
            extras: Optional dictionary of extra metadata stored inside the
                structure JSON file.
            write_index: Whether to write/update ``tensorstore_index.json``.
                Defaults to ``True``.

        Returns:
            The checkpoint directory path (same as *path* but normalized to a
            string).

        Raises:
            ValueError: If *prefix* is empty or not a string, or if the
                TensorStore index keys do not match the PyTree leaf keys.
            FileNotFoundError: If ``tensorstore_index.json`` is missing after
                the save completes.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        root = str(path)
        write_shared_files = fsspec_utils.should_write_shared_checkpoint_files(root)
        if write_shared_files:
            _fs.mkdir(root, exist_ok=True)

        if dtype is not None:
            pytree = jax.tree_util.tree_map(
                lambda x: x.astype(dtype) if _is_array_like(x) and jnp.issubdtype(x.dtype, jnp.floating) else x,
                pytree,
            )

        leaves, treedef = jax.tree_util.tree_flatten(pytree, is_leaf=_is_none)

        leaf_keys_tree = leaf_key_paths(pytree, prefix=prefix, is_leaf=_is_none)
        leaf_keys_full: list[str] = jax.tree_util.tree_leaves(leaf_keys_tree, is_leaf=_is_none)
        if len(leaf_keys_full) != len(leaves):
            raise ValueError(
                f"Mismatch between leaf_keys ({len(leaf_keys_full)}) and leaves ({len(leaves)}). "
                "Ensure treedef and leaves use the same is_leaf and no leaves are dropped."
            )

        arr_mask = [_is_array_like(x) for x in leaves]
        array_keys = [k for k, m in zip(leaf_keys_full, arr_mask, strict=False) if m]
        nonarray_indices = [i for i, m in enumerate(arr_mask) if not m]
        nonarray_payload = {str(i): base64.b64encode(pickle.dumps(leaves[i])).decode("utf-8") for i in nonarray_indices}

        backend = "tensorstore"
        array_relpaths: list[str] = []

        tree_serialize_leaves(
            checkpoint_dir=root,
            pytree=pytree,
            manager=self.global_manager,
            prefix=prefix,
            write_index=write_index and write_shared_files,
        )

        self.global_manager.wait_until_finished()
        if not write_shared_files:
            return root

        index_path = _fs.joinpath(root, "tensorstore_index.json")
        if not _fs.exists(index_path):
            raise FileNotFoundError(f"Missing tensorstore_index.json in {root}")
        idx = json.loads(_fs.read_text(index_path))
        arrays_info = idx.get("prefixes", {}).get(prefix, [])
        if not arrays_info:
            raise ValueError(f"No arrays recorded in index for prefix={prefix!r}")

        relpaths_from_index = [str(info["path"]) for info in arrays_info]
        keys_from_index = [".".join(p.split("/")) for p in relpaths_from_index]
        if set(keys_from_index) != set(array_keys):
            missing = set(array_keys) - set(keys_from_index)
            extra = set(keys_from_index) - set(array_keys)
            raise ValueError(
                f"TensorStore index keys mismatch for prefix={prefix!r}. "
                f"Missing: {sorted(missing)}; Extra: {sorted(extra)}"
            )

        key_to_rel = dict(zip(keys_from_index, relpaths_from_index, strict=False))
        array_relpaths = [key_to_rel[k] for k in array_keys]

        if sum(arr_mask) != len(array_relpaths):
            raise ValueError(
                f"Structure mismatch: arr_mask expects {sum(arr_mask)} arrays, but index provided {len(array_relpaths)}."
            )

        structure: dict[str, object] = {
            "format": "pytree-structure",
            "version": __version__,
            "backend": backend,
            "prefix": prefix,
            "treedef_b64": _treedef_to_b64(treedef),
            "leaf_keys_full": leaf_keys_full,
            "arr_mask": arr_mask,
            "array_keys": array_keys,
            "array_relpaths": array_relpaths,
            "nonarray_payload": nonarray_payload,
            "safetensors_file": None,
            "extras": extras or {},
        }
        _fs.write_text(_structure_path(root, prefix), json.dumps(structure, indent=2))

        meta = CheckpointMetadata(timestamp=datetime.now().isoformat(), custom_metadata=extras)
        _fs.write_text(_fs.joinpath(root, "checkpoint_metadata.json"), json.dumps(meta.to_dict(), indent=2))

        return root

    def load_pytree(
        self,
        path: str | os.PathLike,
        mesh: Mesh,
        *,
        prefix: str,
        shardings: dict[str, tp.Callable] | None = None,
        sharding_rules: tp.Sequence[tuple[str, NamedSharding]] | None = None,
        partition_rules: tp.Sequence[tuple[str, PartitionSpec]] | None = None,
        dtype: jnp.dtype | None = None,
        template: PyTree | None = None,
        strict_shapes: bool = True,
        key_aliases: KeyAliases | None = None,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
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
    ) -> tuple[PyTree, dict]:
        """Load a PyTree previously saved by :meth:`save_pytree`.

        Reads the ``{prefix}_structure.json`` file written during save to
        reconstruct the exact treedef, then deserializes arrays via TensorStore.
        If that structure file is missing and *can_skip_structure* is ``True``,
        this falls back to an array-only nested-dict reconstruction from
        ``tensorstore_index.json``. That compatibility path cannot recover
        non-array leaves without a *template*, but it keeps index-only model
        checkpoints loadable.

        Args:
            path: Checkpoint directory (local or remote URL).
            mesh: JAX mesh used to create shardings for loaded arrays.
            prefix: Logical namespace that matches the one used during save.
            shardings: Optional mapping from leaf key strings to explicit
                :class:`~jax.sharding.NamedSharding` objects.
            sharding_rules: Optional sequence of ``(regex_pattern,
                NamedSharding)`` pairs. For each array key the first matching
                pattern determines its sharding (fallback is fully replicated).
            dtype: Optional dtype to cast loaded arrays to.
            template: Optional PyTree template for shape coercion. When given,
                loaded arrays are matched against the template leaves by key;
                shape mismatches trigger ``ValueError`` if *strict_shapes* is
                ``True``.
            strict_shapes: Whether to raise on shape mismatches when a
                *template* is provided. Defaults to ``True``.
            key_aliases: Optional function that receives a template key and
                returns alternate checkpoint keys to try when the exact key is
                absent. This keeps framework-specific legacy naming outside
                SpectraX while allowing strict template restores.
            callback: Optional per-array callback ``fn(array, key) -> array``
                invoked after each array is loaded.
            chunk_size: If set, arrays are loaded in batches of this size to
                reduce peak memory. Defaults to ``None`` (load all at once).
            can_skip_structure: If ``True``, allow loading from
                ``tensorstore_index.json`` when ``{prefix}_structure.json`` is
                absent. Defaults to ``False`` so exact treedef preservation
                remains the normal checkpoint contract.
            concurrent_gb: In-flight read budget passed to TensorStore, in
                decimal GiB units matching JAX's checkpoint manager.
            tensorstore_io_concurrency: Optional TensorStore file I/O
                concurrency limit.
            tensorstore_copy_concurrency: Optional TensorStore data-copy
                concurrency limit.
            tensorstore_cache_gb: Optional TensorStore cache-pool size in GiB.
            tensorstore_assume_metadata: If ``True``, skip sequential zarr
                metadata existence checks and embed shape/dtype metadata from
                ``tensorstore_index.json`` into TensorStore specs for faster
                opens.
            tensorstore_metadata_workers: Worker count for parallel metadata
                checks when ``tensorstore_assume_metadata`` is ``False``.
            show_progress: Whether process 0 should render a compact tqdm
                progress bar while arrays are loaded.
            progress_every: Refresh progress every N loaded tensors.

        Returns:
            A 2-tuple ``(pytree, metadata)`` where *pytree* has the exact same
            structure as the saved tree (or is coerced to *template* when
            provided), and *metadata* is the ``extras`` dict written at save
            time.

        Raises:
            ValueError: If *prefix* is empty, or if the saved prefix does not
                match the requested prefix, or if any arrays are missing.
            FileNotFoundError: If both ``{prefix}_structure.json`` and
                ``tensorstore_index.json`` are missing, or if any array
                subdirectories are missing.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        if partition_rules is not None and sharding_rules is None:
            sharding_rules = [(pat, NamedSharding(mesh=mesh, spec=spec)) for pat, spec in partition_rules]

        root = str(path)
        struct_path = _structure_path(root, prefix)
        if not _fs.exists(struct_path):
            if not can_skip_structure:
                raise FileNotFoundError(
                    f"Missing {os.path.basename(struct_path)} in {root}. "
                    "Pass can_skip_structure=True to reconstruct arrays from tensorstore_index.json."
                )
            return self._load_pytree_from_index(
                root,
                mesh,
                prefix=prefix,
                shardings=shardings,
                sharding_rules=sharding_rules,
                dtype=dtype,
                template=template,
                strict_shapes=strict_shapes,
                key_aliases=key_aliases,
                callback=callback,
                chunk_size=chunk_size,
                concurrent_gb=concurrent_gb,
                tensorstore_io_concurrency=tensorstore_io_concurrency,
                tensorstore_copy_concurrency=tensorstore_copy_concurrency,
                tensorstore_cache_gb=tensorstore_cache_gb,
                tensorstore_assume_metadata=tensorstore_assume_metadata,
                tensorstore_metadata_workers=tensorstore_metadata_workers,
                show_progress=show_progress,
                progress_every=progress_every,
            )

        struct = json.loads(_fs.read_text(struct_path))
        if struct.get("prefix") != prefix:
            raise ValueError(
                f"Structure recorded for prefix={struct.get('prefix')!r}, "
                f"but you requested prefix={prefix!r}. Use the same prefix you saved with."
            )

        treedef = _treedef_from_b64(struct["treedef_b64"])
        leaf_keys_full: list[str] = struct["leaf_keys_full"]
        arr_mask: list[bool] = struct["arr_mask"]
        if len(arr_mask) != treedef.num_leaves:
            raise ValueError(
                f"Structure/treedef mismatch: arr_mask has {len(arr_mask)} leaves, "
                f"treedef expects {treedef.num_leaves}. The structure file may be stale "
                "or saved with a different JAX PyTree definition."
            )
        array_keys: list[str] = struct["array_keys"]
        metadata = struct.get("extras", {})

        relpaths: list[str] = struct["array_relpaths"]
        if len(relpaths) != len(array_keys):
            raise ValueError("array_relpaths and array_keys length mismatch")

        abs_paths = [_fs.joinpath(root, rp) for rp in relpaths]
        index_arrays = _index_arrays_for_prefix(root, prefix)
        index_info_by_key = {_key_from_relpath(str(info.get("path", ""))): info for info in index_arrays}
        saved_shapes = [_shape_from_index(index_info_by_key.get(key, {}).get("shape")) for key in array_keys]
        storage_dtypes = [_dtype_from_index(index_info_by_key.get(key, {}).get("dtype")) for key in array_keys]
        tensorstore_metadata = (
            _zarr_metadata_for_paths(abs_paths, workers=tensorstore_metadata_workers)
            if tensorstore_assume_metadata
            else None
        )

        missing = (
            []
            if tensorstore_assume_metadata
            else _missing_zarr_metadata(abs_paths, workers=tensorstore_metadata_workers)
        )
        if missing:
            idx = _fs.joinpath(root, "tensorstore_index.json")
            prefixes = []
            if _fs.exists(idx):
                idx_data = json.loads(_fs.read_text(idx))
                prefixes = sorted(list(idx_data.get("prefixes", {}).keys()))
            raise FileNotFoundError(
                f"{len(missing)} arrays missing (example: {missing[0]}). "
                f"Check that the prefix you pass matches the one saved. "
                f"Available prefixes in this directory: {prefixes}"
            )

        sharding_lookup_keys: list[str] = _template_sharding_lookup_keys(
            array_keys,
            template=template,
            prefix=prefix,
            key_aliases=key_aliases,
        )
        apply_shardings: list[object] = _resolve_apply_shardings(
            sharding_lookup_keys,
            mesh=mesh,
            shardings=shardings,
            sharding_rules=sharding_rules,
        )

        if chunk_size is None or chunk_size <= 0:
            array_leaves = self._deserialize_with_paths(
                shardings=apply_shardings,
                paths=abs_paths,
                shapes=saved_shapes,
                storage_dtypes=storage_dtypes,
                target_dtype=dtype,
                concurrent_gb=concurrent_gb,
                tensorstore_io_concurrency=tensorstore_io_concurrency,
                tensorstore_copy_concurrency=tensorstore_copy_concurrency,
                tensorstore_cache_gb=tensorstore_cache_gb,
                tensorstore_assume_metadata=tensorstore_assume_metadata,
                show_progress=show_progress,
                progress_every=progress_every,
                tensorstore_metadata=tensorstore_metadata,
            )
            expected_arrays = sum(arr_mask)
            if len(array_leaves) != expected_arrays:
                raise ValueError(
                    f"Loaded {len(array_leaves)} arrays but structure expects {expected_arrays}. "
                    "Index or structure may be stale."
                )
            if dtype is not None:
                array_leaves = [jnp.asarray(x, dtype=dtype) for x in array_leaves]
            if callback is not None:
                array_leaves = [callback(arr, key) for arr, key in zip(array_leaves, array_keys, strict=False)]
        else:
            array_leaves_by_index: list[jax.Array | None] = [None] * len(abs_paths)
            expected_arrays = sum(arr_mask)
            load_indices = list(range(len(abs_paths)))
            if template is not None:
                load_indices.sort(
                    key=lambda index: _array_nbytes(saved_shapes[index], storage_dtypes[index]),
                    reverse=True,
                )
            for start in range(0, len(load_indices), chunk_size):
                chunk_indices = load_indices[start : start + chunk_size]
                chunk_paths = [abs_paths[index] for index in chunk_indices]
                chunk_shardings = [apply_shardings[index] for index in chunk_indices]
                chunk_keys = [array_keys[index] for index in chunk_indices]
                chunk_arrays = self._deserialize_with_paths(
                    shardings=chunk_shardings,
                    paths=chunk_paths,
                    shapes=[saved_shapes[index] for index in chunk_indices],
                    storage_dtypes=[storage_dtypes[index] for index in chunk_indices],
                    target_dtype=dtype,
                    concurrent_gb=concurrent_gb,
                    tensorstore_io_concurrency=tensorstore_io_concurrency,
                    tensorstore_copy_concurrency=tensorstore_copy_concurrency,
                    tensorstore_cache_gb=tensorstore_cache_gb,
                    tensorstore_assume_metadata=tensorstore_assume_metadata,
                    show_progress=show_progress,
                    progress_every=progress_every,
                    tensorstore_metadata=(
                        [tensorstore_metadata[index] for index in chunk_indices]
                        if tensorstore_metadata is not None
                        else None
                    ),
                )
                if dtype is not None:
                    chunk_arrays = [jnp.asarray(x, dtype=dtype) for x in chunk_arrays]
                if callback is not None:
                    chunk_arrays = [callback(arr, key) for arr, key in zip(chunk_arrays, chunk_keys, strict=False)]
                for index, array in zip(chunk_indices, chunk_arrays, strict=True):
                    array_leaves_by_index[index] = array
            array_leaves = [array for array in array_leaves_by_index if array is not None]
            if len(array_leaves) != expected_arrays:
                raise ValueError(
                    f"Loaded {len(array_leaves)} arrays but structure expects {expected_arrays}. "
                    "Index or structure may be stale."
                )

        if template is None:
            leaves_full = [None] * len(leaf_keys_full)
            it = iter(array_leaves)
            nonarray_payload: dict[str, str] = struct.get("nonarray_payload", {})
            for i, is_arr in enumerate(arr_mask):
                if is_arr:
                    leaves_full[i] = next(it)
                else:
                    payload_b64 = nonarray_payload.get(str(i))
                    if payload_b64 is None:
                        raise ValueError(f"Missing non-array payload for leaf index {i}")
                    leaves_full[i] = pickle.loads(base64.b64decode(payload_b64))
            pytree = jax.tree_util.tree_unflatten(treedef, leaves_full)
            return pytree, metadata

        saved_arrays_by_key = {k: v for k, v in zip(array_keys, array_leaves, strict=False)}

        tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template, is_leaf=_is_none)
        tpl_leaf_keys_tree = leaf_key_paths(template, prefix=prefix, is_leaf=_is_none)
        tpl_leaf_keys_full: list[str] = jax.tree_util.tree_leaves(tpl_leaf_keys_tree, is_leaf=_is_none)
        tpl_arr_mask = [_is_array_like(x) for x in tpl_leaves]

        def _coerce_or_fallback(loaded, expected, key):
            """Coerce a loaded array to match *expected* shape, or raise/fall back.

            Handles three forgiving transformations when *strict_shapes* is
            ``False``:

            1. Drop a leading singleton dimension.
            2. Reshape to matching element count.
            3. Return verbatim if shapes already match.

            When *strict_shapes* is ``True``, any mismatch raises
            :class:`ValueError`.

            Args:
                loaded: Array value read from the checkpoint.
                expected: Template leaf that provides the desired output shape.
                key: Logical checkpoint key used in shape-mismatch errors.

            Returns:
                ``loaded`` unchanged when shapes already match, a reshaped or
                squeezed view when ``strict_shapes`` is disabled and the shape
                can be reconciled, or ``expected`` as the non-strict fallback.

            Raises:
                ValueError: If both leaves are array-like, shapes differ, and
                    ``strict_shapes`` is enabled.
            """
            if not (_is_array_like(loaded) and _is_array_like(expected)):
                return loaded
            if loaded.shape == expected.shape:
                return loaded
            if not strict_shapes and (loaded.ndim == expected.ndim + 1) and (loaded.shape[1:] == expected.shape):
                return loaded[0]
            if not strict_shapes and np.prod(loaded.shape) == np.prod(expected.shape):
                return jnp.reshape(loaded, expected.shape)
            if strict_shapes:
                raise ValueError(f"Array shape mismatch for key '{key}': got {loaded.shape}, expected {expected.shape}.")
            return expected

        alias_hits = 0
        tpl_leaves_full = [None] * len(tpl_leaf_keys_full)
        for i, key in enumerate(tpl_leaf_keys_full):
            if tpl_arr_mask[i]:
                expected = tpl_leaves[i]
                loaded = saved_arrays_by_key.get(key)
                if loaded is None and key_aliases is not None:
                    for alias in key_aliases(key):
                        loaded = saved_arrays_by_key.get(alias)
                        if loaded is not None:
                            alias_hits += 1
                            break
                if loaded is None:
                    if strict_shapes:
                        raise KeyError(f"Missing array for key '{key}' in checkpoint.")
                    tpl_leaves_full[i] = expected
                else:
                    tpl_leaves_full[i] = _coerce_or_fallback(loaded, expected, key)
            else:
                tpl_leaves_full[i] = tpl_leaves[i]

        pytree = jax.tree_util.tree_unflatten(tpl_treedef, tpl_leaves_full)
        if alias_hits:
            logger.info(f"Resolved {alias_hits} checkpoint tensor(s) through caller-provided key aliases.")
        return pytree, metadata

    def _load_pytree_from_index(
        self,
        root: str,
        mesh: Mesh,
        *,
        prefix: str,
        shardings: dict[str, tp.Callable] | None = None,
        sharding_rules: tp.Sequence[tuple[str, NamedSharding]] | None = None,
        dtype: jnp.dtype | None = None,
        template: PyTree | None = None,
        strict_shapes: bool = True,
        key_aliases: KeyAliases | None = None,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        chunk_size: int | None = None,
        concurrent_gb: int = 32,
        tensorstore_io_concurrency: int | None = None,
        tensorstore_copy_concurrency: int | None = None,
        tensorstore_cache_gb: int | None = None,
        tensorstore_assume_metadata: bool = False,
        tensorstore_metadata_workers: int | None = None,
        show_progress: bool = False,
        progress_every: int = 10,
    ) -> tuple[PyTree, dict]:
        """Load an array-only PyTree from ``tensorstore_index.json``.

        This compatibility path is used for hosted checkpoints that include the
        TensorStore leaf index but not SpectraX's exact treedef sidecar. It
        reconstructs a nested dictionary by stripping the requested checkpoint
        prefix from each indexed array path. Non-array leaves are only preserved
        when a caller provides them via *template*.

        Args:
            root: Checkpoint directory containing ``tensorstore_index.json``.
            mesh: JAX mesh used to construct fallback replicated shardings.
            prefix: Logical namespace to load from the TensorStore index.
            shardings: Optional mapping from logical array keys to explicit
                destination shardings.
            sharding_rules: Optional sequence of ``(regex, NamedSharding)``
                rules. The first matching rule supplies an array's sharding.
            dtype: Optional dtype to cast loaded arrays to.
            template: Optional PyTree template used to restore non-array leaves
                and coerce loaded arrays into expected shapes.
            strict_shapes: Whether template shape mismatches should raise.
            key_aliases: Optional function that receives a template key and
                returns alternate index keys to try when the exact key is absent.
            callback: Optional per-array callback ``fn(array, key) -> array``.
            chunk_size: Optional number of arrays to load per batch.
            concurrent_gb: In-flight read budget in decimal GiB units.
            tensorstore_io_concurrency: Optional TensorStore file I/O
                concurrency limit.
            tensorstore_copy_concurrency: Optional TensorStore data-copy
                concurrency limit.
            tensorstore_cache_gb: Optional TensorStore cache-pool size in GiB.
            tensorstore_assume_metadata: Whether to embed index shape/dtype
                metadata into TensorStore specs and skip metadata reads.
            tensorstore_metadata_workers: Optional number of threads used to
                check zarr metadata existence when metadata is not assumed.
            show_progress: Whether process 0 should render a tqdm progress bar.
            progress_every: Refresh progress every N loaded tensors.

        Returns:
            Tuple of ``(pytree, metadata)``. Without a template, *pytree* is a
            nested dictionary reconstructed from indexed keys; with a template,
            it matches the template treedef.

        Raises:
            FileNotFoundError: If ``tensorstore_index.json`` or an indexed
                array path is missing.
            ValueError: If the index format is unsupported, keys are
                inconsistent, or strict template shape checks fail.
        """
        index_path = _fs.joinpath(root, "tensorstore_index.json")
        if not _fs.exists(index_path):
            raise FileNotFoundError(
                f"Missing {os.path.basename(_structure_path(root, prefix))} and tensorstore_index.json in {root}"
            )

        index_data = json.loads(_fs.read_text(index_path))
        if index_data.get("format") != "tensorstore":
            raise ValueError(f"Unsupported tensorstore index format: {index_data.get('format')!r}")

        if "prefixes" in index_data:
            prefixes = index_data["prefixes"]
            if prefix not in prefixes:
                raise ValueError(f"Prefix {prefix!r} not found in tensorstore_index.json. Available: {sorted(prefixes)}")
            arrays_info = prefixes[prefix]
        else:
            arrays_info = index_data.get("arrays", [])

        if not arrays_info:
            raise ValueError(f"No arrays recorded in tensorstore_index.json for prefix={prefix!r}.")

        logger.warning(
            "Missing %s; reconstructing prefix=%r from tensorstore_index.json. "
            "This fallback preserves arrays but cannot recover non-array leaves without a template.",
            os.path.basename(_structure_path(root, prefix)),
            prefix,
        )

        relpaths: list[str] = [info["path"] for info in arrays_info]
        array_keys = [_key_from_relpath(relpath) for relpath in relpaths]
        result_keys = [_strip_prefix_from_key(key, prefix) for key in array_keys]
        abs_paths = [_fs.joinpath(root, relpath) for relpath in relpaths]
        saved_shapes = [_shape_from_index(info.get("shape")) for info in arrays_info]
        storage_dtypes = [_dtype_from_index(info.get("dtype")) for info in arrays_info]
        tensorstore_metadata = (
            _zarr_metadata_for_paths(abs_paths, workers=tensorstore_metadata_workers)
            if tensorstore_assume_metadata
            else None
        )

        missing = (
            []
            if tensorstore_assume_metadata
            else _missing_zarr_metadata(abs_paths, workers=tensorstore_metadata_workers)
        )
        if missing:
            raise FileNotFoundError(f"{len(missing)} arrays missing (example: {missing[0]}).")

        sharding_lookup_keys = _template_sharding_lookup_keys(
            array_keys,
            template=template,
            prefix=prefix,
            key_aliases=key_aliases,
        )
        apply_shardings = _resolve_apply_shardings(
            sharding_lookup_keys,
            mesh=mesh,
            shardings=shardings,
            sharding_rules=sharding_rules,
        )

        array_leaves = []
        if chunk_size is None or chunk_size <= 0:
            array_leaves = self._deserialize_with_paths(
                shardings=apply_shardings,
                paths=abs_paths,
                shapes=saved_shapes,
                storage_dtypes=storage_dtypes,
                target_dtype=dtype,
                concurrent_gb=concurrent_gb,
                tensorstore_io_concurrency=tensorstore_io_concurrency,
                tensorstore_copy_concurrency=tensorstore_copy_concurrency,
                tensorstore_cache_gb=tensorstore_cache_gb,
                tensorstore_assume_metadata=tensorstore_assume_metadata,
                show_progress=show_progress,
                progress_every=progress_every,
                tensorstore_metadata=tensorstore_metadata,
            )
            if dtype is not None:
                array_leaves = [jnp.asarray(x, dtype=dtype) for x in array_leaves]
            if callback is not None:
                array_leaves = [callback(arr, key) for arr, key in zip(array_leaves, array_keys, strict=False)]
        else:
            for start in range(0, len(abs_paths), chunk_size):
                end = min(start + chunk_size, len(abs_paths))
                chunk_arrays = self._deserialize_with_paths(
                    shardings=apply_shardings[start:end],
                    paths=abs_paths[start:end],
                    shapes=saved_shapes[start:end],
                    storage_dtypes=storage_dtypes[start:end],
                    target_dtype=dtype,
                    concurrent_gb=concurrent_gb,
                    tensorstore_io_concurrency=tensorstore_io_concurrency,
                    tensorstore_copy_concurrency=tensorstore_copy_concurrency,
                    tensorstore_cache_gb=tensorstore_cache_gb,
                    tensorstore_assume_metadata=tensorstore_assume_metadata,
                    show_progress=show_progress,
                    progress_every=progress_every,
                    tensorstore_metadata=tensorstore_metadata[start:end] if tensorstore_metadata is not None else None,
                )
                if dtype is not None:
                    chunk_arrays = [jnp.asarray(x, dtype=dtype) for x in chunk_arrays]
                if callback is not None:
                    chunk_arrays = [
                        callback(arr, key) for arr, key in zip(chunk_arrays, array_keys[start:end], strict=False)
                    ]
                array_leaves.extend(chunk_arrays)

        if len(array_leaves) != len(array_keys):
            raise ValueError(
                f"Loaded {len(array_leaves)} arrays but tensorstore_index.json records {len(array_keys)} arrays."
            )

        metadata = _checkpoint_metadata_extras(root)
        saved_arrays_by_key = {key: value for key, value in zip(result_keys, array_leaves, strict=False)}

        if template is None:
            result: dict[str, object] = {}
            for key, value in saved_arrays_by_key.items():
                _insert_nested(result, key, value)
            return result, metadata

        tpl_leaves, tpl_treedef = jax.tree_util.tree_flatten(template, is_leaf=_is_none)
        tpl_leaf_keys_tree = leaf_key_paths(template, prefix=None, is_leaf=_is_none)
        tpl_leaf_keys_full: list[str] = jax.tree_util.tree_leaves(tpl_leaf_keys_tree, is_leaf=_is_none)
        tpl_arr_mask = [_is_array_like(x) for x in tpl_leaves]

        def _coerce_or_fallback(loaded, expected, key):
            """Coerce a loaded index-only array against a template leaf.

            Args:
                loaded: Array value read from the checkpoint.
                expected: Template leaf that provides the desired output shape.
                key: Logical checkpoint key used in error messages.

            Returns:
                ``loaded`` when shapes match, a squeezed or reshaped value when
                non-strict coercion is possible, or ``expected`` as the
                non-strict fallback for unreconcilable mismatches.

            Raises:
                ValueError: If both leaves are array-like, shapes differ, and
                    ``strict_shapes`` is enabled.
            """
            if not (_is_array_like(loaded) and _is_array_like(expected)):
                return loaded
            if loaded.shape == expected.shape:
                return loaded
            if not strict_shapes and (loaded.ndim == expected.ndim + 1) and (loaded.shape[1:] == expected.shape):
                return loaded[0]
            if not strict_shapes and np.prod(loaded.shape) == np.prod(expected.shape):
                return jnp.reshape(loaded, expected.shape)
            if strict_shapes:
                raise ValueError(f"Array shape mismatch for key '{key}': got {loaded.shape}, expected {expected.shape}.")
            return expected

        tpl_leaves_full = [None] * len(tpl_leaf_keys_full)
        alias_hits = 0
        for i, key in enumerate(tpl_leaf_keys_full):
            if tpl_arr_mask[i]:
                expected = tpl_leaves[i]
                loaded = saved_arrays_by_key.get(key)
                if loaded is None and key_aliases is not None:
                    for alias in key_aliases(key):
                        loaded = saved_arrays_by_key.get(alias)
                        if loaded is not None:
                            alias_hits += 1
                            break
                if loaded is None:
                    if strict_shapes:
                        raise KeyError(f"Missing array for key '{key}' in checkpoint.")
                    tpl_leaves_full[i] = expected
                else:
                    tpl_leaves_full[i] = _coerce_or_fallback(loaded, expected, key)
            else:
                tpl_leaves_full[i] = tpl_leaves[i]

        if alias_hits:
            logger.info(f"Resolved {alias_hits} checkpoint tensor(s) through caller-provided key aliases.")
        return jax.tree_util.tree_unflatten(tpl_treedef, tpl_leaves_full), metadata
