# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import hashlib
import json
import os
import pickle
import typing as tp
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from jax.distributed import is_initialized
from jax.experimental import multihost_utils as mh
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from safetensors import flax as safe_flax
from tqdm.autonotebook import tqdm

from eformer import __version__
from eformer.escale import create_cpu_mesh, match_partition_rules
from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.pytree import PyTree, flatten_dict, is_flatten, serialization, unflatten_dict
from eformer.serialization.serialization import leaf_key_paths

from . import fsspec_utils
from .base_manager import CheckpointManager
from .serialization import tree_deserialize_leaves, tree_serialize_leaves
from .utils import derive_base_prefix_from_path, index_filename
from .utils import read_process_array as _read_process_array
from .utils import to_host as _to_host

logger = get_logger("AsyncCheckpointManager")
GLOBAL_CHECKPOINT_TIMEOUT = int(os.getenv("GLOBAL_CHECKPOINT_TIMEOUT", "400"))


def _is_array_like(x):
    """Check if an object is array-like (has shape and dtype attributes).

    Args:
        x: Object to check.

    Returns:
        bool: True if object has both shape and dtype attributes.
    """
    return hasattr(x, "shape") and hasattr(x, "dtype")


def _treedef_to_b64(treedef) -> str:
    """Serialize a JAX tree definition to base64 string.

    Args:
        treedef: JAX tree structure definition.

    Returns:
        str: Base64 encoded string representation of the tree definition.
    """
    return base64.b64encode(pickle.dumps(treedef)).decode("utf-8")


def _treedef_from_b64(s: str):
    """Deserialize a JAX tree definition from base64 string.

    Args:
        s: Base64 encoded string containing tree definition.

    Returns:
        Deserialized JAX tree structure definition.
    """
    return pickle.loads(base64.b64decode(s.encode("utf-8")))


def _structure_path(path: ePathLike | str, prefix: str | None) -> ePath:  # type: ignore
    """Generate the structure metadata file path for a checkpoint.

    Args:
        path: Base directory path for the checkpoint.
        prefix: Optional prefix for the structure file name.

    Returns:
        ePath: Full path to the structure JSON file.
    """
    name = f"{prefix or 'pytree'}_structure.json"
    return ePath(path) / name


def _is_none(x):
    """Check if a value is None.

    Args:
        x: Value to check.

    Returns:
        bool: True if x is None, False otherwise.
    """
    return x is None


def _sync_remote_checkpoint_visibility(path: ePathLike | str, *, scope: str) -> None:
    """Block multihost callers until shared remote checkpoint files are visible."""
    if not fsspec_utils.is_remote_path(path):
        return
    if jax.process_count() <= 1 or not is_initialized():
        return

    token = hashlib.sha1(f"{scope}:{path}".encode()).hexdigest()[:12]
    mh.sync_global_devices(f"eformer-checkpoint-{scope}-{token}")


@dataclass
class CheckpointMetadata:
    """Enhanced metadata for checkpoints with versioning and validation.

    Stores comprehensive metadata about a checkpoint including version information,
    timestamps, checksums for validation, and custom user metadata.

    Attributes:
        version: Version string for the checkpoint format.
        timestamp: ISO format timestamp of when checkpoint was created.
        checksum: Dictionary mapping array keys to SHA256 checksums.
        array_metadata: Dictionary mapping array keys to shape/dtype info.
        framework_version: Version of the framework used to create checkpoint.
        custom_metadata: User-defined metadata dictionary.
    """

    version: str = __version__
    timestamp: str = None
    checksum: dict[str, str] = None
    array_metadata: dict[str, dict] = None
    framework_version: str = None
    custom_metadata: dict = None

    def to_dict(self) -> dict:
        """Convert metadata to dictionary format.

        Returns:
            Dictionary representation of the metadata.
        """
        return {
            "version": self.version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "checksum": self.checksum or {},
            "array_metadata": self.array_metadata or {},
            "framework_version": self.framework_version,
            "custom_metadata": self.custom_metadata or {},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointMetadata":
        """Create CheckpointMetadata from dictionary.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            CheckpointMetadata instance.
        """
        return cls(
            version=data.get("version", __version__),
            timestamp=data.get("timestamp"),
            checksum=data.get("checksum", {}),
            array_metadata=data.get("array_metadata", {}),
            framework_version=data.get("framework_version"),
            custom_metadata=data.get("custom_metadata", {}),
        )


class AsyncCheckpointManager:
    """Checkpoint manager with concurrent operations.

    This manager provides checkpoint saving and loading with support
    for parallel operations, tensorstore backend, validation, and compression.
    Supports both TensorStore (for large-scale distributed checkpoints) and
    SafeTensors (for smaller, single-file checkpoints) formats.

    Key Features:
        - Automatic format detection (TensorStore vs SafeTensors)
        - Parallel I/O operations for faster loading/saving
        - CPU offloading to prevent OOM on accelerators
        - Checksum validation for data integrity
        - Support for sharded checkpoints across multiple files
        - Pattern-based partition rules with preserved ordering

    Attributes:
        float_dtype: Default data type for floating point arrays.
        enable: Whether checkpointing is enabled.
        verbose: Enable verbose output.
        gcs_bucket: Google Cloud Storage bucket name.
        enable_validation: Enable checksum validation.
        enable_compression: Enable compression for tensorstore.
        use_tensorstore: Use tensorstore backend when available.

    Example:
        >>> manager = AsyncCheckpointManager(
        ...     enable_validation=True,
        ...     use_tensorstore=True
        ... )
        >>>
        >>> manager.save(model_state, "checkpoint", mesh=mesh)
        >>>
        >>> rules = [(".*kernel", PartitionSpec("model", None))]
        >>> state, meta = manager.load("checkpoint", mesh, partition_rules=rules)
    """

    def __init__(
        self,
        enable: bool | None = None,
        float_dtype: jnp.dtype = jnp.bfloat16,
        verbose: bool = False,
        gcs_bucket: str | None = None,
        gcs_credentials_path: str | None = None,
        enable_validation: bool = False,
        enable_compression: bool = False,
        use_tensorstore: bool = True,
    ):
        """Initialize the AsyncCheckpointManager.

        Args:
            enable: Whether checkpointing is enabled. If None, auto-detection is used
                based on process index during save operations.
            float_dtype: Default data type for floating point arrays when saving.
                Defaults to jnp.bfloat16 for memory efficiency.
            verbose: Enable verbose logging output. Defaults to False.
            gcs_bucket: Optional Google Cloud Storage bucket name for cloud storage.
                If provided, enables GCS integration.
            gcs_credentials_path: Optional path to GCS service account credentials JSON.
                If None and gcs_bucket is set, uses default credentials.
            enable_validation: Enable checksum validation for data integrity.
                Computes SHA256 checksums during save and verifies during load.
                Defaults to False for performance.
            enable_compression: Enable compression for TensorStore backend.
                Can reduce checkpoint size at the cost of CPU time. Defaults to False.
            use_tensorstore: Use TensorStore backend when available. TensorStore
                provides better performance for large distributed checkpoints.
                Defaults to True.

        Raises:
            RuntimeError: If running with multiple processes and JAX distributed
                is not initialized.

        Example:
            >>> manager = AsyncCheckpointManager(
            ...     float_dtype=jnp.float32,
            ...     use_tensorstore=True,
            ...     enable_validation=True
            ... )
        """
        if jax.process_count() > 1:
            if not is_initialized():
                raise RuntimeError("you should call jax distribution init before running process.")

        self.float_dtype = float_dtype
        self.enable = enable
        self.verbose = verbose
        self.gcs_bucket = gcs_bucket
        self.enable_validation = enable_validation
        self.enable_compression = enable_compression
        self.use_tensorstore = use_tensorstore

        self.gcs_client = None
        self._global_manager = None

        if gcs_bucket:
            self.gcs_client = CheckpointManager.create_gcs_client(gcs_credentials_path)

    def __del__(self):
        """Cleanup executor on deletion.

        Ensures the thread pool executor is properly shutdown when the
        manager is destroyed.
        """

    @property
    def global_manager(self) -> GlobalAsyncCheckpointManager:
        """Get or create the global async checkpoint manager.

        Returns:
            GlobalAsyncCheckpointManager: The singleton manager instance.
        """
        if self._global_manager is None:
            self._global_manager = GlobalAsyncCheckpointManager(timeout_secs=GLOBAL_CHECKPOINT_TIMEOUT)
        return self._global_manager

    @staticmethod
    def _estimate_nbytes(array: jax.Array) -> int:
        """Estimate the number of bytes in an array.

        Args:
            array: JAX array to estimate size for.

        Returns:
            Estimated number of bytes in the array.
        """
        if hasattr(array, "nbytes"):
            return array.nbytes
        elif hasattr(array, "shape") and hasattr(array, "dtype"):
            return np.prod(array.shape) * np.dtype(array.dtype).itemsize
        else:
            return 0

    def _calculate_optimal_chunks(self, shape: tuple, dtype: jnp.dtype) -> list[int] | None:
        """Calculate optimal chunk sizes for an array.

        Aims for chunks of ~64MB for optimal I/O performance. Balances between
        chunk size and number of chunks to optimize read/write operations.

        Args:
            shape: Shape of the array to chunk.
            dtype: Data type of the array.

        Returns:
            List of chunk sizes for each dimension, or None for small arrays
            that don't need chunking.

        Note:
            For very large dimensions (>10000), limits chunk size to 2000 elements
            to avoid overly large chunks.
        """
        if not shape:
            return None

        target_chunk_bytes = 64 * 1024 * 1024

        dtype_size = np.dtype(dtype).itemsize

        total_elements = np.prod(shape)
        total_bytes = total_elements * dtype_size

        if total_bytes < target_chunk_bytes:
            return None

        chunks = []
        remaining_bytes = target_chunk_bytes

        for dim_size in shape:
            elements_per_chunk = min(dim_size, max(1, remaining_bytes // dtype_size))

            if dim_size > 10000:
                elements_per_chunk = min(2000, elements_per_chunk)

            chunks.append(int(elements_per_chunk))
            remaining_bytes = remaining_bytes // max(1, (dim_size // elements_per_chunk))

        return chunks

    @staticmethod
    def compute_checksum(array: jax.Array) -> str:
        """Compute SHA256 checksum for validation.

        Converts array to bytes and computes SHA256 hash for data integrity
        verification.

        Args:
            array: JAX array to compute checksum for.

        Returns:
            SHA256 checksum as hexadecimal string.

        Note:
            Arrays are converted to numpy before hashing for consistency.
        """
        array_bytes = np.asarray(array).tobytes()
        return hashlib.sha256(array_bytes).hexdigest()

    def _validate_checkpoint(self, tree: dict, metadata: CheckpointMetadata) -> bool:
        """Validate checkpoint integrity using checksums.

        Compares computed checksums of loaded arrays against stored checksums
        in metadata to ensure data integrity.

        Args:
            tree: Dictionary containing checkpoint data.
            metadata: Checkpoint metadata containing checksums.

        Returns:
            True if validation passes, False otherwise.

        Note:
            Validation is skipped if enable_validation is False or no checksums
            are present in metadata.
        """
        if not self.enable_validation or not metadata.checksum:
            return True

        flat_tree = flatten_dict(tree) if not is_flatten(tree) else tree
        for key, array in flat_tree.items():
            if key in metadata.checksum:
                computed = self.compute_checksum(array)
                if computed != metadata.checksum[key]:
                    logger.error(f"Checksum mismatch for {key}")
                    return False
        return True

    def save_tree(
        self,
        tree: PyTree,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh | None = None,
        gather_fns: dict[tp.Callable] | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        metadata: dict[str, str] | None = None,
        callback: tp.Callable[[str], None] | None = None,
        prefix: str | None = None,
        do_all_gather: bool = False,
        cpu_offload: bool = False,
    ) -> str:
        """Save checkpoint with parallel shard writing.

        Saves a PyTree structure to disk using either TensorStore or SafeTensors format,
        with support for sharding large checkpoints and parallel I/O operations.

        Args:
            tree: PyTree structure to save.
            path: Path where the checkpoint will be saved.
            mesh: JAX mesh for distributed computation. If None, creates a CPU mesh
                with a warning about potential sharding issues.
            gather_fns: Dictionary of gather functions or bool for device gathering.
                If True, uses jax.device_get for all arrays.
            float_dtype: Data type for floating point arrays. Defaults to self.float_dtype.
            metadata: Additional metadata to save with checkpoint.
            callback: Optional callback function called after save completes.
            prefix: Optional prefix for saving specific tree (e.g., 'model', 'optimizer').
                Used for organizing multiple trees in same directory.
            do_all_gather: Whether to gather all arrays to host before saving. Defaults
                to True for safer and more consistent checkpoint saving.
            cpu_offload: Whether to offload arrays to CPU during gathering. Defaults to
                True to reduce memory pressure on accelerators and prevent OOM errors.

        Returns:
            Path where the checkpoint was saved.

        Note:
            - Automatically chooses between TensorStore (if available and enabled) or
              SafeTensors format based on configuration.
            - When mesh is not provided, a warning is logged and CPU mesh is used as fallback.
            - CPU offloading helps prevent out-of-memory errors on GPUs/TPUs during checkpointing.
            - Arrays are automatically flattened before saving and unflattened when loading.

        Example:
            >>> manager = AsyncCheckpointManager()
            >>> manager.save_tree(
            ...     tree=model_state,
            ...     path="checkpoint",
            ...     mesh=mesh,
            ...     prefix="model",
            ...     cpu_offload=True
            ... )
        """
        if float_dtype is None:
            float_dtype = self.float_dtype

        tree = serialization.to_state_dict(tree)
        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")
        if mesh is None:
            logger.warning("`mesh` should be provided otherwise you will face some sharding issues.")
            mesh = create_cpu_mesh()
        if gather_fns:
            tree = self._gather_fn(tree, gather_fns)

        if do_all_gather and not self.use_tensorstore:
            tree = jax.tree_util.tree_map(
                lambda x: _to_host(x, float_dtype, mesh, cpu_offload),
                tree,
                is_leaf=lambda x: isinstance(x, (jax.Array, np.generic, float, int)),
            )
        elif do_all_gather and self.use_tensorstore and self.verbose:
            logger.warning("Ignoring do_all_gather for TensorStore backend to preserve sharding (TP+FSDP).")

        checkpoint_meta = CheckpointMetadata(timestamp=datetime.now().isoformat(), custom_metadata=metadata)

        if self.enable_validation:
            checkpoint_meta.checksum = {k: self.compute_checksum(v) for k, v in tree.items()}
            checkpoint_meta.array_metadata = {
                k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in tree.items()
            }

        path_str = str(path)

        if self.use_tensorstore:
            out = self._save_tensorstore(tree, path_str, checkpoint_meta, prefix)
        else:
            out = self._save_single(tree, path_str, checkpoint_meta.to_dict())

        if callback:
            callback(path_str)

        return out

    save = save_tree

    def _gather_fn(self, tree: dict, gather_fns: dict[tp.Callable] | bool) -> dict:
        """Gather distributed arrays.

        Performs parallel gathering of distributed arrays using provided gather
        functions or device_get.

        Args:
            tree: Dictionary of arrays to gather.
            gather_fns: Dictionary mapping keys to gather functions, or bool.
                If True, uses jax.device_get for all arrays.
                If dict, applies specific gather function for matching keys.

        Returns:
            Dictionary with gathered arrays.

        Note:
            Arrays without matching gather functions are returned unchanged.
        """
        if isinstance(gather_fns, bool):
            results = {}
            for key, value in tree.items():
                results[key] = jax.device_get(value)
            return results

        if not is_flatten(gather_fns):
            gather_fns = flatten_dict(gather_fns, sep=".")

        results = {}
        for key, value in tree.items():
            if key in gather_fns:
                results[key] = gather_fns[key](value)
            else:
                results[key] = value

        return results

    def _save_single(self, tree: dict, path: str, metadata: dict):
        """Save a single checkpoint file using SafeTensors format.

        Saves the flattened tree dictionary to a single SafeTensors file with
        optional metadata. Non-string metadata values are JSON-serialized.

        Args:
            tree: Flattened dictionary of arrays to save. Keys should be
                dot-separated strings representing the tree path.
            path: Path where the checkpoint file will be saved.
            metadata: Dictionary of metadata to store with the checkpoint.
                Non-string values will be JSON-serialized.

        Returns:
            The path where the checkpoint was saved.

        Note:
            SafeTensors format requires all metadata values to be strings,
            so non-string values are automatically JSON-encoded.
        """
        if metadata:
            metadata = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()}

        safe_flax.save_file(tree, path, metadata)
        return path

    def _save_tensorstore(
        self,
        tree: dict,
        path: str,
        metadata: CheckpointMetadata,
        prefix: str | None = None,
    ) -> str:
        """Save using tensorstore via the core serialization module.

        Leverages TensorStore for efficient array serialization with support for
        zarr format and concurrent writes.

        Args:
            tree: Dictionary of arrays to save (flattened).
            path: Path where the checkpoint will be saved.
            metadata: Checkpoint metadata.
            prefix: Optional prefix for saving specific tree.

        Returns:
            Path where the checkpoint was saved.

        Note:
            Creates a unified index file (tensorstore_index.json) that supports
            multiple prefixes in v2.0 format. Also saves checkpoint metadata
            separately.
        """
        write_shared_files = fsspec_utils.should_write_shared_checkpoint_files(path)

        pytree = unflatten_dict(tree, sep=".")

        tree_serialize_leaves(
            checkpoint_dir=path,
            pytree=pytree,
            manager=self.global_manager,
            prefix=prefix,
            commit_callback=lambda: logger.info("Committed checkpoint to Tensorstore"),
            write_index=write_shared_files,
        )
        self.global_manager.wait_until_finished()

        logger.info("Committed checkpoint to Tensorstore")
        if write_shared_files:
            meta_path = ePath(path) / "checkpoint_metadata.json"
            meta_path.write_text(json.dumps(metadata.to_dict()))
        _sync_remote_checkpoint_visibility(path, scope=f"save-tree:{prefix or 'root'}")

        return path

    def _load_tensorstore_sync(
        self,
        path: str,
        shardings: dict[NamedSharding] | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        mesh: Mesh | None = None,
        prefix: str | None = None,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[dict, dict]:
        """Load checkpoint saved with tensorstore using core deserialization.

        Args:
            path: Path to the tensorstore checkpoint.
            shardings: PyTree of sharding specifications or dict of functions.
            partition_rules: Pattern-based partition rules.
            mesh: JAX mesh for distributed computation.
            prefix: Optional prefix for loading specific tree.
            callback: Optional callback to process each loaded array by key.
            chunk_size: Optional number of arrays to load per batch.

        Returns:
            Tuple of (loaded tree dictionary, metadata dictionary).
        """

        tree = tree_deserialize_leaves(
            checkpoint_dir=path,
            mesh=mesh,
            partition_rules=partition_rules,
            manager=self.global_manager,
            prefix=prefix,
            shardings=shardings,
            callback=callback,
            chunk_size=chunk_size,
        )
        self.global_manager.wait_until_finished()

        meta_path = ePath(path) / "checkpoint_metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {}

        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")

        return tree, metadata

    def load(
        self,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh,
        shardings: dict[NamedSharding] | None | dict[tp.Callable] = None,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        partition_rules: tuple[tuple[str, PartitionSpec]] | None = None,
        dtype: str | jnp.dtype | None = None,
        validate: bool | None = None,
        prefix_filter: str | None = None,
        prefix: str | None = None,
        use_async: bool = True,
        chunk_size: int | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Synchronous load method that can work with or without async.

        Automatically detects checkpoint format (TensorStore or SafeTensors) and
        loads accordingly. Can be called without async/await.

        Args:
            path: Path to the checkpoint directory or file.
            mesh: JAX mesh for distributed computation. Required for proper sharding.
            shardings: PyTree of sharding specifications matching checkpoint structure,
                or dict mapping keys to functions that process/reshard arrays after loading.
            mismatch_allowed: Whether to allow missing shard functions without error.
            callback: Optional callback to process each array after loading.
                Receives (array, key) and returns processed array.
            partition_rules: List of (regex, PartitionSpec) tuples for pattern-based
                sharding. Applied to arrays matching the regex patterns. Preserves
                order of arrays during loading.
            dtype: Data type to cast arrays to after loading.
            validate: Whether to validate checksums. If None, uses self.enable_validation.
            prefix_filter: Deprecated. Use 'prefix' instead.
            prefix: Optional prefix for loading specific tree (e.g., 'model', 'optimizer').
                Required when checkpoint contains multiple prefixes.
            use_async: Whether to use parallel loading (faster) or sequential loading.
            chunk_size: Optional number of arrays to load per batch when using TensorStore.

        Returns:
            Tuple of (loaded tree, metadata dictionary).
            Tree is unflattened to nested structure.

        Raises:
            ValueError: If validation fails or prefix not found.
            FileNotFoundError: If checkpoint doesn't exist.

        Note:
            - Automatically detects TensorStore format by checking for .zarray files
              or tensorstore_index.json.
            - When using partition_rules, the order of loaded arrays is preserved
              to ensure consistent sharding application.

        Example:
            >>> manager = AsyncCheckpointManager()
            >>> rules = [(".*weight", PartitionSpec("model", None))]
            >>> tree, meta = manager.load("checkpoint", mesh, partition_rules=rules)
        """
        path_str = str(path)

        is_tensorstore = False
        path_obj = ePath(path_str)
        if path_obj.is_dir():
            try:
                if fsspec_utils.exists(fsspec_utils.join_path(path_str, "tensorstore_index.json")):
                    is_tensorstore = True
                else:
                    fs, base_path = fsspec.core.url_to_fs(path_str)
                    try:
                        entries = fs.ls(base_path, detail=True)
                    except (OSError, FileNotFoundError):
                        entries = []
                    for entry in entries:
                        if isinstance(entry, dict):
                            entry_path = entry.get("name", "")
                            is_dir = entry.get("type") == "directory"
                        else:
                            entry_path = entry
                            try:
                                is_dir = fs.isdir(entry_path)
                            except (OSError, FileNotFoundError):
                                is_dir = False
                        if not is_dir:
                            continue
                        zarray_path = f"{entry_path.rstrip('/')}/.zarray"
                        if fs.exists(zarray_path):
                            is_tensorstore = True
                            break
            except (ImportError, ModuleNotFoundError):
                # Fall back to ePath for backends without fsspec support (e.g., gs:// without gcsfs).
                if (path_obj / "tensorstore_index.json").exists():
                    is_tensorstore = True
                else:
                    try:
                        for entry in path_obj.iterdir():
                            if entry.is_dir() and (entry / ".zarray").exists():
                                is_tensorstore = True
                                break
                    except (OSError, FileNotFoundError):
                        pass

        if is_tensorstore:
            if use_async:
                tree, metadata = self._load_tensorstore_sync(
                    path=path_str,
                    mesh=mesh,
                    partition_rules=partition_rules,
                    shardings=shardings,
                    prefix=prefix,
                    callback=callback,
                    chunk_size=chunk_size,
                )
            else:
                tree = tree_deserialize_leaves(
                    checkpoint_dir=path_str,
                    mesh=mesh,
                    partition_rules=partition_rules,
                    manager=self.global_manager,
                    prefix=prefix,
                    shardings=shardings,
                    callback=callback,
                    chunk_size=chunk_size,
                )
                self.global_manager.wait_until_finished()
                meta_path = path_obj / "checkpoint_metadata.json"
                if meta_path.exists():
                    metadata = json.loads(meta_path.read_text())
                else:
                    metadata = {}

            if not is_flatten(tree):
                tree = flatten_dict(tree, sep=".")
            tree = unflatten_dict(tree, sep=".")
            return tree, metadata
        else:
            return self.load_tree_parallel(
                path=path,
                shardings=shardings,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
                validate=validate,
                prefix_filter=prefix_filter,
            )

    def load_tree_parallel(
        self,
        path: ePathLike | str | os.PathLike,
        shardings: None | dict[tp.Callable] = None,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        validate: bool | None = None,
        prefix_filter: str | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load checkpoint with parallel shard reading.

        Args:
            path: Path to the checkpoint.
            shardings: PyTree of sharding specifications or dict of functions.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.
            validate: Whether to validate checksums.
            prefix_filter: Optional prefix to filter shards.

        Returns:
            Tuple of (loaded tree, metadata dictionary).

        Raises:
            ValueError: If checkpoint validation fails.
        """
        validate = validate if validate is not None else self.enable_validation

        path_str = str(path)
        base_prefix = derive_base_prefix_from_path(path_str)
        index_path_str = index_filename(base_prefix)

        if ePath(index_path_str).exists():
            return self._load_sharded_parallel(
                index_path_str,
                shardings,
                mismatch_allowed,
                callback,
                dtype,
                validate,
                prefix_filter,
            )

        tree, metadata = CheckpointManager.load_checkpoint(
            path,
            shardings,
            self.verbose,
            mismatch_allowed,
            callback,
            dtype,
            self.gcs_client,
        )

        if validate and metadata:
            meta = CheckpointMetadata.from_dict(metadata)
            if not self._validate_checkpoint(tree, meta):
                raise ValueError("Checkpoint validation failed")

        return tree, metadata

    def _load_sharded_parallel(
        self,
        index_path: str,
        shardings: PyTree | dict[tp.Callable] | None,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
        validate: bool,
        prefix_filter: str | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load sharded checkpoint with parallel reads.

        Args:
            index_path: Path to the index file.
            shardings: PyTree of sharding specifications or dict of functions.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.
            validate: Whether to validate checksums.
            prefix_filter: Optional prefix to filter loaded keys (deprecated).

        Returns:
            Tuple of (loaded tree, metadata dictionary).

        Raises:
            ValueError: If checkpoint validation fails.
        """
        index_data = json.loads(ePath(index_path).read_text())

        weight_map: dict[str, str] = index_data.get("weight_map", {})
        directory = str(ePath(index_path).parent)

        file_to_keys: dict[str, list[str]] = defaultdict(list)
        for k, shard_name in weight_map.items():
            file_to_keys[shard_name].append(k)

        shard_fns = None
        if shardings:
            if isinstance(shardings, dict) and not any(isinstance(v, dict) for v in shardings.values()):
                shard_fns = shardings
            else:
                if not is_flatten(shardings):
                    shard_fns = flatten_dict(shardings, sep=".")
                else:
                    shard_fns = shardings

        tree = {}

        for shard_name, keys in tqdm(
            file_to_keys.items(),
            desc="Loading shards",
            disable=not self.verbose,
        ):
            shard_path = str(ePath(directory) / shard_name)
            shard_tree = self._load_shard_file(
                shard_path,
                keys,
                shard_fns,
                mismatch_allowed,
                callback,
                dtype,
            )
            tree.update(shard_tree)

        tree = unflatten_dict(tree, sep=".")
        metadata = index_data.get("metadata", {})

        if validate and metadata:
            meta = CheckpointMetadata.from_dict(metadata)
            if not self._validate_checkpoint(tree, meta):
                raise ValueError("Checkpoint validation failed")

        return tree, metadata

    def _load_shard_file(
        self,
        shard_path: str,
        keys: list[str],
        shard_fns: dict | None,
        mismatch_allowed: bool,
        callback: tp.Callable | None,
        dtype: str | jnp.dtype | None,
    ) -> dict:
        """Load a single shard file.

        Args:
            shard_path: Path to the shard file.
            keys: List of keys to load from the shard.
            shard_fns: Flat dictionary of functions to apply to shards.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays.
            dtype: Data type to cast arrays to.

        Returns:
            Dictionary with loaded tensors.
        """
        shard_tree = {}
        with safe_flax.safe_open(shard_path, framework="flax") as manager:
            process_func = partial(
                _read_process_array,
                shard_fns=shard_fns,
                mismatch_allowed=mismatch_allowed,
                manager=manager,
                callback=callback,
                dtype=dtype,
            )
            for key in keys:
                k, tensor, _ = process_func(key)
                shard_tree[k] = tensor
        return shard_tree

    @staticmethod
    def is_tensorstore(path) -> bool:
        """Check if a checkpoint path uses TensorStore format.

        Args:
            path: Path to check for TensorStore format.

        Returns:
            bool: True if the path contains or points to a TensorStore checkpoint.
        """
        if str(path).endswith("tensorstore_index.json"):
            return True
        return (ePath(path) / "tensorstore_index.json").exists()

    @staticmethod
    def safe_loadpath(path) -> ePathLike:
        """Convert a checkpoint path to a safe loadable format.

        Strips TensorStore index filename if present to get the base directory.

        Args:
            path: Checkpoint path that may include index filename.

        Returns:
            ePathLike: Cleaned path suitable for loading.
        """
        p = str(path)
        if AsyncCheckpointManager.is_tensorstore(p) and p.endswith("tensorstore_index.json"):
            return ePath(p[: -len("tensorstore_index.json")])
        return ePath(p)

    def save_pytree(
        self,
        pytree: PyTree,
        path: ePathLike | str,
        mesh: Mesh | None = None,
        *,
        prefix: str,
        do_all_gather: bool = False,
        cpu_offload: bool = True,
        dtype: jnp.dtype | None = None,
        extras: dict | None = None,
        write_index: bool = True,
    ) -> str:
        """Save a PyTree with exact structure and prefix.

        Saves a PyTree structure to disk with support for both TensorStore and SafeTensors
        backends. Arrays are saved to <path>/<prefix>/..., while index and structure
        metadata go to <path>/.

        Args:
            pytree: PyTree structure to save.
            path: Directory path where the checkpoint will be saved.
            mesh: JAX mesh for distributed computation.
            prefix: Required prefix for organizing the saved tree (e.g., 'model', 'optimizer').
            do_all_gather: Whether to gather all arrays to host before saving.
            cpu_offload: Whether to offload arrays to CPU during gathering to prevent OOM.
            dtype: Optional data type to cast arrays to before saving.
            extras: Additional metadata to save with the checkpoint.
            write_index: Whether to write the index file (for TensorStore backend).

        Returns:
            str: Path where the checkpoint was saved.

        Raises:
            ValueError: If prefix is empty or not a string.
            FileNotFoundError: If TensorStore index creation fails.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        root = ePath(path)
        write_shared_files = fsspec_utils.should_write_shared_checkpoint_files(path)
        if write_shared_files:
            root.mkdir(parents=True, exist_ok=True)

        if do_all_gather:
            use_dtype = dtype or self.float_dtype
            pytree = jax.tree_util.tree_map(
                lambda x: (_is_array_like(x) and _to_host(x, use_dtype, mesh, cpu_offload)) or x,
                pytree,
                is_leaf=lambda x: isinstance(x, (jax.Array, np.generic, float, int)),
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

        backend = "tensorstore" if self.use_tensorstore else "safetensors"
        array_relpaths: list[str] = []
        safetensors_file = None

        tree_serialize_leaves(
            checkpoint_dir=str(root),
            pytree=pytree,
            manager=self.global_manager,
            prefix=prefix,
            write_index=write_index and write_shared_files,
        )

        self.global_manager.wait_until_finished()
        if not write_shared_files:
            return str(root)

        index_path = root / "tensorstore_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing tensorstore_index.json in {root}")
        idx = json.loads(index_path.read_text())
        arrays_info = idx.get("prefixes", {}).get(prefix, [])
        if not arrays_info:
            raise ValueError(f"No arrays recorded in index for prefix={prefix!r}")

        relpaths_from_index = [info["path"] for info in arrays_info]
        keys_from_index = [".".join(Path(rp).parts) for rp in relpaths_from_index]
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

        structure = {
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
            "safetensors_file": str(safetensors_file) if safetensors_file else None,
            "extras": extras or {},
        }
        _structure_path(root, prefix).write_text(json.dumps(structure, indent=2))

        meta = CheckpointMetadata(timestamp=datetime.now().isoformat(), custom_metadata=extras)
        (root / "checkpoint_metadata.json").write_text(json.dumps(meta.to_dict(), indent=2))

        return str(root)

    def load_pytree(
        self,
        path: ePathLike | str,
        mesh: Mesh,
        *,
        prefix: str,
        shardings: dict[str, tp.Callable] | None = None,
        partition_rules: tp.Sequence[tuple[str, PartitionSpec]] | None = None,
        dtype: jnp.dtype | None = None,
        template: PyTree | None = None,
        strict_shapes: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[PyTree, dict]:
        """Load a PyTree saved by save_pytree with the same prefix.

        Loads a PyTree structure from disk that was previously saved with save_pytree.
        Supports both TensorStore and SafeTensors backends with automatic detection.

        Args:
            path: Directory path containing the saved checkpoint.
            mesh: JAX mesh for distributed computation and array sharding.
            prefix: Required prefix that must match the one used during save.
            shardings: Optional dictionary mapping array keys to sharding functions.
            partition_rules: Optional sequence of (regex, PartitionSpec) tuples for
                pattern-based array sharding.
            dtype: Optional data type to cast arrays to after loading.
            callback: Optional callback to process each loaded array by key.
            chunk_size: Optional number of arrays to load per batch.

        Returns:
            Tuple of (loaded PyTree, metadata dictionary).

        Raises:
            ValueError: If prefix is empty, doesn't match saved prefix, or data is corrupted.
            FileNotFoundError: If structure file or arrays are missing.
        """
        if not prefix or not isinstance(prefix, str):
            raise ValueError("A non-empty string prefix is required")

        root = ePath(path)
        struct_path = _structure_path(root, prefix)
        if not struct_path.exists():
            raise FileNotFoundError(f"Missing pytree_structure.json in {root}")

        struct = json.loads(struct_path.read_text())
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

        def default_sharding():
            return NamedSharding(mesh=mesh, spec=PartitionSpec())

        relpaths: list[str] = struct["array_relpaths"]
        if len(relpaths) != len(array_keys):
            raise ValueError("array_relpaths and array_keys length mismatch")

        abs_paths = [str(root / rp) for rp in relpaths]

        missing = [p for p in abs_paths if not (ePath(p) / ".zarray").exists()]
        if missing:
            idx = root / "tensorstore_index.json"
            prefixes = []
            if idx.exists():
                idx_data = json.loads(idx.read_text())
                prefixes = sorted(list(idx_data.get("prefixes", {}).keys()))
            raise FileNotFoundError(
                f"{len(missing)} arrays missing (example: {missing[0]}). "
                f"Check that the prefix you pass matches the one saved. "
                f"Available prefixes in this directory: {prefixes}"
            )

        if partition_rules is not None:
            matched = match_partition_rules(
                partition_rules,
                {k.replace(".", "/"): i for i, k in enumerate(array_keys)},
                strict=False,
            )
            apply_shardings = [NamedSharding(mesh=mesh, spec=matched[k.replace(".", "/")]) for k in array_keys]
        else:
            apply_shardings = [
                shardings.get(k, default_sharding()) if shardings else default_sharding() for k in array_keys
            ]

        if chunk_size is None or chunk_size <= 0:
            array_leaves = self.global_manager.deserialize_with_paths(shardings=apply_shardings, paths=abs_paths)
            self.global_manager.wait_until_finished()
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
            array_leaves = []
            expected_arrays = sum(arr_mask)
            for start in range(0, len(abs_paths), chunk_size):
                end = min(start + chunk_size, len(abs_paths))
                chunk_paths = abs_paths[start:end]
                chunk_shardings = apply_shardings[start:end]
                chunk_keys = array_keys[start:end]
                chunk_arrays = self.global_manager.deserialize_with_paths(shardings=chunk_shardings, paths=chunk_paths)
                self.global_manager.wait_until_finished()
                if dtype is not None:
                    chunk_arrays = [jnp.asarray(x, dtype=dtype) for x in chunk_arrays]
                if callback is not None:
                    chunk_arrays = [callback(arr, key) for arr, key in zip(chunk_arrays, chunk_keys, strict=False)]
                array_leaves.extend(chunk_arrays)
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
        for i, key in enumerate(tpl_leaf_keys_full):
            if tpl_arr_mask[i]:
                expected = tpl_leaves[i]
                loaded = saved_arrays_by_key.get(key)
                if loaded is None:
                    if strict_shapes:
                        raise KeyError(f"Missing array for key '{key}' in checkpoint.")
                    tpl_leaves_full[i] = expected
                else:
                    tpl_leaves_full[i] = _coerce_or_fallback(loaded, expected, key)
            else:
                tpl_leaves_full[i] = tpl_leaves[i]

        pytree = jax.tree_util.tree_unflatten(tpl_treedef, tpl_leaves_full)
        return pytree, metadata
