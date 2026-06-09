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


import io
import json
import os
import tempfile
import typing as tp
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import msgpack
import numpy
from google.cloud import storage
from jax.sharding import Mesh
from safetensors import flax as safe_flax
from tqdm.autonotebook import tqdm

from eformer.loggings import get_logger
from eformer.paths import ePath, ePathLike
from eformer.pytree import PyTree, flatten_dict, is_flatten, serialization, unflatten_dict

from .utils import (
    derive_base_prefix_from_path,
    estimate_array_nbytes,
    group_keys_by_shard_size,
    index_filename,
    is_gcs_path,
    parse_gcs_path,
    put_dtype,
    shard_filename,
)
from .utils import read_process_array as _read_process_array
from .utils import to_host as _to_host

logger = get_logger(__name__)

ALLOWED_DATA_TYPES = [
    jnp.int4,
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.int64,
    jnp.uint4,
    jnp.uint8,
    jnp.uint16,
    jnp.uint32,
    jnp.uint64,
    jnp.float16,
    jnp.float32,
    jnp.float64,
    jnp.bfloat16,
    jnp.float_,
]


class CheckpointManager:
    """Base checkpoint manager for saving and loading PyTree structures.

    This manager provides functionality for saving and loading checkpoints
    with support for sharding, Google Cloud Storage, and various data formats.

    Attributes:
        float_dtype: Default data type for floating point arrays.
        save_optimizer_state: Whether to save optimizer state.
        checkpoint_dir: Directory for saving checkpoints.
        enable: Whether checkpointing is enabled.
        verbose: Enable verbose output.
        gcs_bucket: Google Cloud Storage bucket name.
        gcs_client: GCS client instance.
    """

    def __init__(
        self,
        checkpoint_dir: ePathLike | str | os.PathLike,
        enable: bool | None = None,
        float_dtype: jnp.dtype = jnp.bfloat16,
        save_optimizer_state: bool = True,
        verbose: bool = False,
        gcs_bucket: str | None = None,
        gcs_credentials_path: str | None = None,
    ):
        """Initialize the CheckpointManager.

        Args:
            checkpoint_dir: Directory for saving/loading checkpoints. Can be a local
                path or a GCS path (gs://bucket/path).
            enable: Whether checkpointing is enabled. If None, auto-detection is used
                based on process index during save operations.
            float_dtype: Default data type for floating point arrays when saving.
                Defaults to jnp.bfloat16 for memory efficiency.
            save_optimizer_state: Whether to include optimizer state in checkpoints.
                Defaults to True.
            verbose: Enable verbose logging output. Defaults to False.
            gcs_bucket: Optional Google Cloud Storage bucket name for cloud storage.
                If provided, enables GCS integration.
            gcs_credentials_path: Optional path to GCS service account credentials JSON.
                If None and gcs_bucket is set, uses default credentials.

        Example:
            >>> manager = CheckpointManager(
            ...     checkpoint_dir="/checkpoints",
            ...     float_dtype=jnp.float32,
            ...     verbose=True
            ... )
            >>> # Or with GCS
            >>> manager = CheckpointManager(
            ...     checkpoint_dir="gs://my-bucket/checkpoints",
            ...     gcs_bucket="my-bucket"
            ... )
        """
        self.float_dtype = float_dtype
        self.save_optimizer_state = save_optimizer_state
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable
        self.verbose = verbose
        self.gcs_bucket = gcs_bucket

        self.gcs_client = None

        if gcs_bucket:
            self.gcs_client = self.create_gcs_client(gcs_credentials_path)

    _estimate_nbytes = staticmethod(estimate_array_nbytes)
    _group_keys_by_shard_size = staticmethod(group_keys_by_shard_size)
    _derive_base_prefix_from_path = staticmethod(derive_base_prefix_from_path)
    _shard_filename = staticmethod(shard_filename)
    _index_filename = staticmethod(index_filename)
    _is_gcs_path = staticmethod(is_gcs_path)
    _parse_gcs_path = _parse_gcs_path_static = staticmethod(parse_gcs_path)

    @staticmethod
    def create_gcs_client(gcs_credentials_path: str | None = None):
        """Create a Google Cloud Storage client.

        Args:
            gcs_credentials_path: Optional path to service account credentials.

        Returns:
            Google Cloud Storage client instance.
        """
        if gcs_credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(gcs_credentials_path)
            gcs_client = storage.Client(credentials=credentials)
        else:
            gcs_client = storage.Client()
        return gcs_client

    @staticmethod
    def load_checkpoint(
        path: ePathLike | str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        gcs_client: storage.Client | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load a checkpoint from local path or GCS.

        Supports:
          - Single safetensors file
          - Sharded safetensors with index (prefix.safetensors.index.json)

        Args:
            path: Path to the checkpoint file or directory.
            shard_fns: Dictionary of functions to apply to specific shards.
            verbose: Enable verbose output.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays after loading.
            dtype: Data type to cast arrays to.
            gcs_client: Optional GCS client instance.

        Returns:
            Tuple of (loaded PyTree or dict, metadata dict).
        """
        path_str = str(path)
        base_prefix = CheckpointManager._derive_base_prefix_from_path(path_str)
        index_path_str = CheckpointManager._index_filename(base_prefix)
        is_gcs = CheckpointManager._is_gcs_path(path)

        if is_gcs and gcs_client is None:
            gcs_client = storage.Client()

        index_exists = False
        if path_str.endswith(".safetensors.index.json"):
            index_exists = True
        else:
            if is_gcs:
                bucket_name, blob_name = CheckpointManager._parse_gcs_path_static(index_path_str)
                bucket = gcs_client.bucket(bucket_name)
                index_blob = bucket.blob(blob_name)
                index_exists = index_blob.exists()
            else:
                index_exists = os.path.exists(index_path_str)

        if index_exists:
            if is_gcs:
                return CheckpointManager._load_sharded_from_gcs_index(
                    index_gcs_path=index_path_str,
                    shard_fns=shard_fns,
                    verbose=verbose,
                    mismatch_allowed=mismatch_allowed,
                    callback=callback,
                    dtype=dtype,
                    gcs_client=gcs_client,
                )
            else:
                return CheckpointManager._load_sharded_from_local_index(
                    index_path=index_path_str,
                    shard_fns=shard_fns,
                    verbose=verbose,
                    mismatch_allowed=mismatch_allowed,
                    callback=callback,
                    dtype=dtype,
                )

        if is_gcs:
            if isinstance(path, ePathLike):
                try:
                    blob = path.blob
                except Exception:
                    bucket_name, blob_name = CheckpointManager._parse_gcs_path_static(path_str)
                    bucket = gcs_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
            else:
                bucket_name, blob_name = CheckpointManager._parse_gcs_path_static(path_str)
                bucket = gcs_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
                blob.download_to_filename(temp_file.name)
                temp_path = temp_file.name
            try:
                return CheckpointManager._load_checkpoint_from_file(
                    path=temp_path,
                    shard_fns=shard_fns,
                    verbose=verbose,
                    mismatch_allowed=mismatch_allowed,
                    callback=callback,
                    dtype=dtype,
                )
            finally:
                os.unlink(temp_path)
        else:
            return CheckpointManager._load_checkpoint_from_file(
                path=path_str,
                shard_fns=shard_fns,
                verbose=verbose,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
            )

    @staticmethod
    def _load_sharded_from_local_index(
        index_path: str,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load sharded checkpoint from local index file.

        Args:
            index_path: Path to the index JSON file.
            shard_fns: Dictionary of functions to apply to specific shards.
            verbose: Enable verbose output.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays after loading.
            dtype: Data type to cast arrays to.

        Returns:
            Tuple of (loaded PyTree or dict, metadata dict).
        """
        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_map: dict[str, str] = index_data.get("weight_map", {})
        directory = os.path.dirname(index_path)
        file_to_keys: dict[str, list[str]] = defaultdict(list)
        for k, shard_name in weight_map.items():
            file_to_keys[shard_name].append(k)

        if shard_fns and not is_flatten(shard_fns):
            shard_fns = flatten_dict(shard_fns, sep=".")

        total_keys = sum(len(v) for v in file_to_keys.values())
        tree: dict[str, jax.Array] = {}
        mismatch_count = 0

        pbar = tqdm(total=total_keys, desc="Loading shards", disable=not verbose)
        for shard_name, keys in file_to_keys.items():
            shard_path = os.path.join(directory, shard_name)
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
                    k, tensor, mm = process_func(key)
                    tree[k] = tensor
                    mismatch_count += mm
                    pbar.update(1)
        pbar.close()

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        tree = unflatten_dict(tree, sep=".")
        metadata = index_data.get("metadata", {})
        return tree, metadata

    @staticmethod
    def _load_sharded_from_gcs_index(
        index_gcs_path: str,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        gcs_client: storage.Client | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load sharded checkpoint from GCS index file.

        Args:
            index_gcs_path: GCS path to the index JSON file.
            shard_fns: Dictionary of functions to apply to specific shards.
            verbose: Enable verbose output.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays after loading.
            dtype: Data type to cast arrays to.
            gcs_client: Optional GCS client instance.

        Returns:
            Tuple of (loaded PyTree or dict, metadata dict).
        """
        if gcs_client is None:
            gcs_client = storage.Client()

        bucket_name, blob_name = CheckpointManager._parse_gcs_path_static(index_gcs_path)
        bucket = gcs_client.bucket(bucket_name)
        index_blob = bucket.blob(blob_name)
        index_bytes = index_blob.download_as_bytes()
        index_data = json.loads(index_bytes.decode("utf-8"))

        weight_map: dict[str, str] = index_data.get("weight_map", {})
        shard_dir = os.path.dirname(blob_name)

        file_to_keys: dict[str, list[str]] = defaultdict(list)
        for k, shard_name in weight_map.items():
            file_to_keys[shard_name].append(k)

        if shard_fns and not is_flatten(shard_fns):
            shard_fns = flatten_dict(shard_fns, sep=".")

        total_keys = sum(len(v) for v in file_to_keys.values())
        tree: dict[str, jax.Array] = {}
        mismatch_count = 0

        pbar = tqdm(total=total_keys, desc="Loading shards (GCS)", disable=not verbose)
        for shard_name, keys in file_to_keys.items():
            shard_blob_path = shard_name if not shard_dir else f"{shard_dir}/{shard_name}"
            shard_blob = bucket.blob(shard_blob_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
                shard_blob.download_to_filename(temp_file.name)
                temp_path = temp_file.name

            try:
                with safe_flax.safe_open(temp_path, framework="flax") as manager:
                    process_func = partial(
                        _read_process_array,
                        shard_fns=shard_fns,
                        mismatch_allowed=mismatch_allowed,
                        manager=manager,
                        callback=callback,
                        dtype=dtype,
                    )
                    for key in keys:
                        k, tensor, mm = process_func(key)
                        tree[k] = tensor
                        mismatch_count += mm
                        pbar.update(1)
            finally:
                os.unlink(temp_path)
        pbar.close()

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        tree = unflatten_dict(tree, sep=".")
        metadata = index_data.get("metadata", {})
        return tree, metadata

    @classmethod
    def save_checkpoint(
        cls,
        tree: PyTree,
        path: ePathLike | str | os.PathLike,
        mesh: Mesh,
        gather_fns: dict[tp.Callable] | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        metadata: dict[str, str] | None = None,
        enable: bool | None = None,
        shard_size_gb: float | None = 5.00,
        write_index_file: bool = True,
    ) -> ePathLike | str | os.PathLike:
        """Save a checkpoint to local path or GCS using SafeTensors.

        If shard_size_gb is provided, the tree is saved as multiple shards of up to that size
        (except the last shard, which may be smaller). An index file 'prefix.safetensors.index.json'
        is also written mapping every tensor name to a shard file.

        Args:
            tree: PyTree structure to save.
            path: Path where the checkpoint will be saved.
            mesh: JAX mesh for distributed computation.
            gather_fns: Dictionary of gather functions or bool for device gathering.
            float_dtype: Data type for floating point arrays.
            verbose: Enable verbose output.
            mismatch_allowed: Whether to allow missing gather functions.
            metadata: Additional metadata to save with checkpoint.
            enable: Whether checkpointing is enabled (None = auto-detect process 0).
            shard_size_gb: Maximum size of each shard in GB.
            write_index_file: Whether to write the index file for sharded saves.

        Returns:
            Path where the checkpoint was saved.
        """
        if enable is None:
            enable = jax.process_index() == 0
        if not enable:
            path = "/dev/null"
        if str(path).startswith("/dev/null"):
            path = "/dev/null"

        if float_dtype is None:
            float_dtype = jnp.bfloat16

        tree = serialization.to_state_dict(tree)
        if not is_flatten(tree):
            tree = flatten_dict(tree, sep=".")

        gather_mismatch_count = 0
        if gather_fns:
            pbar_gather = tqdm(list(tree.keys()), desc="Gathering State", disable=not verbose)
            if isinstance(gather_fns, bool):
                for key in pbar_gather:
                    pbar_gather.update(1)
                    tree[key] = jax.device_get(tree[key])
            else:
                if not is_flatten(gather_fns):
                    gather_fns = flatten_dict(gather_fns, sep=".")

                for key in pbar_gather:
                    callable_func = gather_fns.get(key, None)
                    if callable_func is None:
                        if not mismatch_allowed:
                            raise KeyError(f"Gather Function {key} missing.")
                        gather_mismatch_count += 1
                    else:
                        tree[key] = callable_func(tree[key])

                    pbar_gather.set_postfix(gather_mismatch=gather_mismatch_count)
                    pbar_gather.update(1)

        tree = jax.tree_util.tree_map(
            lambda x: _to_host(x, float_dtype, mesh),
            tree,
            is_leaf=lambda x: isinstance(x, jax.Array | numpy.generic | float | int),
        )

        path_str = str(path)

        if shard_size_gb is not None and shard_size_gb > 0:
            max_bytes = int(shard_size_gb * (1024**3))
            flat_state = tree if is_flatten(tree) else flatten_dict(tree, sep=".")
            shards = cls._group_keys_by_shard_size(flat_state, max_bytes)
            base_prefix = cls._derive_base_prefix_from_path(path_str)
            index_path = cls._index_filename(base_prefix)

            weight_map: dict[str, str] = {}
            total_shards = len(shards)

            if cls._is_gcs_path(path_str):
                cls._save_sharded_to_gcs(
                    flat_state=flat_state,
                    base_prefix=base_prefix,
                    shards=shards,
                    total_shards=total_shards,
                    metadata=metadata,
                    verbose=verbose,
                )
            else:
                cls._save_sharded_to_local(
                    flat_state=flat_state,
                    base_prefix=base_prefix,
                    shards=shards,
                    total_shards=total_shards,
                    metadata=metadata,
                    verbose=verbose,
                )

            for i, shard_keys in enumerate(shards, start=1):
                shard_name = ePath(cls._shard_filename(base_prefix, i, total_shards)).name
                for k in shard_keys:
                    weight_map[k] = shard_name

            if write_index_file:
                index_data = {"metadata": metadata or {}, "weight_map": weight_map}
                if cls._is_gcs_path(path_str):
                    gcs_client = cls.create_gcs_client()
                    idx_bucket, idx_blob = cls._parse_gcs_path(cls._index_filename(base_prefix))
                    bucket = gcs_client.bucket(idx_bucket)
                    blob = bucket.blob(idx_blob)
                    blob.upload_from_string(
                        json.dumps(index_data, ensure_ascii=False).encode("utf-8"),
                        content_type="application/json",
                    )
                else:
                    with open(index_path, "w", encoding="utf-8") as f:
                        json.dump(index_data, f, ensure_ascii=False)

            return cls._index_filename(base_prefix)

        if cls._is_gcs_path(path_str):
            return cls._save_to_gcs(tree, path_str, metadata, verbose)
        else:
            safe_flax.save_file(tensors=tree, filename=path_str, metadata=metadata)
            return path

    @classmethod
    def _save_sharded_to_local(
        cls,
        flat_state: dict[str, jax.Array],
        base_prefix: str,
        shards: list[list[str]],
        total_shards: int,
        metadata: dict[str, str] | None,
        verbose: bool = True,
    ) -> None:
        """Save sharded checkpoint to local filesystem.

        Args:
            flat_state: Flattened state dictionary to save.
            base_prefix: Base prefix for shard filenames.
            shards: List of lists containing keys for each shard.
            total_shards: Total number of shards.
            metadata: Optional metadata to save with each shard.
            verbose: Enable verbose output.
        """
        for i, shard_keys in enumerate(tqdm(shards, desc="Saving shards", disable=not verbose), start=1):
            shard_path = cls._shard_filename(base_prefix, i, total_shards)
            shard_tensors = {k: flat_state[k] for k in shard_keys}
            safe_flax.save_file(tensors=shard_tensors, filename=shard_path, metadata=metadata)

    @classmethod
    def _save_sharded_to_gcs(
        cls,
        flat_state: dict[str, jax.Array],
        base_prefix: str,
        shards: list[list[str]],
        total_shards: int,
        metadata: dict[str, str] | None,
        verbose: bool = True,
    ) -> None:
        """Save sharded checkpoint to Google Cloud Storage.

        Args:
            flat_state: Flattened state dictionary to save.
            base_prefix: Base prefix for shard filenames.
            shards: List of lists containing keys for each shard.
            total_shards: Total number of shards.
            metadata: Optional metadata to save with each shard.
            verbose: Enable verbose output.
        """
        gcs_client = cls.create_gcs_client()
        bucket_name, base_blob_name = cls._parse_gcs_path(base_prefix + ".txt")
        base_dir = os.path.dirname(base_blob_name)
        bucket = gcs_client.bucket(bucket_name)

        for i, shard_keys in enumerate(tqdm(shards, desc="Saving shards to GCS", disable=not verbose), start=1):
            shard_name = ePath(cls._shard_filename(base_prefix, i, total_shards)).name
            shard_blob_name = f"{base_dir}/{shard_name}" if base_dir else shard_name
            shard_blob = bucket.blob(shard_blob_name)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
                shard_path = temp_file.name
            try:
                shard_tensors = {k: flat_state[k] for k in shard_keys}
                safe_flax.save_file(tensors=shard_tensors, filename=shard_path, metadata=metadata)
                shard_blob.upload_from_filename(shard_path)
            finally:
                if os.path.exists(shard_path):
                    os.unlink(shard_path)

    @staticmethod
    def _load_checkpoint_from_file(
        path: ePathLike | str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
    ) -> tuple[PyTree | dict, dict]:
        """Load checkpoint from a single local file.

        Args:
            path: Path to the checkpoint file.
            shard_fns: Dictionary of functions to apply to specific shards.
            verbose: Enable verbose output.
            mismatch_allowed: Whether to allow missing shard functions.
            callback: Optional callback to process arrays after loading.
            dtype: Data type to cast arrays to.

        Returns:
            Tuple of (loaded PyTree or dict, metadata dict).
        """
        with safe_flax.safe_open(str(path), framework="flax") as f:
            metadata = f.metadata()
            keys = list(f.keys())

            if shard_fns and not is_flatten(shard_fns):
                shard_fns = flatten_dict(shard_fns, sep=".")

            process_func = partial(
                _read_process_array,
                shard_fns=shard_fns,
                mismatch_allowed=mismatch_allowed,
                manager=f,
                callback=callback,
                dtype=dtype,
            )
            results = [process_func(key) for key in tqdm(keys, desc="Loading", total=len(keys), disable=not verbose)]

        tree = {key: tensor for key, tensor, _ in results}
        mismatch_count = sum(mismatch for _, _, mismatch in results)

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        tree = unflatten_dict(tree, sep=".")
        return tree, metadata

    @classmethod
    def _save_to_gcs(
        cls,
        tree: dict,
        gcs_path: str,
        metadata: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> str:
        """Save tree to GCS using temporary file.

        Args:
            tree: Dictionary to save.
            gcs_path: GCS path where the checkpoint will be saved.
            metadata: Optional metadata to save with checkpoint.
            verbose: Enable verbose output.

        Returns:
            GCS path where the checkpoint was saved.
        """
        gcs_client = cls.create_gcs_client()

        bucket_name, blob_name = cls._parse_gcs_path(gcs_path)
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
            safe_flax.save_file(tensors=tree, filename=temp_file.name, metadata=metadata)
            temp_path = temp_file.name

        try:
            if verbose:
                logger.info(f"Uploading checkpoint to {gcs_path}")
            blob.upload_from_filename(temp_path)
            return gcs_path
        finally:
            os.unlink(temp_path)

    def save_state_to_gcs_msgpack(
        self,
        tree: PyTree,
        gcs_path: str,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: str | jnp.dtype | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
    ):
        """Save tree to GCS using msgpack format (streaming).

        Args:
            tree: PyTree structure to save.
            gcs_path: GCS path where the checkpoint will be saved.
            gather_fns: Dictionary of gather functions.
            float_dtype: Data type for floating point arrays.
            verbose: Enable verbose output.
            mismatch_allowed: Whether to allow missing gather functions.

        Raises:
            ValueError: If GCS client is not initialized.
            KeyError: If gather function is missing and mismatch not allowed.
        """
        if not self.gcs_client:
            raise ValueError("GCS client not initialized.")

        bucket_name, blob_name = self._parse_gcs_path(gcs_path)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        buffer = io.BytesIO()

        tree = serialization.to_state_dict(tree)
        packer = msgpack.Packer()
        flatten_state = flatten_dict(tree)
        if gather_fns:
            gather_fns = flatten_dict(gather_fns)

        pbar = tqdm(flatten_state.items(), disable=not verbose, desc="Saving State to GCS")

        gather_mismatch_count = 0
        for key, value in pbar:
            if gather_fns:
                try:
                    callable_func = gather_fns.get(key)
                    if callable_func is None:
                        if not mismatch_allowed:
                            raise KeyError(f"Gather Function {key} is None")
                        gather_mismatch_count += 1
                    else:
                        value = callable_func(value)
                except KeyError as k_err:
                    if not mismatch_allowed:
                        raise KeyError(k_err) from None
                    gather_mismatch_count += 1

            pbar.set_postfix(gather_mismatch=gather_mismatch_count)
            value = put_dtype(value, float_dtype)
            buffer.write(packer.pack((key, serialization.to_bytes(value))))

        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

    def load_state_from_gcs_msgpack(
        self,
        gcs_path: str,
        verbose: bool = False,
    ) -> dict:
        """Load tree from GCS msgpack format.

        Args:
            gcs_path: GCS path to the msgpack checkpoint.
            verbose: Enable verbose output.

        Returns:
            Dictionary containing the loaded checkpoint.

        Raises:
            ValueError: If GCS client is not initialized.
        """
        if not self.gcs_client:
            raise ValueError("GCS client not initialized.")

        bucket_name, blob_name = self._parse_gcs_path(gcs_path)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        unpacker = msgpack.Unpacker(buffer, raw=False)
        tree = {}

        for key, value_bytes in tqdm(unpacker, desc="Loading from GCS", disable=not verbose):
            tree[key] = serialization.from_bytes(None, value_bytes)
        return unflatten_dict(tree)
