# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
import re
import tempfile
import typing as tp
from collections import defaultdict
from functools import partial

import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import msgpack
import numpy
from eformer.jaximus import implicit
from flax.serialization import from_bytes, to_bytes, to_state_dict
from flax.struct import PyTreeNode
from google.cloud import storage
from safetensors import flax as safe_flax
from tqdm.autonotebook import tqdm

from easydel.utils.helpers import get_logger

from ..traversals import flatten_dict, is_flatten, unflatten_dict
from .path_utils import EasyPathLike, GCSPath, LocalPath

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

STRING_TO_DTYPE_MAP = {
    "bf16": jnp.bfloat16,
    "bfloat16": jnp.bfloat16,
    "fp16": jnp.float16,
    "float16": jnp.float16,
    "fp32": jnp.float32,
    "float32": jnp.float32,
    "fp64": jnp.float64,
    "float64": jnp.float64,
    "fp8": jnp.float8_e5m2,
    "fp8_e4m3fn": jnp.float8_e4m3fn,
    "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "fp8_e5m2": jnp.float8_e5m2,
    "fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "float8_e5m2": jnp.float8_e5m2,
    "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
}
DTYPE_TO_STRING_MAP = {
    jnp.bfloat16: "bf16",
    jnp.float16: "fp16",
    jnp.float32: "fp32",
    jnp.float64: "fp64",
    jnp.float8_e5m2: "fp8",
    jnp.float8_e4m3fn: "fp8_e4m3fn",
    jnp.float8_e4m3fnuz: "fp8_e4m3fnuz",
    jnp.float8_e4m3b11fnuz: "fp8_e4m3b11fnuz",
    jnp.float8_e5m2: "fp8_e5m2",
    jnp.float8_e5m2fnuz: "fp8_e5m2fnuz",
}


@implicit
def put_dtype(
    array: jax.Array,
    dtype: str | jnp.dtype | None,
) -> jax.Array:
    """
    Get the tensor with the specified data type.

    Args:
            array: The input tensor.
            dtype: The desired data type.

    Returns:
            The tensor with the specified data type.
    """
    if not dtype:
        return array

    if isinstance(dtype, str):
        try:
            dtype = STRING_TO_DTYPE_MAP[dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype string: {dtype}") from e

    if array.dtype in (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64):
        return array.astype(dtype)
    return array  # Return original array if it's not a float


def _read_process_array(
    key,
    shard_fns,
    mismatch_allowed,
    manager,
    callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
    dtype: str | jnp.dtype | None = None,
):
    """Helper function to process a single tensor from a checkpoint."""
    tensor = manager.get_tensor(key)
    mismatch = 0
    if shard_fns:
        try:
            callable_func = shard_fns.get(key)
            if callable_func is None:
                if not mismatch_allowed:
                    raise KeyError(f"Shard Function {key} is None and NoneType OBJ is not callable.")
                mismatch = 1
            else:
                tensor = callable_func(tensor)
        except KeyError as k_err:
            if not mismatch_allowed:
                raise KeyError(k_err) from None
            mismatch = 1

    if callback:
        tensor = callback(tensor, key)
    tensor = put_dtype(tensor, dtype)
    return key, tensor, mismatch


class CheckpointManager:
    """
    A class to manage saving and loading checkpoints with Google Cloud Storage support.
    """

    def __init__(
        self,
        checkpoint_dir: EasyPathLike | str | os.PathLike,
        enable: bool | None = None,
        float_dtype: jnp.dtype = jnp.bfloat16,
        save_optimizer_state: bool = True,
        verbose: bool = False,
        gcs_bucket: str | None = None,
        gcs_credentials_path: str | None = None,
    ):
        self.float_dtype = float_dtype
        self.save_optimizer_state = save_optimizer_state
        self.checkpoint_dir = checkpoint_dir
        self.enable = enable
        self.verbose = verbose
        self.gcs_bucket = gcs_bucket

        self.gcs_client = None

        if gcs_bucket:
            self.gcs_client = self.create_gcs_client(gcs_credentials_path)

    @staticmethod
    def _estimate_nbytes(array: jax.Array) -> int:
        """Estimate number of bytes for a JAX array without device_get."""
        try:
            # Works for most jax dtypes
            itemsize = numpy.dtype(array.dtype).itemsize
            return int(array.size) * int(itemsize)
        except Exception:
            # Fallback for exotic/4-bit dtypes
            if getattr(array, "dtype", None) in (jnp.int4, jnp.uint4):
                return (int(array.size) + 1) // 2
            v = jnp.asarray(array)
            return int(v.size) * int(numpy.dtype(v.dtype).itemsize)

    @staticmethod
    def _group_keys_by_shard_size(
        flat_state: dict[str, jax.Array],
        max_shard_size_bytes: int,
    ) -> list[list[str]]:
        """Group keys into shards under max_shard_size_bytes each."""
        shards: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0

        for k, v in flat_state.items():
            nbytes = CheckpointManager._estimate_nbytes(v)
            if current and current_bytes + nbytes > max_shard_size_bytes:
                shards.append(current)
                current = []
                current_bytes = 0
            current.append(k)
            current_bytes += nbytes

        if current:
            shards.append(current)
        return shards

    @staticmethod
    def _derive_base_prefix_from_path(path_str: str) -> str:
        """
        Normalize a path into its 'base prefix' used for sharded file naming.
        Examples:
          /x/model.safetensors -> /x/model
          /x/model.safetensors.index.json -> /x/model
          /x/model-00001-of-00004.safetensors -> /x/model
        """
        # Strip .safetensors.index.json
        if path_str.endswith(".safetensors.index.json"):
            return path_str[: -len(".safetensors.index.json")]
        # Strip .safetensors
        if path_str.endswith(".safetensors"):
            prefix = path_str[: -len(".safetensors")]
        else:
            prefix = path_str

        # If it's a shard name like ...-00001-of-00004, strip that part too
        m = re.match(r"^(.*)-\d{5}-of-\d{5}$", prefix)
        if m:
            return m.group(1)
        return prefix

    @staticmethod
    def _shard_filename(base_prefix: str, idx: int, total: int) -> str:
        return f"{base_prefix}-{idx:05d}-of-{total:05d}.safetensors"

    @staticmethod
    def _index_filename(base_prefix: str) -> str:
        return f"{base_prefix}.safetensors.index.json"

    @staticmethod
    def create_gcs_client(gcs_credentials_path: str | None = None):
        if gcs_credentials_path:
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(gcs_credentials_path)
            gcs_client = storage.Client(credentials=credentials)
        else:
            gcs_client = storage.Client()
        return gcs_client

    @staticmethod
    def _is_gcs_path(path: str) -> bool:
        """Check if path is a GCS path (starts with gs://)"""
        if isinstance(path, GCSPath):
            return True
        elif isinstance(path, LocalPath):
            return False
        return isinstance(path, str) and path.startswith("gs://")

    @staticmethod
    def _parse_gcs_path(gcs_path: str | EasyPathLike) -> tuple[str, str]:
        """Parse gs://bucket/path into bucket and blob name"""
        gcs_path = str(gcs_path)
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        path_parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""
        return bucket_name, blob_name

    @staticmethod
    def load_checkpoint(
        path: EasyPathLike | str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        gcs_client: storage.Client | None = None,
    ) -> tuple[PyTreeNode | dict, dict]:
        """
        Load a checkpoint from local path or GCS.
        Supports:
          - Single safetensors file
          - Sharded safetensors with index (prefix.safetensors.index.json)
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
            if isinstance(path, EasyPathLike):
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
    ) -> tuple[PyTreeNode | dict, dict]:
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
        state: dict[str, jax.Array] = {}
        mismatch_count = 0

        pbar = tqdm(total=total_keys, desc="Loading shards", disable=not (verbose and jax.process_index() == 0))
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
                    state[k] = tensor
                    mismatch_count += mm
                    pbar.update(1)
        pbar.close()

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        state = unflatten_dict(state, sep=".")
        metadata = index_data.get("metadata", {})
        return state, metadata

    @staticmethod
    def _load_sharded_from_gcs_index(
        index_gcs_path: str,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
        gcs_client: storage.Client | None = None,
    ) -> tuple[PyTreeNode | dict, dict]:
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
        state: dict[str, jax.Array] = {}
        mismatch_count = 0

        pbar = tqdm(total=total_keys, desc="Loading shards (GCS)", disable=not (verbose and jax.process_index() == 0))
        for shard_name, keys in file_to_keys.items():
            shard_blob_path = shard_name if not shard_dir else f"{shard_dir}/{shard_name}"
            shard_blob = bucket.blob(shard_blob_path)

            # Download shard to temp and read via safetensors
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
                        state[k] = tensor
                        mismatch_count += mm
                        pbar.update(1)
            finally:
                os.unlink(temp_path)
        pbar.close()

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        state = unflatten_dict(state, sep=".")
        metadata = index_data.get("metadata", {})
        return state, metadata

    @classmethod
    def save_checkpoint(
        cls,
        state: PyTreeNode,
        path: EasyPathLike | str | os.PathLike,
        gather_fns: dict[tp.Callable] | bool | None = None,
        float_dtype: str | jnp.dtype | None = None,
        verbose: bool = True,
        mismatch_allowed: bool = True,
        metadata: dict[str, str] | None = None,
        enable: bool | None = None,
        shard_size_gb: float | None = 5.00,
        write_index_file: bool = True,
    ) -> EasyPathLike | str | os.PathLike:
        """
        Save a checkpoint to local path or GCS using SafeTensors.

        If shard_size_gb is provided, the state is saved as multiple shards of up to that size
        (except the last shard, which may be smaller). An index file 'prefix.safetensors.index.json'
        is also written mapping every tensor name to a shard file.
        """
        if enable is None:
            enable = jax.process_index() == 0
        if not enable:
            path = "/dev/null"
        if str(path).startswith("/dev/null"):
            path = "/dev/null"

        if float_dtype is None:
            float_dtype = jnp.bfloat16

        state = to_state_dict(state)
        if not is_flatten(state):
            state = flatten_dict(state, sep=".")

        gather_mismatch_count = 0
        if gather_fns:
            pbar_gather = tqdm(list(state.keys()), desc="Gathering State", disable=not (verbose and jax.process_index() == 0))
            if isinstance(gather_fns, bool):
                for key in pbar_gather:
                    pbar_gather.update(1)
                    state[key] = jax.device_get(state[key])
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
                        state[key] = callable_func(state[key])

                    pbar_gather.set_postfix(gather_mismatch=gather_mismatch_count)
                    pbar_gather.update(1)

        def _gather(x):
            return put_dtype(jax.device_get(jnp.array(x)) if not isinstance(x, (jax.Array)) else x, float_dtype)

        state = jax.tree_util.tree_map(
            _gather, state, is_leaf=lambda x: isinstance(x, jax.Array | numpy.generic | float | int)
        )

        path_str = str(path)

        if shard_size_gb is not None and shard_size_gb > 0:
            max_bytes = int(shard_size_gb * (1024**3))
            flat_state = state if is_flatten(state) else flatten_dict(state, sep=".")
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

            # Create weight_map
            for i, shard_keys in enumerate(shards, start=1):
                shard_name = os.path.basename(cls._shard_filename(base_prefix, i, total_shards))
                for k in shard_keys:
                    weight_map[k] = shard_name

            if write_index_file:
                index_data = {"metadata": metadata or {}, "weight_map": weight_map}
                if cls._is_gcs_path(path_str):
                    gcs_client = cls.create_gcs_client()
                    _ = cls._parse_gcs_path(base_prefix + ".txt")
                    idx_bucket, idx_blob = cls._parse_gcs_path(cls._index_filename(base_prefix))
                    bucket = gcs_client.bucket(idx_bucket)
                    blob = bucket.blob(idx_blob)
                    blob.upload_from_string(
                        json.dumps(index_data, ensure_ascii=False).encode("utf-8"),
                        content_type="application/json",
                    )
                elif base_prefix != "/dev/null":
                    with open(index_path, "w", encoding="utf-8") as f:
                        json.dump(index_data, f, ensure_ascii=False)

            return cls._index_filename(base_prefix)

        if cls._is_gcs_path(path_str):
            return cls._save_to_gcs(state, path_str, metadata, verbose)
        else:
            safe_flax.save_file(tensors=state, filename=path_str, metadata=metadata)
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
        for i, shard_keys in enumerate(tqdm(shards, desc="Saving shards", disable=not (verbose and jax.process_index() == 0), start=1)):
            subset = {k: flat_state[k] for k in shard_keys}
            gathered = jax.experimental.multihost_utils.process_allgather(subset)
            if jax.process_index() == 0:
                shard_path = cls._shard_filename(base_prefix, i, total_shards)
                safe_flax.save_file(tensors=gathered, filename=shard_path, metadata=metadata)

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
        gcs_client = cls.create_gcs_client()
        bucket_name, base_blob_name = cls._parse_gcs_path(base_prefix + ".txt")  # hack to reuse parser
        base_dir = os.path.dirname(base_blob_name)
        bucket = gcs_client.bucket(bucket_name)

        for i, shard_keys in enumerate(tqdm(shards, desc="Saving shards to GCS", disable=not (verbose and jax.process_index() == 0)), start=1):
            subset = {k: flat_state[k] for k in shard_keys}
            gathered = jax.experimental.multihost_utils.process_allgather(subset)
            if jax.process_index() == 0:
                shard_name = os.path.basename(cls._shard_filename(base_prefix, i, total_shards))
                shard_blob_name = f"{base_dir}/{shard_name}" if base_dir else shard_name
                shard_blob = bucket.blob(shard_blob_name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
                    shard_path = temp_file.name
                try:
                    safe_flax.save_file(tensors=gathered, filename=shard_path, metadata=metadata)
                    shard_blob.upload_from_filename(shard_path)
                finally:
                    if os.path.exists(shard_path):
                        os.unlink(shard_path)

    @staticmethod
    def _parse_gcs_path_static(gcs_path: str) -> tuple[str, str]:
        """Static version of GCS path parser"""
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {gcs_path}")

        path_parts = gcs_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1] if len(path_parts) > 1 else ""
        return bucket_name, blob_name

    @staticmethod
    def _load_checkpoint_from_file(
        path: EasyPathLike | str | os.PathLike,
        shard_fns: dict[tp.Callable] | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
        callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
        dtype: str | jnp.dtype | None = None,
    ) -> tuple[PyTreeNode | dict, dict]:
        """Original load_checkpoint logic for local files"""
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
            results = [
                process_func(key)
                for key in tqdm(
                    keys,
                    desc="Loading",
                    total=len(keys),
                    disable=not (verbose and jax.process_index() == 0),
                )
            ]

        state = {key: tensor for key, tensor, _ in results}
        mismatch_count = sum(mismatch for _, _, mismatch in results)

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        state = unflatten_dict(state, sep=".")
        return state, metadata

    @classmethod
    def _save_to_gcs(
        cls,
        state: dict,
        gcs_path: str,
        metadata: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> str:
        """Save state to GCS using temporary file"""
        gcs_client = cls.create_gcs_client()

        bucket_name, blob_name = cls._parse_gcs_path(gcs_path)
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
            safe_flax.save_file(tensors=state, filename=temp_file.name, metadata=metadata)
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
        state: PyTreeNode,
        gcs_path: str,
        gather_fns: dict[tp.Callable] | None = None,
        float_dtype: str | jnp.dtype | None = None,
        verbose: bool = False,
        mismatch_allowed: bool = True,
    ):
        """Save state to GCS using msgpack format (streaming)"""
        if not self.gcs_client:
            raise ValueError("GCS client not initialized.")

        bucket_name, blob_name = self._parse_gcs_path(gcs_path)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        buffer = io.BytesIO()

        state = to_state_dict(state)
        packer = msgpack.Packer()
        flatten_state = flatten_dict(state)
        if gather_fns:
            gather_fns = flatten_dict(gather_fns)

        pbar = tqdm(flatten_state.items(), disable=not (verbose and jax.process_index() == 0), desc="Saving State to GCS")

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
            buffer.write(packer.pack((key, to_bytes(value))))

        # Upload buffer to GCS
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="application/octet-stream")

    def load_state_from_gcs_msgpack(
        self,
        gcs_path: str,
        verbose: bool = False,
    ) -> dict:
        """Load state from GCS msgpack format"""
        if not self.gcs_client:
            raise ValueError("GCS client not initialized.")

        bucket_name, blob_name = self._parse_gcs_path(gcs_path)
        bucket = self.gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download to buffer
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)

        # Unpack msgpack data
        unpacker = msgpack.Unpacker(buffer, raw=False)
        state = {}

        for key, value_bytes in tqdm(unpacker, desc="Loading from GCS", disable=not (verbose and jax.process_index() == 0)):
            state[key] = from_bytes(None, value_bytes)
        return unflatten_dict(state)
