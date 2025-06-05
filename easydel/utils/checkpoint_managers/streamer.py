# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
import os
import tempfile
import typing as tp
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
        """
        if (isinstance(path, str) and path.startswith("gs://")) or isinstance(path, EasyPathLike):
            if not gcs_client:
                gcs_client = storage.Client()
            if isinstance(path, EasyPathLike):
                blob = path.blob
            else:
                bucket_name, blob_name = CheckpointManager._parse_gcs_path_static(path)
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
                path=path,
                shard_fns=shard_fns,
                verbose=verbose,
                mismatch_allowed=mismatch_allowed,
                callback=callback,
                dtype=dtype,
            )

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
                    disable=not verbose,
                )
            ]

        state = {key: tensor for key, tensor, _ in results}
        mismatch_count = sum(mismatch for _, _, mismatch in results)

        if verbose and mismatch_count:
            logger.info(f"Sharding mismatch: {mismatch_count}")

        state = unflatten_dict(state, sep=".")
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
    ) -> EasyPathLike | str | os.PathLike:
        """
        Save a checkpoint to local path or GCS using SafeTensors.
        """
        if enable is None:
            enable = jax.process_index() == 0
        if not enable:
            return "/dev/null"

        if float_dtype is None:
            float_dtype = jnp.bfloat16

        state = to_state_dict(state)
        gather_mismatch_count = 0
        if not is_flatten(state):
            state = flatten_dict(state, sep=".")
        if gather_fns:
            pbar_gather = tqdm(
                list(state.keys()),
                desc="Gathering State",
                disable=not verbose,
            )
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
            _gather,
            state,
            is_leaf=lambda x: isinstance(x, jax.Array | numpy.generic | float | int),
        )

        if cls._is_gcs_path(path):
            return cls._save_to_gcs(state, path, metadata, verbose)
        else:
            safe_flax.save_file(tensors=state, filename=str(path), metadata=metadata)
            return path

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

        for key, value_bytes in tqdm(unpacker, desc="Loading from GCS", disable=not verbose):
            state[key] = from_bytes(None, value_bytes)
        return unflatten_dict(state)
