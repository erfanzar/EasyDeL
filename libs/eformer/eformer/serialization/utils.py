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


import re
import typing as tp

import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import numpy
import psutil
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm.autonotebook import tqdm

from eformer.escale import create_cpu_mesh, with_sharding_constraint
from eformer.loggings import get_logger
from eformer.mpric import STRING_TO_DTYPE_MAP, put_dtype
from eformer.pytree import flatten_dict, is_flatten

logger = get_logger(__name__)


def reshard(x, target_sharding):
    """Reshard an array to a new sharding specification.

    Uses JIT compilation with sharding constraints to efficiently move an array
    to a new sharding layout across devices.

    Args:
        x: Input JAX array to reshard.
        target_sharding: Target sharding specification (e.g., NamedSharding).

    Returns:
        The input array resharded according to target_sharding.

    Example:
        >>> from jax.sharding import NamedSharding, PartitionSpec
        >>> new_sharding = NamedSharding(mesh, PartitionSpec("data"))
        >>> resharded = reshard(my_array, new_sharding)
    """

    @jax.jit
    def _move(y):
        return with_sharding_constraint(y, target_sharding)

    return _move(x)


def to_host(x: jax.Array, float_dtype: jnp.floating | None, mesh: Mesh, cpu_offload: bool):
    """Move array to host with optional dtype conversion and CPU offloading.

    Reshards the array to a fully replicated sharding on the provided mesh,
    optionally offloads to CPU, and converts floating point arrays to the
    specified dtype.

    Args:
        x: JAX array to move to host.
        float_dtype: Target dtype for floating point arrays. Can be a string
            (e.g., "float32") or jnp dtype. If None, dtype is unchanged.
        mesh: JAX mesh to use for resharding to replicated layout.
        cpu_offload: If True, additionally reshard to CPU mesh to free
            accelerator memory.

    Returns:
        Array moved to host, potentially with converted dtype.

    Note:
        CPU offloading is useful for preventing OOM during checkpointing by
        moving data off accelerators before serialization.
    """
    if isinstance(x, jax.Array):
        x = reshard(x, NamedSharding(mesh, PartitionSpec()))
        if cpu_offload:
            x = reshard(x, NamedSharding(create_cpu_mesh(), PartitionSpec()))
    if float_dtype:
        dtype = STRING_TO_DTYPE_MAP.get(float_dtype, float_dtype) if isinstance(float_dtype, str) else float_dtype
        if jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(dtype)
    return x


def estimate_array_nbytes(array: jax.Array) -> int:
    """
    Estimate number of bytes for a JAX array without device_get.

    Args:
        array: JAX array to estimate size for

    Returns:
        Estimated size in bytes
    """
    try:
        itemsize = numpy.dtype(array.dtype).itemsize
        return int(array.size) * int(itemsize)
    except Exception:
        if getattr(array, "dtype", None) in (jnp.int4, jnp.uint4):
            return (int(array.size) + 1) // 2
        v = jnp.asarray(array)
        return int(v.size) * int(numpy.dtype(v.dtype).itemsize)


def estimate_available_memory() -> int:
    """
    Dynamically estimate available memory for safe loading.

    Returns:
        Available memory in bytes
    """
    mem = psutil.virtual_memory()
    available = int(mem.available * 0.5)
    try:
        devices = jax.local_devices()
        if devices:
            device_mem = devices[0].memory_stats()
            if device_mem:
                device_available = int(device_mem.get("bytes_limit", 0) * 0.4)
                available = min(available, device_available)
    except Exception:
        pass

    return max(available, 100 * 1024 * 1024)


def derive_base_prefix_from_path(path_str: str) -> str:
    """
    Normalize a path into its 'base prefix' used for sharded file naming.

    Examples:
        /x/model.safetensors -> /x/model
        /x/model.safetensors.index.json -> /x/model
        /x/model-00001-of-00004.safetensors -> /x/model

    Args:
        path_str: Input path string

    Returns:
        Base prefix for the path
    """

    if path_str.endswith(".safetensors.index.json"):
        return path_str[: -len(".safetensors.index.json")]

    if path_str.endswith(".safetensors"):
        prefix = path_str[: -len(".safetensors")]
    else:
        prefix = path_str

    m = re.match(r"^(.*)-\d{5}-of-\d{5}$", prefix)
    if m:
        return m.group(1)
    return prefix


def shard_filename(base_prefix: str, idx: int, total: int) -> str:
    """Generate a standardized shard filename for sharded checkpoints.

    Creates filenames following the pattern: <prefix>-XXXXX-of-YYYYY.safetensors

    Args:
        base_prefix: Base path/prefix for the shard files.
        idx: Shard index (1-indexed).
        total: Total number of shards.

    Returns:
        Full shard filename with zero-padded indices.

    Example:
        >>> shard_filename("/checkpoints/model", 1, 4)
        '/checkpoints/model-00001-of-00004.safetensors'
    """
    return f"{base_prefix}-{idx:05d}-of-{total:05d}.safetensors"


def index_filename(base_prefix: str) -> str:
    """Generate the index filename for a sharded checkpoint.

    Creates the filename for the JSON index file that maps tensor names
    to their shard files.

    Args:
        base_prefix: Base path/prefix for the checkpoint.

    Returns:
        Full path to the index JSON file.

    Example:
        >>> index_filename("/checkpoints/model")
        '/checkpoints/model.safetensors.index.json'
    """
    return f"{base_prefix}.safetensors.index.json"


def is_gcs_path(path: str) -> bool:
    """Check if a path points to Google Cloud Storage.

    Determines whether a path is a GCS path by checking for the gs:// prefix
    or by checking if it's a GCSPath instance.

    Args:
        path: Path string or path object to check.

    Returns:
        True if the path is a GCS path, False otherwise.

    Example:
        >>> is_gcs_path("gs://my-bucket/checkpoint")
        True
        >>> is_gcs_path("/local/path/checkpoint")
        False
    """
    from ..paths import GCSPath, LocalPath

    if isinstance(path, GCSPath):
        return True
    elif isinstance(path, LocalPath):
        return False
    return isinstance(path, str) and path.startswith("gs://")


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """
    Parse gs://bucket/path into bucket and blob name.

    Args:
        gcs_path: GCS path string

    Returns:
        Tuple of (bucket_name, blob_name)
    """
    gcs_path = str(gcs_path)
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""
    return bucket_name, blob_name


def group_keys_by_shard_size(
    flat_state: dict[str, jax.Array],
    max_shard_size_bytes: int,
) -> list[list[str]]:
    """
    Group keys into shards under max_shard_size_bytes each.

    Args:
        flat_state: Flattened state dictionary
        max_shard_size_bytes: Maximum size per shard in bytes

    Returns:
        List of key groups (shards)
    """
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for k, v in flat_state.items():
        nbytes = estimate_array_nbytes(v)
        if current and current_bytes + nbytes > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(k)
        current_bytes += nbytes

    if current:
        shards.append(current)
    return shards


def optimize_shard_layout(state: dict[str, jax.Array], max_shard_size_bytes: int) -> list[list[str]]:
    """
    Optimize shard layout for better loading performance.
    Groups related tensors and considers access patterns.

    Args:
        state: State dictionary with arrays
        max_shard_size_bytes: Maximum size per shard

    Returns:
        Optimized list of key groups
    """
    prefix_groups = {}
    for key in state.keys():
        prefix = key.rsplit(".", 1)[0] if "." in key else "root"
        if prefix not in prefix_groups:
            prefix_groups[prefix] = []
        prefix_groups[prefix].append(key)

    shards = []
    current_shard = []
    current_size = 0

    for _, keys in sorted(prefix_groups.items()):
        group_size = sum(estimate_array_nbytes(state[k]) for k in keys)

        if group_size > max_shard_size_bytes:
            if current_shard:
                shards.append(current_shard)
                current_shard = []
                current_size = 0

            for key in keys:
                key_size = estimate_array_nbytes(state[key])
                if current_size + key_size > max_shard_size_bytes:
                    if current_shard:
                        shards.append(current_shard)
                    current_shard = [key]
                    current_size = key_size
                else:
                    current_shard.append(key)
                    current_size += key_size
        else:
            if current_size + group_size > max_shard_size_bytes:
                shards.append(current_shard)
                current_shard = keys
                current_size = group_size
            else:
                current_shard.extend(keys)
                current_size += group_size

    if current_shard:
        shards.append(current_shard)

    return shards


def read_process_array(
    key: str,
    shard_fns: dict | None,
    mismatch_allowed: bool,
    manager,
    callback: tp.Callable[[jax.Array, str], jax.Array] | None = None,
    dtype: str | jnp.dtype | None = None,
) -> tuple[str, jax.Array, int]:
    """
    Helper function to process a single tensor from a checkpoint.

    Args:
        key: Tensor key
        shard_fns: Shard functions dictionary
        mismatch_allowed: Whether to allow shard function mismatches
        manager: Checkpoint manager instance
        callback: Optional callback for tensor processing
        dtype: Target dtype for conversion

    Returns:
        Tuple of (key, processed_tensor, mismatch_count)
    """
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


def apply_gather_functions(
    state: dict,
    gather_fns: dict | bool,
    mismatch_allowed: bool,
    verbose: bool,
) -> dict:
    """
    Apply gather functions to state.

    Args:
        state: State dictionary
        gather_fns: Gather functions or boolean flag
        mismatch_allowed: Whether to allow mismatches
        verbose: Enable verbose output

    Returns:
        Processed state dictionary
    """
    if isinstance(gather_fns, bool) and gather_fns:
        return {k: jax.device_get(v) for k, v in state.items()}

    if not is_flatten(gather_fns):
        gather_fns = flatten_dict(gather_fns, sep=".")

    processed = {}
    mismatch_count = 0

    pbar = tqdm(state.items(), desc="Gathering state", disable=not verbose)
    for key, value in pbar:
        func = gather_fns.get(key)
        if func:
            processed[key] = func(value)
        elif not mismatch_allowed:
            raise KeyError(f"Gather function for {key} not found")
        else:
            processed[key] = value
            mismatch_count += 1

        if verbose:
            pbar.set_postfix({"mismatches": mismatch_count})

    return processed


def flatten_for_broadcast(state: dict) -> dict:
    """
    Flatten state for efficient broadcasting.

    Args:
        state: State dictionary

    Returns:
        Flattened state dictionary
    """
    if is_flatten(state):
        return state
    return flatten_dict(state, sep=".")


def chunk_tensor_by_memory(
    tensor: jax.Array,
    memory_limit: int,
) -> list[jax.Array]:
    """
    Split tensor into chunks that fit within memory limit.

    Args:
        tensor: Input tensor
        memory_limit: Memory limit in bytes

    Returns:
        List of tensor chunks
    """
    tensor_bytes = estimate_array_nbytes(tensor)
    if tensor_bytes <= memory_limit:
        return [tensor]
    n_chunks = (tensor_bytes + memory_limit - 1) // memory_limit
    if tensor.ndim > 0 and tensor.shape[0] >= n_chunks:
        chunk_size = tensor.shape[0] // n_chunks
        chunks = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else tensor.shape[0]
            chunks.append(tensor[start:end])
        return chunks
    return [tensor]


def broadcast_tensor(
    tensor: jax.Array,
    memory_limit_bytes: int,
    target_sharding: NamedSharding | None = None,
) -> jax.Array:
    """
    Efficiently broadcast tensor from single replica to all devices.

    Args:
        tensor: Input tensor
        memory_limit_bytes: Memory limit for chunking
        target_sharding: Optional target sharding

    Returns:
        Broadcasted tensor
    """
    size_bytes = estimate_array_nbytes(tensor)
    n_chunks = max(1, (size_bytes + memory_limit_bytes - 1) // memory_limit_bytes)

    slices = []
    if tensor.ndim == 0 or n_chunks == 1:
        slices = [slice(None)]
    else:
        total = tensor.shape[0]
        base = total // n_chunks
        rem = total % n_chunks
        start = 0
        for i in range(n_chunks):
            extra = 1 if i < rem else 0
            end = start + base + extra
            slices.append(slice(start, end))
            start = end

    pieces = []
    for slc in slices:
        if jax.process_index() == 0:
            piece = tensor[slc] if slc != slice(None) else tensor
        else:
            if slc == slice(None):
                piece = jnp.zeros_like(tensor)
            else:
                length = slc.stop - slc.start
                piece = jnp.zeros((length, *tensor.shape[1:]), dtype=tensor.dtype)
        b = jax.experimental.multihost_utils.broadcast_one_to_all(piece)
        pieces.append(b)

    result = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
    if target_sharding is not None:
        result = jax.device_put(result, target_sharding)
    return result
