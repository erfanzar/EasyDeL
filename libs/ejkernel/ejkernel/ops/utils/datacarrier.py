# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Data carrier dataclasses for kernel tiling and execution configuration.

This module provides :class:`FwdParams` and :class:`BwdParams` — lightweight
dataclasses that carry the block-size and GPU-execution parameters commonly
needed by forward and backward kernel variants (primarily attention and
matrix-multiplication operations).

All fields default to ``None``, which signals to the kernel that it should
select an appropriate value automatically (via heuristics or autotuning).

Custom hashing:
    Both dataclasses override ``__hash__`` with :func:`hash_fn`, which builds
    a hash from the concatenated string representations of all ``float``,
    ``int``, ``bool``, ``dict``, and ``list`` attributes.  This makes the
    objects usable as dictionary keys and in :class:`~ejkernel.ops.config.ConfigCache`
    lookups without requiring them to be ``frozen=True``.

Classes:
    FwdParams: Block-size and GPU-execution parameters for forward kernels.
    BwdParams: Block-size and GPU-execution parameters for backward kernels.
"""

import hashlib
from dataclasses import dataclass


def get_safe_hash_int(text, algorithm="md5"):
    """Generate an integer hash of text using the specified algorithm.

    Converts any input to a string and computes a hash using the specified
    algorithm from the hashlib module. The hash digest is then converted
    to a big-endian integer.

    Args:
        text: Input to hash (will be converted to string)
        algorithm: Hash algorithm name from hashlib (default: "md5")

    Returns:
        Integer representation of the hash digest

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib
        Exception: If any other error occurs during hash generation

    Example:
        >>> get_safe_hash_int("test_string")
        123456789012345678901234567890
        >>> get_safe_hash_int("test", algorithm="sha256")
        987654321098765432109876543210
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


def hash_fn(self) -> int:
    """Compute an integer hash from the numeric/collection attributes of an object.

    Intended to be assigned as the ``__hash__`` method of a dataclass (e.g.
    ``__hash__ = hash_fn``), providing hashability without requiring the
    dataclass to be ``frozen=True``.

    The hash is derived from the concatenated ``str()`` representations of all
    attribute values whose types are ``float``, ``int``, ``bool``, ``dict``, or
    ``list``.  Attribute values of other types (``None``, ``str``, arbitrary
    objects) are excluded from the hash computation.

    Args:
        self: Dataclass instance whose ``__dict__`` contains the configuration
            attributes to hash.

    Returns:
        An integer hash value.  Two instances with the same numeric/collection
        attribute values will produce the same hash (though not necessarily the
        same string representation for ``str``-typed attributes).

    Note:
        ``None`` values are excluded from the hash, so a parameter left at its
        default of ``None`` does not contribute to the hash.  This means two
        instances that differ only in ``None`` vs a set value will have
        different hashes only when the set value is of a hashable primitive type.
    """
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list))
    return get_safe_hash_int(shu)


@dataclass
class FwdParams:
    """Forward pass parameters for kernel configuration.

    Encapsulates block sizes and execution parameters for forward pass kernels,
    particularly for attention and matrix multiplication operations.

    Attributes:
        blocksize_m: Block size for M dimension (rows of output matrix)
        blocksize_k: Block size for K dimension (reduction dimension)
        blocksize_n: Block size for N dimension (columns of output matrix)
        q_blocksize: Block size for query dimension in attention
        kv_blocksize: Block size for key/value dimension in attention
        blocksize_heads: Block size for head dimension in multi-head attention
        blocksize_keys: Block size for key sequence length
        num_key_splits: Number of splits for key computation
        num_warps: Number of GPU warps for thread block execution
        num_stages: Number of pipeline stages for memory optimization

    Note:
        All parameters are optional (None) to allow automatic selection
        during kernel execution or autotuning.
    """

    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None

    blocksize_heads: int | None = None
    blocksize_keys: int | None = None
    num_key_splits: int | None = None

    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


@dataclass
class BwdParams:
    """Backward pass parameters for kernel configuration.

    Encapsulates block sizes and execution parameters for backward pass kernels,
    used in gradient computation for attention and matrix multiplication operations.

    Attributes:
        blocksize_m: Block size for M dimension (rows of output matrix)
        blocksize_k: Block size for K dimension (reduction dimension)
        blocksize_n: Block size for N dimension (columns of output matrix)
        q_blocksize: Block size for query dimension in attention gradients
        kv_blocksize: Block size for key/value dimension in attention gradients
        num_warps: Number of GPU warps for thread block execution
        num_stages: Number of pipeline stages for memory optimization

    Note:
        Parameters are typically smaller than forward pass due to different
        memory access patterns in gradient computation.
    """

    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn
