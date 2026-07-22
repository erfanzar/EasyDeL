# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Compilation utilities for JAX function optimization.

Provides enhanced JIT compilation with persistent caching to disk,
reducing compilation overhead across script runs.

Functions:
    ejit: Enhanced JIT with persistent caching
    save_compiled_fn: Save compiled function to disk
    load_compiled_fn: Load compiled function from disk
    load_cached_functions: Load multiple cached functions
    hash_fn: Generate hash for function signature

Constants:
    RECOMPILE_FORCE: Force recompilation flag
    ECACHE_COMPILES: Enable compilation caching
    CACHE_DIR: Cache directory path
    COMPILE_FUNC_DIR: Compiled functions directory
    COMPILED_CACHE: In-memory cache of compiled functions

Key Features:
    - Persistent disk caching of compiled functions
    - Automatic cache invalidation on changes
    - Hardware-specific signatures
    - Two-level caching (memory + disk)
    - Graceful fallback on errors

Example:
    >>> from easydel.utils.compiling_utils import ejit
    >>>
    >>> @ejit
    ... def optimized_fn(x, y):
    ...     return x @ y + x.T @ y.T
    >>>
    >>> # First call compiles and caches
    >>> result = optimized_fn(a, b)
    >>> # Next run loads from cache
    >>> result = optimized_fn(a, b)
"""

from __future__ import annotations

import hashlib
import typing as tp

import jax
import numpy as np
from ejkernel.callib import (  # pyright: ignore[reportMissingTypeStubs]
    ejit,
    get_effective_compile_dir,
    load_cached_functions,
    load_compiled_fn,
    save_compiled_fn,
)

from .helpers import check_bool_flag, get_cache_dir

if tp.TYPE_CHECKING:
    from jax._src.stages import Compiled, Lowered

ejit = ejit

__all__ = [
    "ejit",
    "get_hash_of_lowering",
    "get_safe_hash_int",
    "hash_fn",
    "load_cached_functions",
    "load_compiled_fn",
    "save_compiled_fn",
]

P = tp.ParamSpec("P")
R = tp.TypeVar("R")

RECOMPILE_FORCE = check_bool_flag("EASYDEL_RECOMPILE_FORCE", False)
ECACHE_COMPILES = check_bool_flag("EASYDEL_CACHE_COMPILES", False)

CACHE_DIR = get_cache_dir()

# The single source of truth for compiled-function persistence is
# ejkernel.callib._ejit: an environment-fingerprinted cache directory
# (jax/jaxlib/libtpu versions + host CPU identity) with entry-size guards and
# atomic writes, so stale or oversized entries can never crash a reader.
# The save/load helpers above are re-exports of that hardened implementation.
COMPILE_FUNC_DIR = get_effective_compile_dir()

COMPILED_FILE_NAME = "compiled.executable"
COMPILED_CACHE: dict[str, Compiled] = {}


def _get_hardware_signature() -> str:  # pyright: ignore[reportUnusedFunction]
    """Create signature for current JAX hardware environment.

    Returns:
        String representation of available JAX devices.
    """
    return str(jax.devices())


def _get_leaf_signature(leaf: tp.Any) -> tp.Hashable:
    """Generate hashable signature for PyTree leaf.

    Args:
        leaf: Leaf node from PyTree.

    Returns:
        Hashable signature including shape, dtype, and sharding.
    """
    if isinstance(leaf, jax.Array | np.ndarray):
        if hasattr(leaf, "sharding"):
            return (leaf.shape, str(jax.dtypes.canonicalize_dtype(leaf.dtype)), repr(leaf.sharding))
        return (leaf.shape, str(jax.dtypes.canonicalize_dtype(leaf.dtype)))
    return type(leaf)


def _get_args_signature(args: tuple, kwargs: dict) -> str:  # pyright: ignore[reportUnusedFunction]
    """Create signature for function arguments.

    Generates a unique signature based on the PyTree structure,
    shapes, and dtypes of arguments.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        String signature of the arguments.
    """
    arg_leaves, arg_tree = jax.tree_util.tree_flatten((args, kwargs))
    leaf_signatures = tuple(map(_get_leaf_signature, arg_leaves))
    return str((arg_tree, leaf_signatures))





def hash_fn(self) -> int:
    """Generate a deterministic hash for an object based on its ``__dict__`` values.

    Concatenates all scalar and collection attribute values into a string
    and produces an integer hash via ``get_safe_hash_int``.

    Args:
        self: The object to hash.

    Returns:
        Integer hash derived from the object's attributes.
    """
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list))
    return get_safe_hash_int(shu)


def get_safe_hash_int(text, algorithm="md5"):
    """Generate an integer hash of text using the specified algorithm.

    Args:
        text: Input value (will be converted to string).
        algorithm: Name of the hashlib algorithm to use (default: ``"md5"``).

    Returns:
        Integer representation of the hash digest.

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib.
        RuntimeError: If hashing fails for any other reason.
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise RuntimeError(f"Error generating hash: {e!s}") from e


def get_hash_of_lowering(lowered_func: Lowered):
    """Compute a SHA-256 hash of a lowered JAX function's text representation.

    This provides a stable fingerprint for the lowered HLO, enabling
    cache invalidation when the computation graph changes.

    Args:
        lowered_func: A lowered JAX function (output of ``jitted.lower(...)``).

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    text_representation = lowered_func.as_text()
    hash_object = hashlib.sha256(text_representation.encode("utf-8"))
    hash_digest = hash_object.hexdigest()
    return hash_digest
