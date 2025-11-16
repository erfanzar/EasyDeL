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

"""Compilation utilities for JAX function optimization.

Provides enhanced JIT compilation with persistent caching to disk,
reducing compilation overhead across script runs.

Functions:
    ejit: Enhanced JIT with persistent caching
    save_compiled_fn: Save compiled function to disk
    load_compiled_fn: Load compiled function from disk
    load_cached_functions: Load multiple cached functions
    smart_compile: Smart compilation with auto-caching
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
import os
import pickle
import typing as tp
import warnings
from functools import wraps

import jax
import numpy as np
from ejkernel.callib import ejit
from jax._src.interpreters import pxla
from jax.experimental.serialize_executable import deserialize_and_load, serialize

from .helpers import check_bool_flag, get_cache_dir

if tp.TYPE_CHECKING:
    from jax._src.stages import Compiled, Lowered

ejit = ejit
P = tp.ParamSpec("P")
R = tp.TypeVar("R")

RECOMPILE_FORCE = check_bool_flag("EASYDEL_RECOMPILE_FORCE", False)
ECACHE_COMPILES = check_bool_flag("EASYDEL_CACHE_COMPILES", False)

CACHE_DIR = get_cache_dir()

COMPILE_FUNC_DIR = os.getenv("COMPILE_FUNC_DIR", CACHE_DIR / "ejit_compiled_functions")
if not isinstance(COMPILE_FUNC_DIR, str):
    COMPILE_FUNC_DIR.mkdir(parents=True, exist_ok=True)

COMPILED_FILE_NAME = "compiled.executable"
SIGNATURE_FILE_NAME = "compiled.signature"
COMPILED_CACHE: dict[str, Compiled] = {}


def _get_hardware_signature() -> str:
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


def _get_args_signature(args: tuple, kwargs: dict) -> str:
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


def load_cached_functions(verbose: bool = True) -> None:
    """Pre-loads all valid cached functions from disk into the persistent L2 cache."""
    if not COMPILE_FUNC_DIR.exists():
        return

    loaded_count = 0
    for cache_key_dir in COMPILE_FUNC_DIR.iterdir():
        if not cache_key_dir.is_dir():
            continue

        cache_key = cache_key_dir.name
        filepath = cache_key_dir / COMPILED_FILE_NAME

        if filepath.exists():
            try:
                with open(filepath, "rb") as f:
                    serialized, in_tree, out_tree = pickle.load(f)
                compiled_func = deserialize_and_load(serialized, in_tree, out_tree)
                COMPILED_CACHE[cache_key] = compiled_func
                loaded_count += 1
            except Exception as e:
                if verbose:
                    warnings.warn(f"Could not pre-load ejit cache for key {cache_key}. Error: {e}", stacklevel=2)

    if verbose and loaded_count > 0:
        print(f"Pre-loaded {loaded_count} functions into ejit's persistent memory cache.")


def save_compiled_fn(path: str | os.PathLike, fn: Compiled, prefix: str | None = None):
    """Save a compiled JAX function to disk for later reuse.

    Serializes a compiled function along with its input/output tree structures,
    allowing it to be loaded and executed in future Python sessions.

    Args:
        path: Directory path where the compiled function will be saved.
              Will be created if it doesn't exist.
        fn: Compiled JAX function (output of lowered.compile()).
        prefix: Optional prefix for the filename. Useful for organizing
               multiple compiled functions in the same directory.

    Files Created:
        - {prefix}-compiled.executable: Serialized function and metadata

    Example:
        >>> # Compile a function
        >>> jitted = jax.jit(my_function)
        >>> lowered = jitted.lower(sample_input)
        >>> compiled = lowered.compile()
        >>>
        >>> # Save to disk
        >>> from pathlib import Path
        >>> cache_dir = Path("./my_cache")
        >>> save_compiled_fn(cache_dir, compiled, prefix="model_v1")
        >>>
        >>> # File created: ./my_cache/model_v1-compiled.executable

    Raises:
        Warning: If serialization fails (logged, not raised).

    Notes:
        - Compiled functions are hardware-specific
        - Large models may produce large cache files
        - Uses pickle for serialization (standard security caveats apply)
    """
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prefix = prefix or ""
    filename = path / (prefix + "-" + COMPILED_FILE_NAME if prefix else COMPILED_FILE_NAME)
    serialized, in_tree, out_tree = serialize(fn)
    try:
        with open(filename, "wb") as f:
            pickle.dump((serialized, in_tree, out_tree), f)
    except Exception as e:
        warnings.warn(f"Could not save compiled function to {filename}: {e}", stacklevel=2)


def load_compiled_fn(path: str | os.PathLike, prefix: str | None = None):
    """Load a compiled function from disk."""
    prefix = prefix or ""
    filename = path / (prefix + "-" + COMPILED_FILE_NAME)
    (serialized, in_tree, out_tree) = pickle.load(open(filename, "rb"))
    return deserialize_and_load(
        serialized=serialized,
        in_tree=in_tree,
        out_tree=out_tree,
    )


def hash_fn(self) -> int:
    """Generate a hash for an object based on its dictionary values."""
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list))
    return get_safe_hash_int(shu)


def get_safe_hash_int(text, algorithm="md5"):
    """Generate a hash of text using specified algorithm with safety checks."""
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


def get_hash_of_lowering(lowered_func: Lowered):
    text_representation = lowered_func.as_text()
    hash_object = hashlib.sha256(text_representation.encode("utf-8"))
    hash_digest = hash_object.hexdigest()
    return hash_digest


def smart_compile(
    lowered_func: Lowered,
    tag: str | None = None,
    verbose: bool = True,
    cache_key: tuple[str, tuple] | None = None,
) -> tuple[Compiled, tuple[str, tuple] | None]:
    """Compile a lowered JAX function with caching."""
    func_hash = get_hash_of_lowering(lowered_func)
    foldername = str(func_hash) if tag is None else f"{tag}-{func_hash}"
    func_dir = COMPILE_FUNC_DIR / foldername
    filepath = func_dir / COMPILED_FILE_NAME
    signature_filepath = func_dir / SIGNATURE_FILE_NAME
    post_fix = f" (TAG : {tag})" if tag else ""
    signature = cache_key

    if filepath.exists() and not RECOMPILE_FORCE:
        try:
            (serialized, in_tree, out_tree) = pickle.load(open(filepath, "rb"))
            signature = pickle.load(open(signature_filepath, "rb"))
            compiled_func = deserialize_and_load(
                serialized=serialized,
                in_tree=in_tree,
                out_tree=out_tree,
            )
            return compiled_func, signature
        except Exception as e:
            if verbose:
                warnings.warn(
                    f"couldn't load compiled function due to {e}" + post_fix,
                    stacklevel=4,
                )
            compiled_func: Compiled = lowered_func.compile()
            if ECACHE_COMPILES:
                serialized, in_tree, out_tree = serialize(compiled_func)
                func_dir.mkdir(parents=True, exist_ok=True)
                try:
                    pickle.dump((serialized, in_tree, out_tree), open(filepath, "wb"))
                    pickle.dump(cache_key, open(signature_filepath, "wb"))
                except Exception as e:
                    if verbose:
                        warnings.warn(
                            f"couldn't save compiled function due to {e}" + post_fix,
                            stacklevel=4,
                        )
            return compiled_func, signature
    else:
        compiled_func: Compiled = lowered_func.compile()
        if ECACHE_COMPILES:
            try:
                serialized, in_tree, out_tree = serialize(compiled_func)
                func_dir.mkdir(parents=True, exist_ok=True)
                pickle.dump((serialized, in_tree, out_tree), open(filepath, "wb"))
                pickle.dump(cache_key, open(signature_filepath, "wb"))
            except Exception as e:
                if verbose:
                    warnings.warn(
                        f"couldn't save and serialize compiled function due to {e}" + post_fix,
                        stacklevel=4,
                    )
        return compiled_func, signature


class NoCompileContext:
    """Context manager that fails if JAX triggers a new compilation.

    Useful around hot paths that are expected to hit cached executables only.
    """

    def __init__(self, message: str = "JAX attempted to compile a new executable inside ForbidCompile."):
        """Initialize the guard with a custom failure message."""
        self.message = message
        self._original_func = None

    def __enter__(self):
        """Patch JAX's cached lowering to detect compilation cache misses."""
        # Store the original function
        self._original_func = pxla._cached_lowering_to_hlo
        original_cached_func = self._original_func

        @wraps(original_cached_func)
        def wrapper(*args, **kwargs):
            info_before = original_cached_func.cache_info()
            misses_before = info_before.misses

            result = original_cached_func(*args, **kwargs)
            info_after = original_cached_func.cache_info()
            misses_after = info_after.misses

            if misses_after > misses_before:
                raise RuntimeError(self.message)

            return result

        pxla._cached_lowering_to_hlo = wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the cached lowering function."""
        if self._original_func:
            pxla._cached_lowering_to_hlo = self._original_func
        return False
