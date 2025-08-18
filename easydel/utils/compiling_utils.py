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

import functools
import hashlib
import inspect
import os
import pickle
import typing as tp
import warnings

import jax
import numpy as np
from jax.experimental.serialize_executable import deserialize_and_load, serialize

from .helpers import check_bool_flag, get_cache_dir

if tp.TYPE_CHECKING:
    from jax._src.stages import Compiled, Lowered

P = tp.ParamSpec("P")
R = tp.TypeVar("R")

RECOMPILE_FORCE = check_bool_flag("EASYDEL_RECOMPILE_FORCE", False)
ECACHE_COMPILES = check_bool_flag("EASYDEL_CACHE_COMPILES", False)

CACHE_DIR = get_cache_dir()
COMPILE_FUNC_DIR = CACHE_DIR / "ejit_compiled_functions"
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


def ejit(
    func: tp.Callable[P, R] | None = None,
    *,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    in_shardings: tp.Any = None,
    out_shardings: tp.Any = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    backend: str | None = None,
    inline: bool = False,
    abstracted_axes: tp.Any | None = None,
    compiler_options: dict[str, tp.Any] | None = None,
):
    """Enhanced JIT compilation with persistent caching.

    Drop-in replacement for jax.jit that caches compiled functions
    to disk for reuse across script runs, significantly reducing
    compilation overhead.

    Features:
        - Two-level caching (in-memory LRU + persistent disk cache)
        - Automatic cache invalidation on signature changes
        - Graceful fallback to standard jax.jit on errors
        - Support for all jax.jit parameters

    Args:
        func: Function to JIT-compile and cache.
        static_argnums: Indices of static arguments.
        static_argnames: Names of static arguments.
        donate_argnums: Indices of donated arguments.
        in_shardings: Input sharding specifications.
        out_shardings: Output sharding specifications.

    Returns:
        JIT-compiled function with caching.

    Example:
        >>> @ejit
        ... def fast_matmul(a, b):
        ...     return a @ b
        >>> # First call compiles and caches
        >>> result = fast_matmul(x, y)
        >>> # Subsequent calls use cached version
    """
    if func is None:
        return functools.partial(
            ejit,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            abstracted_axes=abstracted_axes,
            backend=backend,
            compiler_options=compiler_options,
            donate_argnames=donate_argnames,
            inline=inline,
            keep_unused=keep_unused,
        )

    if not ECACHE_COMPILES:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(str(COMPILE_FUNC_DIR))
        jax.config.update("jax_compilation_cache_dir", str(COMPILE_FUNC_DIR))
        jitted_function = jax.jit(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
            abstracted_axes=abstracted_axes,
            backend=backend,
            compiler_options=compiler_options,
            donate_argnames=donate_argnames,
            inline=inline,
            keep_unused=keep_unused,
        )
        return jitted_function
    jitted_function = jax.jit(
        func,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        abstracted_axes=abstracted_axes,
        backend=backend,
        compiler_options=compiler_options,
        donate_argnames=donate_argnames,
        inline=inline,
        keep_unused=keep_unused,
    )
    try:
        func_source = inspect.getsource(func)
        hardware_sig = _get_hardware_signature()
        jit_options_sig = str((static_argnums, static_argnames, donate_argnums, in_shardings, out_shardings))
        static_key_part = "".join([func_source, hardware_sig, jit_options_sig])
    except Exception as e:
        warnings.warn(
            f"Could not create static cache key for ejit function '{func.__name__}'. "
            f"Falling back to regular jit. Error: {e}",
            stacklevel=2,
        )

        return jitted_function

    static_arg_indices = (
        set(static_argnums)
        if isinstance(static_argnums, list | tuple)
        else {static_argnums}
        if static_argnums is not None
        else set()
    )
    static_arg_names_set = (
        set(static_argnames)
        if isinstance(static_argnames, list | tuple)
        else {static_argnames}
        if static_argnames is not None
        else set()
    )

    def get_compiled_and_cache(args_sig: str, args, kwargs) -> Compiled | None:
        """
        This inner function handles the "slow path": looking up in the L2 cache,
        on disk, or compiling. It's decorated with LRU cache so it only runs
        once per unique set of argument shapes.
        """
        compilation_key = hashlib.md5((static_key_part + args_sig).encode("utf-8")).hexdigest()

        if compilation_key in COMPILED_CACHE and not RECOMPILE_FORCE:
            return COMPILED_CACHE[compilation_key]

        func_dir = COMPILE_FUNC_DIR / compilation_key
        filepath = func_dir / COMPILED_FILE_NAME
        if filepath.exists() and not RECOMPILE_FORCE:
            try:
                with open(filepath, "rb") as f:
                    serialized, in_tree, out_tree = pickle.load(f)
                compiled_func = deserialize_and_load(serialized, in_tree, out_tree)
                COMPILED_CACHE[compilation_key] = compiled_func
                return compiled_func
            except Exception as e:
                warnings.warn(f"Could not load ejit cache from '{filepath}'. Recompiling. Error: {e}", stacklevel=2)

        try:
            lowered_func = jitted_function.lower(*args, **kwargs)
            compiled_func = lowered_func.compile()

            try:
                serialized, in_tree, out_tree = serialize(compiled_func)
                func_dir.mkdir(parents=True, exist_ok=True)
                with open(filepath, "wb") as f:
                    pickle.dump((serialized, in_tree, out_tree), f)
            except Exception:
                pass

            COMPILED_CACHE[compilation_key] = compiled_func
            return compiled_func
        except Exception as e:
            warnings.warn(f"ejit compilation failed for '{func.__name__}'. Error: {e}", stacklevel=2)
            return None

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            args_sig = _get_args_signature(args, kwargs)
            compiled_func = get_compiled_and_cache(args_sig, args, kwargs)

        except Exception as e:
            warnings.warn(
                f"ejit signature generation failed for '{func.__name__}'. Falling back. Error: {e}", stacklevel=2
            )
            return jitted_function(*args, **kwargs)
        if compiled_func is None:
            return jitted_function(*args, **kwargs)
        dynamic_args = tuple(arg for i, arg in enumerate(args) if i not in static_arg_indices)
        dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in static_arg_names_set}
        return compiled_func(*dynamic_args, **dynamic_kwargs)

    return wrapper


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


if __name__ == "__main__":
    jnp = jax.numpy

    @ejit
    def my_function(x, y):
        return x * y + x

    a = jnp.array([1, 2, 3], dtype=jnp.float32)
    b = jnp.array([4, 5, 6], dtype=jnp.float32)

    result1 = my_function(a, b)  # Compiles and caches on first call
    result2 = my_function(a, b)  # Returns cached result

    c = jnp.array([1, 2, 3], dtype=jnp.float32)
    d = jnp.array([1, 1, 1], dtype=jnp.float32)
    result3 = my_function(c, d)
    print(result1, result2, result3)
