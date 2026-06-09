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
    RECOMPILE_FORCE: Force recompilation flag (``EASYDEL_RECOMPILE_FORCE``).
    ECACHE_COMPILES: Enable two-level compilation caching (``EASYDEL_CACHE_COMPILES``).
    ALLOW_FULL_CACHE: Persist all XLA caches including tiny entries (``ALLOW_FULL_CACHE``).
    CACHE_DIR: Platform-specific base cache directory.
    COMPILE_FUNC_DIR: Directory for serialized compiled executables
        (``COMPILE_FUNC_DIR`` env var or ``<CACHE_DIR>/ejit_compiled_functions``).
    COMPILED_CACHE: Module-level in-memory dict of compiled ``Compiled`` objects.

Key Features:
    - Persistent disk caching of compiled functions
    - Automatic cache invalidation on source, hardware, or argument-signature changes
    - Hardware-specific cache keys (device list included in signature)
    - Two-level caching (in-memory dict + disk pickle)
    - Graceful fallback to standard ``jax.jit`` on any error

Example:
    >>> from ejkernel.callib import ejit
    >>>
    >>> @ejit
    ... def optimized_fn(x, y):
    ...     return x @ y + x.T @ y.T
    >>>
    >>> result = optimized_fn(a, b)
    >>> result = optimized_fn(a, b)  # served from cache on second call
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

from ._utils import check_bool_flag, get_cache_dir

if tp.TYPE_CHECKING:
    from jax._src.stages import Compiled, Lowered

P = tp.ParamSpec("P")
R = tp.TypeVar("R")

RECOMPILE_FORCE = check_bool_flag("EASYDEL_RECOMPILE_FORCE", False)
ECACHE_COMPILES = check_bool_flag("EASYDEL_CACHE_COMPILES", False)
ALLOW_FULL_CACHE = check_bool_flag("ALLOW_FULL_CACHE", False)

CACHE_DIR = get_cache_dir()
COMPILE_FUNC_DIR = os.getenv("COMPILE_FUNC_DIR", CACHE_DIR / "ejit_compiled_functions")

if not isinstance(COMPILE_FUNC_DIR, str) and hasattr(COMPILE_FUNC_DIR, "mkdir"):
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
    compiler_options: dict[str, tp.Any] | None = None,
):
    """Enhanced JIT compilation with persistent caching.

    Drop-in replacement for ``jax.jit`` that caches compiled functions
    to disk for reuse across script runs, significantly reducing
    compilation overhead. Can be used as a bare decorator or called
    with keyword arguments.

    Features:
        - Two-level caching (in-memory dict + persistent disk cache)
        - Automatic cache invalidation on source, hardware, or signature changes
        - Graceful fallback to standard ``jax.jit`` on errors
        - Support for all ``jax.jit`` parameters

    Args:
        func: Function to JIT-compile and cache. When ``None``, returns a
            partial decorator so that ``@ejit(...)`` syntax works.
        static_argnums: Indices of arguments to treat as compile-time constants.
        static_argnames: Names of arguments to treat as compile-time constants.
        donate_argnums: Indices of arguments whose buffers may be donated to outputs.
        in_shardings: Input sharding specifications for distributed execution.
        out_shardings: Output sharding specifications for distributed execution.
        donate_argnames: Names of arguments whose buffers may be donated.
        keep_unused: Whether to keep unused arguments in the compiled function.
        backend: JAX backend to use (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``).
        inline: Whether to inline the function into the caller.
        compiler_options: Dictionary of additional XLA compiler options.

    Returns:
        JIT-compiled function with persistent caching when ``EASYDEL_CACHE_COMPILES``
        is set, otherwise a standard ``jax.jit``-compiled function with XLA cache
        directory configured.

    Note:
        Caching behavior is controlled by two environment variables:

        - ``EASYDEL_CACHE_COMPILES``: Enable the custom two-level cache (default: off).
        - ``EASYDEL_RECOMPILE_FORCE``: Force recompilation ignoring cache (default: off).
        - ``ALLOW_FULL_CACHE``: Enable full XLA persistent caching (default: off).

    Example:
        >>> @ejit
        ... def fast_matmul(a, b):
        ...     return a @ b
        >>>
        >>> result = fast_matmul(x, y)
        >>>
        >>> @ejit(static_argnums=(2,))
        ... def scaled_matmul(a, b, scale):
        ...     return a @ b * scale
    """
    if func is None:
        return functools.partial(
            ejit,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
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
        if ALLOW_FULL_CACHE:
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
            jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

        jitted_function = jax.jit(
            func,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            donate_argnums=donate_argnums,
            in_shardings=in_shardings,
            out_shardings=out_shardings,
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
        backend=backend,
        compiler_options=compiler_options,
        donate_argnames=donate_argnames,
        inline=inline,
        keep_unused=keep_unused,
    )
    jit_options_sig = str((static_argnums, static_argnames, donate_argnums, in_shardings, out_shardings))
    static_key_part: str | None = None
    static_key_disabled = False

    def _resolve_static_key_part() -> str | None:
        """Lazily resolve cache key components to avoid import-time backend init."""
        nonlocal static_key_part, static_key_disabled
        if static_key_disabled:
            return None
        if static_key_part is not None:
            return static_key_part
        try:
            func_source = inspect.getsource(func)
            hardware_sig = _get_hardware_signature()
            static_key_part = "".join([func_source, hardware_sig, jit_options_sig])
            return static_key_part
        except Exception as e:
            static_key_disabled = True
            warnings.warn(
                f"Could not create static cache key for ejit function '{func.__name__}'. "
                f"Falling back to regular jit. Error: {e}",
                stacklevel=2,
            )
            return None

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

    def get_compiled_and_cache(static_key: str, args_sig: str, args, kwargs) -> Compiled | None:
        """Retrieve or compile a function with the given argument signature.

        Implements the three-level lookup and compilation "slow path":

        1. Check in-memory ``COMPILED_CACHE`` dict (L1 / fastest).
        2. Check on-disk pickle under ``COMPILE_FUNC_DIR`` (L2).
        3. Compile via ``jitted_function.lower(...).compile()`` and persist (L3).

        Serialization failures on disk writes are silently swallowed so that a
        read-only filesystem never prevents execution.

        Args:
            static_key: Stable string incorporating function source, hardware
                signature, and JIT options — used as the non-argument portion
                of the cache key.
            args_sig: String signature of the dynamic arguments (shapes, dtypes,
                shardings). Combined with ``static_key`` to form the MD5 hash.
            args: Positional arguments forwarded to ``jitted_function.lower``.
            kwargs: Keyword arguments forwarded to ``jitted_function.lower``.

        Returns:
            The compiled ``Compiled`` object if successful, ``None`` if
            compilation itself failed (caller falls back to ``jitted_function``).
        """
        compilation_key = hashlib.md5((static_key + args_sig).encode("utf-8")).hexdigest()

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
        """Execute the cached compiled function or fall back to jitted version.

        Attempts to resolve a cached compiled executable for the given
        argument signature. If caching or compilation fails at any stage,
        transparently falls back to the standard ``jax.jit`` path.

        Args:
            *args: Positional arguments forwarded to the original function.
            **kwargs: Keyword arguments forwarded to the original function.

        Returns:
            The result of calling the compiled (or jitted) function.
        """
        try:
            local_static_key_part = _resolve_static_key_part()
            if local_static_key_part is None:
                return jitted_function(*args, **kwargs)
            args_sig = _get_args_signature(args, kwargs)
            compiled_func = get_compiled_and_cache(local_static_key_part, args_sig, args, kwargs)

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
    """Pre-load all cached compiled functions from disk into memory.

    Scans the cache directory and loads all valid compiled functions into
    the in-memory cache for faster subsequent lookups.

    Args:
        verbose: If True, print status messages and warnings about loading.

    Note:
        This is useful at startup to warm up the cache before running
        performance-critical code paths.
    """
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
        >>>
        >>> jitted = jax.jit(my_function)
        >>> lowered = jitted.lower(sample_input)
        >>> compiled = lowered.compile()
        >>>
        >>>
        >>> from pathlib import Path
        >>> cache_dir = Path("./my_cache")
        >>> save_compiled_fn(cache_dir, compiled, prefix="model_v1")
        >>>
        >>>

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
    """Load a previously saved compiled function from disk.

    Reconstructs a ``Compiled`` JAX object from the pickle produced by
    :func:`save_compiled_fn`, using ``jax.experimental.serialize_executable``.

    Args:
        path: Directory path where the compiled function was saved.
        prefix: Optional prefix that was used when saving. When provided, the
            filename is ``{prefix}-compiled.executable``; otherwise just
            ``compiled.executable``.

    Returns:
        The deserialized ``Compiled`` JAX function, ready to call directly
        (bypassing re-compilation).

    Raises:
        FileNotFoundError: If the compiled function file doesn't exist at
            the resolved path.
        pickle.UnpicklingError: If the file is corrupt or incompatible.

    Note:
        Compiled functions are hardware- and XLA-version-specific.  Loading
        a file produced on a different GPU model or JAX version will likely
        raise an error inside ``deserialize_and_load``.
    """
    from pathlib import Path

    path = Path(path)
    prefix = prefix or ""
    filename = path / (prefix + "-" + COMPILED_FILE_NAME if prefix else COMPILED_FILE_NAME)
    serialized, in_tree, out_tree = pickle.load(open(filename, "rb"))
    return deserialize_and_load(
        serialized=serialized,
        in_tree=in_tree,
        out_tree=out_tree,
    )


def hash_fn(self) -> int:
    """Generate a deterministic integer hash for an object based on its attribute values.

    Concatenates the string representations of all hashable-type attribute values
    (``float``, ``int``, ``bool``, ``dict``, ``list``) from the object's ``__dict__``
    and produces an MD5-based integer hash.

    Args:
        self: Any object with a ``__dict__`` containing primitive-typed values.

    Returns:
        Deterministic positive integer hash derived from the object's attributes.

    Note:
        Only attributes of type ``float``, ``int``, ``bool``, ``dict``, or ``list``
        contribute to the hash. Other attribute types are silently ignored.
    """
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list))
    return get_safe_hash_int(shu)


def get_safe_hash_int(text, algorithm="md5"):
    """Generate a deterministic integer hash of text using the specified algorithm.

    Converts the input to a string, hashes it with the given ``hashlib``
    algorithm, and returns the digest as a big-endian unsigned integer.

    Args:
        text: Input to hash. Will be converted to ``str`` before hashing.
        algorithm: Name of a ``hashlib`` algorithm (e.g., ``'md5'``, ``'sha256'``).
            Defaults to ``'md5'``.

    Returns:
        Non-negative integer representing the hash digest.

    Raises:
        ValueError: If the specified algorithm is not supported by ``hashlib``.
        Exception: For any other hashing failure.

    Example:
        >>> get_safe_hash_int("hello world")  # doctest: +SKIP
        295242985...
        >>> get_safe_hash_int("hello world", algorithm="sha256")  # doctest: +SKIP
        805318394...
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


def get_hash_of_lowering(lowered_func: Lowered) -> str:
    """Generate a SHA-256 hash of a lowered JAX function.

    Creates a deterministic hash based on the text representation of the
    lowered function, useful for cache key generation.

    Args:
        lowered_func: JAX lowered function object.

    Returns:
        Hexadecimal string of the SHA-256 hash.
    """
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
    """Compile a lowered JAX function with intelligent disk caching.

    Attempts to load a previously compiled version from the disk cache
    (``COMPILE_FUNC_DIR/<tag>-<sha256>/compiled.executable``), falling back
    to fresh compilation if not found or if the cache entry is corrupt.
    Newly compiled functions are persisted to disk only when
    ``EASYDEL_CACHE_COMPILES`` is set; otherwise only the in-memory path via
    :func:`ejit` is used.

    Args:
        lowered_func: A ``jax.stages.Lowered`` object — typically the result
            of calling ``jax.jit(fn).lower(*args)``.
        tag: Optional human-readable tag prepended to the directory name under
            ``COMPILE_FUNC_DIR`` (e.g. a model name). Useful for organizing
            multiple compiled functions in the same cache root.
        verbose: If ``True``, emit warnings via the ``warnings`` module when
            loading or saving fails.
        cache_key: Optional caller-supplied signature tuple that is persisted
            alongside the compiled binary in a ``compiled.signature`` file. On
            a cache hit the stored signature is returned to the caller.

    Returns:
        A tuple ``(compiled_func, effective_cache_key)`` where
        ``compiled_func`` is the ready-to-call ``Compiled`` object and
        ``effective_cache_key`` is either the signature loaded from disk (on
        a clean hit) or the ``cache_key`` argument passed in.

    Note:
        The cache directory key is ``SHA-256(lowered_func.as_text())``, so the
        cache is automatically invalidated whenever the lowered MLIR changes
        (e.g. after a JAX or XLA version upgrade).
    """
    func_hash = get_hash_of_lowering(lowered_func)
    foldername = str(func_hash) if tag is None else f"{tag}-{func_hash}"
    func_dir = COMPILE_FUNC_DIR / foldername
    filepath = func_dir / COMPILED_FILE_NAME
    signature_filepath = func_dir / SIGNATURE_FILE_NAME
    post_fix = f" (TAG : {tag})" if tag else ""
    signature = cache_key

    if filepath.exists() and not RECOMPILE_FORCE:
        try:
            serialized, in_tree, out_tree = pickle.load(open(filepath, "rb"))
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
        """Example function demonstrating ejit usage."""
        return x * y + x

    a = jnp.array([1, 2, 3], dtype=jnp.float32)
    b = jnp.array([4, 5, 6], dtype=jnp.float32)

    result1 = my_function(a, b)
    result2 = my_function(a, b)

    c = jnp.array([1, 2, 3], dtype=jnp.float32)
    d = jnp.array([1, 1, 1], dtype=jnp.float32)
    result3 = my_function(c, d)
    print(result1, result2, result3)
