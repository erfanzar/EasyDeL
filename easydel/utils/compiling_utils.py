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


from __future__ import annotations

import functools
import hashlib
import os
import pickle
import typing as tp
import warnings

import jax
import numpy as np
from jax.experimental.serialize_executable import deserialize_and_load, serialize

from .helpers import check_bool_flag, get_cache_dir

if tp.TYPE_CHECKING:
    from jax._src.stages import Compiled as JAXCompiled
    from jax._src.stages import Lowered as JAXLowered
    from jax.sharding import Mesh, Sharding
    from jax.tree_util import PyTreeDef

    Compiled = JAXCompiled
    Lowered = JAXLowered
else:
    Compiled, Lowered = tp.Any, tp.Any
    PyTreeDef, Mesh, Sharding = tp.Any, tp.Any, tp.Any


P = tp.ParamSpec("P")
R = tp.TypeVar("R")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
Pytree = tp.Any


RECOMPILE_FORCE = check_bool_flag("RECOMPILE_FORCE", False)
ECACHE_COMPILES = check_bool_flag("ECACHE_COMPILES", True)

CACHE_DIR = get_cache_dir()
COMPILE_FUNC_DIR = CACHE_DIR / "compiled_funcs"
COMPILE_FUNC_DIR.mkdir(parents=True, exist_ok=True)
COMPILED_FILE_NAME = "compiled.func"

COMPILED_CACHE: dict[tuple, tp.Any] = {}


def is_jit_wrapped(fn: tp.Any) -> bool:
    return all(
        [
            hasattr(fn, "_fun"),
            hasattr(fn, "lower"),
            hasattr(fn, "eval_shape"),
            hasattr(fn, "trace"),
        ]
    )


def cjit(
    fn: tp.Callable[P, R],
    static_argnums: tuple[int, ...] | None = None,
    static_argnames: tuple[str, ...] | None = None,
    verbose: bool = True,
):
    """
    A decorator that adds caching to a JAX JIT-compiled function.
    The input `fn` must already be a JIT-transformed function (e.g., from @jax.jit).
    """
    assert is_jit_wrapped(fn=fn), "function should be jit wrapped already"

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        static_arg_indices = set(static_argnums) if static_argnums is not None else set()
        dynamic_args = tuple(arg for i, arg in enumerate(args) if i not in static_arg_indices)
        dynamic_kwargs = kwargs.copy()
        if static_argnames is not None:
            for key in static_argnames:
                dynamic_kwargs.pop(key, None)
        signature = get_signature_tree_util(dynamic_args, dynamic_kwargs)
        cache_key = (fn, signature)
        if cache_key in COMPILED_CACHE:
            compiled_func = COMPILED_CACHE[cache_key]
            return compiled_func(*dynamic_args, **dynamic_kwargs)
        lowered_func: Lowered = fn.lower(*args, **kwargs)
        compiled_func = smart_compile(
            lowered_func=lowered_func,
            tag="cached-jit",
            verbose=verbose,
        )
        COMPILED_CACHE[cache_key] = compiled_func

        return compiled_func(*dynamic_args, **dynamic_kwargs)

    return wrapped


def hash_fn(self) -> int:
    """Generate a hash for an object based on its dictionary values.

    Args:
        self: Object to hash

    Returns:
        Integer hash value
    """
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | float | bool | dict | list))
    return get_safe_hash_int(shu)


def get_safe_hash_int(text, algorithm="md5"):
    """Generate a hash of text using specified algorithm with safety checks.

    Args:
        text: Input text to hash
        algorithm: Hash algorithm to use (default: md5)

    Returns:
        Integer representation of the hash digest

    Raises:
        ValueError: If specified algorithm is not supported
        Exception: If any error occurs during hashing
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


_leaf_types = (jax.Array, np.ndarray, int, float, bool, str, bytes, type(None))


def _is_leaf_for_signature(node):
    """Determine if a node should be considered a leaf for signature generation.

    Args:
        node: Node to check

    Returns:
        True if node is a leaf type (array, primitive type, or has shape/dtype attributes), False otherwise
    """
    if isinstance(node, jax.Array | np.ndarray):
        return True
    if isinstance(node, _leaf_types):
        return True
    if not isinstance(node, list | tuple | dict):
        try:
            _ = node.shape
            _ = node.dtype
            return True
        except AttributeError:
            return False
    return False


def get_leaf_signature(leaf: tp.Any) -> tp.Hashable:
    """Generate a hashable signature for a leaf node.

    Args:
        leaf: Input node to generate signature for

    Returns:
        Tuple of (shape, dtype) for array-like objects, type for others
    """
    if isinstance(leaf, jax.Array | np.ndarray) or (hasattr(leaf, "shape") and hasattr(leaf, "dtype")):
        try:
            shape = tuple(leaf.shape)
            dtype_str = str(jax.dtypes.canonicalize_dtype(leaf.dtype))
            return (shape, dtype_str)
        except Exception:
            return type(leaf)
    else:
        return type(leaf)


def get_signature_tree_util(
    args: tuple[tp.Any, ...],
    kwargs: dict[str, tp.Any],
) -> tuple:
    """Generate a signature tree from function arguments.

    Args:
        args: Positional arguments to process
        kwargs: Keyword arguments to process

    Returns:
        Tuple containing tree structure and leaf signatures
    """
    leaves, structure = jax.tree_util.tree_flatten(
        (args, kwargs),
        is_leaf=_is_leaf_for_signature,
    )
    leaf_signatures = leaf_signatures = tuple(map(get_leaf_signature, leaves))
    return (structure, leaf_signatures)


def get_hash_of_lowering(lowered_func: Lowered):
    text_representation = lowered_func.as_text()
    hash_object = hashlib.sha256(text_representation.encode("utf-8"))
    hash_digest = hash_object.hexdigest()
    return hash_digest


def smart_compile(
    lowered_func: Lowered,
    tag: str | None = None,
    verbose: bool = True,
) -> Compiled:
    """Compile a lowered JAX function with caching.

    Args:
        lowered_func: JAX function in lowered form
        tag: Optional tag for the compiled function
        verbose: Whether to show warning messages (default: True)

    Returns:
        Compiled JAX function
    """
    func_hash = get_hash_of_lowering(lowered_func)
    foldername = str(func_hash) if tag is None else f"{tag}-{func_hash}"
    func_dir = COMPILE_FUNC_DIR / foldername
    filepath = func_dir / COMPILED_FILE_NAME
    post_fix = f" (TAG : {tag})" if tag else ""
    if filepath.exists() and not RECOMPILE_FORCE:
        try:
            (serialized, in_tree, out_tree) = pickle.load(open(filepath, "rb"))
            compiled_func = deserialize_and_load(
                serialized=serialized,
                in_tree=in_tree,
                out_tree=out_tree,
            )
            return compiled_func
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
                except Exception as e:
                    if verbose:
                        warnings.warn(
                            f"couldn't save compiled function due to {e}" + post_fix,
                            stacklevel=4,
                        )
            return compiled_func
    else:
        compiled_func: Compiled = lowered_func.compile()
        if ECACHE_COMPILES:
            try:
                serialized, in_tree, out_tree = serialize(compiled_func)
                func_dir.mkdir(parents=True, exist_ok=True)
                pickle.dump((serialized, in_tree, out_tree), open(filepath, "wb"))
            except Exception as e:
                if verbose:
                    warnings.warn(
                        f"couldn't save and serialize compiled function due to {e}" + post_fix,
                        stacklevel=4,
                    )
        return compiled_func


def save_compiled_fn(
    path: str | os.PathLike,
    fn: Compiled,
    prefix: str | None = None,
):
    """Save a compiled function to disk with its serialization metadata.

    Args:
        path: Directory path to save the function
        fn: Compiled function to save
        prefix: Optional prefix for the filename

    Raises:
        Warning: If saving fails
    """
    path.mkdir(parents=True, exist_ok=True)
    prefix = prefix or ""
    filename = path / (prefix + "-" + COMPILED_FILE_NAME)
    serialized, in_tree, out_tree = serialize(fn)
    try:
        pickle.dump((serialized, in_tree, out_tree), open(filename, "wb"))
    except Exception as e:
        warnings.warn(f"couldn't save compiled function due to {e}", stacklevel=4)


def load_compiled_fn(
    path: str | os.PathLike,
    prefix: str | None = None,
):
    """Load a compiled function from disk.

    Args:
        path: Directory path to load from
        prefix: Optional prefix used when saving the function

    Returns:
        Deserialized compiled function

    Raises:
        If file loading or deserialization fails
    """
    prefix = prefix or ""
    filename = path / (prefix + "-" + COMPILED_FILE_NAME)
    (serialized, in_tree, out_tree) = pickle.load(open(filename, "rb"))
    return deserialize_and_load(
        serialized=serialized,
        in_tree=in_tree,
        out_tree=out_tree,
    )


def cache_compiles(
    tag: str | None = None,
    static_argnames: list[str] | None = None,
):
    """Create a decorator for caching compiled functions.

    Args:
        tag: Optional tag for cache identification
        static_argnames: List of static argument names to exclude from signature

    Returns:
        Decorator function for caching compiled functions
    """
    static_argnames = static_argnames or []

    def create_wrapper(func: tp.Callable, tag: str | None = None) -> tp.Callable:
        original_func = getattr(func, "_fun", func)
        func_id = str(
            hashlib.sha256(
                original_func.__code__.co_code,
            )
            .hexdigest()
            .encode("utf-8")
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signature = (func_id, get_signature_tree_util(args, kwargs))
            if signature in COMPILED_CACHE:
                for static_key in static_argnames:
                    kwargs.pop(static_key)
                return COMPILED_CACHE[signature](*args, **kwargs)
            if hasattr(func, "lower"):
                lowered = func.lower(*args, **kwargs)
                for static_key in static_argnames:
                    kwargs.pop(static_key)
                func_hash = get_hash_of_lowering(lowered)
                sig_hash = hashlib.sha256(str(signature).encode()).hexdigest()[:8]
                foldername = f"{tag}-{func_hash}-{sig_hash}" if tag else f"{func_hash}-{sig_hash}"
                func_dir = COMPILE_FUNC_DIR / foldername
                filepath = func_dir / "compiled.func"

                if filepath.exists() and not RECOMPILE_FORCE:
                    with open(filepath, "rb") as f:
                        serialized, in_tree, out_tree = pickle.load(f)
                    compiled_func = deserialize_and_load(
                        serialized=serialized,
                        in_tree=in_tree,
                        out_tree=out_tree,
                    )
                    COMPILED_CACHE[signature] = compiled_func
                    return compiled_func(*args, **kwargs)

                compiled_func = lowered.compile()
                COMPILED_CACHE[signature] = compiled_func

                try:
                    serialized, in_tree, out_tree = serialize(compiled_func)
                    func_dir.mkdir(parents=True, exist_ok=True)
                    with open(filepath, "wb") as f:
                        pickle.dump((serialized, in_tree, out_tree), f)
                except Exception as e:
                    print(f"Failed to cache compilation: {e}")

                return compiled_func(*args, **kwargs)
            return func(*args, **kwargs)

        wrapper._COMPILED_CACHE = COMPILED_CACHE
        return wrapper

    def decorator(func: tp.Callable) -> tp.Callable:
        return create_wrapper(func, tag)

    return decorator


def lower_function(
    func: tp.Callable[P, R],
    func_input_args: P.args,  # type:ignore
    func_input_kwargs: P.kwargs,  # type:ignore
    mesh: Mesh | None = None,
    in_shardings: Pytree = None,
    out_shardings: Pytree = None,
    static_argnums: int | tp.Sequence[int] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
) -> Lowered:
    """Lowers a JAX function with specified configurations."""
    """
    lower a JAX function with optional sharding and mesh configuration.

    Args:
        func: The JAX function to compile.
        func_input_args: Input arguments for the function.
        func_input_kwargs: Input keyword arguments for the function.
        mesh: tp.Optional JAX mesh for distributed execution.
        in_shardings: tp.Optional input sharding specifications.
        out_shardings: tp.Optional output sharding specifications.
        static_argnums: Indices of static arguments.
        donate_argnums: Indices of arguments to donate.

    Returns:
        lowered JAX function.
    """
    jit_options: dict[str, tp.Any] = {
        "in_shardings": in_shardings,
        "out_shardings": out_shardings,
        "static_argnums": static_argnums,
        "donate_argnums": donate_argnums,
    }

    jitted_func = jax.jit(func, **jit_options)

    if mesh is None:
        return jitted_func.lower(*func_input_args, **func_input_kwargs)  # type: ignore[attr-defined]
    else:
        with mesh:  # type: ignore[attr-defined] # mesh is a context manager
            return jitted_func.lower(*func_input_args, **func_input_kwargs)  # type: ignore[attr-defined]


def compile_function(
    func: tp.Callable[P, R],
    func_input_args: P.args,  # type:ignore
    func_input_kwargs: P.kwargs,  # type:ignore
    mesh: Mesh | None = None,
    in_shardings: Pytree = None,
    out_shardings: Pytree = None,
    static_argnums: int | tp.Sequence[int] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
) -> Compiled:
    """
    Compiles a JAX function with optional sharding and mesh configuration.

    Args:
        func: The JAX function to compile.
        func_input_args: Input arguments for the function.
        func_input_kwargs: Input keyword arguments for the function.
        mesh: tp.Optional JAX mesh for distributed execution.
        in_shardings: tp.Optional input sharding specifications.
        out_shardings: tp.Optional output sharding specifications.
        static_argnums: Indices of static arguments.
        donate_argnums: Indices of arguments to donate.

    Returns:
        Compiled JAX function.
    """
    return lower_function(
        func,
        func_input_args,
        func_input_kwargs,
        mesh=mesh,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    ).compile()


if __name__ == "__main__":
    jnp = jax.numpy

    @cjit
    @jax.jit
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
