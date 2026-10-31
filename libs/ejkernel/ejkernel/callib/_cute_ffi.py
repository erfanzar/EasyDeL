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

"""JAX primitive integration for CuTe DSL kernels via TVM-FFI.

This module provides a Triton-style primitive that performs abstract evaluation
from output shape contracts and lowers execution through JAX FFI targets
registered by ``jax_tvm_ffi``.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import threading
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.extend as jex
import jax.numpy as jnp
from jax import tree_util
from jax.interpreters import ad, batching, mlir, xla

_HAS_CUDA_BINDINGS = False
try:
    import cuda.bindings.driver as cuda

    _HAS_CUDA_BINDINGS = True
except Exception:
    cuda = None

_HAS_JAX_TVM_FFI = False
try:
    import jax_tvm_ffi

    _HAS_JAX_TVM_FFI = True
except Exception:
    jax_tvm_ffi = None

CAN_USE_CUTE_PRIMITIVE = False
try:
    import cutlass
    import cutlass.cute as cute

    CAN_USE_CUTE_PRIMITIVE = True
except ModuleNotFoundError:
    pass


if CAN_USE_CUTE_PRIMITIVE:
    _DTYPE_TO_CUTLASS: dict[jnp.dtype, type[cutlass.Numeric]] = {
        jnp.dtype(jnp.float16): cutlass.Float16,
        jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
        jnp.dtype(jnp.float32): cutlass.Float32,
        jnp.dtype(jnp.int8): cutlass.Int8,
        jnp.dtype(jnp.uint8): cutlass.Uint8,
        jnp.dtype(jnp.int32): cutlass.Int32,
        jnp.dtype(jnp.uint32): cutlass.Uint32,
    }
else:  # pragma: no cover
    _DTYPE_TO_CUTLASS = {}


@dataclass(frozen=True)
class _CompiledKernel:
    """Container for a compiled CuTe callable and its FFI target name.

    Attributes:
        target_name: The unique FFI target name used to register the kernel
            with JAX via TVM-FFI.
        compiled: The compiled CuTe callable object produced by
            ``cute.compile``.
    """

    target_name: str
    compiled: Any


_COMPILE_CACHE: dict[tuple[Any, ...], _CompiledKernel] = {}
_COMPILE_LOCK = threading.Lock()
_REGISTERED_TARGETS: set[str] = set()
_REGISTERED_TARGETS_LOCK = threading.Lock()


def _to_shape_dtype_struct(out_shape: Any) -> Any:
    """Normalize output descriptors into ``jax.ShapeDtypeStruct`` leaves.

    Args:
        out_shape: A pytree of objects with ``shape`` and ``dtype`` attributes.

    Returns:
        A pytree with the same structure where each leaf is replaced by a
        ``jax.ShapeDtypeStruct``.
    """
    return tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)


def _shape_dtype_key(shaped: Any) -> tuple[int, jnp.dtype]:
    """Build a stable compile-cache key fragment from a shaped value.

    Args:
        shaped: An object with ``shape`` and ``dtype`` attributes.

    Returns:
        A tuple of ``(rank, dtype)`` suitable for use as a cache key component.
    """
    return (len(tuple(shaped.shape)), jnp.dtype(shaped.dtype))


def _fake_tensor_from_shaped(shaped: Any):
    """Create a fake compact tensor from shape/dtype metadata.

    Builds a symbolic fake tensor via ``cute.runtime.make_fake_compact_tensor``
    suitable for passing to ``cute.compile`` at kernel-compilation time.  Uses
    ``assumed_align=16`` (16 bytes = 128 bits) so that CuTe DSL kernels using
    ``CopyG2SOp`` with 128-bit cp.async copies can pass MLIR alignment
    verification.  JAX/XLA guarantees at least 128-byte alignment for device
    buffers, so this is always safe in practice.

    Args:
        shaped: An object with ``shape`` and ``dtype`` attributes.

    Returns:
        A fake compact tensor with symbolic shape and stride, suitable for
        passing to ``cute.compile`` as a placeholder.

    Raises:
        TypeError: If ``shaped.dtype`` is not in the supported dtype map
            (float16, bfloat16, float32, int8, uint8, int32, uint32).
    """
    dtype = jnp.dtype(shaped.dtype)
    cutlass_dtype = _DTYPE_TO_CUTLASS.get(dtype)
    if cutlass_dtype is None:
        raise TypeError(f"Unsupported dtype for CuTe primitive path: {dtype}")
    rank = len(tuple(shaped.shape))
    sym_shape = tuple(cute.sym_int() for _ in range(rank))
    stride_order = tuple(range(rank - 1, -1, -1))
    return cute.runtime.make_fake_compact_tensor(
        cutlass_dtype,
        sym_shape,
        stride_order=stride_order,
        assumed_align=16,
    )


def _fn_expects_stream(fn: Any) -> bool:
    """Return whether the compiled host launcher expects a leading stream arg.

    Inspects the function's signature to determine if the first positional
    parameter is named ``stream``.

    Args:
        fn: A callable whose signature will be inspected.

    Returns:
        ``True`` if the first positional parameter is named ``stream``,
        ``False`` otherwise or if the signature cannot be inspected.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except Exception:
        return False
    if not params:
        return False
    first = params[0]
    if first.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return False
    return first.name == "stream"


def _make_fake_stream() -> Any:
    """Create a compile-time stream placeholder for TVM-FFI CuTe kernels.

    Tries ``cute.runtime.make_fake_stream`` first, then falls back to a
    null CUDA stream from the ``cuda.bindings`` package.

    Returns:
        A fake or null stream object suitable for CuTe compilation.

    Raises:
        RuntimeError: If neither ``cute.runtime`` nor ``cuda.bindings``
            can provide a stream object.
    """
    try:
        return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    except Exception:
        pass
    if _HAS_CUDA_BINDINGS:
        return cuda.CUstream(0)
    raise RuntimeError("Unable to create a CuTe fake stream for TVM-FFI compilation.")


def _cache_key_hash(cache_key: tuple[Any, ...]) -> str:
    """Build a deterministic hash string from a compile cache key.

    Args:
        cache_key: A tuple of values representing the compilation parameters.

    Returns:
        A SHA-256 hex-digest string derived from the cache key's ``repr``.
    """
    return hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()


def _compile_or_get_kernel(
    *,
    fn: Any,
    in_shaped: tuple[Any, ...],
    out_shaped: tuple[jax.ShapeDtypeStruct, ...],
    compile_options: str | None,
    static_kwargs: tuple[tuple[str, Any], ...],
) -> _CompiledKernel:
    """Compile (or fetch cached) CuTe callable and FFI target metadata.

    Looks up the compile cache by a key derived from the function identity,
    input/output shapes and dtypes, compile options, and static kwargs. On a
    cache miss, compiles the kernel using fake tensors and stores the result.

    Args:
        fn: The ``@cute.jit`` launcher callable to compile.
        in_shaped: Tuple of input shaped objects (with ``shape`` and ``dtype``).
        out_shaped: Tuple of ``jax.ShapeDtypeStruct`` for expected outputs.
        compile_options: Optional options string forwarded to ``cute.compile``.
        static_kwargs: Tuple of ``(name, value)`` pairs for static keyword
            arguments passed at compile time.

    Returns:
        A ``_CompiledKernel`` containing the compiled callable and its
        unique FFI target name.
    """
    cache_key = (
        fn,
        tuple(_shape_dtype_key(arg) for arg in in_shaped),
        tuple(_shape_dtype_key(arg) for arg in out_shaped),
        compile_options,
        static_kwargs,
    )

    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        in_fake = [_fake_tensor_from_shaped(arg) for arg in in_shaped]
        out_fake = [_fake_tensor_from_shaped(arg) for arg in out_shaped]

        expects_stream = _fn_expects_stream(fn)
        compile_kwargs = dict(static_kwargs)

        def _compile(add_stream: bool):
            """Run ``cute.compile`` with or without a leading stream argument.

            Args:
                add_stream: If ``True``, prepend a fake stream to the
                    compile arguments.

            Returns:
                The compiled CuTe callable.
            """
            compile_args: list[Any] = []
            if add_stream:
                compile_args.append(_make_fake_stream())
            compile_args.extend(in_fake)
            compile_args.extend(out_fake)
            if compile_options:
                return cute.compile(fn, *compile_args, options=compile_options, **compile_kwargs)
            return cute.compile(fn, *compile_args, **compile_kwargs)

        try:
            compiled = _compile(expects_stream)
        except Exception:
            compiled = _compile(not expects_stream)

        digest = _cache_key_hash(cache_key)[:24]
        target_name = f"ejkernel_cute_tvm_ffi_{digest}"
        result = _CompiledKernel(target_name=target_name, compiled=compiled)
        _COMPILE_CACHE[cache_key] = result
        return result


def _register_target_once(kernel: _CompiledKernel) -> None:
    """Register a compiled CuTe callable as a JAX FFI target exactly once.

    Uses ``jax_tvm_ffi.register_ffi_target`` to make the compiled kernel
    available to the JAX FFI lowering path. Tries multiple platform strings
    (``gpu``, ``cuda``, ``CUDA``, and unspecified) to accommodate different
    JAX/XLA runtime configurations.

    Args:
        kernel: The ``_CompiledKernel`` to register.

    Raises:
        ValueError: If ``jax_tvm_ffi`` is not installed.
        RuntimeError: If all registration attempts fail.
    """
    if not _HAS_JAX_TVM_FFI:
        raise ValueError("CuTe primitive path requires `jax_tvm_ffi` (apache-tvm-ffi) to register TVM-FFI targets.")

    with _REGISTERED_TARGETS_LOCK:
        if kernel.target_name in _REGISTERED_TARGETS:
            return

        error: Exception | None = None
        register_fns = (
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.compiled, platform="gpu"),
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.compiled, platform="cuda"),
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.compiled, platform="CUDA"),
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.compiled),
        )
        for register_fn in register_fns:
            try:
                register_fn()
                _REGISTERED_TARGETS.add(kernel.target_name)
                return
            except Exception as exc:  # pragma: no cover - exercised only in incompatible runtime envs
                error = exc

        raise RuntimeError(f"Failed to register CuTe TVM-FFI target `{kernel.target_name}`.") from error


def _cute_kernel_call_impl(
    *args_flat,
    fn: Any,
    out_shape_dtype_flat: tuple[jax.ShapeDtypeStruct, ...],
    input_output_aliases: tuple[tuple[int, int], ...],
    compile_options: str | None,
    static_kwargs: tuple[tuple[str, Any], ...],
):
    """Primitive implementation shared by eager and lowering paths.

    Compiles (or retrieves from cache) the CuTe kernel, registers it as a
    JAX FFI target, and dispatches execution via ``jax.ffi.ffi_call``.

    Args:
        *args_flat: Flattened input arrays for the kernel.
        fn: The ``@cute.jit`` launcher callable.
        out_shape_dtype_flat: Tuple of ``jax.ShapeDtypeStruct`` for outputs.
        input_output_aliases: Tuple of ``(input_idx, output_idx)`` pairs for
            in-place aliasing.
        compile_options: Optional options string for ``cute.compile``.
        static_kwargs: Tuple of ``(name, value)`` pairs for static kwargs.

    Returns:
        The output arrays produced by the FFI call.

    Raises:
        ValueError: If CuTe is not available.
    """
    if not CAN_USE_CUTE_PRIMITIVE:
        raise ValueError("CuTe primitive path requires CUTLASS CuTe DSL.")

    kernel = _compile_or_get_kernel(
        fn=fn,
        in_shaped=tuple(args_flat),
        out_shaped=out_shape_dtype_flat,
        compile_options=compile_options,
        static_kwargs=static_kwargs,
    )
    _register_target_once(kernel)

    alias_map = dict(input_output_aliases)
    ffi_call = jax.ffi.ffi_call(
        kernel.target_name,
        result_shape_dtypes=out_shape_dtype_flat,
        input_output_aliases=alias_map,
    )
    return ffi_call(*args_flat)


cute_kernel_call_p = jex.core.Primitive("ejkernel_cute_kernel_call")
cute_kernel_call_p.multiple_results = True
cute_kernel_call_p.def_impl(functools.partial(xla.apply_primitive, cute_kernel_call_p))


@cute_kernel_call_p.def_abstract_eval
def _cute_kernel_call_abstract_eval(*_, out_shape_dtype_flat, **__):
    """Primitive abstract evaluation returning output avals.

    Args:
        *_: Unused positional arguments (input avals).
        out_shape_dtype_flat: Tuple of ``jax.ShapeDtypeStruct`` defining
            the expected output shapes and dtypes.
        **__: Unused keyword arguments.

    Returns:
        List of ``jax.core.ShapedArray`` abstract values for each output.
    """
    return [jax.core.ShapedArray(x.shape, x.dtype) for x in out_shape_dtype_flat]


def _raise_on_jvp(*args, **kwargs):
    """Raise for unsupported automatic differentiation.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always, as the CuTe TVM-FFI primitive does
            not support JVP or transpose rules.
    """
    del args, kwargs
    raise NotImplementedError(
        "CuTe TVM-FFI primitive does not support automatic differentiation. Use `jax.custom_jvp` or `jax.custom_vjp`."
    )


def _raise_on_vmap(*args, **kwargs):
    """Raise for unsupported batching.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always, as the CuTe TVM-FFI primitive does
            not support batching via ``jax.vmap``.
    """
    del args, kwargs
    raise NotImplementedError(
        "CuTe TVM-FFI primitive does not support batching via `jax.vmap`. Use `jax.custom_batching.custom_vmap`."
    )


ad.primitive_jvps[cute_kernel_call_p] = _raise_on_jvp
ad.primitive_transposes[cute_kernel_call_p] = _raise_on_jvp
batching.primitive_batchers[cute_kernel_call_p] = _raise_on_vmap

mlir.register_lowering(
    cute_kernel_call_p,
    mlir.lower_fun(_cute_kernel_call_impl, multiple_results=True),
    platform="cuda",
)


def build_cute_ffi_call(
    fn: Any,
    *,
    output_shape_dtype: Any,
    input_output_aliases: dict[int, int] | None = None,
    compile_options: str | None = "--enable-tvm-ffi",
    **static_kwargs: Any,
):
    """Create a callable that dispatches a CuTe kernel through a JAX primitive.

    Wraps a ``@cute.jit`` launcher so that it can be called from within
    JAX-traced computations (``jax.jit``, ``jax.grad``, etc.).  Compilation
    is deferred until the first call and is keyed on input/output shapes and
    dtypes, so the same callable works for dynamically sized inputs as long as
    the rank and dtype stay fixed.

    Args:
        fn: A ``@cute.jit`` launcher callable whose positional arguments are
            (optionally) a stream, followed by input tensors, followed by output
            tensors.  The function signature is inspected at compile time to
            detect whether a stream argument is expected.
        output_shape_dtype: A pytree of objects with ``shape`` and ``dtype``
            attributes (e.g. ``jax.ShapeDtypeStruct``, ``jax.Array``,
            ``jax.eval_shape`` outputs) describing the output tensors.
        input_output_aliases: Optional ``{input_idx: output_idx}`` alias map
            (using flattened indices) for in-place operations.  Passed through
            to ``jax.ffi.ffi_call``.
        compile_options: Optional options string forwarded to ``cute.compile``.
            Defaults to ``'--enable-tvm-ffi'`` which is required for TVM-FFI
            kernel registration.  Pass ``None`` to omit the flag entirely.
        **static_kwargs: Additional keyword arguments forwarded to
            ``cute.compile`` at compilation time (not at call time).  They are
            included in the compilation cache key.

    Returns:
        A callable ``call(*inputs) -> outputs`` that traces through the
        ``ejkernel_cute_kernel_call`` JAX primitive and lowers to
        ``jax.ffi.ffi_call`` backed by the compiled CuTe kernel.
    """
    out_shape = _to_shape_dtype_struct(output_shape_dtype)
    flat_out_shape, out_tree = tree_util.tree_flatten(out_shape)
    static_items = tuple(static_kwargs.items())
    alias_items = tuple(sorted((input_output_aliases or {}).items()))

    @partial(jax.jit, inline=True)
    def _call(*args):
        """Dispatch the CuTe kernel through the JAX primitive.

        Args:
            *args: Runtime input arrays (pytree-flattened internally).

        Returns:
            Output pytree matching the ``output_shape_dtype`` structure.
        """
        args_flat, _ = tree_util.tree_flatten(args)
        out_flat = cute_kernel_call_p.bind(
            *args_flat,
            fn=fn,
            out_shape_dtype_flat=tuple(flat_out_shape),
            input_output_aliases=alias_items,
            compile_options=compile_options,
            static_kwargs=static_items,
        )
        return tree_util.tree_unflatten(out_tree, out_flat)

    return _call


def has_cute_ffi_support() -> bool:
    """Return whether the CuTe TVM-FFI primitive path can be used.

    Returns:
        ``True`` if both ``cutlass`` (with ``cutlass.cute``) and
        ``jax_tvm_ffi`` are importable, ``False`` otherwise.
    """
    if not CAN_USE_CUTE_PRIMITIVE or not _HAS_JAX_TVM_FFI:
        return False
    return True


__all__ = ["build_cute_ffi_call", "has_cute_ffi_support"]
