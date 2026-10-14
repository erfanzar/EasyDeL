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

"""JAX primitive integration for tile-lang kernels via TVM-FFI.

Mirrors :mod:`ejkernel.callib._cute_ffi`: a custom JAX primitive performs
abstract evaluation from a declared output shape contract and lowers to
``jax.ffi.ffi_call`` against a TVM-FFI target produced by
``tilelang.compile(..., execution_backend="tvm_ffi")``.

Design contract (must match ``_cute_ffi``):
    * The user authors a ``@T.prim_func`` whose parameters list **all**
      buffers, inputs first then outputs. The output buffers must NOT be
      auto-allocated by tile-lang, i.e. ``out_idx`` must be omitted /
      ``None``. JAX pre-allocates them and passes them to the FFI call.
    * ``build_tilelang_ffi_call`` returns a primitive-backed Python
      callable that takes the input arrays positionally and returns the
      output arrays (matching the pytree shape declared via
      ``output_shape_dtype``).
"""

from __future__ import annotations

import functools
import hashlib
import threading
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.extend as jex
from jax import tree_util
from jax.interpreters import ad, batching, mlir, xla

_HAS_JAX_TVM_FFI = False
try:
    import jax_tvm_ffi

    _HAS_JAX_TVM_FFI = True
except Exception:
    jax_tvm_ffi = None

CAN_USE_TILELANG_PRIMITIVE = False
try:
    import tilelang

    CAN_USE_TILELANG_PRIMITIVE = True
except Exception:
    tilelang = None


@dataclass(frozen=True)
class _CompiledTilelangKernel:
    """Container for a compiled tile-lang adapter and its FFI target name.

    Attributes:
        target_name: Unique FFI target name registered with JAX via TVM-FFI.
        kernel: The :class:`tilelang.JITKernel` produced by ``tilelang.compile``.
        adapter_func: The TVM-FFI callable extracted from
            ``kernel.adapter.func`` (this is what JAX will dispatch to).
    """

    target_name: str
    kernel: Any
    adapter_func: Any


_COMPILE_CACHE: dict[tuple[Any, ...], _CompiledTilelangKernel] = {}
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


def _shape_dtype_key(shaped: Any) -> tuple[tuple[int, ...], str]:
    """Build a stable compile-cache key fragment from a shaped value.

    Args:
        shaped: An object with ``shape`` and ``dtype`` attributes.

    Returns:
        A tuple of ``(shape_tuple, dtype_str)`` suitable for use as a cache
        key component.  Shape dimensions are coerced to ``int`` to avoid
        symbolic-int hashability issues.
    """
    return (tuple(int(d) for d in shaped.shape), str(shaped.dtype))


def _cache_key_hash(cache_key: tuple[Any, ...]) -> str:
    """Build a deterministic SHA-256 hex digest from a compile cache key.

    Args:
        cache_key: A tuple of values representing the compilation parameters.

    Returns:
        A SHA-256 hex-digest string derived from the cache key's ``repr``.
    """
    return hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()


def _compile_or_get_kernel(
    *,
    prim_func: Any,
    in_shaped: tuple[Any, ...],
    out_shaped: tuple[jax.ShapeDtypeStruct, ...],
    target: str | None,
    target_host: str | None,
    pass_configs: tuple[tuple[str, Any], ...] | None,
    compile_flags: tuple[str, ...] | None,
) -> _CompiledTilelangKernel:
    """Compile (or fetch cached) tile-lang prim_func and FFI target metadata.

    Looks up the compile cache by a key derived from the prim_func identity,
    input/output shapes + dtypes, target, and compile flags. On a cache miss
    invokes ``tilelang.compile(..., execution_backend='tvm_ffi')`` and
    records the resulting adapter function under a hashed target name.

    Args:
        prim_func: A ``@T.prim_func`` (or ``tilelang.engine.lower.IRModule``)
            whose parameter list represents ALL buffers (inputs + outputs).
        in_shaped: Tuple of input shape/dtype descriptors.
        out_shaped: Tuple of ``jax.ShapeDtypeStruct`` for outputs.
        target: Optional explicit target string forwarded to ``tilelang.compile``.
        target_host: Optional host target forwarded to ``tilelang.compile``.
        pass_configs: Optional sorted tuple of ``(key, value)`` pass options.
        compile_flags: Optional tuple of NVCC flags forwarded as a list.

    Returns:
        A :class:`_CompiledTilelangKernel` containing the compiled kernel
        and its unique FFI target name.

    Raises:
        ValueError: If tile-lang is not installed.
    """
    if not CAN_USE_TILELANG_PRIMITIVE:
        raise ValueError("tile-lang primitive path requires `tilelang` to be installed.")

    cache_key = (
        id(prim_func),
        tuple(_shape_dtype_key(arg) for arg in in_shaped),
        tuple(_shape_dtype_key(arg) for arg in out_shaped),
        target,
        target_host,
        pass_configs,
        compile_flags,
    )

    with _COMPILE_LOCK:
        cached = _COMPILE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        kwargs: dict[str, Any] = {
            "out_idx": None,
            "execution_backend": "tvm_ffi",
        }
        if target is not None:
            kwargs["target"] = target
        if target_host is not None:
            kwargs["target_host"] = target_host
        if pass_configs:
            kwargs["pass_configs"] = dict(pass_configs)
        if compile_flags:
            kwargs["compile_flags"] = list(compile_flags)

        compiled = tilelang.compile(prim_func, **kwargs)
        adapter_func = compiled.adapter.func

        digest = _cache_key_hash(cache_key)[:24]
        target_name = f"ejkernel_tilelang_tvm_ffi_{digest}"
        result = _CompiledTilelangKernel(
            target_name=target_name,
            kernel=compiled,
            adapter_func=adapter_func,
        )
        _COMPILE_CACHE[cache_key] = result
        return result


def _register_target_once(kernel: _CompiledTilelangKernel) -> None:
    """Register a compiled tile-lang callable as a JAX FFI target exactly once.

    Tries multiple platform strings (``gpu``, ``cuda``, ``CUDA``, and
    unspecified) to accommodate different JAX/XLA runtime configurations.

    Args:
        kernel: The :class:`_CompiledTilelangKernel` to register.

    Raises:
        ValueError: If ``jax_tvm_ffi`` is not installed.
        RuntimeError: If all registration attempts fail.
    """
    if not _HAS_JAX_TVM_FFI:
        raise ValueError("tile-lang primitive path requires `jax_tvm_ffi` to register TVM-FFI targets.")

    with _REGISTERED_TARGETS_LOCK:
        if kernel.target_name in _REGISTERED_TARGETS:
            return

        error: Exception | None = None
        register_fns = (
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.adapter_func, platform="gpu"),
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.adapter_func, platform="cuda"),
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.adapter_func, platform="CUDA"),
            lambda: jax_tvm_ffi.register_ffi_target(kernel.target_name, kernel.adapter_func),
        )
        for register_fn in register_fns:
            try:
                register_fn()
                _REGISTERED_TARGETS.add(kernel.target_name)
                return
            except Exception as exc:
                error = exc

        raise RuntimeError(f"Failed to register tile-lang TVM-FFI target `{kernel.target_name}`.") from error


def _tilelang_kernel_call_impl(
    *args_flat,
    prim_func: Any,
    out_shape_dtype_flat: tuple[jax.ShapeDtypeStruct, ...],
    input_output_aliases: tuple[tuple[int, int], ...],
    target: str | None,
    target_host: str | None,
    pass_configs: tuple[tuple[str, Any], ...] | None,
    compile_flags: tuple[str, ...] | None,
):
    """Primitive implementation shared by eager and lowering paths.

    Compiles (or retrieves) the tile-lang kernel, registers it as a JAX
    FFI target, and dispatches execution via ``jax.ffi.ffi_call``.

    Args:
        *args_flat: Flattened input arrays.
        prim_func: The tile-lang ``@T.prim_func``.
        out_shape_dtype_flat: Tuple of ``jax.ShapeDtypeStruct`` for outputs.
        input_output_aliases: Tuple of ``(input_idx, output_idx)`` pairs.
        target: Optional explicit target string for ``tilelang.compile``.
        target_host: Optional host target.
        pass_configs: Optional sorted tuple of pass options.
        compile_flags: Optional tuple of NVCC flags.

    Returns:
        The output arrays produced by the FFI call.

    Raises:
        ValueError: If tile-lang is not available.
    """
    if not CAN_USE_TILELANG_PRIMITIVE:
        raise ValueError("tile-lang primitive path requires `tilelang` to be installed.")

    kernel = _compile_or_get_kernel(
        prim_func=prim_func,
        in_shaped=tuple(args_flat),
        out_shaped=out_shape_dtype_flat,
        target=target,
        target_host=target_host,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
    )
    _register_target_once(kernel)

    alias_map = dict(input_output_aliases)
    ffi_call = jax.ffi.ffi_call(
        kernel.target_name,
        result_shape_dtypes=out_shape_dtype_flat,
        input_output_aliases=alias_map,
    )
    return ffi_call(*args_flat)


tilelang_kernel_call_p = jex.core.Primitive("ejkernel_tilelang_kernel_call")
tilelang_kernel_call_p.multiple_results = True
tilelang_kernel_call_p.def_impl(functools.partial(xla.apply_primitive, tilelang_kernel_call_p))


@tilelang_kernel_call_p.def_abstract_eval
def _tilelang_kernel_call_abstract_eval(*_, out_shape_dtype_flat, **__):
    """Primitive abstract evaluation returning output avals from the contract.

    Args:
        *_: Unused positional arguments (input avals passed by JAX tracing).
        out_shape_dtype_flat: Tuple of ``jax.ShapeDtypeStruct`` defining
            the expected output shapes and dtypes.
        **__: Unused keyword arguments.

    Returns:
        List of ``jax.core.ShapedArray`` abstract values for each output.
    """
    return [jax.core.ShapedArray(x.shape, x.dtype) for x in out_shape_dtype_flat]


def _raise_on_jvp(*args, **kwargs):
    """Raise for unsupported automatic differentiation through the primitive.

    Registered as both the JVP rule and the transpose rule for
    ``tilelang_kernel_call_p``.  Users must provide a ``jax.custom_jvp`` or
    ``jax.custom_vjp`` wrapper if gradients are needed.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always.
    """
    del args, kwargs
    raise NotImplementedError(
        "tile-lang TVM-FFI primitive does not support automatic differentiation. "
        "Use `jax.custom_jvp` or `jax.custom_vjp` to wire fwd/bwd kernels."
    )


def _raise_on_vmap(*args, **kwargs):
    """Raise for unsupported batching through the primitive.

    Registered as the batching rule for ``tilelang_kernel_call_p``.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always, directing the user to ``custom_vmap``.
    """
    del args, kwargs
    raise NotImplementedError(
        "tile-lang TVM-FFI primitive does not support batching via `jax.vmap`. Use `jax.custom_batching.custom_vmap`."
    )


ad.primitive_jvps[tilelang_kernel_call_p] = _raise_on_jvp
ad.primitive_transposes[tilelang_kernel_call_p] = _raise_on_jvp
batching.primitive_batchers[tilelang_kernel_call_p] = _raise_on_vmap

mlir.register_lowering(
    tilelang_kernel_call_p,
    mlir.lower_fun(_tilelang_kernel_call_impl, multiple_results=True),
    platform="cuda",
)


def build_tilelang_ffi_call(
    prim_func: Any,
    *,
    output_shape_dtype: Any,
    input_output_aliases: dict[int, int] | None = None,
    target: str | None = None,
    target_host: str | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
):
    """Create a callable that dispatches a tile-lang kernel through a JAX primitive.

    The wrapped callable accepts the runtime input arrays positionally and
    returns the output arrays whose shapes / dtypes match ``output_shape_dtype``.
    The underlying ``@T.prim_func`` must declare ALL buffers (inputs followed
    by outputs) since ``out_idx=None`` is forced internally — JAX preallocates
    the output buffers and passes them by reference.

    Args:
        prim_func: A ``@T.prim_func`` (or ``IRModule``) whose parameter list
            represents inputs followed by outputs.
        output_shape_dtype: A pytree of objects (e.g. ``jax.ShapeDtypeStruct``,
            ``jax.Array``, ``jax.eval_shape`` outputs) describing the output
            tensors that the kernel writes.
        input_output_aliases: Optional alias map from flattened input index to
            flattened output index (used for in-place ops / state updates).
        target: Optional explicit target string forwarded to ``tilelang.compile``
            (e.g. ``'cuda'``). Falls back to tile-lang's auto-detection.
        target_host: Optional host target forwarded to ``tilelang.compile``.
        pass_configs: Optional dict of pass-pipeline options forwarded to
            ``tilelang.compile``.
        compile_flags: Optional list of NVCC compile flags. A common one on
            current CUDA wheels is ``['-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK']``.

    Returns:
        A callable ``call(*inputs) -> outputs`` that traces through a JAX
        primitive and lowers to ``jax.ffi.ffi_call`` against a TVM-FFI target
        produced by tile-lang.
    """
    out_shape = _to_shape_dtype_struct(output_shape_dtype)
    flat_out_shape, out_tree = tree_util.tree_flatten(out_shape)
    alias_items = tuple(sorted((input_output_aliases or {}).items()))
    pass_items: tuple[tuple[str, Any], ...] | None = tuple(sorted(pass_configs.items())) if pass_configs else None
    flag_items: tuple[str, ...] | None = tuple(compile_flags) if compile_flags else None

    @partial(jax.jit, inline=True)
    def _call(*args):
        """Dispatch the tile-lang kernel through the JAX primitive.

        Args:
            *args: Runtime input arrays (pytree-flattened internally).

        Returns:
            Output pytree matching the ``output_shape_dtype`` structure.
        """
        args_flat, _ = tree_util.tree_flatten(args)
        out_flat = tilelang_kernel_call_p.bind(
            *args_flat,
            prim_func=prim_func,
            out_shape_dtype_flat=tuple(flat_out_shape),
            input_output_aliases=alias_items,
            target=target,
            target_host=target_host,
            pass_configs=pass_items,
            compile_flags=flag_items,
        )
        return tree_util.tree_unflatten(out_tree, out_flat)

    return _call


def has_tilelang_ffi_support() -> bool:
    """Return whether the tile-lang TVM-FFI primitive path can be used.

    Returns:
        ``True`` if both ``tilelang`` and ``jax_tvm_ffi`` are importable,
        ``False`` otherwise.
    """
    return CAN_USE_TILELANG_PRIMITIVE and _HAS_JAX_TVM_FFI


_AUTOTUNE_CACHE: dict[Any, Any] = {}
_AUTOTUNE_LOCK = threading.Lock()


def autotune_tilelang_ffi(
    prim_func_builder: Any,
    configs: list[dict[str, Any]],
    *,
    example_inputs: list,
    output_shape_dtype: Any,
    cache_key: Any,
    input_output_aliases: dict[int, int] | None = None,
    target: str | None = None,
    target_host: str | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
    warmup: int = 3,
    iters: int = 10,
    rep: int | None = None,
    timeout: float | None = None,
):
    """Compile every candidate config, time it on-device, return the fastest.

    This is a lightweight on-device autotuner. For each entry in ``configs``
    we call ``prim_func_builder(**config)`` to get a ``@T.prim_func``, build
    the FFI call, run it ``warmup + iters`` times on the supplied example
    inputs, and keep the config with the lowest median latency. The winning
    callable is cached under ``cache_key`` so the sweep only runs once per
    distinct problem shape.

    Args:
        prim_func_builder: ``callable(**config) -> @T.prim_func``.
        configs: list of kwarg dicts to sweep.
        example_inputs: concrete JAX arrays (correct shapes / dtypes) used to
            time the candidates.
        output_shape_dtype: output contract forwarded to ``build_tilelang_ffi_call``.
        cache_key: hashable key identifying this problem shape.
        input_output_aliases: alias map forwarded to ``build_tilelang_ffi_call``.
        target: Optional explicit target string forwarded to ``tilelang.compile``.
        target_host: Optional host target forwarded to ``tilelang.compile``.
        pass_configs: Optional pass-pipeline options forwarded to ``tilelang.compile``.
        compile_flags: NVCC flags forwarded to ``build_tilelang_ffi_call``.
        warmup: Number of warmup launches per candidate.
        iters: Timing launches per candidate when ``rep`` is not provided.
        rep: TileLang-style alias for timing launches per candidate.
        timeout: Optional wall-clock budget in seconds for the sweep. The
            tuner stops before starting another candidate once the budget has
            elapsed; in-flight compiles are not interrupted.

    Returns:
        The fastest config's primitive-backed callable.
    """
    import statistics
    import time

    with _AUTOTUNE_LOCK:
        cached = _AUTOTUNE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    best_call = None
    best_time = float("inf")
    errors: list[str] = []
    start_time = time.perf_counter()
    warmup = max(0, int(warmup))
    timing_reps = max(1, int(iters if rep is None else rep))

    for config in configs:
        if timeout is not None and time.perf_counter() - start_time >= timeout:
            errors.append(f"autotune timeout reached after {timeout:.3f}s")
            break
        try:
            prim = prim_func_builder(**config)
            call = build_tilelang_ffi_call(
                prim,
                output_shape_dtype=output_shape_dtype,
                input_output_aliases=input_output_aliases,
                target=target,
                target_host=target_host,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
            jit_call = jax.jit(call)
            for _ in range(warmup):
                out = jit_call(*example_inputs)
                jax.block_until_ready(out)
            samples = []
            for _ in range(timing_reps):
                t0 = time.perf_counter()
                out = jit_call(*example_inputs)
                jax.block_until_ready(out)
                samples.append(time.perf_counter() - t0)
            med = statistics.median(samples)
            if med < best_time:
                best_time = med
                best_call = call
        except Exception as exc:
            errors.append(f"{config}: {type(exc).__name__}: {str(exc)[:80]}")
            continue

    if best_call is None:
        raise RuntimeError("tile-lang autotune found no working config. Errors:\n  " + "\n  ".join(errors))

    with _AUTOTUNE_LOCK:
        _AUTOTUNE_CACHE[cache_key] = best_call
    return best_call


__all__ = [
    "autotune_tilelang_ffi",
    "build_tilelang_ffi_call",
    "has_tilelang_ffi_support",
]
