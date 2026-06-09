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

"""tile-lang kernel integration helpers for JAX.

The :func:`tilelang_call` wrapper mirrors :func:`ejkernel.callib._cute_call.cute_call`
and :func:`ejkernel.callib._triton_call.triton_call`. It enforces single-device
placement of array arguments, requires :func:`jax.shard_map` under multi-device
setups, validates output contracts, and either executes a prebuilt primitive
callable or builds one from a tile-lang prim-func / prim-func factory.
"""

from __future__ import annotations

import os
import shlex
import threading
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util
from jax._src import core

from ._tilelang_ffi import autotune_tilelang_ffi, build_tilelang_ffi_call
from ._utils import ShapeDtype

CAN_USE_TILELANG = False
try:
    import tilelang  # noqa: F401

    CAN_USE_TILELANG = True
except ModuleNotFoundError:
    pass

_NAMED_CALL_CACHE: dict[tuple[Any, ...], Callable[..., Any]] = {}
_NAMED_CALL_CACHE_LOCK = threading.Lock()


def _env_int(name: str, value: int | None, default: int, minimum: int) -> int:
    """Resolve an integer option from an explicit value or environment variable.

    Args:
        name: Environment variable name.
        value: Explicit value supplied by the caller.
        default: Default used when neither source is supplied.
        minimum: Smallest accepted value.

    Returns:
        Integer option clamped to ``minimum``.

    Raises:
        ValueError: If the environment variable cannot be parsed as an integer.
    """
    if value is not None:
        return max(minimum, int(value))
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


def _env_optional_float(name: str, value: float | None, default: float | None) -> float | None:
    """Resolve an optional float option from an explicit value or environment variable.

    Args:
        name: Environment variable name.
        value: Explicit value supplied by the caller.
        default: Default used when neither source is supplied.

    Returns:
        Float option, or ``None`` when disabled.

    Raises:
        ValueError: If the environment variable cannot be parsed as a float.
    """
    if value is not None:
        return max(0.0, float(value))
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.lower() in ("", "none", "off", "false"):
        return None
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}.") from exc


def _env_target(value: str | None) -> str | None:
    """Resolve the TileLang compile target from an explicit value or the environment.

    Args:
        value: Explicit target supplied by the caller.

    Returns:
        Target string, or ``None`` for TileLang auto-detection.
    """
    if value is not None:
        return value
    raw = os.environ.get("EJKERNEL_TILELANG_TARGET")
    if raw is None or raw.lower() in ("", "auto", "none", "off"):
        return None
    return raw


def _env_compile_flags(value: list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Resolve TileLang compile flags from an explicit value or the environment.

    Args:
        value: Explicit compile flags supplied by the caller.

    Returns:
        Tuple of compile flags, or ``None`` when no flags should be forwarded.
    """
    raw = os.environ.get("EJKERNEL_TILELANG_COMPILE_FLAGS")
    if raw is not None:
        if raw.lower() in ("", "none", "off", "false"):
            return None
        return tuple(shlex.split(raw))
    return tuple(value) if value else None


def _device_set_from_sharding(sharding: Any) -> set | None:
    """Extract the participating devices from a sharding object.

    Attempts to retrieve device information by checking ``device_set`` and
    ``devices`` attributes on the sharding, calling them if they are callable.
    Attribute access and call errors are silently swallowed.

    Args:
        sharding: A JAX sharding object that may expose ``device_set`` or
            ``devices`` as an attribute or method.

    Returns:
        A set of devices referenced by the sharding, or ``None`` if the
        device information could not be determined.
    """
    for attr_name in ("device_set", "devices"):
        try:
            attr = getattr(sharding, attr_name, None)
        except Exception:
            continue
        if attr is None:
            continue
        try:
            devices = attr() if callable(attr) else attr
        except Exception:
            continue
        if devices is not None:
            return set(devices)
    return None


def _array_device_set(arg: Any) -> set | None:
    """Return the device set for a JAX array or tracer.

    Inspects ``arg`` for device placement by checking ``device``, ``devices``,
    and ``sharding`` attributes on concrete arrays, and ``aval.sharding`` on
    JAX tracers.

    Args:
        arg: A potential JAX array or tracer whose device placement is needed.

    Returns:
        A set containing the device(s) that ``arg`` resides on, or ``None``
        if device information cannot be determined (e.g. for non-array values).
    """
    if isinstance(arg, jax.Array):
        device_attr = getattr(arg, "device", None)
        if device_attr is not None:
            try:
                device = device_attr() if callable(device_attr) else device_attr
            except Exception:
                device = None
            if device is not None:
                return {device}

        devices_attr = getattr(arg, "devices", None)
        if devices_attr is not None:
            try:
                devices = devices_attr() if callable(devices_attr) else devices_attr
            except Exception:
                devices = None
            if devices is not None:
                return set(devices)

        sharding = getattr(arg, "sharding", None)
        if sharding is not None:
            device_set = _device_set_from_sharding(sharding)
            if device_set is not None:
                return device_set

    if isinstance(arg, core.Tracer):
        aval = getattr(arg, "aval", None)
        sharding = getattr(aval, "sharding", None)
        if sharding is not None:
            device_set = _device_set_from_sharding(sharding)
            if device_set is not None:
                return device_set

    return None


def _assert_single_device_args(
    array_args: Sequence[Any],
    device_index: int | None,
    *,
    allow_sharded_tracers: bool,
) -> None:
    """Validate that all array arguments live on a single logical device.

    Ensures all provided array arguments reside on the same single device.
    Optionally allows sharded tracers when running inside ``jax.shard_map``.

    Args:
        array_args: Sequence of array or tracer arguments to validate.
        device_index: Optional requested device index to verify against.
        allow_sharded_tracers: If ``True``, sharded tracers (from
            ``jax.shard_map``) are permitted without raising an error.

    Raises:
        AssertionError: If any argument spans multiple devices (unless it is
            a tracer and ``allow_sharded_tracers`` is ``True``), if arguments
            reside on different devices, or if the detected device conflicts
            with the requested ``device_index``.
    """
    device_sets: list[tuple[int, set, bool]] = []
    for idx, arg in enumerate(array_args):
        devs = _array_device_set(arg)
        if devs is not None:
            device_sets.append((idx, devs, isinstance(arg, core.Tracer)))

    if not device_sets:
        return

    for idx, devs, is_tracer in device_sets:
        if len(devs) != 1:
            if allow_sharded_tracers and is_tracer:
                continue
            raise AssertionError(
                "tilelang_call requires all array arguments to be on a single device. "
                f"Argument {idx} is sharded across {len(devs)} devices. "
                "Use `jax.shard_map` for multi-device execution."
            )

    single_device_sets = [(idx, devs) for idx, devs, _ in device_sets if len(devs) == 1]
    if not single_device_sets:
        return

    first_device = next(iter(single_device_sets[0][1]))
    for idx, devs in single_device_sets[1:]:
        if next(iter(devs)) != first_device:
            raise AssertionError(
                "tilelang_call requires all array arguments to be on the same device. "
                f"Argument {idx} is on a different device than argument {single_device_sets[0][0]}."
            )

    if device_index is None:
        return

    try:
        platform = getattr(first_device, "platform", None)
        devices = jax.devices(platform) if platform else jax.devices()
        if 0 <= device_index < len(devices) and devices[device_index] != first_device:
            raise AssertionError(
                "tilelang_call received inputs on a different device than the requested "
                f"`device={device_index}`. Place inputs on the target device or adjust "
                "the `device` argument."
            )
    except Exception:
        return


def _has_multi_accelerators() -> bool:
    """Check whether more than one non-CPU accelerator is available.

    Returns:
        ``True`` if more than one non-CPU device is visible to JAX,
        ``False`` otherwise (including when device enumeration fails).
    """
    try:
        devices = jax.devices()
    except Exception:
        return False
    accelerator_devices = [device for device in devices if getattr(device, "platform", None) not in (None, "cpu")]
    return len(accelerator_devices) > 1


def _in_shard_map_context() -> bool:
    """Detect whether execution is currently inside a ``jax.shard_map`` context.

    Probes JAX internal thread-local state for an active mesh environment
    or axis environment, which indicates that ``jax.shard_map`` is in effect.

    Returns:
        ``True`` if a shard-map mesh context is detected, ``False`` otherwise.
    """
    try:
        from jax._src import mesh as mesh_lib

        env = getattr(mesh_lib, "thread_resources", None)
        env = getattr(env, "env", None)
        physical_mesh = getattr(env, "physical_mesh", None)
        axis_names = getattr(physical_mesh, "axis_names", None)
        if axis_names:
            return True
    except Exception:
        pass

    try:
        axis_env = core.thread_local_state.trace_state.axis_env
        axis_names = getattr(axis_env, "names", None)
        if axis_names:
            return True
    except Exception:
        pass

    return False


def _leaf_shape_dtype(leaf: Any) -> tuple[tuple[int, ...] | None, jnp.dtype | None]:
    """Read ``(shape, dtype)`` from an output leaf or tracer.

    Inspects the leaf for ``shape`` and ``dtype`` attributes directly, then
    falls back to checking its ``aval`` attribute (for JAX tracers).

    Args:
        leaf: An array, tracer, or shaped object to inspect.

    Returns:
        A tuple of ``(shape, dtype)`` where each element is ``None`` if the
        corresponding attribute could not be determined.
    """
    shape = getattr(leaf, "shape", None)
    dtype = getattr(leaf, "dtype", None)
    if shape is not None and dtype is not None:
        return tuple(shape), jnp.dtype(dtype)

    aval = getattr(leaf, "aval", None)
    shape = getattr(aval, "shape", None)
    dtype = getattr(aval, "dtype", None)
    if shape is not None and dtype is not None:
        return tuple(shape), jnp.dtype(dtype)

    return None, None


def _validate_out_leaves(
    flat_out: Sequence[Any],
    flat_out_shapes: Sequence[jax.ShapeDtypeStruct] | None,
) -> None:
    """Validate explicit output leaves and optional output shape contracts.

    Checks that all output leaves are JAX arrays or tracers with inferrable
    shape and dtype. When ``flat_out_shapes`` is provided, additionally
    verifies that leaves match the expected shapes and dtypes.

    Args:
        flat_out: Flattened sequence of output leaves (arrays or tracers).
        flat_out_shapes: Optional sequence of ``jax.ShapeDtypeStruct``
            specifying the expected shape and dtype for each output leaf.

    Raises:
        ValueError: If ``flat_out`` is empty, or if the number of output
            leaves does not match ``flat_out_shapes``, or if a shape/dtype
            mismatch is detected.
        TypeError: If any leaf is not a JAX array or tracer, or if
            shape/dtype cannot be inferred from a leaf.
    """
    if not flat_out:
        raise ValueError("`out` must contain at least one output array.")

    for i, leaf in enumerate(flat_out):
        if not isinstance(leaf, (jax.Array, core.Tracer)):
            raise TypeError(f"`out` leaves must be JAX arrays/tracers. Got type {type(leaf)!r} at output index {i}.")
        shape, dtype = _leaf_shape_dtype(leaf)
        if shape is None or dtype is None:
            raise TypeError(f"Could not infer shape/dtype from an `out` leaf. Output index: {i}.")

    if flat_out_shapes is None:
        return

    if len(flat_out) != len(flat_out_shapes):
        raise ValueError(
            "Mismatch between number of output leaves and `out_shape` leaves: "
            f"{len(flat_out)} vs {len(flat_out_shapes)}."
        )

    for i, (leaf, spec) in enumerate(zip(flat_out, flat_out_shapes, strict=False)):
        leaf_shape, leaf_dtype = _leaf_shape_dtype(leaf)
        if leaf_shape != tuple(spec.shape):
            raise ValueError(
                f"Output shape mismatch at index {i}: out has shape {leaf_shape}, expected {tuple(spec.shape)}."
            )
        if leaf_dtype != jnp.dtype(spec.dtype):
            raise ValueError(
                f"Output dtype mismatch at index {i}: out has dtype {leaf_dtype}, expected {jnp.dtype(spec.dtype)}."
            )


def _shape_specs_from_out_leaves(flat_out: Sequence[Any]) -> list[jax.ShapeDtypeStruct]:
    """Build shape/dtype structs from explicit ``out`` leaves.

    Args:
        flat_out: Flattened sequence of output leaves (arrays or tracers).

    Returns:
        List of ``jax.ShapeDtypeStruct`` instances inferred from each leaf.

    Raises:
        TypeError: If shape or dtype cannot be inferred from any leaf.
    """
    specs: list[jax.ShapeDtypeStruct] = []
    for i, leaf in enumerate(flat_out):
        shape, dtype = _leaf_shape_dtype(leaf)
        if shape is None or dtype is None:
            raise TypeError(f"Could not infer shape/dtype from an `out` leaf. Output index: {i}.")
        specs.append(jax.ShapeDtypeStruct(shape, dtype))
    return specs


def _shape_key(shape: Any) -> tuple[str, ...]:
    """Normalize shape values to a stable, hashable key.

    Args:
        shape: An iterable of dimension sizes (integers or symbolic values).

    Returns:
        A tuple of string representations suitable for use as a hash key.
    """
    return tuple(str(d) for d in tuple(shape))


def _arg_contract_key(arg: Any) -> tuple[Any, ...]:
    """Build a cache-key fragment for an argument.

    Produces a hashable tuple that uniquely identifies the argument's type
    contract (shape/dtype for arrays, value for scalars, type info for others).

    Args:
        arg: An input argument (array, scalar, dtype, or arbitrary object).

    Returns:
        A hashable tuple encoding the argument's contract for cache lookups.
    """
    if isinstance(arg, (jax.Array, core.Tracer)):
        shape, dtype = _leaf_shape_dtype(arg)
        return ("array", _shape_key(shape or ()), str(dtype))
    if isinstance(arg, (bool, int, float, str, bytes)):
        return ("scalar", type(arg).__name__, arg)
    if isinstance(arg, jnp.dtype):
        return ("dtype", str(arg))
    return ("object", type(arg).__module__, type(arg).__qualname__)


def _out_contract_key(output_contract_shapes: Sequence[jax.ShapeDtypeStruct] | None) -> tuple[Any, ...]:
    """Build a stable key for expected output contracts.

    Args:
        output_contract_shapes: Optional sequence of ``jax.ShapeDtypeStruct``
            describing the expected output shapes and dtypes.

    Returns:
        A hashable tuple encoding the output contract, or an empty tuple
        if ``output_contract_shapes`` is ``None``.
    """
    if output_contract_shapes is None:
        return ()
    return tuple((_shape_key(spec.shape), str(jnp.dtype(spec.dtype))) for spec in output_contract_shapes)


def _shape_dtype_tree(value: Any) -> Any:
    """Convert an output contract tree to ``jax.ShapeDtypeStruct`` leaves.

    Args:
        value: Pytree whose leaves expose ``shape`` and ``dtype``.

    Returns:
        Pytree with the same structure and ``jax.ShapeDtypeStruct`` leaves.
    """
    return tree_util.tree_map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), value)


def _freeze_key(value: Any) -> Any:
    """Convert arbitrary static metadata into a stable hashable cache fragment.

    Args:
        value: Static metadata, config values, callables, dtypes, or pytrees.

    Returns:
        A hashable representation suitable for process-local call caches.
    """
    if isinstance(value, dict):
        return (
            "dict",
            tuple((str(key), _freeze_key(item)) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_freeze_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted((_freeze_key(item) for item in value), key=repr)))
    if isinstance(value, jnp.dtype):
        return ("dtype", str(value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if callable(value):
        return (
            "callable",
            getattr(value, "__module__", type(value).__module__),
            getattr(value, "__qualname__", type(value).__qualname__),
            id(value),
        )
    try:
        hash(value)
    except TypeError:
        return ("object", type(value).__module__, type(value).__qualname__, repr(value))
    return value


def build_tilelang_call(
    prim_func: Any | None = None,
    *,
    kernel: Callable[..., Any] | None = None,
    out_shape: ShapeDtype | Sequence[ShapeDtype] | None = None,
    output_shape_dtype: ShapeDtype | Sequence[ShapeDtype] | None = None,
    args: Sequence[Any] | None = None,
    meta: dict[str, Any] | None = None,
    configs: Sequence[dict[str, Any]] | None = None,
    name: str | None = None,
    input_output_aliases: dict[int, int] | None = None,
    target: str | None = None,
    target_host: str | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
    warmup: int | None = None,
    rep: int | None = None,
    timeout: float | None = None,
    autotune: bool | None = None,
    cache_key: Any | None = None,
) -> Callable[..., Any]:
    """Build, cache, and optionally autotune a primitive-backed TileLang call.

    This is the TileLang analogue of the project Triton caller's kernel
    construction layer: users provide a prim-func factory plus static metadata,
    and this helper owns target selection, compile flags, pass configs, output
    contracts, optional autotune sweeps, and process-local call caching. For
    existing call sites that already create a prim-func, ``prim_func`` can be
    passed directly while still routing compilation through the shared caller.

    Args:
        prim_func: Optional already-created ``@T.prim_func`` or IRModule.
        kernel: Optional callable that returns a ``@T.prim_func`` when invoked
            with ``meta`` and one candidate config.
        out_shape: Output shape/dtype contract returned by the compiled call.
        output_shape_dtype: Backward-compatible alias for ``out_shape``.
        args: Optional example inputs. Required when autotuning is enabled.
        meta: Static keyword arguments passed to ``kernel`` for every build.
        configs: Optional autotune candidate dictionaries. Each dictionary is
            merged after ``meta`` and passed to ``kernel``.
        name: Optional human-readable call name included in the cache key.
        input_output_aliases: Optional flattened input-to-output alias map.
        target: Optional explicit TileLang target, e.g. ``"cuda"``. Defaults
            to ``EJKERNEL_TILELANG_TARGET`` or TileLang auto-detection.
        target_host: Optional TileLang host target.
        pass_configs: Optional TileLang pass pipeline configuration.
        compile_flags: Optional NVCC flags forwarded to TileLang. Defaults to
            ``EJKERNEL_TILELANG_COMPILE_FLAGS`` when that environment variable
            is set.
        warmup: Warmup launches per autotune candidate. Defaults to
            ``EJKERNEL_TILELANG_AUTOTUNE_WARMUP`` or ``25``.
        rep: Timed launches per autotune candidate. Defaults to
            ``EJKERNEL_TILELANG_AUTOTUNE_REP`` or ``100``.
        timeout: Optional autotune wall-clock budget in seconds. Defaults to
            ``EJKERNEL_TILELANG_AUTOTUNE_TIMEOUT`` or ``60``; set the
            environment variable to ``none`` / ``off`` to disable.
        autotune: Force autotune on/off. Defaults to on when ``configs`` are
            supplied and off otherwise.
        cache_key: Optional explicit key for the compiled/autotuned callable.

    Returns:
        Primitive-backed callable that accepts runtime inputs and returns the
        output contract described by ``out_shape``.

    Raises:
        ValueError: If TileLang is unavailable, autotune lacks example inputs,
            both ``prim_func`` and ``kernel`` are provided, or autotune is
            requested while tracing.
    """
    if not CAN_USE_TILELANG:
        raise ValueError("`build_tilelang_call` is only available when `tilelang` is installed.")
    if kernel is not None and prim_func is not None:
        raise ValueError("Provide only one of `prim_func` or `kernel` to `build_tilelang_call`.")
    if kernel is None and prim_func is None:
        raise ValueError("Provide either `prim_func` or `kernel` to `build_tilelang_call`.")
    if out_shape is not None and output_shape_dtype is not None:
        raise ValueError("Provide only one of `out_shape` or `output_shape_dtype` to `build_tilelang_call`.")
    output_contract = out_shape if out_shape is not None else output_shape_dtype
    if output_contract is None:
        raise ValueError("`build_tilelang_call` requires `out_shape` or `output_shape_dtype`.")

    meta_dict = dict(meta or {})
    config_list = tuple(dict(config) for config in configs or ())
    do_autotune = bool(config_list) if autotune is None else bool(autotune)
    if prim_func is not None and do_autotune:
        raise ValueError("TileLang autotune requires a `kernel` prim-func factory, not a direct `prim_func`.")
    if do_autotune and not config_list:
        raise ValueError("`build_tilelang_call` received `autotune=True` without candidate `configs`.")

    resolved_warmup = _env_int("EJKERNEL_TILELANG_AUTOTUNE_WARMUP", warmup, 25, 0)
    resolved_rep = _env_int("EJKERNEL_TILELANG_AUTOTUNE_REP", rep, 100, 1)
    resolved_timeout = _env_optional_float("EJKERNEL_TILELANG_AUTOTUNE_TIMEOUT", timeout, 60)
    resolved_target = _env_target(target)
    resolved_compile_flags = _env_compile_flags(compile_flags)

    output_shape = _shape_dtype_tree(output_contract)
    flat_out_shape, _ = tree_util.tree_flatten(output_shape)
    example_inputs: list[Any] | None = None
    if do_autotune:
        if args is None:
            raise ValueError("`build_tilelang_call` requires concrete `args` when autotuning.")
        flat_args, _ = tree_util.tree_flatten(tuple(args))
        if any(isinstance(arg, core.Tracer) for arg in flat_args):
            raise ValueError("TileLang autotune requires concrete example inputs; disable autotune while tracing.")
        example_inputs = list(flat_args)

    call_key = (
        "build_tilelang_call",
        (
            _freeze_key(cache_key)
            if cache_key is not None
            else (
                str(name) if name is not None else "",
                _freeze_key(prim_func) if prim_func is not None else None,
                _freeze_key(kernel),
                _freeze_key(meta_dict),
                _freeze_key(config_list),
                _out_contract_key(flat_out_shape),
                _freeze_key(input_output_aliases or {}),
                resolved_target,
                target_host,
                _freeze_key(pass_configs or {}),
                resolved_compile_flags,
                bool(do_autotune),
                resolved_warmup,
                resolved_rep,
                resolved_timeout,
            )
        ),
    )
    with _NAMED_CALL_CACHE_LOCK:
        cached_call = _NAMED_CALL_CACHE.get(call_key)
        if cached_call is not None:
            return cached_call

    def _builder(**config: Any) -> Any:
        """Build a prim-func from static metadata plus one candidate config."""
        assert kernel is not None
        params = dict(meta_dict)
        params.update(config)
        return kernel(**params)

    if do_autotune:
        assert example_inputs is not None
        built_call = autotune_tilelang_ffi(
            _builder,
            list(config_list),
            example_inputs=example_inputs,
            output_shape_dtype=output_shape,
            cache_key=call_key,
            input_output_aliases=input_output_aliases,
            target=resolved_target,
            target_host=target_host,
            pass_configs=pass_configs,
            compile_flags=resolved_compile_flags,
            warmup=resolved_warmup,
            rep=resolved_rep,
            timeout=resolved_timeout,
        )
    else:
        built_call = build_tilelang_ffi_call(
            prim_func if prim_func is not None else _builder(),
            output_shape_dtype=output_shape,
            input_output_aliases=input_output_aliases,
            target=resolved_target,
            target_host=target_host,
            pass_configs=pass_configs,
            compile_flags=resolved_compile_flags,
        )

    with _NAMED_CALL_CACHE_LOCK:
        cached_call = _NAMED_CALL_CACHE.get(call_key)
        if cached_call is not None:
            return cached_call
        _NAMED_CALL_CACHE[call_key] = built_call
    return built_call


def tilelang_call(
    *args: Any,
    call: Callable[..., Any] | None = None,
    kernel: Callable[..., Any] | None = None,
    out_shape: ShapeDtype | Sequence[ShapeDtype] | None = None,
    out: Any | None = None,
    name: str | None = None,
    device: int | None = None,
    meta: dict[str, Any] | None = None,
    configs: Sequence[dict[str, Any]] | None = None,
    input_output_aliases: dict[int, int] | None = None,
    target: str | None = None,
    target_host: str | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | tuple[str, ...] | None = None,
    warmup: int | None = None,
    rep: int | None = None,
    timeout: float | None = None,
    autotune: bool | None = None,
    cache_key: Any | None = None,
) -> Any:
    """Execute a tile-lang kernel and return its output(s).

    The caller accepts either an already-built primitive ``call`` or a
    ``kernel`` prim-func factory plus static metadata/configs. ``out_shape``
    and/or ``out`` define the expected output contract used both for
    validation and pytree restructure.

    Args:
        *args: Positional arguments forwarded to the tile-lang kernel callable.
        call: Optional prebuilt tile-lang primitive callable to execute.
        kernel: Optional prim-func factory used to build a primitive callable
            when ``call`` is omitted.
        out_shape: Expected output shape/dtype specification(s).
        out: Optional explicit output array(s) whose shape/dtype/tree are used
            as the output contract. Not used as a destination buffer.
        name: Optional name for the kernel call, used for JAX named scopes
            and internal caching.
        device: Optional device index to validate input placement against.
        meta: Static keyword arguments passed to ``kernel``.
        configs: Optional autotune candidate dictionaries.
        input_output_aliases: Optional flattened input/output alias map.
        target: Optional explicit TileLang target. Defaults to
            ``EJKERNEL_TILELANG_TARGET`` or TileLang auto-detection.
        target_host: Optional TileLang host target.
        pass_configs: Optional TileLang pass-pipeline configuration.
        compile_flags: Optional TileLang/NVCC compile flags. Defaults to
            ``EJKERNEL_TILELANG_COMPILE_FLAGS`` when that environment variable
            is set.
        warmup: Warmup launches per autotune candidate. Defaults to
            ``EJKERNEL_TILELANG_AUTOTUNE_WARMUP`` or ``25``.
        rep: Timed launches per autotune candidate. Defaults to
            ``EJKERNEL_TILELANG_AUTOTUNE_REP`` or ``100``.
        timeout: Optional autotune wall-clock budget in seconds. Defaults to
            ``EJKERNEL_TILELANG_AUTOTUNE_TIMEOUT`` or ``60``.
        autotune: Force autotune on/off. Defaults to on when ``configs`` exist.
        cache_key: Optional explicit compile/autotune cache key.

    Returns:
        The output(s) produced by the tile-lang kernel, unflattened to match
        the ``out`` or ``out_shape`` pytree structure.

    Raises:
        ValueError: If tile-lang is not installed, ``call`` is ``None``,
            neither ``out`` nor ``out_shape`` is provided, or the callable
            returns ``None``.
        AssertionError: If array arguments span multiple devices, or if
            multiple accelerators are detected without an active
            ``jax.shard_map`` context.
    """
    if not CAN_USE_TILELANG:
        raise ValueError("`tilelang_call` is only available when `tilelang` is installed.")
    if call is not None and kernel is not None:
        raise ValueError("Provide only one of `call` or `kernel` to `tilelang_call`.")
    built_call_from_kernel = call is None

    flat_args, _ = tree_util.tree_flatten(args)
    flat_out: list[Any] = []
    out_tree = None
    flat_out_shapes: Sequence[jax.ShapeDtypeStruct] | None = None

    if out_shape is not None:
        out_shape = tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)
        flat_out_shapes, out_tree_from_shape = tree_util.tree_flatten(out_shape)
    else:
        out_tree_from_shape = None

    if out is not None:
        flat_out, out_tree = tree_util.tree_flatten(out)
        _validate_out_leaves(flat_out, flat_out_shapes)
    else:
        if flat_out_shapes is None:
            raise ValueError("Provide either `out` or `out_shape` to `tilelang_call`.")
        out_tree = out_tree_from_shape

    output_contract_shapes = list(flat_out_shapes) if flat_out_shapes is not None else None
    if output_contract_shapes is None and flat_out:
        output_contract_shapes = _shape_specs_from_out_leaves(flat_out)
    if call is None:
        if kernel is None:
            raise ValueError(
                "Provide either `call` or `kernel` to `tilelang_call`. "
                "Use `kernel` plus `meta`/`configs` for Triton-style TileLang construction."
            )
        assert out_tree is not None
        assert output_contract_shapes is not None
        output_contract = tree_util.tree_unflatten(out_tree, output_contract_shapes)
        call = build_tilelang_call(
            kernel=kernel,
            out_shape=output_contract,
            args=args,
            meta=meta,
            configs=configs,
            name=name,
            input_output_aliases=input_output_aliases,
            target=target,
            target_host=target_host,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            warmup=warmup,
            rep=rep,
            timeout=timeout,
            autotune=autotune,
            cache_key=cache_key,
        )

    array_args = [arg for arg in flat_args if isinstance(arg, (jax.Array, core.Tracer))]

    def _coerce_function_output(function_out: Any) -> Any:
        """Validate and restructure the callable's raw output.

        Ensures the callable returned non-None arrays, validates them against
        the output contract, and reshapes the result to match the expected
        output pytree when ``out`` was provided.

        Args:
            function_out: Raw output from the tile-lang kernel callable.

        Returns:
            The validated (and possibly restructured) output.

        Raises:
            ValueError: If ``function_out`` is ``None`` or if the number of
                output leaves does not match the ``out`` specification.
        """
        if function_out is None:
            raise ValueError(
                "`tilelang_call` expected `call` to return output arrays. "
                "Pass a primitive-backed callable instead of a runtime launch object."
            )
        flat_function_out, _ = tree_util.tree_flatten(function_out)
        _validate_out_leaves(flat_function_out, output_contract_shapes)

        if out is None:
            return function_out

        if len(flat_function_out) != len(flat_out):
            raise ValueError(
                "Mismatch between callable output leaves and provided `out` leaves: "
                f"{len(flat_function_out)} vs {len(flat_out)}."
            )

        assert out_tree is not None
        return tree_util.tree_unflatten(out_tree, flat_function_out)

    in_shard_map_context = _in_shard_map_context()
    _assert_single_device_args(array_args, device, allow_sharded_tracers=in_shard_map_context)
    if _has_multi_accelerators() and not in_shard_map_context:
        raise AssertionError(
            "Multiple accelerator devices detected. "
            "tilelang_call must be invoked under `jax.shard_map` in multi-accelerator setups."
        )

    call_to_run = call
    if name is not None and not built_call_from_kernel:
        call_key = (
            str(name),
            tuple(_arg_contract_key(arg) for arg in flat_args),
            _out_contract_key(output_contract_shapes),
            device,
        )
        with _NAMED_CALL_CACHE_LOCK:
            cached_call = _NAMED_CALL_CACHE.get(call_key)
            if cached_call is None:
                _NAMED_CALL_CACHE[call_key] = call_to_run
            else:
                call_to_run = cached_call

    scope_name = "tilelang_call" if name is None else str(name)
    with jax.named_scope(scope_name):
        return _coerce_function_output(call_to_run(*args))


__all__ = ["build_tilelang_call", "tilelang_call"]
