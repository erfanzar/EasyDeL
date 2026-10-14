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


"""Triton kernel integration with JAX for GPU computation.

This module provides the interface for calling Triton kernels from JAX code,
enabling high-performance GPU computation with Triton's programming model
while maintaining JAX's functional semantics.

Key Features:
    - Seamless Triton kernel invocation from JAX
    - CUDA and ROCm (AMD) GPU support
    - Automatic kernel compilation and caching
    - Support for Triton autotuner and heuristics
    - Input-output aliasing for memory efficiency
    - Configurable launch parameters (warps, stages, CTAs)

Key Components:
    triton_call: Main entry point for executing Triton kernels from JAX
    get_triton_type: Convert JAX/NumPy types to Triton type strings
    CompilationResult: Container for compiled kernel binary and metadata

Supported Platforms:
    - CUDA: NVIDIA GPUs via PTX compilation
    - ROCm: AMD GPUs via HSACO compilation

Type Conversions:
    The module handles automatic type conversion between JAX and Triton:
    - JAX arrays -> Triton pointers (*bf16, *fp32, *i32, etc.)
    - Python scalars -> Triton scalars (i32, fp32, etc.)
    - NumPy arrays -> Triton pointers

Example:
    >>> import triton
    >>> import triton.language as tl
    >>> from ejkernel.callib import triton_call
    >>>
    >>> @triton.jit
    ... def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...     pid = tl.program_id(0)
    ...     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ...     mask = offsets < n
    ...     x = tl.load(x_ptr + offsets, mask=mask)
    ...     y = tl.load(y_ptr + offsets, mask=mask)
    ...     tl.store(out_ptr + offsets, x + y, mask=mask)
    >>>
    >>> result = triton_call(
    ...     x, y,
    ...     kernel=add_kernel,
    ...     out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    ...     grid=lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),),
    ...     BLOCK_SIZE=1024,
    ... )

Note:
    Automatic differentiation (JVP/VJP) and vmap are not natively supported.
    Use jax.custom_vjp or jax.custom_vmap for custom gradient/batching rules.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import inspect
import os
import pickle
import shutil
import tempfile
import types
import zlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.dlpack
import jax.extend as jex
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax._src import core, state, util
from jax._src.lib import gpu_triton as triton_kernel_call_lib
from jax.interpreters import ad, batching, mlir, xla
from jaxlib import xla_client

try:
    from jaxlib.plugin_support import import_from_plugin
except ImportError:
    import_from_plugin = None

from ._utils import ShapeDtype, check_bool_flag, get_cache_dir


def _has_triton_kernel_call_runtime(module: Any) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "TritonKernel",
            "TritonKernelCall",
            "TritonAutotunedKernelCall",
            "create_array_parameter",
            "create_scalar_parameter",
            "get_custom_call",
        )
    )


def _load_triton_kernel_call_lib(module: Any) -> Any:
    if _has_triton_kernel_call_runtime(module):
        return module
    if import_from_plugin is None:
        return module

    for plugin_name, xla_platform in (("cuda", "CUDA"), ("rocm", "ROCM")):
        plugin_module = import_from_plugin(plugin_name, "_triton", check_version=False)
        if plugin_module is None or not _has_triton_kernel_call_runtime(plugin_module):
            continue
        xla_client.register_custom_call_target(
            "triton_kernel_call",
            plugin_module.get_custom_call(),
            platform=xla_platform,
        )
        return plugin_module
    return module


triton_kernel_call_lib = _load_triton_kernel_call_lib(triton_kernel_call_lib)

CAN_USE_TRITON = False
try:
    import triton
    import triton._C.libtriton as _triton
    import triton.backends.nvidia.compiler as cb
    import triton.language as tl
    from triton.compiler import code_generator as code_gen
    from triton.compiler import compiler as tc
    from triton.runtime import autotuner

    CAN_USE_TRITON = True
except ModuleNotFoundError:
    pass

try:
    import triton.backends.amd.compiler as hb
except ImportError:
    hb = None
    pass


safe_map, unsafe_map = util.safe_map, map
safe_zip, unsafe_zip = util.safe_zip, zip


_JAX_TO_TRITON_TYPE_MAP = {
    jnp.dtype("bfloat16"): "bf16",
    jnp.dtype("float64"): "fp64",
    jnp.dtype("float32"): "fp32",
    jnp.dtype("float16"): "fp16",
    jnp.dtype("float8_e4m3fn"): "fp8e4nv",
    jnp.dtype("float8_e5m2"): "fp8e5",
    jnp.dtype("float8_e4m3fnuz"): "fp8e4b8",
    jnp.dtype("float8_e5m2fnuz"): "fp8e5b16",
    jnp.dtype("int64"): "i64",
    jnp.dtype("int32"): "i32",
    jnp.dtype("int16"): "i16",
    jnp.dtype("int8"): "i8",
    jnp.dtype("uint64"): "u64",
    jnp.dtype("uint32"): "u32",
    jnp.dtype("uint16"): "u16",
    jnp.dtype("uint8"): "u8",
    jnp.dtype("bool"): "i1",
}

Grid = int | tuple[int] | tuple[int, int] | tuple[int, int, int]
GridOrLambda = Grid | Callable[[dict[str, Any]], Grid]


def _parse_int_env(name: str, default: int) -> int:
    """Parse an integer environment variable with a fallback default.

    Args:
        name: Name of the environment variable.
        default: Default value returned when the variable is unset or
            cannot be parsed as an integer.

    Returns:
        The parsed integer value, or ``default`` on failure.
    """
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


TRITON_CACHE_ENABLED = check_bool_flag(
    "EJKERNEL_TRITON_CACHE_COMPILES",
    check_bool_flag("EASYDEL_TRITON_CACHE_COMPILES", False),
)
TRITON_CACHE_VERBOSE = check_bool_flag(
    "EJKERNEL_TRITON_CACHE_VERBOSE",
    check_bool_flag("EASYDEL_TRITON_CACHE_VERBOSE", False),
)
TRITON_CACHE_MAX_ITEMS = _parse_int_env(
    "EJKERNEL_TRITON_CACHE_MAX_ITEMS",
    _parse_int_env("EASYDEL_TRITON_CACHE_MAX_ITEMS", 256),
)
TRITON_CACHE_DIR = Path(
    os.getenv(
        "EJKERNEL_TRITON_CACHE_DIR",
        os.getenv("EASYDEL_TRITON_CACHE_DIR", str(get_cache_dir() / "triton_kernels")),
    )
)


def normalize_grid(grid: GridOrLambda, metaparams) -> tuple[int, int, int]:
    """Normalize grid specification to a 3D tuple.

    Args:
        grid: Grid specification as int, tuple, or callable returning grid.
        metaparams: Dictionary of metaparameters for callable grid evaluation.

    Returns:
        3D tuple representing normalized grid dimensions.

    Raises:
        ValueError: If grid has more than 3 dimensions.
    """
    if callable(grid):
        grid = grid(metaparams)
    if isinstance(grid, int):
        grid = (grid,)
    elif len(grid) > 3:
        raise ValueError("`grid` should have three or fewer dimensions.")
    return tuple(grid) + (1,) * (3 - len(grid))


def avals_to_layouts(avals):
    """Convert abstract values to layout specifications.

    Args:
        avals: List of abstract values with ndim attribute.

    Returns:
        List of layout specifications as reversed dimension ranges.
    """
    return [list(reversed(range(aval.ndim))) for aval in avals]


def _device_set_from_sharding(sharding) -> set | None:
    """Extract the participating devices from a sharding object.

    Args:
        sharding: JAX sharding object.

    Returns:
        Set of devices referenced by the sharding, or ``None`` if unavailable.
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

    Args:
        arg: Potential array-like argument.

    Returns:
        Set of devices touched by ``arg``. Returns ``None`` for non-array
        values or when device placement cannot be inferred.
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
    """Validate that all array arguments are on one logical device.

    Args:
        array_args: Array or tracer arguments to validate.
        device_index: Optional requested device index.
        allow_sharded_tracers: Whether traced sharded values are allowed.

    Raises:
        AssertionError: If arguments span multiple devices, if arrays are on
            different devices, or if placement conflicts with ``device_index``.
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
                "triton_call requires all array arguments to be on a single device. "
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
                "triton_call requires all array arguments to be on the same device. "
                f"Argument {idx} is on a different device than argument {single_device_sets[0][0]}."
            )

    if device_index is None:
        return

    try:
        platform = getattr(first_device, "platform", None)
        devices = jax.devices(platform) if platform else jax.devices()
        if 0 <= device_index < len(devices) and devices[device_index] != first_device:
            raise AssertionError(
                "triton_call received inputs on a different device than the requested "
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
    """Check whether execution is currently inside a shard_map context.

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


def get_triton_type(obj: Any) -> str:
    """Get Triton type string representation for a given object.

    Args:
        obj: Object to get type for (ShapedArray, AbstractRef, constexpr, or scalar).

    Returns:
        String representation of the Triton type.

    Raises:
        ValueError: For integer overflow cases.
        NotImplementedError: For unsupported object types.
    """
    if isinstance(obj, jax.core.ShapedArray | state.AbstractRef):
        return f"*{_JAX_TO_TRITON_TYPE_MAP[obj.dtype]}"
    if isinstance(obj, tl.constexpr):
        obj = obj.value
    if isinstance(obj, int):
        if -(2**31) <= obj < 2**31:
            return "i32"
        elif 2**31 <= obj < 2**32:
            return "u32"
        elif -(2**63) <= obj < 2**63:
            return "i64"
        elif 2**63 <= obj < 2**64:
            return "u64"
        else:
            raise ValueError(f"integer overflow representing {obj}")
    if isinstance(obj, np.float32 | float):
        return "fp32"
    if isinstance(obj, bool):
        return "B"
    if isinstance(obj, str):
        return "str"
    raise NotImplementedError(f"could not compute type name for {obj}: {type(obj)}")


triton_kernel_call_p = jex.core.Primitive("triton_kernel_call")
triton_kernel_call_p.multiple_results = True
triton_kernel_call_p.def_impl(functools.partial(xla.apply_primitive, triton_kernel_call_p))


@triton_kernel_call_p.def_abstract_eval
def triton_kernel_call_abstract_eval(*_, out_shapes, **__):
    """Abstract evaluation function for triton kernel call primitive.

    Args:
        *_: Unused positional arguments.
        out_shapes: Output shape specifications.
        **: Unused keyword arguments.

    Returns:
        List of ShapedArray objects for outputs.
    """
    return [core.ShapedArray(out_shape.shape, out_shape.dtype) for out_shape in out_shapes]


def aval_size_bytes(aval):
    """Calculate size in bytes for an abstract value.

    Args:
        aval: Abstract value with dtype and size attributes.

    Returns:
        Size in bytes as integer.
    """
    return np.dtype(aval.dtype).itemsize * aval.size


def _normalize_cuda_compute_capability(value: Any) -> int:
    if isinstance(value, tuple):
        if len(value) < 2:
            raise ValueError(f"Invalid CUDA compute capability tuple: {value!r}.")
        return int(value[0]) * 10 + int(value[1])
    if isinstance(value, str):
        value = value.strip()
        if "." in value:
            major, minor = value.split(".", 1)
            return int(major) * 10 + int(minor[:1] or 0)
        value_i = int(value)
        return value_i if value_i >= 20 else value_i * 10
    if isinstance(value, float):
        return round(value * 10)
    value_i = int(value)
    return value_i if value_i >= 20 else value_i * 10


def _get_cuda_compute_capability(device: int) -> int:
    getter = getattr(triton_kernel_call_lib, "get_compute_capability", None)
    if getter is not None:
        return _normalize_cuda_compute_capability(getter(device))

    try:
        devices = jax.devices("gpu")
    except Exception:
        devices = jax.devices()
    try:
        device_obj = devices[int(device)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Unable to resolve CUDA device {device!r} for Triton compilation.") from exc

    compute_capability = getattr(device_obj, "compute_capability", None)
    if compute_capability is None:
        raise ValueError(
            "Unable to determine CUDA compute capability: jaxlib.gpu_triton does not expose "
            "get_compute_capability and the JAX device has no compute_capability attribute."
        )
    return _normalize_cuda_compute_capability(compute_capability)


def get_cuda_backend(device, compute_capability):
    """Create CUDA backend for Triton compilation.

    Args:
        device: CUDA device identifier.
        compute_capability: CUDA compute capability version.

    Returns:
        CUDABackend instance configured for the device.
    """
    target = cb.GPUTarget("cuda", compute_capability, 32)
    backend = cb.CUDABackend(target)
    return backend


def get_hip_backend(device, compute_capability):
    """Create HIP backend for Triton compilation on AMD GPUs.

    Args:
        device: HIP device identifier.
        compute_capability: GPU architecture specification.

    Returns:
        HIPBackend instance configured for the device.
    """
    arch = triton_kernel_call_lib.get_arch_details(device)
    arch = arch.split(":")[0]
    target = hb.GPUTarget("hip", arch, 64)
    backend = hb.HIPBackend(target)
    return backend


@dataclasses.dataclass
class CompilationResult:
    """Result of Triton kernel compilation containing binary and metadata.

    Attributes:
        binary: Compiled binary code (PTX for CUDA, HSACO path for ROCm).
        name: Name of the compiled kernel function.
        shared_mem_bytes: Amount of shared memory required in bytes.
        cluster_dims: Cluster dimensions for the kernel launch.
        ttgir: Triton GPU IR representation (optional).
        llir: LLVM IR representation (optional).
    """

    binary: str
    name: str
    shared_mem_bytes: int
    cluster_dims: tuple
    ttgir: str | None
    llir: str | None


def compile_ttir_inplace(
    ttir,
    backend: [cb.CUDABackend | hb.HIPBackend],  # type: ignore
    options: [cb.CUDAOptions | hb.HIPOptions],  # type: ignore
    compute_capability,
    platform,
):
    """Compile Triton IR to platform-specific binary in-place.

    Args:
        ttir: Triton IR module to compile.
        backend: Platform-specific backend (CUDA or HIP).
        options: Compilation options for the backend.
        compute_capability: Target compute capability.
        platform: Target platform ('cuda' or 'rocm').

    Returns:
        CompilationResult containing compiled binary and metadata.

    Raises:
        ValueError: For unsupported platforms.
    """
    if platform == "cuda":
        return compile_ttir_to_ptx_inplace(
            ttir,
            backend,
            options,
            compute_capability,
        )

    elif platform == "rocm":
        return compile_ttir_to_hsaco_inplace(
            ttir,
            backend,
            options,
            compute_capability,
        )
    else:
        raise ValueError("Unsupported device.")


def compile_ttir_to_ptx_inplace(
    ttir,
    cuda_backend: cb.CUDABackend,
    cuda_options: cb.CUDAOptions,
    compute_capability,
) -> CompilationResult:
    """Compile Triton IR to PTX binary for CUDA devices.

    Args:
        ttir: Triton IR module to compile.
        cuda_backend: CUDA compilation backend.
        cuda_options: CUDA-specific compilation options.
        compute_capability: CUDA compute capability version.

    Returns:
        CompilationResult with PTX binary and metadata.

    Raises:
        ValueError: If compilation passes fail.
    """
    if cuda_options.debug:
        print(ttir)
    try:
        metadata = {}
        opt_ttir = cuda_backend.make_ttir(ttir, metadata, cuda_options, compute_capability)
        ttgir = cuda_backend.make_ttgir(
            opt_ttir,
            metadata,
            cuda_options,
            compute_capability,
        )
    except RuntimeError as e:
        ttir.dump()
        raise ValueError("TTIR->TTGIR pass failed!") from e
    if cuda_options.debug:
        print(ttgir)
    try:
        llir = cuda_backend.make_llir(
            ttgir,
            metadata,
            cuda_options,
            compute_capability,
        )
    except RuntimeError as e:
        ttgir.dump()
        raise ValueError("TTGIR->LLIR pass failed!") from e
    shared_mem_bytes = metadata["shared"]
    if cuda_options.debug:
        print(llir)
    ptx = cuda_backend.make_ptx(
        llir,
        metadata,
        cuda_options,
        compute_capability,
    )
    if cuda_options.debug:
        print(ptx)
    name = metadata["name"]
    cluster_dims = metadata.get("cluster_dims", (0, 0, 0))
    return CompilationResult(
        binary=ptx,
        name=name,
        shared_mem_bytes=shared_mem_bytes,
        cluster_dims=cluster_dims,
        ttgir=None,
        llir=None,
    )


def compile_ttir_to_hsaco_inplace(
    ttir,
    hip_backend: hb.HIPBackend,  # type: ignore
    hip_options: hb.HIPOptions,  # type: ignore
    compute_capability,
) -> CompilationResult:
    """Compile Triton IR to HSACO binary for AMD ROCm devices.

    Args:
        ttir: Triton IR module to compile.
        hip_backend: HIP compilation backend.
        hip_options: HIP-specific compilation options.
        compute_capability: GPU architecture specification.

    Returns:
        CompilationResult with HSACO binary path and metadata.

    Raises:
        ValueError: If compilation passes fail.
    """
    if hip_options.debug:
        print(ttir)
    try:
        metadata = {}
        opt_ttir = hip_backend.make_ttir(ttir, metadata, hip_options)
        ttgir = hip_backend.make_ttgir(opt_ttir, metadata, hip_options)
    except RuntimeError as e:
        ttir.dump()
        raise ValueError("TTIR->TTGIR pass failed!") from e
    if hip_options.debug:
        print(ttgir)
    try:
        llir = hip_backend.make_llir(ttgir, metadata, hip_options)
    except RuntimeError as e:
        ttgir.dump()
        raise ValueError("TTGIR->LLIR pass failed!") from e
    shared_mem_bytes = metadata["shared"]
    if hip_options.debug:
        print(llir)

    amdgcn = hip_backend.make_amdgcn(llir, metadata, hip_options)
    hsaco = hip_backend.make_hsaco(amdgcn, metadata, hip_options)

    name = metadata["name"]
    cluster_dims = (0, 0, 0)
    fd, hsaco_path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(hsaco)
    return CompilationResult(
        binary=hsaco_path,
        name=name,
        shared_mem_bytes=shared_mem_bytes,
        cluster_dims=cluster_dims,
        ttgir=None,
        llir=None,
    )


def _log_triton_cache(msg: str) -> None:
    """Print a cache diagnostic message when verbose logging is enabled.

    Args:
        msg: The message string to print.
    """
    if TRITON_CACHE_VERBOSE:
        print(msg)


def _get_triton_cache_dir() -> Path:
    """Return the Triton kernel cache directory, creating it if needed.

    Returns:
        Path to the Triton kernel cache directory.
    """
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return TRITON_CACHE_DIR


def _triton_kernel_source_hash(fn: triton.JITFunction) -> str:
    """Compute a SHA-256 hash of a Triton JIT function's source code.

    Falls back to hashing the bytecode, constants, and names if the source
    text is unavailable (e.g. for dynamically generated functions).

    Args:
        fn: A Triton ``JITFunction`` whose source will be hashed.

    Returns:
        Hex-digest string of the SHA-256 hash.
    """
    try:
        source = inspect.getsource(fn.fn)
    except (OSError, TypeError):
        code = fn.fn.__code__
        source = repr((code.co_code, code.co_consts, code.co_names))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _make_triton_cache_key(
    *,
    fn: triton.JITFunction,
    platform: str,
    compute_capability: int,
    signature: tuple[tuple[str, str], ...],
    specialization: tuple[str, ...],
    constants: tuple[tuple[str, Any], ...],
    num_warps: int,
    num_stages: int,
    num_ctas: int,
    enable_fp_fusion: bool,
) -> str:
    """Build a deterministic SHA-256 cache key for a Triton kernel compilation.

    Incorporates Triton/JAX versions, platform, compute capability, kernel
    source hash, type signature, specialization attributes, constants, and
    launch parameters to produce a unique key.

    Args:
        fn: The Triton ``JITFunction`` being compiled.
        platform: Target platform (e.g. ``"cuda"`` or ``"rocm"``).
        compute_capability: GPU compute capability version.
        signature: Tuple of ``(param_name, type_string)`` pairs.
        specialization: Tuple of specialization attribute strings.
        constants: Tuple of ``(param_name, value)`` pairs for constants.
        num_warps: Number of warps per thread block.
        num_stages: Number of pipeline stages.
        num_ctas: Number of cooperative thread arrays.
        enable_fp_fusion: Whether floating-point fusion is enabled.

    Returns:
        A SHA-256 hex-digest string uniquely identifying this compilation.
    """
    source_hash = _triton_kernel_source_hash(fn)
    payload = (
        triton.__version__,
        jax.__version__,
        platform,
        compute_capability,
        fn.__module__,
        fn.fn.__name__,
        source_hash,
        signature,
        specialization,
        constants,
        num_warps,
        num_stages,
        num_ctas,
        enable_fp_fusion,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _invalidate_triton_cache_entry(cache_key: str, reason: str) -> None:
    """Remove a Triton kernel cache entry from disk.

    Args:
        cache_key: The cache key identifying the entry to remove.
        reason: Human-readable reason for the invalidation (logged).
    """
    cache_dir = _get_triton_cache_dir() / cache_key
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    _log_triton_cache(f"[triton-cache] invalidated {cache_key}: {reason}")


def _load_triton_kernel_cache(
    cache_key: str,
    *,
    platform: str,
    compute_capability: int,
) -> CompilationResult | None:
    """Load a compiled Triton kernel from the on-disk cache.

    Validates platform and compute capability against the stored metadata.
    If the cache entry is corrupt or mismatched it is automatically
    invalidated.

    Args:
        cache_key: SHA-256 hex-digest identifying the cached kernel.
        platform: Expected platform (``"cuda"`` or ``"rocm"``).
        compute_capability: Expected GPU compute capability.

    Returns:
        A ``CompilationResult`` on a successful cache hit, or ``None`` if
        the entry does not exist or failed validation.
    """
    cache_path = _get_triton_cache_dir() / cache_key / "kernel.pkl"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid cache payload")
        if data.get("platform") != platform:
            raise ValueError("platform mismatch")
        if data.get("compute_capability") != compute_capability:
            raise ValueError("compute capability mismatch")
        binary = data.get("binary")
        if not isinstance(binary, (bytes, bytearray)):
            raise ValueError("invalid binary payload")
        if platform == "cuda":
            binary = binary.decode("utf-8")
        return CompilationResult(
            binary=binary,
            name=data["name"],
            shared_mem_bytes=data["shared_mem_bytes"],
            cluster_dims=tuple(data.get("cluster_dims", (0, 0, 0))),
            ttgir=None,
            llir=None,
        )
    except Exception as exc:
        _invalidate_triton_cache_entry(cache_key, f"load failed: {exc}")
        return None


def _save_triton_kernel_cache(
    cache_key: str,
    compilation_result: CompilationResult,
    *,
    platform: str,
    compute_capability: int,
    ttir: str | None,
) -> None:
    """Persist a compiled Triton kernel to the on-disk cache.

    Writes the binary, metadata, and optional TTIR to a pickle file using
    an atomic rename to avoid partial writes.

    Args:
        cache_key: SHA-256 hex-digest identifying this compilation.
        compilation_result: The compiled kernel binary and metadata.
        platform: Target platform (``"cuda"`` or ``"rocm"``).
        compute_capability: GPU compute capability used during compilation.
        ttir: Optional Triton IR text to store alongside the binary
            (typically only when debug mode is enabled).
    """
    cache_dir = _get_triton_cache_dir() / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "kernel.pkl"
    tmp_path = cache_dir / "kernel.pkl.tmp"

    if platform == "rocm":
        with open(compilation_result.binary, "rb") as f:
            binary = f.read()
    else:
        binary = compilation_result.binary.encode("utf-8")

    payload = {
        "platform": platform,
        "compute_capability": compute_capability,
        "binary": binary,
        "name": compilation_result.name,
        "shared_mem_bytes": compilation_result.shared_mem_bytes,
        "cluster_dims": compilation_result.cluster_dims,
        "ttir": ttir,
    }
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp_path, cache_path)


_COMPILED_KERNEL_CACHE: OrderedDict[str, triton_kernel_call_lib.TritonKernel] = OrderedDict()  # type: ignore


def _lru_get(cache: OrderedDict, key: str):
    """Retrieve a value from an LRU OrderedDict cache, promoting it on hit.

    Args:
        cache: The ``OrderedDict`` acting as an LRU cache.
        key: The cache key to look up.

    Returns:
        The cached value if found, or ``None`` on a miss.
    """
    kernel = cache.get(key)
    if kernel is not None:
        cache.move_to_end(key)
    return kernel


def _lru_set(cache: OrderedDict, key: str, value) -> None:
    """Insert or update a value in an LRU OrderedDict cache.

    Moves the entry to the end (most recently used) and evicts the oldest
    entry if the cache exceeds ``TRITON_CACHE_MAX_ITEMS``.

    Args:
        cache: The ``OrderedDict`` acting as an LRU cache.
        key: The cache key.
        value: The value to store.
    """
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > TRITON_CACHE_MAX_ITEMS:
        cache.popitem(last=False)


def get_or_create_triton_kernel(
    backend_init_func,
    platform,
    fn,
    arg_dtypes,
    scalar_args,
    device,
    *,
    num_warps,
    num_stages,
    num_ctas,
    compute_capability,
    enable_fp_fusion,
    metaparams,
    dump: bool,
) -> tuple[triton_kernel_call_lib.TritonKernel, Any]:  # type: ignore
    """Get or create a compiled Triton kernel with caching.

    Args:
        backend_init_func: Function to initialize backend for compilation.
        platform: Target platform string.
        fn: Triton JIT function to compile.
        arg_dtypes: Argument data types.
        scalar_args: Scalar argument specifications.
        device: Target device identifier.
        num_warps: Number of warps per thread block.
        num_stages: Number of pipeline stages.
        num_ctas: Number of cooperative thread arrays.
        compute_capability: Target compute capability.
        enable_fp_fusion: Whether to enable floating point fusion.
        metaparams: Kernel metaparameters.
        dump: Whether to dump debug information.

    Returns:
        Tuple of compiled TritonKernel and specialization attributes.

    Raises:
        ValueError: For unsupported configurations or compilation errors.
    """
    if num_warps is None:
        num_warps = 4
    if num_stages is None:
        num_stages = 3
    if compute_capability is None:
        compute_capability = (
            _get_cuda_compute_capability(device)
            if platform == "cuda"
            else triton_kernel_call_lib.get_arch_details(device)
        )
    if platform == "cuda" and num_ctas > 1 and compute_capability < 90:
        raise ValueError("num_ctas > 1 unsupported before Hopper.")

    backend = backend_init_func(device, compute_capability)

    signature = {fn.arg_names[i]: v for i, v in enumerate(arg_dtypes)}
    alignments = [16] * len(arg_dtypes)
    for i, _, value in scalar_args:
        alignments[i] = value
    scalar_values = {i: value for i, _, value in scalar_args}
    if hasattr(backend, "get_arg_specialization"):
        specialize_extra = backend.get_arg_specialization
        if specialize_impl := getattr(triton.runtime.jit, "specialize_impl", None):
            specialize_impl = functools.partial(specialize_impl, specialize_extra=specialize_extra)
        else:
            create_specialize_impl = triton.runtime.jit.create_specialize_impl
            if len(inspect.signature(create_specialize_impl).parameters) == 0:
                specialize_impl = functools.partial(create_specialize_impl(), specialize_extra=specialize_extra)
            else:
                specialize_impl = create_specialize_impl(specialize_extra)
        specialization = [
            specialize_impl(
                types.SimpleNamespace(data_ptr=lambda alignment=alignment: alignment, dtype=arg_dtype.removeprefix("*")),
            )
            for arg_dtype, alignment in safe_zip(arg_dtypes, alignments)
        ]
    else:
        specialize_impl = triton.runtime.jit.native_specialize_impl
        specialization = []
        for i, (arg_dtype, alignment) in enumerate(safe_zip(arg_dtypes, alignments)):
            if i in scalar_values:
                specialization.append(specialize_impl(backend, scalar_values[i], False, True, True))
            else:
                arg = types.SimpleNamespace(
                    data_ptr=lambda alignment=alignment: alignment,
                    dtype=arg_dtype.removeprefix("*"),
                )
                specialization.append(specialize_impl(backend, arg, False, True, True))
    attrs = {}
    for i, (_, attr) in enumerate(specialization):
        if not isinstance(attr, str):
            attr = ""
        attrs[(i,)] = backend.parse_attr(attr)
    constants = dict(metaparams)
    constants.update({k: None for _, k, v in scalar_args if v is None})
    constants.update({fn.arg_names[i]: 1 for i, _, v in scalar_args if v == 1})
    for constant in constants:
        signature[constant] = "constexpr"

    signature_key = tuple(signature.items())
    specialization_key = tuple(repr(item) for item in specialization)
    constants_key = tuple(sorted(constants.items()))

    cache_key = _make_triton_cache_key(
        fn=fn,
        platform=platform,
        compute_capability=compute_capability,
        signature=signature_key,
        specialization=specialization_key,
        constants=constants_key,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
        enable_fp_fusion=enable_fp_fusion,
    )
    kernel = _lru_get(_COMPILED_KERNEL_CACHE, cache_key)
    if kernel is not None:
        _log_triton_cache(f"[triton-cache] hit (memory) key={cache_key}")

    if kernel is None and TRITON_CACHE_ENABLED:
        cached = _load_triton_kernel_cache(
            cache_key,
            platform=platform,
            compute_capability=compute_capability,
        )
        if cached is not None:
            _log_triton_cache(f"[triton-cache] hit (disk) key={cache_key}")
            if platform == "rocm":
                fd, hsaco_path = tempfile.mkstemp()
                with os.fdopen(fd, "wb") as f:
                    f.write(cached.binary)
                cached_binary = hsaco_path
            else:
                cached_binary = cached.binary
            try:
                kernel = triton_kernel_call_lib.TritonKernel(
                    cached.name,
                    num_warps,
                    num_ctas,
                    cached.shared_mem_bytes,
                    cached_binary,
                    "",
                    compute_capability,
                )
            except TypeError:
                kernel = triton_kernel_call_lib.TritonKernel(
                    cached.name,
                    num_warps,
                    cached.shared_mem_bytes,
                    cached_binary,
                    "",
                    compute_capability,
                    *cached.cluster_dims,
                )
            _lru_set(_COMPILED_KERNEL_CACHE, cache_key, kernel)
        else:
            _log_triton_cache(f"[triton-cache] miss (disk) key={cache_key}")

    if kernel is None:
        opts = {
            "num_warps": num_warps,
            "num_stages": num_stages,
            "num_ctas": num_ctas,
            "optimize_epilogue": False,
            "debug": dump,
            "enable_fp_fusion": enable_fp_fusion,
        }

        options = backend.parse_options(opts)
        context = _triton.ir.context()
        _triton.ir.load_dialects(context)
        backend.load_dialects(context)
        codegen_fns = backend.get_codegen_implementation(options)

        module = code_gen.ast_to_ttir(
            fn,
            tc.ASTSource(fn, constexprs=constants, signature=signature, attrs=attrs),
            options=options,
            codegen_fns=codegen_fns,
            context=context,
            module_map=backend.get_module_map(),
        )
        ttir = str(module)

        compilation_result = compile_ttir_inplace(module, backend, options, compute_capability, platform)

        kernel_name = compilation_result.name
        try:
            kernel = triton_kernel_call_lib.TritonKernel(
                kernel_name,
                num_warps,
                num_ctas,
                compilation_result.shared_mem_bytes,
                compilation_result.binary,
                ttir,
                compute_capability,
            )
        except TypeError:
            kernel = triton_kernel_call_lib.TritonKernel(
                kernel_name,
                num_warps,
                compilation_result.shared_mem_bytes,
                compilation_result.binary,
                ttir,
                compute_capability,
                *compilation_result.cluster_dims,
            )

        _lru_set(_COMPILED_KERNEL_CACHE, cache_key, kernel)
        if TRITON_CACHE_ENABLED:
            try:
                _save_triton_kernel_cache(
                    cache_key,
                    compilation_result,
                    platform=platform,
                    compute_capability=compute_capability,
                    ttir=ttir if dump else None,
                )
                _log_triton_cache(f"[triton-cache] saved key={cache_key}")
            except Exception as exc:
                _log_triton_cache(f"[triton-cache] save failed key={cache_key}: {exc}")

    return kernel, attrs


def triton_kernel_call_lowering(
    backend_init_func,
    ctx,
    *array_args,
    fn,
    scalar_args,
    name,
    custom_call_target_name,
    out_shapes,
    grid,
    num_warps,
    num_stages,
    num_ctas,
    device,
    compute_capability,
    enable_fp_fusion,
    input_output_aliases,
    zeroed_outputs,
    debug,
    serialized_metadata,
    **metaparams,
):
    """Lower Triton kernel call to platform-specific implementation.

    This function handles the compilation and lowering of Triton kernels for
    execution, including autotuning support and platform-specific optimizations.

    Args:
        backend_init_func: Function to initialize the compilation backend.
        ctx: Lowering context containing type information.
        *array_args: Array arguments to the kernel.
        fn: Triton kernel function to execute.
        scalar_args: Scalar argument specifications.
        name: Name for the kernel call.
        custom_call_target_name: Target name for the custom call.
        out_shapes: Output tensor shapes.
        grid: Kernel launch grid specification.
        num_warps: Number of warps per thread block.
        num_stages: Number of pipeline stages.
        num_ctas: Number of cooperative thread arrays.
        device: Target device identifier.
        compute_capability: Target compute capability.
        enable_fp_fusion: Whether to enable floating point fusion.
        input_output_aliases: Input-output aliasing specifications.
        zeroed_outputs: Outputs to zero-initialize.
        debug: Whether to enable debug output.
        serialized_metadata: Serialized kernel metadata.
        **metaparams: Additional kernel metaparameters.

    Returns:
        Lowered kernel call rule result.
    """
    kernel_call_name = name
    args = list(ctx.avals_in)
    arg_dtypes = list(safe_map(get_triton_type, ctx.avals_in))
    for idx, dtype, v in scalar_args:
        args.insert(idx, v)
        arg_dtypes.insert(idx, dtype)
    args.extend(ctx.avals_out)
    arg_dtypes.extend(safe_map(get_triton_type, ctx.avals_out))
    named_args = dict(unsafe_zip(fn.arg_names, args))

    default_config = triton.Config(
        {},
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )

    def unwrap_triton_kernel(kernel, configs):
        """Recursively unwrap Triton kernel wrappers to the underlying JIT function.

        Handles the Triton decorator stack (``Autotuner``, ``Heuristics``, and
        ``JITFunction``), extracting the base JIT function and accumulating
        the resolved configuration list. Supports both ``@autotune @heuristics @jit``
        and ``@heuristics @autotune @jit`` decorator orderings.

        Args:
            kernel: A Triton kernel which may be an ``Autotuner``, ``Heuristics``
                wrapper, or a base ``JITFunction``.
            configs: List of ``triton.Config`` objects accumulated so far.

        Returns:
            Tuple of (jit_function, resolved_configs) where ``jit_function`` is
            the underlying ``triton.JITFunction`` and ``resolved_configs`` is the
            list of configurations after pruning and heuristic evaluation.

        Raises:
            NotImplementedError: If any config uses a ``pre_hook``, which is
                not supported in the JAX-Triton bridge.
        """

        if isinstance(kernel, autotuner.Autotuner):
            prev_early_config_prune_fn = kernel.early_config_prune

            def prune_configs(configs, named_args, **kwargs):
                """Filter autotuner configs to those compatible with the current metaparams.

                Removes configs whose ``pre_hook`` is set (unsupported) and
                retains only those whose keyword arguments are consistent with
                the caller-supplied ``metaparams``. Delegates to any previously
                registered ``early_config_prune`` function afterwards.

                Args:
                    configs: List of ``triton.Config`` candidates.
                    named_args: Dictionary of named arguments for the kernel.
                    **kwargs: Additional keyword arguments (unused).

                Returns:
                    Pruned list of ``triton.Config`` objects.

                Raises:
                    NotImplementedError: If any config has a ``pre_hook``.
                """
                pruned_configs = []
                for config in configs:
                    if config.pre_hook is not None:
                        raise NotImplementedError("`pre_hook` is not supported")

                    if all(config.kwargs.get(k, v) == v for k, v in metaparams.items()):
                        pruned_configs.append(config)
                if prev_early_config_prune_fn is not None:
                    pruned_configs = prev_early_config_prune_fn(pruned_configs, named_args)
                return pruned_configs

            kernel.early_config_prune = prune_configs
            kernel.nargs = named_args
            tuned_configs = kernel.prune_configs(metaparams)
            return unwrap_triton_kernel(kernel.fn, tuned_configs)

        if isinstance(kernel, autotuner.Heuristics):
            inner_fn, configs = unwrap_triton_kernel(kernel.fn, configs)
            updated_configs = []
            for config in configs:
                kwargs = config.kwargs.copy()
                for name, heuristic in kernel.values.items():
                    kwargs[name] = heuristic({**named_args, **metaparams, **kwargs})
                updated_config = copy.copy(config)
                updated_config.kwargs = kwargs
                updated_configs.append(updated_config)
            return inner_fn, updated_configs

        return kernel, configs

    fn, configs = unwrap_triton_kernel(fn, [default_config])

    if not isinstance(fn, triton.JITFunction):
        raise ValueError("`kernel` must be a Triton `JITFunction`, `Heuristics` or `Autotuner`.")

    outputs_offset = len(ctx.avals_in) + len(scalar_args)
    config_params = []
    for config in configs:
        config_metaparams = {**metaparams, **config.kwargs}
        config_grid = normalize_grid(grid, config_metaparams)

        config_zeroed_outputs = zeroed_outputs
        if callable(zeroed_outputs):
            config_zeroed_outputs = config_zeroed_outputs(config_metaparams)

        zeroed_params_with_sizes = {
            i + outputs_offset: aval_size_bytes(ctx.avals_out[i]) for i in sorted(config_zeroed_outputs)
        }

        config_params.append(
            dict(
                metaparams=tuple(sorted(config_metaparams.items())),
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                num_ctas=config.num_ctas,
                grid=config_grid,
                zeroed_params_with_sizes=tuple(zeroed_params_with_sizes.items()),
            )
        )

    kernel_calls = []
    for params in config_params:
        kernel, specialization_attr = get_or_create_triton_kernel(
            backend_init_func,
            ctx.module_context.platforms[0],
            fn,
            arg_dtypes,
            scalar_args,
            device,
            num_warps=params["num_warps"],
            num_stages=params["num_stages"],
            num_ctas=params["num_ctas"],
            compute_capability=compute_capability,
            enable_fp_fusion=enable_fp_fusion,
            metaparams=dict(params["metaparams"]),
            dump=debug,
        )

        kernel_params = []
        zeroed_params_with_sizes = dict(params["zeroed_params_with_sizes"])
        equal_to_1 = {i for i, _, v in scalar_args if v == 1}
        for i, (arg, dtype) in enumerate(safe_zip(args, arg_dtypes)):
            if isinstance(arg, core.ShapedArray):
                arg_attrs = specialization_attr[(i,)]
                kernel_params.append(
                    triton_kernel_call_lib.create_array_parameter(
                        zeroed_params_with_sizes.get(i, 0),
                        16 if (["tt.divisibility", 16] in arg_attrs) else 0,
                    )
                )
            elif i not in equal_to_1:
                kernel_params.append(triton_kernel_call_lib.create_scalar_parameter(arg, dtype))

        kernel_calls.append(
            triton_kernel_call_lib.TritonKernelCall(
                kernel,
                params["grid"][0],
                params["grid"][1],
                params["grid"][2],
                kernel_params,
            )
        )

    if len(kernel_calls) > 1:
        named_scalar_args = {fn.arg_names[i]: v for i, _, v in scalar_args}
        input_output_aliases_with_sizes = tuple(
            (input_idx, output_idx, aval_size_bytes(ctx.avals_in[input_idx]))
            for input_idx, output_idx in input_output_aliases
        )
        kernel_call = triton_kernel_call_lib.TritonAutotunedKernelCall(
            f"{kernel_call_name} ({fn.fn.__name__}) {named_scalar_args}",
            [(call, str(config)) for call, config in safe_zip(kernel_calls, configs)],
            input_output_aliases_with_sizes,
        )
    else:
        kernel_call = kernel_calls[0]

    call_proto = kernel_call.to_proto(kernel_call_name, serialized_metadata)
    rule = jax.ffi.ffi_lowering(
        custom_call_target_name,
        api_version=2,
        backend_config=zlib.compress(call_proto),
        operand_output_aliases=dict(input_output_aliases),
    )
    return rule(ctx, *array_args)


mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, get_cuda_backend),
    platform="cuda",
)

mlir.register_lowering(
    triton_kernel_call_p,
    functools.partial(triton_kernel_call_lowering, get_hip_backend),
    platform="rocm",
)


def triton_kernel_call_raise_on_jvp(*args, **kwargs):
    """Raise error for automatic differentiation on Triton kernels.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always, as JVP is not supported.
    """
    del args, kwargs
    raise NotImplementedError(
        "jax_triton.triton_call does not support automatic differentiation. Use "
        "jax.custom_jvp or jax.custom_vjp to implement a custom automatic "
        "differentiation rule for your kernel."
    )


ad.primitive_jvps[triton_kernel_call_p] = triton_kernel_call_raise_on_jvp


def triton_kernel_call_raise_on_vmap(*args, **kwargs):
    """Raise error for batching with vmap on Triton kernels.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Raises:
        NotImplementedError: Always, as vmap is not supported.
    """
    del args, kwargs
    raise NotImplementedError(
        "jax_triton.triton_call does not support batching with jax.vmap. Use "
        "jax.custom_batching.custom_vmap to implement a custom batching rule for "
        "your kernel."
    )


batching.primitive_batchers[triton_kernel_call_p] = triton_kernel_call_raise_on_vmap


def triton_call(
    *args: jax.Array | bool | int | float | np.float32,
    kernel: triton.JITFunction | triton.runtime.Heuristics | triton.runtime.Autotuner,
    out_shape: ShapeDtype | Sequence[ShapeDtype],
    grid: GridOrLambda,
    name: str = "",
    custom_call_target_name: str = "triton_kernel_call",
    num_warps: int | None = None,
    num_stages: int | None = None,
    num_ctas: int = 1,
    device: int = 0,
    compute_capability: int | None = None,
    enable_fp_fusion: bool = True,
    input_output_aliases: dict[int, int] | None = None,
    zeroed_outputs: Sequence[int] | Callable[[dict[str, Any]], Sequence[int]] = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
    **metaparams: Any,
) -> Any:
    """Call a Triton kernel from JAX with specified parameters.

    This is the main entry point for executing Triton kernels within JAX
    computations. It handles compilation, optimization, and execution of
    GPU kernels written in Triton.

    Args:
        *args: Input arguments to the kernel (arrays and scalars).
        kernel: Triton kernel function, heuristics, or autotuner to execute.
        out_shape: Expected output shape(s) and dtype(s).
        grid: Kernel launch grid specification or callable returning grid.
        name: Optional name for the kernel call.
        custom_call_target_name: Target name for the custom call.
        num_warps: Number of warps per thread block (default: 4).
        num_stages: Number of pipeline stages (default: 3).
        num_ctas: Number of cooperative thread arrays.
        device: Target device identifier.
        compute_capability: Target compute capability (auto-detected if None).
        enable_fp_fusion: Whether to enable floating point fusion.
        input_output_aliases: Mapping of input indices to output indices for aliasing.
        zeroed_outputs: Indices of outputs to zero-initialize or callable returning them.
        debug: Whether to enable debug output during compilation.
        serialized_metadata: Additional serialized kernel metadata.
        **metaparams: Additional kernel metaparameters.

    Returns:
        JAX array(s) containing the kernel execution results.

    Raises:
        ValueError: If Triton is not installed or for invalid configurations.
        AssertionError: If array arguments are not on a single device, or if
            multiple accelerators are available and the call is not under
            ``jax.shard_map``.
    """
    if not CAN_USE_TRITON:
        raise ValueError("`triton_call` is only available when `triton` is installed.")
    out_shape = tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)
    flat_args, _ = tree_util.tree_flatten(args)
    flat_out_shapes, out_tree = tree_util.tree_flatten(out_shape)

    array_args = []
    scalar_args = []
    for i, arg in enumerate(flat_args):
        if isinstance(arg, bool | int | float):
            scalar_args.append((i, get_triton_type(arg), arg))
        elif isinstance(arg, np.float32):
            scalar_args.append((i, get_triton_type(arg), float(arg)))
        else:
            array_args.append(arg)

    in_shard_map_context = _in_shard_map_context()
    _assert_single_device_args(array_args, device, allow_sharded_tracers=in_shard_map_context)
    if _has_multi_accelerators() and not in_shard_map_context:
        raise AssertionError(
            "Multiple accelerator devices detected. "
            "triton_call must be invoked under `jax.shard_map` in multi-accelerator setups."
        )

    if input_output_aliases is None:
        input_output_aliases = {}

    out_flat = triton_kernel_call_p.bind(
        *array_args,
        fn=kernel,
        scalar_args=tuple(scalar_args),
        name=name,
        custom_call_target_name=custom_call_target_name,
        out_shapes=tuple(flat_out_shapes),
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
        device=device,
        compute_capability=compute_capability,
        enable_fp_fusion=enable_fp_fusion,
        input_output_aliases=tuple(input_output_aliases.items()),
        zeroed_outputs=zeroed_outputs,
        debug=debug,
        serialized_metadata=serialized_metadata,
        **metaparams,
    )
    return tree_util.tree_unflatten(out_tree, out_flat)
