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


"""Utility functions for the ejKernel library.

This module provides a collection of utility functions for kernel development,
including mathematical operations, array manipulation, hardware detection,
performance testing, distributed synchronization, and JAX mesh / sharding helpers.

The utilities support both Triton and JAX-based kernel implementations, with
specific handling for AMD CDNA/RDNA GPU architectures and distributed training
scenarios using ``jax.distributed``.

Key Functions:
    cdiv: Ceiling division for int and JAX arrays.
    next_power_of_2: Smallest power of 2 >= x.
    strides_from_shape / get_strides: Row-major (C-contiguous) stride computation.
    get_stride / kw_strides: Per-dimension stride access and kernel kwarg helpers.
    is_hip / is_cdna / is_rdna: AMD GPU architecture detection via Triton runtime.
    safe_autotune: Triton autotuning with graceful fallback.
    barrier_sync: Multi-process JAX barrier using the distributed client.
    numeric_gen / random_dense: Reproducible random array generation for testing.
    make_mesh / get_qkv_shardings / get_segments_shardings: JAX mesh and sharding
        helpers for (dp, fsdp, tp, sp) parallelism configurations.
    make_dummy_rpa_inputs: Full dummy input set for Ragged Paged Attention testing.
    generate_block_indices: Random sparse-attention block index generator.
    get_tpu_generation: TPU generation detection by parsing ``device_kind``.
    calculate_blocksize_and_wraps: Triton block-size / warp-count selection.

Constants:
    CDNA_ARCHS: AMD CDNA GPU architecture identifiers (MI100/MI200/MI300 series).
    RDNA_ARCHS: AMD RDNA GPU architecture identifiers (RX 6000/7000 series).
    Layouts: Type alias for supported attention tensor layout format strings.
    DEBUG_GLOBAL_RNG: Module-level JAX PRNGKey used by numeric_gen/random_dense.

Note:
    ``get_abs_err`` and ``get_err_ratio`` call PyTorch-style methods (``.detach()``,
    ``.square()``, ``.flatten().abs().max().item()``) and therefore only work with
    PyTorch tensors, not JAX arrays. They are likely legacy helpers; avoid using
    them with JAX arrays.

Example:
    >>> from ejkernel.utils import cdiv, next_power_of_2, is_hip
    >>>
    >>> # Calculate number of tiles
    >>> num_tiles = cdiv(seq_len, block_size)
    >>>
    >>> # Get optimal block size
    >>> block_size = next_power_of_2(head_dim)
"""

import functools
import re
import typing as tp
from typing import Literal, overload

import jax
import numpy
import numpy as np

try:
    import triton  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
from jax import Array
from jax import numpy as jnp
from jax.sharding import PartitionSpec as Ps

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
"""TypeVar bound to ``Callable`` for use in decorator type annotations."""

DEBUG_GLOBAL_RNG = None
"""Global RNG key used by :func:`numeric_gen` and :func:`random_dense` for reproducible test generation."""

CDNA_ARCHS = ["gfx940", "gfx941", "gfx942", "gfx90a", "gfx908"]
"""AMD CDNA (Compute DNA) architecture identifiers: MI100, MI200, MI300 series GPUs."""

RDNA_ARCHS = ["gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"]
"""AMD RDNA (Radeon DNA) architecture identifiers: RX 6000, RX 7000 series GPUs."""

Layouts: type = Literal["bhsd", "bshd", "thd"]
"""Type alias for supported attention tensor layout formats."""


@overload
def cdiv(a: int, b: int) -> int: ...


@overload
def cdiv(a: int, b: jax.Array) -> jax.Array: ...


@overload
def cdiv(a: jax.Array, b: int) -> jax.Array: ...


@overload
def cdiv(a: jax.Array, b: jax.Array) -> jax.Array: ...


def cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
    """Ceiling division operation.

    Computes the ceiling division of a by b, which is equivalent to (a + b - 1) // b.

    Args:
            a: Dividend, can be an integer or a JAX array.
            b: Divisor, can be an integer or a JAX array.

    Returns:
            The ceiling division result with the same type as inputs.
    """
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    return jax.lax.div(a + b - 1, b)


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Calculate row-major (C-contiguous) strides for an array with the given shape.

    The stride for dimension ``i`` is the number of elements between consecutive
    elements along that dimension, assuming a contiguous C-order layout.

    Args:
        shape: A tuple of integers representing the dimensions of an array.

    Returns:
        A tuple of integers representing the C-contiguous strides. The last
        dimension always has stride 1.

    Example:
        >>> strides_from_shape((2, 3, 4))
        (12, 4, 1)
    """
    size = np.prod(shape)
    strides = []
    for s in shape:
        size = size // s
        strides.append(int(size))
    return tuple(strides)


def get_stride(shape: tuple[int, ...] | jax.Array, index=0) -> int:
    """Get the C-contiguous stride at a specific dimension index.

    Delegates to :func:`get_strides` and returns the element at ``index``.
    If an array is passed instead of a shape tuple, its ``.shape`` attribute
    is used automatically.

    Args:
        shape: Shape tuple ``(d0, d1, ..., dn)`` or a JAX array whose
            ``.shape`` is used.
        index: The dimension index to retrieve the stride for. Defaults to 0.

    Returns:
        The stride (number of elements) at the specified dimension.

    Example:
        >>> get_stride((2, 3, 4), index=1)
        4
    """
    return get_strides(shape)[index]


def next_power_of_2(x: int) -> int:
    """Returns the next power of two greater than or equal to `x`.

    Args:
            x: A non-negative integer.

    Returns:
            The smallest power of 2 greater than or equal to x.

    Raises:
            ValueError: If x is negative.
    """
    if x < 0:
        raise ValueError("`next_power_of_2` requires a non-negative integer.")
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def safe_autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
) -> tp.Callable[[F], F]:
    """Safely apply Triton autotuning with fallback on failure.

    Wraps a function with Triton's autotuning capability, gracefully falling back
    to the original function if autotuning fails. This ensures kernel execution
    continues even if autotuning encounters issues.

    Args:
        configs: List of triton.Config objects to test during autotuning.
        key: List of argument names whose values define the autotuning key.
        prune_configs_by: Optional dict mapping metric names to pruning functions.
        reset_to_zero: List of argument names to reset to zero between runs.
        restore_value: List of argument names to restore after autotuning.
        pre_hook: Optional function to call before each autotuning run.
        post_hook: Optional function to call after each autotuning run.
        warmup: Number of warmup runs before measuring performance.
        rep: Number of repetitions for each configuration.

    Returns:
        A decorator that applies autotuning to the wrapped function.

    Example:
        >>> @safe_autotune(
        ...     configs=[triton.Config({'BLOCK_SIZE': 128})],
        ...     key=['n_elements']
        ... )
        ... def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        ...     pass
    """
    try:
        from triton.runtime.autotuner import Autotuner  # type: ignore

        def decorator(fn: F) -> F:
            """Wrap ``fn`` with Triton Autotuner, falling back to ``fn`` on failure."""
            try:
                return Autotuner(
                    fn,
                    fn.arg_names,
                    configs,
                    key,
                    reset_to_zero,
                    restore_value,
                    pre_hook=pre_hook,
                    post_hook=post_hook,
                    prune_configs_by=prune_configs_by,
                    warmup=warmup,
                    rep=rep,
                )
            except Exception:
                return fn

        return decorator
    except (Exception, RuntimeError) as err:
        print(f"Couldn't autotune given function due to {err}")

        def decorator(fn: F) -> F:
            """No-op fallback decorator when Triton autotuning is unavailable."""
            return fn

        return decorator


def dtype_index(x: jnp.array) -> int:
    """Get numeric index for array dtype.

    Maps JAX array dtypes to numeric indices for use in kernel dispatch
    and configuration.

    Args:
        x: JAX array whose dtype to index.

    Returns:
        Integer index corresponding to the dtype:
            1 for float16
            2 for bfloat16
            3 for float32

    Raises:
        ValueError: If the dtype is not supported.
    """
    if x.dtype == jnp.float16:
        return 1
    if x.dtype == jnp.bfloat16:
        return 2
    if x.dtype == jnp.float32:
        return 3
    raise ValueError(x.dtype)


def get_sharding(arr: jax.Array):
    """Retrieve the sharding specification of a JAX array.

    Safely extracts the ``sharding`` attribute from a JAX array without
    raising an error if the array is not sharded.

    Args:
        arr: A JAX array, possibly distributed across devices.

    Returns:
        The ``jax.sharding.Sharding`` object if the array is sharded,
        or ``None`` if the array has no sharding attribute.
    """
    return getattr(arr, "sharding", None)


def get_strides(shape: tuple[int, ...] | jax.Array) -> tuple[int, ...]:
    """Calculate row-major (C-contiguous) strides for a given shape.

    Equivalent to :func:`strides_from_shape` but also accepts a JAX array
    directly, extracting its ``.shape`` attribute automatically.

    Args:
        shape: Shape tuple ``(d0, d1, ..., dn)`` or a JAX array whose
            ``.shape`` attribute is used.

    Returns:
        Tuple of integer strides in the same order as the shape dimensions.
        The last element is always 1 (innermost stride).

    Example:
        >>> get_strides((2, 3, 4))
        (12, 4, 1)
    """
    if hasattr(shape, "shape"):
        shape = shape.shape
    size = numpy.prod(shape)
    strides = []
    for s in shape:
        size = int(size // s)
        strides.append(size)
    return tuple(strides)


def get_padded_headsize(size):
    """Calculate padded head size for optimal memory alignment.

    Rounds up the head size to the smallest power of 2 that is >= ``size``,
    with a minimum of 16. This is used in attention kernels to ensure
    favourable memory-access patterns.

    Note:
        Unlike :func:`next_power_of_2`, this function uses a bit-length
        trick (``1 << (size - 1).bit_length()``) rather than a loop, and
        enforces a floor of 16 regardless of input.

    Args:
        size: Original head dimension size (positive integer).

    Returns:
        Padded head size as a power of 2, at least 16.

    Example:
        >>> get_padded_headsize(13)
        16
        >>> get_padded_headsize(20)
        32
    """
    padded_d_model = 1 << (size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def kw_strides(x: Array | None, *stride_names: str):
    """Generate stride keyword arguments for kernel calls.

    Creates a dictionary mapping stride names to their corresponding values
    for use as keyword arguments in kernel invocations.

    Args:
        x: JAX array to get strides from, or None for zero strides.
        *stride_names: Names for each dimension's stride.

    Returns:
        Dictionary mapping "stride_{name}" to stride values.

    Example:
        >>> arr = jnp.ones((2, 3, 4))
        >>> kw_strides(arr, 'batch', 'seq', 'head')
        {'stride_batch': 12, 'stride_seq': 4, 'stride_head': 1}
    """
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": get_stride(x, i) for i, s in enumerate(stride_names)}


def narrow(x, dim: int, start: int, length: int):
    """Narrow a tensor along a specific dimension.

    Extracts a contiguous slice of length `length` starting at `start`
    along the specified dimension, similar to PyTorch's narrow operation.

    Args:
        x: Input array to narrow.
        dim: Dimension along which to narrow.
        start: Starting index of the slice.
        length: Length of the slice to extract.

    Returns:
        Narrowed array with reduced size along the specified dimension.

    Example:
        >>> x = jnp.arange(20).reshape(4, 5)
        >>> narrow(x, dim=1, start=1, length=3).shape
        (4, 3)
    """
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start, start + length)
    return x[tuple(slices)]


def get_input_shapes() -> list[tuple[int, int, int, int, int, int]]:
    """Generate test input shapes for benchmarking and testing.

    Creates a list of input shape configurations with varying batch sizes
    and sequence lengths for comprehensive kernel testing. Batch size
    decreases as sequence length increases to keep total work roughly
    constant.

    Returns:
        List of 6-tuples ``(batch, num_groups, seq_len, heads, gqa_ratio, head_dim)``
        covering sequence lengths from 256 to 131072 with GQA ratios of 1 and 2.
    """
    cases = [(max(1, 2 ** (16 - i)), 1, 2**i, 16, 1, 128) for i in range(8, 18)] + [
        (max(1, 2 ** (16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)
    ]
    return cases


@functools.cache
def is_hip():
    """Check if running on AMD HIP backend.

    Returns:
        True if the current Triton target uses HIP backend, False otherwise.
    """
    if triton is None:
        return False
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


@functools.cache
def is_cdna():
    """Check if running on AMD CDNA architecture.

    CDNA (Compute DNA) architectures include MI100, MI200 series GPUs.

    Returns:
        True if running on CDNA architecture (gfx940, gfx941, etc.), False otherwise.
    """
    if triton is None:
        return False
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in CDNA_ARCHS
    except Exception:
        return False


@functools.cache
def is_rdna():
    """Check if running on AMD RDNA architecture.

    RDNA (Radeon DNA) architectures include RX 6000, 7000 series GPUs.

    Returns:
        True if running on RDNA architecture (gfx1030, gfx1100, etc.), False otherwise.
    """
    if triton is None:
        return False
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in RDNA_ARCHS
    except Exception:
        return False


def calculate_blocksize_and_wraps(n):
    """Calculate optimal block size and number of warps for Triton kernels.

    Determines the appropriate block size (as power of 2) and number of warps
    based on the input size, with architecture-specific adjustments for HIP.

    Args:
        n: Input size to calculate block configuration for.

    Returns:
        Tuple of (block_size, num_warps) optimized for the input size.

    Raises:
        RuntimeError: If the required block size exceeds MAX_FUSED_SIZE (65536).

    Example:
        >>> calculate_blocksize_and_wraps(1024)
        (1024, 4)
        >>> calculate_blocksize_and_wraps(10000)
        (16384, 16)
    """
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError()
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def numeric_gen(*shape, dtype: str | jnp.dtype = jnp.float16, method: str = "normal"):
    """Generate random numeric arrays for testing and debugging.

    Creates random arrays using JAX's random number generation with a global
    debug RNG state for reproducibility.

    Args:
        *shape: Dimensions of the array to generate.
        dtype: Data type of the generated array. Defaults to float16.
        method: Random generation method from jax.random. Defaults to "normal".

    Returns:
        Random JAX array with specified shape and dtype.

    Raises:
        AssertionError: If the specified method is not available in jax.random.

    Example:
        >>> arr = numeric_gen(2, 3, 4, dtype=jnp.float32, method="uniform")
        >>> arr.shape
        (2, 3, 4)
    """
    global DEBUG_GLOBAL_RNG
    if DEBUG_GLOBAL_RNG is None:
        DEBUG_GLOBAL_RNG = jax.random.PRNGKey(0)
    DEBUG_GLOBAL_RNG, key = jax.random.split(DEBUG_GLOBAL_RNG, 2)
    method = getattr(jax.random, method, None)
    assert method is not None, "unsupported method in `jax.random`."
    return method(key=key, shape=shape, dtype=dtype)


def random_dense(
    *shape,
    dtype: str | jnp.dtype = jnp.float16,
    limit: int | None = 1,
) -> jnp.ndarray:
    """Generate a random dense array with uniform distribution.

    Creates a random array with values uniformly distributed in [-limit, limit],
    optionally casting through bfloat16 for numerical stability.

    Args:
        *shape: Dimensions of the array to generate.
        dtype: Output data type. Defaults to float16.
        limit: Maximum absolute value. If None, defaults to 1/prod(shape).

    Returns:
        Random JAX array with specified shape and dtype.

    Example:
        >>> arr = random_dense(2, 3, 4, dtype=jnp.float32)
        >>> arr.shape
        (2, 3, 4)
    """
    global DEBUG_GLOBAL_RNG
    if DEBUG_GLOBAL_RNG is None:
        DEBUG_GLOBAL_RNG = jax.random.PRNGKey(0)
    DEBUG_GLOBAL_RNG, key = jax.random.split(DEBUG_GLOBAL_RNG, 2)
    if limit is None:
        limit = 1 / np.prod(shape)
    x = jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return x.astype(jnp.bfloat16).astype(dtype)


def get_abs_err(x, y):
    """Calculate maximum absolute error between two arrays.

    Warning:
        This function uses PyTorch-style tensor methods (``.detach()``,
        ``.flatten()``, ``.abs()``, ``.max()``, ``.item()``). It only works
        with PyTorch tensors — passing JAX arrays will raise an
        ``AttributeError``. This appears to be a legacy helper retained for
        PyTorch-based test scripts.

    Args:
        x: First PyTorch tensor.
        y: Second PyTorch tensor.

    Returns:
        Maximum absolute difference as a Python float.
    """
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    """Calculate relative error ratio between two arrays.

    Computes the root mean square error (RMSE) between ``x`` and ``y``,
    normalized by the RMS magnitude of ``x`` (the reference).

    Warning:
        This function uses PyTorch-style tensor methods (``.detach()``,
        ``.flatten()``, ``.square()``, ``.mean()``, ``.sqrt()``, ``.item()``).
        It only works with PyTorch tensors — passing JAX arrays will raise an
        ``AttributeError``. This appears to be a legacy helper retained for
        PyTorch-based test scripts.

    Args:
        x: Reference PyTorch tensor.
        y: PyTorch tensor to compare against the reference.

    Returns:
        Relative error ratio ``RMSE(x, y) / (RMS(x) + 1e-8)`` as a Python float.
    """
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    """Assert that two arrays are numerically close within tolerance.

    Compares arrays using both absolute error and a relative error ratio.
    If the absolute error is below ``err_atol`` the check passes unconditionally.
    Otherwise the error ratio (from :func:`get_err_ratio`) is compared against
    ``ratio``.

    Warning:
        Internally calls :func:`get_abs_err` and :func:`get_err_ratio`, which
        use PyTorch-style tensor methods. This function only works with PyTorch
        tensors, not JAX arrays.

    The pass/fail logic is:
        1. If ``abs_atol <= err_atol`` → pass (print message, return).
        2. If ``warning=True`` **or** ``error_rate < 0.01`` **or** ``abs_atol <= 0.3``:
           - If ``error_rate > ratio`` → emit a ``warnings.warn``.
        3. Otherwise → ``assert error_rate < ratio``.

    Args:
        prefix: String prefix shown in the diagnostic message.
        ref: Reference PyTorch tensor.
        tri: PyTorch tensor to test against the reference.
        ratio: Maximum allowed relative error ratio.
        warning: If ``True``, failures emit a ``UserWarning`` instead of
            raising ``AssertionError``. Defaults to ``False``.
        err_atol: Absolute error threshold below which the check always passes.
            Defaults to 1e-6.

    Raises:
        AssertionError: If ``error_rate >= ratio`` and ``warning=False`` and
            neither of the soft-pass conditions is met.

    Example:
        >>> ref = jnp.ones((10,))
        >>> test = ref + 1e-7
        >>> assert_close("Test", ref, test, ratio=0.01)
    """
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (error_rate < 0.01 or abs_atol <= 0.3):
        if error_rate > ratio:
            import warnings

            warnings.warn(msg, stacklevel=1)
    else:
        assert error_rate < ratio, msg


def is_fp8(x):
    """Check if an array uses FP8 dtype and if hardware supports it.

    Args:
        x: Array to check for FP8 dtype.

    Returns:
        True if array is FP8 and hardware supports it, False if not FP8.

    Raises:
        RuntimeError: If array is FP8 but hardware doesn't support it.
    """
    if x.dtype in {jnp.float8_e4m3fnuz, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float8_e5m2fnuz}:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError("This device does not support fp8")
    else:
        return False


@functools.cache
def get_gpu_arch() -> str:
    """Get the architecture identifier string of the current GPU.

    Queries the Triton runtime driver for the active GPU target architecture.
    Results are cached for the lifetime of the process.

    Returns:
        Architecture identifier string (e.g., "gfx942", "sm_80"), or an
        empty string if Triton is not available or the query fails.
    """
    if triton is None:
        return ""
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return ""


def arch_supports_fp8() -> bool:
    """Check if the current GPU architecture supports FP8 arithmetic.

    Currently only AMD MI300 series (gfx942) on the HIP backend provides
    hardware FP8 support. This function is used as a guard before invoking
    FP8-specific kernel paths.

    Returns:
        True if running on AMD HIP with gfx942 architecture, False otherwise.
    """
    return is_hip() and get_gpu_arch() in ("gfx942")


def generate_block_indices(
    batch: int,
    num_query_blocks: int,
    heads: int,
    selected_blocks: int,
    block_size: int,
    seed: int = 42,
) -> jax.Array:
    """Generate random block indices for sparse attention benchmarks.

    This function generates a tensor of block indices where each token attends to
    a random selection of previous key blocks. The indices are sorted in ascending order.
    Returns per-token format: each token in a query block gets the same block indices.

    Args:
        batch: Batch size.
        num_query_blocks: Number of query blocks.
        heads: Number of attention heads (typically kv_heads for GQA).
        selected_blocks: Number of key blocks each query block should attend to.
        block_size: Size of each block.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (batch, seq_len, heads, selected_blocks) containing sorted
        block indices in per-token format. Positions beyond available blocks are filled with -1.

    Example:
        >>>
        >>> indices = generate_block_indices(batch=2, num_query_blocks=4, heads=8, selected_blocks=2, block_size=64)
        >>> indices.shape
        (2, 256, 8, 2)
    """
    seq_len = num_query_blocks * block_size
    rng = np.random.default_rng(seed)
    block_indices = np.full((batch, seq_len, heads, selected_blocks), -1, dtype=np.int32)

    for b in range(batch):
        for qb in range(num_query_blocks):
            num_available_blocks = qb + 1
            token_start = qb * block_size
            token_end = (qb + 1) * block_size
            for h in range(heads):
                perm = rng.permutation(num_available_blocks)[:selected_blocks]
                if num_available_blocks < selected_blocks:
                    perm = np.pad(perm, (0, selected_blocks - num_available_blocks), constant_values=-1)
                perm_sorted = np.sort(perm)
                block_indices[b, token_start:token_end, h, :] = perm_sorted

    return jnp.asarray(block_indices)


_sync_counter = 0
"""Global counter for generating unique barrier names in :func:`barrier_sync`."""


def barrier_sync(timeout: float = 200):
    """Synchronize all JAX processes at a barrier point.

    Blocks execution until all processes in the distributed JAX runtime reach
    this barrier. This is essential for ensuring consistency across distributed
    training, especially before/after collective operations or checkpointing.

    The function uses a global counter to create unique barrier names, allowing
    multiple barriers to be used sequentially without conflicts.

    Args:
        timeout: Maximum time to wait for all processes to reach the barrier,
            in seconds. Defaults to 200 seconds (3.33 minutes). If the timeout
            is exceeded, a RuntimeError will be raised by the underlying JAX
            distributed client.

    Returns:
        None

    Raises:
        RuntimeError: If the JAX distributed client is not initialized. This
            typically means JAX was not started in distributed mode or the
            distributed runtime failed to initialize.

    Note:
        - This function is a no-op when running with a single process
          (jax.process_count() == 1), allowing code to work seamlessly
          in both single and multi-process environments.
        - Each call increments a global counter to ensure unique barrier names,
          preventing conflicts when multiple barriers are used in sequence.
        - The timeout is converted to milliseconds for the underlying JAX API.

    Example:
        >>>
        >>> model = train_step(model, batch)
        >>> barrier_sync()
        >>> if jax.process_index() == 0:
        ...     save_checkpoint(model)
        >>> barrier_sync()

        >>>
        >>> barrier_sync(timeout=600)

    Warning:
        Ensure all processes call barrier_sync() the same number of times and
        in the same order, or deadlocks may occur. Conditional barriers based
        on process rank should be avoided.
    """
    global _sync_counter
    if jax.process_count() == 1:
        return
    import jax._src.distributed as distributed

    client = distributed.global_state.client

    if client is None:
        raise RuntimeError("barrier_sync requires jax distributed client to be initialized")

    _sync_counter += 1
    client.wait_at_barrier(f"easy_barrier_sync_{_sync_counter}", timeout_in_ms=int(timeout * 1000.0))


def _align_to(x: int, multiple: int) -> int:
    """Round up a value to the nearest multiple.

    Args:
        x: Value to align.
        multiple: Alignment boundary.

    Returns:
        Smallest value >= x that is divisible by multiple.
    """
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Calculate lanes per 32-bit slot for dtype packing.

    Determines how many elements of the given dtype fit into a 32-bit slot,
    used for memory-efficient paged attention layouts.

    Args:
        dtype: JAX/NumPy dtype to calculate packing for.

    Returns:
        Number of lanes per 32-bit slot (2 for fp16/bf16, 1 for fp32).

    Raises:
        ValueError: If dtype is not 16-bit or 32-bit float.
    """
    bw = jnp.dtype(dtype).itemsize * 8
    if bw not in (16, 32):
        raise ValueError(f"Only 16/32-bit floats supported for packing, got {dtype} ({bw} bits).")
    return 32 // bw


def make_dummy_rpa_inputs(
    *,
    rng_seed: int = 0,
    num_seqs: int = 4,
    pages_per_seq: int = 3,
    page_size: int = 16,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 80,
    kv_dtype: jnp.dtype = jnp.float32,
    q_dtype: jnp.dtype | None = None,
    kv_len_max: int | None = None,
    total_q: int | None = None,
    total_num_pages: int | None = None,
    decode_prefill_mixed: tuple[int, int, int] | None = None,
):
    """Generate dummy inputs for Ragged Paged Attention testing and benchmarking.

    Creates a complete set of test inputs including queries, keys, values,
    kv_cache, and metadata tensors that satisfy all kernel constraints.

    Args:
        rng_seed: Random seed for reproducibility.
        num_seqs: Number of sequences in the batch.
        pages_per_seq: Number of pages allocated per sequence (padded width).
        page_size: Number of tokens per page.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads.
        head_dim: Dimension of each attention head (will be padded to 128).
        kv_dtype: Data type for keys and values (16/32-bit float).
        q_dtype: Data type for queries (defaults to kv_dtype if None).
        kv_len_max: Maximum KV length per sequence (defaults to pages_per_seq * page_size).
        total_q: Total query tokens across all sequences. If set, uses deterministic lengths.
        total_num_pages: Physical kv_cache pages (can be < num_seqs * pages_per_seq).
        decode_prefill_mixed: Tuple (decode_end, prefill_end, total) for mixed batching.

    Returns:
        Dictionary containing:
            - queries: (sum_q, num_q_heads, head_dim) [q_dtype]
            - keys, values: (sum_q, num_kv_heads, head_dim) [kv_dtype]
            - kv_cache: (total_pages, page_size, x2_per_pack, pack, head_dim_aligned) [kv_dtype]
            - kv_lens: (num_seqs,) [int32]
            - block_tables: (num_seqs * pages_per_seq,) [int32]
            - query_start_loc: (num_seqs + 1,) [int32]
            - distribution: (3,) [int32]
            - _meta: Metadata dict for debugging

    Raises:
        ValueError: If total_q is invalid, page_size is not divisible by packing,
            or total_num_pages is insufficient for the given kv_lens.

    Example:
        >>> inputs = make_dummy_rpa_inputs(
        ...     total_q=1024, num_q_heads=8, head_dim=128,
        ...     num_kv_heads=4, kv_dtype=jnp.bfloat16,
        ...     page_size=64, pages_per_seq=16, total_num_pages=2**17
        ... )
        >>> inputs['queries'].shape
        (1024, 8, 128)

    Notes:
        - pages_per_seq is treated as the (padded) page-table width
        - If total_num_pages < num_seqs * pages_per_seq, the page table is padded
          with safe in-bounds indices
    """
    if q_dtype is None:
        q_dtype = kv_dtype

    pack = _dtype_packing(kv_dtype)
    if page_size % pack != 0:
        raise ValueError(f"page_size ({page_size}) must be divisible by packing ({pack}).")

    head_dim_aligned = _align_to(head_dim, 128)
    kv_len_cap = pages_per_seq * page_size
    if kv_len_max is None:
        kv_len_max = kv_len_cap
    kv_len_max = min(kv_len_max, kv_len_cap)

    key = jax.random.PRNGKey(rng_seed)
    key_kv, key_q, key_data = jax.random.split(key, 3)

    if total_q is None:
        kv_lens = jax.random.randint(key_kv, (num_seqs,), minval=1, maxval=kv_len_max + 1, dtype=jnp.int32)
        rnd = jax.random.uniform(key_q, (num_seqs,), minval=0.0, maxval=1.0)
        q_lens = jnp.maximum(1, jnp.ceil(rnd * kv_lens).astype(jnp.int32))
    else:
        if total_q < num_seqs:
            raise ValueError(f"total_q ({total_q}) must be >= num_seqs ({num_seqs}) so each sequence has >= 1 token.")
        if total_q > num_seqs * kv_len_max:
            raise ValueError(
                f"total_q ({total_q}) must be <= num_seqs*kv_len_max ({num_seqs}*{kv_len_max}) "
                "to satisfy q_len <= kv_len <= kv_len_max per sequence."
            )
        base = total_q // num_seqs
        rem = total_q % num_seqs
        q_lens = jnp.full((num_seqs,), base, dtype=jnp.int32)
        if rem:
            q_lens = q_lens.at[:rem].add(jnp.int32(1))
        kv_lens = q_lens

    query_start_loc = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(q_lens, dtype=jnp.int32)])
    sum_q = int(query_start_loc[-1])

    if decode_prefill_mixed is None:
        distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    else:
        i, j, k = decode_prefill_mixed
        if not (0 <= i <= j <= k == num_seqs):
            raise ValueError("distribution must satisfy 0 <= i <= j <= k == num_seqs.")
        distribution = jnp.array([i, j, k], dtype=jnp.int32)

    pages_needed = (kv_lens + jnp.int32(page_size) - jnp.int32(1)) // jnp.int32(page_size)
    pages_needed_np = np.asarray(pages_needed, dtype=np.int32)
    total_pages_needed = int(pages_needed_np.sum())
    max_pages_needed = int(pages_needed_np.max())

    total_pages_used_full = num_seqs * pages_per_seq

    if total_num_pages is None:
        total_pages = total_pages_used_full
        total_pages_used = total_pages_used_full
        seq_bases = jnp.arange(num_seqs, dtype=jnp.int32) * jnp.int32(pages_per_seq)
        per_seq = (seq_bases[:, None] + jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :]).reshape(-1)
        block_tables = per_seq
    else:
        total_pages = int(total_num_pages)
        if total_pages >= total_pages_used_full:
            total_pages_used = total_pages_used_full
            seq_bases = jnp.arange(num_seqs, dtype=jnp.int32) * jnp.int32(pages_per_seq)
            per_seq = (seq_bases[:, None] + jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :]).reshape(-1)
            block_tables = per_seq
        else:
            if total_pages < total_pages_needed:
                raise ValueError(
                    f"total_num_pages ({total_pages}) must be >= total_pages_needed ({total_pages_needed}) "
                    "given kv_lens and page_size."
                )
            if max_pages_needed > pages_per_seq:
                raise ValueError(
                    f"pages_per_seq ({pages_per_seq}) must be >= max(ceil(kv_len/page_size)) ({max_pages_needed})."
                )
            needs_padding = bool((pages_needed_np < pages_per_seq).any())
            if needs_padding and total_pages == total_pages_needed:
                raise ValueError(
                    "total_num_pages must be > total_pages_needed when pages_per_seq is a padded capacity; "
                    "increase total_num_pages or reduce kv_lens/pages_per_seq."
                )
            pad_page = np.int32(total_pages_needed) if needs_padding else np.int32(0)

            block_tables_2d = np.full((num_seqs, pages_per_seq), pad_page, dtype=np.int32)
            cursor = 0
            for seq_idx, pn in enumerate(pages_needed_np.tolist()):
                pn_i = int(pn)
                if pn_i <= 0:
                    continue
                block_tables_2d[seq_idx, :pn_i] = np.arange(cursor, cursor + pn_i, dtype=np.int32)
                cursor += pn_i
            total_pages_used = cursor
            block_tables = jnp.asarray(block_tables_2d.reshape(-1), dtype=jnp.int32)

    kv_cache_shape = (
        total_pages,
        page_size,
        _align_to(num_kv_heads * 2, pack) // pack,
        pack,
        head_dim_aligned,
    )
    kcache = jax.random.normal(key_data, kv_cache_shape, dtype=kv_dtype)
    key_qry, key_key, key_val = jax.random.split(jax.random.PRNGKey(rng_seed ^ 0xC0FFEE), 3)
    queries = jax.random.normal(key_qry, (sum_q, num_q_heads, head_dim), dtype=q_dtype)
    keys = jax.random.normal(key_key, (sum_q, num_kv_heads, head_dim), dtype=kv_dtype)
    values = jax.random.normal(key_val, (sum_q, num_kv_heads, head_dim), dtype=kv_dtype)

    return dict(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kcache,
        kv_lens=kv_lens.astype(jnp.int32),
        block_tables=block_tables.astype(jnp.int32),
        query_start_loc=query_start_loc.astype(jnp.int32),
        distribution=distribution.astype(jnp.int32),
        _meta=dict(
            num_seqs=num_seqs,
            pages_per_seq=pages_per_seq,
            page_size=page_size,
            total_pages=total_pages,
            total_pages_used=total_pages_used,
            q_lens=q_lens,
            kv_lens=kv_lens,
            head_dim=head_dim,
            head_dim_aligned=head_dim_aligned,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            packing=pack,
            dtypes=dict(q=q_dtype, kv=kv_dtype),
        ),
    )


def get_tpu_generation() -> int:
    """Detect and return the current TPU generation.

    Queries JAX devices to determine the TPU generation (v3, v4, v5, etc.)
    by parsing the device_kind string.

    Returns:
        Integer representing TPU generation (3, 4, 5, etc.).
        Returns 0 if no TPU is detected or generation cannot be determined.

    Example:
        >>> gen = get_tpu_generation()
        >>> if gen >= 4:
        ...     print("Using TPU v4+ optimizations")
    """
    try:
        devices = jax.devices("tpu")

        if not devices:
            return 0
        device_kind = devices[0].device_kind
        match = re.search(r"v(\d+)", device_kind)

        if match:
            return int(match.group(1))

        return 0

    except (RuntimeError, IndexError):
        return 0


def make_mesh(mesh_axis: tuple[int, int, int, int]):
    """Create a JAX mesh with standard sharding axes.

    Creates a device mesh with axes named for data parallelism (dp),
    fully-sharded data parallelism (fsdp), tensor parallelism (tp),
    and sequence parallelism (sp).

    Args:
        mesh_axis: Tuple of (dp, fsdp, tp, sp) axis sizes.

    Returns:
        JAX Mesh with named axes ("dp", "fsdp", "tp", "sp").

    Example:
        >>> mesh = make_mesh((2, 1, 4, 1))  # 2 data parallel, 4 tensor parallel
    """
    return jax.make_mesh(mesh_axis, ("dp", "fsdp", "tp", "sp"))


def get_qkv_shardings(layout: Literal["bhsd", "bshd", "thd"]):
    """Get sharding specifications for attention QKV tensors based on layout.

    Returns PartitionSpecs for queries, keys, and values that are compatible
    with the given tensor layout and a standard (dp, fsdp, tp, sp) mesh created
    by :func:`make_mesh`.

    The returned specs come in two groups of three:
    - Non-sequence-parallel specs (``q_spec, k_spec, v_spec``): no ``sp`` axis.
    - Sequence-parallel specs (``sq_spec, sk_spec, sv_spec``): adds the ``sp``
      axis to the sequence/token dimension.

    Args:
        layout: Tensor layout format string:
            - ``"bhsd"``: tensors have shape ``[batch, heads, seq, dim]``.
              Batch+fsdp on axis 0, tp on heads (axis 1).
            - ``"bshd"``: tensors have shape ``[batch, seq, heads, dim]``.
              Batch+fsdp on axis 0, tp on heads (axis 2).
            - ``"thd"``: packed/ragged tensors. The PartitionSpec has four axes
              ``(None, tp, (dp, fsdp), None)``; the exact dimension ordering
              used in practice should be confirmed against the calling site.

    Returns:
        Tuple of 6 ``PartitionSpec`` objects:
        ``(q_spec, k_spec, v_spec, sq_spec, sk_spec, sv_spec)``.
        The first three are non-sequence-parallel; the last three include the
        ``sp`` axis for sequence-parallel training.

    Raises:
        ValueError: If ``layout`` is not one of ``"bhsd"``, ``"bshd"``, ``"thd"``.
    """
    if layout == "bhsd":
        qps = Ps(("dp", "fsdp"), "tp", None, None)
        kps = Ps(("dp", "fsdp"), "tp", None, None)
        vps = Ps(("dp", "fsdp"), "tp", None, None)

        sqps = Ps(("dp", "fsdp"), "tp", "sp", None)
        skps = Ps(("dp", "fsdp"), "tp", "sp", None)
        svps = Ps(("dp", "fsdp"), "tp", "sp", None)
    elif layout == "bshd":
        qps = Ps(("dp", "fsdp"), None, "tp", None)
        kps = Ps(("dp", "fsdp"), None, "tp", None)
        vps = Ps(("dp", "fsdp"), None, "tp", None)

        sqps = Ps(("dp", "fsdp"), "sp", "tp", None)
        skps = Ps(("dp", "fsdp"), "sp", "tp", None)
        svps = Ps(("dp", "fsdp"), "sp", "tp", None)
    elif layout == "thd":
        qps = Ps(None, "tp", ("dp", "fsdp"), None)
        kps = Ps(None, "tp", ("dp", "fsdp"), None)
        vps = Ps(None, "tp", ("dp", "fsdp"), None)

        sqps = Ps("sp", "tp", ("dp", "fsdp"), None)
        skps = Ps("sp", "tp", ("dp", "fsdp"), None)
        svps = Ps("sp", "tp", ("dp", "fsdp"), None)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    return qps, kps, vps, sqps, skps, svps


def get_segments_shardings():
    """Get sharding specifications for segment ID tensors.

    Returns PartitionSpecs for query and key/value segment IDs,
    compatible with a standard (dp, fsdp, tp, sp) mesh.

    Returns:
        Tuple of 4 PartitionSpecs: (q_spec, kv_spec, sq_spec, skv_spec)
        where the 's' prefix indicates sequence-parallel variants.
    """
    qps = Ps(("dp", "fsdp"), None)
    kvps = Ps(("dp", "fsdp"), None)
    sqps = Ps(("dp", "fsdp"), None)
    skvps = Ps(("dp", "fsdp"), "sp")
    return qps, kvps, sqps, skvps
