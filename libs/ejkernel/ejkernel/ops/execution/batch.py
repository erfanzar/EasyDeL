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


"""Batch processing utilities for vectorized and parallel execution.

This module provides utilities for efficiently executing kernels over batched
data using JAX's vmap and pmap transformations while maintaining the benefits
of automatic configuration selection.

Key Functions:
    vmap_with_config: Vectorized execution with shared configuration selection
    pmap_with_config: Parallel execution across devices with shared configuration

The Challenge:
    JAX's vmap and pmap transformations apply functions element-wise across
    batch dimensions. However, configuration selection can be expensive and
    shouldn't be repeated for every element in a batch.

The Solution:
    These utilities select a configuration once using a representative sample
    (typically the first element), then apply that configuration to all elements
    in the batch using the appropriate JAX transformation.

Benefits:
    - Amortized configuration selection cost across batch elements
    - Consistent configuration for all elements in a batch
    - Full compatibility with JAX transformations
    - Automatic handling of different input axis specifications
    - Support for both CPU vectorization (vmap) and multi-device parallelism (pmap)

Example Usage:
    >>>
    >>> vmapped_fn = vmap_with_config(executor, kernel, in_axes=0)
    >>> batch_result = vmapped_fn(batch_input)
    >>>
    >>>
    >>> pmapped_fn = pmap_with_config(executor, kernel, in_axes=0)
    >>> device_result = pmapped_fn(device_sharded_input)

Note:
    Configuration selection uses the first element along the specified axis
    as a representative sample. This assumes that optimal configuration is
    consistent across the batch, which is typically true for homogeneous data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax


def vmap_with_config(executor, kernel, in_axes=0) -> Callable[..., Any]:
    """Vectorized execution with shared configuration selection.

    Creates a vectorized version of kernel execution where configuration is
    selected once using a representative sample, then applied to all elements
    in the batch via jax.vmap.

    This approach significantly reduces overhead compared to selecting configuration
    for each batch element individually, while maintaining optimal performance.

    Args:
        executor: Executor instance for running the kernel
        kernel: Kernel to execute vectorially
        in_axes: Input axes specification for vmap (default: 0)
            - int: Same axis for all arguments
            - tuple/list: Per-argument axis specification
            - None: Broadcast argument (no vectorization)

    Returns:
        Function that performs vectorized execution with shared config selection

    Example:
        >>>
        >>> cache = ConfigCache()
        >>> selector = ConfigSelectorChain(cache)
        >>> executor = Executor(selector)
        >>>
        >>>
        >>> vmapped_matmul = vmap_with_config(executor, matmul_kernel, in_axes=0)
        >>>
        >>>
        >>> batch_x = jnp.array([x1, x2, x3, ...])
        >>> batch_y = jnp.array([y1, y2, y3, ...])
        >>> batch_result = vmapped_matmul(batch_x, batch_y)

    Note:
        The representative sample is obtained by taking the first element
        along each specified axis. This sample is used for configuration
        selection but not included in the final batch computation.
    """

    def wrapped(*args, **kwargs):
        """Select a configuration using a representative sample, then vmap over the batch."""

        def slice0(x, axis):
            """Return the first element of ``x`` along ``axis``, or ``x`` unchanged."""
            if axis is None or not isinstance(x, jax.Array):
                return x
            return jax.lax.index_in_dim(x, 0, axis, keepdims=False)

        in_axes_tree = jax.tree.map(lambda _: in_axes, args) if not isinstance(in_axes, tuple | list) else in_axes

        sample = jax.tree.map(slice0, args, in_axes_tree)

        _ = executor(kernel, *sample, stamp=False, **kwargs)

        def fn(*a, **k):
            """Execute ``kernel`` via ``executor`` — vectorized by ``jax.vmap``."""
            return executor(kernel, *a, **k)

        return jax.vmap(fn, in_axes=in_axes)(*args, **kwargs)

    return wrapped


def pmap_with_config(executor, kernel, in_axes=0, axis_name="devices"):
    """Parallel execution across devices with shared configuration selection.

    Creates a parallel version of kernel execution for multi-device computation
    where configuration is selected once using data from the first device, then
    applied to all devices via jax.pmap.

    This enables efficient multi-device execution while avoiding redundant
    configuration selection on each device.

    Args:
        executor: Executor instance for running the kernel
        kernel: Kernel to execute in parallel across devices
        in_axes: Input axes specification for pmap (default: 0)
            - int: Same axis for all arguments (typically device axis)
            - tuple/list: Per-argument axis specification
            - None: Broadcast argument to all devices
        axis_name: Name for the parallel axis (default: "devices")
            Used for collective operations and debugging

    Returns:
        Function that performs parallel execution with shared config selection

    Example:
        >>>
        >>> cache = ConfigCache()
        >>> selector = ConfigSelectorChain(cache)
        >>> executor = Executor(selector)
        >>>
        >>>
        >>> pmapped_matmul = pmap_with_config(executor, matmul_kernel, in_axes=0)
        >>>
        >>>
        >>> devices = jax.devices()
        >>> x_sharded = jax.device_put_sharded([x1, x2, x3, x4], devices)
        >>> y_sharded = jax.device_put_sharded([y1, y2, y3, y4], devices)
        >>>
        >>>
        >>> result_sharded = pmapped_matmul(x_sharded, y_sharded)

    Note:
        Configuration selection uses data from the first device (index 0)
        as a representative sample. This assumes optimal configuration is
        consistent across devices, which is typically true for homogeneous
        hardware setups.
    """

    def wrapped(*args, **kwargs):
        """Select a configuration using device-0 data, then pmap over all devices."""

        local_args = jax.tree.map(lambda x: x[0] if isinstance(x, jax.Array) else x, args)

        _ = executor(kernel, *local_args, stamp=False, **kwargs)

        def fn(*a, **k):
            """Execute ``kernel`` via ``executor`` — parallelised by ``jax.pmap``."""
            return executor(kernel, *a, **k)

        return jax.pmap(fn, in_axes=in_axes, axis_name=axis_name)(*args, **kwargs)

    return wrapped
