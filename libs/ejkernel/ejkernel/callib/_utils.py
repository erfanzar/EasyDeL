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


"""Utility functions for the callib module.

This module provides common utility functions used across the callib package,
including mathematical operations, array shape calculations, system utilities,
and environment configuration helpers.

Key Functions:
    - cdiv: Ceiling division for integers and JAX arrays
    - strides_from_shape: Calculate strides for contiguous arrays
    - next_power_of_2: Find next power of 2
    - get_cache_dir: Get platform-specific cache directory
    - quiet: Context manager for suppressing output
    - check_bool_flag: Parse boolean environment variables

Protocols:
    - ShapeDtype: Protocol for array-like objects with shape and dtype
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, overload

import jax
import jax.dlpack
import numpy as np


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
    """Calculate the strides for a contiguous array with the given shape.

    Computes the number of elements to skip in memory to advance by one
    position along each dimension, assuming row-major (C-style) layout.

    Args:
        shape: A tuple of integers representing the dimensions of an array.

    Returns:
        A tuple of integers representing the strides of a contiguous array.
        The stride for dimension i is the product of all dimensions after i.

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


class ShapeDtype(Protocol):
    """Protocol for objects that have shape and dtype attributes.

    This protocol defines the interface for array-like objects that provide
    shape and dtype information, commonly used in tensor operations.
    Compatible types include ``jax.ShapeDtypeStruct``, ``jax.Array``,
    and ``numpy.ndarray``.

    Example:
        >>> def process(x: ShapeDtype) -> None:
        ...     print(f"Shape: {x.shape}, Dtype: {x.dtype}")
        >>> process(jax.ShapeDtypeStruct((4, 8), jnp.float32))
        Shape: (4, 8), Dtype: float32
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensions of the array-like object.

        Returns:
            Tuple of integers representing the size of each dimension.
        """
        ...

    @property
    def dtype(self) -> np.dtype:
        """Return the data type of the array-like object.

        Returns:
            NumPy dtype describing the element type.
        """
        ...


def get_cache_dir() -> Path:
    """Get the EasyDeL cache directory.

    Returns the platform-specific cache directory for EasyDeL.
    Creates the directory if it doesn't exist.

    Returns:
        Path to the cache directory.

    Example:
        >>> cache_dir = get_cache_dir()
        >>> print(cache_dir)
        /home/user/.cache/ejkernel-cache
    """
    home_dir = Path.home()
    app_name = "ejkernel-cache"
    if os.name == "nt":
        cache_dir = Path(os.getenv("LOCALAPPDATA", home_dir / "AppData" / "Local")) / app_name
    elif os.name == "posix":
        if "darwin" in os.sys.platform:
            cache_dir = home_dir / "Library" / "Caches" / app_name
        else:
            cache_dir = home_dir / ".cache" / app_name
    else:
        cache_dir = home_dir / ".cache" / app_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class DummyStream:
    """A null device-like stream that discards all writes.

    Used for suppressing output by replacing stdout/stderr.
    All write and flush operations are no-ops.
    """

    def write(self, *args, **kwargs):
        """Discard all write operations.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        pass

    def flush(self, *args, **kwargs):
        """Discard all flush operations.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        pass


@contextmanager
def quiet(suppress_stdout=True, suppress_stderr=True):
    """Context manager to temporarily suppress stdout and/or stderr output.

    Replaces stdout/stderr with null streams to discard all output.
    Restores original streams on exit.

    Args:
        suppress_stdout: Whether to suppress stdout.
        suppress_stderr: Whether to suppress stderr.

    Yields:
        None

    Example:
        >>> with quiet():
        ...     print("This won't be displayed")
        ...     noisy_function()
        >>> print("This will be displayed")

    Note:
        This will suppress ALL output to the specified streams within
        the context, including output from C extensions and system calls.
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        if suppress_stdout:
            sys.stdout = DummyStream()
        if suppress_stderr:
            sys.stderr = DummyStream()
        yield

    finally:
        if suppress_stdout:
            sys.stdout = original_stdout
        if suppress_stderr:
            sys.stderr = original_stderr


def check_bool_flag(name: str, default: bool = True) -> bool:
    """Parse boolean environment variable.

    Interprets various string representations as boolean values.
    Accepts: 'true', 'yes', 'ok', '1', 'easy' (case-insensitive).

    Args:
        name: Environment variable name.
        default: Default value if variable not set.

    Returns:
        Boolean interpretation of the environment variable.

    Example:
        >>> os.environ['DEBUG'] = 'yes'
        >>> check_bool_flag('DEBUG')
        True
        >>> check_bool_flag('MISSING', default=False)
        False
    """
    default = "1" if default else "0"
    return str(os.getenv(name, default)).lower() in ["true", "yes", "ok", "1", "easy"]
