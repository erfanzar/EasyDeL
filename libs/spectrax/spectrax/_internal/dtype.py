# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dtype utilities compatible with eformer APIs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

STRING_TO_DTYPE_MAP = {
    "bf16": jnp.bfloat16,
    "bfloat16": jnp.bfloat16,
    "fp16": jnp.float16,
    "float16": jnp.float16,
    "fp32": jnp.float32,
    "float32": jnp.float32,
    "fp64": jnp.float64,
    "float64": jnp.float64,
    "fp8": jnp.float8_e5m2,
    "fp8_e4m3fn": jnp.float8_e4m3fn,
    "fp8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "fp8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "fp8_e5m2": jnp.float8_e5m2,
    "fp8_e5m2fnuz": jnp.float8_e5m2fnuz,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e4m3fnuz": jnp.float8_e4m3fnuz,
    "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "float8_e5m2": jnp.float8_e5m2,
    "float8_e5m2fnuz": jnp.float8_e5m2fnuz,
}


DTYPE_TO_STRING_MAP = {
    jnp.bfloat16: "bf16",
    jnp.float16: "fp16",
    jnp.float32: "fp32",
    jnp.float64: "fp64",
    jnp.float8_e5m2: "fp8",
    jnp.float8_e4m3fn: "fp8_e4m3fn",
    jnp.float8_e4m3fnuz: "fp8_e4m3fnuz",
    jnp.float8_e4m3b11fnuz: "fp8_e4m3b11fnuz",
    jnp.float8_e5m2: "fp8_e5m2",
    jnp.float8_e5m2fnuz: "fp8_e5m2fnuz",
}


DTYPE_MAPPING = {
    "bf16": jnp.bfloat16,
    "f16": jnp.float16,
    "f32": jnp.float32,
    "f64": jnp.float64,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
    "float64": jnp.float64,
    "f8_e4m3": jnp.float8_e4m3fn,
    "f8_e5m2": jnp.float8_e5m2,
    "float8_e4m3": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
}


def put_dtype(array: jax.Array, dtype: str | jnp.dtype | None) -> jax.Array:
    """Convert a JAX array to the specified data type.

    Args:
        array: The input JAX array.
        dtype: Target dtype as string or jnp.dtype, or None to return unchanged.

    Returns:
        The array cast to the requested dtype, or the original array if
        dtype is None or the array is not a standard floating-point type.
    """
    if not dtype:
        return array

    if isinstance(dtype, str):
        try:
            dtype = STRING_TO_DTYPE_MAP[dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype string: {dtype}") from e

    if array.dtype in (jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64):
        return array.astype(dtype)
    return array
