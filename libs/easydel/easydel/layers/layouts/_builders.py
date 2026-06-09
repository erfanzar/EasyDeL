# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Builder helpers for fused column-parallel transformer projections.

This module provides the canonical entry points to build the fused QKV and
fused gate/up :class:`ColumnParallelLinear` instances used by attention
and MLP blocks, plus the activation-time splitters that undo the fusion
for downstream consumers. Layouts are described declaratively by
:class:`FusedColumnLayout` (built via :func:`dense_qkv_layout` and
:func:`dense_gate_up_layout`) and carried on the linear's ``layout``
attribute so that checkpoint reform rules and runtime splits agree.

Public functions:
    :func:`build_fused_gate_up_projection`: Build a fused MLP gate/up
        projection backed by a dense gate/up layout.
    :func:`build_fused_qkv_projection`: Build a fused attention QKV
        projection backed by a dense Q/K/V layout.
    :func:`split_fused_gate_up_projection`: TP-aware split of a fused
        gate/up activation back into ``(gate, up)``.
    :func:`split_fused_qkv_projection`: TP-aware split of a fused QKV
        activation back into ``(q, k, v)`` with a contiguous fallback.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias, cast

import jax
import jax.numpy as jnp
import spectrax as spx
from jax.typing import DTypeLike

from easydel.layers.linears import ColumnParallelLinear

from ._dense import dense_gate_up_layout, dense_qkv_layout
from ._runtime import split_interleaved_pair_last_axis, split_interleaved_segments_last_axis

if TYPE_CHECKING:
    from easydel.infra.base_config import EasyDeLBaseConfig
    from easydel.infra.sharding import TensorLayout

Initializer: TypeAlias = Callable[[jax.Array, tuple[int, ...], DTypeLike], jax.Array]
ReformRule: TypeAlias = dict[str, object]
ReformParam: TypeAlias = dict[str, ReformRule]


def _default_kernel_init(config: EasyDeLBaseConfig) -> Initializer:
    """Return the canonical normal-initializer for fused linears.

    Mirrors the per-layer default used throughout EasyDeL when the caller
    does not supply ``kernel_init`` explicitly. The standard deviation is
    taken from ``config.initializer_range`` so that fused QKV / gate-up
    projections match the variance of the equivalent split layers.

    Args:
        config: Owning model config; only ``initializer_range`` is read.

    Returns:
        A :class:`jax.nn.initializers.Initializer` callable suitable for
        :class:`ColumnParallelLinear` ``kernel_init``.
    """
    return jax.nn.initializers.normal(config.initializer_range)


def build_fused_gate_up_projection(
    *,
    config: EasyDeLBaseConfig,
    intermediate_size: int,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    precision: jax.lax.PrecisionLike,
    rngs: spx.Rngs,
    use_bias: bool = False,
    target_prefix: str = "gate_up_proj",
    gate_prefix: str = "gate_proj",
    up_prefix: str = "up_proj",
    kernel_init: Initializer | None = None,
    sharding_layout: "TensorLayout | object | None" = None,
    bias_sharding_layout: "TensorLayout | object | None" = None,
) -> ColumnParallelLinear:
    """Build a column-parallel fused ``[gate | up]`` MLP projection.

    Creates a :class:`ColumnParallelLinear` whose output axis stores both
    halves of the SwiGLU pre-activation and attaches a
    :class:`FusedColumnLayout` so that downstream splitters and
    checkpoint reform rules know how to undo the fusion under tensor
    parallelism.

    Args:
        config: Owning model config; ``hidden_size`` is the input width.
        intermediate_size: Width of the MLP intermediate (per branch).
            The fused output width is ``2 * intermediate_size``.
        dtype: Compute dtype for the projection.
        param_dtype: Storage dtype for the fused weight.
        precision: JAX matmul precision flag.
        rngs: SpecTrax RNGs supplying the parameter initializer key.
        use_bias: When ``True`` adds learnable biases (rare for SwiGLU).
        target_prefix: Reform-rule prefix for the fused tensor name.
        gate_prefix: Source-tensor prefix for the gate half (HF naming).
        up_prefix: Source-tensor prefix for the up half (HF naming).
        kernel_init: Optional initializer override; defaults to
            :func:`_default_kernel_init`.
        sharding_layout: Optional explicit weight layout for the fused kernel.
        bias_sharding_layout: Optional explicit bias layout.

    Returns:
        :class:`ColumnParallelLinear` mapping
        ``hidden_size -> 2 * intermediate_size`` with a fused gate/up layout.
    """
    layout = dense_gate_up_layout(
        intermediate_size,
        gate_prefix=gate_prefix,
        up_prefix=up_prefix,
    )
    return ColumnParallelLinear(
        config.hidden_size,
        layout.segment_sizes,
        rngs=rngs,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=kernel_init or _default_kernel_init(config),
        layout=layout,
        sharding_layout=sharding_layout,
        bias_sharding_layout=bias_sharding_layout,
    )


def split_fused_gate_up_projection(
    gate_up: jax.Array,
    *,
    config: EasyDeLBaseConfig,
) -> tuple[jax.Array, jax.Array]:
    """Split a fused ``[gate | up]`` activation into its two halves.

    TP-aware: when tensor parallelism is active the fused activation is
    rank-interleaved on the last axis, so a naive split would produce
    incorrect tensors. Delegates to :func:`split_interleaved_pair_last_axis`
    which falls back to a contiguous split when no TP mesh axis applies.

    Args:
        gate_up: Activation of shape ``[..., 2 * intermediate_size]``.
        config: Owning model config used to look up the active TP mesh axis.

    Returns:
        Tuple ``(gate, up)`` each of shape ``[..., intermediate_size]``.
    """
    return split_interleaved_pair_last_axis(gate_up, config=config)


def build_fused_qkv_projection(
    *,
    config: EasyDeLBaseConfig,
    q_size: int,
    kv_size: int,
    dtype: DTypeLike,
    param_dtype: DTypeLike,
    precision: jax.lax.PrecisionLike,
    rngs: spx.Rngs,
    use_bias: bool = False,
    target_prefix: str = "qkv_proj",
    query_prefix: str = "q_proj",
    key_prefix: str = "k_proj",
    value_prefix: str = "v_proj",
    kernel_init: Initializer | None = None,
) -> ColumnParallelLinear:
    """Build a column-parallel fused ``[Q | K | V]`` attention projection.

    Creates a :class:`ColumnParallelLinear` whose output axis carries
    ``Q | K | V`` concatenated (with ``num_kv_heads <= num_heads`` for
    GQA), and attaches a :class:`FusedColumnLayout` so that downstream
    splitters and checkpoint reform rules know how to undo the fusion
    under tensor parallelism.

    Args:
        config: Owning model config; ``hidden_size`` is the input width.
        q_size: Width of the query slice (``num_heads * head_dim``).
        kv_size: Width of the key (and value) slice
            (``num_kv_heads * head_dim``).
        dtype: Compute dtype for the projection.
        param_dtype: Storage dtype for the fused weight.
        precision: JAX matmul precision flag.
        rngs: SpecTrax RNGs supplying the parameter initializer key.
        use_bias: When ``True`` adds learnable biases (rare for attention).
        target_prefix: Reform-rule prefix for the fused tensor name.
        query_prefix: Source-tensor prefix for the Q half (HF naming).
        key_prefix: Source-tensor prefix for the K half (HF naming).
        value_prefix: Source-tensor prefix for the V half (HF naming).
        kernel_init: Optional initializer override; defaults to
            :func:`_default_kernel_init`.

    Returns:
        :class:`ColumnParallelLinear` mapping
        ``hidden_size -> q_size + 2 * kv_size`` with a fused QKV layout.
    """
    layout = dense_qkv_layout(
        q_size,
        kv_size,
        query_prefix=query_prefix,
        key_prefix=key_prefix,
        value_prefix=value_prefix,
    )
    return ColumnParallelLinear(
        config.hidden_size,
        layout.out_features,
        rngs=rngs,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=kernel_init or _default_kernel_init(config),
        layout=layout,
    )


def split_fused_qkv_projection(
    qkv: jax.Array,
    *,
    q_size: int,
    kv_size: int,
    config: EasyDeLBaseConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split a fused QKV activation into ``(q, k, v)``.

    TP-aware: when the active tensor-parallel size divides each segment
    cleanly, the activation is rank-interleaved on the last axis and
    must be unpacked accordingly. Falls back to a contiguous
    ``[q_size, q_size + kv_size]`` split when no TP applies (single
    device, training without TP, or unfusable shapes).

    Args:
        qkv: Activation of shape ``[..., q_size + 2 * kv_size]``.
        q_size: Width of the query slice.
        kv_size: Width of each of the key and value slices.
        config: Owning model config used to look up the TP mesh axis.

    Returns:
        Tuple ``(q, k, v)`` with shapes ``[..., q_size]``,
        ``[..., kv_size]``, ``[..., kv_size]``.
    """
    parts = split_interleaved_segments_last_axis(
        qkv,
        (q_size, kv_size, kv_size),
        config=config,
    )
    if parts is not None:
        return cast(tuple[jax.Array, jax.Array, jax.Array], parts)
    return cast(tuple[jax.Array, jax.Array, jax.Array], tuple(jnp.split(qkv, (q_size, q_size + kv_size), axis=-1)))
