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

"""XLA reference for the fused indexer scoring: the same math with the ``[B, H, S, T]`` logits materialised.

This is the numerical contract of the Pallas TPU kernel: the kernel must equal it in value and in all three
cotangents (``jax.grad`` through this function is the reference backward).
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ejkernel.ops import FwdParams

_PRECISION_ALIASES = {
    "default": "default",
    "fastest": "default",
    "bfloat16": "default",
    "high": "high",
    "tensorfloat32": "high",
    "highest": "highest",
    "float32": "highest",
}


def precision_key(precision: tp.Any) -> str:
    """Normalise ``precision`` (``None`` / str / ``jax.lax.Precision`` / an equal pair) to ``default|high|highest``.

    Args:
        precision: Caller precision.

    Returns:
        The normalised static precision name.

    Raises:
        ValueError: For an unknown spelling or a pair of different precisions.
    """
    if precision is None:
        return "default"
    if isinstance(precision, (tuple, list)):
        keys = {precision_key(p) for p in precision}
        if len(keys) != 1:
            raise ValueError(f"indexer_scores: per-operand precision {precision!r} is not supported")
        return keys.pop()
    if isinstance(precision, jax.lax.Precision):
        return _PRECISION_ALIASES[precision.name.lower()]
    if isinstance(precision, str) and precision.lower() in _PRECISION_ALIASES:
        return _PRECISION_ALIASES[precision.lower()]
    raise ValueError(f"precision must be None, 'default', 'high', 'highest' or a jax.lax.Precision, got {precision!r}")


def indexer_scores_xla(
    query: Float[Array, "batch num_heads q_len head_dim"],
    index_k: Float[Array, "batch num_entries head_dim"],
    weights: Float[Array, "batch num_heads q_len 1"],
    fwd_params: FwdParams | None = None,
    precision: tp.Any = None,
) -> Float[Array, "batch q_len num_entries"]:
    """``sum_h weights[b,h,s] * relu(query[b,h,s] . index_k[b,t])`` -> fp32 ``[B, S, T]``.

    Args:
        query: Index queries ``[B, H, S, d]``.
        index_k: Index keys ``[B, T, d]``; cast to the query dtype for the contraction.
        weights: Per-head, per-query weights ``[B, H, S, 1]`` (applied in fp32).
        fwd_params: Ignored (tiling belongs to the Pallas kernel).
        precision: ``None`` / ``"default"`` contracts in the operand dtype; ``"high"`` / ``"highest"`` (or a
            ``jax.lax.Precision``) upcasts the operands to fp32 and contracts at that precision.

    Returns:
        fp32 ``[B, S, T]`` scores.
    """
    del fwd_params
    key = precision_key(precision)
    index_k = index_k.astype(query.dtype)
    if key != "default":
        query, index_k = query.astype(jnp.float32), index_k.astype(jnp.float32)
    lax_precision = {
        "default": jax.lax.Precision.DEFAULT,
        "high": jax.lax.Precision.HIGH,
        "highest": jax.lax.Precision.HIGHEST,
    }[key]
    logits = jnp.einsum("bhsd,btd->bhst", query, index_k, preferred_element_type=jnp.float32, precision=lax_precision)
    return jnp.sum(jax.nn.relu(logits) * weights.astype(jnp.float32), axis=1)


__all__ = ("indexer_scores_xla", "precision_key")
