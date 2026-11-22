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


"""Backward rules for XLA mean pooling custom VJP.

The forward rule ``_mean_pooling_fwd`` and backward rule ``_mean_pooling_bwd``
are registered on ``_mean_pooling_core`` via ``defvjp``. The raw forward
functions are injected by ``_xla_impl_fwd`` after both modules load to avoid
recursive custom-VJP calls.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _mean_pooling_fwd(
    x: Float[Array, "... hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch hidden_dim"], tuple]:
    """Forward rule for the mean pooling custom VJP.

    Runs ``_mean_pooling_core`` and saves the minimal residuals needed by the
    backward rule: the shape of ``x`` (to reconstruct gradient shapes) and
    ``cu_seqlens`` (to determine which branch to use).

    Note:
        The raw forward functions are injected by ``_xla_impl_fwd`` after
        both modules are loaded (late-binding pattern).

    Args:
        x: Input tensor, either ``[total_tokens, hidden_dim]`` (varlen) or
            ``[batch, seq_len, hidden_dim]`` (fixed-length).
        cu_seqlens: Cumulative sequence lengths ``[num_seqs + 1]``, or None.

    Returns:
        Tuple of ``(output, residual)`` where:
            - ``output``: mean-pooled result ``[batch_or_num_seqs, hidden_dim]``
            - ``residual``: ``(x.shape, cu_seqlens)`` for use in the backward pass
    """
    out = _mean_pooling_varlen(x, cu_seqlens) if cu_seqlens is not None else _mean_pooling_fixed(x)
    residual = (x.shape, cu_seqlens)
    return out, residual


def _mean_pooling_bwd(
    residual: tuple,
    g: Float[Array, "batch hidden_dim"],
) -> tuple[Float[Array, "... hidden_dim"], None]:
    """Backward rule for the mean pooling custom VJP.

    Computes the gradient of the mean pooling loss w.r.t. ``x``.  The
    gradient of a mean over ``seq_len`` values is ``g / seq_len`` broadcast
    back to every token in the sequence.

    Args:
        residual: Tuple ``(x_shape, cu_seqlens)`` saved by the forward rule.
            ``x_shape`` is the original shape of the input ``x``.
        g: Upstream gradient of shape ``[num_seqs, hidden_dim]`` (varlen) or
            ``[batch, hidden_dim]`` (fixed-length).

    Returns:
        Pair ``(dx, None)`` where ``dx`` matches the shape of the
        forward input ``x`` and ``None`` denotes no gradient for boundaries:
            - varlen:  ``[total_tokens, hidden_dim]``
            - fixed:   ``[batch, seq_len, hidden_dim]``
    """
    x_shape, cu_seqlens = residual

    if cu_seqlens is not None:
        num_seqs = cu_seqlens.shape[0] - 1
        if num_seqs == 0:
            dx = jnp.zeros(x_shape, dtype=g.dtype)
        else:
            positions = jnp.arange(x_shape[0], dtype=cu_seqlens.dtype)
            seq_ids = jnp.searchsorted(cu_seqlens[1:], positions, side="right")
            valid = (positions >= cu_seqlens[0]) & (positions < cu_seqlens[-1])
            safe_ids = jnp.minimum(seq_ids, num_seqs - 1)
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            dx = jnp.where(valid[:, None], g[safe_ids] / jnp.maximum(lengths[safe_ids, None], 1), 0)
    else:
        seq_len = x_shape[1]
        dx = jnp.broadcast_to(g[:, None, :] / seq_len, x_shape)

    return dx, None


__all__ = ("_mean_pooling_bwd", "_mean_pooling_fwd")
