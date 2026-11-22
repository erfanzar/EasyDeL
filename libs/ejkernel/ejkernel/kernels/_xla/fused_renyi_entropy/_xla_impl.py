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

"""Pure-JAX fused per-row Renyi entropy of a softmax distribution.

For ``p = softmax(z / T)`` and order ``alpha > 0``::

    H_alpha(p) = log(sum_v p_v^alpha) / (1 - alpha)          (alpha != 1)
    H_1(p)     = -sum_v p_v log p_v                          (Shannon limit)

The ``alpha != 1`` case needs only two logsumexps over the vocabulary,
``log sum p^alpha = LSE(alpha * z~) - alpha * LSE(z~)`` with ``z~ = z / T``, so
it streams and shards exactly like a softmax normalizer and never materializes
``p``. ``alpha == 1.0`` selects the Shannon branch exactly (avoiding the
``1 / (1 - alpha)`` blow-up), ``alpha = 2`` is collision entropy, and
``alpha -> inf`` approaches min-entropy ``-log max_v p_v``.

Gradients come from plain autodiff (no custom VJP). Logits must be finite:
``-inf`` entries make the Shannon branch evaluate ``0 * -inf``.
"""

from __future__ import annotations

import jax.numpy as jnp

from .._row_loss import (
    check_reduction,
    check_temperature,
    log_softmax,
    logsumexp,
    reduce_rows,
    row_weights,
    vocab_sum,
)


def renyi_entropy_rows(z, alpha: float, vocab_axis: str | None = None, *, normalized: bool = False):
    """Per-row Renyi entropy of ``softmax(z)`` over the last (possibly sharded) axis.

    Args:
        z: float32 ``(..., V)`` (or ``(..., V_local)`` when ``vocab_axis`` is set) logits.
        alpha: Renyi order ``> 0``; ``1.0`` selects the exact Shannon branch.
        vocab_axis: Mesh axis the vocabulary is sharded over, or ``None``.
        normalized: ``True`` when ``z`` is already a log-probability tensor
            (``logsumexp(z) == 0``), which skips the normalizer reduction.

    Returns:
        float32 ``(...,)`` per-row entropies.
    """
    if alpha == 1.0:
        log_p = z if normalized else log_softmax(z, vocab_axis)
        return -vocab_sum(jnp.exp(log_p) * log_p, vocab_axis)
    a = jnp.float32(alpha)
    lse_alpha = logsumexp(a * z, vocab_axis)
    if not normalized:
        lse_alpha = lse_alpha - a * logsumexp(z, vocab_axis)
    return lse_alpha / jnp.float32(1.0 - alpha)


def fused_renyi_entropy(
    logits,
    weights=None,
    *,
    alpha: float = 1.0,
    reduction: str = "mean",
    temperature: float = 1.0,
    vocab_parallel_axis: str | None = None,
):
    """JAX/XLA fused per-row Renyi entropy over the vocabulary axis.

    Args:
        logits: ``(..., V)`` logits (differentiable; ``stop_gradient`` outside
            for a detached diagnostic).
        weights: Optional ``logits.shape[:-1]`` per-row weights / mask.
        alpha: Renyi order ``> 0``. ``1.0`` is exact Shannon; ``2.0`` is
            collision entropy.
        reduction: ``"none"`` / ``"sum"`` / ``"mean"`` (mean over
            ``max(sum(weights), 1e-8)``).
        temperature: Softmax temperature applied to the logits. The entropy is
            NOT ``T**2``-rescaled (it is not a KL-family term); callers that
            compose it into a ``T**2``-scaled objective apply their own scale.
        vocab_parallel_axis: Mesh axis name for vocab parallelism; requires
            per-shard inputs inside ``shard_map``.

    Returns:
        float32 per-row array (``reduction="none"``) or scalar.

    Raises:
        ValueError: On an invalid ``reduction``, non-positive ``temperature`` or
            ``alpha``, or a ``weights`` shape that differs from ``logits.shape[:-1]``.
    """
    check_reduction(reduction)
    check_temperature(temperature)
    if not float(alpha) > 0.0:
        raise ValueError(f"alpha must be positive; got {alpha}")

    wts = row_weights(weights, logits.shape[:-1])
    z = logits.astype(jnp.float32) / jnp.float32(temperature)
    per_row = renyi_entropy_rows(z, float(alpha), vocab_parallel_axis) * wts
    return reduce_rows(per_row, wts, reduction)
