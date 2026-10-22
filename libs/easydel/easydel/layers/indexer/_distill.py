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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer-facing entry point of the indexer warm-up objective (the registered ``indexer_distill_kl`` operation)."""

from __future__ import annotations

import typing as tp

import jax
from jaxtyping import Array

__all__ = ("indexer_distillation_loss",)


def indexer_distillation_loss(
    q: Array,
    entries: Array,
    scores: tp.Any,
    *,
    scaling: float,
    window_kv: Array | None = None,
    sinks: Array | None = None,
    token_valid: Array | None = None,
    window: int | None = None,
    precision: jax.lax.PrecisionLike = None,
    block_q: int = 64,
    entry_group: int = 1,
    entry_valid: Array | None = None,
    q_positions: Array | None = None,
    entry_positions: Array | None = None,
    q_segment_ids: Array | None = None,
    entry_segment_ids: Array | None = None,
) -> Array:
    """Indexer warm-up KL ``KL(stop_grad(p) || softmax(scores))``; scalar fp32 mean over queries reaching an entry.

    ``p`` is the dense attention distribution over ``entries`` (sink-augmented softmax over
    ``[window tokens | entries | sink]``, summed over heads, L1-normalized per query). Only ``scores`` receives
    gradient. ``window`` / ``window_kv`` and ``sinks`` are optional: without them the target is a plain softmax over
    the entries (token DSA, entries = tokens). See :class:`easydel.operations.kernels.IndexerDistillKLOp`.

    Args:
        q: Attention queries ``[B, H, S, D]``.
        entries: Candidate keys ``[B, T, D]`` (shared) or ``[B, H, T, D]``.
        scores: Indexer scores ``[B, S, T]`` (``-inf`` = unreachable) or lazy
            :class:`~easydel.layers.indexer.IndexerScoreInputs`.
        scaling: Attention logit scale.
        window_kv: Sliding-window token keys; required with ``window``.
        sinks: Optional per-head sinks ``[H]``.
        token_valid: Optional per-key validity of the window region ``[B, S]``.
        window: Sliding-window size, or ``None`` for no token region.
        precision: Matmul precision.
        block_q: Queries per tile.
        entry_group: Entries per indexer candidate (pooled / block indexers); ``scores`` then has
            ``ceil(T / entry_group)`` columns and the target is the attention mass summed per group.
        entry_valid: Optional per-entry validity ``[B, T]`` on the attention side.
        q_positions: Optional query positions ``[B, S]``; with ``entry_positions`` ``[B, T]`` enforces token-level
            causality on the attention side.
        entry_positions: See ``q_positions``.
        q_segment_ids: Optional packed-sequence ids ``[B, S]``; with ``entry_segment_ids`` ``[B, T]`` restricts the
            attention side to the query's segment.
        entry_segment_ids: See ``q_segment_ids``.

    Returns:
        Scalar fp32 loss.
    """
    # Imported lazily: ``easydel.operations`` imports ``easydel.layers``.
    from easydel.operations.kernels.indexer_distill import indexer_distill_kl_op

    return indexer_distill_kl_op()(
        q,
        entries,
        scores,
        scaling=scaling,
        window_kv=window_kv,
        sinks=sinks,
        token_valid=token_valid,
        window=window,
        precision=precision,
        block_q=block_q,
        entry_group=entry_group,
        entry_valid=entry_valid,
        q_positions=q_positions,
        entry_positions=entry_positions,
        q_segment_ids=q_segment_ids,
        entry_segment_ids=entry_segment_ids,
    ).loss
