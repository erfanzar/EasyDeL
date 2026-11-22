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

"""Registry interface for TPU Pallas fused cross-entropy.

This module is intentionally thin: it owns the kernel-registry entry and type
checking boundary, then delegates all execution to ``_pallas_impl_fwd``. Keeping
the registry wrapper small avoids duplicating validation logic between the
public operation layer and the Pallas implementation.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import fused_cross_entropy_pallas


@kernel_registry.register("fused_cross_entropy", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def fused_cross_entropy(
    logits: Float[Array, "... vocab_size"],
    targets: Int[Array, "..."] | None = None,
    weights: Float[Array, "..."] | None = None,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    soft_targets: Float[Array, "... vocab_size"] | None = None,
    reduction: str = "mean",
    vocab_parallel_axis: str | None = None,
    block_v: int = 0,
    block_m: int = 0,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Dispatch fused sparse cross-entropy to the TPU Pallas implementation.

    Args:
        logits: ``(..., vocab_size)`` local or full-vocab logits.
        targets: Sparse integer token ids with shape ``logits.shape[:-1]``.
        weights: Optional per-row weights. Zero weights enable row-block early
            exit in the Pallas kernel.
        ignore_index: Target sentinel excluded from loss and gradient.
        label_smoothing: Sparse CE label smoothing. Supported for replicated
            vocab; vocab-parallel Pallas currently requires zero smoothing.
        z_loss: Optional Mesh-TF/PaLM-style logit z-loss coefficient.
        soft_targets: Dense target distribution. Falls back to the XLA path.
        reduction: ``"none"``, ``"sum"``, or ``"mean"``.
        vocab_parallel_axis: Mesh axis name for vocab sharding inside
            ``shard_map``. When present, the Pallas implementation uses JAX
            collectives to merge softmax statistics over that axis.
        block_v: Optional vocab tile override; stale small values are floored by
            the implementation's TPU defaults.
        block_m: Optional row tile override; the implementation chooses its TPU
            default for the public path.

    Returns:
        ``(loss, per_row_correct)`` where ``loss`` follows ``reduction`` and
        ``per_row_correct`` is an accuracy helper array, or ``-1`` sentinels
        when accuracy is not meaningful for the local shard.
    """
    return fused_cross_entropy_pallas(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        soft_targets=soft_targets,
        reduction=reduction,
        vocab_parallel_axis=vocab_parallel_axis,
        block_v=block_v,
        block_m=block_m,
    )


__all__ = ["fused_cross_entropy"]
