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

"""Registry entry point for the XLA fused cross-entropy."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import fused_cross_entropy as _fused_ce_xla


@kernel_registry.register("fused_cross_entropy", Platform.XLA, Backend.ANY)
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
    """XLA reference for fused cross-entropy with analytic VJP.

    Matches the TileLang implementation's contract bit-for-bit (numerics
    aside): same sparse/dense modes, same ``label_smoothing`` /
    ``z_loss`` semantics, same masking, same gradient formula. Use this
    on non-NVIDIA backends (TPU/CPU) or when TileLang is unavailable.

    ``block_v`` / ``block_m`` are accepted for signature compatibility
    with the TileLang interface; XLA doesn't honour them (autotune
    happens inside XLA itself).
    """
    del block_v, block_m
    return _fused_ce_xla(
        logits,
        targets,
        weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        soft_targets=soft_targets,
        reduction=reduction,
        vocab_parallel_axis=vocab_parallel_axis,
    )


__all__ = ("fused_cross_entropy",)
