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

"""tile-lang fused cross-entropy public interface.

Registers ``fused_cross_entropy`` for ``Platform.TILELANG / Backend.GPU``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._impl import fused_cross_entropy_tilelang


@kernel_registry.register("fused_cross_entropy", Platform.TILELANG, Backend.GPU)
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
    """Tile-lang fused cross-entropy with analytic backward.

    Two target modes:
      * **Sparse** (default): integer ``targets``. Optional
        ``label_smoothing`` and ``z_loss`` regularisation are folded into
        the kernel at build time (zero cost when both are 0).
      * **Dense**: pass ``soft_targets`` (full probability distribution).
        ``targets`` is ignored; ``label_smoothing`` rejected (apply it
        externally to the target distribution).

    Args:
        logits: ``(..., V)`` predicted logits. Supported dtypes:
            float16, bfloat16, float32.
        targets: Integer token ids with shape ``logits.shape[:-1]``.
            Required when ``soft_targets`` is ``None``. Positions equal
            to ``ignore_index`` contribute zero loss and zero gradient.
        weights: Optional per-token weights with shape ``logits.shape[:-1]``.
            When ``None`` in sparse mode, a 0/1 mask is built from
            ``targets != ignore_index``.
        ignore_index: Sparse-mode sentinel for ignored positions.
        label_smoothing: Smoothing coefficient ``α ∈ [0, 1)``. Adds the
            normalising constant ``-[(1-α) log(1-α) + (V-1) lc log(lc)]``
            with ``lc = α/(V-1)`` so a perfect prediction still has
            loss 0.
        z_loss: Coefficient for ``z_loss · lse²`` auxiliary loss
            (Mesh-TF / PaLM-style logit-magnitude regularisation).
        soft_targets: ``(..., V)`` dense probability targets. Switches
            to the dense kernel pair (one extra ``N·V`` HBM read for
            fwd and bwd vs sparse).
        reduction: ``"none"`` / ``"sum"`` / ``"mean"``.
        vocab_parallel_axis: Mesh axis along which ``V`` is sharded
            (sparse + no smoothing/z_loss only).

    Returns:
        Scalar for ``"mean"`` / ``"sum"``; otherwise ``logits.shape[:-1]``.

    Raises:
        EjkernelRuntimeError: On invalid shapes or argument combos.
    """
    if soft_targets is None:
        if targets is None:
            raise EjkernelRuntimeError(
                "fused_cross_entropy: either `targets` (sparse) or `soft_targets` (dense) must be provided."
            )
        if targets.shape != logits.shape[:-1]:
            raise EjkernelRuntimeError(
                f"fused_cross_entropy: targets.shape={targets.shape} must equal logits.shape[:-1]={logits.shape[:-1]}"
            )
        if weights is not None and weights.shape != targets.shape:
            raise EjkernelRuntimeError(
                f"fused_cross_entropy: weights.shape={weights.shape} must equal targets.shape={targets.shape}"
            )
    else:
        if soft_targets.shape != logits.shape:
            raise EjkernelRuntimeError(
                f"fused_cross_entropy: soft_targets.shape={soft_targets.shape} must equal logits.shape={logits.shape}"
            )
        if weights is not None and weights.shape != logits.shape[:-1]:
            raise EjkernelRuntimeError(
                f"fused_cross_entropy: weights.shape={weights.shape} must equal logits.shape[:-1]={logits.shape[:-1]}"
            )
    return fused_cross_entropy_tilelang(
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
