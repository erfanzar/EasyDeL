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

"""Registry entries for the XLA mHC collapse / expand (``Platform.XLA``, ``Backend.ANY``)."""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import mhc_collapse as _collapse_impl
from ._xla_impl_fwd import mhc_expand as _expand_impl


@kernel_registry.register("mhc_collapse", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def mhc_collapse(
    x: Float[Array, "batch seq hc width"],
    fn: Float[Array, "mix hc width"],
    base: Float[Array, "mix"],
    scale: Float[Array, "three"],
    *,
    hidden_size: int | None = None,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    rms_eps: float = 1e-6,
    axis_name: str | tuple[str, ...] | None = None,
    tile_rows: int = 256,
    fn_pieces: int = 3,
    grad_pieces: int = 3,
    interpret: bool | None = None,
) -> tuple[Float[Array, "batch seq hc"], Float[Array, "batch seq hc hc"], Float[Array, "batch seq width"]]:
    """mHC collapse (XLA reference): ``(post, comb, collapsed)``; see :mod:`._xla_impl_fwd`.

    ``tile_rows`` / ``fn_pieces`` / ``grad_pieces`` / ``interpret`` are the Pallas implementation's arguments,
    accepted and ignored so both registrations share one call signature.

    Args:
        x: Residual streams ``[B, S, hc, W]``.
        fn: Gate predictor ``[(2 + hc) hc, hc, W]``.
        base: Gate biases ``[(2 + hc) hc]``.
        scale: Logit scales ``[3]``.
        hidden_size: Full hidden width for the RMS denominator (``None``: from the streams).
        sinkhorn_iters: Sinkhorn-Knopp iterations.
        eps: Sinkhorn / ``pre`` floor.
        rms_eps: RMS epsilon.
        axis_name: Mesh axis the hidden is sharded over (inside ``shard_map``), or ``None``.
        tile_rows: Ignored (Pallas only).
        fn_pieces: Ignored (Pallas only).
        grad_pieces: Ignored (Pallas only).
        interpret: Ignored (Pallas only).

    Returns:
        ``post [B, S, hc]`` fp32, ``comb [B, S, hc, hc]`` fp32, ``collapsed [B, S, W]`` in ``x.dtype``.
    """
    del tile_rows, fn_pieces, grad_pieces, interpret
    return _collapse_impl(
        x,
        fn,
        base,
        scale,
        hidden_size=hidden_size,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        rms_eps=rms_eps,
        axis_name=axis_name,
    )


@kernel_registry.register("mhc_expand", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def mhc_expand(
    x: Float[Array, "batch seq hc width"],
    out: Float[Array, "batch seq width"],
    post: Float[Array, "batch seq hc"],
    comb: Float[Array, "batch seq hc hc"],
    *,
    axis_name: str | tuple[str, ...] | None = None,
    tile_rows: int = 256,
    interpret: bool | None = None,
) -> Float[Array, "batch seq hc width"]:
    """mHC expand (XLA reference): the new streams; see :mod:`._xla_impl_fwd`.

    Args:
        x: Residual streams ``[B, S, hc, W]``.
        out: Sub-layer output ``[B, S, W]``.
        post: Write gates ``[B, S, hc]``.
        comb: Mixer ``[B, S, hc, hc]``.
        axis_name: Accepted for signature parity; the expand has no cross-device sum.
        tile_rows: Ignored (Pallas only).
        interpret: Ignored (Pallas only).

    Returns:
        New streams ``[B, S, hc, W]`` in ``x.dtype``.
    """
    del axis_name, tile_rows, interpret
    return _expand_impl(x, out, post, comb)


__all__ = ("mhc_collapse", "mhc_expand")
