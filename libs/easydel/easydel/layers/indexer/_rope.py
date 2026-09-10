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

"""RoPE helpers for indexer heads.

Indexer heads frequently use a *narrower* rotary width than the attention
heads they serve (HF truncates the cos/sin tables to ``index_head_dim`` or a
partial-rotary factor), and different families split the rotated channels
differently: DeepSeek/GLM use split-half (NeoX ``rotate_half``) or adjacent
(interleaved) pairing, sometimes on the trailing slice only. The helpers here
cover all three conventions behind one entry point.
"""

from __future__ import annotations

from jax import numpy as jnp
from jaxtyping import Array, Float

__all__ = ("apply_indexer_rope", "truncate_cos_sin")


def truncate_cos_sin(
    cos: Float[Array, "... rotary"],
    sin: Float[Array, "... rotary"],
    width: int | None,
) -> tuple[Float[Array, "... rope"], Float[Array, "... rope"]]:
    """Truncate cos/sin tables to the indexer's rotary width.

    Args:
        cos: Cosine table (any leading axes; last axis is the rotary dim).
        sin: Sine table, same shape as ``cos``.
        width: Target rotary width; ``None`` keeps the tables as-is. For
            split-half tables that are pre-doubled (``[half | half]``) pass
            the *full* doubled width; for interleaved tables pass the width
            before halving (rounded down to even).

    Returns:
        ``(cos, sin)`` truncated to ``width`` (or unchanged).
    """
    if width is None:
        return cos, sin
    return cos[..., :width], sin[..., :width]


def _rotate_half(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    """NeoX split-half rotation partner: ``[-x2, x1]`` for ``x = [x1, x2]``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_indexer_rope(
    x: Float[Array, "... dim"],
    cos: Float[Array, "... rope"],
    sin: Float[Array, "... rope"],
    *,
    style: str = "split_half",
) -> Float[Array, "... dim"]:
    """Apply indexer-head RoPE in the requested channel convention.

    Args:
        x: Tensor whose trailing axis carries the head dim (queries
            ``[B, S, H, D]`` or keys ``[B, S, D]``; broadcastable leading
            axes are fine).
        cos: Cosines at the target positions, truncated to the rotary width.
            Split-half tables may be pre-doubled (``[half | half]``).
        sin: Sines, same shape as ``cos``.
        style: ``"split_half"`` rotates ``[x1, x2] -> [x1 c - x2 s, x2 c + x1 s]``
            (NeoX). ``"interleaved"`` rotates adjacent pairs
            ``[x0, x1] -> [x0 c - x1 s, x1 c + x0 s]``. ``"none"`` returns
            ``x`` unchanged.

    Returns:
        The rotated tensor, same shape as ``x``. Channels past the rotary
        width pass through unchanged.

    Raises:
        ValueError: On an unknown ``style`` or an odd rotary width for the
            interleaved style.
    """
    if style == "none":
        return x
    if style not in ("split_half", "interleaved"):
        raise ValueError(f"rope style must be 'split_half', 'interleaved' or 'none', got {style!r}.")

    rotary_dim = cos.shape[-1]
    if style == "split_half" and rotary_dim <= x.shape[-1]:
        # Pre-doubled tables ([half | half]) carry the full rotated width.
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        rotated = x_rot * cos + _rotate_half(x_rot) * sin
        if rotary_dim == x.shape[-1]:
            return rotated
        return jnp.concatenate([rotated, x_pass], axis=-1)

    # Interleaved (adjacent-channel) pairing on the leading even slice.
    width = min(rotary_dim * 2, x.shape[-1])
    width -= width % 2
    x_rot, x_pass = x[..., :width], x[..., width:]
    cos_w, sin_w = cos[..., : width // 2], sin[..., : width // 2]
    x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
    rotated = jnp.stack([x1 * cos_w - x2 * sin_w, x2 * cos_w + x1 * sin_w], axis=-1)
    rotated = rotated.reshape(x_rot.shape)
    if width == x.shape[-1]:
        return rotated
    return jnp.concatenate([rotated, x_pass], axis=-1)
