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

"""XLA reference for one manifold-constrained hyper-connection (mHC) site.

A decoder sub-layer site keeps ``hc`` residual streams ``x [B, S, hc, W]`` (``W`` = the hidden width this device
holds; models shard the hidden over tensor parallelism). The site

1. **collapses** the streams into the sub-layer input and predicts the mixing gates::

       inv_rms = rsqrt(sum_{h,d} x^2 / (hc * hidden_size) + rms_eps)          # over the FULL hidden
       logits  = einsum("shd,mhd->sm", x, fn) * inv_rms                       # m = (2 + hc) * hc
       pre     = sigmoid(logits[:hc] * scale[0] + base[:hc]) + eps
       post    = 2 sigmoid(logits[hc:2hc] * scale[1] + base[hc:2hc])
       comb    = sinkhorn(softmax(logits[2hc:].reshape(hc, hc) * scale[2] + base[2hc:]) + eps)
       collapsed = einsum("sh,shd->sd", pre, x)

2. **expands** the sub-layer output ``out [B, S, W]`` back onto the streams::

       x' = post[..., None] * out[..., None, :] + einsum("sji,sjd->sid", comb, x)

The RMS normalisation scales the ``[.., m]`` logits instead of the streams: ``(x r) fn^T == (x fn^T) r`` is a
reassociation of fp32 sums only, and no ``[B, S, hc * W]`` fp32 row is materialised.

Both contractions over the hidden are partial when ``W`` is a tensor-parallel shard; ``axis_name`` names the mesh
axis to sum them over (the sums are ``[B, S, 1 + m]`` fp32, tiny). Gate math is fp32; ``collapsed`` and ``x'`` are
in the streams' dtype, and the expand quantises ``post`` / ``comb`` to that dtype first, as the residual write of
the model layers does. This module is the contract the Pallas kernels in
``ejkernel.kernels._pallas.tpu.mhc_fused`` reproduce; the gate helpers here are shared with them.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

LANES = 128


def mix_width(hc: int) -> int:
    """Width of the gate predictor ``fn``.

    Args:
        hc: Number of residual streams.

    Returns:
        ``(2 + hc) * hc``.
    """
    return (2 + hc) * hc


def check_hc(hc: int) -> None:
    """Validate that ``hc`` fits the packed 128-lane gate and stats rows.

    Args:
        hc: Number of residual streams.

    Raises:
        ValueError: The gate row (``2 hc + hc^2`` lanes) or the stats row (``1 + mix`` lanes) exceeds ``LANES``.
    """
    if hc < 1 or 2 * hc + hc * hc > LANES or 1 + mix_width(hc) > LANES:
        raise ValueError(f"mhc: hc={hc} does not fit the {LANES}-lane packed gate row")


def axis_product(axis_name: str | tuple[str, ...] | None) -> int:
    """Product of the sizes of the named mesh axes (``1`` for ``None``); must run inside ``shard_map``.

    Args:
        axis_name: Mesh axis name, tuple of names, or ``None``.

    Returns:
        The number of hidden shards the axes describe.
    """
    if axis_name is None:
        return 1
    names = (axis_name,) if isinstance(axis_name, str) else tuple(axis_name)
    return int(math.prod(int(jax.lax.axis_size(a)) for a in names))


def resolve_hidden_size(local_width: int, hidden_size: int | None, axis_name) -> int:
    """The FULL hidden width used as the RMS denominator.

    The default takes it from the streams (``local_width`` times the number of hidden shards), which is what the
    unfused model layers normalise by; an explicit ``hidden_size`` overrides it.

    Args:
        local_width: Hidden width held by this device.
        hidden_size: Optional explicit full hidden width.
        axis_name: Mesh axis (or axes) the hidden is sharded over, or ``None``.

    Returns:
        The full hidden width.
    """
    if hidden_size is not None:
        return int(hidden_size)
    return int(local_width) * axis_product(axis_name)


def sinkhorn_comb(comb_logits: Array, iters: int, eps: float) -> Array:
    """Project ``softmax(logits) + eps`` onto the doubly-stochastic manifold.

    Column normalisation first, then ``iters - 1`` alternating row / column steps, every denominator floored by
    ``eps`` -- the same recurrence as ``ejkernel.modules.sinkhorn_knopp``.

    Args:
        comb_logits: Mixer logits ``[..., hc, hc]`` (fp32).
        iters: Sinkhorn-Knopp iterations (static, >= 1).
        eps: Softmax offset and denominator floor.

    Returns:
        The mixer ``[..., hc, hc]`` fp32.
    """
    c = jax.nn.softmax(comb_logits, axis=-1) + eps
    c = c / (jnp.sum(c, axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (jnp.sum(c, axis=-1, keepdims=True) + eps)
        c = c / (jnp.sum(c, axis=-2, keepdims=True) + eps)
    return c


def gates_from_logits(
    mix_logits: Array, base: Array, scale: Array, *, hc: int, sinkhorn_iters: int, eps: float
) -> tuple[Array, Array, Array]:
    """``(pre, post, comb)`` fp32 from RMS-normalised ``[..., mix]`` predictor logits.

    The three scales are applied as one gathered vector (``scale[[0]*hc + [1]*hc + [2]*hc*hc]``), so the scale
    cotangent is a single segment sum rather than three scalar pads.

    Args:
        mix_logits: Normalised logits ``[..., mix]`` (fp32).
        base: Gate biases ``[mix]``.
        scale: Logit scales ``[3]``.
        hc: Number of streams.
        sinkhorn_iters: Sinkhorn-Knopp iterations.
        eps: ``pre`` and Sinkhorn floor.

    Returns:
        ``pre [..., hc]``, ``post [..., hc]``, ``comb [..., hc, hc]``, all fp32.
    """
    base = jnp.asarray(base, jnp.float32)
    scale = jnp.asarray(scale, jnp.float32)
    lead = mix_logits.shape[:-1]
    scale_indices = np.asarray([0] * hc + [1] * hc + [2] * (hc * hc), np.int32)
    gated = mix_logits * scale[scale_indices] + base
    pre = jax.nn.sigmoid(gated[..., :hc]) + eps
    post = 2.0 * jax.nn.sigmoid(gated[..., hc : 2 * hc])
    comb = sinkhorn_comb(gated[..., 2 * hc :].reshape(*lead, hc, hc), sinkhorn_iters, eps)
    return pre, post, comb


def gates_from_stats(
    stats: Array,
    base: Array,
    scale: Array,
    *,
    hc: int,
    hidden_size: int,
    rms_eps: float,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[Array, Array, Array]:
    """``(pre, post, comb)`` from a packed ``[rows, LANES]`` fp32 stats row.

    Args:
        stats: Lane 0 = sum of squares over the FULL hidden, lanes ``1..mix`` = the un-normalised ``fn``
            projection.
        base: Gate biases ``[mix]``.
        scale: Logit scales ``[3]``.
        hc: Number of streams.
        hidden_size: Full hidden width (RMS denominator ``hc * hidden_size``).
        rms_eps: RMS epsilon.
        sinkhorn_iters: Sinkhorn-Knopp iterations.
        eps: ``pre`` and Sinkhorn floor.

    Returns:
        ``pre [rows, hc]``, ``post [rows, hc]``, ``comb [rows, hc, hc]`` fp32.
    """
    mix = mix_width(hc)
    inv_rms = jax.lax.rsqrt(stats[:, 0] / (hc * hidden_size) + rms_eps)
    logits = stats[:, 1 : 1 + mix] * inv_rms[:, None]
    return gates_from_logits(logits, base, scale, hc=hc, sinkhorn_iters=sinkhorn_iters, eps=eps)


def pack_gates(pre: Array, post: Array, comb: Array) -> Array:
    """Pack gates into one lane-dense ``[rows, LANES]`` fp32 row per token.

    Args:
        pre: Read gates ``[rows, hc]``.
        post: Write gates ``[rows, hc]``.
        comb: Mixer ``[rows, hc, hc]``.

    Returns:
        ``[rows, LANES]`` fp32 with the lane layout documented on :data:`LANES`.
    """
    rows, hc = pre.shape
    body = jnp.concatenate(
        [pre.astype(jnp.float32), post.astype(jnp.float32), comb.astype(jnp.float32).reshape(rows, hc * hc)], axis=-1
    )
    return jnp.pad(body, ((0, 0), (0, LANES - body.shape[-1])))


def unpack_gates(gates: Array, hc: int) -> tuple[Array, Array, Array]:
    """Inverse of :func:`pack_gates`.

    Args:
        gates: Packed ``[rows, LANES]`` row.
        hc: Number of streams.

    Returns:
        ``pre [rows, hc]``, ``post [rows, hc]``, ``comb [rows, hc, hc]``.
    """
    rows = gates.shape[0]
    pre = gates[:, :hc]
    post = gates[:, hc : 2 * hc]
    comb = gates[:, 2 * hc : 2 * hc + hc * hc].reshape(rows, hc, hc)
    return pre, post, comb


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
) -> tuple[Float[Array, "batch seq hc"], Float[Array, "batch seq hc hc"], Float[Array, "batch seq width"]]:
    """Reference collapse: ``(post, comb, collapsed)`` -- see the module docstring.

    Args:
        x: Residual streams ``[B, S, hc, W]`` (``W`` may be a tensor-parallel shard of the hidden).
        fn: Gate predictor ``[(2 + hc) hc, hc, W]`` -- the same hidden shard as ``x``.
        base: Gate biases ``[(2 + hc) hc]``.
        scale: Per-gate logit scales ``[3]`` (pre, post, comb).
        hidden_size: FULL hidden width for the RMS denominator ``hc * hidden_size``; ``None`` takes it from the
            streams (``W`` times the size of ``axis_name``).
        sinkhorn_iters: Sinkhorn-Knopp iterations for ``comb``.
        eps: Sinkhorn / ``pre`` floor.
        rms_eps: RMS epsilon.
        axis_name: Mesh axis (or axes) over which ``W`` is sharded; the two hidden contractions are summed over
            it. ``None`` when ``W`` is the full hidden.

    Returns:
        ``post [B, S, hc]`` fp32, ``comb [B, S, hc, hc]`` fp32 (doubly stochastic), ``collapsed [B, S, W]`` in
        ``x.dtype``.
    """
    hc = x.shape[2]
    check_hc(hc)
    hidden_size = resolve_hidden_size(x.shape[3], hidden_size, axis_name)
    x32 = x.astype(jnp.float32)
    fn3 = fn.astype(jnp.float32)
    sumsq = jnp.sum(jnp.square(x32), axis=(2, 3))
    mixp = jnp.einsum("bshd,mhd->bsm", x32, fn3, precision=jax.lax.Precision.HIGHEST)
    if axis_name is not None:
        sumsq = jax.lax.psum(sumsq, axis_name)
        mixp = jax.lax.psum(mixp, axis_name)
    inv_rms = jax.lax.rsqrt(sumsq / (hc * hidden_size) + rms_eps)
    logits = mixp * inv_rms[..., None]
    pre, post, comb = gates_from_logits(logits, base, scale, hc=hc, sinkhorn_iters=sinkhorn_iters, eps=eps)
    collapsed = jnp.sum(pre[..., None] * x32, axis=2)
    return post, comb, collapsed.astype(x.dtype)


def mhc_expand(
    x: Float[Array, "batch seq hc width"],
    out: Float[Array, "batch seq width"],
    post: Float[Array, "batch seq hc"],
    comb: Float[Array, "batch seq hc hc"],
) -> Float[Array, "batch seq hc width"]:
    """Reference expand with ``post`` / ``comb`` first cast to ``x.dtype``.

    Args:
        x: Residual streams ``[B, S, hc, W]``.
        out: Sub-layer output ``[B, S, W]``.
        post: Write gates ``[B, S, hc]``.
        comb: Mixer ``[B, S, hc, hc]`` indexed ``[source, target]``.

    Returns:
        ``post[..., None] * out[..., None, :] + einsum("bsji,bsjd->bsid", comb, x)`` in ``x.dtype``. The stream mix
        runs at HIGHEST precision, so fp32 streams get an fp32 contract (bf16 streams are exact either way).
    """
    dtype = x.dtype
    return post.astype(dtype)[..., None] * out.astype(dtype)[..., None, :] + jnp.einsum(
        "bsji,bsjd->bsid", comb.astype(dtype), x, precision=jax.lax.Precision.HIGHEST
    )


__all__ = (
    "LANES",
    "axis_product",
    "check_hc",
    "gates_from_logits",
    "gates_from_stats",
    "mhc_collapse",
    "mhc_expand",
    "mix_width",
    "pack_gates",
    "resolve_hidden_size",
    "sinkhorn_comb",
    "unpack_gates",
)
