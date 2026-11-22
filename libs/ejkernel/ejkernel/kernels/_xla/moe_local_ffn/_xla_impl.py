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

"""Experts-over-tp local expert FFN: one ``custom_vjp`` over dropless dynamic row windows.

Layout: the token stream ``x [T, H]`` is replicated over the expert-sharding axis and this device owns ``E_local``
full-width experts (``w_gate`` / ``w_up`` ``[E_local, H, M]``, ``w_down`` ``[E_local, M, H]``). Every token routes to
``k`` experts; routes whose expert id lies outside ``[0, E_local)`` belong to another device and contribute nothing
here. The result is this device's partial ``out[t] = sum_{j local} weights[t, j] * FFN_{experts[t, j]}(x[t])``; the
caller reduces it over the expert-sharding axis.

Per token chunk of ``tokens_per_chunk`` tokens:

1. One stable sort puts the chunk's local routes first, grouped by local expert.
2. The local prefix is processed in row windows of ``rows_per_window`` rows. The window count is bounded statically
   by ``ceil(chunk * k / rows_per_window)``, but windows past the actual local row count are skipped with
   ``lax.cond``: routing skew never drops a route, and balanced routing pays only for the rows it has.
3. Per window: gather the token rows, grouped gate / up matmuls, activation, grouped down matmul, then a token-major
   weighted gather-sum adds the window's rows into an fp32 ``[chunk, H]`` accumulator.

The backward is hand-written (no autodiff through the loops; only the inputs are saved). Per window it recomputes
gate / up, gathers the output-cotangent rows, forms ``dh = d_out @ w_down^T``, takes the routing-weight gradient as
``<dh, h>`` (no down-projection recompute), accumulates the weight gradients in place across windows and chunks, and
gathers the input gradient back token-major.

Backends plug in through :class:`LocalFfnPrimitives`. This module provides the XLA primitives (``take`` gathers,
``ragged_dot`` / ``ragged_dot_general``) and is the mandatory fallback.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

F32 = jnp.float32
I32 = jnp.int32

Activation = Callable[[jax.Array, jax.Array], jax.Array]


class LocalFfnPrimitives(tp.NamedTuple):
    """Backend primitives the local-FFN core is written against.

    Attributes:
        gather: ``(src [n, H], rows [r]) -> src[rows]`` in ``src.dtype``.
        gather_sum: ``(y [r, H], idx [t, k], w [t, k]) -> sum_j w[:, j] * y[idx[:, j]]`` in fp32; slots with a
            negative index contribute nothing (and should move no data).
        gmm: ``(lhs [m, K], rhs [G, K, N], group_sizes, transpose_rhs=False) -> [m, N]`` in ``lhs.dtype``
            (``rhs`` is ``[G, N, K]`` with ``transpose_rhs``). Rows past ``sum(group_sizes)`` are unspecified.
        tgmm: ``(lhs [m, K], rhs [m, N], group_sizes, acc [G, K, N]) -> acc + lhs_g^T @ rhs_g`` per group in
            ``acc.dtype``. Rows past ``sum(group_sizes)`` must not contribute.
    """

    gather: Callable[[jax.Array, jax.Array], jax.Array]
    gather_sum: Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
    gmm: Callable[..., jax.Array]
    tgmm: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def default_activation(gate: jax.Array, up: jax.Array) -> jax.Array:
    """SwiGLU: ``silu(gate) * up``."""
    return jax.nn.silu(gate) * up


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def plan_local_ffn(
    tokens: int,
    top_k: int,
    tokens_per_chunk: int | None = None,
    rows_per_window: int | None = None,
    *,
    token_align: int = 1,
    row_align: int = 1,
) -> tuple[int, int, int, int]:
    """Static ``(chunk_tokens, window_rows, num_chunks, max_windows_per_chunk)``.

    ``tokens_per_chunk`` (default: all tokens) is capped at ``tokens`` and rounded up to ``token_align``; the token
    stream is zero-padded to a whole number of chunks. ``rows_per_window`` (default: every route of a chunk) is capped
    at ``chunk * top_k`` and rounded up to ``row_align``.

    Raises:
        ValueError: For non-positive sizes.
    """
    if tokens <= 0 or top_k <= 0:
        raise ValueError(f"tokens and top_k must be positive, got tokens={tokens}, top_k={top_k}")
    tc = tokens if tokens_per_chunk is None else int(tokens_per_chunk)
    if tc <= 0:
        raise ValueError(f"tokens_per_chunk must be positive, got {tokens_per_chunk}")
    tc = _round_up(min(tc, tokens), token_align)
    routes = tc * top_k
    r = routes if rows_per_window is None else int(rows_per_window)
    if r <= 0:
        raise ValueError(f"rows_per_window must be positive, got {rows_per_window}")
    r = _round_up(min(r, routes), row_align)
    return tc, r, -(-tokens // tc), -(-routes // r)


def check_local_ffn_shapes(x, weights, experts, w_gate, w_up, w_down) -> None:
    """Raise ``ValueError`` unless the operands form one local-FFN call."""
    if x.ndim != 2:
        raise ValueError(f"x must be [tokens, hidden], got shape {tuple(x.shape)}")
    tokens, hidden = x.shape
    if experts.ndim != 2 or experts.shape[0] != tokens:
        raise ValueError(f"experts must be [tokens={tokens}, k], got shape {tuple(experts.shape)}")
    if tuple(weights.shape) != tuple(experts.shape):
        raise ValueError(f"weights {tuple(weights.shape)} and experts {tuple(experts.shape)} must match")
    if not jnp.issubdtype(experts.dtype, jnp.integer):
        raise ValueError(f"experts must be integer expert ids, got {experts.dtype}")
    if w_gate.ndim != 3 or w_gate.shape[0] < 1 or w_gate.shape[1] != hidden:
        raise ValueError(f"w_gate must be [experts, hidden={hidden}, inter], got shape {tuple(w_gate.shape)}")
    num_local, _, inter = w_gate.shape
    if tuple(w_up.shape) != tuple(w_gate.shape):
        raise ValueError(f"w_up {tuple(w_up.shape)} must match w_gate {tuple(w_gate.shape)}")
    if tuple(w_down.shape) != (num_local, inter, hidden):
        raise ValueError(f"w_down must be {(num_local, inter, hidden)}, got {tuple(w_down.shape)}")


def _chunk_routes(experts_chunk: jax.Array, num_local: int, padded_rows: int):
    """Sort one chunk's routes local-first; return order, inverse, expert row bounds, local row count, local mask."""
    shape = experts_chunk.shape
    flat = experts_chunk.reshape(-1)
    key = jnp.where((flat >= 0) & (flat < num_local), flat, num_local).astype(I32)
    order = jnp.argsort(key, stable=True).astype(I32)
    n = order.shape[0]
    # A second sort (not a 1-D scatter, which is slow on TPU) inverts the permutation.
    inverse = jnp.argsort(order).astype(I32)
    counts = jnp.bincount(key, length=num_local + 1)[:num_local].astype(I32)
    ends = jnp.cumsum(counts).astype(I32)
    starts = ends - counts
    if padded_rows > n:
        order = jnp.pad(order, (0, padded_rows - n))
    return order, inverse.reshape(shape), starts, ends, ends[-1], (key < num_local).reshape(shape)


def _window(i, rows_per_window: int, order, inverse, starts, ends, local):
    """Start, rows, per-expert group sizes, in-window slot mask and token-major slot indices of row window ``i``."""
    c0 = i * rows_per_window
    rows = lax.dynamic_slice_in_dim(order, c0, rows_per_window)
    group_sizes = jnp.clip(jnp.minimum(ends, c0 + rows_per_window) - jnp.maximum(starts, c0), 0).astype(I32)
    pos = inverse - c0
    in_window = local & (pos >= 0) & (pos < rows_per_window)
    # -1 marks a slot outside this window (another window's, or another device's route): the gather-sum skips it
    # without moving a row, which matters because only ~1/ep of a token's k slots are local to a device.
    return c0, rows, group_sizes, in_window, jnp.where(in_window, pos, -1)


def _forward(prims: LocalFfnPrimitives, activation: Activation, tc: int, r: int, x, weights, experts, wg, wu, wd):
    tokens, hidden = x.shape
    top_k = experts.shape[1]
    num_local = wg.shape[0]
    num_windows = -(-tc * top_k // r)

    def chunk(c, out):
        t0 = c * tc
        xc = lax.dynamic_slice_in_dim(x, t0, tc)
        wc = lax.dynamic_slice_in_dim(weights, t0, tc).astype(F32)
        routes = _chunk_routes(lax.dynamic_slice_in_dim(experts, t0, tc), num_local, num_windows * r)
        order, inverse, starts, ends, n_local, local = routes

        def window(i, acc):
            def run(acc):
                _, rows, gs, in_window, idx = _window(i, r, order, inverse, starts, ends, local)
                xr = prims.gather(xc, rows // top_k)
                h = activation(prims.gmm(xr, wg, gs), prims.gmm(xr, wu, gs)).astype(x.dtype)
                y = prims.gmm(h, wd, gs)
                return acc + prims.gather_sum(y, idx, jnp.where(in_window, wc, 0.0))

            return lax.cond(i * r < n_local, run, lambda a: a, acc)

        acc = lax.fori_loop(0, num_windows, window, jnp.zeros((tc, hidden), F32))
        return lax.dynamic_update_slice_in_dim(out, acc.astype(out.dtype), t0, 0)

    return lax.fori_loop(0, tokens // tc, chunk, jnp.zeros_like(x))


def _backward(prims: LocalFfnPrimitives, activation: Activation, tc: int, r: int, x, weights, experts, wg, wu, wd, dout):
    tokens, hidden = x.shape
    top_k = experts.shape[1]
    num_local = wg.shape[0]
    num_windows = -(-tc * top_k // r)

    def chunk(c, carry):
        dx, dw, dwg, dwu, dwd = carry
        t0 = c * tc
        xc = lax.dynamic_slice_in_dim(x, t0, tc)
        wc = lax.dynamic_slice_in_dim(weights, t0, tc).astype(F32)
        doc = lax.dynamic_slice_in_dim(dout, t0, tc)
        routes = _chunk_routes(lax.dynamic_slice_in_dim(experts, t0, tc), num_local, num_windows * r)
        order, inverse, starts, ends, n_local, local = routes
        w_flat = wc.reshape(-1)

        def window(i, state):
            def run(state):
                dxc, dwc, dwg, dwu, dwd = state
                c0, rows, gs, in_window, idx = _window(i, r, order, inverse, starts, ends, local)
                valid = ((c0 + jnp.arange(r, dtype=I32)) < n_local)[:, None]
                # Grouped-matmul rows past the window's valid prefix are unspecified (unwritten on TPU, possibly
                # NaN), so every weight-gradient matmul operand computed from them is masked with ``where`` -- a zero
                # scale would not clear a NaN. Those rows are never gathered back token-major.
                scale = jnp.where(valid[:, 0], w_flat[rows], 0.0)[:, None]
                tok = rows // top_k
                xr = prims.gather(xc, tok)
                h, act_vjp = jax.vjp(activation, prims.gmm(xr, wg, gs), prims.gmm(xr, wu, gs))
                dor = prims.gather(doc, tok)
                dh = prims.gmm(dor, wd, gs, transpose_rhs=True).astype(F32)
                row_dot = jnp.sum(dh * h.astype(F32), axis=-1)
                dg, du = act_vjp((dh * scale).astype(h.dtype))
                dg = jnp.where(valid, dg, 0).astype(x.dtype)
                du = jnp.where(valid, du, 0).astype(x.dtype)
                hw = jnp.where(valid, h.astype(F32) * scale, 0.0).astype(x.dtype)
                dwd = prims.tgmm(hw, dor, gs, dwd)
                dwg = prims.tgmm(xr, dg, gs, dwg)
                dwu = prims.tgmm(xr, du, gs, dwu)
                dxr = prims.gmm(dg, wg, gs, transpose_rhs=True) + prims.gmm(du, wu, gs, transpose_rhs=True)
                dxc = dxc + prims.gather_sum(dxr, idx, in_window.astype(F32))
                dwc = dwc + jnp.where(in_window, row_dot[idx], 0.0)
                return dxc, dwc, dwg, dwu, dwd

            return lax.cond(i * r < n_local, run, lambda s: s, state)

        init = (jnp.zeros((tc, hidden), F32), jnp.zeros((tc, top_k), F32), dwg, dwu, dwd)
        dxc, dwc, dwg, dwu, dwd = lax.fori_loop(0, num_windows, window, init)
        dx = lax.dynamic_update_slice_in_dim(dx, dxc.astype(dx.dtype), t0, 0)
        dw = lax.dynamic_update_slice_in_dim(dw, dwc, t0, 0)
        return dx, dw, dwg, dwu, dwd

    init = (jnp.zeros_like(x), jnp.zeros(weights.shape, F32), jnp.zeros_like(wg), jnp.zeros_like(wu), jnp.zeros_like(wd))
    return lax.fori_loop(0, tokens // tc, chunk, init)


def _make_local_ffn(prims: LocalFfnPrimitives, activation: Activation, tc: int, r: int):
    @jax.custom_vjp
    def ffn(x, weights, experts, wg, wu, wd):
        return _forward(prims, activation, tc, r, x, weights, experts, wg, wu, wd)

    def ffn_fwd(x, weights, experts, wg, wu, wd):
        return _forward(prims, activation, tc, r, x, weights, experts, wg, wu, wd), (x, weights, experts, wg, wu, wd)

    def ffn_bwd(res, dout):
        x, weights, experts, wg, wu, wd = res
        dx, dw, dwg, dwu, dwd = _backward(
            prims, activation, tc, r, x, weights, experts, wg, wu, wd, dout.astype(x.dtype)
        )
        return dx, dw.astype(weights.dtype), np.zeros(experts.shape, jax.dtypes.float0), dwg, dwu, dwd

    ffn.defvjp(ffn_fwd, ffn_bwd)
    return ffn


def local_ffn(
    prims: LocalFfnPrimitives,
    x: jax.Array,
    weights: jax.Array,
    experts: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    *,
    activation: Activation | None = None,
    tokens_per_chunk: int | None = None,
    rows_per_window: int | None = None,
    token_align: int = 1,
    row_align: int = 1,
) -> jax.Array:
    """Run the local expert FFN with ``prims`` (see the module docstring); differentiable in every float input.

    ``activation`` must be a pure function of ``(gate, up)`` that closes over no traced values.
    """
    check_local_ffn_shapes(x, weights, experts, w_gate, w_up, w_down)
    tokens, top_k = experts.shape
    tc, r, num_chunks, _ = plan_local_ffn(
        tokens, top_k, tokens_per_chunk, rows_per_window, token_align=token_align, row_align=row_align
    )
    experts = experts.astype(I32)
    pad = num_chunks * tc - tokens
    if pad:
        x = jnp.pad(x, ((0, pad), (0, 0)))
        weights = jnp.pad(weights, ((0, pad), (0, 0)))
        experts = jnp.pad(experts, ((0, pad), (0, 0)), constant_values=-1)
    out = _make_local_ffn(prims, activation or default_activation, tc, r)(x, weights, experts, w_gate, w_up, w_down)
    return out[:tokens] if pad else out


def xla_primitives(dtype) -> LocalFfnPrimitives:
    """XLA primitives: ``take`` gathers, ``ragged_dot`` (fp32 at ``HIGHEST``) and its ragged-contraction form."""
    precision = lax.Precision.HIGHEST if jnp.dtype(dtype) == jnp.float32 else None
    tgmm_dims = lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(([0], [0]), ([], [])),
        lhs_ragged_dimensions=[0],
        rhs_group_dimensions=[],
    )

    def gather(src, rows):
        return jnp.take(src, rows, axis=0, mode="clip")

    def gather_sum(y, idx, w):
        # One [tokens, H] term per slot keeps the temporary at [tokens, H] (a full [tokens * k, H] take would not).
        acc = None
        for j in range(idx.shape[1]):
            live = (idx[:, j] >= 0)[:, None]
            term = jnp.take(y, idx[:, j], axis=0, mode="clip").astype(F32) * w[:, j : j + 1].astype(F32)
            term = jnp.where(live, term, 0.0)
            acc = term if acc is None else acc + term
        return acc

    def gmm(lhs, rhs, group_sizes, transpose_rhs=False):
        if transpose_rhs:
            rhs = jnp.swapaxes(rhs, 1, 2)
        return lax.ragged_dot(lhs, rhs, group_sizes, precision=precision, preferred_element_type=lhs.dtype)

    def tgmm(lhs, rhs, group_sizes, acc):
        part = lax.ragged_dot_general(lhs, rhs, group_sizes, tgmm_dims, precision=precision, preferred_element_type=F32)
        return (acc.astype(F32) + part).astype(acc.dtype)

    return LocalFfnPrimitives(gather=gather, gather_sum=gather_sum, gmm=gmm, tgmm=tgmm)


def moe_local_ffn_xla(
    x: jax.Array,
    weights: jax.Array,
    experts: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    activation: Activation | None = None,
    tokens_per_chunk: int | None = None,
    rows_per_window: int | None = None,
) -> jax.Array:
    """XLA local expert FFN (see the module docstring)."""
    return local_ffn(
        xla_primitives(x.dtype),
        x,
        weights,
        experts,
        w_gate,
        w_up,
        w_down,
        activation=activation,
        tokens_per_chunk=tokens_per_chunk,
        rows_per_window=rows_per_window,
    )


__all__ = (
    "LocalFfnPrimitives",
    "check_local_ffn_shapes",
    "default_activation",
    "local_ffn",
    "moe_local_ffn_xla",
    "plan_local_ffn",
    "xla_primitives",
)
