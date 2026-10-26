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

"""Registry entry (Pallas, TPU) for the fused chunked per-channel-decay KDA, differentiable via ``jax.custom_vjp``.

Forward: :func:`kda_chunk_pallas_fwd` (one ``pallas_call``), which also emits the state entering every chunk as the
backward residual (``[B, H, C, K, V]`` fp32 -- the per-chunk carry a rematerialised scan would also keep).

Backward: ``backward="pallas"`` runs :func:`kda_chunk_pallas_bwd` (one reverse-sweep ``pallas_call``);
``backward="xla"`` runs a reversed ``lax.scan`` whose body takes ``jax.vjp`` of the same per-chunk math. The
custom_vjp boundary sits on the prepared fp32 inputs; the elementwise prologue (L2 norm, scale, padding, packed-row
masking) is plain JAX and differentiates on its own.

Shapes the kernel does not tile (``K`` or ``V`` not a multiple of 128, ``chunk_size`` not a power of two in
``[8, 128]``, ``num_heads`` not a multiple of ``block_h``) and non-TPU backends route to the XLA implementation.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.kda_chunk._xla_impl import (
    LANES,
    META_ROW_RUN,
    batched_chunk_step,
    kda_chunk_xla,
    per_chunk_cumsum,
    per_chunk_reverse_cumsum,
    prepare_inputs,
    resolve_precision,
    validate_options,
)
from ._pallas_impl_bwd import kda_chunk_pallas_bwd
from ._pallas_impl_fwd import kda_chunk_pallas_fwd, pack_segment_meta


def is_tileable(k_dim: int, v_dim: int, num_heads: int, chunk_size: int, block_h: int) -> bool:
    """Whether the Pallas kernel tiles this shape (otherwise the XLA implementation runs)."""
    cs = int(chunk_size)
    return (
        k_dim % LANES == 0
        and v_dim % LANES == 0
        and 8 <= cs <= LANES
        and cs & (cs - 1) == 0
        and int(block_h) > 0
        and num_heads % int(block_h) == 0
    )


def _to_chunks(x: Array, chunk_size: int) -> Array:
    """``[B, H, Lp, *] -> [C, B, H, cs, *]``."""
    B, H, Lp = x.shape[:3]
    y = x.reshape(B, H, Lp // chunk_size, chunk_size, *x.shape[3:])
    return jnp.moveaxis(y, 2, 0)


def _from_chunks(x: Array) -> Array:
    """``[C, B, H, cs, *] -> [B, H, Lp, *]``."""
    y = jnp.moveaxis(x, 0, 2)
    B, H, C, cs = y.shape[:4]
    return y.reshape(B, H, C * cs, *y.shape[4:])


@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
def _core(q, k, v, beta, g, state0, meta, chunk_size, block_h, precision, backward, bwd_reduce, use_states):
    """Fused forward on the prepared fp32 inputs. Returns ``(out, final_state, chunk_states_in)``."""
    del backward, bwd_reduce, use_states
    b4 = beta[..., None]
    return kda_chunk_pallas_fwd(
        q,
        k,
        k * b4,
        v * b4,
        per_chunk_cumsum(g, chunk_size),
        state0,
        meta,
        chunk_size=chunk_size,
        block_h=block_h,
        precision=precision,
    )


def _core_fwd(q, k, v, beta, g, state0, meta, chunk_size, block_h, precision, backward, bwd_reduce, use_states):
    outs = _core(q, k, v, beta, g, state0, meta, chunk_size, block_h, precision, backward, bwd_reduce, use_states)
    return outs, (q, k, v, beta, g, meta, outs[2])


def _core_bwd(chunk_size, block_h, precision, backward, bwd_reduce, use_states, res, cts):
    q, k, v, beta, g, meta, states_in = res
    d_out, d_final, d_states = cts
    d_meta = None if meta is None else jnp.zeros_like(meta)
    b4 = beta[..., None]
    if backward == "pallas":
        dq, dk_direct, dkb, dvb, dgc, d_state0 = kda_chunk_pallas_bwd(
            q,
            k,
            k * b4,
            v * b4,
            per_chunk_cumsum(g, chunk_size),
            states_in,
            d_out,
            d_final,
            d_states if use_states else None,
            meta,
            chunk_size=chunk_size,
            block_h=block_h,
            precision=precision,
            reduce_mode=bwd_reduce,
        )
        dk = dk_direct + dkb * b4
        dv = dvb * b4
        dbeta = jnp.sum(dkb * k + dvb * v, axis=-1)
        dg = per_chunk_reverse_cumsum(dgc, chunk_size)
        return dq, dk, dv, dbeta, dg, d_state0, d_meta

    prec = resolve_precision(precision)
    xs = {
        "q": _to_chunks(q, chunk_size),
        "k": _to_chunks(k, chunk_size),
        "v": _to_chunks(v, chunk_size),
        "beta": _to_chunks(beta, chunk_size),
        "g": _to_chunks(g, chunk_size),
        "state_in": jnp.moveaxis(states_in, 2, 0),
        "d_out": _to_chunks(d_out, chunk_size),
        "d_state_in": jnp.moveaxis(d_states, 2, 0),
    }
    if meta is not None:
        xs["run"] = jnp.moveaxis(meta[:, :, META_ROW_RUN, :chunk_size], 1, 0)  # [C, B, cs]

    def body(ct_state, x):
        run = x.get("run")

        def f(state, q_c, k_c, v_c, b_c, g_c):
            return batched_chunk_step(state, q_c, k_c, v_c, b_c, g_c, run, prec, inverse="masked")

        _, vjp = jax.vjp(f, x["state_in"], x["q"], x["k"], x["v"], x["beta"], x["g"])
        d_state, dq, dk, dv, dbeta, dg = vjp((ct_state, x["d_out"]))
        return d_state + x["d_state_in"], (dq, dk, dv, dbeta, dg)

    d_state0, grads = lax.scan(body, d_final, xs, reverse=True)
    dq, dk, dv, dbeta, dg = (_from_chunks(x) for x in grads)
    return dq, dk, dv, dbeta, dg, d_state0, d_meta


_core.defvjp(_core_fwd, _core_bwd)


@kernel_registry.register("kda_chunk", Platform.PALLAS, Backend.TPU, priority=10)
def kda_chunk(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len v_dim"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len head_dim"],
    initial_state: Float[Array, "batch num_heads head_dim v_dim"] | None = None,
    segment_ids: Int[Array, "batch seq_len"] | None = None,
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
    continue_initial_state: bool = False,
    return_chunk_states: bool = False,
    precision: str = "highest",
    block_h: int = 1,
    backward: str = "pallas",
    bwd_reduce: str = "vpu",
) -> tuple[Array, ...]:
    """Fused chunked KDA with per-channel decay on TPU (same contract as the XLA entry).

    Args:
        query: Queries ``[B, H, L, K]``.
        key: Keys ``[B, H, L, K]``.
        value: Values ``[B, H, L, V]``.
        beta: Delta-rule gates ``[B, H, L]``.
        decay: Per-token per-channel log-decay ``[B, H, L, K]`` (``<= 0``).
        initial_state: Optional state ``[B, H, K, V]``.
        segment_ids: Optional packed-row ids ``[B, L]`` (negative = padding).
        chunk_size: Chunk length (power of two in ``[8, 128]`` for the kernel).
        use_qk_l2norm: L2-normalise queries and keys first.
        continue_initial_state: Cached-continuation packed-row semantics (see the XLA entry).
        return_chunk_states: Also return the state after every chunk ``[B, C, H, K, V]``.
        precision: ``"highest"`` or ``"default"`` for the non-inverse dots (the chunk inverse is always HIGHEST).
        block_h: Heads per grid step.
        backward: ``"pallas"`` (fused reverse-sweep kernel) or ``"xla"`` (reversed-scan recompute).
        bwd_reduce: ``"vpu"`` or ``"mxu"`` evaluation of the backward's weighted reductions.

    Returns:
        ``(out [B, H, L, V] fp32, final_state [B, H, K, V] fp32)`` (+ ``chunk_states``).
    """
    validate_options(chunk_size, precision, backward, bwd_reduce)
    _, H, L, K = query.shape
    V = value.shape[-1]
    if jax.default_backend() != "tpu" or not is_tileable(K, V, H, chunk_size, block_h):
        if int(block_h) <= 0:
            raise ValueError(f"block_h must be positive, got {block_h}")
        return kda_chunk_xla(
            query,
            key,
            value,
            beta,
            decay,
            initial_state,
            segment_ids,
            chunk_size=int(chunk_size),
            use_qk_l2norm=use_qk_l2norm,
            continue_initial_state=continue_initial_state,
            return_chunk_states=return_chunk_states,
            precision=precision,
        )
    cs = int(chunk_size)
    p = prepare_inputs(
        query, key, value, beta, decay, initial_state, segment_ids, cs, use_qk_l2norm, continue_initial_state
    )
    meta = None if p.local_run is None else pack_segment_meta(p.local_run)
    out, final, states_in = _core(
        p.q, p.k, p.v, p.beta, p.g, p.state, meta, cs, int(block_h), precision, backward, bwd_reduce,
        bool(return_chunk_states),
    )  # fmt: skip
    out = out[:, :, :L, :]
    if return_chunk_states:
        post = jnp.concatenate([states_in[:, :, 1:], final[:, :, None]], axis=2)  # [B, H, C, K, V]
        return out, final, post.transpose(0, 2, 1, 3, 4)
    return out, final


__all__ = ("is_tileable", "kda_chunk")
