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

"""Shared helpers for TileLang kernel parity tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import ejkernel.kernels._tilelang  # noqa: F401
from ejkernel.kernels._registry import Backend, Platform, kernel_registry

_SEED = 1234


_FP16_FWD_TOL = 5e-3


_FP16_BWD_TOL = 1e-2


def _max_abs(a, b):
    """Max absolute diff between two arrays, computed in fp32."""
    return float(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)).max())


def _max_rel(a, b):
    """Max relative diff = max_abs / max(|b|)."""
    return _max_abs(a, b) / max(float(jnp.abs(b.astype(jnp.float32)).max()), 1e-9)


def _tl(algo):
    return kernel_registry.get(algo, platform=Platform.TILELANG, backend=Backend.GPU)


def _xla(algo):
    return kernel_registry.get(algo, platform=Platform.XLA, backend=Backend.ANY)


def _randn(key, shape, dtype=jnp.float16, scale=0.1):
    return (jax.random.normal(key, shape) * scale).astype(dtype)


def _state_space_v1_inputs(seed, batch=1, seq=4, dim=32, state=8):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    x = _randn(keys[0], (batch, seq, dim))
    A = -(jax.random.uniform(keys[1], (dim, state)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(keys[2], (batch, seq, state))
    C = _randn(keys[3], (batch, seq, state))
    Dsk = _randn(keys[4], (dim,))
    dt = (jax.random.uniform(keys[5], (batch, seq, dim)) * 0.5 + 0.1).astype(jnp.float16)
    return x, A, Bp, C, Dsk, dt


def _state_space_v2_inputs(seed, batch=1, seq=4, heads=2, dim=8, state=8):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 6)
    x = _randn(keys[0], (batch, seq, heads, dim))
    A = -(jax.random.uniform(keys[1], (heads,)) * 0.5 + 0.05).astype(jnp.float16)
    Bp = _randn(keys[2], (batch, seq, 1, state))
    C = _randn(keys[3], (batch, seq, 1, state))
    Dsk = _randn(keys[4], (heads,))
    dt = (jax.random.uniform(keys[5], (batch, seq, heads)) * 0.5 + 0.1).astype(jnp.float16)
    return x, A, Bp, C, Dsk, dt


_FA_FEATURES = ["bias", "mask", "sliding_window", "softcap", "sink", "segment_ids", "gqa", "normalize_off"]


def _fa_feature_inputs(feat):
    """Build ``(q, k, v, kwargs)`` exercising one FA score-space feature."""
    B, N, H, D = 1, 64, 4, 64
    ks = jax.random.split(jax.random.PRNGKey(_SEED), 8)
    q = _randn(ks[0], (B, N, H, D))
    num_kv = 2 if feat == "gqa" else H
    k = _randn(ks[1], (B, N, num_kv, D))
    v = _randn(ks[2], (B, N, num_kv, D))
    kw = {}
    if feat == "bias":
        kw["bias"] = _randn(ks[3], (B, H, N, N), scale=0.2)
    elif feat == "mask":
        kw["attention_mask"] = jax.random.uniform(ks[4], (B, 1, N, N)) > 0.3
    elif feat == "sliding_window":
        kw["sliding_window"] = 16
    elif feat == "softcap":
        kw["logits_soft_cap"] = 5.0
    elif feat == "sink":
        kw["softmax_aux"] = _randn(ks[5], (4,), scale=0.5)
    elif feat == "segment_ids":
        seg = jnp.broadcast_to((jnp.arange(N) < 32).astype(jnp.int32)[None, :], (B, N))
        kw["q_segment_ids"] = seg
        kw["kv_segment_ids"] = seg
    elif feat == "normalize_off":
        kw["normalize_output"] = False
    return q, k, v, kw


def _scan_recurrent_ref(q, k, v, scale, g=None, g_gamma=None, gk=None, gv=None, initial_state=None, reverse=False):
    """Reference: ``jax.lax.scan`` of the canonical linear-attention step.

    The XLA backend ships a hand-written custom_vjp whose gradient
    convention does not exactly match ``jax.grad`` through ``jax.lax.scan``
    (the latter is the mathematical ground truth and the spec the tile-lang
    kernel implements). We use the scan reference here so the assertion is
    against the autograd ground truth.
    """

    def step(h, qkv):
        qt, kt, vt, gt, gkt, gvt = qkv
        h_new = (
            h
            * jnp.exp(gt)[..., :, None]
            * decay[:, :, None, None]
            * jnp.exp(gkt)[..., :, None]
            * jnp.exp(gvt)[..., None, :]
            + kt[..., :, None] * vt[..., None, :]
        )
        o = jnp.einsum("bhij,bhi->bhj", h_new, qt) * scale
        return h_new, o

    B, _S, H, Dq = q.shape
    Dv = v.shape[-1]
    if k.shape[2] != H:
        reps = H // k.shape[2]
        k = jnp.repeat(k, repeats=reps, axis=2)
        v = jnp.repeat(v, repeats=reps, axis=2)
    if g_gamma is None:
        g_gamma = jnp.zeros((H,), dtype=jnp.float32)
    if g_gamma.ndim == 1:
        decay = jnp.exp(jnp.broadcast_to(g_gamma[None, :], (B, H)))
    else:
        decay = jnp.exp(jnp.broadcast_to(g_gamma, (B, H)))
    qT = jnp.transpose(q.astype(jnp.float32), (1, 0, 2, 3))
    kT = jnp.transpose(k.astype(jnp.float32), (1, 0, 2, 3))
    vT = jnp.transpose(v.astype(jnp.float32), (1, 0, 2, 3))
    if g is None:
        g = jnp.zeros_like(q)
    if gk is None:
        gk = jnp.zeros_like(q)
    if gv is None:
        gv = jnp.zeros_like(v)
    gT = jnp.transpose(g.astype(jnp.float32), (1, 0, 2, 3))
    gkT = jnp.transpose(gk.astype(jnp.float32), (1, 0, 2, 3))
    gvT = jnp.transpose(gv.astype(jnp.float32), (1, 0, 2, 3))
    if reverse:
        qT = jnp.flip(qT, axis=0)
        kT = jnp.flip(kT, axis=0)
        vT = jnp.flip(vT, axis=0)
        gT = jnp.flip(gT, axis=0)
        gkT = jnp.flip(gkT, axis=0)
        gvT = jnp.flip(gvT, axis=0)
    h0 = jnp.zeros((B, H, Dq, Dv), jnp.float32) if initial_state is None else initial_state.astype(jnp.float32)
    h_final, oT = jax.lax.scan(step, h0, (qT, kT, vT, gT, gkT, gvT))
    if reverse:
        oT = jnp.flip(oT, axis=0)
    o = jnp.transpose(oT, (1, 0, 2, 3)).astype(q.dtype)
    return o, h_final


def _make_mla_ragged_inputs(seed: int = _SEED + 6):
    key = jax.random.PRNGKey(seed)
    k0, k1, k2, k3, k4 = jax.random.split(key, 5)
    num_seqs = 2
    pages_per_seq = 3
    page_size = 4
    num_pages = 7
    page_size_per_pack = 2
    kv_packing = 2
    total_tokens = 5
    num_q_heads = 3
    nope_dim = 8
    pe_dim = 8
    queries_nope = _randn(k0, (total_tokens, num_q_heads, nope_dim))
    queries_pe = _randn(k1, (total_tokens, num_q_heads, pe_dim))
    keys_values = _randn(k2, (total_tokens, nope_dim))
    keys_pe = _randn(k3, (total_tokens, pe_dim))
    kv_cache = _randn(k4, (num_pages, page_size_per_pack, kv_packing, 256))
    kv_lens = jnp.array([7, 5], dtype=jnp.int32)
    block_tables = jnp.array([4, 1, 6, 2, 5, 0], dtype=jnp.int32)
    query_start_loc = jnp.array([0, 3, 5], dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    return (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        pages_per_seq,
        page_size,
    )


def _deepseek_inputs():
    B, S, Hq, D, Hkv, L, Hi, Di = 1, 64, 4, 64, 4, 128, 2, 64
    ks = jax.random.split(jax.random.PRNGKey(_SEED), 7)
    return (
        _randn(ks[0], (B, S, Hq, D)),
        _randn(ks[1], (B, S, L)),
        _randn(ks[2], (L, Hkv, D)),
        _randn(ks[3], (L, Hkv, D)),
        _randn(ks[4], (B, S, Hi, Di)),
        _randn(ks[5], (B, S, Di)),
        _randn(ks[6], (B, S, Hi), scale=0.5),
    )


def _ssm1_scan_ref(x, A, Bp, C, Dsk, dt):
    def step(h, inp):
        x_t, B_t, C_t, dt_t = inp
        dA = jnp.exp(A * dt_t[:, None])
        dBx = dt_t[:, None] * B_t[None, :] * x_t[:, None]
        h_new = dA * h + dBx
        y = jnp.sum(h_new * C_t[None, :], axis=-1) + Dsk * x_t
        return h_new, y

    xT = jnp.transpose(x.astype(jnp.float32), (1, 0, 2))[:, 0]
    BT = jnp.transpose(Bp.astype(jnp.float32), (1, 0, 2))[:, 0]
    CT = jnp.transpose(C.astype(jnp.float32), (1, 0, 2))[:, 0]
    dtT = jnp.transpose(dt.astype(jnp.float32), (1, 0, 2))[:, 0]
    h0 = jnp.zeros((x.shape[-1], A.shape[-1]), jnp.float32)
    _, yT = jax.lax.scan(step, h0, (xT, BT, CT, dtT))
    return yT[None].astype(x.dtype)
