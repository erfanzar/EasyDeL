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

"""XLA ``kda_chunk`` vs an independent per-token recurrent oracle (values, chunk states, gradients, packed rows)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry

XLA = kernel_registry.get("kda_chunk", platform=Platform.XLA, backend=Backend.ANY)


def recurrent_oracle(q, k, v, beta, g, state0=None, seg=None, continue_initial_state=False, chunk_size=None):
    """Per-token KDA recurrence with elementwise fp32 math (no matmuls), written independently of the kernel.

    Returns ``(out [B, H, L, V], final [B, H, K, V], states_at_chunk_ends [B, C, H, K, V] | None)``.
    """
    B, H, L, K = q.shape
    V = v.shape[-1]

    def l2(x):
        return x / jnp.sqrt(jnp.sum(x * x, -1, keepdims=True) + 1e-6)

    q = l2(q.astype(jnp.float32)) * K**-0.5
    k = l2(k.astype(jnp.float32))
    s = jnp.zeros((B, H, K, V), jnp.float32) if state0 is None else state0.astype(jnp.float32)
    seg_t = jnp.zeros((B, L), jnp.int32) if seg is None else seg.astype(jnp.int32)
    prev0 = jnp.zeros((B,), jnp.int32) if seg is None else jnp.full((B,), -1, jnp.int32)

    def step(carry, x):
        s, prev = carry
        qt, kt, vt, bt, gt, st = x
        valid = st >= 0
        reset = valid & (st != prev)
        if continue_initial_state:
            reset = reset & (prev >= 0)
        old = s
        s = jnp.where(reset[:, None, None, None], 0.0, s)
        s = s * jnp.exp(gt)[..., None]
        mem = jnp.sum(s * kt[..., None], axis=-2)
        delta = (vt - mem) * bt[..., None]
        s = s + kt[..., None] * delta[..., None, :]
        o = jnp.sum(s * qt[..., None], axis=-2)
        o = jnp.where(valid[:, None, None], o, 0.0)
        pad_state = old if continue_initial_state else jnp.zeros_like(s)
        s = jnp.where(valid[:, None, None, None], s, pad_state)
        nxt = jnp.where(valid, st, prev if continue_initial_state else -1)
        return (s, nxt), (o, s)

    xs = (
        q.transpose(2, 0, 1, 3),
        k.transpose(2, 0, 1, 3),
        v.astype(jnp.float32).transpose(2, 0, 1, 3),
        beta.astype(jnp.float32).transpose(2, 0, 1),
        g.astype(jnp.float32).transpose(2, 0, 1, 3),
        seg_t.T,
    )
    (final, _), (out, states) = jax.lax.scan(step, (s, prev0), xs)
    chunk_states = None
    if chunk_size is not None:
        ends = np.minimum(np.arange(chunk_size - 1, L + chunk_size - 1, chunk_size), L - 1)
        chunk_states = states[ends].transpose(1, 0, 2, 3, 4)
    return out.transpose(1, 2, 0, 3), final, chunk_states


def make_inputs(seed, B=2, H=2, L=256, K=128, V=128, strong=False):
    rng = np.random.RandomState(seed)
    f = lambda *s: jnp.asarray(rng.randn(*s).astype(np.float32))  # noqa: E731
    q, k, v = f(B, H, L, K), f(B, H, L, K), f(B, H, L, V)
    beta = jax.nn.sigmoid(f(B, H, L))
    g = -5.0 * jax.nn.sigmoid(f(B, H, L, K)) if strong else -0.1 * jax.nn.softplus(f(B, H, L, K))
    s0 = f(B, H, K, V) * 0.1
    return q, k, v, beta, g, s0


def packed_ids(B, L, boundaries=(37, 64, 65, 130, 200), pad_rows=((1, 230, None),), lead_pad=None):
    seg = np.zeros((B, L), np.int32)
    for t0 in boundaries:
        if t0 < L:
            seg[:, t0:] += 1
    for row, start, stop in pad_rows:
        seg[row, start:stop] = -1
    if lead_pad is not None:
        seg[0, :lead_pad] = -1
    return jnp.asarray(seg)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def run_xla(*args, **kw):
    return jax.jit(functools.partial(XLA, **kw))(*args)


@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("dims", [(16, 32), (128, 128)], ids=["k16v32", "k128"])
def test_unsegmented_matches_oracle(chunk_size, dims):
    K, V = dims
    q, k, v, beta, g, s0 = make_inputs(0, K=K, V=V)
    out, fin = run_xla(q, k, v, beta, g, s0, chunk_size=chunk_size)
    o_ref, f_ref, _ = recurrent_oracle(q, k, v, beta, g, s0)
    assert out.shape == (2, 2, 256, V) and out.dtype == jnp.float32 and fin.shape == (2, 2, K, V)
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5


@pytest.mark.parametrize("strong", [False, True], ids=["weak_decay", "strong_decay"])
def test_odd_length_and_chunk_states(strong):
    q, k, v, beta, g, s0 = make_inputs(1, L=150, strong=strong)
    out, fin, st = run_xla(q, k, v, beta, g, s0, chunk_size=64, return_chunk_states=True)
    o_ref, f_ref, st_ref = recurrent_oracle(q, k, v, beta, g, s0, chunk_size=64)
    assert st.shape == (2, 3, 2, 128, 128)
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5 and rel(st, st_ref) < 2e-5


@pytest.mark.parametrize("continue_initial_state", [False, True], ids=["reset", "continue"])
def test_packed_rows_match_oracle(continue_initial_state):
    q, k, v, beta, g, s0 = make_inputs(2)
    seg = packed_ids(2, 256, pad_rows=((1, 230, None), (1, 100, 110)), lead_pad=5)
    kw = dict(continue_initial_state=continue_initial_state)
    out, fin, st = run_xla(q, k, v, beta, g, s0, seg, chunk_size=64, return_chunk_states=True, **kw)
    o_ref, f_ref, st_ref = recurrent_oracle(q, k, v, beta, g, s0, seg, chunk_size=64, **kw)
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5 and rel(st, st_ref) < 2e-5
    # the boundaries bite: without segments the outputs differ after the first boundary
    plain, _ = run_xla(q, k, v, beta, g, s0, chunk_size=64)
    assert np.abs(np.asarray(plain[:, :, 40:64]) - np.asarray(out[:, :, 40:64])).max() > 1e-3


def test_gradients_match_oracle():
    q, k, v, beta, g, s0 = make_inputs(3, L=128, strong=True)
    seg = packed_ids(2, 128, boundaries=(37, 64, 65), pad_rows=((1, 110, None),))
    rng = np.random.RandomState(4)
    w_out = jnp.asarray(rng.randn(2, 2, 128, 128).astype(np.float32))
    w_fin = jnp.asarray(rng.randn(2, 2, 128, 128).astype(np.float32))

    def loss(fn, *a):
        out, fin = fn(*a)[:2]
        return jnp.sum(out * w_out) + jnp.sum(fin * w_fin)

    for s in (None, seg):
        f_k = lambda q, k, v, b, g, s0, s=s: XLA(q, k, v, b, g, s0, s, chunk_size=32)  # noqa: E731
        f_r = lambda q, k, v, b, g, s0, s=s: recurrent_oracle(q, k, v, b, g, s0, s)  # noqa: E731
        g_k = jax.jit(jax.grad(functools.partial(loss, f_k), argnums=range(6)))(q, k, v, beta, g, s0)
        g_r = jax.jit(jax.grad(functools.partial(loss, f_r), argnums=range(6)))(q, k, v, beta, g, s0)
        for a, b, name in zip(g_k, g_r, ("dq", "dk", "dv", "dbeta", "dg", "ds0"), strict=True):
            assert np.isfinite(np.asarray(a)).all(), name
            assert rel(a, b) < 1e-4, (name, rel(a, b))


def test_bf16_inputs():
    q, k, v, beta, g, s0 = make_inputs(5)
    q, k, v = (x.astype(jnp.bfloat16) for x in (q, k, v))
    out, fin = run_xla(q, k, v, beta, g, s0, chunk_size=64)
    o_ref, f_ref, _ = recurrent_oracle(q, k, v, beta, g, s0)
    assert out.dtype == jnp.float32
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5


@pytest.mark.parametrize(
    "kw",
    [dict(precision="bf16"), dict(backward="cuda"), dict(bwd_reduce="tensor"), dict(chunk_size=0), dict(block_h=0)],
)
def test_invalid_arguments_raise(kw):
    q, k, v, beta, g, _ = make_inputs(6, B=1, H=1, L=16, K=16, V=16)
    with pytest.raises(ValueError):
        XLA(q, k, v, beta, g, **kw)


def test_registry_signatures_agree():
    assert kernel_registry.validate_signatures("kda_chunk")
