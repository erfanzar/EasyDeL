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

"""Pallas TPU ``kda_chunk`` (Mosaic-compiled forward and backward) vs the per-token recurrent oracle and XLA."""

from __future__ import annotations

import functools
import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

_ORACLE_PATH = pathlib.Path(__file__).resolve().parents[2] / "_xla" / "test_kda_chunk.py"
_spec = importlib.util.spec_from_file_location("_kda_chunk_xla_test", _ORACLE_PATH)
_oracle_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle_mod)
recurrent_oracle = _oracle_mod.recurrent_oracle
make_inputs = _oracle_mod.make_inputs
packed_ids = _oracle_mod.packed_ids
rel = _oracle_mod.rel

PALLAS = kernel_registry.get("kda_chunk", platform=Platform.PALLAS, backend=Backend.TPU)
XLA = kernel_registry.get("kda_chunk", platform=Platform.XLA, backend=Backend.ANY)


def run(fn, *args, **kw):
    return jax.jit(functools.partial(fn, **kw))(*args)


@pytest.mark.parametrize("block_h", [1, 2])
@pytest.mark.parametrize("chunk_size", [32, 64])
def test_forward_unsegmented_matches_oracle(chunk_size, block_h):
    q, k, v, beta, g, s0 = make_inputs(0)
    out, fin = run(PALLAS, q, k, v, beta, g, s0, chunk_size=chunk_size, block_h=block_h)
    o_ref, f_ref, _ = recurrent_oracle(q, k, v, beta, g, s0)
    assert out.shape == (2, 2, 256, 128) and out.dtype == jnp.float32
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5


@pytest.mark.parametrize("continue_initial_state", [False, True], ids=["reset", "continue"])
@pytest.mark.parametrize("strong", [False, True], ids=["weak_decay", "strong_decay"])
def test_forward_packed_rows_and_chunk_states(strong, continue_initial_state):
    q, k, v, beta, g, s0 = make_inputs(1, L=250, strong=strong)
    seg = packed_ids(2, 250, boundaries=(5, 95, 98, 162, 200), pad_rows=((1, 230, None), (1, 100, 110)), lead_pad=3)
    kw = dict(continue_initial_state=continue_initial_state)
    out, fin, st = run(PALLAS, q, k, v, beta, g, s0, seg, chunk_size=64, return_chunk_states=True, **kw)
    o_ref, f_ref, st_ref = recurrent_oracle(q, k, v, beta, g, s0, seg, chunk_size=64, **kw)
    assert st.shape == (2, 4, 2, 128, 128)
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5 and rel(st, st_ref) < 2e-5
    o_x, f_x, st_x = run(XLA, q, k, v, beta, g, s0, seg, chunk_size=64, return_chunk_states=True, **kw)
    assert rel(out, o_x) < 2e-5 and rel(fin, f_x) < 2e-5 and rel(st, st_x) < 2e-5


@pytest.mark.parametrize(
    ("backward", "bwd_reduce"), [("pallas", "vpu"), ("pallas", "mxu"), ("xla", "vpu")], ids=["pallas", "mxu", "xla"]
)
@pytest.mark.parametrize("segmented", [False, True], ids=["plain", "packed"])
def test_gradients_match_oracle(segmented, backward, bwd_reduce):
    q, k, v, beta, g, s0 = make_inputs(2, L=192, strong=True)
    seg = packed_ids(2, 192, boundaries=(37, 64, 65, 130), pad_rows=((1, 170, None),)) if segmented else None
    rng = np.random.RandomState(3)
    w_out = jnp.asarray(rng.randn(2, 2, 192, 128).astype(np.float32))
    w_fin = jnp.asarray(rng.randn(2, 2, 128, 128).astype(np.float32))

    def loss(fn, *a):
        out, fin = fn(*a)[:2]
        return jnp.sum(out * w_out) + jnp.sum(fin * w_fin)

    f_k = functools.partial(PALLAS, chunk_size=64, backward=backward, bwd_reduce=bwd_reduce)
    f_r = recurrent_oracle
    args = (q, k, v, beta, g, s0, seg)
    g_k = jax.jit(jax.grad(functools.partial(loss, f_k), argnums=range(6)))(*args)
    g_r = jax.jit(jax.grad(functools.partial(loss, f_r), argnums=range(6)))(*args)
    for a, b, name in zip(g_k, g_r, ("dq", "dk", "dv", "dbeta", "dg", "ds0"), strict=True):
        assert np.isfinite(np.asarray(a)).all(), name
        assert rel(a, b) < 1e-4, (name, rel(a, b))


def test_chunk_state_cotangent_reaches_inputs():
    q, k, v, beta, g, s0 = make_inputs(4, L=192)
    seg = packed_ids(2, 192, boundaries=(37, 64, 65, 130), pad_rows=((1, 170, None),))
    w = jnp.asarray(np.random.RandomState(5).randn(2, 3, 2, 128, 128).astype(np.float32))

    def loss(fn, *a):
        _, fin, st = fn(*a)
        return jnp.sum(st * w) + jnp.sum(fin)

    g_p = jax.jit(jax.grad(functools.partial(loss, functools.partial(PALLAS, chunk_size=64, return_chunk_states=True)),
                           argnums=range(6)))(q, k, v, beta, g, s0, seg)  # fmt: skip
    g_x = jax.jit(jax.grad(functools.partial(loss, functools.partial(XLA, chunk_size=64, return_chunk_states=True)),
                           argnums=range(6)))(q, k, v, beta, g, s0, seg)  # fmt: skip
    for a, b in zip(g_p, g_x, strict=True):
        assert rel(a, b) < 1e-4


def test_block_h_two_gradients():
    q, k, v, beta, g, s0 = make_inputs(6, H=4, L=128)

    def loss(fn, *a):
        out, fin = fn(*a)
        return jnp.sum(out**2) + jnp.sum(fin)

    g_p = jax.jit(jax.grad(functools.partial(loss, functools.partial(PALLAS, chunk_size=64, block_h=2)),
                           argnums=range(6)))(q, k, v, beta, g, s0)  # fmt: skip
    g_x = jax.jit(jax.grad(functools.partial(loss, functools.partial(XLA, chunk_size=64)), argnums=range(6)))(
        q, k, v, beta, g, s0
    )
    for a, b in zip(g_p, g_x, strict=True):
        assert rel(a, b) < 1e-4


def test_bf16_inputs_and_default_precision():
    q, k, v, beta, g, s0 = make_inputs(7)
    qb, kb, vb = (x.astype(jnp.bfloat16) for x in (q, k, v))
    out, fin = run(PALLAS, qb, kb, vb, beta, g, s0, chunk_size=64)
    o_ref, f_ref, _ = recurrent_oracle(qb, kb, vb, beta, g, s0)
    assert out.dtype == jnp.float32
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5
    out_d, _ = run(PALLAS, q, k, v, beta, g, s0, chunk_size=64, precision="default")
    o_ref, _, _ = recurrent_oracle(q, k, v, beta, g, s0)
    assert rel(out_d, o_ref) < 5e-2  # bf16 MXU passes for the non-inverse dots


@pytest.mark.parametrize(
    ("dims", "chunk_size", "heads", "block_h"),
    [((64, 128), 64, 2, 1), ((128, 96), 64, 2, 1), ((128, 128), 48, 2, 1), ((128, 128), 64, 3, 2)],
    ids=["k64", "v96", "cs48", "heads_not_multiple"],
)
def test_untileable_shapes_route_to_xla(dims, chunk_size, heads, block_h):
    K, V = dims
    q, k, v, beta, g, s0 = make_inputs(8, H=heads, L=100, K=K, V=V)
    out, fin = run(PALLAS, q, k, v, beta, g, s0, chunk_size=chunk_size, block_h=block_h)
    o_x, f_x = run(XLA, q, k, v, beta, g, s0, chunk_size=chunk_size)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(o_x))
    np.testing.assert_array_equal(np.asarray(fin), np.asarray(f_x))


@pytest.mark.parametrize("kw", [dict(backward="cuda"), dict(bwd_reduce="tensor"), dict(precision="bf16")])
def test_invalid_arguments_raise(kw):
    q, k, v, beta, g, _ = make_inputs(9, B=1, H=1, L=64)
    with pytest.raises(ValueError):
        PALLAS(q, k, v, beta, g, **kw)
