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


def _grads(fn, args, w_out, w_fin):
    def loss(*a):
        out, fin = fn(*a)[:2]
        return jnp.sum(out * w_out) + jnp.sum(fin * w_fin)

    return jax.jit(jax.grad(loss, argnums=range(6)))(*args)


def _nrel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize("block_h", [1, 2])
@pytest.mark.parametrize("segmented", [False, True], ids=["plain", "packed"])
def test_factored_pairwise_values_and_gradients_match_oracle(segmented, block_h):
    """With the gate bound the kernel factors the pairwise decay into MXU matmuls; at HIGHEST it matches the oracle."""
    q, k, v, beta, g, s0 = make_inputs(20, L=192, strong=True)
    assert float(g.min()) >= -5.0
    seg = packed_ids(2, 192, boundaries=(37, 64, 65, 130), pad_rows=((1, 170, None),)) if segmented else None
    rng = np.random.RandomState(21)
    w_out = jnp.asarray(rng.randn(2, 2, 192, 128).astype(np.float32))
    w_fin = jnp.asarray(rng.randn(2, 2, 128, 128).astype(np.float32))
    f_k = functools.partial(PALLAS, chunk_size=64, block_h=block_h, decay_lower_bound=-5.0)
    args = (q, k, v, beta, g, s0, seg)
    out, fin = jax.jit(f_k)(*args)
    o_ref, f_ref, _ = recurrent_oracle(*args)
    assert rel(out, o_ref) < 2e-5 and rel(fin, f_ref) < 2e-5
    g_k = _grads(f_k, args, w_out, w_fin)
    g_r = _grads(recurrent_oracle, args, w_out, w_fin)
    for a, b, name in zip(g_k, g_r, ("dq", "dk", "dv", "dbeta", "dg", "ds0"), strict=True):
        assert np.isfinite(np.asarray(a)).all(), name
        assert rel(a, b) < 1e-4, (name, rel(a, b))


def test_factored_pairwise_default_precision_keeps_the_dense_error_budget():
    """At ``precision="default"`` the factored path's error vs fp32 stays within 2x the dense kernel's.

    A single bf16 pass on the backward's decay reductions loses accuracy in the log-decay gradient, so the factored
    path runs them (and the chunk inverse) at bf16x3.
    """
    q, k, v, beta, g, s0 = make_inputs(22, L=256, strong=True)
    rng = np.random.RandomState(23)
    w_out = jnp.asarray(rng.randn(2, 2, 256, 128).astype(np.float32))
    w_fin = jnp.asarray(rng.randn(2, 2, 128, 128).astype(np.float32))
    args = (q, k, v, beta, g, s0, None)
    dense = functools.partial(PALLAS, chunk_size=64, precision="default")
    factored = functools.partial(PALLAS, chunk_size=64, precision="default", decay_lower_bound=-5.0)
    o_ref, _, _ = recurrent_oracle(*args)
    e_dense, e_fact = _nrel(jax.jit(dense)(*args)[0], o_ref), _nrel(jax.jit(factored)(*args)[0], o_ref)
    assert e_fact <= 2.0 * e_dense + 1e-4, (e_fact, e_dense)
    g_r = _grads(recurrent_oracle, args, w_out, w_fin)
    g_d = _grads(dense, args, w_out, w_fin)
    g_f = _grads(factored, args, w_out, w_fin)
    for a, b, r, name in zip(g_f, g_d, g_r, ("dq", "dk", "dv", "dbeta", "dg", "ds0"), strict=True):
        assert np.isfinite(np.asarray(a)).all(), name
        assert _nrel(a, r) <= 2.0 * _nrel(b, r) + 1e-4, (name, _nrel(a, r), _nrel(b, r))


@pytest.mark.parametrize("precision", ["highest", "default"])
@pytest.mark.parametrize("segmented", [False, True], ids=["plain", "packed"])
def test_factored_head_batched_block_matches_single_head_blocks(segmented, precision):
    """``block_h=4`` runs the factored chunk math as one head-batched program; it must equal ``block_h=1``.

    Covers the forward outputs, the final and per-chunk states, and every gradient, with the chunk-state cotangent in
    the loss so the backward's ``d_chunk_states_in`` path is exercised; at HIGHEST it also matches the oracle.
    """
    q, k, v, beta, g, s0 = make_inputs(30, H=4, L=192, strong=True)
    assert float(g.min()) >= -5.0
    seg = packed_ids(2, 192, boundaries=(37, 64, 65, 130), pad_rows=((1, 170, None),)) if segmented else None
    rng = np.random.RandomState(31)
    w_out = jnp.asarray(rng.randn(2, 4, 192, 128).astype(np.float32))
    w_fin = jnp.asarray(rng.randn(2, 4, 128, 128).astype(np.float32))
    w_st = jnp.asarray(rng.randn(2, 3, 4, 128, 128).astype(np.float32))
    args = (q, k, v, beta, g, s0, seg)

    def fn(block_h):
        return functools.partial(
            PALLAS, chunk_size=64, block_h=block_h, precision=precision, decay_lower_bound=-5.0, return_chunk_states=True
        )

    def loss(f, *a):
        out, fin, st = f(*a)
        return jnp.sum(out * w_out) + jnp.sum(fin * w_fin) + jnp.sum(st * w_st)

    one, four = fn(1), fn(4)
    vals_1, vals_4 = jax.jit(one)(*args), jax.jit(four)(*args)
    for a, b, name in zip(vals_4, vals_1, ("out", "final", "chunk_states"), strict=True):
        assert a.shape == b.shape and a.dtype == jnp.float32, name
        assert rel(a, b) <= 1e-6, (name, rel(a, b))
    grads_1 = jax.jit(jax.grad(functools.partial(loss, one), argnums=range(6)))(*args)
    grads_4 = jax.jit(jax.grad(functools.partial(loss, four), argnums=range(6)))(*args)
    for a, b, name in zip(grads_4, grads_1, ("dq", "dk", "dv", "dbeta", "dg", "ds0"), strict=True):
        assert np.isfinite(np.asarray(a)).all(), name
        assert rel(a, b) <= 1e-6, (name, rel(a, b))
    if precision == "highest":
        o_ref, f_ref, st_ref = recurrent_oracle(*args, chunk_size=64)
        assert rel(vals_4[0], o_ref) < 2e-5 and rel(vals_4[1], f_ref) < 2e-5 and rel(vals_4[2], st_ref) < 2e-5


def test_dense_pairwise_opts_out_and_the_bound_changes_the_kernel():
    q, k, v, beta, g, s0 = make_inputs(24, L=128, strong=True)
    plain = run(PALLAS, q, k, v, beta, g, s0, chunk_size=64)
    dense = run(PALLAS, q, k, v, beta, g, s0, chunk_size=64, decay_lower_bound=-5.0, pairwise="dense")
    factored = run(PALLAS, q, k, v, beta, g, s0, chunk_size=64, decay_lower_bound=-5.0)
    for a, b in zip(dense, plain, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(factored[0]), np.asarray(plain[0]))


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


@pytest.mark.parametrize("reverse", [False, True], ids=["prefix", "suffix"])
def test_in_kernel_chunk_cumsum_keeps_fp32_accuracy(reverse):
    """``chunk_cumsum`` inside a ``pallas_call`` matches float64 prefix / suffix sums to fp32 rounding.

    It computes the kernels' chunk prefix sums of ``g`` (and the backward's suffix sums of ``dg_cumsum``) as a
    triangular ones-matrix MXU matmul at HIGHEST; a single bfloat16 pass would round every term to 8 mantissa bits
    (error ~1e-3 of the running magnitude).
    """
    from ejkernel.kernels._pallas.tpu.kda_chunk._pallas_impl_fwd import chunk_cumsum
    from jax.experimental import pallas as pl

    h, cs, kdim = 8, 64, 128
    rng = np.random.default_rng(31)
    gate = rng.uniform(-5.0, 0.0, (2, h // 2, cs, kdim))  # strongly decaying log-gates down to the -5 bound
    mixed = 5.0 * rng.standard_normal((2, h // 2, cs, kdim))  # mixed-sign cotangents
    x = np.concatenate([gate, mixed], axis=1).astype(np.float32)

    def kernel(x_ref, o_ref):
        o_ref[...] = chunk_cumsum(x_ref[...], reverse=reverse)

    spec = pl.BlockSpec((None, h, cs, kdim), lambda i: (i, 0, 0, 0))
    out = pl.pallas_call(
        kernel, grid=(2,), in_specs=[spec], out_specs=spec, out_shape=jax.ShapeDtypeStruct(x.shape, jnp.float32)
    )(jnp.asarray(x))
    x64 = x.astype(np.float64)
    if reverse:
        ref = np.cumsum(x64[..., ::-1, :], axis=-2)[..., ::-1, :]
        scale = np.cumsum(np.abs(x64)[..., ::-1, :], axis=-2)[..., ::-1, :]
    else:
        ref = np.cumsum(x64, axis=-2)
        scale = np.cumsum(np.abs(x64), axis=-2)
    out = np.asarray(out, np.float64)
    assert out.shape == x.shape
    assert np.max(np.abs(out - ref) / scale) < 1e-5


@pytest.mark.parametrize("kw", [dict(backward="cuda"), dict(bwd_reduce="tensor"), dict(precision="bf16")])
def test_invalid_arguments_raise(kw):
    q, k, v, beta, g, _ = make_inputs(9, B=1, H=1, L=64)
    with pytest.raises(ValueError):
        PALLAS(q, k, v, beta, g, **kw)
