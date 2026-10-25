# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``scoring_kernel="pallas"`` (the ``indexer_scores`` operation) equals the dense ``"xla"`` scoring path.

Covers the shared helper against a float64 NumPy oracle (values and cotangents), every indexer that routes dense
scoring through it (token / pool :class:`SparseIndexer`, :class:`CompressedIndexerScorer`, shared-key scores), the
4-device ``shard_map`` dispatch, the operation registration, and config validation.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from easydel.layers.indexer import (
    CompressedIndexerConfig,
    CompressedIndexerScorer,
    IndexerConfig,
    IndexerKind,
    SharedKeyIndexerConfig,
    SparseIndexer,
    shared_key_scores,
)
from easydel.layers.indexer._shared_key import fused_head_scores
from easydel.operations import OperationRegistry
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

HI = jax.lax.Precision.HIGHEST


def _f64(x):
    return np.asarray(jnp.asarray(x, jnp.float32), np.float64)


def _oracle(q, k, w, cot):
    """Float64 ``sum_h w[b,s,h] relu(q[b,s,h] . k[b,t])`` and its analytic cotangents for ``[B,S,H,D]`` queries."""
    q, k, w, cot = _f64(q), _f64(k), _f64(w), _f64(cot)
    a = np.einsum("bshd,btd->bsht", q, k)
    out = np.sum(np.maximum(a, 0.0) * w[..., None], axis=2)
    g = cot[:, :, None] * w[..., None] * (a > 0)
    dq = np.einsum("bsht,btd->bshd", g, k)
    dk = np.einsum("bsht,bshd->btd", g, q)
    dw = np.sum(cot[:, :, None] * np.maximum(a, 0.0), axis=-1)
    return out, (dq, dk, dw)


def _rel(a, b) -> float:
    a, b = _f64(a), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _inputs(batch, seq, heads, entries, dim, seed=0):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((batch, seq, heads, dim)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((batch, entries, dim)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((batch, seq, heads)), jnp.float32)
    cot = jnp.asarray(rng.standard_normal((batch, seq, entries)), jnp.float32)
    return q, k, w, cot


def test_operation_is_registered():
    assert OperationRegistry.get("indexer_scores").get_impl_name() == "indexer_scores"


@pytest.mark.parametrize(("batch", "seq", "heads", "entries", "dim"), [(2, 40, 4, 96, 16), (1, 130, 8, 300, 64)])
def test_fused_head_scores_matches_float64_oracle(batch, seq, heads, entries, dim):
    q, k, w, cot = _inputs(batch, seq, heads, entries, dim, seed=seq)
    want, want_g = _oracle(q, k, w, cot)
    got = jax.jit(lambda a, b, c: fused_head_scores(a, b, c, precision=HI))(q, k, w)
    assert got.shape == (batch, seq, entries) and got.dtype == jnp.float32
    assert _rel(got, want) < 1e-5
    grads = jax.jit(
        jax.grad(lambda a, b, c: jnp.sum(fused_head_scores(a, b, c, precision=HI) * cot), argnums=(0, 1, 2))
    )(q, k, w)
    for name, g, ref in zip(("q", "k", "w"), grads, want_g, strict=True):
        assert _rel(g, ref) < 1e-5, (name, _rel(g, ref))


def _sparse(kind, kernel, **extra):
    base = dict(
        kind=kind,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=12,
        hidden_size=32,
        q_input_dim=32,
        score_activation="relu",
        head_reduction="weighted",
        rope_style="none",
        scoring_kernel=kernel,
    )
    if kind == IndexerKind.POOL:
        base.update(packed_state="key_gate_valid", stop_gradient=True, kpool_size=4, select_tail=True)
    base.update(extra)
    return SparseIndexer(
        IndexerConfig(**base), dtype=jnp.float32, param_dtype=jnp.float32, precision=HI, rngs=spx.Rngs(0)
    )


@pytest.mark.parametrize("reduction", ["weighted", "uniform"])
def test_sparse_token_indexer_on_off_equivalence(reduction):
    off, on = (
        _sparse(IndexerKind.TOKEN, "xla", head_reduction=reduction),
        _sparse(IndexerKind.TOKEN, "pallas", head_reduction=reduction),
    )
    hidden = jax.random.normal(jax.random.PRNGKey(0), (2, 24, 32), jnp.float32)
    q_res = jax.random.normal(jax.random.PRNGKey(1), (2, 24, 32), jnp.float32)
    a = off(hidden_states=hidden, q_resid=q_res)
    b = on(hidden_states=hidden, q_resid=q_res)
    assert _rel(b.score_proxy, _f64(a.score_proxy)) < 1e-5
    np.testing.assert_array_equal(np.asarray(a.topk_indices), np.asarray(b.topk_indices))
    cot = jax.random.normal(jax.random.PRNGKey(2), a.score_proxy.shape, jnp.float32)

    def grad(module):
        return jax.grad(lambda h: jnp.sum(module(hidden_states=h, q_resid=q_res).score_proxy * cot))(hidden)

    assert _rel(grad(on), _f64(grad(off))) < 1e-5


@pytest.mark.parametrize("cached", [False, True])
def test_sparse_pool_indexer_on_off_equivalence(cached):
    off, on = _sparse(IndexerKind.POOL, "xla"), _sparse(IndexerKind.POOL, "pallas")
    hidden = jax.random.normal(jax.random.PRNGKey(3), (2, 24, 32), jnp.float32)
    q_res = jax.random.normal(jax.random.PRNGKey(4), (2, 24, 32), jnp.float32)
    mask = jnp.ones((2, 24), jnp.bool_).at[0, :5].set(False)
    packed = off(hidden_states=hidden, q_resid=q_res, attention_mask=mask).packed_state if cached else None
    a = off(hidden_states=hidden, q_resid=q_res, cached_packed=packed)
    b = on(hidden_states=hidden, q_resid=q_res, cached_packed=packed)
    np.testing.assert_array_equal(np.asarray(a.topk_indices), np.asarray(b.topk_indices))
    np.testing.assert_array_equal(np.asarray(a.topk_mask), np.asarray(b.topk_mask))


def test_compressed_scorer_on_off_equivalence():
    def scorer(kernel):
        config = CompressedIndexerConfig(16, 8, 4, 32, 4, 4, scoring_kernel=kernel)
        return CompressedIndexerScorer(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=HI, rngs=spx.Rngs(5)
        )

    off, on = scorer("xla"), scorer("pallas")
    rng = np.random.default_rng(23)
    q, keys, hidden = (jnp.asarray(rng.normal(size=s), jnp.float32) for s in ((2, 33, 4, 32), (2, 50, 32), (2, 33, 16)))
    a, b = off(q, keys, hidden), on(q, keys, hidden)
    assert _rel(b, _f64(a)) < 1e-5
    cot = jnp.asarray(rng.normal(size=a.shape), jnp.float32)
    ga = jax.grad(lambda x: jnp.sum(off(x, keys, hidden) * cot))(q)
    gb = jax.grad(lambda x: jnp.sum(on(x, keys, hidden) * cot))(q)
    assert _rel(gb, _f64(ga)) < 1e-5
    # decode (one query) keeps the dense path and is exactly the dense result
    np.testing.assert_array_equal(
        np.asarray(on(q[:, :1], keys, hidden[:, :1])), np.asarray(off(q[:, :1], keys, hidden[:, :1]))
    )


def test_shared_key_scores_on_off_equivalence():
    q, k, w, cot = _inputs(2, 48, 4, 24, 16, seed=31)
    positions = jnp.broadcast_to(jnp.arange(48, dtype=jnp.int32), (2, 48)).at[1, -4:].set(-1)
    off = shared_key_scores(q, w, k, positions, 2, HI)
    on = shared_key_scores(q, w, k, positions, 2, HI, scoring_kernel="pallas")
    finite = np.isfinite(np.asarray(off))
    np.testing.assert_array_equal(finite, np.isfinite(np.asarray(on)))
    assert _rel(np.asarray(on)[finite], np.asarray(off, np.float64)[finite]) < 1e-5
    masked_cot = jnp.where(jnp.asarray(finite), cot, 0.0)

    def grad(kernel):
        def loss(a, b, c):
            s = shared_key_scores(a, c, b, positions, 2, HI, scoring_kernel=kernel)
            return jnp.sum(jnp.where(jnp.isfinite(s), s, 0.0) * masked_cot)

        return jax.grad(loss, argnums=(0, 1, 2))(q, k, w)

    for gb, ga in zip(grad("pallas"), grad("xla"), strict=True):
        assert _rel(gb, _f64(ga)) < 1e-5


def _mesh_source(topology):
    if jax.device_count() < 4:
        pytest.skip("Requires four devices")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(topology), ("data", "tokens", "features"))
    axis = spx.PartitionAxis(
        batch_axis="data",
        query_sequence_axis="tokens",
        hidden_state_axis="features",
        decode_batch_axis="data",
        decode_hidden_state_axis="features",
    )
    return SimpleNamespace(mesh=mesh, runtime_sharding_resolver=coerce_runtime_sharding_resolver(axis, mesh=mesh))


@pytest.mark.parametrize("topology", [(4, 1, 1), (1, 4, 1), (2, 2, 1), (1, 1, 4)])
def test_fused_head_scores_on_four_device_mesh(topology):
    source = _mesh_source(topology)
    q, k, w, cot = _inputs(4, 64, 8, 192, 32, seed=41)
    want, want_g = _oracle(q, k, w, cot)
    placed = [
        jax.device_put(x, NamedSharding(source.mesh, spec))
        for x, spec in ((q, P("data", "tokens")), (k, P("data")), (w, P("data", "tokens")))
    ]

    def fn(a, b, c):
        return fused_head_scores(a, b, c, precision=HI, mesh_source=source)

    got = jax.jit(fn)(*placed)
    assert _rel(got, want) < 1e-5
    grads = jax.jit(jax.grad(lambda a, b, c: jnp.sum(fn(a, b, c) * cot), argnums=(0, 1, 2)))(*placed)
    for g, ref in zip(grads, want_g, strict=True):
        assert _rel(g, ref) < 1e-5
    # without a layout provider a multi-device call stays correct (GSPMD-partitioned XLA reference)
    assert _rel(jax.jit(lambda a, b, c: fused_head_scores(a, b, c, precision=HI))(*placed), want) < 1e-5


def test_config_validation():
    with pytest.raises(ValueError, match="scoring_kernel"):
        IndexerConfig(scoring_kernel="triton")
    with pytest.raises(ValueError, match="relu"):
        IndexerConfig(scoring_kernel="pallas", score_activation="none")
    assert IndexerConfig().scoring_kernel == "xla"
    with pytest.raises(ValueError, match="scoring_kernel"):
        CompressedIndexerConfig(8, 4, 2, 4, 2, 2, scoring_kernel="dense")
    with pytest.raises(ValueError, match="scoring_kernel"):
        SharedKeyIndexerConfig(8, 6, 4, 4, 3, ratio=2, key_input_dim=5, scoring_kernel="auto")
    q, k, w, _ = _inputs(1, 2, 1, 2, 2)
    positions = jnp.zeros((1, 2), jnp.int32)
    with pytest.raises(ValueError, match="scoring_kernel"):
        shared_key_scores(q, w, k, positions, 1, scoring_kernel="x")
