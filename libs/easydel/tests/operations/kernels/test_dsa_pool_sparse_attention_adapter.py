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

"""Tests for the opt-in ``DsaPoolSparseAttn`` EasyDeL operation adapter.

Registry wiring, the public-ejkernel-only import boundary, value and gradients (query, latent, pooled
straight-through term) against an independent float64 numpy reference on the Pallas and XLA platforms, tiles taken
from ``operation_configs``, and the head-parallel ``shard_map`` path on a multi-device mesh.
"""

from __future__ import annotations

import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh


def _selection(rng, batch, seq, kpool, select, n_tails=3):
    """Causal whole-pool selection + up to ``n_tails`` tail keys; batch row 0 query 9 selects nothing."""
    pool_mask = np.zeros((batch, seq, seq // kpool), bool)
    tails = -np.ones((batch, seq, n_tails), np.int32)
    n_sel = max(1, select // kpool)
    for b in range(batch):
        for s in range(seq):
            hi = (s + 1) // kpool
            if hi:
                lo = max(0, hi - 3 * n_sel)
                pool_mask[b, s, rng.choice(np.arange(lo, hi), size=min(n_sel, hi - lo), replace=False)] = True
            t = rng.integers(0, n_tails + 1)
            if t:
                tails[b, s, :t] = np.sort(rng.integers(0, s + 1, size=t))
    pool_mask[0, 9] = False
    tails[0, 9] = -1
    return pool_mask, tails


def _np_reference(q, lat, pool_mask, tails, kpool, scale, ste, g):
    """float64 attention over the open set (set semantics, empty rows -> 0) and its analytic gradients."""
    q, lat, g = (np.asarray(x, np.float64) for x in (q, lat, g))
    batch, seq, _ = pool_mask.shape
    open_ = np.repeat(pool_mask, kpool, axis=-1)[:, :, :seq].copy()
    for b in range(batch):
        for s in range(seq):
            for t in tails[b, s]:
                if t >= 0:
                    open_[b, s, t] = True
    open_ &= np.tril(np.ones((seq, seq), bool))[None]
    del ste  # straight-through term: value 0 for any input, only its cotangent (the pool-sum of dS) matters
    logits = np.einsum("bhsd,bld->bhsl", q, lat) * scale
    mask = np.broadcast_to(open_[:, None], logits.shape)
    m = np.where(mask, logits, -np.inf).max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    p = np.where(mask, np.exp(logits - m), 0.0)
    denom = p.sum(-1, keepdims=True)
    p = p / np.where(denom == 0, 1.0, denom)
    out = np.einsum("bhsl,bld->bhsd", p, lat)
    ds = p * (np.einsum("bhsd,bld->bhsl", g, lat) - np.sum(g * out, -1, keepdims=True))
    dq = scale * np.einsum("bhsl,bld->bhsd", ds, lat)
    dlat = np.einsum("bhsl,bhsd->bld", p, g) + scale * np.einsum("bhsl,bhsd->bld", ds, q)
    dste = ds.sum(1).reshape(batch, seq, -1, kpool).sum(-1)
    return out, dq, dlat, dste


def _metadata(mesh=None, operation_configs=None):
    from easydel.operations._operation_meta import OperationMetadata

    return OperationMetadata(
        runtime_dtype=jnp.float32,
        runtime_softmax_dtype=jnp.float32,
        _stored_mesh=mesh,
        operation_configs=operation_configs,
    )


def test_registered_and_requirements():
    from easydel.operations import OperationRegistry
    from easydel.operations.kernels import DsaPoolSparseAttn

    assert DsaPoolSparseAttn.get_impl_name() == "dsa_pool_sparse_attention"
    assert OperationRegistry.get("dsa_pool_sparse_attention") is DsaPoolSparseAttn
    req = DsaPoolSparseAttn.get_requirements()
    assert req is not None


def test_no_private_ejkernel_imports():
    from easydel.operations.kernels import dsa_pool_sparse_attention as adapter_mod

    src = pathlib.Path(adapter_mod.__file__).read_text()
    forbidden = r"ejkernel\.kernels\.|_pallas|_xla|_triton|pallas_call|jax\.experimental\.pallas|\bpl\."
    hits = [ln for ln in src.splitlines() if re.search(forbidden, ln)]
    assert not hits, f"forbidden private-kernel references: {hits}"
    assert "from ejkernel.modules import dsa_sparse_attention" in src


@pytest.mark.parametrize("platform", [None, "xla"])
def test_value_and_gradients_match_numpy(platform):
    from easydel.operations.kernels import DsaPoolSparseAttn

    rng = np.random.default_rng(0)
    batch, heads, seq, dim, kpool = 2, 4, 512, 128, 4
    pool_mask, tails = _selection(rng, batch, seq, kpool, select=96)
    q = rng.standard_normal((batch, heads, seq, dim)).astype(np.float32)
    lat = rng.standard_normal((batch, seq, dim)).astype(np.float32)
    ste = (0.3 * rng.standard_normal(pool_mask.shape)).astype(np.float32)
    g = rng.standard_normal((batch, heads, seq, dim)).astype(np.float32)
    scale = dim**-0.5
    op = DsaPoolSparseAttn(_metadata(operation_configs={"dsa_pool_sparse_attention": {"block_q": 128, "block_kv": 256}}))
    pm, tl = jnp.asarray(pool_mask), jnp.asarray(tails)

    def f(q, lat, ste):
        res = op(q, lat, pm, tl, ste, kpool=kpool, softmax_scale=scale, platform=platform)
        return res.attention_outputs

    out, vjp = jax.vjp(jax.jit(f), jnp.asarray(q), jnp.asarray(lat), jnp.asarray(ste))
    grads = vjp(jnp.asarray(g))
    want = _np_reference(q, lat, pool_mask, tails, kpool, scale, ste, g)
    for got, ref, name in zip((out, *grads), want, ("out", "dq", "d_latent", "d_ste_pool"), strict=True):
        got = np.asarray(got, np.float64)
        assert np.isfinite(got).all(), name
        np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4 * np.abs(ref).max(), err_msg=name)
    assert np.all(np.asarray(out)[0, :, 9] == 0.0)  # a query with no open key


def test_shard_map_path_on_a_mesh():
    from easydel.operations.kernels import DsaPoolSparseAttn

    n = jax.device_count()
    if n < 2:
        pytest.skip("needs a multi-device mesh")
    devices = np.array(jax.devices()).reshape(1, 1, n // 2 if n >= 4 else 1, 1, min(n, 2), 1)
    mesh = Mesh(devices, ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    rng = np.random.default_rng(1)
    batch, heads, seq, dim, kpool = 2, 4, 512, 128, 4
    pool_mask, tails = _selection(rng, batch, seq, kpool, select=96)
    q = jnp.asarray(rng.standard_normal((batch, heads, seq, dim)), jnp.float32)
    lat = jnp.asarray(rng.standard_normal((batch, seq, dim)), jnp.float32)
    pm, tl = jnp.asarray(pool_mask), jnp.asarray(tails)
    op = DsaPoolSparseAttn(_metadata(mesh=mesh))
    cfg = {"block_q": 256, "block_kv": 256}

    def loss(q, lat, op=op):
        out = op(q, lat, pm, tl, None, kpool=kpool, cfg=cfg).attention_outputs
        return jnp.sum(out**2), out

    with mesh:
        (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(q, lat)
    ref_op = DsaPoolSparseAttn(_metadata())
    (_, ref), ref_grads = jax.value_and_grad(
        lambda q, lat: _xla_loss(ref_op, q, lat, pm, tl, kpool), (0, 1), has_aux=True
    )(q, lat)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-4, atol=2e-4)
    for a, b in zip(grads, ref_grads, strict=True):
        b = np.asarray(b)
        np.testing.assert_allclose(np.asarray(a), b, rtol=2e-4, atol=2e-4 * np.abs(b).max())


def _xla_loss(op, q, lat, pm, tl, kpool):
    out = op(q, lat, pm, tl, None, kpool=kpool, platform="xla").attention_outputs
    return jnp.sum(out**2), out
