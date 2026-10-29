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

"""Tests for the opt-in ``DsaGatherAttn`` EasyDeL operation adapter.

Registry wiring, the public-ejkernel-only import boundary, value and ``(dq, dlatent)`` against an independent float64
numpy reference on the Pallas and XLA platforms, and the sequence-split ``shard_map`` layout on a multi-device mesh
(queries split over the head axis's devices, every head local) against the unsharded XLA reference.
"""

from __future__ import annotations

import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh


def _selection(rng, batch, seq, kpool, k_sel):
    """Distinct causal complete-pool ids per query (``-1`` padded), random order; row 0 query 3 selects nothing."""
    idx = -np.ones((batch, seq, k_sel), np.int32)
    for b in range(batch):
        for s in range(seq):
            vis = (s + 1) // kpool
            if vis:
                picks = rng.choice(vis, size=min(k_sel, vis), replace=False)
                idx[b, s, : len(picks)] = picks
            idx[b, s] = rng.permutation(idx[b, s])
    idx[0, 3] = -1
    return idx


def _np_reference(q, lat, idx, kpool, scale, g):
    """float64 attention over the selected pools + own-pool tail and its analytic ``(dq, dlatent)``."""
    q, lat, g = (np.asarray(x, np.float64) for x in (q, lat, g))
    batch, seq, _ = idx.shape
    open_ = np.zeros((batch, seq, lat.shape[1]), bool)
    for b in range(batch):
        for s in range(seq):
            for p in idx[b, s]:
                if p >= 0:
                    open_[b, s, p * kpool : (p + 1) * kpool] = True
            if s % kpool != kpool - 1:
                open_[b, s, (s // kpool) * kpool : s + 1] = True
    logits = np.einsum("bnhd,bsd->bnhs", q, lat) * scale
    mask = np.broadcast_to(open_[:, :, None], logits.shape)
    m = np.where(mask, logits, -np.inf).max(-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    p = np.where(mask, np.exp(logits - m), 0.0)
    denom = p.sum(-1, keepdims=True)
    p = p / np.where(denom == 0, 1.0, denom)
    out = np.einsum("bnhs,bsd->bnhd", p, lat)
    ds = p * (np.einsum("bnhd,bsd->bnhs", g, lat) - np.sum(g * out, -1, keepdims=True))
    dq = scale * np.einsum("bnhs,bsd->bnhd", ds, lat)
    dlat = np.einsum("bnhs,bnhd->bsd", p, g) + scale * np.einsum("bnhs,bnhd->bsd", ds, q)
    return out, dq, dlat


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
    from easydel.operations.kernels import DsaGatherAttn

    assert DsaGatherAttn.get_impl_name() == "dsa_gather_attention"
    assert OperationRegistry.get("dsa_gather_attention") is DsaGatherAttn
    assert DsaGatherAttn.get_requirements() is not None


def test_no_private_ejkernel_imports():
    from easydel.operations.kernels import dsa_gather_attention as adapter_mod

    src = pathlib.Path(adapter_mod.__file__).read_text()
    forbidden = r"ejkernel\.kernels\.|_pallas|_xla\b|_triton|pallas_call|jax\.experimental\.pallas|\bpl\."
    hits = [ln for ln in src.splitlines() if re.search(forbidden, ln)]
    assert not hits, f"forbidden private-kernel references: {hits}"
    assert "from ejkernel.modules import dsa_gather_attention" in src


@pytest.mark.parametrize("platform", [None, "xla"])
def test_value_and_gradients_match_numpy(platform):
    from easydel.operations.kernels import DsaGatherAttn

    rng = np.random.default_rng(0)
    batch, heads, seq, dim, kpool, k_sel = 2, 4, 256, 128, 4, 12
    idx = _selection(rng, batch, seq, kpool, k_sel)
    q = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    lat = rng.standard_normal((batch, seq, dim)).astype(np.float32)
    g = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
    scale = dim**-0.5
    op = DsaGatherAttn(_metadata(operation_configs={"dsa_gather_attention": {"block_q": 8, "pools_per_step": 32}}))
    ids = jnp.asarray(idx)

    def f(q_, lat_):
        return op(q_, lat_, ids, None, kpool=kpool, softmax_scale=scale, platform=platform).attention_outputs

    out, vjp = jax.vjp(jax.jit(f), jnp.asarray(q), jnp.asarray(lat))
    grads = vjp(jnp.asarray(g))
    want = _np_reference(q, lat, idx, kpool, scale, g)
    for got, ref, name in zip((out, *grads), want, ("out", "dq", "d_latent"), strict=True):
        got = np.asarray(got, np.float64)
        assert np.isfinite(got).all(), name
        np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4 * np.abs(ref).max(), err_msg=name)
    # Query 3 of row 0: own pool complete (3 % 4 == 3) and nothing selected, so no key is open: output exactly 0.
    assert np.all(np.asarray(out)[0, 3] == 0.0)


def test_sequence_split_layout_on_a_mesh():
    from easydel.operations.kernels import DsaGatherAttn

    n = jax.device_count()
    if n < 2 or jax.default_backend() != "tpu":
        pytest.skip("needs a multi-device TPU mesh")
    devices = np.array(jax.devices()).reshape(1, 1, n // 2 if n >= 4 else 1, 1, min(n, 2), 1)
    mesh = Mesh(devices, ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    rng = np.random.default_rng(1)
    batch, heads, seq, dim, kpool, k_sel = 2, 4, 512, 128, 4, 24
    idx = jnp.asarray(_selection(rng, batch, seq, kpool, k_sel))
    q = jnp.asarray(rng.standard_normal((batch, seq, heads, dim)), jnp.float32)
    lat = jnp.asarray(rng.standard_normal((batch, seq, dim)), jnp.float32)
    op = DsaGatherAttn(_metadata(mesh=mesh))
    layout = op.layout(batch=batch, seq=seq, num_heads=heads)
    assert layout is not None and "tp" in layout.seq, f"the query sequence must be split over tp: {layout}"

    def loss(q_, lat_, op=op):
        out = op(q_, lat_, idx, None, kpool=kpool).attention_outputs
        return jnp.sum(out**2), out

    with mesh:
        (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(q, lat)
    ref_op = DsaGatherAttn(_metadata())

    def ref_loss(q_, lat_):
        out = ref_op(q_, lat_, idx, None, kpool=kpool, platform="xla").attention_outputs
        return jnp.sum(out**2), out

    (_, ref), ref_grads = jax.value_and_grad(ref_loss, (0, 1), has_aux=True)(q, lat)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=2e-4, atol=2e-4)
    for a, b in zip(grads, ref_grads, strict=True):
        b = np.asarray(b)
        np.testing.assert_allclose(np.asarray(a), b, rtol=2e-4, atol=2e-4 * np.abs(b).max())
