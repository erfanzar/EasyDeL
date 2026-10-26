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

"""Pool-bitmap block-sparse DSA attention, Pallas TPU (real Mosaic compilation) vs the dense XLA reference.

Forward value, ``(dq, d_latent, d_ste_pool)`` through both backwards (fused, two-pass) on clustered / spread
selections and on an adversarial batch (a fully padded row = q-blocks with ``count == 0``, three packed segments,
a padded tail); fp32 and bf16; kpool 4 and 2; the per-pair cost variants; the visit modes (``dense`` bitwise
``sparse``, ``budget`` vs the reference on the visited set); the fused backward's core-group split; sequence
padding; the public module operation incl. its ``shard_map`` path; a production-shaped compile (H=16, D=512).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.dsa_sparse_attention._tables import build_tables
from ejkernel.modules import DsaSparseAttention, DsaSparseAttentionConfig
from ejkernel.modules import dsa_sparse_attention as dsa_op
from jax.sharding import Mesh, PartitionSpec

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

PALLAS = kernel_registry.get("dsa_sparse_attention", platform=Platform.PALLAS, backend=Backend.TPU)
XLA_REF = kernel_registry.get("dsa_sparse_attention", platform=Platform.XLA, backend=Backend.ANY)


def random_selection(rng, batch, seq, kpool, select, n_tails=3, clustered=True):
    """Causal pool selection of ~``select`` keys per query (whole pools only) plus up to ``n_tails`` tail keys."""
    pool_mask = np.zeros((batch, seq, -(-seq // kpool)), bool)
    tails = -np.ones((batch, seq, n_tails), np.int32)
    n_sel = max(1, select // kpool)
    for b in range(batch):
        for s in range(seq):
            hi = (s + 1) // kpool
            if hi:
                lo = max(0, hi - 3 * n_sel) if clustered else 0
                pool_mask[b, s, rng.choice(np.arange(lo, hi), size=min(n_sel, hi - lo), replace=False)] = True
            t = rng.integers(0, n_tails + 1)
            if t:
                tails[b, s, :t] = np.sort(rng.integers(0, s + 1, size=t))
    return jnp.asarray(pool_mask), jnp.asarray(tails)


def adversarial_selection(seq=1024, kpool=4, n_tails=3):
    """Row 0 fully padded (count == 0 everywhere); row 1 three packed segments; row 2 a 256-position padded tail."""
    rng = np.random.default_rng(96)
    pool_mask = np.zeros((3, seq, seq // kpool), bool)
    tails = -np.ones((3, seq, n_tails), np.int32)
    for lo, hi in ((0, 300), (300, 700), (700, seq)):
        for s in range(lo, hi):
            first = lo // kpool + (1 if lo % kpool else 0)
            last = (s + 1) // kpool
            if last > first:
                pool_mask[1, s, rng.choice(np.arange(first, last), size=min(8, last - first), replace=False)] = True
            rem = (s + 1) % kpool
            if rem:
                tails[1, s, :rem] = np.arange(s + 1 - rem, s + 1)
            else:
                tails[1, s, 0] = s
    for s in range(seq - 256):
        last = (s + 1) // kpool
        if last:
            pool_mask[2, s, rng.choice(np.arange(last), size=min(8, last), replace=False)] = True
        rem = (s + 1) % kpool
        tails[2, s, :rem] = np.arange(s + 1 - rem, s + 1)
        if not rem:
            tails[2, s, 0] = s
    return jnp.asarray(pool_mask), jnp.asarray(tails)


def _arrays(rng, batch, heads, seq, dim, dtype):
    q = jnp.asarray(rng.standard_normal((batch, heads, seq, dim)), dtype)
    lat = jnp.asarray(rng.standard_normal((batch, seq, dim)), dtype)
    g = jnp.asarray(rng.standard_normal((batch, heads, seq, dim)), dtype)
    return q, lat, g


def _value_and_grads(fn, q, lat, pool_mask, tails, g, **kw):
    ste = jnp.zeros(pool_mask.shape, jnp.float32)

    def loss(q, lat, ste):
        out = fn(q, lat, pool_mask, tails, ste, **kw)
        return jnp.sum(out.astype(jnp.float32) * g.astype(jnp.float32)), out

    (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(q, lat, ste)
    return (out, *grads)


def _assert_close(got, want, tol, name):
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    assert np.all(np.isfinite(got)), name
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol * max(float(np.abs(want).max()), 1e-6), err_msg=name)


NAMES = ("out", "dq", "d_latent", "d_ste_pool")


@pytest.mark.parametrize(
    ("seq", "kpool", "bq", "bk", "dtype", "clustered"),
    [
        (1024, 4, 256, 512, jnp.float32, True),
        (1024, 4, 128, 128, jnp.float32, False),
        (768, 2, 256, 128, jnp.float32, True),  # kpool 2: 64 keys per bitmap word
        (1024, 4, 256, 256, jnp.bfloat16, True),
    ],
)
@pytest.mark.parametrize("backward", ["fused", "two_pass"])
def test_matches_reference_value_and_gradients(seq, kpool, bq, bk, dtype, clustered, backward):
    rng = np.random.default_rng(seq + bk + bq)
    pool_mask, tails = random_selection(rng, 1, seq, kpool, select=seq // 4, clustered=clustered)
    _, _, count, _ = build_tables(pool_mask, tails, kpool=kpool, block_q=bq, block_kv=bk)
    assert float(count.mean()) < seq // bk  # some block pairs are actually skipped
    q, lat, g = _arrays(rng, 1, 4, seq, 128, dtype)
    kw = dict(kpool=kpool, block_q=bq, block_kv=bk)
    got = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, backward=backward, **kw)
    want = _value_and_grads(
        XLA_REF, q.astype(jnp.float32), lat.astype(jnp.float32), pool_mask, tails, g.astype(jnp.float32), **kw
    )
    tol = 3e-2 if dtype == jnp.bfloat16 else 2e-4
    for a, b, name in zip(got, want, NAMES, strict=True):
        _assert_close(a, b, tol, name)
    assert got[0].dtype == dtype and got[0].shape == q.shape
    assert float(jnp.abs(got[3]).max()) > 0  # the indexer gradient flows


@pytest.mark.parametrize("backward", ["fused", "two_pass"])
def test_adversarial_batch_padded_and_packed_rows(backward):
    """Fully padded rows (count == 0 q-blocks: the clamped DMA index), packed segments, a padded tail."""
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    q, lat, g = _arrays(np.random.default_rng(1), 3, 2, seq, 128, jnp.float32)
    kw = dict(kpool=kpool, block_q=bq, block_kv=bk)
    got = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, backward=backward, **kw)
    want = _value_and_grads(XLA_REF, q, lat, pool_mask, tails, g, **kw)
    for a, b, name in zip(got, want, NAMES, strict=True):
        _assert_close(a, b, 2e-4, name)
    out, dq, dlat, _ = (np.asarray(x) for x in got)
    assert np.all(out[0] == 0.0) and np.all(dq[0] == 0.0) and np.all(dlat[0] == 0.0)  # the fully padded row
    assert np.all(out[2, :, -256:] == 0.0)  # the padded tail


@pytest.mark.parametrize(
    "variant",
    [
        dict(fwd_nt=True),
        dict(fwd_lean=True),
        dict(stale_tau=32.0),
        dict(stale_tau=0.0),
        dict(backward="two_pass", bwd_lean=True, pool_sum="matmul"),
        dict(backward="fused", pool_sum="matmul"),
        dict(backward="fused", fused_groups=1),
    ],
    ids=lambda v: "+".join(f"{k}={v[k]}" for k in v),
)
def test_variants_match_base(variant):
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    q, lat, g = _arrays(np.random.default_rng(2), 3, 2, seq, 128, jnp.float32)
    kw = dict(kpool=kpool, block_q=bq, block_kv=bk)
    base = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, **kw)
    got = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, **variant, **kw)
    for a, b, name in zip(got, base, NAMES, strict=True):
        if name == "out" and (variant.get("fwd_nt") or variant.get("fwd_lean")):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"{name}: must be bitwise the base")
        else:
            _assert_close(a, b, 1e-4, name)


def test_fused_group_split_is_a_reorder():
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    q, lat, g = _arrays(np.random.default_rng(3), 3, 2, seq, 128, jnp.bfloat16)
    kw = dict(kpool=kpool, block_q=bq, block_kv=bk, backward="fused")
    g1 = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, fused_groups=1, **kw)
    g2 = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, fused_groups=2, **kw)
    np.testing.assert_array_equal(np.asarray(g1[1]), np.asarray(g2[1]))  # dq is per q-block: bitwise
    _assert_close(g2[2], g1[2], 1e-2, "d_latent")  # partial sums in another order, then one bf16 cast
    np.testing.assert_allclose(np.asarray(g2[3]), np.asarray(g1[3]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dense_mode_is_bitwise_sparse(dtype):
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    q, lat, g = _arrays(np.random.default_rng(4), 3, 2, seq, 128, dtype)
    for backward in ("fused", "two_pass"):
        kw = dict(kpool=kpool, block_q=bq, block_kv=bk, backward=backward)
        s = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, mode="sparse", **kw)
        d = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, mode="dense", **kw)
        for a, b, name in zip(s, d, NAMES, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"{backward} {name}")


def test_budget_mode_matches_reference_on_the_visited_set():
    seq, kpool, bq, bk = 1024, 4, 256, 256
    pool_mask, tails = adversarial_selection(seq, kpool)
    q, lat, g = _arrays(np.random.default_rng(5), 3, 2, seq, 128, jnp.float32)
    kw = dict(kpool=kpool, block_q=bq, block_kv=bk, mode="budget:2")
    got = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, **kw)
    want = _value_and_grads(XLA_REF, q, lat, pool_mask, tails, g, **kw)
    for a, b, name in zip(got, want, NAMES, strict=True):
        _assert_close(a, b, 2e-4, name)
    full = XLA_REF(q, lat, pool_mask, tails, kpool=kpool)
    assert float(jnp.abs(got[0] - full).max()) > 1e-2  # dropping kv-blocks is a real numerics change


def test_unaligned_sequence_is_padded():
    rng = np.random.default_rng(6)
    seq, kpool = 1000, 4
    pool_mask, tails = random_selection(rng, 1, seq, kpool, select=128)
    q, lat, g = _arrays(rng, 1, 2, seq, 128, jnp.float32)
    kw = dict(kpool=kpool, block_q=256, block_kv=512)
    got = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, **kw)
    want = _value_and_grads(XLA_REF, q, lat, pool_mask, tails, g, **kw)
    for a, b, name in zip(got, want, NAMES, strict=True):
        assert a.shape == b.shape, name
        _assert_close(a, b, 2e-4, name)


def test_invalid_arguments_raise():
    pool_mask, tails = random_selection(np.random.default_rng(0), 1, 256, 4, select=32)
    q = jnp.zeros((1, 1, 256, 128), jnp.float32)
    for bad in (
        dict(backward="cwa"),
        dict(mode="gather"),
        dict(pool_sum="sum"),
        dict(fused_groups=3),
        dict(block_kv=64),
        dict(block_q=100),
    ):
        with pytest.raises((ValueError, Exception)):
            PALLAS(q, q[:, 0], pool_mask, tails, kpool=4, **bad)


def test_public_operation_dispatch_and_shard_map():
    assert "create_shard_map_wrapper" in DsaSparseAttention.__dict__
    seq, kpool = 1024, 4
    pool_mask, tails = adversarial_selection(seq, kpool)
    pool_mask, tails = jnp.concatenate([pool_mask, pool_mask[1:2]]), jnp.concatenate([tails, tails[1:2]])
    q, lat, _ = _arrays(np.random.default_rng(7), 4, 4, seq, 128, jnp.float32)
    cfg = DsaSparseAttentionConfig(block_q=256, block_kv=256)
    pallas = dsa_op(q, lat, pool_mask, tails, kpool=kpool, cfg=cfg, platform="pallas")
    xla = dsa_op(q, lat, pool_mask, tails, kpool=kpool, cfg=cfg, platform="xla")
    _assert_close(pallas, xla, 2e-4, "pallas vs xla through the public op")
    n = jax.device_count()
    mesh = Mesh(np.array(jax.devices()).reshape(n // 2 if n > 1 else 1, min(n, 2)), ("dp", "tp"))
    in_specs = (PartitionSpec("dp", "tp", None, None), PartitionSpec("dp", None, None))
    in_specs = (*in_specs, PartitionSpec("dp", None, None), PartitionSpec("dp", None, None))
    out_spec = PartitionSpec("dp", "tp", None, None)

    def loss(q, lat):
        out = dsa_op(q, lat, pool_mask, tails, kpool=kpool, cfg=cfg, mesh=mesh, in_specs=in_specs, out_specs=out_spec)
        return jnp.sum(out**2), out

    (_, sharded), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(q, lat)
    _assert_close(sharded, xla, 2e-4, "shard_map forward")
    ref_grads = jax.grad(lambda q, lat: jnp.sum(XLA_REF(q, lat, pool_mask, tails, kpool=kpool) ** 2), (0, 1))(q, lat)
    for a, b, name in zip(grads, ref_grads, ("dq", "d_latent"), strict=True):
        _assert_close(a, b, 2e-4, f"shard_map {name}")


@pytest.mark.parametrize("backward", ["fused", "two_pass"])
def test_production_shape_compiles_and_matches(backward):
    """H=16, D=512, bf16, 256 x 512 tiles: the scoped-VMEM budget and the (8, 128) layout at the real shape."""
    rng = np.random.default_rng(8)
    seq, kpool = 2048, 4
    pool_mask, tails = random_selection(rng, 1, seq, kpool, select=512)
    q, lat, g = _arrays(rng, 1, 16, seq, 512, jnp.bfloat16)
    kw = dict(kpool=kpool, block_q=256, block_kv=512)
    got = _value_and_grads(PALLAS, q, lat, pool_mask, tails, g, backward=backward, **kw)
    want = _value_and_grads(
        XLA_REF, q.astype(jnp.float32), lat.astype(jnp.float32), pool_mask, tails, g.astype(jnp.float32), **kw
    )
    for a, b, name in zip(got, want, NAMES, strict=True):
        _assert_close(a, b, 3e-2, name)
