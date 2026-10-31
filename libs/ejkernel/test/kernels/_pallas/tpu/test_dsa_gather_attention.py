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

"""Per-query DMA-gather DSA attention (Pallas TPU) against an independent float64 reference and the XLA fallback.

Covers: value and ``(dq, dlatent)`` gradients, query shards with absolute ``positions`` (sequence-parallel callers),
invalid queries (negative positions), ``-1`` and non-causal selected ids (dropped), queries with no open key, padding
of the query block and of the pool grid, ``include_tail=False``, fp32 and bf16 latents, the remat residual names, and
a sequence-split ``shard_map`` on a multi-chip mesh.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.errors import EjkernelRuntimeError
from ejkernel.kernels import Backend, Platform, kernel_registry
from ejkernel.modules import DsaGatherAttentionConfig
from ejkernel.modules import dsa_gather_attention as gather_op
from jax.sharding import Mesh, PartitionSpec

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernel")

PALLAS = kernel_registry.get("dsa_gather_attention", platform=Platform.PALLAS, backend=Backend.TPU)
XLA = kernel_registry.get("dsa_gather_attention", platform=Platform.XLA, backend=Backend.ANY)


def _selection(rng, batch, n, seq, kpool, k_sel, q_offset):
    """Distinct pool ids in random order with ``-1`` holes and some non-causal ids (which must be dropped)."""
    num_pools = seq // kpool
    idx = np.stack(
        [np.stack([rng.choice(num_pools, size=k_sel, replace=False) for _ in range(n)]) for _ in range(batch)]
    )
    pos = q_offset + np.arange(n)
    # Bias most picks to be causal so the attention is not dominated by the tail.
    for b in range(batch):
        for i in range(n):
            vis = (pos[i] + 1) // kpool
            if vis >= k_sel:
                causal = rng.choice(vis, size=k_sel, replace=False)
                keep = rng.random(k_sel) < 0.8
                idx[b, i] = np.where(keep, causal, idx[b, i])
                _, first = np.unique(idx[b, i], return_index=True)
                dup = np.ones(k_sel, bool)
                dup[first] = False
                idx[b, i][dup] = -1
    holes = rng.random(idx.shape) < 0.15
    idx = np.where(holes, -1, idx).astype(np.int32)
    positions = np.broadcast_to(pos[None], (batch, n)).astype(np.int32).copy()
    return idx, positions


def _open_mask(indices, positions, kpool, seq, include_tail=True):
    batch, n, _ = indices.shape
    m = np.zeros((batch, n, seq), bool)
    for b in range(batch):
        for i in range(n):
            pos = int(positions[b, i])
            if pos < 0:
                continue
            for p in indices[b, i]:
                if p >= 0 and p * kpool + kpool - 1 <= pos:
                    m[b, i, p * kpool : (p + 1) * kpool] = True
            if include_tail and pos % kpool != kpool - 1:
                m[b, i, (pos // kpool) * kpool : pos + 1] = True
    return m


def _np_reference(q, lat, open_, scale, g):
    """float64 attention over ``open_`` [B, N, S] and its analytic ``(dq, dlatent)``; empty rows give 0."""
    q, lat, g = (np.asarray(x, np.float64) for x in (q, lat, g))
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


def _value_and_grads(fn, q, lat, idx, positions, g, **kw):
    def loss(q_, l_):
        out = fn(q_, l_, idx, positions, **kw)
        return jnp.sum(out.astype(jnp.float32) * g.astype(jnp.float32)), out

    (_, out), (dq, dlat) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))(q, lat)
    return out, dq, dlat


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _arrays(rng, batch, n, heads, seq, dim, dtype):
    q = jnp.asarray(rng.standard_normal((batch, n, heads, dim)), dtype)
    lat = jnp.asarray(rng.standard_normal((batch, seq, dim)), dtype)
    g = jnp.asarray(rng.standard_normal((batch, n, heads, dim)), dtype)
    return q, lat, g


@pytest.mark.parametrize(
    ("batch", "n", "seq", "q_offset", "k_sel", "include_tail"),
    [
        (2, 64, 256, 0, 16, True),  # queries are the sequence prefix
        (1, 48, 512, 400, 24, True),  # a query shard at an offset; 48 % block_q(16) == 0
        (1, 40, 256, 216, 12, False),  # 40 % 16 != 0 (query padding), no tail
    ],
)
def test_matches_float64_reference_fp32(batch, n, seq, q_offset, k_sel, include_tail):
    rng = np.random.default_rng(seq + n)
    kpool, heads, dim = 4, 4, 128
    idx, positions = _selection(rng, batch, n, seq, kpool, k_sel, q_offset)
    positions[0, 5] = -1  # an invalid query
    if q_offset == 0:
        idx[0, 3] = -1  # pos 3: own pool complete, nothing selected -> no open key -> 0
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.float32)
    scale = dim**-0.5
    out, dq, dlat = _value_and_grads(
        PALLAS, q, lat, jnp.asarray(idx), jnp.asarray(positions), g, kpool=kpool, include_tail=include_tail
    )
    want = _np_reference(q, lat, _open_mask(idx, positions, kpool, seq, include_tail), scale, g)
    for got, ref, name in zip((out, dq, dlat), want, ("out", "dq", "dlatent"), strict=True):
        assert got.shape == ref.shape, name
        assert _rel(got, ref) < 2e-4, (name, _rel(got, ref))
    np.testing.assert_array_equal(np.asarray(out[0, 5]), 0.0)
    if q_offset == 0:
        np.testing.assert_array_equal(np.asarray(out[0, 3]), 0.0)


def test_bf16_matches_xla_fallback_at_glm_shape():
    """bf16 values and gradients match the XLA fallback at H=64, D=512, kpool=4 and 512 selected pools per query."""
    rng = np.random.default_rng(7)
    batch, n, seq, heads, dim, kpool, k_sel = 1, 256, 8192, 64, 512, 4, 512
    q_offset = seq - n
    idx, positions = _selection(rng, batch, n, seq, kpool, k_sel, q_offset)
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.bfloat16)
    args = (q, lat, jnp.asarray(idx), jnp.asarray(positions), g)
    got = _value_and_grads(PALLAS, *args, kpool=kpool)
    want = _value_and_grads(XLA, *args, kpool=kpool)
    for a, b, name in zip(got, want, ("out", "dq", "dlatent"), strict=True):
        assert a.dtype == jnp.bfloat16, name
        assert _rel(a, b) < 1e-2, (name, _rel(a, b))


def test_unaligned_sequence_pads_the_pool_grid():
    rng = np.random.default_rng(3)
    batch, n, seq, heads, dim, kpool = 1, 32, 250, 2, 128, 4  # 250 keys: the last pool is incomplete
    idx, positions = _selection(rng, batch, n, 248, kpool, 8, seq - n)
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.float32)
    got = _value_and_grads(PALLAS, q, lat, jnp.asarray(idx), jnp.asarray(positions), g, kpool=kpool)
    want = _np_reference(q, lat, _open_mask(idx, positions, kpool, seq), dim**-0.5, g)
    for a, b, name in zip(got, want, ("out", "dq", "dlatent"), strict=True):
        assert _rel(a, b) < 2e-4, (name, _rel(a, b))


@pytest.mark.parametrize(
    "cfg",
    [
        dict(block_q=8, pools_per_step=32, num_buffers=2),
        dict(pools_per_step=64, num_buffers=6),  # padded multi-group path: 65 slots -> 2 x 64
        dict(pools_per_step=13, num_buffers=3),  # multi-group, no padding: 65 slots -> 5 x 13
        dict(num_cores=1),
    ],
)
def test_tiling_options_do_not_change_the_result(cfg):
    """Explicit tilings match the default automatic group plan (one 65-pool group per query at this shape)."""
    rng = np.random.default_rng(11)
    batch, n, seq, heads, dim, kpool = 1, 64, 1024, 8, 128, 4
    idx, positions = _selection(rng, batch, n, seq, kpool, 64, seq - n)
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.float32)
    args = (q, lat, jnp.asarray(idx), jnp.asarray(positions), g)
    base = _value_and_grads(PALLAS, *args, kpool=kpool)
    other = _value_and_grads(PALLAS, *args, kpool=kpool, **cfg)
    np.testing.assert_allclose(np.asarray(other[0]), np.asarray(base[0]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(other[1]), np.asarray(base[1]), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(other[2]), np.asarray(base[2]), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("budget", [None, 8 * 2**20, 2 * 2**20])
def test_group_plan_is_even_and_fits(budget):
    """The automatic plan covers every slot with evenly sized groups (no padded group) inside the VMEM budget."""
    from ejkernel.kernels._pallas.tpu.dsa_gather_attention import dsa_gather_group_plan

    slots = 513  # index_topk 2048 / kpool 4 = 512 selected pools, plus the own-pool tail
    g, ng, nbuf = dsa_gather_group_plan(
        slots, kpool=4, head_dim=512, num_heads=64, itemsize=2, block_q=16, vmem_budget=budget
    )
    assert g * ng >= slots and g * (ng - 1) < slots, (g, ng)
    assert ng == -(-slots // g)
    assert nbuf == (2 if ng == 1 else 4)
    if budget is None:
        assert (g, ng) == (513, 1), "the GLM shape fits one group per query in the default budget"
    else:
        big = dsa_gather_group_plan(slots, kpool=4, head_dim=512, num_heads=64, itemsize=2, block_q=16)
        assert ng > big[1], "a smaller budget must split into more groups"


def test_named_residuals_skip_the_forward_recompute():
    """Saving the tagged residuals under remat: bitwise gradients and one forward kernel launch fewer."""
    from ejkernel.kernels._pallas.tpu.dsa_gather_attention import RESIDUAL_NAME, TABLES_NAME

    rng = np.random.default_rng(12)
    batch, n, seq, heads, dim, kpool = 1, 64, 512, 4, 128, 4
    idx, positions = _selection(rng, batch, n, seq, kpool, 16, seq - n)
    idx, positions = jnp.asarray(idx), jnp.asarray(positions)
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.float32)

    def body(q_, l_):
        return jnp.tanh(PALLAS(q_, l_, idx, positions, kpool=kpool))

    def grad_fn(policy):
        f = jax.checkpoint(body, policy=policy)
        return jax.jit(jax.value_and_grad(lambda q_, l_: jnp.sum(f(q_, l_) * g), argnums=(0, 1)))

    recompute = grad_fn(jax.checkpoint_policies.nothing_saveable)
    saved = grad_fn(jax.checkpoint_policies.save_only_these_names(RESIDUAL_NAME, TABLES_NAME))
    (v_s, g_s), (v_r, g_r) = saved(q, lat), recompute(q, lat)
    np.testing.assert_array_equal(np.asarray(v_s), np.asarray(v_r))
    for a, b in zip(g_s, g_r, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    target = 'custom_call_target="tpu_custom_call"'
    n_recompute = recompute.lower(q, lat).compile().as_text().count(target)
    n_saved = saved.lower(q, lat).compile().as_text().count(target)
    assert n_saved == n_recompute - 1, (n_saved, n_recompute)


def test_sequence_split_shard_map_matches_one_device():
    """Queries split over the sequence on a multi-chip mesh (latent replicated) equal the single-device call."""
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("needs 2+ TPU chips")
    n_dev = 4 if len(devices) >= 4 else 2
    mesh = Mesh(np.asarray(devices[:n_dev]), ("sp",))
    rng = np.random.default_rng(5)
    batch, seq, heads, dim, kpool = 1, 1024, 8, 128, 4
    idx, positions = _selection(rng, batch, seq, seq, kpool, 32, 0)
    idx, positions = jnp.asarray(idx), jnp.asarray(positions)
    q, lat, g = _arrays(rng, batch, seq, heads, dim=dim, seq=seq, dtype=jnp.float32)
    spec_q = PartitionSpec(None, "sp", None, None)
    specs = (spec_q, PartitionSpec(None, None, None), PartitionSpec(None, "sp", None), PartitionSpec(None, "sp"))
    cfg = DsaGatherAttentionConfig(platform="pallas", backend="tpu")

    def sharded(q_, l_, i_, p_):
        return gather_op(q_, l_, i_, p_, kpool=kpool, cfg=cfg, mesh=mesh, in_specs=specs, out_specs=spec_q)

    def single(q_, l_, i_, p_):
        return gather_op(q_, l_, i_, p_, kpool=kpool, cfg=cfg)

    got = _value_and_grads(sharded, q, lat, idx, positions, g)
    want = _value_and_grads(single, q, lat, idx, positions, g)
    for a, b, name in zip(got, want, ("out", "dq", "dlatent"), strict=True):
        assert _rel(a, b) < 1e-5, (name, _rel(a, b))


def _bwd_hlo(q, lat, idx, positions, g, **kw):
    def loss(q_, l_):
        return jnp.sum(PALLAS(q_, l_, idx, positions, kpool=4, **kw).astype(jnp.float32) * g)

    return jax.jit(jax.grad(loss, argnums=(0, 1))).lower(q, lat).compile().as_text()


def test_fused_backward_matches_the_three_dma_backward_and_is_the_default():
    """One pool group per query: the fused-row backward is chosen by default and equals the three-DMA backward."""
    rng = np.random.default_rng(13)
    batch, n, seq, heads, dim, kpool = 2, 48, 512, 8, 128, 4
    idx, positions = _selection(rng, batch, n, seq, kpool, 24, seq - n)  # 25 slots -> padded to 26 when fused
    positions[1, 7] = -1
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.float32)
    idx, positions = jnp.asarray(idx), jnp.asarray(positions)
    args = (q, lat, idx, positions, g)
    fused = _value_and_grads(PALLAS, *args, kpool=kpool, fused_backward=True)
    dense = _value_and_grads(PALLAS, *args, kpool=kpool, fused_backward=False)
    auto = _value_and_grads(PALLAS, *args, kpool=kpool)
    want = _np_reference(q, lat, _open_mask(np.asarray(idx), np.asarray(positions), kpool, seq), dim**-0.5, g)
    for got in (fused, dense):
        for a, b, name in zip(got, want, ("out", "dq", "dlatent"), strict=True):
            assert _rel(a, b) < 2e-4, (name, _rel(a, b))
    for a, b, name in zip(fused, dense, ("out", "dq", "dlatent"), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6, err_msg=name)
    for a, b in zip(auto, fused, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert "dsa_gather_bwd_fused" in _bwd_hlo(*args)
    assert "dsa_gather_bwd_fused" not in _bwd_hlo(*args, fused_backward=False)


def test_fused_backward_bf16_matches_xla_at_glm_shape():
    rng = np.random.default_rng(14)
    batch, n, seq, heads, dim, kpool, k_sel = 1, 128, 8192, 64, 512, 4, 512
    idx, positions = _selection(rng, batch, n, seq, kpool, k_sel, seq - n)
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.bfloat16)
    args = (q, lat, jnp.asarray(idx), jnp.asarray(positions), g)
    got = _value_and_grads(PALLAS, *args, kpool=kpool, fused_backward=True)
    want = _value_and_grads(XLA, *args, kpool=kpool)
    for a, b, name in zip(got, want, ("out", "dq", "dlatent"), strict=True):
        assert _rel(a, b) < 1e-2, (name, _rel(a, b))


def test_fused_backward_true_is_not_silently_ignored():
    """Several pool groups per query cannot use the fused backward: True raises, None falls back to three DMAs."""
    rng = np.random.default_rng(15)
    batch, n, seq, heads, dim, kpool = 1, 32, 1024, 4, 128, 4
    idx, positions = _selection(rng, batch, n, seq, kpool, 64, seq - n)
    q, lat, g = _arrays(rng, batch, n, heads, seq, dim, jnp.float32)
    args = (q, lat, jnp.asarray(idx), jnp.asarray(positions), g)
    # The registered entry re-raises kernel errors as EjkernelRuntimeError with the same message.
    with pytest.raises((ValueError, EjkernelRuntimeError), match="fused_backward=True is not supported"):
        _value_and_grads(PALLAS, *args, kpool=kpool, pools_per_step=13, fused_backward=True)
    assert "dsa_gather_bwd_fused" not in _bwd_hlo(*args, pools_per_step=13)
