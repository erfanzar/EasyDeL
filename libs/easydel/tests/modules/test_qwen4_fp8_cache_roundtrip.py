"""Opt-in TPU4 regression for Qwen4's concrete FP8 v3 paged-QSA path.

Run in a complete EasyDeL checkout/environment (not the review-file overlay):
  JAX_PLATFORMS=tpu ENABLE_DISTRIBUTED_INIT=0 python -m pytest -q -s \
    /Users/erfan/.xerxes-scratch/test_qwen4_fp8_cache_roundtrip.py::test_fp8_paged_cache_tp4_roundtrip

Collection imports neither JAX nor EasyDeL and does not initialize hardware.
This tests the production allocator, Qwen v3 writer and gathered attention,
not the Pallas ragged kernel, indexer selection, or end-to-end model logits.
"""

import os
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(
    os.environ.get("JAX_PLATFORMS") != "tpu" or os.environ.get("ENABLE_DISTRIBUTED_INIT") != "0",
    reason="explicit opt-in required: JAX_PLATFORMS=tpu ENABLE_DISTRIBUTED_INIT=0",
)
def test_fp8_paged_cache_tp4_roundtrip():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from easydel.caching import RaggedPagesCacheConfig, RaggedPagesMetadata
    from easydel.infra.sharding import coerce_runtime_sharding_resolver, mesh_axis_size
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpAttention,
        Qwen4ExpPagedQSAView,
    )
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = jax.local_devices(backend="tpu")
    assert len(devices) == 4, "this hardware regression requires exactly four local TPU devices"
    assert all(d.platform == "tpu" for d in devices)
    mesh = Mesh(np.asarray(devices), ("tp",))
    replicated = NamedSharding(mesh, P())
    query_sharding = NamedSharding(mesh, P(None, None, "tp", None))
    resolver = coerce_runtime_sharding_resolver(None, mesh=mesh)
    assert mesh_axis_size(mesh, resolver.paxis.kv_head_axis) == 4

    # Production layout resolution must undo logical-head padding when FP8
    # cannot shard complete packed K/V pairs across TP4. Capacity is hard-capped.
    cfg = RaggedPagesCacheConfig.create(
        mesh=mesh,
        runtime_sharding_resolver=resolver,
        kvdtype=jnp.float8_e4m3fn,
        num_hidden_layers=1,
        num_kv_heads=2,
        kv_head_dim_size=256,
        max_model_length=8,
        page_size=4,
        version="v3",
        max_cache_tokens=12,
    )
    assert cfg.num_pages == 3
    assert cfg.logical_num_kv_heads == cfg.num_kv_heads == 2
    assert cfg.kv_head_shards == 1
    assert cfg.kvdtype == jnp.dtype(jnp.float8_e4m3fn)
    # Same model-specific sidecar annotation used by Qwen paged-cache setup.
    object.__setattr__(cfg, "qwen4_indexer_head_dim", 8)
    view = Qwen4ExpPagedQSAView.init(cfg, mesh=mesh, runtime_sharding_resolver=resolver)
    assert isinstance(view, Qwen4ExpPagedQSAView)
    assert view.kv_pages.dtype == jnp.float8_e4m3fn
    assert view.kv_pages.sharding.is_fully_replicated
    assert len(view.kv_pages.addressable_shards) == 4
    assert view.key_pages.shape == (3, 4, 2, 256)

    def metadata(length, query_tokens):
        return RaggedPagesMetadata(
            pages_tables=jax.device_put(jnp.array([[2, 0]], jnp.int32), replicated),
            context_lens=jax.device_put(jnp.array([length], jnp.int32), replicated),
            query_start_loc=jax.device_put(jnp.array([0, query_tokens], jnp.int32), replicated),
            num_seqs=jax.device_put(jnp.array([1], jnp.int32), replicated),
            request_distribution=jax.device_put(jnp.array([0, 1, 1], jnp.int32), replicated),
            version="v3",
            page_size=4,
        )

    rng = np.random.default_rng(15)
    key = jax.device_put(jnp.asarray(rng.normal(0, 0.4, (1, 5, 2, 256)), jnp.bfloat16), replicated)
    # Distinct logical-head means catch accidental head duplication/permutation.
    values = rng.normal(0, 0.3, (1, 5, 2, 256)).astype(np.float32)
    values[:, :, 0] -= 0.75
    values[:, :, 1] += 0.75
    values[:, 4] += 0.5  # decode token must actually be written before attention
    value = jax.device_put(jnp.asarray(values, jnp.bfloat16), replicated)
    query = jax.device_put(jnp.asarray(rng.normal(0, 0.4, (1, 1, 8, 256)), jnp.bfloat16), query_sharding)
    selected = jax.device_put(jnp.array([[[0, 1, 2, 3, 4, -1]]], jnp.int32), replicated)
    attention = SimpleNamespace(head_dim=256)  # methods only read this scalar; view is real
    prefill_meta, decode_meta = metadata(4, 4), metadata(5, 1)

    @jax.jit
    def prefill(v, k, val):
        return Qwen4ExpAttention._write_v3_paged_kv(attention, k, val, v, prefill_meta)

    # Keep KV replicas but force queries AND results to have physical TP4 head
    # sharding. A merely four-device mesh around single-device math is not enough.
    @jax.jit(out_shardings=(None, query_sharding))
    def decode(v, q, k, val, sel):
        updated = Qwen4ExpAttention._write_v3_paged_kv(attention, k, val, v, decode_meta)
        out = Qwen4ExpAttention._paged_gather_attention(attention, q, updated, decode_meta, sel)
        return updated, out

    with mesh:
        view = prefill(view, key[:, :4], value[:, :4])
        view, got = decode(view, query, key[:, 4:], value[:, 4:], selected)
        got.block_until_ready()

    assert got.shape == (1, 1, 8, 256)
    assert got.sharding.is_equivalent_to(query_sharding, ndim=4)
    assert len(got.addressable_shards) == 4
    assert {shard.data.shape for shard in got.addressable_shards} == {(1, 1, 2, 256)}
    assert view.kv_pages.dtype == jnp.float8_e4m3fn
    assert view.kv_pages.sharding.is_fully_replicated
    physical, offsets = np.array([2, 2, 2, 2, 0]), np.array([0, 1, 2, 3, 0])
    decoded_key = np.asarray(key.astype(jnp.float8_e4m3fn).astype(jnp.float32))[0]
    decoded_value = np.asarray(value.astype(jnp.float8_e4m3fn).astype(jnp.float32))[0]
    np.testing.assert_array_equal(np.asarray(view.key_pages.astype(jnp.float32))[physical, offsets], decoded_key)
    np.testing.assert_array_equal(np.asarray(view.value_pages.astype(jnp.float32))[physical, offsets], decoded_value)
    # The unused physical page is a guard against sequential/logical-page writes.
    np.testing.assert_array_equal(np.asarray(view.kv_pages.astype(jnp.float32))[1], 0)

    def reference(k, v):
        q = np.asarray(query.astype(jnp.float32))[0, 0].reshape(2, 4, 256)
        scores = np.einsum("hgd,shd->hgs", q, k) / 16.0
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        return np.einsum("hgs,shd->hgd", probs, v).reshape(1, 1, 8, 256)

    actual = np.asarray(got.astype(jnp.float32))
    assert np.isfinite(actual).all()
    # Production gathered FP8 attention also rounds probabilities/output to FP8;
    # these bounds allow that quantization, not just BF16 accumulation error.
    # They are explicit initial regression bounds, not hardware-calibrated claims.
    np.testing.assert_allclose(actual, reference(decoded_key, decoded_value), rtol=0.08, atol=0.06)
    np.testing.assert_allclose(
        actual,
        reference(np.asarray(key.astype(jnp.float32))[0], np.asarray(value.astype(jnp.float32))[0]),
        rtol=0.10,
        atol=0.08,
    )
