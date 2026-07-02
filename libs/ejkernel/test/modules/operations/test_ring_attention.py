from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from ejkernel.modules.operations import RingAttentionConfig, ring_attention

from ._utils import assert_allclose, dense_attention_reference, device_platform, rand_qkv


def test_ring_attention_axis_name_none_matches_dense_reference_xla():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(key, batch=1, q_len=16, kv_len=16, q_heads=4, kv_heads=4, head_dim=32, dtype=jnp.bfloat16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 16, 16), dtype=jnp.float32).astype(jnp.bfloat16)
    softmax_aux = jnp.array([0.0, -0.5], dtype=jnp.bfloat16)  # 2 sinks, shared across all heads

    out = ring_attention(
        q,
        k,
        v,
        softmax_aux,
        bias,
        axis_name=None,
        causal=True,
        sliding_window=(8, 0),
        logits_soft_cap=10.0,
        softmax_scale=32**-0.5,
        platform="xla",
    )
    ref_out, _ = dense_attention_reference(
        q,
        k,
        v,
        bias=bias,
        causal=True,
        sliding_window=(8, 0),
        logits_soft_cap=10.0,
        softmax_scale=32**-0.5,
        softmax_aux=softmax_aux,
    )

    assert out.shape == (1, 16, 4, 32)
    assert_allclose(out, ref_out, atol=0.2)


@pytest.mark.skipif(jax.local_device_count() < 4, reason="requires at least 4 local devices for sequence sharding")
def test_ring_attention_shard_map_accepts_dict_config():
    key = jax.random.PRNGKey(3)
    q, k, v = rand_qkv(key, batch=1, q_len=16, kv_len=16, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.float32)
    cfg = RingAttentionConfig(platform="xla", backend="any")
    seq_spec = PartitionSpec(None, "sp", None, None)

    mesh = Mesh(np.array(jax.devices()[:4]).reshape(4), ("sp",))
    with mesh:
        out = ring_attention(
            q,
            k,
            v,
            axis_name="sp",
            causal=True,
            platform="xla",
            cfg=cfg.to_dict(),
            mesh=mesh,
            in_specs=(seq_spec, seq_spec, seq_spec, None, None),
            out_specs=seq_spec,
        )

    ref_out, _ = dense_attention_reference(q, k, v, causal=True)
    assert_allclose(out, ref_out, atol=2e-4)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_ring_attention_pallas_matches_xla_on_tpu():
    key = jax.random.PRNGKey(2)
    q, k, v = rand_qkv(key, batch=1, q_len=128, kv_len=128, q_heads=4, kv_heads=4, head_dim=32, dtype=jnp.bfloat16)
    out_xla = ring_attention(q, k, v, axis_name=None, causal=True, platform="xla")
    out_pallas = ring_attention(q, k, v, axis_name=None, causal=True, platform="pallas")
    assert_allclose(out_pallas, out_xla, atol=0.25)
