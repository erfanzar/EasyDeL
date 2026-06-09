from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import ragged_decode_attention

from ._utils import assert_allclose, dense_attention_reference, device_platform, rand_qkv


def test_ragged_decode_attention_matches_dense_reference_and_supports_query_rank_4():
    key = jax.random.PRNGKey(0)
    batch, seq_len, q_heads, kv_heads, head_dim = 3, 16, 4, 2, 32

    q3, k, v = rand_qkv(key, batch=batch, q_len=1, kv_len=seq_len, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim)
    q3 = q3[:, 0, :, :]

    sequence_start = jnp.array([0, 2, 0], dtype=jnp.int32)
    sequence_end = jnp.array([8, 12, 16], dtype=jnp.int32)
    sliding_window = (6, 0)
    logits_soft_cap = 10.0
    softmax_aux = jnp.array([0.1, -0.2], dtype=jnp.bfloat16)

    out3 = ragged_decode_attention(
        q3,
        k,
        v,
        sequence_start,
        sequence_end,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=head_dim**-0.5,
        platform="xla",
    )
    out4 = ragged_decode_attention(
        q3[:, None, :, :],
        k,
        v,
        sequence_start,
        sequence_end,
        softmax_aux,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=head_dim**-0.5,
        platform="xla",
    )

    assert out3.shape == (batch, q_heads, head_dim)
    assert out4.shape == (batch, 1, q_heads, head_dim)
    assert_allclose(out4[:, 0], out3, atol=0.0)

    q_pos = sequence_end - 1
    kv_pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    left, right = sliding_window
    mask = (kv_pos >= sequence_start[:, None]) & (kv_pos < sequence_end[:, None])
    mask = mask & (kv_pos >= (q_pos[:, None] - left)) & (kv_pos <= (q_pos[:, None] + right))
    mask = mask[:, None, None, :]

    ref_out, _ = dense_attention_reference(
        q3[:, None, :, :],
        k,
        v,
        attention_mask=mask,
        softmax_scale=head_dim**-0.5,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
    assert_allclose(out3, ref_out[:, 0], atol=0.25)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_ragged_decode_attention_pallas_matches_xla_on_tpu():
    key = jax.random.PRNGKey(1)
    batch, seq_len, q_heads, kv_heads, head_dim = 2, 32, 4, 2, 32
    q3, k, v = rand_qkv(key, batch=batch, q_len=1, kv_len=seq_len, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim)
    q3 = q3[:, 0, :, :]
    sequence_start = jnp.zeros((batch,), dtype=jnp.int32)
    sequence_end = jnp.array([16, 32], dtype=jnp.int32)

    out_xla = ragged_decode_attention(
        q3,
        k,
        v,
        sequence_start,
        sequence_end,
        None,
        platform="xla",
    )
    out_pallas = ragged_decode_attention(
        q3,
        k,
        v,
        sequence_start,
        sequence_end,
        None,
        platform="pallas",
    )

    assert_allclose(out_pallas, out_xla, atol=0.35)
