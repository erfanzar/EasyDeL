from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import flash_attention
from ejkernel.types import MaskInfo

from ._utils import assert_allclose, dense_attention_reference, device_platform, has_triton, rand_qkv


def test_flash_attention_matches_dense_reference_with_bias():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=10, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 8, 10), dtype=jnp.float32).astype(jnp.bfloat16)

    out = flash_attention(q, k, v, bias, softmax_scale=32**-0.5, platform="xla")
    ref_out, _ = dense_attention_reference(q, k, v, bias=bias, softmax_scale=32**-0.5)
    assert out.shape == (2, 8, 4, 32)
    assert_allclose(out, ref_out, atol=0.15)


def test_flash_attention_segments_sliding_window_logits_soft_cap_and_softmax_aux_match_reference_xla():
    key = jax.random.PRNGKey(2)
    q, k, v = rand_qkv(key, batch=2, q_len=12, kv_len=12, q_heads=4, kv_heads=2, head_dim=16, dtype=jnp.bfloat16)

    seg = jnp.array(
        [
            [0] * 4 + [1] * 4 + [-1] * 4,
            [0] * 6 + [1] * 3 + [-1] * 3,
        ],
        dtype=jnp.int32,
    )
    mask_info = MaskInfo.from_segments(q_segment_ids=seg)

    softmax_aux = jax.random.normal(jax.random.PRNGKey(3), (2,), dtype=jnp.float32).astype(jnp.bfloat16)
    sliding_window = (5, 1)
    logits_soft_cap = 6.0
    scale = 16**-0.5

    out = flash_attention(
        q,
        k,
        v,
        None,
        None,
        None,
        softmax_aux,
        mask_info=mask_info,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        platform="xla",
    )

    seg_mask = (seg[:, None, :, None] == seg[:, None, None, :]) & (seg[:, None, :, None] >= 0)
    ref_out, _ = dense_attention_reference(
        q,
        k,
        v,
        attention_mask=seg_mask,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        softmax_aux=softmax_aux,
    )

    assert_allclose(out, ref_out, atol=0.15)


def test_flash_attention_dropout_seed_changes_output():
    key = jax.random.PRNGKey(4)
    q, k, v = rand_qkv(key, batch=1, q_len=16, kv_len=16, q_heads=4, kv_heads=4, head_dim=32, dtype=jnp.bfloat16)

    out0 = flash_attention(q, k, v, dropout_prob=0.25, dropout_seed=0, platform="xla")
    out1 = flash_attention(q, k, v, dropout_prob=0.25, dropout_seed=1, platform="xla")

    assert out0.shape == q.shape
    assert out1.shape == q.shape
    assert not jnp.allclose(out0, out1)


@pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUDA validation")
def test_flash_attention_cuda_softcap_matches_xla():
    key = jax.random.PRNGKey(8)
    q, k, v = rand_qkv(key, batch=1, q_len=16, kv_len=16, q_heads=2, kv_heads=2, head_dim=32, dtype=jnp.float16)
    out_cuda = flash_attention(q, k, v, causal=False, logits_soft_cap=6.0, platform="cuda")
    out_xla = flash_attention(q, k, v, causal=False, logits_soft_cap=6.0, platform="xla")
    assert_allclose(out_cuda, out_xla, atol=0.25, rtol=0.15)


@pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUDA validation")
def test_flash_attention_cuda_backward_matches_xla():
    key = jax.random.PRNGKey(9)
    q, k, v = rand_qkv(key, batch=1, q_len=8, kv_len=8, q_heads=2, kv_heads=2, head_dim=32, dtype=jnp.float16)

    def loss(q_in, k_in, v_in, platform):
        out = flash_attention(q_in, k_in, v_in, causal=True, platform=platform)
        return jnp.sum(out)

    dq_cuda = jax.grad(loss, argnums=0)(q, k, v, "cuda")
    dq_xla = jax.grad(loss, argnums=0)(q, k, v, "xla")
    assert_allclose(dq_cuda, dq_xla, atol=0.25, rtol=0.2)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_flash_attention_pallas_matches_xla_on_tpu_with_sliding_window_and_soft_cap():
    key = jax.random.PRNGKey(5)
    q, k, v = rand_qkv(key, batch=2, q_len=128, kv_len=128, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)

    out_xla = flash_attention(q, k, v, sliding_window=(64, 0), logits_soft_cap=10.0, causal=True, platform="xla")
    out_pallas = flash_attention(q, k, v, sliding_window=(64, 0), logits_soft_cap=10.0, causal=True, platform="pallas")

    assert_allclose(out_pallas, out_xla, atol=0.2)


@pytest.mark.skipif(device_platform() != "gpu", reason="GPU-only CUDA vs Triton comparison")
@pytest.mark.skipif(not has_triton(), reason="Triton not installed")
@pytest.mark.parametrize(
    "head_dim, sliding_window",
    [
        (64, None),
        (64, (32, 32)),
        (128, None),
        (128, (32, 32)),
    ],
)
def test_flash_attention_cuda_matches_triton(head_dim: int, sliding_window: tuple[int, int] | None):
    key = jax.random.PRNGKey(12 + head_dim)
    q, k, v = rand_qkv(
        key,
        batch=2,
        q_len=64,
        kv_len=64,
        q_heads=4,
        kv_heads=4,
        head_dim=head_dim,
        dtype=jnp.float16,
    )

    out_cuda = flash_attention(q, k, v, causal=True, sliding_window=sliding_window, platform="cuda")
    out_triton = flash_attention(q, k, v, causal=True, sliding_window=sliding_window, platform="triton")

    assert_allclose(out_cuda, out_triton, atol=0.2, rtol=0.1)
