from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.modules.operations import BlockSparseAttention, blocksparse_attention
from ejkernel.modules.operations.configs import BlockSparseAttentionConfig
from ejkernel.ops import BwdParams, FwdParams
from ejkernel.types import MaskInfo

from ._utils import assert_allclose, dense_attention_reference, rand_qkv


def test_blocksparse_attention_matches_dense_reference_xla():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(key, batch=1, q_len=8, kv_len=8, q_heads=4, kv_heads=2, head_dim=16, dtype=jnp.bfloat16)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    softmax_aux = jax.random.normal(jax.random.PRNGKey(1), (4,), dtype=jnp.float32).astype(jnp.bfloat16)
    sliding_window = (3, 1)
    logits_soft_cap = 5.0
    scale = 16**-0.5

    out = blocksparse_attention(
        q_t,
        k_t,
        v_t,
        softmax_aux,
        None,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        platform="xla",
    )
    ref_out, _ = dense_attention_reference(
        q,
        k,
        v,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        softmax_aux=softmax_aux,
    )

    assert out.shape == (1, 4, 8, 16)
    assert_allclose(out.transpose(0, 2, 1, 3), ref_out, atol=0.15)


def test_blocksparse_attention_purify_zeros_padded_queries():
    key = jax.random.PRNGKey(2)
    q, k, v = rand_qkv(
        key,
        batch=1,
        q_len=8,
        kv_len=8,
        q_heads=4,
        kv_heads=4,
        head_dim=16,
        dtype=jnp.bfloat16,
    )
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    seg = jnp.array([[0, 0, 0, 0, 1, 1, -1, -1]], dtype=jnp.int32)
    mask_info = MaskInfo.from_segments(q_segment_ids=seg)

    out = blocksparse_attention(
        q_t,
        k_t,
        v_t,
        None,
        None,
        mask_info=mask_info,
        purify=True,
        platform="xla",
        causal=False,
    )
    assert jnp.all(out[:, :, 6:, :] == 0)


def test_blocksparse_attention_bias_raises_on_xla():
    key = jax.random.PRNGKey(3)
    q, k, v = rand_qkv(key, batch=1, q_len=8, kv_len=8, q_heads=4, kv_heads=4, head_dim=16, dtype=jnp.bfloat16)
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    bias = jnp.zeros((1, 4, 8, 8), dtype=jnp.bfloat16)
    with pytest.raises(EjkernelRuntimeError):
        blocksparse_attention(q_t, k_t, v_t, None, bias, platform="xla")


def test_blocksparse_attention_run_forwards_cfg_fwd_and_bwd_params(monkeypatch):
    kernel = BlockSparseAttention()
    captured: dict[str, object] = {}

    def _fake_impl(**kwargs):
        captured.update(kwargs)
        return jnp.zeros_like(kwargs["query"])

    monkeypatch.setattr(kernel, "get_impl", lambda _cfg: _fake_impl)

    q = jnp.ones((1, 2, 8, 16), dtype=jnp.float32)
    k = jnp.ones((1, 2, 8, 16), dtype=jnp.float32)
    v = jnp.ones((1, 2, 8, 16), dtype=jnp.float32)

    fwd_params = FwdParams(q_blocksize=256, kv_blocksize=128, num_stages=2, num_warps=4)
    bwd_params = BwdParams(q_blocksize=512, kv_blocksize=1024, num_stages=2, num_warps=4)
    cfg = BlockSparseAttentionConfig(fwd_params=fwd_params, bwd_params=bwd_params, platform="pallas", backend="tpu")

    _ = kernel.run(query=q, key=k, value=v, causal=False, cfg=cfg)

    assert captured["fwd_params"] is fwd_params
    assert captured["bwd_params"] is bwd_params
