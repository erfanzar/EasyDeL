from __future__ import annotations

import importlib

import jax.numpy as jnp

from ejkernel.ops import BwdParams, FwdParams

tpu_kernel = importlib.import_module("ejkernel.kernels._pallas.tpu.blocksparse_attention._kernel")


def test_blocksparse_attention_tpu_bwd_q_blocks_use_q_blocksize(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_make_splash_mha(*, mask, block_sizes, logits_soft_cap, head_shards, q_seq_shards):
        del mask, logits_soft_cap, head_shards, q_seq_shards
        captured["block_sizes"] = block_sizes

        def _fake_kernel(*, q, k, v, segment_ids, sinks):
            del k, segment_ids, sinks
            return jnp.zeros((q.shape[0], q.shape[1], v.shape[-1]), dtype=v.dtype)

        return _fake_kernel

    monkeypatch.setattr(tpu_kernel, "make_splash_mha", _fake_make_splash_mha)

    q = jnp.ones((1, 2, 24, 8), dtype=jnp.float32)
    k = jnp.ones((1, 2, 40, 8), dtype=jnp.float32)
    v = jnp.ones((1, 2, 40, 8), dtype=jnp.float32)

    fwd_params = FwdParams(q_blocksize=8, kv_blocksize=16, num_stages=2, num_warps=4)
    bwd_params = BwdParams(q_blocksize=16, kv_blocksize=32, num_stages=2, num_warps=4)

    out = tpu_kernel.blocksparse_attention.__wrapped__(
        query=q,
        key=k,
        value=v,
        causal=False,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
    )

    block_sizes = captured["block_sizes"]
    assert out.shape == (1, 2, 24, 8)
    assert block_sizes.block_q_dkv == min(bwd_params.q_blocksize, q.shape[2])
    assert block_sizes.block_q_dq == min(bwd_params.q_blocksize, q.shape[2])
    assert block_sizes.block_kv_dkv == min(bwd_params.kv_blocksize, v.shape[2])
    assert block_sizes.block_kv_dq == min(bwd_params.kv_blocksize, v.shape[2])
