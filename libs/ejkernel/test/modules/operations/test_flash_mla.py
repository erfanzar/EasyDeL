from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla.flash_mla import flash_mla as xla_flash_mla
from ejkernel.modules.operations import flash_mla

from ._utils import assert_allclose


def _mla_dense_reference(
    query,
    key_value,
    w_kc,
    w_vc,
    b_q=None,
    b_k=None,
    *,
    softmax_scale=None,
    causal=False,
    attention_mask=None,
    bias=None,
    logits_soft_cap=None,
    softmax_aux=None,
    sliding_window=None,
):
    """Reference MLA computed entirely in float32.

    Decomposes MLA into standard Q/K/V attention so we can verify flash_mla
    against independently-written code.
    """
    batch, seq_q, q_heads, _q_dim = query.shape
    _, kv_len, _kv_lora_rank = key_value.shape
    _, kv_heads, d_nope = w_kc.shape
    _, _, _v_dim = w_vc.shape

    q_f = query.astype(jnp.float32)
    kv_f = key_value.astype(jnp.float32)
    w_kc_f = w_kc.astype(jnp.float32)
    w_vc_f = w_vc.astype(jnp.float32)

    reps = q_heads // kv_heads

    # Project compressed latent → keys / values
    k_nope = jnp.einsum("btr,rhd->bthd", kv_f, w_kc_f)  # [B, T, kv_h, d_nope]
    v = jnp.einsum("btr,rhd->bthd", kv_f, w_vc_f)  # [B, T, kv_h, v_dim]

    # GQA expansion
    k_nope = jnp.repeat(k_nope, reps, axis=2)
    v = jnp.repeat(v, reps, axis=2)

    # Build full Q and K (possibly with RoPE concatenation)
    if b_k is None:
        q_full = q_f
        k_full = k_nope
        d_scale = float(d_nope)
    elif b_q is None:
        # Fused RoPE: query = [nope | rope]
        rope_dim = b_k.shape[2]
        q_full = q_f
        k_rope = jnp.broadcast_to(
            b_k.astype(jnp.float32)[:, :, None, :],
            (batch, kv_len, q_heads, rope_dim),
        )
        k_full = jnp.concatenate([k_nope, k_rope], axis=-1)
        d_scale = float(d_nope + rope_dim)
    else:
        # Decoupled RoPE: query = nope only; b_q/b_k are per-token shared
        rope_dim = b_k.shape[2]
        b_q_exp = jnp.broadcast_to(
            b_q.astype(jnp.float32)[:, :, None, :],
            (batch, seq_q, q_heads, rope_dim),
        )
        k_rope = jnp.broadcast_to(
            b_k.astype(jnp.float32)[:, :, None, :],
            (batch, kv_len, q_heads, rope_dim),
        )
        q_full = jnp.concatenate([q_f, b_q_exp], axis=-1)
        k_full = jnp.concatenate([k_nope, k_rope], axis=-1)
        d_scale = float(d_nope + rope_dim)

    scale = 1.0 / math.sqrt(d_scale) if softmax_scale is None else float(softmax_scale)

    logits = jnp.einsum("bqhd,bkhd->bhqk", q_full, k_full) * scale

    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    if sliding_window is not None:
        left_w, right_w = (sliding_window, sliding_window) if isinstance(sliding_window, int) else sliding_window
        q_pos = jnp.arange(seq_q)[:, None]
        k_pos = jnp.arange(kv_len)[None, :]
        win = (k_pos >= q_pos - left_w) & (k_pos <= q_pos + right_w)
        logits = jnp.where(win[None, None], logits, jnp.finfo(jnp.float32).min)

    if causal:
        cm = jnp.tril(jnp.ones((seq_q, kv_len), dtype=jnp.bool_))
        logits = jnp.where(cm[None, None], logits, jnp.finfo(jnp.float32).min)

    if bias is not None:
        logits = logits + bias.astype(jnp.float32)
    elif attention_mask is not None:
        m = attention_mask
        if m.dtype != jnp.bool_:
            m = m.astype(jnp.bool_)
        logits = jnp.where(m, logits, jnp.finfo(jnp.float32).min)

    if softmax_aux is not None:
        aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux.ndim == 1 and aux.shape[0] != q_heads:
            aux = jnp.broadcast_to(aux[None, :], (q_heads, aux.shape[0]))
        elif aux.ndim == 1:
            aux = aux[:, None]
        sinks = jnp.broadcast_to(aux[None, :, None, :], (batch, q_heads, seq_q, aux.shape[-1]))
        combined = jnp.concatenate([logits, sinks], axis=-1)
        combined = combined - jnp.max(combined, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined, axis=-1)
        weights = probs[..., :kv_len]
    else:
        weights = jax.nn.softmax(logits, axis=-1)

    out = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return out.astype(query.dtype)


def _rand_mla_inputs(key, *, batch, seq_len, q_heads, kv_heads, d_nope, kv_rank, v_dim):
    """Generate random MLA inputs (no RoPE)."""
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], (batch, seq_len, q_heads, d_nope), dtype=jnp.float32)
    c = jax.random.normal(keys[1], (batch, seq_len, kv_rank), dtype=jnp.float32)
    wkc = jax.random.normal(keys[2], (kv_rank, kv_heads, d_nope), dtype=jnp.float32) * 0.1
    wvc = jax.random.normal(keys[3], (kv_rank, kv_heads, v_dim), dtype=jnp.float32) * 0.1
    return q, c, wkc, wvc


def test_flash_mla_basic_runs_on_xla():
    b, t, q_heads, kv_heads, head_dim = 1, 8, 4, 2, 16
    kv_lora_rank = 8
    rope_dim = 4
    d_nope = head_dim - rope_dim
    key = jax.random.PRNGKey(0)
    key, kq, kkv, kwk, kwv, kbk = jax.random.split(key, 6)

    query = jax.random.normal(kq, (b, t, q_heads, head_dim), dtype=jnp.float32)
    key_value = jax.random.normal(kkv, (b, t, kv_lora_rank), dtype=jnp.float32)
    w_kc = jax.random.normal(kwk, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32)
    w_vc = jax.random.normal(kwv, (kv_lora_rank, kv_heads, head_dim), dtype=jnp.float32)
    b_k = jax.random.normal(kbk, (b, t, rope_dim), dtype=jnp.float32)

    out = flash_mla(query, key_value, w_kc, w_vc, None, b_k, causal=True, platform="xla")
    assert out.shape == (b, t, q_heads, head_dim)
    assert jnp.isfinite(out).all()


def test_flash_mla_no_rope_causal_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(10),
        batch=2,
        seq_len=16,
        q_heads=4,
        kv_heads=2,
        d_nope=32,
        kv_rank=64,
        v_dim=32,
    )
    out = xla_flash_mla(q, c, wkc, wvc, causal=True)
    ref = _mla_dense_reference(q, c, wkc, wvc, causal=True)
    assert out.shape == ref.shape
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_fused_rope_matches_reference():
    key = jax.random.PRNGKey(20)
    batch, seq, q_heads, kv_heads = 2, 16, 4, 2
    d_nope, rope_dim, kv_rank, v_dim = 24, 8, 64, 24

    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (batch, seq, q_heads, d_nope + rope_dim)) * 0.5
    c = jax.random.normal(keys[1], (batch, seq, kv_rank))
    wkc = jax.random.normal(keys[2], (kv_rank, kv_heads, d_nope)) * 0.1
    wvc = jax.random.normal(keys[3], (kv_rank, kv_heads, v_dim)) * 0.1
    b_k = jax.random.normal(keys[4], (batch, seq, rope_dim)) * 0.5

    out = xla_flash_mla(q, c, wkc, wvc, b_k=b_k, causal=True)
    ref = _mla_dense_reference(q, c, wkc, wvc, b_k=b_k, causal=True)
    assert out.shape == ref.shape
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_decoupled_rope_matches_reference():
    key = jax.random.PRNGKey(30)
    batch, seq, q_heads, kv_heads = 2, 16, 4, 2
    d_nope, rope_dim, kv_rank, v_dim = 24, 8, 64, 24

    keys = jax.random.split(key, 6)
    q = jax.random.normal(keys[0], (batch, seq, q_heads, d_nope)) * 0.5
    c = jax.random.normal(keys[1], (batch, seq, kv_rank))
    wkc = jax.random.normal(keys[2], (kv_rank, kv_heads, d_nope)) * 0.1
    wvc = jax.random.normal(keys[3], (kv_rank, kv_heads, v_dim)) * 0.1
    b_q = jax.random.normal(keys[4], (batch, seq, rope_dim)) * 0.5
    b_k = jax.random.normal(keys[5], (batch, seq, rope_dim)) * 0.5

    out = xla_flash_mla(q, c, wkc, wvc, b_q=b_q, b_k=b_k, causal=True)
    ref = _mla_dense_reference(q, c, wkc, wvc, b_q=b_q, b_k=b_k, causal=True)
    assert out.shape == ref.shape
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_non_causal_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(40),
        batch=1,
        seq_len=12,
        q_heads=8,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    out = xla_flash_mla(q, c, wkc, wvc, causal=False)
    ref = _mla_dense_reference(q, c, wkc, wvc, causal=False)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_gqa_multiple_ratios():
    """Test GQA with various q_heads / kv_heads ratios."""
    for q_heads, kv_heads in [(4, 1), (4, 2), (8, 2), (6, 3)]:
        q, c, wkc, wvc = _rand_mla_inputs(
            jax.random.PRNGKey(q_heads * 100 + kv_heads),
            batch=1,
            seq_len=12,
            q_heads=q_heads,
            kv_heads=kv_heads,
            d_nope=16,
            kv_rank=32,
            v_dim=16,
        )
        out = xla_flash_mla(q, c, wkc, wvc, causal=True)
        ref = _mla_dense_reference(q, c, wkc, wvc, causal=True)
        assert out.shape == ref.shape, f"GQA {q_heads}:{kv_heads}"
        assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_attention_mask_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(50),
        batch=2,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    # Mask that blocks last 4 KV positions for first batch
    mask = jnp.ones((2, 1, 12, 12), dtype=jnp.bool_)
    mask = mask.at[0, :, :, 8:].set(False)

    out = xla_flash_mla(q, c, wkc, wvc, attention_mask=mask)
    # Broadcast mask to [batch, q_heads, seq_q, kv_len] for reference
    mask_ref = jnp.broadcast_to(mask, (2, 4, 12, 12))
    ref = _mla_dense_reference(q, c, wkc, wvc, attention_mask=mask_ref)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_bias_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(60),
        batch=1,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    bias = jax.random.normal(jax.random.PRNGKey(61), (1, 4, 12, 12)) * 0.5

    out = xla_flash_mla(q, c, wkc, wvc, bias=bias)
    ref = _mla_dense_reference(q, c, wkc, wvc, bias=bias)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_logits_soft_cap_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(70),
        batch=2,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    out = xla_flash_mla(q, c, wkc, wvc, causal=True, logits_soft_cap=10.0)
    ref = _mla_dense_reference(q, c, wkc, wvc, causal=True, logits_soft_cap=10.0)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_sliding_window_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(80),
        batch=1,
        seq_len=16,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    out = xla_flash_mla(q, c, wkc, wvc, sliding_window=4)
    ref = _mla_dense_reference(q, c, wkc, wvc, sliding_window=4)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_asymmetric_sliding_window_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(85),
        batch=1,
        seq_len=16,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    out = xla_flash_mla(q, c, wkc, wvc, sliding_window=(6, 2))
    ref = _mla_dense_reference(q, c, wkc, wvc, sliding_window=(6, 2))
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_softmax_aux_matches_reference():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(90),
        batch=2,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    sinks = jnp.zeros((3,))  # 3 shared sink logits

    out = xla_flash_mla(q, c, wkc, wvc, causal=True, softmax_aux=sinks)
    ref = _mla_dense_reference(q, c, wkc, wvc, causal=True, softmax_aux=sinks)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_combined_features_match_reference():
    """Test multiple features used together."""
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(100),
        batch=2,
        seq_len=16,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    kwargs = dict(causal=True, sliding_window=6, logits_soft_cap=15.0)

    out = xla_flash_mla(q, c, wkc, wvc, **kwargs)
    ref = _mla_dense_reference(q, c, wkc, wvc, **kwargs)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_rope_with_bias_and_causal():
    """Fused RoPE + additive bias + causal = everything together."""
    key = jax.random.PRNGKey(110)
    batch, seq, q_heads, kv_heads = 1, 12, 4, 2
    d_nope, rope_dim, kv_rank, v_dim = 16, 8, 32, 16

    keys = jax.random.split(key, 6)
    q = jax.random.normal(keys[0], (batch, seq, q_heads, d_nope + rope_dim)) * 0.5
    c = jax.random.normal(keys[1], (batch, seq, kv_rank))
    wkc = jax.random.normal(keys[2], (kv_rank, kv_heads, d_nope)) * 0.1
    wvc = jax.random.normal(keys[3], (kv_rank, kv_heads, v_dim)) * 0.1
    b_k = jax.random.normal(keys[4], (batch, seq, rope_dim)) * 0.5
    bias = jax.random.normal(keys[5], (batch, q_heads, seq, seq)) * 0.3

    out = xla_flash_mla(q, c, wkc, wvc, b_k=b_k, causal=True, bias=bias)
    ref = _mla_dense_reference(q, c, wkc, wvc, b_k=b_k, causal=True, bias=bias)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_dropout_produces_different_outputs():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(120),
        batch=2,
        seq_len=16,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    rng1 = jax.random.PRNGKey(0)
    rng2 = jax.random.PRNGKey(1)

    out1 = xla_flash_mla(q, c, wkc, wvc, causal=True, deterministic=False, dropout_rng=rng1, dropout_prob=0.3)
    out2 = xla_flash_mla(q, c, wkc, wvc, causal=True, deterministic=False, dropout_rng=rng2, dropout_prob=0.3)
    out_det = xla_flash_mla(q, c, wkc, wvc, causal=True)

    assert out1.shape == out_det.shape
    assert jnp.isfinite(out1).all()
    assert jnp.isfinite(out2).all()
    # Different RNG keys → different outputs
    assert not jnp.allclose(out1, out2)
    # Deterministic ≠ dropout
    assert not jnp.allclose(out1, out_det)


def test_flash_mla_deterministic_ignores_dropout():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(130),
        batch=1,
        seq_len=8,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    rng = jax.random.PRNGKey(42)

    out1 = xla_flash_mla(q, c, wkc, wvc, causal=True, deterministic=True, dropout_rng=rng, dropout_prob=0.5)
    out2 = xla_flash_mla(q, c, wkc, wvc, causal=True)

    # deterministic=True should ignore dropout_prob entirely
    assert_allclose(out1, out2, atol=1e-6)


def test_flash_mla_dropout_without_rng_raises():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(131),
        batch=1,
        seq_len=8,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )

    with pytest.raises(ValueError, match="dropout_rng must be provided"):
        xla_flash_mla(q, c, wkc, wvc, causal=True, deterministic=False, dropout_prob=0.5)


def test_flash_mla_seq_len_1():
    """Single-token sequence (decode step)."""
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(140),
        batch=2,
        seq_len=1,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    out = xla_flash_mla(q, c, wkc, wvc, causal=True)
    ref = _mla_dense_reference(q, c, wkc, wvc, causal=True)
    assert out.shape == (2, 1, 4, 16)
    assert_allclose(out, ref, atol=1e-4)


def test_flash_mla_custom_softmax_scale():
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(150),
        batch=1,
        seq_len=8,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    scale = 0.05
    out = xla_flash_mla(q, c, wkc, wvc, causal=True, softmax_scale=scale)
    ref = _mla_dense_reference(q, c, wkc, wvc, causal=True, softmax_scale=scale)
    assert_allclose(out, ref, atol=1e-4)


def _grad_loss_fn(fn, q, c, wkc, wvc, b_q=None, b_k=None, **kwargs):
    """Compute scalar loss = sum(fn(...)) and return gradients of all inputs."""

    def loss_fn(q_, c_, wkc_, wvc_, bq_, bk_):
        out = fn(q_, c_, wkc_, wvc_, b_q=bq_, b_k=bk_, **kwargs)
        return jnp.sum(out)

    return jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, c, wkc, wvc, b_q, b_k)


def test_flash_mla_xla_backward_no_rope():
    """XLA backward: no RoPE, causal. Compare grads against reference."""
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(200),
        batch=2,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    kwargs = dict(causal=True)
    grads_xla = _grad_loss_fn(xla_flash_mla, q, c, wkc, wvc, **kwargs)
    grads_ref = _grad_loss_fn(_mla_dense_reference, q, c, wkc, wvc, **kwargs)

    names = ["dq", "dc", "dwkc", "dwvc", "db_q", "db_k"]
    for name, g_xla, g_ref in zip(names, grads_xla, grads_ref, strict=False):
        if g_xla is None and g_ref is None:
            continue
        assert_allclose(g_xla, g_ref, atol=1e-3, err_msg=f"XLA bwd {name}")


def test_flash_mla_xla_backward_fused_rope():
    """XLA backward: fused RoPE, causal."""
    key = jax.random.PRNGKey(210)
    batch, seq, q_heads, kv_heads = 2, 12, 4, 2
    d_nope, rope_dim, kv_rank, v_dim = 16, 8, 32, 16

    keys = jax.random.split(key, 5)
    q = jax.random.normal(keys[0], (batch, seq, q_heads, d_nope + rope_dim)) * 0.5
    c = jax.random.normal(keys[1], (batch, seq, kv_rank))
    wkc = jax.random.normal(keys[2], (kv_rank, kv_heads, d_nope)) * 0.1
    wvc = jax.random.normal(keys[3], (kv_rank, kv_heads, v_dim)) * 0.1
    b_k = jax.random.normal(keys[4], (batch, seq, rope_dim)) * 0.5

    kwargs = dict(causal=True)
    grads_xla = _grad_loss_fn(xla_flash_mla, q, c, wkc, wvc, b_k=b_k, **kwargs)
    grads_ref = _grad_loss_fn(_mla_dense_reference, q, c, wkc, wvc, b_k=b_k, **kwargs)

    names = ["dq", "dc", "dwkc", "dwvc", "db_q", "db_k"]
    for name, g_xla, g_ref in zip(names, grads_xla, grads_ref, strict=False):
        if g_xla is None and g_ref is None:
            continue
        assert_allclose(g_xla, g_ref, atol=1e-3, err_msg=f"XLA bwd fused {name}")


def test_flash_mla_xla_backward_decoupled_rope():
    """XLA backward: decoupled RoPE, causal."""
    key = jax.random.PRNGKey(220)
    batch, seq, q_heads, kv_heads = 2, 12, 4, 2
    d_nope, rope_dim, kv_rank, v_dim = 16, 8, 32, 16

    keys = jax.random.split(key, 6)
    q = jax.random.normal(keys[0], (batch, seq, q_heads, d_nope)) * 0.5
    c = jax.random.normal(keys[1], (batch, seq, kv_rank))
    wkc = jax.random.normal(keys[2], (kv_rank, kv_heads, d_nope)) * 0.1
    wvc = jax.random.normal(keys[3], (kv_rank, kv_heads, v_dim)) * 0.1
    b_q = jax.random.normal(keys[4], (batch, seq, rope_dim)) * 0.5
    b_k = jax.random.normal(keys[5], (batch, seq, rope_dim)) * 0.5

    kwargs = dict(causal=True)
    grads_xla = _grad_loss_fn(xla_flash_mla, q, c, wkc, wvc, b_q=b_q, b_k=b_k, **kwargs)
    grads_ref = _grad_loss_fn(_mla_dense_reference, q, c, wkc, wvc, b_q=b_q, b_k=b_k, **kwargs)

    names = ["dq", "dc", "dwkc", "dwvc", "db_q", "db_k"]
    for name, g_xla, g_ref in zip(names, grads_xla, grads_ref, strict=False):
        if g_xla is None and g_ref is None:
            continue
        assert_allclose(g_xla, g_ref, atol=4e-3, err_msg=f"XLA bwd decoupled {name}")


def test_flash_mla_xla_backward_with_bias():
    """XLA backward: no RoPE, causal, with additive bias."""
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(230),
        batch=1,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    bias = jax.random.normal(jax.random.PRNGKey(231), (1, 4, 12, 12)) * 0.3

    def loss_with_bias(fn, q_, c_, wkc_, wvc_, bias_):
        out = fn(q_, c_, wkc_, wvc_, causal=True, bias=bias_)
        return jnp.sum(out)

    grads_xla = jax.grad(loss_with_bias, argnums=(1, 2, 3, 4, 5))(xla_flash_mla, q, c, wkc, wvc, bias)
    grads_ref = jax.grad(loss_with_bias, argnums=(1, 2, 3, 4, 5))(_mla_dense_reference, q, c, wkc, wvc, bias)

    names = ["dq", "dc", "dwkc", "dwvc", "dbias"]
    for name, g_xla, g_ref in zip(names, grads_xla, grads_ref, strict=False):
        assert_allclose(g_xla, g_ref, atol=1e-3, err_msg=f"XLA bwd bias {name}")


def test_flash_mla_xla_backward_soft_cap():
    """XLA backward: no RoPE, causal, with logits soft cap."""
    q, c, wkc, wvc = _rand_mla_inputs(
        jax.random.PRNGKey(240),
        batch=2,
        seq_len=12,
        q_heads=4,
        kv_heads=2,
        d_nope=16,
        kv_rank=32,
        v_dim=16,
    )
    kwargs = dict(causal=True, logits_soft_cap=10.0)
    grads_xla = _grad_loss_fn(xla_flash_mla, q, c, wkc, wvc, **kwargs)
    grads_ref = _grad_loss_fn(_mla_dense_reference, q, c, wkc, wvc, **kwargs)

    names = ["dq", "dc", "dwkc", "dwvc", "db_q", "db_k"]
    for name, g_xla, g_ref in zip(names, grads_xla, grads_ref, strict=False):
        if g_xla is None and g_ref is None:
            continue
        assert_allclose(g_xla, g_ref, atol=1e-3, err_msg=f"XLA bwd softcap {name}")
