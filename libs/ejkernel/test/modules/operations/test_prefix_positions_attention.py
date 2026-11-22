"""Absolute q/kv positions with segment ids when ``q_len != kv_len`` (queries continuing a kv prefix).

The kv sequence is a left-padded prefix (length ``P``) followed by the queries' own tokens
(length ``C``), so a query at absolute position ``P + i`` must see exactly the keys with
``kv_pos <= q_pos`` and an equal, non-negative segment id. Outputs and q/k/v gradients are
compared against an independent dense reference in fp32 at highest matmul precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._xla.flash_attention import flash_attention as xla_flash_kernel
from ejkernel.kernels._xla.scaled_dot_product_attention import scaled_dot_product_attention as xla_sdpa_kernel
from ejkernel.modules.operations import flash_attention, scaled_dot_product_attention
from ejkernel.types import MaskInfo

HQ, HKV, D = 2, 1, 128
TOL = 1e-4


def _inputs(batch: int, prefix: int, completion: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kv_len = prefix + completion
    query = rng.standard_normal((batch, completion, HQ, D)).astype(np.float32)
    key = rng.standard_normal((batch, kv_len, HKV, D)).astype(np.float32)
    value = rng.standard_normal((batch, kv_len, HKV, D)).astype(np.float32)
    prompt_valid = np.zeros((batch, prefix), dtype=bool)
    completion_valid = np.zeros((batch, completion), dtype=bool)
    for row in range(batch):
        prompt_len = max(1, prefix - (row * 3 + 1) % max(prefix // 2, 1))
        prompt_valid[row, prefix - prompt_len :] = True  # left padded
        completion_len = max(1, completion - (row * 2 + 1) % max(completion // 2, 1))
        completion_valid[row, :completion_len] = True  # right padded
    q_seg = np.where(completion_valid, 0, -1).astype(np.int32)
    kv_seg = np.concatenate([np.where(prompt_valid, 0, -1), q_seg], axis=1).astype(np.int32)
    q_pos = np.broadcast_to(prefix + np.arange(completion), (batch, completion)).astype(np.int32)
    kv_pos = np.broadcast_to(np.arange(kv_len), (batch, kv_len)).astype(np.int32)
    cot = rng.standard_normal((batch, completion, HQ, D)).astype(np.float32)
    return query, key, value, q_seg, kv_seg, q_pos, kv_pos, cot


def _visibility(q_seg, kv_seg, q_pos, kv_pos):
    return (
        (q_seg[:, :, None] >= 0)
        & (kv_seg[:, None, :] >= 0)
        & (q_seg[:, :, None] == kv_seg[:, None, :])
        & (kv_pos[:, None, :] <= q_pos[:, :, None])
    )


def _reference(query, key, value, visible):
    key = jnp.repeat(key, HQ // HKV, axis=2)
    value = jnp.repeat(value, HQ // HKV, axis=2)
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=jax.lax.Precision.HIGHEST) * (D**-0.5)
    logits = jnp.where(visible[:, None], logits, -1e30)
    weights = jnp.where(visible[:, None], jax.nn.softmax(logits, axis=-1), 0.0)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, value, precision=jax.lax.Precision.HIGHEST)


def _mask(representation, q_seg, kv_seg, q_pos, kv_pos):
    if representation == "segments":
        return MaskInfo.from_segments(
            q_segment_ids=jnp.asarray(q_seg),
            kv_segment_ids=jnp.asarray(kv_seg),
            q_positions=jnp.asarray(q_pos),
            kv_positions=jnp.asarray(kv_pos),
        )
    visible = _visibility(q_seg, kv_seg, q_pos, kv_pos)
    return MaskInfo(
        _attention_mask=jnp.asarray(visible[:, None]),
        q_positions=jnp.asarray(q_pos),
        kv_positions=jnp.asarray(kv_pos),
        causal_mask_baked_in=True,
    )


def _call(op, platform, q, k, v, mask):
    if op == "sdpa":
        return scaled_dot_product_attention(q, k, v, mask_info=mask, causal=True, platform=platform)
    # The XLA flash backend honours an explicit matmul precision (DEFAULT would be bf16 passes on TPU);
    # the Pallas TPU kernel only accepts DEFAULT and accumulates fp32 inputs in fp32.
    precision = jax.lax.Precision.HIGHEST if platform == "xla" else jax.lax.Precision.DEFAULT
    return flash_attention(q, k, v, mask_info=mask, causal=True, platform=platform, precision=precision)


def _errors(op, platform, representation, prefix, completion):
    query, key, value, q_seg, kv_seg, q_pos, kv_pos, cot = _inputs(2, prefix, completion)
    visible = jnp.asarray(_visibility(q_seg, kv_seg, q_pos, kv_pos))
    q_valid = jnp.asarray(q_seg >= 0)[:, :, None, None].astype(jnp.float32)
    mask = _mask(representation, q_seg, kv_seg, q_pos, kv_pos)

    def op_loss(q, k, v):
        out = _call(op, platform, q, k, v, mask)
        return jnp.sum(out * cot * q_valid), out

    def ref_loss(q, k, v):
        out = _reference(q, k, v, visible)
        return jnp.sum(out * cot * q_valid), out

    with jax.default_matmul_precision("highest"):
        (_, out), grads = jax.value_and_grad(op_loss, argnums=(0, 1, 2), has_aux=True)(query, key, value)
        (_, ref), ref_grads = jax.value_and_grad(ref_loss, argnums=(0, 1, 2), has_aux=True)(query, key, value)
    rows = np.asarray(q_seg >= 0)
    errors = {"out": float(np.max(np.abs(np.asarray(out)[rows] - np.asarray(ref)[rows])))}
    for label, got, want in zip(("dq", "dk", "dv"), grads, ref_grads, strict=True):
        errors[label] = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    print(f"[prefix-positions] {op} {platform} {representation} P={prefix} C={completion} errors={errors}")
    return errors


# Only the segments-plus-positions representation (no stored dense mask) takes the
# position-aware path; a stored dense mask keeps the pre-existing behaviour (tested below).
CASES = [
    pytest.param("sdpa", "xla", "segments", 12, 8),
    pytest.param("sdpa", "xla", "segments", 256, 128),
    pytest.param("flash", "xla", "segments", 12, 8),
    pytest.param("flash", "xla", "segments", 256, 128),
    pytest.param("flash", None, "segments", 128, 128),
    pytest.param("flash", None, "segments", 256, 128),
]


@pytest.mark.parametrize("op,platform,representation,prefix,completion", CASES)
def test_prefix_positions_match_dense_reference(op, platform, representation, prefix, completion):
    errors = _errors(op, platform, representation, prefix, completion)
    assert all(np.isfinite(v) and v <= TOL for v in errors.values()), errors


def _segment_inputs(q_len: int, kv_len: int):
    """Self-attention style packed segments without positions (the pre-existing path)."""
    query, key, value, *_ = _inputs(2, kv_len - q_len, q_len, seed=3)
    query, key, value = (jnp.asarray(x) for x in (query, key, value))
    seg = np.array([[0] * (q_len // 2) + [1] * (q_len - q_len // 2 - 2) + [-1] * 2] * 2, dtype=np.int32)
    return query, key, value, seg


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_without_positions_is_unchanged(causal):
    query, key, value, seg = _segment_inputs(128, 128)
    mask = MaskInfo.from_segments(jnp.asarray(seg))
    out = scaled_dot_product_attention(query, key, value, mask_info=mask, causal=causal, platform="xla")
    legacy = xla_sdpa_kernel(
        jnp.asarray(query),
        jnp.asarray(key),
        jnp.asarray(value),
        attention_mask=mask.get_or_compute_attention_mask(),
        causal=causal,
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(legacy), atol=1e-6, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_unequal_lengths_without_positions_is_unchanged(causal):
    """q_len != kv_len without positions keeps the index-based backend path (no position term)."""
    query, key, value, q_seg, kv_seg, *_ = _inputs(2, 128, 128, seed=5)
    query, key, value = (jnp.asarray(x) for x in (query, key, value))
    plain = MaskInfo.from_segments(jnp.asarray(q_seg), jnp.asarray(kv_seg))
    out = scaled_dot_product_attention(query, key, value, mask_info=plain, causal=causal, platform="xla")
    legacy = xla_sdpa_kernel(
        jnp.asarray(query),
        jnp.asarray(key),
        jnp.asarray(value),
        attention_mask=plain.get_or_compute_attention_mask(),
        causal=causal,
    )
    np.testing.assert_allclose(np.asarray(out), np.asarray(legacy), atol=1e-6, rtol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_positions_with_equal_lengths_take_the_legacy_path(causal):
    """With ``q_len == kv_len`` positions do not alter the result (bit-identical to positions=None)."""
    query, key, value, seg = _segment_inputs(128, 128)
    plain = MaskInfo.from_segments(jnp.asarray(seg))
    positions = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32), (2, 128))
    with_pos = MaskInfo.from_segments(jnp.asarray(seg), q_positions=positions, kv_positions=positions)
    a = scaled_dot_product_attention(query, key, value, mask_info=plain, causal=causal, platform="xla")
    b = scaled_dot_product_attention(query, key, value, mask_info=with_pos, causal=causal, platform="xla")
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("platform", ["xla", None])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_without_positions_is_unchanged(platform, causal):
    query, key, value, seg = _segment_inputs(128, 128)
    plain = MaskInfo.from_segments(jnp.asarray(seg))
    positions = jnp.broadcast_to(jnp.arange(128, dtype=jnp.int32), (2, 128))
    with_pos = MaskInfo.from_segments(jnp.asarray(seg), q_positions=positions, kv_positions=positions)
    a = flash_attention(query, key, value, mask_info=plain, causal=causal, platform=platform)
    b = flash_attention(query, key, value, mask_info=with_pos, causal=causal, platform=platform)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    if platform == "xla":
        legacy = xla_flash_kernel(
            jnp.asarray(query),
            jnp.asarray(key),
            jnp.asarray(value),
            causal=causal,
            q_segment_ids=jnp.asarray(seg),
            kv_segment_ids=jnp.asarray(seg),
        )
        legacy = jnp.where(jnp.asarray(seg >= 0)[:, :, None, None], legacy, 0)
        np.testing.assert_allclose(np.asarray(a), np.asarray(legacy), atol=1e-6, rtol=0)


def test_flash_positions_refuse_logits_soft_cap():
    query, key, value, q_seg, kv_seg, q_pos, kv_pos, _ = _inputs(2, 12, 8)
    mask = _mask("segments", q_seg, kv_seg, q_pos, kv_pos)
    with pytest.raises(NotImplementedError, match="logits_soft_cap"):
        flash_attention(query, key, value, mask_info=mask, causal=True, logits_soft_cap=30.0, platform="xla")


def _cache_style(prefix: int, completion: int):
    """Positions plus a baked dense mask (what a KV cache attaches on cached steps)."""
    query, key, value, q_seg, kv_seg, q_pos, kv_pos, _ = _inputs(2, prefix, completion, seed=7)
    query, key, value = (jnp.asarray(x) for x in (query, key, value))
    dense = jnp.asarray(_visibility(q_seg, kv_seg, q_pos, kv_pos)[:, None])
    with_pos = MaskInfo(_attention_mask=dense, q_positions=jnp.asarray(q_pos), kv_positions=jnp.asarray(kv_pos))
    plain = MaskInfo(_attention_mask=dense)
    return query, key, value, dense, with_pos, plain


def test_cache_style_mask_does_not_take_the_position_path():
    *_, with_pos, plain = _cache_style(12, 8)
    assert with_pos.has_explicit_positions
    assert not with_pos.uses_position_causality()
    assert not plain.uses_position_causality()
    _, _, _, q_seg, kv_seg, q_pos, kv_pos, _ = _inputs(2, 12, 8)
    assert _mask("segments", q_seg, kv_seg, q_pos, kv_pos).uses_position_causality()


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_cache_style_positions_with_dense_mask_is_unchanged(causal):
    query, key, value, dense, with_pos, plain = _cache_style(128, 128)
    a = scaled_dot_product_attention(query, key, value, mask_info=with_pos, causal=causal, platform="xla")
    b = scaled_dot_product_attention(query, key, value, mask_info=plain, causal=causal, platform="xla")
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    legacy = xla_sdpa_kernel(query, key, value, attention_mask=dense, causal=causal)
    np.testing.assert_allclose(np.asarray(a), np.asarray(legacy), atol=1e-6, rtol=0)


@pytest.mark.parametrize("platform", ["xla", None])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_cache_style_positions_with_dense_mask_is_unchanged(platform, causal):
    query, key, value, dense, with_pos, plain = _cache_style(128, 128)
    a = flash_attention(query, key, value, mask_info=with_pos, causal=causal, platform=platform)
    b = flash_attention(query, key, value, mask_info=plain, causal=causal, platform=platform)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    if platform == "xla":
        legacy = xla_flash_kernel(query, key, value, attention_mask=dense, causal=causal)
        np.testing.assert_allclose(np.asarray(a), np.asarray(legacy), atol=1e-6, rtol=0)
