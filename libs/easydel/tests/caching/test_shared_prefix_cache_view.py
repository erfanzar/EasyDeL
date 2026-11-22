"""Prefix capture/fork cache views: gather layout, fork mask and gradient routing."""

import easydel.caching.shared_prefix as shared_prefix
import jax
import numpy as np
import pytest
from easydel.caching.shared_prefix import PrefixCaptureView, PrefixForkView
from ejkernel.types import MaskInfo
from jax import numpy as jnp

B, P, C, G, H, D = 2, 3, 2, 2, 1, 4


def _arrays():
    rng = np.random.default_rng(0)
    prefix_key = rng.standard_normal((B, P, H, D)).astype(np.float32)
    prefix_value = rng.standard_normal((B, P, H, D)).astype(np.float32)
    key = rng.standard_normal((B * G, C, H, D)).astype(np.float32)
    value = rng.standard_normal((B * G, C, H, D)).astype(np.float32)
    prefix_seg = np.array([[-1, 0, 0], [0, 0, 0]], np.int32)
    q_seg = np.array([[0, 0], [0, -1], [0, 0], [0, -1]], np.int32)
    return prefix_key, prefix_value, key, value, prefix_seg, q_seg


def _fork(prefix_key, prefix_value, prefix_seg):
    return PrefixForkView(
        prefix_key=jnp.asarray(prefix_key),
        prefix_value=jnp.asarray(prefix_value),
        prefix_segment_ids=jnp.asarray(prefix_seg),
        num_generations=G,
    )


def test_fork_view_gathers_group_prefix_and_builds_absolute_positions():
    prefix_key, prefix_value, key, value, prefix_seg, q_seg = _arrays()
    view = _fork(prefix_key, prefix_value, prefix_seg)
    k, v, mask, _ = view.concatenate_to_cache(
        key=jnp.asarray(key), value=jnp.asarray(value), mask_info=MaskInfo.from_segments(jnp.asarray(q_seg))
    )
    rows = np.arange(B * G) // G
    np.testing.assert_array_equal(np.asarray(k), np.concatenate([prefix_key[rows], key], axis=1))
    np.testing.assert_array_equal(np.asarray(v), np.concatenate([prefix_value[rows], value], axis=1))
    np.testing.assert_array_equal(np.asarray(mask.q_segment_ids), q_seg)
    np.testing.assert_array_equal(np.asarray(mask.kv_segment_ids), np.concatenate([prefix_seg[rows], q_seg], axis=1))
    np.testing.assert_array_equal(np.asarray(mask.q_positions), np.broadcast_to(P + np.arange(C), (B * G, C)))
    np.testing.assert_array_equal(np.asarray(mask.kv_positions), np.broadcast_to(np.arange(P + C), (B * G, P + C)))


def test_fork_view_gradient_sums_over_group_rows():
    prefix_key, prefix_value, key, value, prefix_seg, q_seg = _arrays()
    weights = np.random.default_rng(1).standard_normal((B * G, P + C, H, D)).astype(np.float32)

    def loss(pk):
        view = _fork(pk, prefix_value, prefix_seg)
        k, _, _, _ = view.concatenate_to_cache(
            key=jnp.asarray(key), value=jnp.asarray(value), mask_info=MaskInfo.from_segments(jnp.asarray(q_seg))
        )
        return jnp.sum(k * weights)

    grad = jax.grad(loss)(jnp.asarray(prefix_key))
    expected = weights[:, :P].reshape(B, G, P, H, D).sum(axis=1)
    np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-6)


def test_capture_view_passes_states_through_and_records_them():
    prefix_key, prefix_value, *_ = _arrays()
    k, _, mask, view = PrefixCaptureView().concatenate_to_cache(
        key=jnp.asarray(prefix_key), value=jnp.asarray(prefix_value), mask_info=None
    )
    np.testing.assert_array_equal(np.asarray(k), prefix_key)
    np.testing.assert_array_equal(np.asarray(view.value), prefix_value)
    assert mask is None


def test_fork_view_rejects_batch_not_equal_to_prompts_times_groups():
    prefix_key, prefix_value, key, value, prefix_seg, q_seg = _arrays()
    view = _fork(prefix_key, prefix_value, prefix_seg)
    with pytest.raises(ValueError, match="num_generations"):
        view.concatenate_to_cache(
            key=jnp.asarray(key[:3]),
            value=jnp.asarray(value[:3]),
            mask_info=MaskInfo.from_segments(jnp.asarray(q_seg[:3])),
        )


def _dense_fork_reference(query, prefix_key, prefix_value, key, value, prefix_seg, q_seg, groups, scale):
    """Softmax attention over the materialized ``concat(prefix[r // G], completion[r])`` keys."""
    rows = jnp.arange(query.shape[0]) // groups
    reps = query.shape[2] // key.shape[2]
    full_key = jnp.repeat(jnp.concatenate([prefix_key[rows], key], axis=1), reps, axis=2)
    full_value = jnp.repeat(jnp.concatenate([prefix_value[rows], value], axis=1), reps, axis=2)
    prefix_len, completion_len = prefix_key.shape[1], key.shape[1]
    kv_valid = jnp.concatenate([prefix_seg[rows] >= 0, q_seg >= 0], axis=1)
    causal = jnp.arange(prefix_len + completion_len)[None, :] <= prefix_len + jnp.arange(completion_len)[:, None]
    valid = kv_valid[:, None, None, :] & causal[None, None]
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, full_key) * scale
    weights = jax.nn.softmax(jnp.where(valid, logits, -jnp.inf), axis=-1)
    out = jnp.einsum("bhqk,bkhd->bqhd", weights, full_value)
    return jnp.where((q_seg >= 0)[:, :, None, None], out, 0)


@pytest.mark.parametrize("query_block", [256, 3, 1])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_grouped_fork_attention_matches_concatenated_softmax_and_gradients(monkeypatch, groups, query_block):
    """Exact LSE merge of the shared-prefix and causal own-completion parts, including causal query blocks."""
    monkeypatch.setattr(shared_prefix, "GROUPED_FORK_QUERY_BLOCK", query_block)
    prompts, prefix_len, completion_len, kv_heads, reps, dim = 2, 5, 7, 2, 2, 8
    rng = np.random.default_rng(groups)
    prefix_key = jnp.asarray(rng.standard_normal((prompts, prefix_len, kv_heads, dim)), jnp.float32)
    prefix_value = jnp.asarray(rng.standard_normal((prompts, prefix_len, kv_heads, dim)), jnp.float32)
    rows = prompts * groups
    query = jnp.asarray(rng.standard_normal((rows, completion_len, kv_heads * reps, dim)), jnp.float32)
    key = jnp.asarray(rng.standard_normal((rows, completion_len, kv_heads, dim)), jnp.float32)
    value = jnp.asarray(rng.standard_normal((rows, completion_len, kv_heads, dim)), jnp.float32)
    prefix_seg = np.zeros((prompts, prefix_len), np.int32)
    prefix_seg[0, :2] = -1  # left-padded ragged prompt
    lengths = rng.integers(1, completion_len + 1, rows)
    q_seg = np.where(np.arange(completion_len)[None, :] < lengths[:, None], 0, -1).astype(np.int32)
    prefix_seg, q_seg = jnp.asarray(prefix_seg), jnp.asarray(q_seg)
    cotangent = jnp.asarray(rng.standard_normal(query.shape), jnp.float32)
    scale = dim**-0.5

    def grouped(q, pk, pv, k, v):
        view = PrefixForkView(prefix_key=pk, prefix_value=pv, prefix_segment_ids=prefix_seg, num_generations=groups)
        out = view.grouped_fork_attention(
            query=q,
            key=k,
            value=v,
            mask_info=MaskInfo.from_segments(q_seg),
            softmax_scale=scale,
            runtime_dtype=jnp.float32,
            softmax_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        return out, jnp.sum(out * cotangent)

    def reference(q, pk, pv, k, v):
        out = _dense_fork_reference(q, pk, pv, k, v, prefix_seg, q_seg, groups, scale)
        return out, jnp.sum(out * cotangent)

    args = (query, prefix_key, prefix_value, key, value)
    with jax.default_matmul_precision("highest"):
        out, _ = grouped(*args)
        ref, _ = reference(*args)
        grads = jax.grad(lambda *a: grouped(*a)[1], argnums=(0, 1, 2, 3, 4))(*args)
        ref_grads = jax.grad(lambda *a: reference(*a)[1], argnums=(0, 1, 2, 3, 4))(*args)
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-5, atol=1e-5)
    for name, g, r in zip(("query", "prefix_key", "prefix_value", "key", "value"), grads, ref_grads, strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(r), rtol=1e-4, atol=1e-5, err_msg=name)
