"""Dense and paged attention numerics and cache writes against a NumPy reference (accelerator-only)."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.operations import OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_paged_reference_contract import _cache, _metadata, _storage_shapes


def _reference(q, k, v, visible, sinks, *, normalized, scale, cap, keep=None, probability=0.0):
    k = np.repeat(k, q.shape[2] // k.shape[2], axis=2)
    v = np.repeat(v, q.shape[2] // v.shape[2], axis=2)
    logits = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)) * scale
    logits = cap * np.tanh(logits / cap)
    logits = np.where(visible, logits, -np.inf)
    sink_logits = np.broadcast_to(sinks[None, :, None, :], (*logits.shape[:-1], sinks.shape[-1]))
    all_logits = np.concatenate((logits, sink_logits), axis=-1)
    numerator = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
    if normalized:
        numerator /= numerator.sum(axis=-1, keepdims=True)
    weights = numerator[..., : k.shape[1]]
    if keep is not None:
        weights = np.where(keep, weights / (1.0 - probability), 0.0)
    return np.einsum("bhqk,bkhd->bqhd", weights, v), weights


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_dense_weight_dropout_with_sinks_masks_softcap_and_normalization(normalized, compiled):
    rng = np.random.default_rng(255)
    q = rng.normal(size=(1, 3, 2, 4)).astype(np.float32)
    k = rng.normal(size=(1, 5, 1, 4)).astype(np.float32)
    v = rng.normal(size=(1, 5, 1, 6)).astype(np.float32)
    visible = np.array([[[[True, False, True, False, False], [True] * 5, [False] * 5]]])
    sinks = np.array([[0.3], [-0.1]], dtype=np.float32)
    key = jax.random.key(17)
    keep = np.asarray(jax.random.bernoulli(key, 0.75, (1, 2, 3, 5)))
    operation = OperationRegistry.create("vanilla", _metadata())

    def forward(q, k, v, key):
        return operation(
            q,
            k,
            v,
            mask_info=MaskInfo(_attention_mask=jnp.asarray(visible)),
            softmax_aux=jnp.asarray(sinks),
            softmax_scale=0.5,
            logits_soft_cap=1.3,
            normalize_output=normalized,
            deterministic=False,
            dropout_prob=0.25,
            dropout_rng=key,
            precision=jax.lax.Precision.HIGHEST,
        )

    if compiled:
        forward = jax.jit(forward)
    result = forward(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), key)
    expected, weights = _reference(
        q, k, v, visible, sinks, normalized=normalized, scale=0.5, cap=1.3, keep=keep, probability=0.25
    )
    np.testing.assert_allclose(result.attention_outputs, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(result.attention_weights, weights, rtol=3e-5, atol=3e-6)
    repeated = forward(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), key)
    np.testing.assert_array_equal(result.attention_outputs, repeated.attention_outputs)
    assert result.attention_weights.dtype == jnp.float32
    np.testing.assert_array_equal(result.attention_outputs[:, 2], 0.0)


def test_unnormalized_padding_mask_keeps_batch_axis_when_query_length_is_one():
    operation = OperationRegistry.create("vanilla", _metadata())
    query = jnp.zeros((3, 1, 1, 4), dtype=jnp.float32)
    keys = jnp.zeros((3, 2, 1, 4), dtype=jnp.float32)
    values = jnp.broadcast_to(jnp.array([2.0, 8.0])[None, :, None, None], (3, 2, 1, 4))
    mask = MaskInfo(_attention_mask=jnp.array([[True, False], [False, True], [False, False]]))
    result = operation(query, keys, values, mask_info=mask, normalize_output=False, softmax_aux=jnp.zeros((1, 1)))
    expected = np.broadcast_to(np.array([2.0, 8.0, 0.0])[:, None, None, None], (3, 1, 1, 4))
    np.testing.assert_array_equal(result.attention_outputs, expected)


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("head_dim", [4, 64])
@pytest.mark.parametrize("kind", ["ring", "dropout", "unnormalized_dropout"])
@pytest.mark.parametrize("compiled", [False, True])
def test_continuous_batching_preserves_history_request_isolation_and_page_writes(layout, head_dim, kind, compiled):
    rng = np.random.default_rng(199)
    q = rng.normal(size=(4, 2, head_dim)).astype(np.float32)
    k = rng.normal(size=(4, 1, head_dim)).astype(np.float32)
    v = rng.normal(size=(4, 1, head_dim)).astype(np.float32)
    storage = [rng.normal(size=shape).astype(np.float32) for shape in _storage_shapes(layout, head_dim)]
    tables = np.array([[2, 0], [3, 1], [-1, -1]], dtype=np.int32)
    contexts = np.array([3, 2, 0], dtype=np.int32)
    starts = np.array([0, 1, 3, 3], dtype=np.int32)
    # Decode request 0 retains two historical tokens on non-contiguous pages;
    # request 1 is a two-token prefill; request 2 and token 3 are padding.
    paired = layout == "v3" and head_dim == 64
    if layout == "unified":
        old_keys, old_values = storage[0].copy(), storage[1].copy()
    else:
        flat = storage[0].reshape(4, 2, -1, storage[0].shape[-1])
        old_keys = (flat[..., :head_dim] if paired else flat[:, :, 0::2, :head_dim])[:, :, :1].copy()
        old_values = (flat[..., head_dim : 2 * head_dim] if paired else flat[:, :, 1::2, :head_dim])[:, :, :1].copy()
    expected_keys, expected_values = old_keys.copy(), old_values.copy()
    for token, page, offset in [(0, 0, 0), (1, 3, 0), (2, 3, 1)]:
        expected_keys[page, offset] = k[token]
        expected_values[page, offset] = v[token]
    sinks = np.array([[0.2], [-0.1]], dtype=np.float32)
    normalized = kind != "unnormalized_dropout"
    probability = 0.0 if kind == "ring" else 0.25
    key = jax.random.key(29)
    expected = np.zeros_like(q)
    for row, (start, end, context) in enumerate([(0, 1, 3), (1, 3, 2)]):
        keys = expected_keys[tables[row]].reshape(1, 4, 1, head_dim)
        values = expected_values[tables[row]].reshape(1, 4, 1, head_dim)
        visible = np.zeros((1, 1, 4, 4), dtype=bool)
        for token in range(start, end):
            position = context - (end - start) + token - start
            visible[0, 0, token, : position + 1] = True
        keep = None
        if probability:
            keep = np.asarray(jax.random.bernoulli(jax.random.fold_in(key, row), 1 - probability, (1, 2, 4, 4)))
        result, _ = _reference(
            q[None],
            keys,
            values,
            visible,
            sinks,
            normalized=normalized,
            scale=0.5,
            cap=1.3,
            keep=keep,
            probability=probability,
        )
        expected[start:end] = result[0, start:end]
    name = (
        ("ring" if layout == "unified" else "ring_paged")
        if kind == "ring"
        else {
            "v2": "ragged_page_attention_v2",
            "v3": "ragged_page_attention_v3",
            "unified": "unified_attention",
        }[layout]
    )
    operation = OperationRegistry.create(name, _metadata())

    def forward(q, k, v, *storage):
        cache = _cache(layout, head_dim, storage)
        runtime = RaggedPagesMetadata(
            jnp.asarray(tables),
            jnp.asarray(contexts),
            jnp.asarray(starts),
            jnp.array([2]),
            request_distribution=jnp.array([1, 2, 2]),
            page_size=2,
        )
        result = operation(
            query=q,
            key=k,
            value=v,
            cache_view=cache,
            compute_dtype=jnp.float32,
            cache_metadata=runtime,
            deterministic=kind == "ring",
            dropout_prob=probability,
            dropout_rng=key,
            softmax_scale=0.5,
            logits_soft_cap=1.3,
            softmax_aux=jnp.asarray(sinks),
            normalize_output=normalized,
            precision=jax.lax.Precision.HIGHEST,
        )
        cache = result.cache_view
        arrays = (cache.key_cache, cache.value_cache) if layout == "unified" else (cache.kv_pages,)
        return result.attention_outputs, arrays

    if compiled:
        forward = jax.jit(forward)
    output, updated_storage = forward(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), *map(jnp.asarray, storage))
    np.testing.assert_allclose(output, expected, rtol=5e-5, atol=5e-6)
    np.testing.assert_array_equal(output[3], 0.0)
    if layout == "unified":
        actual_keys, actual_values = updated_storage
    else:
        flat = np.asarray(updated_storage[0]).reshape(4, 2, -1, storage[0].shape[-1])
        actual_keys = (flat[..., :head_dim] if paired else flat[:, :, 0::2, :head_dim])[:, :, :1]
        actual_values = (flat[..., head_dim : 2 * head_dim] if paired else flat[:, :, 1::2, :head_dim])[:, :, :1]
    np.testing.assert_array_equal(actual_keys, expected_keys)
    np.testing.assert_array_equal(actual_values, expected_values)
    # An unrelated page is bitwise unchanged, including physical storage padding.
    for old, updated in zip(storage, updated_storage, strict=True):
        np.testing.assert_array_equal(updated[1], old[1])
