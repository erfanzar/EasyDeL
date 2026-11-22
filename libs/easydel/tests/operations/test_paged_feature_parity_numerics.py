"""Independent rank-major, visibility and populated-cache parity (accelerator only)."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.operations import OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_paged_reference_contract import _cache, _metadata, _storage_shapes


@pytest.mark.parametrize("kind", ["ring", "dropout", "cache_only"])
@pytest.mark.parametrize("segments", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_rank_major_bthd_pairwise_features_preserve_independent_reference(kind, segments, compiled):
    # Zero QK deliberately exposes visibility: causal row 1 must see BOTH
    # tokens, rather than the diagonal-only result of segment-ID conversion.
    q = np.zeros((2, 2, 2, 4), dtype=np.float32)
    k = np.zeros((2, 2, 1, 4), dtype=np.float32)
    v = np.broadcast_to(np.array([1, 3, 7, 11], dtype=np.float32).reshape(2, 2, 1, 1), (2, 2, 1, 4)).copy()
    pages = [np.zeros(shape, dtype=np.float32) for shape in _storage_shapes("unified", 4)]
    pages[1][0] = v[0]
    pages[1][2] = v[1]
    bias = np.zeros((2, 2, 2, 4), dtype=np.float32)
    bias[1, :, :, 0] = -0.4
    slopes = np.array([0.2, 0.5], dtype=np.float32)
    qs = np.array([[1, 1], [2, 3]], dtype=np.int32)
    ks = np.array([[1, 1, -1, -1], [2, 3, -1, -1]], dtype=np.int32)
    custom = np.ones((2, 1, 2, 4), dtype=bool)
    custom[1, :, 1, 0] = False
    rng = jax.random.key(19)
    probability = 0.0 if kind == "ring" else 0.25
    expected = np.zeros_like(q)
    for row in range(2):
        visible = np.arange(4)[None, :] <= np.arange(2)[:, None]
        visible &= np.arange(4)[None, :] < 2
        visible &= (qs[row, :, None] == ks[row, None, :]) if segments else custom[row, 0]
        logits = bias[row] + slopes[:, None, None] * np.arange(4)[None, None, :]
        logits = np.where(visible[None], logits, -np.inf)
        # Include a shared sink of logit zero, matching the operation below.
        all_logits = np.concatenate((logits, np.zeros((2, 2, 1))), axis=-1)
        weights = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        weights = weights[..., :4]
        if probability:
            keep = np.asarray(jax.random.bernoulli(jax.random.fold_in(rng, row), 1 - probability, (1, 2, 4, 4)))
            weights = np.where(keep[0, :, row * 2 : row * 2 + 2], weights / (1 - probability), 0)
        vals = np.concatenate((v[row, :, 0], np.zeros((2, 4))), axis=0)
        expected[row] = np.einsum("hqk,kd->qhd", weights, vals)

    operation = OperationRegistry.create(
        "unified_attention" if kind == "cache_only" else "paged_attention_reference", _metadata()
    )

    def forward(q, k, v, key_pages, value_pages):
        cache = _cache("unified", 4, (key_pages, value_pages))
        runtime = RaggedPagesMetadata(
            jnp.array([[[0, 1]], [[2, 3]]]),
            jnp.array([[2], [2]]),
            jnp.array([[0, 2], [0, 2]]),
            jnp.array([[2], [2]]),
            page_size=2,
        )
        result = operation(
            query=q,
            **({} if kind == "cache_only" else {"key": k, "value": v}),
            cache_view=cache,
            cache_metadata=runtime,
            mask_info=MaskInfo.from_segments(jnp.asarray(qs), jnp.asarray(ks))
            if segments
            else MaskInfo(_attention_mask=jnp.asarray(custom)),
            bias=jnp.asarray(bias),
            alibi_slopes=jnp.asarray(slopes),
            softmax_aux=jnp.zeros((1,)),
            dense_impl="ring" if kind == "ring" else "vanilla",
            deterministic=not probability,
            dropout_prob=probability,
            dropout_rng=rng,
            compute_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
        )
        return result.attention_outputs, result.cache_view.key_cache, result.cache_view.value_cache

    if compiled:
        forward = jax.jit(forward)
    output, updated_keys, updated_values = forward(*map(jnp.asarray, (q, k, v, *pages)))
    np.testing.assert_allclose(output, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(updated_keys, pages[0])
    np.testing.assert_array_equal(updated_values, pages[1])
