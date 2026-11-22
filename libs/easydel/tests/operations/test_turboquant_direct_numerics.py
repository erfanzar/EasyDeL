"""Compressed-cache attention numerics against an independent NumPy reference (accelerator-only)."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.caching.turboquant_ragged_page import TurboQuantRaggedPagesCacheView
from easydel.operations import OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_compressed_cache_ring_safety import PAGE_FIELDS, _compressed_cache, _constants
from .test_paged_reference_contract import _metadata


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    # Quantization/reconstruction also contains fp32 dots, in addition to
    # attention's explicitly requested precision. Keep the NumPy comparison
    # independent of TPU's bf16-speed DEFAULT dot algorithm.
    with jax.default_matmul_precision("highest"):
        yield


def _numpy_compress(x, constants, *, keys):
    """Independent nearest-centroid quantization and little-endian byte packing."""
    norm = np.linalg.norm(x, axis=-1)
    rotated = (x / np.maximum(norm[..., None], 1e-8)) @ constants.rotation_matrix.T
    book = constants.key_codebook if keys else constants.value_codebook
    codes = np.argmin(np.abs(rotated[..., None] - book), axis=-1).astype(np.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    if not keys:
        return packed, norm.astype(jnp.bfloat16)
    residual = rotated - book[codes]
    signs = residual @ constants.qjl_projection.T >= 0
    packed_signs = np.sum(signs.reshape(*signs.shape[:-1], -1, 8) * (1 << np.arange(8)), axis=-1).astype(np.uint8)
    norms = np.stack((norm, np.linalg.norm(residual, axis=-1)), axis=-1).astype(jnp.bfloat16)
    return packed, packed_signs, norms


def _numpy_decompress(pages, constants):
    ki, ks, kn, vi, vn = pages

    def codes(packed):
        return np.stack((packed & 15, packed >> 4), axis=-1).reshape(*packed.shape[:-1], -1)

    centroids = constants.key_codebook[codes(ki)]
    signs = (((ks[..., None] >> np.arange(8)) & 1).reshape(*ks.shape[:-1], -1) * 2 - 1).astype(np.float32)
    norms = kn.astype(np.float32)
    effective = centroids * norms[..., :1]
    effective += (signs @ constants.qjl_projection) * norms[..., 1:] * np.sqrt(np.pi / 2) / constants.qjl_dim
    keys = effective @ constants.rotation_matrix
    values = (constants.value_codebook[codes(vi)] @ constants.rotation_matrix) * vn.astype(np.float32)[..., None]
    return keys, values


@pytest.mark.parametrize("name", ["paged_attention_reference", "ragged_page_attention_v2", "ragged_page_attention_v3"])
@pytest.mark.parametrize("dropout,normalized", [(True, True), (False, False), (True, False)])
@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("combined_mask", [False, True])
def test_direct_compressed_history_writes_qjl_and_dropout_match_numpy(name, dropout, normalized, update, combined_mask):
    rng = np.random.default_rng(562)
    constants = _constants()
    query = rng.normal(size=(2, 2, 2, 8)).astype(np.float32)
    key = rng.normal(size=(2, 2, 1, 8)).astype(np.float32)
    value = rng.normal(size=(2, 2, 1, 8)).astype(np.float32)
    initial_k = rng.normal(size=(4, 2, 1, 8)).astype(np.float32)
    initial_v = rng.normal(size=(4, 2, 1, 8)).astype(np.float32)
    storage = (*_numpy_compress(initial_k, constants, keys=True), *_numpy_compress(initial_v, constants, keys=False))
    expected_pages = [page.copy() for page in storage]
    tables = np.array([[2, 0], [3, 1]], dtype=np.int32)
    contexts = np.array([3, 2], dtype=np.int32)
    if update:
        new = (
            *_numpy_compress(key.reshape(4, 1, 8), constants, keys=True),
            *_numpy_compress(value.reshape(4, 1, 8), constants, keys=False),
        )
        destinations, cells = [2, 0, 3, 3], [1, 0, 0, 1]
        for pages, data in zip(expected_pages, new, strict=True):
            pages[destinations, cells] = data
    expected_keys, expected_values = _numpy_decompress(expected_pages, constants)
    sinks = np.array([[0.2], [-0.3]], dtype=np.float32)
    bias = rng.normal(size=(2, 2, 2, 4)).astype(np.float32) * 0.1
    qs = np.array([[0, 1], [2, 2]], dtype=np.int32)
    ks = np.array([[0, 0, 1, -1], [2, 2, -1, -1]], dtype=np.int32)
    pairwise = np.ones((2, 1, 2, 4), dtype=bool)
    pairwise[1, 0, 1, 0] = False
    dropout_key = jax.random.key(24)
    output = np.zeros_like(query)
    for row in range(2):
        # Each request sees its own table and historical prefix; QJL scores
        # include the residual term, not only Lloyd-Max centroid reconstruction.
        k = expected_keys[tables[row]].reshape(4, 1, 8).repeat(2, axis=1)
        v = expected_values[tables[row]].reshape(4, 1, 8).repeat(2, axis=1)
        positions = contexts[row] - 2 + np.arange(2)
        visible = (np.arange(4)[None, :] < contexts[row]) & (np.arange(4)[None, :] <= positions[:, None])
        if combined_mask:
            visible &= (qs[row, :, None] == ks[row, None, :]) & (ks[row, None, :] >= 0) & pairwise[row, 0]
        logits = np.einsum("qhd,khd->hqk", query[row], k) * 0.5 + bias[row]
        logits = np.where(visible[None], logits, -np.inf)
        all_logits = np.concatenate((logits, np.broadcast_to(sinks[:, None], (2, 2, 1))), axis=-1)
        weights = np.exp(all_logits - np.max(all_logits, axis=-1, keepdims=True))
        if normalized:
            weights /= weights.sum(axis=-1, keepdims=True)
        weights = weights[..., :4]
        if dropout:
            # The public paged RNG contract folds each request row into the
            # key and draws against all packed query rows, not only active ones.
            keep = np.asarray(jax.random.bernoulli(jax.random.fold_in(dropout_key, row), 0.75, (1, 2, 4, 4)))
            weights = np.where(keep[0, :, 2 * row : 2 * row + 2], weights / 0.75, 0.0)
        output[row] = np.einsum("hqk,khd->qhd", weights, v)

    view = _compressed_cache(jnp.zeros((), dtype=jnp.bfloat16), [jnp.asarray(page) for page in storage], constants)
    runtime = RaggedPagesMetadata(
        jnp.asarray(tables), jnp.asarray(contexts), jnp.array([0, 2, 4]), jnp.asarray(contexts), page_size=2
    )
    operation = OperationRegistry.create(name, _metadata())
    result = operation(
        query=jnp.asarray(query),
        key=jnp.asarray(key) if update else None,
        value=jnp.asarray(value) if update else None,
        cache_view=view,
        cache_metadata=runtime,
        mask_info=MaskInfo(
            _attention_mask=jnp.asarray(pairwise), _q_segment_ids=jnp.asarray(qs), _kv_segment_ids=jnp.asarray(ks)
        )
        if combined_mask
        else None,
        softmax_scale=0.5,
        softmax_aux=jnp.asarray(sinks),
        bias=jnp.asarray(bias),
        deterministic=not dropout,
        dropout_prob=0.25,
        dropout_rng=dropout_key,
        normalize_output=normalized,
        compute_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    np.testing.assert_allclose(result.attention_outputs, output, rtol=3e-4, atol=3e-5)
    assert isinstance(result.cache_view, TurboQuantRaggedPagesCacheView)
    assert result.cache_view.kv_pages is view.kv_pages
    for name, expected in zip(PAGE_FIELDS, expected_pages, strict=True):
        np.testing.assert_array_equal(getattr(result.cache_view, name), expected)
