"""Independent soft-cap and MLA paged-dropout regressions; accelerator only."""

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesMetadata
from easydel.caching.mla_ragged_page import MLARaggedPagesCacheView
from easydel.layers.attention import FlexibleAttentionModule
from easydel.operations import OperationMetadata, OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_mla_stochastic_contract import _cache_config, _config


@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("dropout", [False, True])
def test_vanilla_zero_softcap_is_disabled(segmented, dropout):
    config = _config()
    operation = OperationRegistry.create("vanilla", OperationMetadata(base_config=config, runtime_dtype=jnp.float32))
    rng = np.random.default_rng(418)
    q, k, v = (rng.normal(size=(1, 3, 2, 4)).astype(np.float32) for _ in range(3))
    ids = np.array([[0, 0, 1]], dtype=np.int32)
    visibility = ids[:, :, None] == ids[:, None, :] if segmented else np.ones((1, 3, 3), dtype=bool)
    logits = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)) * 0.5
    logits = np.where(visibility[:, None], logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    key = jax.random.key(42)
    if dropout:
        weights *= np.asarray(jax.random.bernoulli(key, 0.75, weights.shape)) / 0.75
    expected = np.einsum("bhqk,bkhd->bqhd", weights, v)
    result = operation(
        *map(jnp.asarray, (q, k, v)),
        mask_info=MaskInfo.from_segments(jnp.asarray(ids), jnp.asarray(ids)) if segmented else None,
        logits_soft_cap=0.0,
        softmax_scale=0.5,
        deterministic=not dropout,
        dropout_prob=0.25 if dropout else 0.0,
        dropout_rng=key,
        precision=jax.lax.Precision.HIGHEST,
        compute_dtype=jnp.float32,
    )
    np.testing.assert_allclose(result.attention_outputs, expected, rtol=2e-5, atol=2e-6)
    assert np.isfinite(result.attention_outputs).all()


@pytest.mark.parametrize("through_layer", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("rope_dim", [0, 2])
def test_mla_stochastic_history_and_component_padding_match_numpy(through_layer, compiled, rope_dim):
    config = _config()
    operation = (
        FlexibleAttentionModule(config, softmax_scale=0.5)
        if through_layer
        else OperationRegistry.create(
            "paged_attention_reference", OperationMetadata(base_config=config, runtime_dtype=jnp.float32)
        )
    )
    cache_config = _cache_config(rope_dim=rope_dim)
    rng = np.random.default_rng(532)
    qn = rng.normal(size=(1, 4, 2, 3)).astype(np.float32)
    qp = rng.normal(size=(1, 4, 2, rope_dim)).astype(np.float32)
    kv = rng.normal(size=(1, 4, 3)).astype(np.float32)
    kp = rng.normal(size=(1, 4, rope_dim)).astype(np.float32)
    pages = np.full(cache_config.get_shape_and_axes()[0], 37.0, dtype=np.float32)
    flat = pages.reshape(4, 2, -1)
    flat[2, 0, :3] = rng.normal(size=3)
    flat[2, 0, 128 : 128 + rope_dim] = rng.normal(size=rope_dim)
    expected_pages = pages.copy()
    target = expected_pages.reshape(4, 2, -1)
    for token, (page, offset) in enumerate(((2, 1), (0, 0), (3, 0), (3, 1))):
        target[page, offset, :3] = kv[0, token]
        target[page, offset, 128 : 128 + rope_dim] = kp[0, token]
    key = jax.random.key(83)
    expected = np.zeros_like(qn)
    for row, (start, end, context, table) in enumerate(((0, 2, 3, [2, 0]), (2, 4, 2, [3, 1]))):
        history = target[table].reshape(4, -1)
        dense_k = np.concatenate((history[:, :3], history[:, 128 : 128 + rope_dim]), axis=-1)
        queries = np.concatenate((qn[0], qp[0]), axis=-1).astype(np.float64)
        logits = np.einsum("qhd,kd->hqk", queries, dense_k) * 0.5
        positions = np.arange(4) - start + context - (end - start)
        visible = (np.arange(4)[None, :] < context) & (np.arange(4)[None, :] <= positions[:, None])
        visible &= (np.arange(4)[:, None] >= start) & (np.arange(4)[:, None] < end)
        # Only active rows contribute; compute those independently of padding.
        active_logits = np.where(visible[None, start:end], logits[:, start:end], -np.inf)
        weights = np.exp(active_logits - active_logits.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        keep = np.asarray(jax.random.bernoulli(jax.random.fold_in(key, row), 0.75, (1, 2, 4, 4)))
        weights *= keep[0, :, start:end] / 0.75
        expected[0, start:end] = np.einsum("hqk,kd->qhd", weights, history[:, :3])

    def forward(qn, qp, kv, kp, pages):
        runtime = RaggedPagesMetadata(
            jnp.array([[2, 0], [3, 1]], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            jnp.array([0, 2, 4], dtype=jnp.int32),
            jnp.array([3, 2], dtype=jnp.int32),
            page_size=2,
        )
        common = dict(
            cache_view=MLARaggedPagesCacheView(cache_config, 0, pages),
            cache_metadata=runtime,
            queries_nope=qn,
            queries_pe=qp,
            keys_values=kv,
            keys_pe=kp,
            deterministic=False,
            dropout_prob=0.25,
            dropout_rng=key,
            precision=jax.lax.Precision.HIGHEST,
            compute_dtype=jnp.float32,
            softmax_scale=0.5,
        )
        combined = jnp.concatenate((qn, qp), axis=-1)
        result = operation(combined, None, None, mode=None, **common) if through_layer else operation(combined, **common)
        return result.attention_outputs, result.cache_view.kv_pages

    if compiled:
        forward = jax.jit(forward)
    output, updated = forward(*map(jnp.asarray, (qn, qp, kv, kp, pages)))
    np.testing.assert_allclose(output, expected, rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(updated, expected_pages)
