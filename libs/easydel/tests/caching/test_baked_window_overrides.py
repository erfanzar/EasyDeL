"""Explicit local windows must intersect caller-baked masks without a sliding cache."""

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCacheConfig, TransformerCacheView
from easydel.caching.transformer.cache import AttnMaskDetail
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.utils import AttnMaskType
from easydel.layers.attention import AttentionModule
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def _concatenate_baked_mask(cache_kind, window):
    attention = AttentionModule.__new__(AttentionModule)
    attention.config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    attention.dtype = jnp.float32
    config = TransformerCacheConfig.create(batch_size=4, sequence_length=3, num_hidden_layers=1, num_heads=1, head_dim=1)

    def concatenate(tokens):
        segments = jnp.zeros((4, 3), jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), (4, 3))
        # Every pair is visible under the original size-3 window. The baked
        # marker does not identify a later caller's narrower explicit window.
        mask = MaskInfo(
            _attention_mask=jnp.ones((4, 1, 3, 3), jnp.bool_),
            _q_segment_ids=segments,
            _kv_segment_ids=segments,
            q_positions=positions,
            kv_positions=positions,
            sliding_window_baked_in=True,
        )
        view = None
        if cache_kind != "none":
            view = TransformerCacheView(
                key=jnp.zeros_like(tokens),
                value=jnp.zeros_like(tokens),
                indexes=jnp.zeros((4,), jnp.int32),
                starts=jnp.zeros((4,), jnp.int32),
                metadata=config,
                maximum_sequence_length=3,
                segment_ids=segments,
                masking_details=AttnMaskDetail(mask_type=AttnMaskType.FULL, size=3) if cache_kind == "full" else None,
            )
        key, value, mask, _, updated, metadata = attention.concatenate(
            query=tokens,
            key=tokens,
            value=tokens,
            mask_info=mask,
            mode=common_types.MODE_PREFILL,
            cache_view=view,
            sliding_window=window,
            causal=False,
        )
        return key, value, mask, updated, metadata

    return concatenate


@pytest.mark.cpu_ok
@pytest.mark.parametrize("cache_kind", ["none", "dense", "full"])
@pytest.mark.parametrize("window", [1, 2, (0, 0), (1, 0)])
def test_baked_explicit_window_override_shapes_abstract(cache_kind, window):
    key, value, mask, updated, metadata = jax.eval_shape(
        _concatenate_baked_mask(cache_kind, window), jax.ShapeDtypeStruct((4, 3, 1, 1), jnp.float32)
    )
    assert key.shape == value.shape == (4, 3, 1, 1)
    assert mask.attention_mask.shape == (4, 1, 3, 3)
    assert mask.q_positions.shape == mask.kv_positions.shape == (4, 3)
    if cache_kind == "none":
        assert updated is None and metadata is None
    else:
        assert updated.key.shape == updated.value.shape == (4, 3, 1, 1)
        assert updated.indexes.shape == metadata.indexes.shape == (4,)


@pytest.mark.parametrize("cache_kind", ["none", "dense", "full"])
@pytest.mark.parametrize("window", [1, 2, (0, 0), (1, 0)])
def test_baked_explicit_window_override_matches_visibility_and_softmax(cache_kind, window):
    tokens = np.broadcast_to(np.array([1, 2, 4], np.float32)[None, :, None, None], (4, 3, 1, 1))
    key, value, mask, updated, metadata = jax.jit(_concatenate_baked_mask(cache_kind, window))(jnp.asarray(tokens))
    left, right = (window - 1,) * 2 if isinstance(window, int) else window
    positions = np.arange(3)
    relative = positions[None, :] - positions[:, None]
    expected = (relative >= -left) & (relative <= right)
    np.testing.assert_array_equal(mask.attention_mask, np.broadcast_to(expected, (4, 1, 3, 3)))
    np.testing.assert_array_equal(mask.q_positions, np.broadcast_to(positions, (4, 3)))
    np.testing.assert_array_equal(mask.kv_positions, np.broadcast_to(positions, (4, 3)))
    np.testing.assert_array_equal(key, tokens)
    np.testing.assert_array_equal(value, tokens)
    if cache_kind != "none":
        np.testing.assert_array_equal(updated.indexes, [3] * 4)
        np.testing.assert_array_equal(metadata.indexes, [3] * 4)

    scores = tokens[0, :, 0, 0, None] * tokens[0, None, :, 0, 0] * 0.25
    reference_weights = np.exp(np.where(expected, scores, -np.inf) - scores.max(axis=-1, keepdims=True))
    reference_weights /= reference_weights.sum(axis=-1, keepdims=True)
    reference = reference_weights @ tokens[0, :, 0, 0]
    logits = jnp.einsum("bqhd,bkhd->bhqk", jnp.asarray(tokens) * 0.25, key, precision=jax.lax.Precision.HIGHEST)
    probabilities = jax.nn.softmax(jnp.where(mask.attention_mask, logits.astype(jnp.float32), -jnp.inf), axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value, precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(output)[..., 0, 0], np.broadcast_to(reference, (4, 3)), rtol=2e-6, atol=2e-6)
