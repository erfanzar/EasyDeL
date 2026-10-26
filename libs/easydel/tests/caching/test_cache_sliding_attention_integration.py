"""Cache-owned sliding coordinates stay aligned through outer attention preparation."""

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCacheConfig, TransformerCacheView, TransformerMetadata
from easydel.caching.transformer.cache import AttnMaskDetail
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.utils import AttnMaskType
from easydel.layers.attention import AttentionModule
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def _append(causal, extra_window, postpadded):
    attention = AttentionModule.__new__(AttentionModule)
    attention.config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1))
    attention.dtype = jnp.float32
    config = TransformerCacheConfig.create(
        batch_size=4, sequence_length=16, num_hidden_layers=1, num_heads=1, head_dim=1, auxiliary_dim=1
    )

    def append(history, tokens):
        # Four persisted keys correspond to absolute positions 3..6, not 0..3.
        view = TransformerCacheView(
            key=history,
            value=history,
            indexes=jnp.full((4,), 7, jnp.int32),
            starts=jnp.zeros((4,), jnp.int32),
            metadata=config,
            maximum_sequence_length=16,
            auxiliary_state=history[..., 0],
            segment_ids=jnp.zeros((4, 4), jnp.int32),
            masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=4),
        )
        mask = MaskInfo.from_segments(jnp.zeros((4, 3), jnp.int32))
        metadata = TransformerMetadata(postpadded=postpadded)
        features, selection_mask, pending = view.update_auxiliary(tokens[..., 0], mask, metadata, causal=causal)
        key, value, result_mask, _, updated, _ = attention.concatenate(
            query=tokens,
            key=tokens,
            value=tokens,
            mask_info=mask,
            mode=common_types.MODE_DECODE,
            cache_view=pending,
            cache_metadata=metadata,
            sliding_window=extra_window,
            causal=causal,
        )
        return key, value, result_mask, updated, features, selection_mask

    return append


@pytest.mark.cpu_ok
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("extra_window", [None, 2, (1, 0)])
@pytest.mark.parametrize("postpadded", [False, True])
def test_outer_attention_keeps_auxiliary_union_shapes_and_coordinates_abstract(causal, extra_window, postpadded):
    key, value, mask, updated, features, selection_mask = jax.eval_shape(
        _append(causal, extra_window, postpadded),
        jax.ShapeDtypeStruct((4, 4, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((4, 3, 1, 1), jnp.float32),
    )
    assert key.shape == value.shape == (4, 7, 1, 1)
    assert features.shape == (4, 7, 1)
    assert mask.attention_mask.shape == selection_mask.attention_mask.shape == (4, 1, 3, 7)
    assert mask.q_positions.shape == selection_mask.q_positions.shape == (4, 3)
    assert mask.kv_positions.shape == selection_mask.kv_positions.shape == (4, 7)
    assert updated.key.shape == (4, 4, 1, 1)
    assert updated.auxiliary_state.shape == (4, 4, 1)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("extra_window", [None, 2, (1, 0)])
@pytest.mark.parametrize("postpadded", [False, True])
def test_outer_attention_matches_absolute_sliding_reference_with_auxiliary(causal, extra_window, postpadded):
    history = np.broadcast_to(np.arange(3, 7, dtype=np.float32)[None, :, None, None], (4, 4, 1, 1))
    tokens = np.broadcast_to(np.arange(7, 10, dtype=np.float32)[None, :, None, None], (4, 3, 1, 1))
    key, value, mask, updated, features, selection_mask = jax.jit(_append(causal, extra_window, postpadded))(
        jnp.asarray(history), jnp.asarray(tokens)
    )
    key_positions = np.arange(3, 10)
    query_positions = np.arange(7, 10)
    relative = key_positions[None, :] - query_positions[:, None]
    expected = (relative >= -3) & (relative <= (0 if causal else 3))
    np.testing.assert_array_equal(selection_mask.attention_mask, np.broadcast_to(expected, (4, 1, 3, 7)))
    if extra_window is not None:
        left, right = (extra_window - 1,) * 2 if isinstance(extra_window, int) else extra_window
        expected &= (relative >= -left) & (relative <= right)
    np.testing.assert_array_equal(mask.attention_mask, np.broadcast_to(expected, (4, 1, 3, 7)))
    np.testing.assert_array_equal(mask.q_positions, np.broadcast_to(query_positions, (4, 3)))
    np.testing.assert_array_equal(mask.kv_positions, np.broadcast_to(key_positions, (4, 7)))
    np.testing.assert_array_equal(key[..., 0], features)
    np.testing.assert_array_equal(updated.key[..., 0], updated.auxiliary_state)
    np.testing.assert_array_equal(updated.indexes, [10] * 4)
    # Independent NumPy softmax over actual absolute positions, not production masks.
    scores = tokens[0, :, 0, 0, None] * key_positions[None, :] * 0.05
    reference_weights = np.exp(np.where(expected, scores, -np.inf) - scores.max(axis=-1, keepdims=True))
    reference_weights /= reference_weights.sum(axis=-1, keepdims=True)
    reference = reference_weights @ key_positions
    logits = jnp.einsum("bqhd,bkhd->bhqk", jnp.asarray(tokens) * 0.05, key, precision=jax.lax.Precision.HIGHEST)
    probabilities = jax.nn.softmax(jnp.where(mask.attention_mask, logits.astype(jnp.float32), -jnp.inf), axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", probabilities, value, precision=jax.lax.Precision.HIGHEST)
    np.testing.assert_allclose(np.asarray(output)[..., 0, 0], np.broadcast_to(reference, (4, 3)), rtol=2e-6, atol=2e-6)
