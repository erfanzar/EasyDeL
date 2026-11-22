"""Abstract-only fixed-shape auxiliary-cache and bidirectional-mask contracts."""

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCache, TransformerCacheConfig, TransformerCacheView, TransformerMetadata
from easydel.caching.transformer.cache import AttnMaskDetail
from easydel.infra.utils import AttnMaskType
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from jax.sharding import Mesh
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


def _config(auxiliary_dim=None):
    return TransformerCacheConfig.create(
        batch_size=1, sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=2, auxiliary_dim=auxiliary_dim
    )


@pytest.mark.parametrize("dimension", [0, -1])
def test_auxiliary_config_rejects_nonpositive_width(dimension):
    with pytest.raises(ValueError, match="auxiliary_dim must be positive"):
        _config(dimension)


@pytest.mark.parametrize("dimension", [None, 4, 9])
@pytest.mark.parametrize("sliding", [False, True])
def test_cache_allocation_preallocates_optional_auxiliary_pytree(dimension, sliding):
    config = _config(dimension)
    mesh = Mesh(np.array(jax.devices()[:1]), ("tp",))
    details = {0: AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=3)} if sliding else None
    cache = jax.eval_shape(
        lambda: TransformerCache.init_cache(mesh=mesh, config=config, dtype=jnp.float32, mask_type_details=details)
    )
    view = cache.views[0]
    assert view.key.shape == (1, 3 if sliding else 8, 1, 2)
    if dimension is None:
        assert view.read_auxiliary() is None
    else:
        assert view.read_auxiliary().shape == (1, 3 if sliding else 8, dimension)
        assert view.read_auxiliary().dtype == jnp.float32
        leaves, structure = jax.tree_util.tree_flatten(cache)
        restored = jax.tree_util.tree_unflatten(structure, leaves)
        assert restored.views[0].read_auxiliary().shape == view.read_auxiliary().shape
        reset = jax.eval_shape(lambda x: x.reset(), view)
        assert reset.read_auxiliary().shape == view.read_auxiliary().shape


@pytest.mark.parametrize("sliding", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_size", [1, 4])
def test_auxiliary_then_kv_append_preserves_compiled_state_shapes(sliding, causal, block_size):
    capacity = 3 if sliding else 8
    config = _config(9)

    def append(old, auxiliary, tokens, features, indexes, segments):
        view = TransformerCacheView(
            key=old,
            value=old,
            indexes=indexes,
            starts=jnp.zeros_like(indexes),
            metadata=config,
            maximum_sequence_length=8,
            segment_ids=segments,
            auxiliary_state=auxiliary,
            masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=capacity) if sliding else None,
        )
        original_mask = MaskInfo.from_attention_mask(jnp.ones((1, 1, block_size, block_size), jnp.bool_))
        metadata = TransformerMetadata(postpadded=True, indexes=indexes)
        scored, selection_mask, pending = view.update_auxiliary(features, original_mask, metadata, causal=causal)
        key, _, mask, updated, _ = pending.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=metadata,
            mask_info=original_mask,
            causal=causal,
        )
        return scored, selection_mask.attention_mask, key, mask.attention_mask, updated

    shape = jax.ShapeDtypeStruct
    scored, selection_mask, key, kv_mask, updated = jax.eval_shape(
        append,
        shape((1, capacity, 1, 2), jnp.float32),
        shape((1, capacity, 9), jnp.float32),
        shape((1, block_size, 1, 2), jnp.float32),
        shape((1, block_size, 9), jnp.float32),
        shape((1,), jnp.int32),
        shape((1, capacity), jnp.int32),
    )
    attention_width = capacity + block_size if sliding and block_size > 1 else capacity
    assert scored.shape == (1, attention_width, 9)
    assert key.shape == (1, attention_width, 1, 2)
    assert selection_mask.shape == kv_mask.shape == (1, 1, block_size, attention_width)
    assert updated.auxiliary_state.shape == (1, capacity, 9)
    assert updated.segment_ids.shape == (1, capacity)


@pytest.mark.parametrize("postpadded", [False, True])
@pytest.mark.parametrize("block_size", [1, 2])
def test_lazy_segment_mask_stays_lazy_through_auxiliary_then_kv_tracing(postpadded, block_size):
    def append(old, auxiliary, tokens, features, indexes, segments):
        view = TransformerCacheView(
            key=old,
            value=old,
            indexes=indexes,
            starts=jnp.zeros_like(indexes),
            metadata=_config(3),
            maximum_sequence_length=8,
            segment_ids=segments,
            auxiliary_state=auxiliary,
        )
        original_mask = MaskInfo.from_segments(jnp.zeros((1, block_size), jnp.int32))
        metadata = TransformerMetadata(postpadded=True) if postpadded else None
        _, selection_mask, pending = view.update_auxiliary(features, original_mask, metadata)
        assert original_mask._attention_mask is None
        _, _, attention_mask, updated, _ = pending.concatenate_to_cache(
            query=tokens,
            key=tokens,
            value=tokens,
            mode=common_types.MODE_PREFILL,
            quantizer=lambda x: x,
            cache_metadata=metadata,
            mask_info=original_mask,
        )
        assert original_mask._attention_mask is None
        return selection_mask, attention_mask, updated

    shape = jax.ShapeDtypeStruct
    selection_mask, attention_mask, updated = jax.eval_shape(
        append,
        shape((1, 8, 1, 2), jnp.float32),
        shape((1, 8, 3), jnp.float32),
        shape((1, block_size, 1, 2), jnp.float32),
        shape((1, block_size, 3), jnp.float32),
        shape((1,), jnp.int32),
        shape((1, 8), jnp.int32),
    )
    assert selection_mask.attention_mask.shape == attention_mask.attention_mask.shape == (1, 1, block_size, 8)
    assert selection_mask.q_segment_ids.shape == attention_mask.q_segment_ids.shape == (1, block_size)
    assert updated.segment_ids.shape == (1, 8)


def test_auxiliary_append_requires_preallocated_storage():
    view = TransformerCacheView(
        key=jax.ShapeDtypeStruct((1, 8, 1, 2), jnp.float32),
        value=None,
        indexes=jax.ShapeDtypeStruct((1,), jnp.int32),
        starts=jax.ShapeDtypeStruct((1,), jnp.int32),
        metadata=_config(),
        maximum_sequence_length=8,
    )
    with pytest.raises(ValueError, match="Request auxiliary_dim"):
        view.update_auxiliary(None, None)
