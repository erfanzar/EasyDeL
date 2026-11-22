"""Scalar MTP row isolation (trace-only on CPU; numerical checks need an accelerator)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCache, TransformerCacheConfig, TransformerCacheView
from easydel.inference.speculative import _mtp_forward_core, _scatter_view_row, _slice_view_row
from jax import numpy as jnp


def _view(key, indexes, segments):
    config = TransformerCacheConfig.create(
        batch_size=key.shape[0], sequence_length=8, num_hidden_layers=1, num_heads=1, head_dim=1
    )
    return TransformerCacheView(
        key=key,
        value=key,
        indexes=indexes,
        starts=jnp.zeros_like(indexes),
        metadata=config,
        maximum_sequence_length=8,
        segment_ids=segments,
    )


def _scalar_forward(key, indexes, segments, token, row):
    view = _view(key, indexes, segments)

    class Layer:
        layer_idx = 0

        def __call__(self, h, mask, pos, mode, cache, metadata, frequencies):
            kv = h.reshape(1, 1, 1, 1)
            _, _, _, updated, _ = cache.concatenate_to_cache(
                query=kv,
                key=kv,
                value=kv,
                mode=mode,
                quantizer=lambda x: x,
                cache_metadata=metadata,
                mask_info=mask,
            )
            return h, updated

    mtp = SimpleNamespace(
        layers=[Layer()],
        pre_fc_norm_embedding=lambda x: x,
        pre_fc_norm_hidden=lambda x: x,
        fc=lambda x: x[..., :1],
        norm=lambda x: x,
    )
    _, cache = _mtp_forward_core(
        mtp,
        token,
        None,
        jnp.zeros((1, 1), jnp.int32),
        token,
        jnp.zeros((1, 1), jnp.int32),
        TransformerCache(views=[view]),
        row,
    )
    return cache.views[0]


@pytest.mark.cpu_ok
@pytest.mark.parametrize("legacy", [False, True])
def test_scalar_mtp_cache_row_history_traces_with_batch_one_forward(legacy):
    segments = None if legacy else jax.ShapeDtypeStruct((4, 8), jnp.int32)
    result = jax.eval_shape(
        _scalar_forward,
        jax.ShapeDtypeStruct((4, 8, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((4,), jnp.int32),
        segments,
        jax.ShapeDtypeStruct((1, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    assert result.key.shape == result.value.shape == (4, 8, 1, 1)
    assert result.segment_ids.shape == (4, 8)
    assert result.indexes.shape == result.starts.shape == (4,)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("full_history,updated_history", [(False, False), (False, True), (True, False), (True, True)])
def test_slice_scatter_optional_history_shapes(full_history, updated_history):
    def update(key, indexes, history, row):
        view = _view(key, indexes, history if full_history else None)
        selected = _slice_view_row(view, row)
        assert selected.segment_ids is None or selected.segment_ids.shape == (1, 8)
        new_history = jnp.zeros((1, 8), jnp.int32) if updated_history else None
        return _scatter_view_row(view, selected.replace(segment_ids=new_history), row)

    result = jax.eval_shape(
        update,
        jax.ShapeDtypeStruct((4, 8, 1, 1), jnp.float32),
        jax.ShapeDtypeStruct((4,), jnp.int32),
        jax.ShapeDtypeStruct((4, 8), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32),
    )
    if full_history or updated_history:
        assert result.segment_ids.shape == (4, 8)
    else:
        assert result.segment_ids is None


@pytest.mark.parametrize("row", [0, 1, 2, 3])
def test_scatter_history_and_scalar_mtp_preserve_all_other_pool_rows(row):
    key = jnp.arange(32, dtype=jnp.float32).reshape(4, 8, 1, 1)
    indexes = jnp.array([2, 3, 4, 5], jnp.int32)
    segments = jnp.where(jnp.arange(8)[None, :] < indexes[:, None], jnp.arange(4)[:, None] + 10, -1)
    original = _view(key, indexes, segments)
    selected = _slice_view_row(original, jnp.array(row, jnp.int32))
    np.testing.assert_array_equal(selected.segment_ids, np.asarray(segments)[row : row + 1])
    changed = selected.replace(segment_ids=jnp.full((1, 8), 99, jnp.int32))
    scattered = _scatter_view_row(original, changed, jnp.array(row, jnp.int32))
    expected = np.asarray(segments).copy()
    expected[row] = 99
    np.testing.assert_array_equal(scattered.segment_ids, expected)
    forwarded = _scalar_forward(key, indexes, segments, jnp.full((1, 1, 1), 42.0), jnp.array(row, jnp.int32))
    other = [i for i in range(4) if i != row]
    np.testing.assert_array_equal(np.asarray(forwarded.segment_ids)[other], np.asarray(segments)[other])
    np.testing.assert_array_equal(np.asarray(forwarded.key)[other], np.asarray(key)[other])
    np.testing.assert_array_equal(np.asarray(forwarded.indexes)[other], np.asarray(indexes)[other])
    assert int(forwarded.indexes[row]) == int(indexes[row]) + 1
    assert int(forwarded.segment_ids[row, indexes[row]]) == int(segments[row, indexes[row] - 1])


def test_legacy_scatter_promotes_history_without_marking_other_valid_rows_as_padding():
    indexes = jnp.array([2, 3, 4, 0], jnp.int32)
    original = _view(jnp.zeros((4, 8, 1, 1)), indexes, None)
    selected = _slice_view_row(original, jnp.array(1, jnp.int32))
    updated = selected.replace(segment_ids=jnp.full((1, 8), 99, jnp.int32))
    result = _scatter_view_row(original, updated, jnp.array(1, jnp.int32))
    expected = np.where(np.arange(8)[None, :] < np.array([2, 3, 4, 0])[:, None], 0, -1)
    expected[1] = 99
    np.testing.assert_array_equal(result.segment_ids, expected)
