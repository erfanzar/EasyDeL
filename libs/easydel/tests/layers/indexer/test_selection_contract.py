# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

"""Independent output/gradient tests for the common indexer selection contract."""

import jax
import numpy as np
import pytest
from easydel.layers.indexer import BaseIndexer, IndexerSelection, SelectionSpec
from jax import numpy as jnp


@pytest.mark.parametrize("leading", [(), (2,), (2, 3), (2, 4, 3)])
def test_selection_scatter_matches_set_reference(leading):
    rng = np.random.default_rng(11)
    indices = rng.integers(-2, 9, (*leading, 6), dtype=np.int32)
    expected = np.zeros((*leading, 7), dtype=bool)
    for row in np.ndindex(leading):
        for value in indices[row]:
            if 0 <= value < 7:
                expected[(*row, value)] = True
    actual = jax.jit(lambda x: IndexerSelection(x).to_mask(7))(jnp.asarray(indices))
    np.testing.assert_array_equal(actual, expected)


def test_supplied_mask_is_used_only_for_its_own_domain():
    indices = jnp.asarray([[2, -1]])
    supplied = jnp.asarray([[False, False, True, False]])
    marker = jnp.asarray([[True, False, False, False]])  # deliberately not the indices
    assert np.array_equal(IndexerSelection(indices, mask=marker).to_mask(4), marker)
    # A different domain width ignores the mask and scatters the indices.
    np.testing.assert_array_equal(IndexerSelection(indices, mask=marker).to_mask(5), [[0, 0, 1, 0, 0]])
    np.testing.assert_array_equal(
        IndexerSelection(indices, mask=supplied).to_bias(4, mask_value=-1.0), [[-1, -1, 0, -1]]
    )


@pytest.mark.parametrize("size", [0, 4])
def test_empty_selection_and_empty_domain(size):
    result = IndexerSelection(jnp.empty((2, 3, 0), dtype=jnp.int32)).to_mask(size)
    assert result.shape == (2, 3, size)
    assert not np.asarray(result).any()
    result = IndexerSelection(jnp.asarray([[0, -1, 100]])).to_mask(0)
    assert result.shape == (1, 0)


def test_grouped_ranking_validity_and_local_budget():
    scores = jnp.asarray([[[9.0, 2.0, jnp.inf, 5.0], [0.0, 8.0, 7.0, -jnp.inf]]])
    valid = jnp.asarray([[[False, True, True, True], [True, True, True, False]]])
    result = jax.jit(lambda s, v: BaseIndexer.select_candidates(s, 2, v))(scores, valid)
    # An invalid high scorer cannot consume a slot; +inf local blocks do.
    np.testing.assert_array_equal(result.indices, [[[2, 3], [1, 2]]])
    np.testing.assert_array_equal(result.to_mask(4), [[[False, False, True, True], [False, True, True, False]]])


def test_invalid_scores_and_clamped_budget():
    result = BaseIndexer.select_candidates(jnp.asarray([[jnp.nan, -jnp.inf, 1.0]]), 7)
    np.testing.assert_array_equal(result.indices, [[2, -1, -1]])
    assert BaseIndexer.select_candidates(jnp.empty((2, 0)), 3).indices.shape == (2, 0)
    assert BaseIndexer.select_candidates(jnp.ones((2, 3)), 0).indices.shape == (2, 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_bias_forward_and_selected_score_gradient(dtype):
    indices = jnp.asarray([[[2, -1, 0], [-1, -1, -1]]])
    scores = jnp.asarray([[[1.2, -jnp.inf, -0.7, 3.0], [-jnp.inf] * 4]], dtype)

    def bias(s):
        return IndexerSelection(indices, s).to_bias(4, dtype, mask_value=-1000.0)

    expected = np.asarray([[[0.0, -1000.0, 0.0, -1000.0], [-1000.0] * 4]])
    np.testing.assert_array_equal(jax.jit(bias)(scores).astype(jnp.float32), expected)
    grad = jax.grad(lambda s: bias(s).astype(jnp.float32).sum())(scores)
    np.testing.assert_array_equal(grad.astype(jnp.float32), [[[1, 0, 1, 0], [0, 0, 0, 0]]])


def test_selection_domains_are_explicit_and_validated():
    spec = SelectionSpec("block", "token", grouped=True)
    assert spec.grouped and spec.output_unit == "token"
    with pytest.raises(ValueError, match="Compressed entries"):
        SelectionSpec("compressed_entry", "token")
    with pytest.raises(ValueError, match="score_proxy"):
        IndexerSelection(jnp.asarray([[0]]), jnp.ones((1, 2))).to_bias(3)
    with pytest.raises(ValueError, match="non-negative"):
        BaseIndexer.select_candidates(jnp.ones((2, 3)), -1)
