"""Public convolution layout API and independent interleaving references."""

from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest
from ejkernel.modules import operations
from ejkernel.modules.operations import reorder_conv_features_for_sharding
from ejkernel.modules.operations.ragged_causal_conv1d import _reorder_concatenated_tensor_for_sharding
from jax import numpy as jnp


@pytest.mark.cpu_ok
def test_public_reorder_export_preserves_legacy_signature():
    """The supported export retains old keyword names and the private alias."""
    assert "reorder_conv_features_for_sharding" in operations.__all__
    assert reorder_conv_features_for_sharding is _reorder_concatenated_tensor_for_sharding
    parameters = inspect.signature(reorder_conv_features_for_sharding).parameters
    assert tuple(parameters) == ("concatenated_tensor", "split_sizes", "n_shards", "dim")
    assert all(parameter.default is inspect.Parameter.empty for parameter in parameters.values())


@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("n_shards", [1, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.int32])
def test_reorder_matches_independent_group_chunk_reference(dim, n_shards, dtype):
    """Interleaving preserves shape/dtype on feature, token and kernel axes."""
    shape = (12, 3) if dim in (0, -2) else (3, 12)
    host = np.arange(36).reshape(shape)
    groups = np.split(host, [4, 6], axis=dim)
    chunks = [np.split(group, n_shards, axis=dim) for group in groups]
    expected = np.concatenate([group[shard] for shard in range(n_shards) for group in chunks], axis=dim)
    result = reorder_conv_features_for_sharding(
        concatenated_tensor=jnp.asarray(host, dtype=dtype), split_sizes=(4, 2, 6), n_shards=n_shards, dim=dim
    )
    assert result.shape == shape
    assert result.dtype == jnp.dtype(dtype)
    np.testing.assert_array_equal(np.asarray(jax.device_get(result)), expected)
