"""Segment-only vanilla attention keeps public masking/bias behavior."""

import jax
import numpy as np
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.operations import OperationMetadata, OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def test_segmented_vanilla_matches_reference_without_materializing_init_bias():
    batch = jax.device_count()
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    metadata = OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, base_config=config)
    op = OperationRegistry.create("vanilla", metadata)
    query = jnp.zeros((batch, 3, 1, 2))
    values = jnp.broadcast_to(jnp.array([2.0, 4.0, 8.0])[None, :, None, None], (batch, 3, 1, 2))
    segments = jnp.broadcast_to(jnp.array([0, 0, 1])[None, :], (batch, 3))
    mask = MaskInfo.from_segments(q_segment_ids=segments, kv_segment_ids=segments)

    def init_bias():
        raise AssertionError("Segment-only masks must not materialize dense bias")

    result = op(query, query, values, mask_info=mask, init_bias=init_bias)
    expected = np.broadcast_to(np.array([3.0, 3.0, 8.0])[None, :, None, None], (batch, 3, 1, 2))
    np.testing.assert_allclose(np.asarray(result.attention_outputs), expected, rtol=1e-5)

    bias = jnp.broadcast_to(jnp.array([0.0, np.log(3.0), 0.0])[None, None, None, :], (batch, 1, 3, 3))
    biased = op(query, query, values, mask_info=mask, bias=bias, init_bias=init_bias)
    expected_biased = np.broadcast_to(np.array([3.5, 3.5, 8.0])[None, :, None, None], (batch, 3, 1, 2))
    np.testing.assert_allclose(np.asarray(biased.attention_outputs), expected_biased, rtol=1e-5)
