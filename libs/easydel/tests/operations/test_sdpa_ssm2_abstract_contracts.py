"""Abstract public operation/layer output contracts, without array computation."""

from types import SimpleNamespace

import jax
import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLBackends
from easydel.layers.attention import FlexibleAttentionModule
from easydel.operations import OperationMetadata, OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("through_layer", [False, True])
def test_sdpa_upcast_public_output_dtype_and_cached_mask_layout(through_layer):
    # This is a single-request contract, not a batch-sharding test. Keep its
    # mesh independent of the host's device count (four on the TPU runner).
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1), attn_mechanism="sdpa", attn_dtype=jnp.bfloat16)
    metadata = OperationMetadata(runtime_dtype=jnp.bfloat16, base_config=config)
    operation = (
        FlexibleAttentionModule(config, softmax_scale=0.5)
        if through_layer
        else OperationRegistry.create("sdpa", metadata)
    )
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in [(1, 2, 1, 4), (1, 10, 1, 4), (1, 10, 1, 4)]]

    def forward(q, k, v):
        mask = MaskInfo(
            _attention_mask=jnp.ones((1, 1, 2, 10), dtype=jnp.bool_),
            causal_mask_baked_in=True,
            sliding_window_baked_in=True,
        )
        return operation(
            q, k, v, mode=None, mask_info=mask, causal=True, sliding_window=3, compute_dtype=jnp.float32
        ).attention_outputs

    result = jax.eval_shape(forward, *descriptors)
    assert (result.shape, result.dtype) == ((1, 2, 1, 4), jnp.float32)


@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("gated", [False, True])
def test_registered_segmented_ssm2_output_and_state_layout(length, gated):
    operation = OperationRegistry.create("ssm2", SimpleNamespace(backend=EasyDeLBackends.CPU))
    x = jax.ShapeDtypeStruct((2, length, 2, 3), jnp.float32)
    state = jax.ShapeDtypeStruct((2, 2, 3, 4), jnp.float32)
    ids = jax.ShapeDtypeStruct((2, length), jnp.int32)

    def forward(x, state, ids):
        result = operation(
            x,
            jnp.zeros((2,)),
            jnp.ones((2, length, 1, 4)),
            jnp.ones((2, length, 1, 4)),
            jnp.ones((2,)),
            jnp.ones((2, length, 2)),
            ssm_state=state,
            segment_ids=ids,
            gate=jnp.ones((2, length, 6)) if gated else None,
            use_gated_rmsnorm=gated,
        )
        return result.attention_outputs, result.ssm_state

    output, final_state = jax.eval_shape(forward, x, state, ids)
    assert (output.shape, output.dtype) == (x.shape, jnp.float32)
    assert (final_state.shape, final_state.dtype) == (state.shape, jnp.float32)
