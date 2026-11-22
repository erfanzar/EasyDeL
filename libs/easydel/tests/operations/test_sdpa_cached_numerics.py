"""Independent registered SDPA cached visibility and upcast parity; accelerator only."""

import jax
import numpy as np
import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule
from easydel.operations import OperationMetadata, OperationRegistry
from ejkernel.types import MaskInfo
from jax import numpy as jnp


@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("through_layer", [False, True])
def test_sdpa_cached_absolute_masks_and_float32_override(window, through_layer):
    config = EasyDeLBaseConfig(
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="sdpa",
        attn_dtype=jnp.bfloat16,
        attn_softmax_dtype=jnp.float32,
    )
    metadata = OperationMetadata(runtime_dtype=jnp.bfloat16, runtime_softmax_dtype=jnp.float32, base_config=config)
    operation = (
        FlexibleAttentionModule(base_config=config, softmax_scale=0.5)
        if through_layer
        else OperationRegistry.create("sdpa", metadata)
    )
    rng = np.random.default_rng(513)
    q = rng.normal(size=(1, 2, 1, 4)).astype(np.float32)
    k = rng.normal(size=(1, 10, 1, 4)).astype(np.float32)
    v = rng.normal(size=(1, 10, 1, 4)).astype(np.float32)
    absolute = np.array([8, 9])
    visible = np.arange(10)[None, :] <= absolute[:, None]
    if window is not None:
        visible &= np.arange(10)[None, :] >= absolute[:, None] - window + 1
    mask = MaskInfo(
        _attention_mask=jnp.asarray(visible[None, None]),
        causal_mask_baked_in=True,
        sliding_window_baked_in=window is not None,
    )
    logits = np.einsum("bqhd,bkhd->bhqk", q.astype(np.float64), k.astype(np.float64)) * 0.5
    logits = np.where(visible[None, None], logits, -np.inf)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhqk,bkhd->bqhd", weights, v)
    with jax.default_matmul_precision("highest"):
        result = operation(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            mask_info=mask,
            mode=None,
            causal=True,
            sliding_window=window,
            compute_dtype=jnp.float32,
        )
    np.testing.assert_allclose(result.attention_outputs, expected, rtol=2e-5, atol=2e-6)
    assert result.attention_outputs.dtype == jnp.float32
