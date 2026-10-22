"""Packed vision/MTP regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base import flatten_mrope_position_ids_for_text
from easydel.modules.qwen2_vl import Qwen2VLVisionConfig
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionAttention
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_mtp_outputs import _model


def test_compiled_multisegment_vision_attention_matches_segmented_reference():
    config = Qwen2VLVisionConfig(
        depth=1,
        embed_dim=16,
        num_heads=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    attention = Qwen2VLVisionAttention(
        config,
        layer_idx=0,
        dim=16,
        num_heads=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(1),
    )
    hidden = jnp.arange(6 * 16, dtype=jnp.float32).reshape(6, 16) / 91
    cos, sin = jnp.ones((6, 8)), jnp.zeros((6, 8))
    boundaries = jnp.array([0, 2, 6], dtype=jnp.int32)
    output = jax.jit(lambda h, cu: attention(h, cu, (cos, sin)))(hidden, boundaries)
    # Independent dense attention for each packed image/frame; rotary is identity.
    qkv = np.asarray(attention.qkv(hidden)).reshape(6, 3, 2, 8)
    chunks = []
    for start, end in [(0, 2), (2, 6)]:
        q, k, v = (qkv[start:end, i].transpose(1, 0, 2) for i in range(3))
        scores = q @ k.swapaxes(-1, -2) / np.sqrt(8)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        chunks.append((weights @ v).transpose(1, 0, 2).reshape(end - start, 16))
    reference = attention.proj(jnp.asarray(np.concatenate(chunks)))
    np.testing.assert_allclose(output, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_automatic_mtp_reuses_packed_mask_and_explicit_positions(moe, vision):
    model = _model(moe, vision)
    ids = jnp.array([[1, 2, 3, 4, 5, 6]], dtype=jnp.int32)
    segments = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)
    positions = jnp.array([[2, 3, 4, 7, 8, 9]], dtype=jnp.int32)
    mask = MaskInfo.from_segments(segments)
    output = model(input_ids=ids, mask_info=mask, position_ids=positions)
    manual = model.compute_mtp_outputs(output, ids, mask_info=mask, position_ids=positions)
    manual_logits = model.compute_mtp_logits(manual)
    np.testing.assert_allclose(output.mtp_logits, manual_logits, rtol=2e-5, atol=2e-5)
    # Only t=0 and t=3 have skip-one targets within their packed example.
    logits = np.asarray(output.mtp_logits, dtype=np.float64)[0, [0, 3]]
    reference = (np.logaddexp.reduce(logits, axis=-1) - logits[np.arange(2), [3, 6]]).mean() * 0.3
    np.testing.assert_allclose(output.mtp_loss, reference, rtol=2e-5, atol=2e-5)
    changed = model(input_ids=ids.at[:, 3:].set(jnp.array([[7, 8, 9]])), mask_info=mask, position_ids=positions)
    np.testing.assert_allclose(output.mtp_logits[:, :3], changed.mtp_logits[:, :3], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("moe", [False, True])
def test_automatic_vision_mtp_reuses_inferred_visual_positions(moe):
    model = _model(moe, True)
    ids = jnp.array([[29, 31, 31, 31, 31, 5]], dtype=jnp.int32)
    grid = np.array([[1, 4, 4]], dtype=np.int32)
    image_embeds = jnp.arange(4 * 32, dtype=jnp.float32).reshape(4, 32) / 113
    output = model(input_ids=ids, image_embeds=image_embeds, image_grid_thw=grid)
    positions, _ = model.model.get_rope_index(input_ids=ids, image_grid_thw=grid)
    positions = flatten_mrope_position_ids_for_text(model.config.text_config, positions)
    reference = model.compute_mtp_logits(model.compute_mtp_outputs(output, ids, position_ids=positions))
    np.testing.assert_allclose(output.mtp_logits, reference, rtol=2e-5, atol=2e-5)


def test_multimodal_moe_public_router_logits_and_configured_auxiliary_loss():
    model = _model(True, True)
    ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    output = model(input_ids=ids, output_router_logits=True)
    assert output.router_logits is not None
    assert len(output.router_logits) == 1
    assert float(output.aux_loss - output.mtp_loss) > 0.0
    disabled = model(input_ids=ids, output_router_logits=False)
    assert disabled.router_logits is None
    np.testing.assert_allclose(disabled.aux_loss, disabled.mtp_loss, rtol=2e-5, atol=2e-5)
