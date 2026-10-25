"""SigLIP supervised-loss regressions; numeric tests require an accelerator."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.siglip.configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from easydel.modules.siglip.modeling_siglip import (
    MultiheadAttention,
    SiglipForImageClassification,
    SiglipImageClassifierOutput,
    SiglipTextModel,
    SiglipTextModelOutput,
    SiglipVisionEmbeddings,
    SiglipVisionModel,
    SiglipVisionModelOutput,
)
from jax import numpy as jnp


@pytest.mark.cpu_ok
def test_public_embedding_output_containers_remain_available():
    vision = SiglipVisionModelOutput(image_embeds="image", last_hidden_state="patches")
    text = SiglipTextModelOutput(text_embeds="text", last_hidden_state="tokens")
    assert vision["image_embeds"] == "image"
    assert text["text_embeds"] == "text"
    assert vision.last_hidden_state == "patches"
    assert text.last_hidden_state == "tokens"


@pytest.mark.cpu_ok
def test_classifier_output_preserves_supervised_fields_and_optional_loss():
    output = SiglipImageClassifierOutput(logits="scores", loss="loss", hidden_states=("features",))
    assert output.logits == "scores"
    assert output["loss"] == "loss"
    assert output.hidden_states == ("features",)
    unsupervised = SiglipImageClassifierOutput(logits="scores")
    assert unsupervised.logits == "scores"
    assert unsupervised.loss is None


@pytest.mark.parametrize(
    "problem_type,num_labels", [("single_label_classification", 3), ("multi_label_classification", 3), ("regression", 1)]
)
def test_image_classification_returns_supervised_loss(problem_type, num_labels):
    config = SiglipConfig(
        text_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 8,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 4,
            "patch_size": 2,
        },
        num_labels=num_labels,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = SiglipForImageClassification(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    pixels = jnp.linspace(-1, 1, 2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    if problem_type == "single_label_classification":
        labels = jnp.array([0, 2], dtype=jnp.int32)
    elif problem_type == "multi_label_classification":
        labels = jnp.array([[1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
    else:
        labels = jnp.array([0.5, -0.5], dtype=jnp.float32)
    # Leave problem_type unset to test the documented dtype/num_labels inference.
    result = model(pixel_values=pixels, labels=labels)
    logits = np.asarray(result.logits)
    targets = np.asarray(labels)
    if problem_type == "single_label_classification":
        log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
        expected = -log_probs[np.arange(2), targets].mean()
    elif problem_type == "multi_label_classification":
        expected = (np.logaddexp(0, logits) - logits * targets).mean()
    else:
        expected = np.square(logits[:, 0] - targets).mean()
    assert result.logits.shape == (2, num_labels)
    np.testing.assert_allclose(np.asarray(result.loss), expected, rtol=1e-5, atol=1e-6)
    assert model(pixel_values=pixels).loss is None
    np.testing.assert_array_equal(np.asarray(model(pixel_values=pixels).logits), logits)


def test_towers_expose_documented_embedding_output_bundles_without_changing_defaults():
    common = dict(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    text = SiglipTextModel(
        SiglipTextConfig(vocab_size=16, max_position_embeddings=8, **common),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    vision = SiglipVisionModel(
        SiglipVisionConfig(image_size=4, patch_size=2, **common),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(1),
    )
    text.eval()
    vision.eval()
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    pixels = jnp.ones((1, 3, 4, 4), jnp.float32)
    raw_text = text(input_ids=ids)
    raw_vision = vision(pixel_values=pixels)
    text_embeds = text(input_ids=ids, return_embeds=True)
    image_embeds = vision(pixel_values=pixels, return_embeds=True)
    assert isinstance(text_embeds, SiglipTextModelOutput)
    assert isinstance(image_embeds, SiglipVisionModelOutput)
    np.testing.assert_array_equal(np.asarray(text_embeds.text_embeds), np.asarray(raw_text.pooler_output))
    np.testing.assert_array_equal(np.asarray(text_embeds.last_hidden_state), np.asarray(raw_text[0]))
    np.testing.assert_array_equal(np.asarray(image_embeds.last_hidden_state), np.asarray(raw_vision[0]))
    expected_image = np.asarray(raw_vision.pooler_output)
    expected_image = expected_image / np.linalg.norm(expected_image, axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(image_embeds.image_embeds), expected_image, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype,param_dtype", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16)])
@pytest.mark.parametrize("interpolate", [False, True])
def test_patch_embeddings_preserve_separate_compute_and_storage_dtypes(dtype, param_dtype, interpolate):
    config = SiglipVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=4,
        patch_size=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    layer = SiglipVisionEmbeddings(config, dtype=dtype, param_dtype=param_dtype, rngs=spx.Rngs(0))
    assert layer.patch_embedding.weight.value.dtype == param_dtype
    assert layer.patch_embedding.bias.value.dtype == param_dtype
    assert layer.position_embedding.weight.value.dtype == param_dtype
    size = 6 if interpolate else 4
    output = layer(jnp.ones((1, 3, size, size), jnp.float32), interpolate_pos_encoding=interpolate)
    assert output.dtype == dtype
    assert output.shape == (1, (size // 2) ** 2, 8)


def test_pooling_attention_matches_independent_fp32_softmax_with_bfloat16_values():
    layer = MultiheadAttention(
        8,
        2,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    identity = jnp.eye(8, dtype=jnp.bfloat16)
    layer.in_proj_weight.value = jnp.concatenate([identity] * 3, axis=-1)
    layer.in_proj_bias.value = jnp.zeros((24,), jnp.bfloat16)
    layer.out_proj.weight.value = identity
    layer.out_proj.bias.value = jnp.zeros((8,), jnp.bfloat16)
    query = jnp.array([[[1.0, -0.5, 2.0, 0.25, -1.0, 0.5, 0.25, 2.0]]], jnp.bfloat16)
    key = jnp.linspace(-2, 2, 5 * 8).reshape(1, 5, 8).astype(jnp.bfloat16)
    value = jnp.cos(jnp.arange(5 * 8, dtype=jnp.float32)).reshape(1, 5, 8).astype(jnp.bfloat16)
    q = np.asarray(query, dtype=np.float32).reshape(1, 1, 2, 4)
    k = np.asarray(key, dtype=np.float32).reshape(1, 5, 2, 4)
    v = np.asarray(value, dtype=np.float32).reshape(1, 5, 2, 4)
    scores = np.einsum("bqhd,bkhd->bhqk", q * 0.5, k)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    # The attention contract accumulates softmax in fp32, then rounds only the
    # probabilities for the value contraction, and finally the bf16 output.
    rounded = np.asarray(jnp.asarray(probabilities, dtype=jnp.bfloat16), dtype=np.float32)
    expected = jnp.asarray(np.einsum("bhqk,bkhd->bqhd", rounded, v).reshape(1, 1, 8), dtype=jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(layer(query, key, value)), np.asarray(expected))
