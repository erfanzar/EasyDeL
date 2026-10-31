"""Pixtral runtime image geometry and dtype/output contracts (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.pixtral import PixtralVisionConfig, PixtralVisionModel, PixtralVisionModelOutput
from jax import numpy as jnp


def vision_model(dtype=jnp.float32, num_hidden_layers=1, param_dtype=jnp.float32, **config_kwargs):
    config = PixtralVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        head_dim=8,
        image_size=24,
        patch_size=8,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **config_kwargs,
    )
    config.set_model_mesh(
        jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    )
    model = PixtralVisionModel(
        config, dtype=dtype, param_dtype=param_dtype, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(3)
    )
    model.eval()
    return model


def test_vision_patch_weights_honor_parameter_storage_dtype():
    model = vision_model(dtype=jnp.bfloat16)
    assert model.patch_conv.weight.value.dtype == jnp.float32
    pixels = jnp.ones((1, 3, 8, 8), jnp.bfloat16)
    outputs = model(pixel_values=pixels)
    assert outputs.last_hidden_state.dtype == jnp.bfloat16
    compiled = jax.jit(lambda p: model(pixel_values=p))(pixels)
    assert compiled.last_hidden_state.dtype == jnp.bfloat16
    assert model.patch_conv.weight.value.dtype == jnp.float32


@pytest.mark.parametrize("dtype,param_dtype", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16)])
def test_patch_embedding_computes_in_activation_dtype_without_changing_storage(dtype, param_dtype):
    model = vision_model(dtype=dtype, param_dtype=param_dtype)
    embedding = model.get_embedding()
    # Exactly representable operands give an independent patch sum reference.
    weights = np.full(embedding.weight.value.shape, 0.125, dtype=np.float32)
    embedding.weight.value = jnp.asarray(weights, dtype=param_dtype)
    pixels = np.arange(8 * 8 * 3, dtype=np.float32).reshape(1, 8, 8, 3) % 4
    expected = np.full((1, 1, 1, 16), np.sum(pixels) * 0.125, dtype=np.float32)
    for forward in (embedding, jax.jit(lambda x: embedding(x))):
        actual = forward(jnp.asarray(pixels, dtype=dtype))
        assert actual.dtype == dtype
        np.testing.assert_array_equal(np.asarray(actual, dtype=np.float32), expected)
        assert embedding.weight.value.dtype == param_dtype
        np.testing.assert_array_equal(np.asarray(embedding.weight.value, dtype=np.float32), weights)


def test_vision_resolves_config_output_flags_and_explicit_overrides():
    model = vision_model(output_attentions=True, output_hidden_states=True)
    pixels = jnp.ones((1, 3, 8, 16), jnp.float32)
    outputs = model(pixel_values=pixels)
    assert len(outputs.hidden_states) == 2
    assert len(outputs.attentions) == 1
    assert outputs.hidden_states[-1].shape == (1, 2, 16)
    assert outputs.attentions[0].shape == (1, 2, 2, 2)
    disabled = model(pixel_values=pixels, output_hidden_states=False, output_attentions=False)
    assert disabled.hidden_states is None
    assert disabled.attentions is None


def test_runtime_image_sizes_pack_only_genuine_rectangular_patches():
    model = vision_model(num_hidden_layers=2, output_hidden_states=True, output_attentions=True)
    pixels = np.random.default_rng(12).normal(size=(2, 3, 16, 24)).astype(np.float32)
    sizes = np.array([[8, 16], [16, 8]], np.int32)
    cropped = [jnp.asarray(pixels[0, :, :8, :16]), jnp.asarray(pixels[1, :, :16, :8])]
    reference = model(pixel_values=cropped)
    concrete = model(pixel_values=jnp.asarray(pixels), image_sizes=sizes)
    np.testing.assert_allclose(concrete.last_hidden_state, reference.last_hidden_state, rtol=2e-5, atol=2e-5)
    compiled = jax.jit(lambda p, s: model(pixel_values=p, image_sizes=s))
    packed = compiled(jnp.asarray(pixels), jnp.asarray(sizes))
    assert isinstance(packed, PixtralVisionModelOutput)
    assert packed.last_hidden_state.shape == (1, 12, 16)
    np.testing.assert_array_equal(packed.patch_lengths, [2, 2])
    np.testing.assert_array_equal(packed.patch_grid_sizes, [[1, 2], [2, 1]])
    np.testing.assert_array_equal(packed.patch_attention_mask, [[True] * 4 + [False] * 8])
    for actual, expected in zip(packed.hidden_states, reference.hidden_states, strict=True):
        np.testing.assert_allclose(actual[:, :4], expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(actual[:, 4:], 0)
    np.testing.assert_allclose(packed.last_hidden_state[:, :4], reference.last_hidden_state, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(packed.last_hidden_state[:, 4:], 0)
    for actual, expected in zip(packed.attentions, reference.attentions, strict=True):
        np.testing.assert_allclose(actual[..., :4, :4], expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(actual[..., 4:, :], 0)
        np.testing.assert_array_equal(actual[..., :, 4:], 0)
    # Fixed capacity is an explicit option for identical eager/compiled layouts.
    eager_packed = model(pixel_values=jnp.asarray(pixels), image_sizes=sizes, pad_to_max_patches=True)
    np.testing.assert_allclose(eager_packed.last_hidden_state, packed.last_hidden_state, rtol=2e-5, atol=2e-5)
    # Padded pixels cannot contaminate valid patches or become attention keys.
    pixels[0, :, 8:, :] = 100
    pixels[0, :, :, 16:] = -100
    pixels[1, :, :, 8:] = 100
    perturbed = compiled(jnp.asarray(pixels), jnp.asarray(sizes))
    np.testing.assert_allclose(perturbed.last_hidden_state, packed.last_hidden_state, rtol=2e-5, atol=2e-5)
