"""Hardware-only numerical regressions for shared vision primitives."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.kimi_vl.modeling_kimi_vl import MoonVisionPatchEmbed, _apply_rope
from easydel.modules.llama4.modeling_llama4 import Llama4TextL2Norm, vision_apply_rotary_emb
from jax import numpy as jnp


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_llama4_scaleless_norm_keeps_fp32_product_before_cast(dtype):
    x = jnp.asarray([[[0.37, -2.9, 1.13, 4.71], [0.09, -0.31, 0.17, 0.22]]], dtype=dtype)
    values = np.asarray(x).astype(np.float32)
    expected = values / np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + 0.1)
    expected = jnp.asarray(expected, dtype=dtype)
    output = Llama4TextL2Norm(eps=0.1)(x)
    assert output.dtype == dtype
    np.testing.assert_allclose(np.asarray(output, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=1e-5)


@pytest.mark.parametrize("family", ["kimi_vl", "llama4"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_vision_rotary_matches_complex_multiplication(family, dtype):
    query = jnp.arange(2 * 3 * 2 * 8, dtype=jnp.float32).reshape(2, 3, 2, 8).astype(dtype) / 10
    key = -query / 2
    angles = np.arange(12, dtype=np.float32).reshape(3, 4) / 7
    frequencies = np.exp(1j * angles).astype(np.complex64)

    def reference(x):
        values = np.asarray(x).astype(np.float32).reshape(2, 3, 2, 4, 2)
        complex_values = values[..., 0] + 1j * values[..., 1]
        rotated = complex_values * frequencies[None, :, None, :]
        real_values = np.stack((rotated.real, rotated.imag), axis=-1).reshape(x.shape)
        return np.asarray(jnp.asarray(real_values, dtype=dtype), dtype=np.float32)

    if family == "kimi_vl":
        rotated_query, rotated_key = _apply_rope(query, key, jnp.asarray(frequencies))
    else:
        rotated_query, rotated_key = vision_apply_rotary_emb(query, key, jnp.asarray(frequencies))
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert rotated_query.dtype == dtype
    assert rotated_key.dtype == dtype
    tolerance = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(np.asarray(rotated_query, dtype=np.float32), reference(query), rtol=tolerance, atol=1e-6)
    np.testing.assert_allclose(np.asarray(rotated_key, dtype=np.float32), reference(key), rtol=tolerance, atol=1e-6)


def test_moon_patch_embedding_keeps_parameter_storage_dtype():
    embed = MoonVisionPatchEmbed(
        out_dim=4,
        patch_size=2,
        pos_emb_height=2,
        pos_emb_width=2,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    assert embed.proj.weight.value.dtype == jnp.float32
    assert embed.proj.bias.value.dtype == jnp.float32
    assert embed.proj.precision == jax.lax.Precision.HIGHEST

    # A full forward must cast convolution operands, not the stored leaves.
    # Four NCHW patches form a 2x2 image grid; choose exact values so the
    # independent flattened-patch matrix product is stable in bf16.
    pixels = jnp.arange(4 * 3 * 2 * 2, dtype=jnp.float32).reshape(4, 3, 2, 2) / 16
    kernel = jnp.ones(embed.proj.weight.value.shape, dtype=jnp.float32) / 8
    embed.proj.weight.value = kernel
    embed.proj.bias.value = jnp.zeros((4,), dtype=jnp.float32)
    embed.pos_emb.weight.value = jnp.zeros((4, 2, 2), dtype=jnp.float32)
    output = embed(pixels, [(2, 2)])
    expected = np.asarray(pixels).sum(axis=(1, 2, 3))[:, None] / 8
    expected = np.broadcast_to(expected, (4, 4))
    assert output.shape == (4, 4)
    assert output.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(output, dtype=np.float32), expected, rtol=1e-2)
    assert embed.proj.weight.value.dtype == jnp.float32
    assert embed.proj.bias.value.dtype == jnp.float32
