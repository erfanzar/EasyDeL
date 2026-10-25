"""Multimodal layout composes with independently referenced frequency scaling."""

import math
from copy import deepcopy

import numpy as np
import pytest
from easydel.layers.rotary import MultiModalRotaryEmbedding, get_frequencies, get_inv_frequencies, get_rope
from jax import numpy as jnp


def _scaling(kind, layout):
    return {
        "rope_type": kind,
        "factor": 2.0,
        "original_max_position_embeddings": 32,
        "attention_factor": 1.25,
        "mrope_section": [2, 2, 2],
        "mrope_interleaved": layout == "interleaved",
        "repetition_style": layout == "repeated",
    }


@pytest.mark.cpu_ok
@pytest.mark.parametrize("kind", ["default", "mrope", "linear", "dynamic", "yarn"])
def test_multimodal_factory_keeps_frequency_configuration(kind):
    scaling = _scaling(kind, "interleaved")
    before = deepcopy(scaling)
    module = get_rope(16, 12, 128, 4096, rope_scaling=scaling)
    assert isinstance(module, MultiModalRotaryEmbedding)
    assert module.mrope_section == (2, 2, 2)
    assert module.mrope_interleaved is True
    assert module.rotary_dim == 12
    assert module.rope_scaling == (None if kind in ("default", "mrope") else scaling)
    assert scaling == before


@pytest.mark.cpu_ok
def test_hunyuan_legacy_alpha_still_uses_folded_base_with_multimodal_layout():
    from easydel.modules.hunyuan_vl import HunYuanVLTextConfig

    config = HunYuanVLTextConfig(
        hidden_size=32,
        num_attention_heads=2,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "dynamic", "alpha": 2.0, "factor": 4.0},
    )
    rope = get_rope(16, 16, config.max_position_embeddings, config.rope_theta, rope_scaling=config.rope_scaling)
    assert isinstance(rope, MultiModalRotaryEmbedding)
    assert rope.base == pytest.approx(10000.0 * 2.0 ** (16.0 / 14.0))
    assert rope.rope_scaling is None
    assert sum(rope.mrope_section) == 8


def _reference_inverse(kind, length=128):
    inverse = 4096.0 ** (-np.arange(0, 12, 2, dtype=np.float32) / 12)
    if kind == "linear":
        inverse = inverse / 2
    elif kind == "dynamic":
        bases = 4096.0 * ((2 * np.maximum(np.asarray(length), 32) / 32) - 1) ** (12 / 10)
        inverse = bases[..., None] ** (-np.arange(0, 12, 2, dtype=np.float32) / 12)
    elif kind == "yarn":
        low = max(math.floor(12 * math.log(32 / (32 * 2 * math.pi)) / (2 * math.log(4096))), 0)
        high = min(math.ceil(12 * math.log(32 / (2 * math.pi)) / (2 * math.log(4096))), 11)
        if high == low:
            high += 0.001
        extrapolation = 1 - np.clip((np.arange(6) - low) / (high - low), 0, 1)
        inverse = inverse / 2 * (1 - extrapolation) + inverse * extrapolation
    return inverse


def _reference_tables(kind, positions):
    inverse = _reference_inverse(kind, positions + 1) if kind == "dynamic" else _reference_inverse(kind)
    angles = positions[..., None] * inverse
    magnitude = 1.25 if kind == "yarn" else 1.0
    return np.cos(angles) * magnitude, np.sin(angles) * magnitude


def _merge(half, layout):
    if layout == "interleaved":
        # Equal sections cover the complete half-width: T,H,W,T,H,W.
        merged = np.stack([half[channel % 3, ..., channel] for channel in range(6)], axis=-1)
        return np.concatenate([merged, merged], axis=-1)
    doubled = np.concatenate([half, half], axis=-1)
    widths = [2] * 6 if layout == "repeated" else [4] * 3
    chunks = np.split(doubled, np.cumsum(widths)[:-1], axis=-1)
    return np.concatenate([chunk[index % 3] for index, chunk in enumerate(chunks)], axis=-1)


@pytest.mark.parametrize("kind", ["default", "mrope", "linear", "dynamic", "yarn"])
@pytest.mark.parametrize("layout", ["chunked", "repeated", "interleaved"])
@pytest.mark.parametrize("neox", [True, False])
def test_scaled_multimodal_direct_cache_and_tables_match_numpy(kind, layout, neox):
    scaling = _scaling(kind, layout)
    rope = get_rope(16, 12, 128, 4096, is_neox_style=neox, rope_scaling=scaling, dtype=jnp.float32)
    positions = np.array([[[1, 40]] * 4, [[2, 41]] * 4, [[3, 42]] * 4], dtype=np.int32)
    offsets = np.array([[1, 2]] * 4, dtype=np.int32)
    shifted = positions + offsets
    host = np.arange(4 * 2 * 16, dtype=np.float32).reshape(4, 2, 1, 16) / 64
    cos_half, sin_half = _reference_tables(kind, shifted)
    cos, sin = _merge(cos_half, layout), _merge(sin_half, layout)
    if not neox:
        cos, sin = np.repeat(cos[..., :6], 2, axis=-1), np.repeat(sin[..., :6], 2, axis=-1)
    x = host[..., :12]
    if neox:
        rotated = np.concatenate([-x[..., 6:], x[..., :6]], axis=-1)
    else:
        pairs = x.reshape(4, 2, 1, 6, 2)
        rotated = np.stack([-pairs[..., 1], pairs[..., 0]], axis=-1).reshape(x.shape)
    expected = np.concatenate([x * cos[:, :, None] + rotated * sin[:, :, None], host[..., 12:]], axis=-1)
    cache = get_frequencies(16, 12, 128, 4096, scaling)
    cache_host = np.asarray(cache)
    ref_cos, ref_sin = _reference_tables(kind, np.arange(cache.shape[0]))
    np.testing.assert_allclose(cache_host, np.concatenate([ref_cos, ref_sin], axis=-1), rtol=2e-5, atol=2e-5)
    inv = get_inv_frequencies(16, 12, 128, 4096, scaling)
    np.testing.assert_allclose(np.asarray(inv), _reference_inverse(kind), rtol=1e-5, atol=1e-7)
    for frequency_cache in (None, cache):
        query, key = rope(
            jnp.asarray(positions),
            jnp.asarray(host),
            jnp.asarray(host * 2),
            offsets=jnp.asarray(offsets),
            frequencies=frequency_cache,
        )
        assert query.dtype == key.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(query), expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(np.asarray(key), expected * 2, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(np.asarray(query[..., 12:]), host[..., 12:])
    actual_cos, actual_sin = rope.compute_cos_sin(jnp.asarray(shifted))
    np.testing.assert_allclose(np.asarray(actual_cos), cos, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual_sin), sin, rtol=2e-5, atol=2e-5)
