"""Rotary embeddings: DeepSeek module offsets/GPT-J layout, yarn inverse-frequency parity, rope registry."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.rotary import (
    DeepseekScalingRotaryEmbedding,
    get_frequencies,
    get_inv_frequencies,
    get_rope,
)
from easydel.layers.rotary._modules import AVAILABLE_ROPE_TYPES
from jax import numpy as jnp

_DEEPSEEK = {
    "rope_type": "deepseek_yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 64,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
}


def _qk(seq=8, heads=2, dim=16):
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((1, seq, heads, dim)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((1, seq, heads, dim)), jnp.float32)
    return q, k, jnp.arange(seq)[None, :]


@pytest.mark.parametrize("neox", [True, False])
def test_deepseek_rope_offsets_shift_positions(neox):
    rope = get_rope(
        head_size=16, rotary_dim=16, max_position=256, base=10000, is_neox_style=neox, rope_scaling=_DEEPSEEK
    )
    assert isinstance(rope, DeepseekScalingRotaryEmbedding)
    q, k, pos = _qk()
    shifted_q, shifted_k = rope(pos, q, k, offsets=jnp.full_like(pos, 5))
    ref_q, ref_k = rope(pos + 5, q, k)
    plain_q, _ = rope(pos, q, k)
    np.testing.assert_allclose(np.asarray(shifted_q), np.asarray(ref_q), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(shifted_k), np.asarray(ref_k), rtol=1e-5, atol=1e-5)
    assert not np.allclose(np.asarray(shifted_q), np.asarray(plain_q))
    assert shifted_q.shape == q.shape


def test_yarn_inv_frequencies_without_original_max_position():
    scaling = {"rope_type": "yarn", "factor": 2.0}
    inv = get_inv_frequencies(head_size=16, rotary_dim=16, max_position=128, base=10000, rope_scaling=scaling)
    assert inv.shape == (8,)
    cache = get_frequencies(head_size=16, rotary_dim=16, max_position=128, base=10000, rope_scaling=scaling)
    assert cache.shape[-1] == 16


def test_deepseek_yarn_inv_frequencies_honor_truncate():
    base = {**_DEEPSEEK, "beta_fast": 32, "beta_slow": 1}
    kw = {"head_size": 32, "rotary_dim": 32, "max_position": 256, "base": 10000}
    on = get_inv_frequencies(rope_scaling={**base, "truncate": True}, **kw)
    off = get_inv_frequencies(rope_scaling={**base, "truncate": False}, **kw)
    assert on.shape == off.shape == (16,)
    assert not np.allclose(np.asarray(on), np.asarray(off))


@pytest.mark.parametrize("neox", [True, False])
@pytest.mark.parametrize("rotary_dim", [4, 8])
@pytest.mark.parametrize("batched_positions", [True, False])
@pytest.mark.parametrize("offset_layout", [None, "sequence", "batch"])
def test_deepseek_rope_rotation_matches_numpy_and_honors_dtype(neox, rotary_dim, batched_positions, offset_layout):
    rope = DeepseekScalingRotaryEmbedding(
        head_size=8,
        rotary_dim=rotary_dim,
        max_position_embeddings=16,
        base=10000,
        is_neox_style=neox,
        dtype=jnp.bfloat16,
        scaling_factor=2.0,
    )
    host = np.arange(32, dtype=np.float32).reshape(1, 2, 2, 8) / 8
    angles = np.arange(16 * rotary_dim // 2, dtype=np.float32).reshape(16, rotary_dim // 2) / 32
    cache = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    positions = np.array([1, 3], dtype=np.int32)
    if batched_positions:
        positions = positions[None, :]
    offsets = None if offset_layout is None else np.array([1, 2], dtype=np.int32)
    if offset_layout == "batch":
        offsets = offsets[None, :]
    q, k = rope(
        jnp.asarray(positions),
        jnp.asarray(host),
        jnp.asarray(host * 2),
        offsets=None if offsets is None else jnp.asarray(offsets),
        frequencies=SimpleNamespace(value=jnp.asarray(cache)),
    )
    selected_positions = (positions if offsets is None else positions + offsets).reshape(1, 2)
    x = host[..., :rotary_dim]
    if neox:
        half = rotary_dim // 2
        rotated = np.concatenate((-x[..., half:], x[..., :half]), axis=-1)
        cos = np.concatenate((np.cos(angles[selected_positions]),) * 2, axis=-1)
        sin = np.concatenate((np.sin(angles[selected_positions]),) * 2, axis=-1)
    else:
        pairs = x.reshape(*x.shape[:-1], rotary_dim // 2, 2)
        rotated = np.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(x.shape)
        cos = np.repeat(np.cos(angles[selected_positions]), 2, axis=-1)
        sin = np.repeat(np.sin(angles[selected_positions]), 2, axis=-1)
    expected = np.concatenate((x * cos[:, :, None] + rotated * sin[:, :, None], host[..., rotary_dim:]), axis=-1)
    assert q.shape == k.shape == host.shape
    assert q.dtype == k.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(q, dtype=np.float32), expected, rtol=5e-3, atol=5e-3)
    np.testing.assert_allclose(np.asarray(k, dtype=np.float32), expected * 2, rtol=5e-3, atol=5e-3)
    np.testing.assert_array_equal(np.asarray(q[..., rotary_dim:], dtype=np.float32), host[..., rotary_dim:])


def test_linear_frequency_multiple_scaling_regimes_keep_all_positions():
    from easydel.layers.rotary._compute_fns import compute_linear_frequencies

    out = compute_linear_frequencies(10000, 4, 4, [1.0, 2.0])
    inverse = np.array([1.0, 0.01], dtype=np.float32)
    positions = np.concatenate((np.arange(4), np.arange(8) / 2))
    angles = positions[:, None] * inverse[None]
    expected = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-6)


def test_yarn_frequency_magnitude_uses_shared_mscale_formula():
    out = get_frequencies(
        head_size=16, rotary_dim=16, max_position=32, base=10000, rope_scaling={"rope_type": "yarn", "factor": 4.0}
    )
    cos, sin = np.split(np.asarray(out), 2, axis=-1)
    expected_squared_magnitude = (1.0 + 0.1 * math.log(4.0)) ** 2
    np.testing.assert_allclose(cos**2 + sin**2, expected_squared_magnitude, rtol=1e-5, atol=1e-6)


def test_rope_registry_dispatches_custom_frequency_cache(monkeypatch):
    class CustomRotary:
        @staticmethod
        def compute_frequencies(*, head_size, rotary_dim, max_position, base, rope_scaling):
            positions = jnp.arange(max_position, dtype=jnp.float32) / rope_scaling["frequency_divisor"]
            angles = jnp.broadcast_to(positions[:, None], (max_position, rotary_dim // 2))
            return jnp.concatenate((jnp.cos(angles), jnp.sin(angles)), axis=-1)

    monkeypatch.setitem(AVAILABLE_ROPE_TYPES, "review_custom_cache", CustomRotary)
    actual = get_frequencies(
        head_size=4,
        rotary_dim=4,
        max_position=4,
        base=10000,
        rope_scaling={"rope_type": "review_custom_cache", "frequency_divisor": 3.0},
    )
    angles = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None] / 3.0, (4, 2))
    expected = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("rope_type", ["yarn", "deepseek_yarn"])
@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize("original_max", [None, 64])
def test_yarn_module_factory_preserves_truncate_and_context_fallback(rope_type, truncate, original_max):
    from easydel.layers.rotary import YaRNScalingRotaryEmbedding

    scaling = {"rope_type": rope_type, "factor": 4.0, "truncate": truncate}
    if original_max is not None:
        scaling["original_max_position_embeddings"] = original_max
    rope = get_rope(head_size=32, rotary_dim=32, max_position=256, base=10000, rope_scaling=scaling)
    expected_class = YaRNScalingRotaryEmbedding if rope_type == "yarn" else DeepseekScalingRotaryEmbedding
    assert type(rope) is expected_class
    assert rope.truncate is truncate
    assert rope.max_position_embeddings == (256 if original_max is None else original_max)
    assert rope.scaling_factor == 4.0


@pytest.mark.cpu_ok
def test_rope_registry_drives_module_factory(monkeypatch):
    from easydel.layers.rotary._modules import rope_wrapper

    class CustomRotary:
        def __init__(self, **kwargs):
            self.options = kwargs

    # Restore registry entries automatically even if an assertion fails.
    monkeypatch.setitem(AVAILABLE_ROPE_TYPES, "review_custom", CustomRotary)
    rope_wrapper("review_custom")(CustomRotary)
    out = get_rope(
        head_size=16,
        rotary_dim=16,
        max_position=64,
        base=10000,
        rope_scaling={"rope_type": "review_custom", "custom_factor": 3.0},
    )
    assert isinstance(out, CustomRotary)
    assert out.options["custom_factor"] == 3.0
    assert out.options["max_position_embeddings"] == 64
    assert out.options["rotary_dim"] == 16

    monkeypatch.setitem(AVAILABLE_ROPE_TYPES, "default", CustomRotary)
    assert isinstance(get_rope(head_size=8, rotary_dim=8, max_position=32, base=10000), CustomRotary)
    with pytest.raises(ValueError, match="deepseek_yarn"):
        get_rope(head_size=16, rotary_dim=16, max_position=64, base=10000, rope_scaling={"rope_type": "bogus"})
