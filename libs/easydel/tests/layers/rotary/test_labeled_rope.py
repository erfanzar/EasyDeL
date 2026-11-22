# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity of the label-keyed rope helpers against independent float64 references.

References: transformers' ``_compute_yarn_parameters`` defaulting (ported to NumPy) for the frequencies, and
DeepSeek's reference ``apply_rotary_emb`` (consecutive channel pairs viewed as complex numbers, multiplied by
``freqs_cis``; the inverse uses the conjugate) for the rotation.
"""

import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.rotary import (
    apply_rope_interleaved_partial,
    compute_labeled_inv_frequencies,
    cos_sin_from_inv_frequencies,
    labeled_rope_cos_sin,
)


def _ref_inv_freq(params: dict, head_dim: int, max_pos: int) -> tuple[np.ndarray, float]:
    base = float(params["rope_theta"])
    dim = int(head_dim * float(params.get("partial_rotary_factor", 1.0)))
    pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float64) / dim)
    if params.get("rope_type", "default") == "default":
        return 1.0 / pos_freqs, 1.0
    original = params.get("original_max_position_embeddings", max_pos)
    factor = params.get("factor") or max_pos / original
    beta_fast, beta_slow = params.get("beta_fast") or 32, params.get("beta_slow") or 1

    def correction_dim(rotations):
        return (dim * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if params.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float64) - low) / (high - low), 0, 1)
    extrap_weight = 1 - ramp
    inv = (1.0 / (factor * pos_freqs)) * (1 - extrap_weight) + (1.0 / pos_freqs) * extrap_weight
    attention = params.get("attention_factor")
    if attention is None:
        attention = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
    return inv, float(attention)


def _ref_rotate(x: np.ndarray, angles: np.ndarray, scale: float, inverse: bool) -> np.ndarray:
    """DeepSeek reference rotation on the trailing ``2 * angles.shape[-1]`` channels of ``x [B, H, S, D]``."""
    half = angles.shape[-1]
    nope, rope = x[..., : -2 * half], x[..., -2 * half :]
    z = rope.reshape(*rope.shape[:-1], half, 2)
    z = z[..., 0] + 1j * z[..., 1]
    freqs_cis = scale * np.exp(1j * angles)[:, None]  # [B, 1, S, half]
    z = z * (np.conj(freqs_cis) if inverse else freqs_cis)
    rotated = np.stack([z.real, z.imag], axis=-1).reshape(rope.shape)
    return np.concatenate([nope, rotated], axis=-1)


PARAMS = {
    "main": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.125},
    "compress": {
        "rope_type": "yarn",
        "rope_theta": 160000.0,
        "partial_rotary_factor": 0.125,
        "factor": 16.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    },
    "yarn_defaults": {"rope_type": "yarn", "rope_theta": 10000.0, "original_max_position_embeddings": 4096},
}


@pytest.mark.parametrize("label", sorted(PARAMS))
def test_labeled_inv_frequencies_match_transformers_yarn(label):
    head_dim, max_pos = 512, 1048576
    got, got_scale = compute_labeled_inv_frequencies(PARAMS[label], head_dim, max_pos)
    want, want_scale = _ref_inv_freq(PARAMS[label], head_dim, max_pos)
    # float32 ``base ** (2i / d)``: a few ulp on CPU, looser on TPU's transcendental units.
    np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=1e-5, atol=0)
    assert got_scale == pytest.approx(want_scale, rel=1e-12)


def test_labeled_rope_rejects_unknown_type():
    with pytest.raises(NotImplementedError):
        compute_labeled_inv_frequencies({"rope_type": "longrope", "rope_theta": 1e4}, 64, 4096)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("label", ["main", "compress"])
def test_partial_interleaved_rotation_matches_deepseek_reference(label, inverse):
    rng = np.random.default_rng(0)
    head_dim, batch, heads, seq = 512, 2, 3, 7
    x = rng.normal(size=(batch, heads, seq, head_dim)).astype(np.float32)
    # Bounded so the float32 angle ``position * inv_freq`` stays within ~1e-4 rad of the float64 reference; at
    # positions near 2e5 the float32 ulp alone is ~1.6e-2 rad, which no float32 implementation can meet.
    positions = rng.integers(0, 1024, size=(batch, seq)).astype(np.int32)
    config = SimpleNamespace(
        head_dim=head_dim, max_position_embeddings=1048576, rope_parameters_for_label=lambda lab: PARAMS[lab]
    )
    cos, sin = labeled_rope_cos_sin(config, label, jnp.asarray(positions), jnp.float32)
    got = apply_rope_interleaved_partial(jnp.asarray(x), cos, -sin if inverse else sin, unsqueeze_dim=1)

    # Frequencies are pinned by ``test_labeled_inv_frequencies_match_transformers_yarn``; here the reference angles use
    # the helper's float32 frequencies so this test isolates the rotation (pairing, sign, slice, scaling), which is
    # independent code. Otherwise a few-ulp frequency difference (TPU ``pow``) times the angle dominates.
    inv = np.asarray(compute_labeled_inv_frequencies(PARAMS[label], head_dim, 1048576)[0], np.float64)
    scale = _ref_inv_freq(PARAMS[label], head_dim, 1048576)[1]
    angles = positions.astype(np.float64)[..., None] * inv[None, None, :]
    want = _ref_rotate(x.astype(np.float64), angles, scale, inverse)
    # A wrong pairing, sign or channel slice is an O(1) error; float32 angle/trig error is well under 5e-3.
    np.testing.assert_allclose(np.asarray(got, np.float64), want, rtol=0, atol=5e-3 * max(1.0, scale))
    np.testing.assert_array_equal(
        np.asarray(got)[..., : head_dim - 2 * inv.shape[0]], x[..., : head_dim - 2 * inv.shape[0]]
    )


def test_rotation_round_trip_and_dtype():
    x = jnp.asarray(np.random.default_rng(1).normal(size=(1, 5, 4, 256)), jnp.bfloat16)  # [B, S, H, D]
    inv = compute_labeled_inv_frequencies(PARAMS["main"], 512, 4096)[0]
    cos, sin = cos_sin_from_inv_frequencies(inv, jnp.arange(5, dtype=jnp.int32)[None], jnp.float32)
    y = apply_rope_interleaved_partial(x, cos, sin, unsqueeze_dim=2)
    back = apply_rope_interleaved_partial(y, cos, -sin, unsqueeze_dim=2)
    assert y.dtype == jnp.bfloat16 and y.shape == x.shape
    np.testing.assert_allclose(np.asarray(back, np.float32), np.asarray(x, np.float32), atol=3e-2)
