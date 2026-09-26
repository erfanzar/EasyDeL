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

"""RoPE cos/sin tables vs transformers' ``ROPE_INIT_FUNCTIONS``.

References are built from HF's inverse frequencies and attention scaling in
float64; the EasyDeL tables are float32, so small configs keep every rotation
angle below a few thousand radians and a ``2e-4`` absolute tolerance holds.
"""

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from easydel.layers.rotary import RopeConfig, apply_phi3_rope, get_frequencies  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

ATOL = 2e-4
HEAD_DIM = 32
NUM_HEADS = 2


def _hf_config(rope_parameters, max_position_embeddings, head_dim=HEAD_DIM):
    return transformers.LlamaConfig(
        hidden_size=head_dim * NUM_HEADS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=head_dim,
        num_hidden_layers=1,
        intermediate_size=8,
        vocab_size=16,
        max_position_embeddings=max_position_embeddings,
        rope_parameters=dict(rope_parameters),
    )


def _hf_cos_sin(config, rope_type, positions, seq_len=None):
    """float64 cos/sin (``rotary_dim // 2`` wide) for ``positions`` from HF's init fn."""
    inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu", seq_len=seq_len)
    angles = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq.double().numpy()[None, :]
    return np.cos(angles) * attention_scaling, np.sin(angles) * attention_scaling


def _ed_cos_sin(rope_parameters, max_position, base, head_dim=HEAD_DIM, partial_rotary_factor=1.0):
    """EasyDeL basic-layout table split into ``(cos, sin)`` halves."""
    frequencies = get_frequencies(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=base,
        rope_scaling=RopeConfig.from_dict(rope_parameters).to_dict(),
        partial_rotary_factor=partial_rotary_factor,
    )
    frequencies = np.asarray(frequencies, dtype=np.float64)
    cos, sin = np.split(frequencies, 2, axis=-1)
    return cos, sin


def test_llama3_table_covers_max_position_embeddings():
    """Positions past ``original_max_position_embeddings`` must not clamp to the last row."""
    params = {
        "rope_type": "llama3",
        "rope_theta": 500000.0,
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    }
    max_position = 512
    cos, sin = _ed_cos_sin(params, max_position, params["rope_theta"])
    assert cos.shape[0] == max_position
    ref_cos, ref_sin = _hf_cos_sin(_hf_config(params, max_position), "llama3", np.arange(max_position))
    np.testing.assert_allclose(cos, ref_cos, atol=ATOL)
    np.testing.assert_allclose(sin, ref_sin, atol=ATOL)


@pytest.mark.parametrize("factor", [2.0, 4.0])
def test_dynamic_ntk_matches_hf_per_sequence_length(factor):
    """Inside the original window the table is unscaled; beyond it row ``p`` uses HF's base for ``p + 1``."""
    max_position = 128
    params = {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": factor}
    cos, sin = _ed_cos_sin(params, max_position, params["rope_theta"])
    config = _hf_config(params, max_position)

    # Every sequence that fits in the original window: exact HF (unscaled) table.
    inside = np.arange(max_position)
    ref_cos, ref_sin = _hf_cos_sin(config, "dynamic", inside, seq_len=max_position)
    np.testing.assert_allclose(cos[:max_position], ref_cos, atol=ATOL)
    np.testing.assert_allclose(sin[:max_position], ref_sin, atol=ATOL)

    # Beyond it: the base HF uses when that token is the newest one.
    for position in (max_position, max_position + 7, int(max_position * factor) - 1):
        ref_cos, ref_sin = _hf_cos_sin(config, "dynamic", [position], seq_len=position + 1)
        np.testing.assert_allclose(cos[position], ref_cos[0], atol=ATOL)
        np.testing.assert_allclose(sin[position], ref_sin[0], atol=ATOL)


@pytest.mark.parametrize("truncate", [True, False])
@pytest.mark.parametrize("attention_factor", [None, 1.5])
def test_yarn_attention_factor_and_truncate_match_hf(truncate, attention_factor):
    params = {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 4.0,
        "original_max_position_embeddings": 64,
        "beta_fast": 32,
        "beta_slow": 1,
        "truncate": truncate,
    }
    if attention_factor is not None:
        params["attention_factor"] = attention_factor
    max_position = 256
    cos, sin = _ed_cos_sin(params, max_position, params["rope_theta"])
    assert cos.shape[0] >= max_position
    ref_cos, ref_sin = _hf_cos_sin(_hf_config(params, max_position), "yarn", np.arange(max_position))
    np.testing.assert_allclose(cos[:max_position], ref_cos, atol=ATOL)
    np.testing.assert_allclose(sin[:max_position], ref_sin, atol=ATOL)


def test_yarn_table_never_shorter_than_served_context():
    """``original * factor`` < ``max_position`` must still cover ``max_position`` rows."""
    params = {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 2.0, "original_max_position_embeddings": 64}
    cos, _ = _ed_cos_sin(params, 512, params["rope_theta"])
    assert cos.shape[0] >= 512


def _phi3_split(frequencies):
    """Phi-3 layout ``(1, L, 2*rotary_dim)`` -> float64 ``(cos, sin)`` halves of width ``rotary_dim // 2``."""
    frequencies = np.asarray(frequencies, dtype=np.float64)[0]
    cos, sin = np.split(frequencies, 2, axis=-1)
    half = cos.shape[-1] // 2
    np.testing.assert_array_equal(cos[:, :half], cos[:, half:])
    return cos[:, :half], sin[:, :half]


@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.75])
def test_longrope_short_inside_original_long_beyond(partial_rotary_factor):
    """Short factors for rows < original (exact HF for such sequences), long factors beyond; partial rotary ok."""
    original = 64
    max_position = 256
    rotary_dim = int(HEAD_DIM * partial_rotary_factor)
    rng = np.random.default_rng(0)
    params = {
        "rope_type": "longrope",
        "rope_theta": 10000.0,
        "original_max_position_embeddings": original,
        "short_factor": [float(x) for x in rng.uniform(1.0, 1.5, rotary_dim // 2)],
        "long_factor": [float(x) for x in rng.uniform(1.5, 8.0, rotary_dim // 2)],
        "partial_rotary_factor": partial_rotary_factor,
    }
    frequencies = get_frequencies(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=max_position,
        base=params["rope_theta"],
        rope_scaling=RopeConfig.from_dict(params).to_dict(),
        partial_rotary_factor=partial_rotary_factor,
    )
    assert frequencies.shape == (1, max_position, 2 * rotary_dim)
    cos, sin = _phi3_split(frequencies)
    config = _hf_config(params, max_position)

    inside = np.arange(original)
    ref_cos, ref_sin = _hf_cos_sin(config, "longrope", inside, seq_len=original)
    np.testing.assert_allclose(cos[:original], ref_cos, atol=ATOL)
    np.testing.assert_allclose(sin[:original], ref_sin, atol=ATOL)

    beyond = np.arange(original, max_position)
    ref_cos, ref_sin = _hf_cos_sin(config, "longrope", beyond, seq_len=max_position)
    np.testing.assert_allclose(cos[original:], ref_cos, atol=ATOL)
    np.testing.assert_allclose(sin[original:], ref_sin, atol=ATOL)


def test_apply_phi3_rope_partial_rotary_matches_hf():
    """Phi-4-mini style: only the first ``rotary_dim`` channels rotate, the rest pass through."""
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

    rotary_dim = 24
    seq_len = 16
    params = {
        "rope_type": "longrope",
        "original_max_position_embeddings": 64,
        "short_factor": [1.0] * (rotary_dim // 2),
        "long_factor": [2.0] * (rotary_dim // 2),
    }
    frequencies = get_frequencies(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position=128,
        base=10000.0,
        rope_scaling=RopeConfig.from_dict(params).to_dict(),
        partial_rotary_factor=rotary_dim / HEAD_DIM,
    )
    rng = np.random.default_rng(1)
    query = rng.standard_normal((1, seq_len, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    key = rng.standard_normal((1, seq_len, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    positions = np.arange(seq_len)
    q_out, k_out = apply_phi3_rope(jnp.asarray(query), jnp.asarray(key), jnp.asarray(positions)[None], frequencies)

    emb = np.asarray(frequencies, dtype=np.float32)[0, positions]
    cos, sin = np.split(emb, 2, axis=-1)
    ref_q, ref_k = apply_rotary_pos_emb(
        torch.from_numpy(query).transpose(1, 2),
        torch.from_numpy(key).transpose(1, 2),
        torch.from_numpy(cos)[None],
        torch.from_numpy(sin)[None],
    )
    np.testing.assert_allclose(np.asarray(q_out), ref_q.transpose(1, 2).numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(k_out), ref_k.transpose(1, 2).numpy(), atol=1e-5)
    np.testing.assert_array_equal(np.asarray(q_out)[..., rotary_dim:], query[..., rotary_dim:])
