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

"""Qwen-VL family mRoPE / attention-bias parity against the HF torch reference.

Covers:
* Qwen3.5 ``_get_rope_index_from_mm_token_types`` vs HF ``Qwen3_5Model.get_rope_index``
  (meshgrid T/H/W ordering and per-frame video grid splitting).
* Qwen2-VL chunked mRoPE channel layout vs HF ``apply_multimodal_rotary_pos_emb``.
* Qwen2-VL text attention bias layout (Q/K/V biased, O bias-free) and vision head count.
* Qwen3-VL(-MoE) text config keeping ``rope_parameters`` (mrope + theta) across reloads.
"""

import types

import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
import torch
from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention, Qwen2VLVisionBlock
from easydel.modules.qwen2_vl.qwen2_vl_configuration import Qwen2VLTextConfig, Qwen2VLVisionConfig
from easydel.modules.qwen3_5.modeling_qwen3_5 import _get_rope_index_from_mm_token_types
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLConfig, Qwen3VLTextConfig
from easydel.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import Qwen3VLMoeTextConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model


def _hf_qwen3_5_rope_index(input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask, sms):
    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=sms))
    )
    fake.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, fake)
    return Qwen3_5Model.get_rope_index(
        fake,
        torch.as_tensor(input_ids),
        torch.as_tensor(mm_token_type_ids),
        None if image_grid_thw is None else torch.as_tensor(image_grid_thw),
        None if video_grid_thw is None else torch.as_tensor(video_grid_thw).clone(),
        None if attention_mask is None else torch.as_tensor(attention_mask),
    )


def _mixed_batch():
    # text(3) | image t=1 h=4 w=6 (-> 2x3 merged) | text(2) | video frame0 (2x2) | ts(1) | video frame1 (2x2) | text(2)
    types_row = [0, 0, 0] + [1] * 6 + [0, 0] + [2] * 4 + [0] + [2] * 4 + [0, 0]
    length = len(types_row)
    ids = np.arange(length, dtype=np.int64)[None] + 100
    mm = np.asarray([types_row], dtype=np.int64)
    # Second row: same sample shifted right by 2 with left padding.
    ids = np.concatenate([ids, np.concatenate([np.zeros((1, 2), np.int64), ids[:, :-2]], axis=1)], axis=0)
    mm = np.concatenate([mm, np.asarray([[0, 0, *types_row[:-2]]], dtype=np.int64)], axis=0)
    attn = np.stack([np.ones(length, np.int64), np.asarray([0, 0] + [1] * (length - 2), np.int64)])
    image_grid = np.asarray([[1, 4, 6], [1, 4, 6]], dtype=np.int64)
    video_grid = np.asarray([[2, 4, 4], [2, 4, 4]], dtype=np.int64)
    return ids, mm, image_grid, video_grid, attn


def test_qwen3_5_rope_index_matches_hf():
    ids, mm, image_grid, video_grid, attn = _mixed_batch()
    hf_pos, hf_delta = _hf_qwen3_5_rope_index(ids, mm, image_grid, video_grid, attn, sms=2)
    ed_pos, ed_delta = _get_rope_index_from_mm_token_types(
        input_ids=ids,
        mm_token_type_ids=mm,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
        attention_mask=attn,
        spatial_merge_size=2,
        return_jax_arrays=False,
    )
    np.testing.assert_array_equal(ed_pos, hf_pos.numpy())
    np.testing.assert_array_equal(ed_delta, hf_delta.numpy())


def test_qwen3_5_rope_index_multi_frame_image_grid_matches_hf():
    # A T>1 grid exercises the temporal axis of the meshgrid ordering.
    types_row = [0, 0] + [1] * 12 + [0]
    ids = np.arange(len(types_row), dtype=np.int64)[None]
    mm = np.asarray([types_row], dtype=np.int64)
    image_grid = np.asarray([[2, 4, 6]], dtype=np.int64)
    hf_pos, hf_delta = _hf_qwen3_5_rope_index(ids, mm, image_grid, None, None, sms=2)
    ed_pos, ed_delta = _get_rope_index_from_mm_token_types(
        input_ids=ids,
        mm_token_type_ids=mm,
        image_grid_thw=image_grid,
        spatial_merge_size=2,
        return_jax_arrays=False,
    )
    np.testing.assert_array_equal(ed_pos, hf_pos.numpy())
    np.testing.assert_array_equal(ed_delta, hf_delta.numpy())


def test_qwen3_5_rope_index_video_split_is_idempotent():
    """Callers that pre-split video grids per frame (Qwen4-Exp) get the same positions."""
    ids, mm, image_grid, video_grid, attn = _mixed_batch()
    pre_split = np.repeat(video_grid, video_grid[:, 0], axis=0)
    pre_split[:, 0] = 1
    kwargs = dict(
        input_ids=ids,
        mm_token_type_ids=mm,
        image_grid_thw=image_grid,
        attention_mask=attn,
        spatial_merge_size=2,
        return_jax_arrays=False,
    )
    a, _ = _get_rope_index_from_mm_token_types(video_grid_thw=video_grid, **kwargs)
    b, _ = _get_rope_index_from_mm_token_types(video_grid_thw=pre_split, **kwargs)
    np.testing.assert_array_equal(a, b)


def test_qwen2_vl_mrope_layout_matches_hf():
    mrope_section = [2, 3, 3]
    head_dim = 2 * sum(mrope_section)
    config = Qwen2VLTextConfig(
        hidden_size=4 * head_dim,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        rope_scaling={"type": "mrope", "mrope_section": mrope_section},
    )
    rope = config.get_basic_rope(dtype=jnp.float32, head_size=head_dim, rotary_dim=head_dim, base=config.rope_theta)
    assert rope.repetition_style

    rng = np.random.default_rng(0)
    batch, seq, heads = 2, 7, 3
    positions = rng.integers(0, 50, size=(3, batch, seq)).astype(np.int32)
    q = rng.standard_normal((batch, seq, heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((batch, seq, heads, head_dim)).astype(np.float32)

    ed_q, ed_k = rope(jnp.asarray(positions), jnp.asarray(q), jnp.asarray(k))

    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.as_tensor(positions, dtype=torch.float32)[..., None] * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    hf_q, hf_k = apply_multimodal_rotary_pos_emb(
        torch.as_tensor(q).transpose(1, 2),
        torch.as_tensor(k).transpose(1, 2),
        emb.cos(),
        emb.sin(),
        mrope_section,
    )
    np.testing.assert_allclose(np.asarray(ed_q), hf_q.transpose(1, 2).numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(ed_k), hf_k.transpose(1, 2).numpy(), rtol=1e-4, atol=1e-4)


def test_qwen2_vl_text_attention_bias_layout():
    config = Qwen2VLTextConfig(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    assert config.attention_bias is True
    attn = Qwen2VLAttention(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0), layer_idx=0)
    assert attn.query_key_value_projection.bias is not None
    assert attn.o_proj.bias is None
    assert any(key.rstrip("$").endswith("bias") for key in attn.reform_param), "HF q/k/v biases must be loaded"


def test_qwen2_vl_vision_attention_uses_config_num_heads():
    config = Qwen2VLVisionConfig(depth=1, embed_dim=32, num_heads=4, hidden_size=32, mlp_ratio=2)
    block = Qwen2VLVisionBlock(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert block.attn.num_heads == 4
    assert block.attn.head_dim == 8


_ROPE_PARAMETERS = {
    "rope_type": "default",
    "mrope_section": [2, 3, 3],
    "mrope_interleaved": True,
    "rope_theta": 5_000_000.0,
}


@pytest.mark.parametrize("config_cls", [Qwen3VLTextConfig, Qwen3VLMoeTextConfig])
def test_qwen3_vl_text_config_keeps_rope_parameters(config_cls):
    config = config_cls(num_hidden_layers=2, rope_parameters=dict(_ROPE_PARAMETERS))
    assert config.rope_theta == _ROPE_PARAMETERS["rope_theta"]
    assert config.rope_scaling["mrope_section"] == _ROPE_PARAMETERS["mrope_section"]
    assert config.rope_scaling["mrope_interleaved"] is True

    restored = config_cls.from_dict(config.to_dict())
    assert restored.rope_theta == _ROPE_PARAMETERS["rope_theta"]
    assert restored.rope_parameters["mrope_section"] == _ROPE_PARAMETERS["mrope_section"]
    assert restored.rope_parameters["mrope_interleaved"] is True
    assert restored.rope_parameters["rope_theta"] == _ROPE_PARAMETERS["rope_theta"]


def test_qwen3_vl_composite_config_keeps_text_rope_parameters():
    text = {"num_hidden_layers": 2, "rope_parameters": dict(_ROPE_PARAMETERS)}
    config = Qwen3VLConfig(text_config=text)
    assert config.text_config.rope_theta == _ROPE_PARAMETERS["rope_theta"]
    assert config.text_config.rope_scaling["mrope_section"] == _ROPE_PARAMETERS["mrope_section"]


def test_qwen3_vl_text_config_explicit_rope_scaling_still_wins():
    config = Qwen3VLTextConfig(
        num_hidden_layers=2,
        rope_theta=1234.0,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True},
    )
    assert config.rope_theta == 1234.0
    assert config.rope_scaling["mrope_section"] == [1, 1, 2]
