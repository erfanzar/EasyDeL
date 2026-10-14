# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-Omni vision DeepStack and masked text residual regression tests."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextModel,
    Qwen3OmniMoeVisionEncoder,
)
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import (
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeVisionConfig,
)
from jax import numpy as jnp


def test_omni_vision_adds_hf_interpolated_positions_before_blocks():
    torch = pytest.importorskip("torch")
    vision_utils = pytest.importorskip("transformers.vision_utils")
    config = Qwen3OmniMoeVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=16,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
        scan_layers=True,
    )
    model = Qwen3OmniMoeVisionEncoder(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    grid = np.array([[1, 4, 2], [2, 2, 4]], dtype=np.int32)
    pixel_values = np.random.default_rng(0).standard_normal((24, 12)).astype(np.float32)

    indices, weights = vision_utils.get_vision_bilinear_indices_and_weights(
        torch.from_numpy(grid), num_grid_per_side=4, spatial_merge_size=2
    )
    table = torch.from_numpy(np.asarray(model.pos_embed.weight.value))
    expected_positions = (table[indices] * weights[:, :, None]).sum(0).numpy()
    np.testing.assert_allclose(np.asarray(model.fast_pos_embed_interpolate(grid)), expected_positions, atol=1e-6)

    patch_states = model.patch_embed(pixel_values)
    rope = model.rot_pos_emb(grid, max_grid_size=4)
    position_ids = vision_utils.get_vision_position_ids(torch.from_numpy(grid), spatial_merge_size=2).numpy()
    inv_freq = 1.0 / (10000 ** (np.arange(0, model._head_dim_ro, 2, dtype=np.float32) / model._head_dim_ro))
    expected_rope = (position_ids[:, :, None] * inv_freq[None, None, :]).reshape(len(position_ids), -1)
    np.testing.assert_allclose(np.asarray(rope), expected_rope, atol=1e-6)
    cu_seqlens = jnp.array([0, 8, 16, 24], dtype=jnp.int32)
    full_rope = jnp.concatenate([rope, rope], axis=-1)  # blocks take the HF cat(freqs, freqs) table
    expected_states = model.blocks[0](patch_states + jnp.asarray(expected_positions), cu_seqlens, full_rope)
    expected_merged = model.merger(expected_states)
    merged, deepstack = model(pixel_values, grid, max_grid_size=4)
    np.testing.assert_allclose(np.asarray(merged), np.asarray(expected_merged), rtol=1e-5, atol=1e-5)
    assert deepstack == []


def test_omni_vision_returns_deepstack_features_when_scan_configured():
    config = Qwen3OmniMoeVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        patch_size=4,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=16,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1],
        scan_layers=True,
    )
    model = Qwen3OmniMoeVisionEncoder(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    pixel_values = np.random.default_rng(0).standard_normal((16, 48)).astype(np.float32)
    grid_thw = np.array([[1, 4, 4]], dtype=np.int32)
    merged, deepstack = model(pixel_values, grid_thw, max_grid_size=8)
    assert merged.shape == (4, 16)
    assert len(deepstack) == 2
    assert all(feature.shape == merged.shape for feature in deepstack)
    assert np.isfinite(np.asarray(deepstack[0])).all()
    assert not np.allclose(np.asarray(deepstack[0]), np.asarray(deepstack[1]))


def test_omni_empty_deepstack_list_runs_under_scan_layers():
    """An empty DeepStack list (deepstack_visual_indexes=[]) keeps the lax.scan path and must not
    evaluate a traced layer index as a Python bool (r10 S10-3)."""
    config = Qwen3OmniMoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        decoder_sparse_step=2,
        mlp_only_layers=[0, 1],
        scan_layers=True,
    )
    model = Qwen3OmniMoeThinkerTextModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    embeddings = jnp.asarray(np.random.default_rng(2).standard_normal((1, 3, 16)), dtype=jnp.float32)
    mask = jnp.array([[False, True, False]])
    baseline = np.asarray(model(inputs_embeds=embeddings).last_hidden_state)
    out = np.asarray(
        model(inputs_embeds=embeddings, visual_pos_masks=mask, deepstack_visual_embeds=[]).last_hidden_state
    )
    np.testing.assert_allclose(out, baseline, rtol=1e-5, atol=1e-5)


def test_omni_deepstack_changes_only_visual_tokens_after_first_text_layer():
    config = Qwen3OmniMoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        decoder_sparse_step=2,
        mlp_only_layers=[0],
        scan_layers=True,
    )
    model = Qwen3OmniMoeThinkerTextModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    embeddings = jnp.asarray(np.random.default_rng(1).standard_normal((1, 3, 16)), dtype=jnp.float32)
    mask = jnp.array([[False, True, False]])
    visual_features = jnp.ones((1, 16), dtype=jnp.float32)
    baseline = np.asarray(model(inputs_embeds=embeddings).last_hidden_state)
    with_visual = np.asarray(
        model(
            inputs_embeds=embeddings,
            visual_pos_masks=mask,
            deepstack_visual_embeds=[visual_features],
        ).last_hidden_state
    )
    np.testing.assert_allclose(with_visual[:, [0, 2], :], baseline[:, [0, 2], :], rtol=1e-5, atol=1e-5)
    assert not np.allclose(with_visual[:, 1, :], baseline[:, 1, :])
