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

"""Cross-layer DSA selection must follow the config's full/shared schedule."""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from ejkernel.types import MaskInfo
from spectrax import common_types


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
def test_shared_indexer_reuses_previous_selection(family):
    """Changing the prior selection changes shared attention, not a full layer."""
    common = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=2,
        index_head_dim=8,
        index_n_heads=2,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "dense"],
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    if family == "glm5_next":
        config = ed.Glm5NextTextConfig(
            **common,
            qk_rope_head_dim=0,
            index_kpool=1,
            layer_types=["deepseek_sparse_attention"] * 2,
        )
        model_type = ed.Glm5NextForCausalLM
    else:
        config = ed.GlmMoeDsaConfig(**common, qk_rope_head_dim=8)
        model_type = ed.GlmMoeDsaForCausalLM

    with config.mesh:
        model = model_type(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(11))
        layer = model.model.layers[1].self_attn
        hidden = jax.random.normal(jax.random.key(12), (1, 8, 32))
        mask = MaskInfo.dynamic_init(input_ids=jnp.ones((1, 8), jnp.int32))
        positions = jnp.arange(8)[None, :]
        zero = jnp.zeros((1, 8, 2), jnp.int32)
        self_only = jnp.broadcast_to(positions[..., None], zero.shape)

        def attend(previous):
            return layer.forward_mla(
                hidden,
                mask,
                positions,
                common_types.MODE_TRAIN,
                frequencies=model.model.frequencies,
                previous_topk_indices=previous,
                return_topk_indices=True,
            )

        first, selected_first = attend(zero)
        second, selected_second = attend(self_only)
        np.testing.assert_array_equal(np.asarray(selected_first), np.asarray(zero))
        np.testing.assert_array_equal(np.asarray(selected_second), np.asarray(self_only))
        assert not np.allclose(np.asarray(first.attention_output), np.asarray(second.attention_output), atol=1e-5)
        with pytest.raises(ValueError, match="preceding full layer"):
            attend(None)
        # Exercise the real stack carry, not only the attention layer in isolation.
        logits = model(input_ids=jnp.arange(1, 9, dtype=jnp.int32)[None, :]).logits
        assert logits.shape == (1, 8, config.vocab_size)
        assert bool(jnp.isfinite(logits).all())


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
def test_full_schedule_ignores_previous_selection(family):
    """An all-full config continues computing its own selection regardless of carry."""
    common = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=2,
        index_head_dim=8,
        index_n_heads=2,
        indexer_types=["full"],
        mlp_layer_types=["dense"],
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    if family == "glm5_next":
        config = ed.Glm5NextTextConfig(
            **common,
            qk_rope_head_dim=0,
            index_kpool=1,
            layer_types=["deepseek_sparse_attention"],
        )
        model_type = ed.Glm5NextForCausalLM
    else:
        config = ed.GlmMoeDsaConfig(**common, qk_rope_head_dim=8)
        model_type = ed.GlmMoeDsaForCausalLM

    with config.mesh:
        model = model_type(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(13))
        layer = model.model.layers[0].self_attn
        hidden = jax.random.normal(jax.random.key(14), (1, 8, 32))
        mask = MaskInfo.dynamic_init(input_ids=jnp.ones((1, 8), jnp.int32))
        positions = jnp.arange(8)[None, :]
        first, first_indices = layer.forward_mla(
            hidden,
            mask,
            positions,
            common_types.MODE_TRAIN,
            frequencies=model.model.frequencies,
            return_topk_indices=True,
        )
        second, second_indices = layer.forward_mla(
            hidden,
            mask,
            positions,
            common_types.MODE_TRAIN,
            frequencies=model.model.frequencies,
            previous_topk_indices=jnp.full((1, 8, 2), 7, jnp.int32),
            return_topk_indices=True,
        )
        np.testing.assert_array_equal(np.asarray(first_indices), np.asarray(second_indices))
        np.testing.assert_array_equal(np.asarray(first.attention_output), np.asarray(second.attention_output))
