# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Public GLM DSA attention and KDA checkpoint contracts (accelerator only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.glm5_next.glm5_next_configuration import Glm5NextTextConfig
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextDSAAttention, Glm5NextLinearAttention
from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention
from ejkernel.types import MaskInfo
from jax import lax
from jax import numpy as jnp
from spectrax.common_types import MODE_TRAIN


def _config(family, **overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        attention_dropout=0.0,
        output_attentions=False,
        n_routed_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    if family == "glm5_next":
        kwargs.update(qk_rope_head_dim=0, index_kpool=2, layer_types=["deepseek_sparse_attention"])
        config_type = Glm5NextTextConfig
    else:
        kwargs.update(qk_rope_head_dim=2)
        config_type = GlmMoeDsaConfig
    kwargs.update(overrides)
    config = config_type(**kwargs)
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    return config


def _layer(family, config):
    layer_type = Glm5NextDSAAttention if family == "glm5_next" else GlmMoeDsaAttention
    return layer_type(
        config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )


def _inputs():
    hidden = jnp.asarray(np.sin(np.arange(128, dtype=np.float32)).reshape(1, 8, 16))
    positions = jnp.arange(8)[None, :]
    mask = MaskInfo.from_segments(jnp.zeros((1, 8), dtype=jnp.int32))
    return hidden, positions, mask


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
@pytest.mark.parametrize("config_output_attentions", [False, True])
def test_per_call_attention_output_overrides_config(family, config_output_attentions):
    config = _config(family, output_attentions=config_output_attentions, dsa_sparse_training=False)
    with config.mesh:
        layer = _layer(family, config)
        hidden, positions, mask = _inputs()
        visible = layer(hidden, mask, positions, MODE_TRAIN, output_attentions=True)
        hidden_weights = layer(hidden, mask, positions, MODE_TRAIN, output_attentions=False)
    weights = np.asarray(visible.attention_weight)
    assert weights.shape == (1, 2, 8, 8)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(weights[..., np.triu_indices(8, k=1)[0], np.triu_indices(8, k=1)[1]] == 0)
    assert hidden_weights.attention_weight is None
    np.testing.assert_array_equal(np.asarray(visible.attention_output), np.asarray(hidden_weights.attention_output))


@pytest.mark.parametrize("family", ["glm5_next", "glm_moe_dsa"])
def test_force_sparse_preserves_training_dropout_and_eval_determinism(family):
    config = _config(family, attention_dropout=0.5, dsa_sparse_training=True)
    with config.mesh:
        forced = _layer(family, config)
        dense = _layer(family, config)
        # Both layers start from identical parameters and independent identical
        # dropout RNG streams. Change only the policy between the calls.
        hidden, positions, mask = _inputs()
        config.dsa_sparse_training = True
        first = forced(hidden, mask, positions, MODE_TRAIN).attention_output
        second = forced(hidden, mask, positions, MODE_TRAIN).attention_output
        config.dsa_sparse_training = False
        expected_first = dense(hidden, mask, positions, MODE_TRAIN).attention_output
        expected_second = dense(hidden, mask, positions, MODE_TRAIN).attention_output
        forced.eval()
        inference_first = forced(hidden, mask, positions, MODE_TRAIN).attention_output
        inference_second = forced(hidden, mask, positions, MODE_TRAIN).attention_output
    np.testing.assert_array_equal(np.asarray(first), np.asarray(expected_first))
    np.testing.assert_array_equal(np.asarray(second), np.asarray(expected_second))
    assert not np.array_equal(np.asarray(first), np.asarray(second))
    np.testing.assert_array_equal(np.asarray(inference_first), np.asarray(inference_second))


def test_kda_convolution_checkpoint_parameters_honor_storage_dtype():
    config = _config(
        "glm5_next",
        layer_types=["linear_attention"],
        linear_num_heads=2,
        linear_head_dim=4,
        linear_conv_kernel_dim=3,
    )
    with config.mesh:
        layer = Glm5NextLinearAttention(
            config,
            layer_idx=0,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        graph, state = spx.export(layer)
        restored = spx.bind(graph, state)
    parameter_dtypes = {
        path: getattr(leaf, "value", leaf).dtype
        for collection, path, leaf in state.items()
        if collection == "parameters"
    }
    for projection in ("q", "k", "v"):
        assert parameter_dtypes[f"{projection}_conv1d.weight"] == jnp.float32
        assert getattr(restored, f"{projection}_conv1d").weight.value.dtype == jnp.float32
    assert restored.dtype == jnp.bfloat16
