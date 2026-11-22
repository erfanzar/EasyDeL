"""InternVL, HyV3, InternLM2 and Kimi-Linear numerics (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import KDACacheConfig, KDACacheView
from easydel.modules.hy_v3 import HyV3Config, HyV3Model
from easydel.modules.internlm2 import InternLM2Config
from easydel.modules.internlm2.modeling_internlm2 import InternLM2Attention
from easydel.modules.internvl import InternVLConfig
from easydel.modules.internvl.modeling_internvl import InternVLModel, InternVLMultiModalProjector
from easydel.modules.kimi_linear import KimiLinearConfig
from easydel.modules.kimi_linear.modeling_kimi_linear import KimiDeltaAttention
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


@pytest.mark.parametrize("selection", [-1, [-1, -2]])
@pytest.mark.parametrize("strategy", ["default", "full"])
def test_internvl_multilayer_and_full_features_match_numpy_projection(selection, strategy):
    config = InternVLConfig(
        vision_config={
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "image_size": 8,
            "patch_size": 2,
        },
        text_config={
            "model_type": "qwen2",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 16,
            "vocab_size": 16,
        },
        vision_feature_layer=selection,
        vision_feature_select_strategy=strategy,
        projector_hidden_act="relu",
    )
    rng = np.random.default_rng(17)
    states = tuple(rng.normal(size=(4, 17, 8)).astype(np.float32) for _ in range(3))
    selected = states[-1] if selection == -1 else np.concatenate([states[index] for index in selection], axis=-1)
    grid = selected[:, 1:].reshape(4, 4, 4, -1)
    # Independent spatial-neighborhood construction in spatial row/column order.
    blocks = []
    for row in range(0, 4, 2):
        for col in range(0, 4, 2):
            blocks.append(np.concatenate([grid[:, row + dr, col + dc] for dr in range(2) for dc in range(2)], axis=-1))
    shuffled = np.stack(blocks, axis=1)
    if strategy == "full":
        shuffled = np.concatenate([np.tile(selected[:, :1], (1, 1, 4)), shuffled], axis=1)
    with config.mesh:
        projector = InternVLMultiModalProjector(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
        )
        assert projector.linear_1.weight.value.shape[0] == shuffled.shape[-1]
        tower_output = SimpleNamespace(
            last_hidden_state=jnp.asarray(states[-1]), hidden_states=tuple(map(jnp.asarray, states))
        )
        model = SimpleNamespace(
            config=config,
            vision_feature_layer=selection,
            vision_feature_select_strategy=strategy,
            vision_tower=lambda *args, **kwargs: tower_output,
            multi_modal_projector=projector,
            pixel_shuffle=InternVLModel.pixel_shuffle,
        )
        actual = InternVLModel.get_image_features(model, jnp.zeros((4, 3, 8, 8)))
        compiled = jax.jit(lambda pixels: InternVLModel.get_image_features(model, pixels))(jnp.zeros((4, 3, 8, 8)))
        norm = projector.layer_norm
        centered = shuffled - shuffled.mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + norm.epsilon)
        normalized = normalized * np.asarray(norm.weight.value) + np.asarray(norm.bias.value)
        expected = np.maximum(
            normalized @ np.asarray(projector.linear_1.weight.value) + np.asarray(projector.linear_1.bias.value), 0
        )
        expected = expected @ np.asarray(projector.linear_2.weight.value) + np.asarray(projector.linear_2.bias.value)
    assert actual.shape == (4, 5 if strategy == "full" else 4, 8)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(compiled, expected, atol=1e-6, rtol=1e-4)


def test_hy_v3_hidden_states_include_normalized_final_layer():
    config = HyV3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        mlp_layer_types=["dense", "dense"],
        attn_mechanism="vanilla",
    )
    with config.mesh:
        model = HyV3Model(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
        )
        ids = jnp.tile(jnp.arange(4)[None], (4, 1))
        output = model(input_ids=ids, output_hidden_states=True)
        embeddings = model.embed_tokens(ids)
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    np.testing.assert_array_equal(output.hidden_states[0], embeddings)
    np.testing.assert_array_equal(output.hidden_states[-1], output.last_hidden_state)
    assert output.hidden_states[config.num_hidden_layers].shape == (4, 4, 16)


def test_internlm2_attention_dropout_changes_training_but_not_evaluation():
    config = InternLM2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_dropout=0.5,
        max_position_embeddings=16,
        attn_mechanism="vanilla",
    )
    with config.mesh:
        attention = InternLM2Attention(
            config,
            layer_idx=0,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        hidden = jax.random.normal(jax.random.PRNGKey(3), (4, 8, 16))
        positions = jnp.tile(jnp.arange(8)[None], (4, 1))

        def forward():
            return attention(
                hidden_states=hidden, mask_info=None, position_ids=positions, mode=common_types.MODE_TRAIN
            ).attention_output

        attention.eval()
        first_eval, second_eval = forward(), forward()
        attention.train()
        first_train, second_train = forward(), forward()
        attention.eval()
        third_eval = forward()
    np.testing.assert_array_equal(first_eval, second_eval)
    np.testing.assert_array_equal(first_eval, third_eval)
    assert not np.array_equal(first_train, second_train)
    assert not np.array_equal(first_train, first_eval)


@pytest.mark.parametrize("padded", [False, True])
def test_kimi_dense_kda_cached_multitoken_chunks_match_full_prefill(padded):
    config = KimiLinearConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "chunk_size": 4,
        },
    )
    with config.mesh:
        attention = KimiDeltaAttention(
            config, 0, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
        )
        hidden = jax.random.normal(jax.random.PRNGKey(12), (4, 8, 16))
        mask = jnp.ones((4, 8), dtype=jnp.bool_)
        if padded:
            mask = mask.at[1, 6:].set(False).at[2, :2].set(False).at[3, :1].set(False)
        full_mask = MaskInfo.dynamic_init(inputs_embeds=hidden, attention_mask=mask)
        full = attention(hidden, mask_info=full_mask)
        cache_config = KDACacheConfig.create(
            num_hidden_layers=1,
            partition_axis=config.partition_axis,
            batch_size=4,
            key_dim=16,
            value_dim=16,
            d_conv=4,
            recurrent_state_shape=(2, 8, 8),
        )
        cache = KDACacheView.init(cache_config, dtype=jnp.float32)
        whole_cached = attention(hidden, mask_info=full_mask, cache_view=cache)
        pieces = []
        for start in (0, 4):
            chunk = hidden[:, start : start + 4]
            chunk_mask = MaskInfo.dynamic_init(inputs_embeds=chunk, attention_mask=mask[:, start : start + 4])
            result = attention(chunk, mask_info=chunk_mask, cache_view=cache)
            cache = result.cache_view
            pieces.append(result.attention_output)
        padding = hidden[:, :4]
        padding_mask = MaskInfo.dynamic_init(inputs_embeds=padding, attention_mask=jnp.zeros((4, 4), dtype=jnp.bool_))
        after_padding = attention(padding, mask_info=padding_mask, cache_view=cache)
    np.testing.assert_allclose(jnp.concatenate(pieces, axis=1), full.attention_output, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(whole_cached.attention_output, full.attention_output, atol=1e-5, rtol=1e-4)
    for name in ("q_conv_state", "k_conv_state", "v_conv_state", "recurrent_state"):
        np.testing.assert_allclose(getattr(cache, name), getattr(whole_cached.cache_view, name), atol=1e-5, rtol=1e-4)
        np.testing.assert_array_equal(getattr(after_padding.cache_view, name), getattr(cache, name))
    np.testing.assert_array_equal(after_padding.attention_output, np.zeros((4, 4, 16), dtype=np.float32))
