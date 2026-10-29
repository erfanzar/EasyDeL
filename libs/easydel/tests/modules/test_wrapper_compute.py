"""Task-wrapper array-computation regressions (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base import BaseCausalLMModule, BaseConditionalGenerationModule, BaseVisionLanguageModule
from easydel.modules.arctic import ArcticConfig, ArcticForSequenceClassification, ArcticModel
from easydel.modules.arctic.modeling_arctic import ArcticMoeBlock
from jax import numpy as jnp


def tiny_config(**kwargs):
    values = dict(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        # The fused expert axis includes all four FSDP devices on the test TPU.
        num_local_experts=4,
        num_experts_per_tok=1,
        moe_layer_frequency=2,
        max_position_embeddings=16,
    )
    values.update(kwargs)
    return ArcticConfig(**values)


class FixtureBackbone(spx.Module):
    """A deterministic backbone exposing exactly the public task-wrapper inputs/outputs."""

    def __init__(self, config, **kwargs):
        self.config = config

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden = inputs_embeds
        if hidden is None:
            hidden = jnp.repeat(input_ids[..., None].astype(jnp.float32), self.config.hidden_size, axis=-1)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=None,
            attentions=None,
            past_key_values="updated-cache",
            router_logits=(jnp.zeros((hidden.shape[0] * hidden.shape[1], 2)),),
            image_hidden_states=jnp.full((1, 2, self.config.hidden_size), 3.0),
            video_hidden_states=jnp.full((1, 4, self.config.hidden_size), 7.0),
        )


class FixtureVLM(BaseVisionLanguageModule):
    _auto_register = False
    _supports_video = True
    _uses_mrope = True

    def get_image_features(self, pixel_values, **kwargs):
        return pixel_values


def make_vlm(**kwargs):
    return FixtureVLM(
        config=tiny_config(),
        base_model_class=FixtureBackbone,
        create_lm_head=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
        **kwargs,
    )


def test_vlm_returns_video_features_and_scaled_router_loss():
    model = make_vlm(router_aux_loss_coef=0.25)
    out = model(input_ids=jnp.array([[1, 2, 3]]), apply_lm_head=False)
    # A uniform two-expert, top-1 router has independent Switch loss 1.0.
    np.testing.assert_allclose(out.aux_loss, 0.25)
    np.testing.assert_array_equal(out.image_hidden_states, np.full((1, 2, 8), 3.0))
    np.testing.assert_array_equal(out.video_hidden_states, np.full((1, 4, 8), 7.0))
    assert out.past_key_values == "updated-cache"


def test_vlm_per_layer_router_losses_take_precedence_and_are_scaled_once():
    model = make_vlm(router_aux_loss_coef=0.25)
    out = SimpleNamespace(all_router_losses=(jnp.array(2.0), jnp.array(4.0)), router_logits=None)
    np.testing.assert_allclose(model.compute_router_aux_loss(out), 1.5)


@pytest.mark.parametrize("wrapper", [BaseCausalLMModule, BaseConditionalGenerationModule])
@pytest.mark.parametrize("constructor_tie,config_tie", [(True, False), (False, True), (False, False)])
def test_lm_head_respects_constructor_and_config_tying(wrapper, constructor_tie, config_tie):
    model = wrapper(
        config=tiny_config(tie_word_embeddings=config_tie),
        base_model_class=ArcticModel,
        tie_word_embeddings=constructor_tie,
        precision=jax.lax.Precision.HIGHEST,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(1),
    )
    hidden = jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 8) / 16
    actual = model.apply_lm_head(hidden)
    if constructor_tie or config_tie:
        expected = np.asarray(hidden) @ np.asarray(model.get_embedding().weight.value).T
    else:
        expected = model.get_lm_head()(hidden)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.compute_lm_logits(hidden), expected, rtol=1e-5, atol=1e-6)


def test_arctic_classifier_uses_mask_and_returns_updated_cache():
    model = ArcticForSequenceClassification(
        config=tiny_config(num_labels=2, pad_token_id=0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(2),
    )
    # Isolate wrapper behavior while preserving its real classifier projection.
    model.model = FixtureBackbone(model.config)
    ids = jnp.array([[1, 2, 3, 4], [0, 5, 6, 0]])
    mask = jnp.array([[1, 1, 0, 0], [0, 1, 1, 0]])
    out = model(input_ids=ids, attention_mask=mask, past_key_values="old-cache")
    last_values = jnp.repeat(jnp.array([2.0, 6.0])[:, None], 8, axis=-1)
    np.testing.assert_allclose(out.logits, model.score(last_values), rtol=1e-5, atol=1e-6)
    assert out.past_key_values == "updated-cache"


@pytest.mark.parametrize("method", ["fused_moe", "dense_moe", "standard_moe"])
def test_arctic_capacity_train_eval_minimum_and_drop_toggle(method):
    config = tiny_config(
        moe_layer_frequency=1,
        moe_method=method,
        moe_train_capacity_factor=2.0,
        moe_eval_capacity_factor=4.0,
    )
    config.moe_force_xla_gmm = True
    block = ArcticMoeBlock(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3), layer_idx=0)
    # Equal scores send all four tokens to the same expert, forcing overflow.
    block.gate.weight.value = jnp.zeros_like(block.gate.weight.value)
    hidden = jnp.ones((1, 4, 8), dtype=jnp.float32)
    config.moe_token_dropping = False
    baseline, _ = block(hidden)
    assert np.any(np.abs(np.asarray(baseline)) > 0)
    config.moe_token_dropping = True
    block.train()
    train, _ = block(hidden)
    np.testing.assert_allclose(train[:, :2], baseline[:, :2], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(train[:, 2:], np.zeros((1, 2, 8)))
    block.eval()
    evaluation, _ = block(hidden)
    np.testing.assert_allclose(evaluation, baseline, rtol=1e-5, atol=1e-7)
    block.train()
    config.moe_min_capacity = 3
    minimum, _ = block(hidden)
    np.testing.assert_allclose(minimum[:, :3], baseline[:, :3], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(minimum[:, 3:], np.zeros((1, 1, 8)))


def test_video_and_mrope_settings_drive_public_position_helpers():
    model = make_vlm(temporal_patch_size=4, tokens_per_second=3.0, spatial_merge_size=2)
    assert model.compute_video_grid(8, 16, 16, 4) == (2, 2, 2)
    temporal = model.get_temporal_positions(jnp.array([[2, 2, 2]]), second_per_grid_ts=[2.0])
    np.testing.assert_array_equal(temporal, [0, 0, 0, 0, 6, 6, 6, 6])
    t, h, w = model.compute_vision_positions((2, 4, 4), start_position=5)
    np.testing.assert_array_equal(t, [5, 5, 5, 5, 6, 6, 6, 6])
    np.testing.assert_array_equal(h, [5, 5, 6, 6, 5, 5, 6, 6])
    np.testing.assert_array_equal(w, [5, 6, 5, 6, 5, 6, 5, 6])


def test_vlm_multimodal_mask_extension_recognizes_both_modalities():
    model = make_vlm(image_token_index=3, video_token_index=4)
    np.testing.assert_array_equal(model._get_multimodal_mask(jnp.array([[1, 3, 2, 4]])), [[False, True, False, True]])


def test_constructor_mrope_allocation_changes_backbone_frequencies():
    from easydel.layers.rotary import MultiModalRotaryEmbedding

    class RotaryBackbone(FixtureBackbone):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.rotary = MultiModalRotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=16,
                base=10000.0,
                is_neox_style=True,
                dtype=jnp.float32,
                mrope_section=(2, 1, 1),
                mrope_interleaved=False,
                repetition_style=True,
            )

    def make_rotary_vlm(**kwargs):
        return FixtureVLM(
            config=tiny_config(),
            base_model_class=RotaryBackbone,
            create_lm_head=False,
            rngs=spx.Rngs(0),
            **kwargs,
        )

    default = make_rotary_vlm()
    custom = make_rotary_vlm(mrope_section=(1, 2, 1))
    assert default.base_model.rotary.mrope_section == (2, 1, 1)
    assert custom.base_model.rotary.mrope_section == (1, 2, 1)
    positions = jnp.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    cosine, sine = custom.base_model.rotary.compute_cos_sin(positions, dtype=jnp.float32)
    inverse_frequencies = 10000.0 ** (-np.arange(0, 8, 2, dtype=np.float32) / 8)
    # Repeated half-width THW sections select T0, H1, H2, W3
    # in both halves, independently of the production rotary implementation.
    axes = [0, 1, 1, 2]
    phases = np.stack([np.asarray(positions)[axis] * inverse_frequencies[i] for i, axis in enumerate(axes)], axis=-1)
    phases = np.concatenate([phases, phases], axis=-1)
    np.testing.assert_allclose(cosine, np.cos(phases), atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(sine, np.sin(phases), atol=1e-6, rtol=1e-5)
    default_cosine, _ = default.base_model.rotary.compute_cos_sin(positions, dtype=jnp.float32)
    assert not np.allclose(default_cosine, cosine)


@pytest.mark.parametrize("chunk_size", [2, 3])
@pytest.mark.parametrize("top_k", [1, 2])
def test_arctic_capacity_is_invariant_under_moe_chunking(chunk_size, top_k):
    config = tiny_config(
        moe_layer_frequency=1,
        moe_method="fused_moe",
        num_experts_per_tok=top_k,
        # Keep the raw capacity below the floor for both one and two routes;
        # selection must count the full block's assignments before chunking.
        moe_train_capacity_factor=2.0 / top_k,
        moe_min_capacity=3,
    )
    config.moe_force_xla_gmm = True
    block = ArcticMoeBlock(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3), layer_idx=0)
    block.gate.weight.value = jnp.zeros_like(block.gate.weight.value)
    hidden = jnp.ones((1, 4, 8), dtype=jnp.float32)
    block.train()
    config.moe_chunk_size = None
    full, _ = block(hidden)
    config.moe_chunk_size = chunk_size
    chunked, _ = block(hidden)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-7)
    assert np.any(np.abs(np.asarray(chunked[:, :3])) > 0)
    np.testing.assert_array_equal(chunked[:, 3:], np.zeros((1, 1, 8)))


def test_base_scalar_clipping_stage_matches_full_logit_cap():
    model = BaseCausalLMModule(
        config=tiny_config(),
        base_model_class=ArcticModel,
        logit_cap=0.25,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(1),
    )
    hidden = jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 8)
    expected = model.compute_lm_logits(hidden)
    np.testing.assert_allclose(model.make_lm_head_fn()(hidden), expected, atol=1e-6, rtol=1e-5)
    clip = model.get_base_lm_logit_postprocess()
    np.testing.assert_allclose(clip(model.apply_lm_head(hidden)), expected, atol=1e-6, rtol=1e-5)
