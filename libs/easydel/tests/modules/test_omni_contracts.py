"""Public Omni generation, cached-output, and precomputed-embedding regressions."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_omni_moe import modeling_qwen3_omni_moe as omni
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeThinkerConfig,
)
from jax import numpy as jnp


@pytest.mark.cpu_ok
def test_predictor_wrapper_preserves_returned_cache_without_array_execution(monkeypatch):
    cache, states, logits = object(), object(), object()
    trunk_output = SimpleNamespace(last_hidden_state=states, past_key_values=cache, hidden_states=None, attentions=None)
    wrapper = SimpleNamespace(model=lambda **kwargs: trunk_output, lm_head=[lambda inputs: logits])
    monkeypatch.setattr(omni.jnp, "stack", lambda values, axis: values[0])
    output = omni.Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration.forward(
        wrapper, inputs_embeds=states, past_key_values=object()
    )
    assert output.past_key_values is cache
    assert output.logits is logits


@pytest.mark.cpu_ok
@pytest.mark.parametrize("media", ["image_grid_thw", "video_grid_thw", "audio_seqlens"])
def test_generation_preparation_routes_omni_positions_without_text_trunk_hook(media):
    tokens, grid, positions, deltas, cache = (object() for _ in range(5))
    trunk = SimpleNamespace(prepare_inputs_for_generation=lambda **kwargs: {"past_key_values": cache})
    thinker = SimpleNamespace(base_model=trunk, get_rope_index=lambda **kwargs: (positions, deltas))
    result = omni.Qwen3OmniMoeThinkerForConditionalGeneration.prepare_inputs_for_generation(
        thinker, tokens, max_length=16, pad_token_id=0, **{media: grid}
    )
    assert result["past_key_values"] is cache
    assert result["position_ids"] is positions
    assert result["rope_deltas"] is deltas
    assert result[media] is grid


@pytest.mark.cpu_ok
def test_outer_wrapper_preserves_embedding_info_and_forward_residuals():
    embeds, info, visual_mask, residuals, output = (object() for _ in range(5))

    class Thinker:
        def compute_embedding_with_info(self, input_ids, **kwargs):
            return embeds, info

        def __call__(self, **kwargs):
            assert kwargs["inputs_embeds"] is embeds
            assert kwargs["visual_pos_masks"] is visual_mask
            assert kwargs["deepstack_visual_embeds"] is residuals
            return output

    wrapper = SimpleNamespace(thinker=Thinker())
    assert omni.Qwen3OmniMoeForConditionalGeneration.compute_embedding_with_info(wrapper, object()) == (embeds, info)
    assert (
        omni.Qwen3OmniMoeForConditionalGeneration.forward(
            wrapper, inputs_embeds=embeds, visual_pos_masks=visual_mask, deepstack_visual_embeds=residuals
        )
        is output
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("media", ["image", "video"])
def test_feature_getters_keep_array_default_and_offer_lossless_residual_bundle(media):
    features, residuals = object(), [object()]
    holder = SimpleNamespace(visual=lambda *args: (features, residuals))
    method = (
        omni.Qwen3OmniMoeThinkerForConditionalGeneration.get_image_features
        if media == "image"
        else omni.Qwen3OmniMoeThinkerForConditionalGeneration.get_video_features
    )
    kwargs = {f"{media}_max_grid_size": 2}
    assert method(holder, object(), object(), **kwargs) is features
    returned_features, returned_residuals = method(holder, object(), object(), return_deepstack=True, **kwargs)
    assert returned_features == (features,)
    assert returned_residuals is residuals


def _thinker_config():
    return Qwen3OmniMoeThinkerConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            mlp_only_layers=[0],
            max_position_embeddings=128,
            attn_dtype=jnp.float32,
            kvdtype=jnp.float32,
            attn_mechanism="vanilla",
            scan_layers=False,
        ),
        vision_config=dict(
            hidden_size=8,
            intermediate_size=16,
            out_hidden_size=16,
            depth=1,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
            attn_mechanism="vanilla",
            scan_layers=False,
        ),
        audio_config=dict(
            num_mel_bins=8,
            d_model=8,
            encoder_layers=1,
            encoder_attention_heads=2,
            encoder_ffn_dim=16,
            downsample_hidden_size=2,
            output_dim=16,
            n_window=50,
            n_window_infer=100,
            attn_mechanism="vanilla",
            scan_layers=False,
        ),
        image_token_id=60,
        video_token_id=61,
        audio_token_id=62,
        vision_start_token_id=57,
        vision_end_token_id=59,
        audio_start_token_id=58,
    )


@pytest.mark.parametrize("kind", ["image", "video", "audio_video", "mixed"])
def test_public_rope_index_matches_hf_with_padding_and_timed_audio_video(kind):
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    config = _thinker_config()
    holder = SimpleNamespace(config=config, spatial_merge_size=2)
    holder.get_llm_pos_ids_for_vision = lambda *args: (
        hf.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_llm_pos_ids_for_vision(holder, *args)
    )
    # Four merged spatial tokens for each of two video frames. The audio/video
    # fixture has two audio tokens; ties must put the first video frame first.
    images = np.array([[1, 4, 4]], dtype=np.int32) if kind in ("image", "mixed") else None
    videos = np.array([[2, 4, 4]], dtype=np.int32) if kind != "image" else None
    if kind == "image":
        tokens = [1, 57, *([60] * 4), 59, 2]
    elif kind == "video":
        tokens = [1, 57, *([61] * 8), 59, 2]
    elif kind == "audio_video":
        tokens = [1, 57, 58, *([61] * 4), 62, 62, *([61] * 4), 63, 59, 2]
    else:
        tokens = [1, 58, 62, 62, 63, 57, *([60] * 4), 59, 57, *([61] * 8), 59, 2]
    ids = np.array([[0, 0, *tokens]], dtype=np.int32)
    mask = ids != 0
    audio = np.array([16], dtype=np.int32) if kind in ("audio_video", "mixed") else None
    kwargs = dict(
        input_ids=torch.tensor(ids),
        image_grid_thw=None if images is None else torch.tensor(images),
        video_grid_thw=None if videos is None else torch.tensor(videos),
        attention_mask=torch.tensor(mask),
        use_audio_in_video=kind == "audio_video",
        audio_seqlens=None if audio is None else torch.tensor(audio),
        second_per_grids=None if videos is None else torch.tensor([0.5]),
    )
    expected = hf.Qwen3OmniMoePreTrainedModelForConditionalGeneration.get_rope_index(holder, **kwargs)
    actual = omni.Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index(
        holder,
        jnp.asarray(ids),
        None if images is None else jnp.asarray(images),
        None if videos is None else jnp.asarray(videos),
        jnp.asarray(mask),
        kind == "audio_video",
        None if audio is None else jnp.asarray(audio),
        None if videos is None else jnp.array([0.5]),
    )
    traced = jax.jit(
        lambda token_ids: omni.Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index(
            holder,
            token_ids,
            None if images is None else jnp.asarray(images),
            None if videos is None else jnp.asarray(videos),
            jnp.asarray(mask),
            kind == "audio_video",
            None if audio is None else jnp.asarray(audio),
            None if videos is None else jnp.array([0.5]),
        )
    )(jnp.asarray(ids))
    for result in (actual, traced):
        for value, reference in zip(result, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(value), reference.numpy())


@pytest.mark.parametrize("media", ["image", "video"])
def test_raw_and_precomputed_visual_forward_keep_nonzero_deepstack_residuals(media):
    config = _thinker_config()
    with jax.default_matmul_precision("highest"):
        model = omni.Qwen3OmniMoeThinkerForConditionalGeneration(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(4)
        )

        class Vision(spx.Module):
            def forward(self, pixels, grid, max_grid):
                features = jnp.arange(16, dtype=jnp.float32)[None, :] / 16
                return features, [jnp.cos(features * 3)]

        model.visual = Vision()
        token = config.image_token_id if media == "image" else config.video_token_id
        ids = jnp.array([[1, 57, token, 59, 2]])
        kwargs = (
            dict(pixel_values=jnp.ones((4, 12)), image_grid_thw=jnp.array([[1, 2, 2]]), image_max_grid_size=2)
            if media == "image"
            else dict(
                pixel_values_videos=jnp.ones((4, 12)), video_grid_thw=jnp.array([[1, 2, 2]]), video_max_grid_size=2
            )
        )
        embeddings, info = model.compute_embedding_with_info(ids, **kwargs)
        cached_features, cached_residuals = getattr(model, f"get_{media}_features")(return_deepstack=True, **kwargs)
        cache_kwargs = {key: value for key, value in kwargs.items() if not key.startswith("pixel_values")}
        cache_kwargs[f"{media}_embeds"] = cached_features[0]
        cache_kwargs[f"deepstack_{media}_embeds"] = cached_residuals
        cached_embeddings, cached_info = model.compute_embedding_with_info(ids, **cache_kwargs)
        np.testing.assert_array_equal(np.asarray(cached_embeddings), np.asarray(embeddings))
        for actual_layer, expected_layer in zip(
            cached_info.deepstack_visual_embeds, info.deepstack_visual_embeds, strict=True
        ):
            np.testing.assert_array_equal(np.asarray(actual_layer), np.asarray(expected_layer))
        raw = model(input_ids=ids, apply_lm_head=False, **kwargs).last_hidden_state
        replay = model(
            inputs_embeds=embeddings,
            position_ids=info.position_ids,
            visual_pos_masks=info.visual_pos_masks,
            deepstack_visual_embeds=info.deepstack_visual_embeds,
            apply_lm_head=False,
        ).last_hidden_state
        aligned = [jnp.zeros_like(embeddings).at[:, 2, :].set(layer[0]) for layer in info.deepstack_visual_embeds]
        dense_replay = model(
            inputs_embeds=embeddings,
            position_ids=info.position_ids,
            visual_pos_masks=info.visual_pos_masks,
            deepstack_visual_embeds=aligned,
            apply_lm_head=False,
        ).last_hidden_state
        compiled_raw = jax.jit(
            lambda tokens: (
                model(
                    input_ids=tokens,
                    apply_lm_head=False,
                    **kwargs,
                ).last_hidden_state
            )
        )(ids)
        no_residual = model(
            inputs_embeds=embeddings, position_ids=info.position_ids, apply_lm_head=False
        ).last_hidden_state
    np.testing.assert_allclose(np.asarray(raw), np.asarray(dense_replay), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(raw), np.asarray(compiled_raw), rtol=2e-5, atol=2e-5)
    assert info.rope_deltas is not None
    assert len(info.deepstack_visual_embeds) == 1
    np.testing.assert_allclose(np.asarray(raw), np.asarray(replay), rtol=2e-5, atol=2e-5)
    assert not np.allclose(np.asarray(raw[:, 2]), np.asarray(no_residual[:, 2]), rtol=1e-4, atol=1e-4)


def test_code_predictor_trunk_and_wrapper_return_updated_cache():
    config = Qwen3OmniMoeTalkerCodePredictorConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_code_groups=2,
        max_position_embeddings=16,
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        attn_mechanism="vanilla",
        scan_layers=False,
    )
    model = omni.Qwen3OmniMoeTalkerCodePredictorForConditionalGeneration(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(6)
    )
    states = jnp.ones((1, 2, 16), dtype=jnp.float32)
    cache = model.model.init_cache(batch_size=1, max_length=4)
    trunk = model.model(inputs_embeds=states, past_key_values=cache)
    assert trunk.past_key_values is not None
    output = model(inputs_embeds=states, past_key_values=model.model.init_cache(batch_size=1, max_length=4))
    assert output.past_key_values is not None
    kwargs = model.model.update_inputs_for_generation(output, {"position_ids": jnp.array([[0, 1]])})
    continued = model(inputs_embeds=states[:, :1], **kwargs)
    for view in continued.past_key_values.views:
        np.testing.assert_array_equal(np.asarray(view.indexes), [3])


def test_code2wav_transformer_matches_hf_with_nonuniform_positions():
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        layer_scale_initial_scale=0.7,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        scan_layers=False,
    )
    hf_config = hf.Qwen3OmniMoeCode2WavConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        layer_scale_initial_scale=0.7,
    )
    hf_config._attn_implementation = "eager"
    reference = hf.Qwen3OmniMoeCode2WavTransformerModel(hf_config).eval()
    with jax.default_matmul_precision("highest"):
        model = omni.Qwen3OmniMoeCode2WavTransformerModel(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(9)
        )
        layer, target = model.layers[0], reference.layers[0]
        with torch.no_grad():
            q, k, v = np.split(np.asarray(layer.self_attn.qkv_proj.weight.value), 3, axis=-1)
            for name, weight in zip(("q_proj", "k_proj", "v_proj"), (q, k, v), strict=True):
                getattr(target.self_attn, name).weight.copy_(torch.tensor(weight.T.copy()))
            target.self_attn.o_proj.weight.copy_(torch.tensor(np.asarray(layer.self_attn.o_proj.weight.value).T.copy()))
            gate, up = np.split(np.asarray(layer.mlp.gate_up_proj.weight.value), 2, axis=-1)
            target.mlp.gate_proj.weight.copy_(torch.tensor(gate.T.copy()))
            target.mlp.up_proj.weight.copy_(torch.tensor(up.T.copy()))
            target.mlp.down_proj.weight.copy_(torch.tensor(np.asarray(layer.mlp.down_proj.weight.value).T.copy()))
            for name in ("input_layernorm", "post_attention_layernorm"):
                getattr(target, name).weight.copy_(torch.tensor(np.asarray(getattr(layer, name).weight.value).copy()))
            for name in ("self_attn_layer_scale", "mlp_layer_scale"):
                getattr(target, name).scale.copy_(torch.tensor(np.asarray(getattr(layer, name).weight.value).copy()))
            reference.norm.weight.copy_(torch.tensor(np.asarray(model.norm.weight.value).copy()))
            states = np.random.default_rng(10).normal(size=(2, 4, 16)).astype(np.float32)
            positions = np.array([[0, 2, 5, 9], [1, 3, 6, 10]], dtype=np.int32)
            # These are rotary coordinates, not packed-sequence boundaries.
            # Without an explicit mask HF interprets every non-unit position
            # increment as a new packed sequence, making this an invalid RoPE
            # parity reference (in this fixture it reduces attention to self).
            attention_mask = np.ones(states.shape[:2], dtype=np.bool_)
            expected = reference(
                inputs_embeds=torch.tensor(states),
                attention_mask=torch.tensor(attention_mask),
                position_ids=torch.tensor(positions),
                use_cache=False,
            ).last_hidden_state.numpy()
        actual = model(
            jnp.asarray(states), attention_mask=jnp.asarray(attention_mask), position_ids=jnp.asarray(positions)
        ).last_hidden_state
        regular = model(jnp.asarray(states), attention_mask=jnp.asarray(attention_mask)).last_hidden_state
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-4, atol=2e-4)
    assert not np.allclose(np.asarray(actual), np.asarray(regular), rtol=1e-5, atol=1e-5)
