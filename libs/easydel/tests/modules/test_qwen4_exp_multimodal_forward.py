# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import jax.numpy as jnp
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel


class _DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(image_token_id=9, video_token_id=10)
        self.embedding_calls = []
        self.language_calls = []

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        batch, seq = input_ids.shape
        return jnp.zeros((3, batch, seq), jnp.int32), jnp.full((batch, 1), 5, jnp.int32)

    def compute_embedding(self, input_ids, **kwargs):
        self.embedding_calls.append((input_ids, kwargs))
        return jnp.full((*input_ids.shape, 4), 3.0, jnp.float32)

    def language_model(self, **kwargs):
        self.language_calls.append(kwargs)
        return SimpleNamespace(rope_deltas=None)


def test_multimodal_forward_merges_vision_and_preserves_ids_for_ple():
    model = _DummyModel()
    ids = jnp.array([[1, 9, 2]], jnp.int32)
    out = Qwen4ExpModel.forward(
        model,
        input_ids=ids,
        pixel_values=jnp.ones((1, 4)),
        image_grid_thw=jnp.ones((1, 3), jnp.int32),
    )
    assert len(model.embedding_calls) == 1
    call = model.language_calls[-1]
    assert call["input_ids"] is None
    assert call["ple_input_ids"] is ids
    assert jnp.all(call["inputs_embeds"] == 3)
    assert jnp.array_equal(out.rope_deltas, jnp.array([[5]], jnp.int32))


def test_multimodal_rope_state_does_not_leak_into_later_text_batch():
    model = _DummyModel()
    ids = jnp.array([[1, 9, 2]], jnp.int32)
    Qwen4ExpModel.forward(
        model,
        input_ids=ids,
        pixel_values=jnp.ones((1, 4)),
        image_grid_thw=jnp.ones((1, 3), jnp.int32),
    )
    out = Qwen4ExpModel.forward(model, input_ids=jnp.array([[3, 4]], jnp.int32))
    assert out.rope_deltas is None


def test_language_model_only_does_not_construct_vision_tower(monkeypatch):
    import easydel.modules.qwen4_exp.modeling_qwen4_exp as modeling
    import spectrax as spx
    from easydel.modules.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig

    def forbidden(*args, **kwargs):
        raise AssertionError("vision tower was constructed")

    monkeypatch.setattr(modeling, "Qwen4ExpVisionTransformer", forbidden)
    text = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        full_attention_interval=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        indexer_n_heads=None,
        indexer_kv_heads=None,
        indexer_head_dim=None,
        indexer_budget=None,
        indexer_compress_ratio=None,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        mtp_num_hidden_layers=0,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = modeling.Qwen4ExpModel(
        Qwen4ExpConfig(text_config=text, language_model_only=True),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    assert model.visual is None


def test_packed_depthwise_conv_resets_at_segment_boundaries():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _packed_depthwise_causal_conv

    x = jnp.array([[[1.0], [2.0], [10.0], [20.0]]])
    segments = jnp.array([[0, 0, 1, 1]], jnp.int32)
    kernel = jnp.ones((3, 1, 1), jnp.float32)
    got = _packed_depthwise_causal_conv(x, kernel, segments, dilation=1)
    # First token of document 1 must not consume [1,2] from document 0.
    assert jnp.array_equal(got[..., 0], jnp.array([[1.0, 3.0, 10.0, 30.0]]))


def test_get_rope_index_accepts_dynamic_jit_inputs():
    from types import SimpleNamespace

    dummy = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=9,
            video_token_id=10,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
    )

    @__import__("jax").jit
    def build(ids, mask):
        return Qwen4ExpModel.get_rope_index(dummy, ids, attention_mask=mask)

    ids = jnp.array([[1, 2, 3, 4]], jnp.int32)
    pos, delta = build(ids, jnp.ones_like(ids))
    assert pos.shape == (3, 1, 4)
    assert delta.shape == (1, 1)


def test_precomputed_multimodal_embeddings_are_forwarded_to_merge():
    model = _DummyModel()
    ids = jnp.array([[1, 9, 10]], jnp.int32)
    image_embeds = jnp.ones((1, 4), jnp.float32)
    video_embeds = jnp.full((1, 4), 2.0, jnp.float32)
    Qwen4ExpModel.forward(model, input_ids=ids, image_embeds=image_embeds, video_embeds=video_embeds)
    kwargs = model.embedding_calls[-1][1]
    assert kwargs["image_embeds"] is image_embeds
    assert kwargs["video_embeds"] is video_embeds


def test_compiled_multimodal_forward_requires_precomputed_position_ids():
    import jax
    import pytest

    model = _DummyModel()

    @jax.jit
    def run(ids, grid):
        return Qwen4ExpModel.forward(model, input_ids=ids, image_grid_thw=grid)

    with pytest.raises(ValueError, match="host-precomputed position_ids"):
        run(jnp.array([[1, 9, 2]], jnp.int32), jnp.ones((1, 3), jnp.int32))


def test_conditional_mtp_defaults_position_rows_and_mask_info():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    captured = {}

    class MTP:
        embed_tokens = None

        def __call__(self, streams, embeds, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(last_hidden_state=embeds)

    dummy = SimpleNamespace(
        mtp=MTP(),
        model=SimpleNamespace(
            language_model=SimpleNamespace(
                embed_tokens=lambda ids: jnp.ones((*ids.shape, 4), jnp.float32),
                frequencies=None,
            )
        ),
    )
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    out = Qwen4ExpForConditionalGeneration.compute_mtp_outputs(dummy, jnp.ones((1, 3, 2, 4), jnp.float32), ids)
    assert out is not None
    assert captured["position_ids"].shape == (3, 1, 3)
    assert captured["mask_info"] is not None


def test_mtp_context_preserves_padding_positions_and_explicit_segments():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import _resolve_qwen4_mtp_context
    from ejkernel.types import MaskInfo

    ids = jnp.array([[0, 0, 5, 6]], jnp.int32)
    attention_mask = jnp.array([[0, 0, 1, 1]], jnp.int32)
    segment_ids = jnp.array([[-1, -1, 7, 7]], jnp.int32)
    packed = MaskInfo.from_segments(q_segment_ids=segment_ids, kv_segment_ids=segment_ids)
    mask, rows = _resolve_qwen4_mtp_context(ids, None, attention_mask, packed, None)
    assert jnp.array_equal(mask._q_segment_ids, segment_ids)
    assert jnp.array_equal(rows[0], mask.q_position_ids)
    assert rows.shape == (3, 1, 4)


def test_ple_right_padding_caches_last_live_token_history():
    import jax.numpy as jnp

    from tests.modules.spmd.test_qwen4_exp import _config, _model

    config = _config(num_hidden_layers=1, layer_types=["linear_attention"], ple_layer_ids=[1], mtp_num_hidden_layers=0)
    ple = _model(config).model.layers[0].ple
    ids = jnp.array([[1, 2, 0, 0], [3, 4, 5, 0]], jnp.int32)
    mask = jnp.array([[1, 1, 0, 0], [1, 1, 1, 0]], bool)
    segments = jnp.where(mask, 0, -1).astype(jnp.int32)
    _, context, _, segment_context = ple(
        jnp.zeros((2, 4, config.hidden_size * config.hc_count), jnp.float32),
        ids,
        conv_mask=mask,
        segment_ids=segments,
    )
    assert jnp.array_equal(context, jnp.array([[1, 2], [4, 5]], jnp.int32))
    assert jnp.array_equal(segment_context, jnp.zeros((2, 2), jnp.int32))


def test_packed_mtp_shift_and_loss_do_not_cross_segments():
    from easydel.modules.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpForCausalLM,
        _packed_mtp_next_ids,
    )
    from ejkernel.types.mask import MaskInfo

    ids = jnp.array([[1, 2, 3, 4, 5, 6]], jnp.int32)
    segments = jnp.array([[0, 0, 0, 1, 1, 1]], jnp.int32)
    mask = MaskInfo.from_segments(segments)
    next_ids, resolved_segments = _packed_mtp_next_ids(ids, mask)
    assert jnp.array_equal(next_ids, jnp.array([[2, 3, 0, 5, 6, 0]], jnp.int32))

    vocab = 8
    logits = jnp.full((1, 6, vocab), -10.0)
    # Only t=0 -> label[2] and t=3 -> label[5] are valid skip-two targets.
    logits = logits.at[0, 0, 3].set(10.0).at[0, 3, 6].set(10.0)
    loss = Qwen4ExpForCausalLM.compute_mtp_loss(logits, ids, segment_ids=resolved_segments)
    assert float(loss) < 1e-5
    bad = logits.at[0, 0, :].set(0.0).at[0, 3, :].set(0.0)
    assert float(Qwen4ExpForCausalLM.compute_mtp_loss(bad, ids, segment_ids=resolved_segments)) > 1.0
