# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.
"""Real-model training coverage; no mocked vision/embedding/decoder paths.

Run against a complete checkout with its normal EasyDeL/SpectraX dependencies.
The hybrid optimizer test and each modality case are independently selectable.
Do not outer-jit the vision test: the shared tower consumes host grid metadata.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.modules.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from easydel.modules.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from spectrax import common_types


@pytest.fixture(autouse=True)
def _strict_fp32():
    # FP32 storage alone does not request FP32 dot arithmetic on TPU.
    with jax.default_matmul_precision("highest"):
        yield
    # These are deliberately separate compilations, not a growing model suite.
    jax.clear_caches()


def _text_config(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=2,
        indexer_compress_ratio=2,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],  # one-indexed: PLE feeds the real GDN layer
        ple_embed_dim=32,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=2,
        seed=1234,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        eos_token_id=0,
        bos_token_id=1,
        pad_token_id=2,
        norm_topk_prob=True,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        attn_dtype=jnp.float32,
        mamba_ssm_dtype="float32",
        mtp=None,
        mtp_num_hidden_layers=0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


def _model(cls, config):
    return cls(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )


def _assert_gradients(loss, grads, *branches):
    assert np.isfinite(np.asarray(loss)).all(), f"nonfinite loss: {loss}"
    named = [
        (jax.tree_util.keystr(path), np.asarray(grad))
        for path, grad in jax.tree_util.tree_flatten_with_path(grads)[0]
        if hasattr(grad, "size") and grad.size
    ]
    assert named, "no parameter gradients were returned"
    for name, grad in named:
        assert np.isfinite(grad).all(), f"nonfinite gradient: {name}"
    for branch in branches:
        # Tuple fragments distinguish nested branches without depending on
        # whether a layer stack uses numeric keys or a stacked parameter axis.
        matched = [(name, grad) for name, grad in named if all(part in name for part in branch)]
        assert matched, f"missing gradient branch {branch}; paths: {[name for name, _ in named]}"
        assert any(np.any(grad != 0) for _, grad in matched), f"all-zero gradient branch {branch}"


def test_hybrid_gdn_ple_qsa_lm_loss_updates_parameters():
    """A causal language loss must train each hybrid branch, not just the head."""
    model = _model(Qwen4ExpForCausalLM, _text_config())
    ids = jnp.array([[1, 7, 13, 19, 23, 31, 37, 43]], jnp.int32)
    state = EasyDeLState.create(model=model, tx=optax.adam(1e-3), init_opt_state=True)

    def loss_fn(graphstate):
        out = state.merge(graphstate)(input_ids=ids, mode=common_types.MODE_TRAIN)
        return optax.softmax_cross_entropy_with_integer_labels(out.logits[:, :-1].astype(jnp.float32), ids[:, 1:]).mean()

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(state.graphstate)
    branches = (
        ("linear_attn",),
        ("ple_embedding",),
        ("ple", "value_proj"),
        ("ple", "conv1d"),
        ("self_attn", "qkv_proj"),
        ("indexer",),
    )
    _assert_gradients(loss, grads, *branches)
    updated = state.apply_gradients(grads=grads)
    before = dict(
        (jax.tree_util.keystr(p), np.asarray(v)) for p, v in jax.tree_util.tree_flatten_with_path(state.graphstate)[0]
    )
    after = dict(
        (jax.tree_util.keystr(p), np.asarray(v)) for p, v in jax.tree_util.tree_flatten_with_path(updated.graphstate)[0]
    )
    assert before.keys() == after.keys()
    for name, value in after.items():
        assert np.isfinite(value).all(), f"nonfinite updated parameter: {name}"
    for branch in branches:
        assert any(
            np.any(after[name] != value) for name, value in before.items() if all(part in name for part in branch)
        ), f"optimizer did not move branch {branch}"


@pytest.mark.parametrize("modality", ["image", "video"])
def test_real_vision_merge_trains_ple_language_path(modality):
    """Raw patches -> real vision -> placeholder merge -> PLE -> language loss.

    One GDN+PLE decoder layer suffices here; the separate hybrid test exercises
    QSA. Video has two temporal groups, each with its own placeholder/delimiters.
    The loss is on trailing text, never a standalone vision/embedding objective.
    """
    text = _text_config(num_hidden_layers=1, layer_types=["linear_attention"])
    vision = Qwen4ExpVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=text.hidden_size,
        num_position_embeddings=4,
        deepstack_visual_indexes=[],
        attn_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config = Qwen4ExpConfig(
        text_config=text,
        vision_config=vision,
        language_model_only=False,
        image_token_id=9,
        video_token_id=10,
        vision_start_token_id=11,
        vision_end_token_id=12,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = _model(Qwen4ExpForConditionalGeneration, config)
    frames = 1 if modality == "image" else 2
    token = config.image_token_id if modality == "image" else config.video_token_id
    ids = jnp.array([[1] + [11, token, 12] * frames + [17, 21, 22]], jnp.int32)
    # Each merged visual token consumes 2x2 patches; each flattened patch has
    # C * temporal_patch_size * patch_size**2 = 24 pixel values.
    pixels = jnp.asarray(np.random.default_rng(7).normal(size=(frames * 4, 24)), jnp.float32)
    grid = np.array([[frames, 2, 2]], np.int32)  # host metadata, not a traced input
    if modality == "image":
        kwargs = dict(pixel_values=pixels, image_grid_thw=grid, image_max_grid_size=2)
        features = model.model.get_image_features(pixels, grid, 2)
    else:
        kwargs = dict(pixel_values_videos=pixels, video_grid_thw=grid, video_max_grid_size=2)
        features = model.model.get_video_features(pixels, grid, 2)
    # The shared Qwen3-VL tower returns (merged_features, deepstack_features).
    # Read its tensor for the independent merge oracle; do not replace or mock
    # get_*_features / compute_embedding, which must handle the real return API.
    raw_features = features[0] if isinstance(features, tuple) else features
    assert raw_features.shape == (frames, text.hidden_size)
    assert np.isfinite(np.asarray(raw_features)).all()
    embeddings = model.model.language_model.embed_tokens(ids)
    merged = model.compute_embedding(ids, **kwargs)
    mask = np.asarray(ids == token)
    np.testing.assert_allclose(np.asarray(merged)[mask], np.asarray(raw_features), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(merged)[~mask], np.asarray(embeddings)[~mask])
    assert np.any(np.asarray(merged)[mask] != np.asarray(embeddings)[mask])

    # Precompute real multimodal mRoPE rows on the host before differentiation.
    position_ids, _ = model.model.get_rope_index(
        ids,
        image_grid_thw=grid if modality == "image" else None,
        video_grid_thw=grid if modality == "video" else None,
    )
    out = model(input_ids=ids, position_ids=position_ids, mode=common_types.MODE_TRAIN, **kwargs)
    # Independent decoder call proves the wrapper preserves original hash IDs
    # alongside visual embeddings rather than silently falling back to tokens.
    expected = model.model.language_model(
        inputs_embeds=merged,
        ple_input_ids=ids,
        position_ids=position_ids,
        mode=common_types.MODE_TRAIN,
    )
    expected_logits = model.lm_head(expected.last_hidden_state)
    np.testing.assert_allclose(np.asarray(out.logits), np.asarray(expected_logits), rtol=1e-5, atol=1e-6)

    state = EasyDeLState.create(model=model, tx=optax.sgd(1e-3), init_opt_state=True)

    def loss_fn(graphstate):
        result = state.merge(graphstate)(
            input_ids=ids, position_ids=position_ids, mode=common_types.MODE_TRAIN, **kwargs
        )
        # Predict the final two ordinary text tokens from earlier causal states.
        return optax.softmax_cross_entropy_with_integer_labels(
            result.logits[:, -3:-1].astype(jnp.float32), ids[:, -2:]
        ).mean()

    loss, grads = jax.jit(jax.value_and_grad(loss_fn))(state.graphstate)
    _assert_gradients(
        loss,
        grads,
        ("visual", "patch_embed"),
        ("visual", "blocks"),
        ("visual", "merger"),
        ("language_model", "ple_embedding"),
        ("language_model", "ple", "value_proj"),
        ("language_model", "linear_attn"),
    )
