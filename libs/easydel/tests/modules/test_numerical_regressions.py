"""Qwen3-VL, Qwen3-VL-MoE and Qwen4-Exp numerical regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP, Qwen3VLTextModel, Qwen3VLVisionPatchEmbed
from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLTextConfig, Qwen3VLVisionConfig
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextSparseBlock,
    get_rope_index,
)
from easydel.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import Qwen3VLMoeTextConfig
from easydel.modules.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel, Qwen4ExpPLELayer, Qwen4ExpVisionTransformer
from easydel.modules.qwen4_exp.qwen4_exp_configuration import Qwen4ExpTextConfig, Qwen4ExpVisionConfig
from jax import numpy as jnp


@pytest.mark.parametrize("compute,storage", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16)])
def test_patch_embed_preserves_storage_and_compute_dtypes(compute, storage):
    config = Qwen3VLVisionConfig(
        hidden_size=4, in_channels=1, patch_size=2, temporal_patch_size=2, sharding_axis_dims=(1, 1, 1, 1, 1, 1)
    )
    layer = Qwen3VLVisionPatchEmbed(config, dtype=compute, param_dtype=storage, rngs=spx.Rngs(0))
    patches = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 16
    result = layer(patches)
    assert layer.proj.weight.value.dtype == storage
    assert layer.proj.bias.value.dtype == storage
    assert result.dtype == compute
    expected = patches.astype(compute) @ layer.proj.weight.value.astype(compute).reshape(8, 4)
    expected += layer.proj.bias.value.astype(compute)
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float32), np.asarray(expected, dtype=np.float32), rtol=0.02, atol=0.02
    )


def test_first_scanned_forward_does_not_cache_a_tracer():
    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = Qwen3VLTextModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    with jax.checking_leaks():
        first = model(input_ids=ids, trace=False).last_hidden_state
    second = model(input_ids=ids, trace=False).last_hidden_state
    assert first.shape == (1, 3, 16)
    assert np.isfinite(np.asarray(first)).all()
    np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("modality,frames", [("image", 1), ("video", 2)])
def test_qwen4_real_vision_infers_grid_bound(modality, frames):
    config = Qwen4ExpVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        num_position_embeddings=4,
        deepstack_visual_indexes=[],
        attn_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    visual = Qwen4ExpVisionTransformer(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model = object.__new__(Qwen4ExpModel)
    object.__setattr__(model, "visual", visual)
    pixels = jnp.asarray(np.random.default_rng(0).normal(size=(4 * frames, 24)), jnp.float32)
    grid = ((frames, 2, 2),)
    encode = model.get_image_features if modality == "image" else model.get_video_features
    automatic = encode(pixels, grid)
    explicit = encode(pixels, grid, 2)
    assert automatic.shape == (frames, 16)
    assert np.isfinite(np.asarray(automatic)).all()
    np.testing.assert_allclose(np.asarray(automatic), np.asarray(explicit), rtol=1e-5, atol=1e-5)


def test_moe_rope_indices_match_independent_mixed_image_positions():
    positions, deltas = get_rope_index(
        np.array([[7, 9, 9, 9, 9, 5]]), image_grid_thw=np.array([[1, 2, 2]]), vision_start_token_id=7, image_token_id=9
    )
    expected = np.array([[[0, 1, 1, 1, 1, 3]], [[0, 1, 1, 2, 2, 3]], [[0, 1, 2, 1, 2, 3]]])
    np.testing.assert_array_equal(np.asarray(positions), expected)
    np.testing.assert_array_equal(np.asarray(deltas), [[-2]])


def _moe_config(normalize, top_k=1):
    return Qwen3VLMoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=top_k,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        norm_topk_prob=normalize,
        moe_force_xla_gmm=True,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


@pytest.mark.parametrize("top_k", [1, 2])
def test_norm_topk_prob_changes_expert_output_by_selected_probability(top_k):
    normalized = Qwen3VLMoeTextSparseBlock(
        _moe_config(True, top_k), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    raw = Qwen3VLMoeTextSparseBlock(
        _moe_config(False, top_k), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    hidden = jnp.asarray(np.random.default_rng(0).normal(size=(1, 3, 16)), jnp.float32)
    norm_out, norm_logits = normalized(hidden)
    raw_out, raw_logits = raw(hidden)
    probabilities = jax.nn.softmax(norm_logits.astype(jnp.float32), axis=-1)
    selected_mass = jnp.sort(probabilities, axis=-1)[..., -top_k:].sum(axis=-1).reshape(1, 3, 1)
    np.testing.assert_allclose(np.asarray(raw_logits), np.asarray(norm_logits), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(raw_out), np.asarray(norm_out * selected_mass), rtol=2e-4, atol=2e-5)
    assert not np.allclose(np.asarray(raw_out), np.asarray(norm_out), rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("cls", [Qwen3VLTextMLP, Qwen3VLMoeTextMLP])
def test_dense_moe_mlp_preserves_output_and_projection_layout(cls):
    config = _moe_config(True)
    layer = cls(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
        layer_idx=0,
    )
    random = np.random.default_rng(42)
    gate = random.normal(scale=0.2, size=(16, 24)).astype(np.float32)
    up = random.normal(scale=0.2, size=(16, 24)).astype(np.float32)
    down = random.normal(scale=0.2, size=(24, 16)).astype(np.float32)
    hidden = random.normal(size=(1, 2, 16)).astype(np.float32)
    # A TP=1 fused runtime tensor is [hidden, gate || up]; the HF source
    # tensors remain independent gate_proj/up_proj paths, not a second MLP.
    layer.gate_up_proj.weight.value = jnp.asarray(np.concatenate([gate, up], axis=-1))
    layer.down_proj.weight.value = jnp.asarray(down)
    gate_logits = hidden @ gate
    reference = ((gate_logits / (1 + np.exp(-gate_logits))) * (hidden @ up)) @ down
    np.testing.assert_allclose(np.asarray(layer(jnp.asarray(hidden))), reference, rtol=2e-5, atol=2e-6)
    assert layer.gate_up_proj.weight.value.shape == (16, 48)
    assert layer.down_proj.weight.value.shape == (24, 16)
    assert set(layer.reform_param) == {"gate_up_proj.weight$"}
    assert layer.reform_param["gate_up_proj.weight$"]["sources"] == ("gate_proj.weight", "up_proj.weight")


def test_ple_packed_prefill_masks_convolution_across_documents():
    config = Qwen4ExpTextConfig(
        vocab_size=32,
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["linear_attention"],
        hc_count=2,
        ple_layer_ids=[1],
        ple_embed_dim=8,
        ple_conv_kernel_size=3,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=1,
        split_ngram_parts=1,
        eos_token_id=0,
        indexer_n_heads=None,
        indexer_kv_heads=None,
        indexer_head_dim=None,
        indexer_budget=None,
        indexer_compress_ratio=None,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    layer = Qwen4ExpPLELayer(
        config,
        layer_idx=0,
        ple_layer_index=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    hidden = jnp.asarray(np.random.default_rng(0).normal(size=(1, 6, 8)), jnp.float32)
    ids = jnp.array([[1, 2, 3, 4, 5, 6]], jnp.int32)
    segments = jnp.array([[0, 0, 0, 1, 1, 1]], jnp.int32)
    # Zero convolution isolates the public PLE gated value, then an independent
    # NumPy RMSNorm and segmented/dilated convolution predict the full output.
    gated, _, _, _ = layer(hidden, ids, segment_ids=segments)
    values = np.asarray(gated).reshape(1, 6, 2, 4)
    normalized = values / np.sqrt(np.mean(values**2, axis=-1, keepdims=True) + config.rms_norm_eps)
    normalized = normalized.reshape(1, 6, 8) * (1 + np.asarray(layer.norm_conv.weight.value))
    layer.conv1d.value = jnp.ones((3, 1, 8), jnp.float32)
    conv = np.zeros((1, 6, 8), np.float32)
    for query in range(6):
        for tap in range(3):
            key = query - 4 + tap * 2
            if key >= 0 and np.asarray(segments)[0, key] == np.asarray(segments)[0, query]:
                conv[:, query] += normalized[:, key]
    reference = np.asarray(gated) + conv / (1 + np.exp(-conv))
    result, tokens, state, context_segments = layer(hidden, ids, segment_ids=segments)
    np.testing.assert_allclose(np.asarray(result), reference, rtol=2e-5, atol=2e-6)
    expected_state = normalized[:, -4:].transpose(0, 2, 1).copy()
    expected_state[:, :, 0] = 0  # retained token 2 belongs to the previous document
    np.testing.assert_allclose(np.asarray(state), expected_state, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(np.asarray(tokens), [[6]])
    np.testing.assert_array_equal(np.asarray(context_segments), [[1]])
