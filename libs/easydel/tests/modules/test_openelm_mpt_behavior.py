"""OpenELM, MPT, OLMo and OPT numerics on tiny models (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.mosaic_mpt.modeling_mosaic import MptAttention
from easydel.modules.mosaic_mpt.mosaic_configuration import MptConfig
from easydel.modules.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel
from easydel.modules.muse_glimmer.muse_glimmer_configuration import MuseGlimmerTextConfig
from easydel.modules.olmo2.modeling_olmo2 import Olmo2Attention, Olmo2ForSequenceClassification
from easydel.modules.olmo2.olmo2_configuration import Olmo2Config
from easydel.modules.olmo3.modeling_olmo3 import Olmo3ForSequenceClassification
from easydel.modules.olmo3.olmo3_configuration import Olmo3Config
from easydel.modules.openelm.modeling_openelm import OpenELMForCausalLM
from easydel.modules.openelm.openelm_configuration import OpenELMConfig
from easydel.modules.opt.modeling_opt import OPTAttention, OPTDecoder, OPTLearnedPositionalEmbedding
from easydel.modules.opt.opt_configuration import OPTConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def openelm_config(**kwargs):
    return OpenELMConfig(
        vocab_size=16,
        num_transformer_layers=1,
        model_dim=16,
        head_dim=8,
        ffn_multipliers=2.0,
        ffn_dim_divisor=8,
        max_context_length=8,
        attn_mechanism="vanilla",
        **kwargs,
    )


def test_openelm_embedding_inputs_and_single_lm_head():
    model = OpenELMForCausalLM(openelm_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    ids = jnp.array([[1, 2, 3]])
    embedding = model.transformer.get_embedding()(ids)
    from_ids = model(input_ids=ids)
    from_embeds = model(inputs_embeds=embedding)
    np.testing.assert_allclose(from_ids.logits, from_embeds.logits, rtol=1e-5, atol=1e-5)
    assert from_embeds.logits.shape == (1, 3, 16)
    assert not hasattr(model.transformer, "classifier")
    for inputs in ({}, {"input_ids": ids, "inputs_embeds": embedding}):
        with pytest.raises(ValueError, match="exactly one"):
            model.transformer(**inputs)


@pytest.mark.parametrize("policy", ["nothing_saveable", "dots_saveable"])
def test_openelm_remat_graph_roundtrip_matches_plain_model(policy):
    baseline = OpenELMForCausalLM(openelm_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    checkpointed = OpenELMForCausalLM(
        openelm_config(gradient_checkpointing=policy),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    baseline.eval()
    checkpointed.eval()
    restored = checkpointed.merge_module(*checkpointed.split_module())
    ids = jnp.array([[1, 2]])
    np.testing.assert_allclose(restored(input_ids=ids).logits, baseline(input_ids=ids).logits, rtol=1e-5, atol=1e-5)


def test_openelm_attention_does_not_mutate_shared_config():
    config = openelm_config()
    original = config.to_dict()
    model = OpenELMForCausalLM(config, rngs=spx.Rngs(0))
    assert config.to_dict() == original
    assert model.transformer.layers[0].attn.num_heads == config.num_query_heads[0]


@pytest.mark.parametrize("attn_type,kv_heads", [("multihead_attention", 4), ("multiquery_attention", 1)])
def test_mpt_fused_attention_matches_explicit_reference(attn_type, kv_heads):
    config = MptConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_seq_len=4,
        attn_config={"attn_type": attn_type},
        attn_mechanism="vanilla",
    )
    attn = MptAttention(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0), layer_idx=0)
    attn.eval()
    x = jnp.arange(48, dtype=jnp.float32).reshape(1, 3, 16) / 48
    mask = MaskInfo.dynamic_init(inputs_embeds=x, attention_mask=jnp.ones((1, 3), dtype=bool))
    actual = attn(x, mask, mask.q_position_ids, common_types.MODE_TRAIN).attention_output
    qkv = attn.Wqkv(x)
    assert qkv.shape[-1] == 16 + 2 * kv_heads * 4
    q, k, v = jnp.split(qkv, (16, 16 + kv_heads * 4), axis=-1)
    q = q.reshape(1, 3, 4, 4)
    k = jnp.repeat(k.reshape(1, 3, kv_heads, 4), 4 // kv_heads, axis=2)
    v = jnp.repeat(v.reshape(1, 3, kv_heads, 4), 4 // kv_heads, axis=2)
    # ALiBi depends only on key positions; any per-query constant cancels in softmax.
    slopes = 2.0 ** (-8.0 * jnp.arange(1, 5) / 4)
    bias = slopes[None, :, None, None] * (jnp.arange(3) - 2)[None, None, None, :]
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * 0.5 + bias
    logits = jnp.where(jnp.arange(3)[None, :] <= jnp.arange(3)[:, None], logits, -jnp.inf)
    probabilities = jax.nn.softmax(logits, axis=-1)
    expected = attn.out_proj(jnp.einsum("bhqk,bkhd->bqhd", probabilities, v).reshape(1, 3, 16))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_mpt_attention_has_no_second_residual_dropout():
    config = MptConfig(
        d_model=16, n_heads=4, n_layers=1, max_seq_len=4, attn_config={"attn_pdrop": 0.5}, attn_mechanism="vanilla"
    )
    attn = MptAttention(config, dtype=jnp.float32, rngs=spx.Rngs(0), layer_idx=0)
    # Isolate residual dropout from independently configurable attention-weight dropout.
    attn.attention_performer.dropout_prob = 0.0
    x = jnp.ones((1, 3, 16))
    mask = MaskInfo.dynamic_init(inputs_embeds=x)
    attn.eval()
    expected = attn(x, mask, mask.q_position_ids, common_types.MODE_TRAIN).attention_output
    attn.train()
    actual = attn(x, mask, mask.q_position_ids, common_types.MODE_TRAIN).attention_output
    np.testing.assert_array_equal(actual, expected)


def test_olmo2_qk_norm_storage_and_output_dtype():
    config = Olmo2Config(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        num_hidden_layers=1,
        attn_mechanism="vanilla",
    )
    attn = Olmo2Attention(config, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0), layer_idx=0)
    for norm, width in ((attn.query_normalization, 16), (attn.key_normalization, 8)):
        assert norm.weight.value.dtype == jnp.float32
        assert norm(jnp.ones((1, 2, width), dtype=jnp.bfloat16)).dtype == jnp.bfloat16


@pytest.mark.parametrize(
    "config_class,model_class",
    [
        (Olmo2Config, Olmo2ForSequenceClassification),
        (Olmo3Config, Olmo3ForSequenceClassification),
    ],
)
@pytest.mark.parametrize("structured_mask", [False, True])
def test_olmo_classifier_pools_last_unmasked_token(config_class, model_class, structured_mask):
    config = config_class(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=0,
        num_labels=2,
        layer_types=["full_attention"],
        attn_mechanism="vanilla",
    )
    model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    ids = jnp.array([[0, 2, 3, 0], [4, 5, 0, 0]])
    mask = jnp.array([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    kwargs = (
        {"mask_info": MaskInfo.dynamic_init(input_ids=ids, attention_mask=mask)}
        if structured_mask
        else {
            "attention_mask": mask,
        }
    )
    actual = model(input_ids=ids, apply_lm_head=False, **kwargs)
    hidden = model.model(input_ids=ids, **kwargs).last_hidden_state
    expected = model.score(hidden)[jnp.arange(2), jnp.array([2, 1])]
    np.testing.assert_allclose(actual.logits, expected, rtol=1e-5, atol=1e-5)


def opt_config(**kwargs):
    return OPTConfig(
        vocab_size=16,
        hidden_size=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        attn_mechanism="vanilla",
        **kwargs,
    )


def test_opt_bias_and_norm_parameter_layout():
    decoder = OPTDecoder(opt_config(enable_bias=False, layer_norm_elementwise_affine=False), rngs=spx.Rngs(0))
    layer = decoder.layers[0]
    for linear in (
        layer.self_attn.q_proj,
        layer.self_attn.k_proj,
        layer.self_attn.v_proj,
        layer.self_attn.out_proj,
        layer.fc1,
        layer.fc2,
    ):
        assert linear.bias is None
    for norm in (layer.self_attn_layer_norm, layer.final_layer_norm, decoder.final_layer_norm):
        assert not norm.use_scale
        assert not norm.use_bias
    decoder.eval()
    assert decoder(jnp.array([[2, 3]])).last_hidden_state.shape == (1, 2, 16)


def test_opt_attention_probability_dropout_train_eval():
    config = opt_config(dropout=0, attention_dropout=0.5)
    attn = OPTAttention(config, 16, 2, dropout=0.5, causal=True, dtype=jnp.float32, rngs=spx.Rngs(0))
    x = jnp.arange(64, dtype=jnp.float32).reshape(1, 4, 16) / 64
    mask = MaskInfo.dynamic_init(inputs_embeds=x)
    attn.eval()
    expected = attn(x, mask, common_types.MODE_TRAIN).attention_output
    np.testing.assert_array_equal(attn(x, mask, common_types.MODE_TRAIN).attention_output, expected)
    attn.train()
    actual = attn(x, mask, common_types.MODE_TRAIN).attention_output
    assert not np.array_equal(actual, expected)


def test_opt_layerdrop_drops_whole_residual_update():
    decoder = OPTDecoder(
        opt_config(dropout=0, layerdrop=1.0, _remove_final_layer_norm=True), dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    ids = jnp.array([[2, 3, 4]])
    positions = jnp.array([[0, 1, 2]])
    expected = decoder.embed_tokens(ids) + decoder.embed_positions(positions)
    decoder.train()
    np.testing.assert_array_equal(decoder(ids, position_ids=positions).last_hidden_state, expected)
    decoder.eval()
    assert not np.array_equal(decoder(ids, position_ids=positions).last_hidden_state, expected)


def test_opt_embedding_dropout_and_offset_lookup():
    config = OPTConfig(
        vocab_size=16,
        hidden_size=16,
        num_hidden_layers=0,
        num_attention_heads=2,
        dropout=0.5,
        do_layer_norm_before=False,
        max_position_embeddings=8,
        attn_mechanism="vanilla",
    )
    decoder = OPTDecoder(config, dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = jnp.array([[2, 3, 4]])
    positions = jnp.array([[0, 1, 2]])
    assert isinstance(decoder.embed_positions, OPTLearnedPositionalEmbedding)
    np.testing.assert_array_equal(
        decoder.embed_positions(positions), decoder.embed_positions.weight.value[positions + 2]
    )
    decoder.eval()
    expected = decoder(ids, position_ids=positions).last_hidden_state
    decoder.train()
    actual = decoder(ids, position_ids=positions).last_hidden_state
    assert not np.array_equal(actual, expected)
    assert np.any(np.asarray(actual) == 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_muse_scaleless_norm_keeps_fp32_multiply(dtype):
    config = MuseGlimmerTextConfig(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=8,
        rms_norm_eps=1e-3,
        attn_mechanism="vanilla",
    )
    model = MuseGlimmerTextModel(config, dtype=dtype, param_dtype=dtype, rngs=spx.Rngs(0))
    x = jnp.asarray([[0.003, 0.7, -4.1, 2.3], [10.1, -0.2, 0.5, 0.01]], dtype=dtype)
    model.embed_tokens.weight.value = jnp.zeros((8, 4), dtype=dtype).at[:2].set(x)
    promoted = x.astype(jnp.float32)
    expected = (promoted * jax.lax.rsqrt(jnp.mean(promoted**2, axis=-1, keepdims=True) + 1e-3)).astype(dtype)
    actual = model.embed_input_ids(jnp.array([[0, 1]]))
    np.testing.assert_array_equal(actual[0], expected)
    assert actual.dtype == dtype
