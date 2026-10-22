"""OPT, MPT, OLMo3 and OpenELM numerical regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import TransformerCache, TransformerCacheConfig, TransformerCacheView
from easydel.inference.esurge.runners.executors.model_executor import _project_stage_lm_head
from easydel.modules.mosaic_mpt.modeling_mosaic import MptForCausalLM, MptModel
from easydel.modules.mosaic_mpt.mosaic_configuration import MptConfig
from easydel.modules.olmo3.modeling_olmo3 import Olmo3Attention
from easydel.modules.olmo3.olmo3_configuration import Olmo3Config
from easydel.modules.openelm.modeling_openelm import OpenELMForCausalLM
from easydel.modules.openelm.openelm_configuration import OpenELMConfig
from easydel.modules.opt.modeling_opt import OPTDecoder, OPTForCausalLM
from easydel.modules.opt.opt_configuration import OPTConfig
from jax import numpy as jnp


def _mpt_config(**kwargs):
    # Attention has its own compute dtype, independent of the model dtype.
    # Pin it for the tight float32 NumPy reference comparisons below.
    return MptConfig(
        vocab_size=16,
        d_model=16,
        n_heads=4,
        n_layers=1,
        max_seq_len=8,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        **kwargs,
    )


def _opt_config(**kwargs):
    return OPTConfig(
        vocab_size=16,
        hidden_size=16,
        ffn_dim=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=8,
        attn_mechanism="vanilla",
        **kwargs,
    )


@pytest.mark.parametrize("tied", [False, True])
def test_opt_projected_embedding_head_width_and_logits(tied):
    config = _opt_config(word_embed_proj_dim=8, tie_word_embeddings=tied)
    model = OPTForCausalLM(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    ids = jnp.array([[1, 2, 3]])
    hidden = model(input_ids=ids, apply_lm_head=False).last_hidden_state
    assert config.hidden_size == 16
    assert hidden.shape == (1, 3, 8)
    assert model.lm_head.weight.value.shape == (8, 16)
    kernel = model.get_embedding().weight.value.T if tied else model.lm_head.weight.value
    expected = np.asarray(hidden) @ np.asarray(kernel)
    # Projected OPT must retain its embedding width in both ordinary inference
    # and the trace-safe head used by chunked training/serving paths.
    for actual in (model(input_ids=ids).logits, model.make_lm_head_fn()(hidden)):
        assert actual.shape == (1, 3, config.vocab_size)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["mpt", "opt", "openelm"])
def test_per_call_attention_request_overrides_config(family):
    if family == "mpt":
        cls, config, heads = MptForCausalLM, _mpt_config(output_attentions=False), 4
    elif family == "opt":
        cls, config, heads = OPTForCausalLM, _opt_config(output_attentions=False), 2
    else:
        config = OpenELMConfig(
            vocab_size=16,
            num_transformer_layers=1,
            model_dim=16,
            head_dim=8,
            qkv_multipliers=1.0,
            ffn_multipliers=2.0,
            ffn_dim_divisor=8,
            max_context_length=8,
            output_attentions=False,
            attn_mechanism="vanilla",
        )
        cls, heads = OpenELMForCausalLM, config.num_query_heads[0]
    model = cls(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    output = model(input_ids=jnp.array([[1, 2, 3]]), output_attentions=True)
    weights = np.asarray(output.attentions[0])
    assert weights.shape == (1, heads, 3, 3)
    np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-6)
    np.testing.assert_array_equal(weights[..., np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]], 0)
    assert np.all(weights[..., 2, :2] > 0)


def test_mpt_full_model_nondefault_alibi_matches_independent_reference():
    config = _mpt_config(attn_config={"alibi_bias_max": 3.0})
    model = MptModel(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    # Zero Q/K makes attention probabilities depend only on configured ALiBi
    # and causality, independently of embeddings, normalization or projections.
    model.blocks[0].attn.Wqkv.weight.value = jnp.zeros_like(model.blocks[0].attn.Wqkv.weight.value)
    output = model(input_ids=jnp.array([[1, 2, 3]]), output_attentions=True)
    slopes = 2.0 ** (-3.0 * np.arange(1, 5, dtype=np.float32) / 4)
    logits = slopes[:, None, None] * np.arange(3, dtype=np.float32)[None, None, :]
    logits = np.where(np.arange(3)[None, :] <= np.arange(3)[:, None], logits, -np.inf)
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    expected = exp / exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(output.attentions[0][0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scale,expected_scale", [(None, 1.0), (0.0, 0.0), (2.5, 2.5), ("inv_sqrt_d_model", 0.25)])
def test_mpt_full_trace_safe_and_exported_head_apply_scale(scale, expected_scale):
    model = MptForCausalLM(
        _mpt_config(logit_scale=scale, use_lm_head=False, tie_word_embeddings=False),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    model.eval()
    ids = jnp.array([[1, 2, 3]])
    hidden = model(input_ids=ids, apply_lm_head=False).last_hidden_state
    # MPT's generic task projection is untied here; compare with an independent
    # NumPy matrix product, not with another production postprocessing path.
    expected = np.asarray(hidden) @ np.asarray(model.lm_head.weight.value) * expected_scale
    full = model(input_ids=ids).logits
    trace_safe = model.make_lm_head_fn()(hidden)
    graphdef, state = spx.export(model.lm_head)
    split = _project_stage_lm_head(
        graphdef,
        state,
        hidden,
        uses_tied_projection=False,
        postprocess=model.get_lm_logit_postprocess(),
        scope="mpt_scale_regression",
    )
    for actual in (full, trace_safe, split):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_olmo3_qk_norm_float32_compute_with_bfloat16_storage():
    config = Olmo3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["full_attention"],
        attn_mechanism="vanilla",
    )
    attention = Olmo3Attention(config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0), layer_idx=0)
    for norm, width in ((attention.query_normalization, 16), (attention.key_normalization, 8)):
        x = np.linspace(-2.71, 3.19, width * 2, dtype=np.float32).reshape(1, 2, width)
        stored_weight = jnp.asarray(np.linspace(0.73, 1.29, width), jnp.bfloat16)
        norm.weight.value = stored_weight
        expected = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + config.rms_norm_eps)
        expected *= np.asarray(stored_weight, dtype=np.float32)
        actual = norm(jnp.asarray(x))
        assert norm.weight.value.dtype == jnp.bfloat16
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("layerdrop", [1e-30, 1.0])
def test_opt_layerdrop_preserves_legacy_cache_without_segment_history(layerdrop):
    decoder = OPTDecoder(
        _opt_config(dropout=0.0, attention_dropout=0.0, layerdrop=layerdrop, _remove_final_layer_norm=True),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    decoder.train()
    metadata = TransformerCacheConfig.create(
        batch_size=1, sequence_length=8, num_hidden_layers=1, num_heads=2, head_dim=8
    )
    view = TransformerCacheView(
        key=jnp.ones((1, 8, 2, 8)),
        value=jnp.full((1, 8, 2, 8), 2.0),
        indexes=jnp.array([2], jnp.int32),
        starts=jnp.array([0], jnp.int32),
        metadata=metadata,
        maximum_sequence_length=8,
    )
    output = decoder(jnp.array([[3]]), position_ids=jnp.array([[2]]), past_key_values=TransformerCache(views=[view]))
    updated = output.past_key_values.views[0]
    expected_index = 2 if layerdrop == 1.0 else 3
    np.testing.assert_array_equal(updated.indexes, [expected_index])
    if layerdrop == 1.0:
        np.testing.assert_array_equal(updated.key, view.key)
        np.testing.assert_array_equal(updated.value, view.value)
        np.testing.assert_array_equal(updated.starts, view.starts)
        np.testing.assert_array_equal(updated.segment_ids, [[0, 0, -1, -1, -1, -1, -1, -1]])
    else:
        np.testing.assert_array_equal(updated.key[:, :2], view.key[:, :2])
        np.testing.assert_array_equal(updated.segment_ids[:, :3], [[0, 0, 0]])
