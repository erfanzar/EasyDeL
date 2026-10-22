"""RoBERTa regressions (accelerator only)."""

import inspect

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForQuestionAnswering,
    RobertaModel,
    dot_product_attention_weights,
)
from easydel.modules.roberta.roberta_configuration import RobertaConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def _config():
    return RobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


@pytest.mark.cpu_ok
def test_question_answering_forward_inputs_are_optional():
    params = inspect.signature(RobertaForQuestionAnswering.forward).parameters
    for name in ("input_ids", "attention_mask", "token_type_ids", "position_ids", "head_mask"):
        assert params[name].default is None, name


def test_head_mask_attention_weights_are_deterministic_without_rng():
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    q = jax.random.normal(k1, (1, 4, 2, 8))
    k = jax.random.normal(k2, (1, 4, 2, 8))
    plain = dot_product_attention_weights(q, k)
    eval_like = dot_product_attention_weights(q, k, dropout_rate=0.5, broadcast_dropout=True)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(eval_like))
    dropped = dot_product_attention_weights(q, k, dropout_rate=0.5, dropout_rng=jax.random.PRNGKey(3))
    assert not np.allclose(np.asarray(plain), np.asarray(dropped))


@pytest.mark.parametrize("masked_suffix", [False, True])
def test_eval_with_head_mask_is_deterministic_and_matches_all_ones_mask(masked_suffix):
    model = RobertaModel(_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    ids = jnp.array([[3, 4, 5, 6]], jnp.int32)
    mask = jnp.array([[1, 1, 1, 0 if masked_suffix else 1]], jnp.int32)
    ones = jnp.ones((2, 2), jnp.float32)
    first = model(input_ids=ids, attention_mask=mask, head_mask=ones, output_attentions=True)
    second = model(input_ids=ids, attention_mask=mask, head_mask=ones)
    ref = model(input_ids=ids, attention_mask=mask).last_hidden_state
    np.testing.assert_array_equal(np.asarray(first.last_hidden_state), np.asarray(second.last_hidden_state))
    np.testing.assert_allclose(np.asarray(first.last_hidden_state), np.asarray(ref), rtol=1e-4, atol=1e-4)
    if masked_suffix:
        for weights in first.attentions:
            # MaskInfo also masks padded queries. A fully masked row has equal
            # finite-minimum logits, whose softmax is uniform (not all zero).
            # Real queries must never attend to the masked suffix key.
            np.testing.assert_array_equal(np.asarray(weights[..., :-1, -1]), 0)
            np.testing.assert_allclose(np.asarray(weights[..., -1, :]), np.full((1, 2, 4), 0.25), atol=1e-6)


@pytest.mark.parametrize("lazy_bias", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_attention_accepts_optional_array_or_callable_bias(lazy_bias, with_bias):
    query = jnp.ones((1, 2, 1, 4), dtype=jnp.float32)
    bias = jnp.array([[[[0.0, -10000.0], [0.0, -10000.0]]]]) if with_bias else None
    init_bias = (lambda: bias) if lazy_bias else bias
    weights = dot_product_attention_weights(query, query, init_bias=init_bias)
    # Uniform query/key scores leave either a uniform distribution or just the unmasked key.
    expected = np.array([[[[1.0, 0.0], [1.0, 0.0]]]]) if with_bias else np.full((1, 1, 2, 2), 0.5)
    np.testing.assert_allclose(np.asarray(weights), expected, rtol=1e-6, atol=1e-6)


def test_output_dense_uses_requested_param_dtype():
    model = RobertaModel(_config(), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    assert model.encoder.layer[0].output.dense.weight.value.dtype == jnp.bfloat16


def test_causal_lm_get_decoder_returns_backbone():
    model = RobertaForCausalLM(_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert model.get_decoder() is model.roberta
    assert isinstance(model.get_decoder(), RobertaModel)


def test_question_answering_runs_with_input_ids_only():
    model = RobertaForQuestionAnswering(_config(), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    out = model(input_ids=jnp.array([[3, 4, 5]], jnp.int32))
    assert out.start_logits.shape == (1, 3)
    assert out.end_logits.shape == (1, 3)


def test_causal_lm_honours_precomputed_mask_info():
    config = _config()
    config.is_decoder = True
    model = RobertaForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    ids = jnp.array([[3, 4, 5, 6]], jnp.int32)
    mask = jnp.array([[1, 0, 1, 1]], jnp.int32)
    reference = model(input_ids=ids, attention_mask=mask).logits
    result = model(input_ids=ids, attention_mask=jnp.ones_like(ids), mask_info=MaskInfo.from_attention_mask(mask)).logits
    np.testing.assert_allclose(np.asarray(result), np.asarray(reference), rtol=1e-5, atol=1e-5)
    unmasked = model(input_ids=ids, attention_mask=jnp.ones_like(ids)).logits
    assert not np.allclose(np.asarray(result), np.asarray(unmasked))


def test_attention_dtype_and_broadcast_dropout():
    query = jnp.ones((2, 4, 2, 8), dtype=jnp.bfloat16)
    weights = dot_product_attention_weights(
        query,
        query,
        softmax_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        dropout_rate=0.5,
        broadcast_dropout=True,
        dropout_rng=jax.random.PRNGKey(1),
    )
    assert weights.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(weights[0]), np.asarray(weights[1]))


def test_decoder_head_mask_preserves_causality_and_future_independence():
    config = _config()
    config.is_decoder = True
    model = RobertaForCausalLM(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    ids = jnp.array([[3, 4, 5, 6]], jnp.int32)
    ones = jnp.ones((2, 2), jnp.float32)
    masked = model(input_ids=ids, head_mask=ones, output_attentions=True)
    plain = model(input_ids=ids)
    np.testing.assert_allclose(np.asarray(masked.logits), np.asarray(plain.logits), rtol=1e-5, atol=1e-5)
    future_changed = model(input_ids=ids.at[:, 2:].set(jnp.array([[8, 9]])), head_mask=ones)
    np.testing.assert_allclose(np.asarray(masked.logits[:, :2]), np.asarray(future_changed.logits[:, :2]), atol=1e-6)
    for weights in masked.attentions:
        np.testing.assert_array_equal(np.triu(np.asarray(weights), k=1), 0)


@pytest.mark.parametrize("with_head_mask", [False, True])
def test_decoder_retains_cache_and_matches_full_prefix(with_head_mask):
    config = _config()
    config.is_decoder = True
    model = RobertaForCausalLM(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    ids = jnp.array([[3, 4, 5, 6, 7]], jnp.int32)
    kwargs = {"head_mask": jnp.ones((2, 2), jnp.float32)} if with_head_mask else {}
    cache = model.init_cache(batch_size=1, max_length=8)
    prompt = model(input_ids=ids[:, :3], past_key_values=cache, **kwargs)
    assert prompt.past_key_values is not None
    for view in prompt.past_key_values.views:
        np.testing.assert_array_equal(np.asarray(view.indexes), [3])
    cache = prompt.past_key_values
    for position in (3, 4):
        output = model(input_ids=ids[:, position : position + 1], past_key_values=cache, **kwargs)
        assert output.past_key_values is not None
        full = model(input_ids=ids[:, : position + 1], **kwargs)
        np.testing.assert_allclose(
            np.asarray(output.logits[:, -1]), np.asarray(full.logits[:, -1]), rtol=1e-5, atol=1e-5
        )
        for view in output.past_key_values.views:
            np.testing.assert_array_equal(np.asarray(view.indexes), [position + 1])
        cache = output.past_key_values
