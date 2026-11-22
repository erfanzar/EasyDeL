"""Hardware-only joint-vocabulary forward, weight tying and generation regressions."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.loss_utils import CausalLMLossStrategy, LossConfig
from easydel.modules.llama import VisionLlamaConfig, VisionLlamaForCausalLM
from jax import numpy as jnp


def _model(tie_vision_embeddings=False, dtype=jnp.float32, sample_mode="all"):
    config = VisionLlamaConfig(
        vocab_size=16,
        vision_vocab_size=8,
        tie_vision_embeddings=tie_vision_embeddings,
        sample_mode=sample_mode,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        kvdtype=dtype,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        pad_token_id=0,
        eos_token_id=None,
        bos_token_id=1,
    )
    with config.mesh:
        return VisionLlamaForCausalLM(config, dtype=dtype, param_dtype=jnp.float32, rngs=spx.Rngs(0))


@pytest.mark.parametrize("tie_vision_embeddings", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_joint_forward_embeddings_and_logits_match_independent_projection(tie_vision_embeddings, dtype):
    model = _model(tie_vision_embeddings, dtype)
    ids = jnp.asarray([[1, 16, 3, 23]], dtype=jnp.int32)
    embeddings = model.model.embed_input_ids(ids)
    text_weight = np.asarray(model.model.embed_tokens.weight.value)
    vision_weight = np.asarray(model.model.vision_embed_tokens.weight.value)
    expected_embeddings = np.stack((text_weight[1], vision_weight[0], text_weight[3], vision_weight[7]))[None]
    np.testing.assert_array_equal(np.asarray(embeddings), np.asarray(jnp.asarray(expected_embeddings, dtype=dtype)))
    output = model(input_ids=ids)
    assert output.logits.shape == (1, 4, model.config.total_vocab_size)
    assert model.model.embed_tokens.weight.value.shape == (16, 16)
    assert model.lm_head.weight.value.shape == (16, 16)
    assert model.model.vision_embed_tokens.weight.value.dtype == jnp.float32
    assert embeddings.dtype == dtype
    hidden = np.asarray(output.last_hidden_state).astype(np.float32)
    if tie_vision_embeddings:
        projection_weight = np.asarray(jnp.asarray(vision_weight.T, dtype=dtype)).astype(np.float32)
    else:
        projection_weight = np.asarray(model.vision_lm_head.weight.value.astype(dtype)).astype(np.float32)
    expected_vision_logits = hidden @ projection_weight
    np.testing.assert_allclose(
        np.asarray(output.logits[..., 16:]).astype(np.float32), expected_vision_logits, rtol=2e-2, atol=2e-3
    )
    np.testing.assert_allclose(
        np.asarray(model.make_lm_head_fn()(output.last_hidden_state)).astype(np.float32),
        np.asarray(output.logits).astype(np.float32),
        rtol=2e-2,
        atol=2e-3,
    )
    if tie_vision_embeddings:
        # The output must use the current input table, not a copied initializer.
        before = model.compute_lm_logits(output.last_hidden_state)[..., 16:]
        model.model.vision_embed_tokens.weight.value = model.model.vision_embed_tokens.weight.value * 2
        after = model.compute_lm_logits(output.last_hidden_state)[..., 16:]
        np.testing.assert_allclose(np.asarray(after), np.asarray(before * 2), rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("tie_vision_embeddings", [False, True])
def test_dense_and_chunked_training_losses_include_vision_targets(tie_vision_embeddings):
    model = _model(tie_vision_embeddings=tie_vision_embeddings)
    ids = jnp.asarray([[1, 16, 3, 23]], dtype=jnp.int32)
    strategy = CausalLMLossStrategy()
    dense_output = model(input_ids=ids)
    # Independent stable cross-entropy over the actual joint output vocabulary.
    logits = np.asarray(dense_output.logits[:, :-1]).astype(np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    targets = np.asarray(ids[:, 1:])
    expected_loss = -np.take_along_axis(log_probs, targets[..., None], axis=-1).mean()
    for chunk_size in (None, 2):
        loss_config = LossConfig(chunk_token_size=chunk_size)
        plan = strategy.plan_forward(module=model, labels=ids, loss_config=loss_config, batch={}, loss_kwargs={})
        assert plan.forward_kwargs == ({"apply_lm_head": False} if chunk_size else {})
        output = model(input_ids=ids, **plan.forward_kwargs) if chunk_size else dense_output
        metrics = strategy.compute(
            module=model,
            outputs=output,
            labels=ids,
            loss_config=loss_config,
            batch={},
            loss_kwargs={},
            paxis=None,
            forward_plan=plan,
        )
        np.testing.assert_allclose(np.asarray(metrics.loss), expected_loss, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("sample_mode", ["all", "random", "text", "vision"])
def test_generation_respects_namespace_and_reproducible_random_sampling(sample_mode):
    model = _model(sample_mode=sample_mode)
    ids = jnp.asarray([[1, 18]], dtype=jnp.int32)
    kwargs = dict(max_new_tokens=2, top_k=0, pad_token_id=0, eos_token_id=None, trace=False)
    result = model.generate(ids, prng_key=jax.random.PRNGKey(17), **kwargs)
    assert result.sequences.shape == (1, 4)
    generated = np.asarray(result.sequences[:, 2:])
    assert np.all((0 <= generated) & (generated < 24))
    if sample_mode == "text":
        assert np.all(generated < 16)
    if sample_mode == "vision":
        assert np.all(generated >= 16)
    if sample_mode == "random":
        repeated = model.generate(ids, prng_key=jax.random.PRNGKey(17), **kwargs)
        np.testing.assert_array_equal(np.asarray(result.sequences), np.asarray(repeated.sequences))


def test_joint_cached_continuation_matches_full_sequence():
    model = _model(tie_vision_embeddings=True)
    ids = jnp.asarray([[1, 17, 2, 22]], dtype=jnp.int32)
    full = model(input_ids=ids)
    cache = model.init_cache(batch_size=1, max_length=8)
    prefill = model(input_ids=ids[:, :3], past_key_values=cache)
    continuation = model(
        input_ids=ids[:, 3:], position_ids=jnp.asarray([[3]], dtype=jnp.int32), past_key_values=prefill.past_key_values
    )
    np.testing.assert_allclose(np.asarray(continuation.logits), np.asarray(full.logits[:, 3:]), rtol=2e-3, atol=2e-3)
