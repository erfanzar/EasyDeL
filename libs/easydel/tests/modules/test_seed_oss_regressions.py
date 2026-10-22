"""Seed-OSS regressions; numerical tests require an accelerator."""

import inspect

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.seed_oss.modeling_seed_oss import SeedOssForSequenceClassification, SeedOssModel
from easydel.modules.seed_oss.seed_oss_configuration import SeedOssConfig
from jax import numpy as jnp


def _config(**kwargs):
    return SeedOssConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


def test_legacy_residual_dropout_does_not_change_checkpoint_forward():
    models = [
        SeedOssModel(
            _config(residual_dropout=rate), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0, dropout=1)
        )
        for rate in (0.0, 0.5)
    ]
    ids = jnp.array([[1, 2, 3, 4]], jnp.int32)
    for training in (False, True):
        for model in models:
            model.train() if training else model.eval()
        expected, actual = [model(input_ids=ids).last_hidden_state for model in models]
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_resid_pdrop_applies_in_training_only():
    model = SeedOssModel(
        _config(resid_pdrop=0.5), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0, dropout=1)
    )
    ids = jnp.array([[1, 2, 3, 4]], jnp.int32)
    model.eval()
    a = model(input_ids=ids).last_hidden_state
    b = model(input_ids=ids).last_hidden_state
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    model.train()
    c = model(input_ids=ids).last_hidden_state
    assert not np.allclose(np.asarray(a), np.asarray(c))


def test_sequence_classification_pools_last_attended_token():
    config = _config(residual_dropout=0.0, num_labels=3, pad_token_id=0)
    model = SeedOssForSequenceClassification(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    # Masked suffix tokens deliberately differ from pad_token_id: pooling must use the mask.
    ids = jnp.array([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]], jnp.int32)
    mask = jnp.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], jnp.int32)
    logits = model(input_ids=ids, attention_mask=mask).logits
    hidden = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    full = model.score(hidden)
    expected = jnp.stack([full[0, 2], full[1, 4]])
    assert logits.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected), rtol=1e-5, atol=1e-5)
    # The compatibility switch must continue to bypass the head while returning hidden states.
    skipped = model(input_ids=ids, attention_mask=mask, apply_lm_head=False, output_hidden_states=True)
    assert skipped.logits is None
    assert skipped.hidden_states[-1].shape == (2, 5, 16)


@pytest.mark.parametrize("batch_size,pad_token_id", [(1, None), (1, 0), (2, 0)])
@pytest.mark.parametrize("with_mask", [False, True])
def test_sequence_classification_pools_embeddings_only(batch_size, pad_token_id, with_mask):
    config = _config(residual_dropout=0.0, num_labels=3, pad_token_id=pad_token_id)
    model = SeedOssForSequenceClassification(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    model.eval()
    # Supply neither input_ids nor, for the compatibility regression, attention_mask.
    embeds = jnp.arange(batch_size * 5 * 16, dtype=jnp.float32).reshape(batch_size, 5, 16) / 100
    mask = jnp.tile(jnp.array([[1, 1, 1, 0, 0]], jnp.int32), (batch_size, 1)) if with_mask else None
    output = model(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True)
    hidden = model.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    expected = model.score(hidden[:, 2 if with_mask else -1])
    assert output.logits.shape == (batch_size, 3)
    np.testing.assert_allclose(np.asarray(output.logits), np.asarray(expected), rtol=1e-5, atol=1e-5)
    assert output.hidden_states[-1].shape == (batch_size, 5, 16)
    skipped = model(inputs_embeds=embeds, attention_mask=mask, apply_lm_head=False, output_hidden_states=True)
    assert skipped.logits is None
    assert skipped.hidden_states[-1].shape == (batch_size, 5, 16)


@pytest.mark.cpu_ok
def test_sequence_classification_keeps_optional_masking_and_head_switch_api():
    parameters = inspect.signature(SeedOssForSequenceClassification.forward).parameters
    for name in ("input_ids", "inputs_embeds", "attention_mask", "mask_info", "position_ids", "segment_ids"):
        assert parameters[name].default is None
        assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["apply_lm_head"].default is True


@pytest.mark.parametrize("apply_lm_head", [True, False])
@pytest.mark.parametrize("mask_ndim", [2, 3, 4])
@pytest.mark.parametrize(
    "ids,mask,segments,final_indices",
    [
        ([9, 5, 6, 7, 8], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [3, 4]),
        ([5, 6, 7, 8, 9], [1, 1, 1, 1, 0], [0, 0, 1, 1, 1], [2, 3]),
        ([5, 6, 9, 7, 8], [1, 1, 0, 1, 1], [0, 0, 1, 1, 1], [3, 4]),
    ],
    ids=["left_padding", "right_padding", "masked_segment_prefix"],
)
def test_packed_classification_preserves_padding_and_segment_positions(
    ids, mask, segments, final_indices, apply_lm_head, mask_ndim
):
    model = SeedOssForSequenceClassification(
        _config(residual_dropout=0.0, num_labels=3, pad_token_id=0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    model.eval()
    attention_mask = np.array([mask], dtype=bool)
    if mask_ndim > 2:
        attention_mask = attention_mask[:, :, None] & attention_mask[:, None, :]
        attention_mask &= np.tri(len(ids), dtype=bool)[None]
        if mask_ndim == 4:
            attention_mask = attention_mask[:, None]
    packed = model(
        input_ids=jnp.array([ids], jnp.int32),
        attention_mask=jnp.array(attention_mask),
        segment_ids=jnp.array([segments], jnp.int32),
        apply_lm_head=apply_lm_head,
        output_hidden_states=True,
    )
    # The last packed sequence must be independent of both earlier sequences and
    # masked non-pad tokens, with rotary positions starting from zero as usual.
    standalone = model(
        input_ids=jnp.array([[7, 8]], jnp.int32),
        apply_lm_head=apply_lm_head,
        output_hidden_states=True,
    )
    np.testing.assert_allclose(
        np.asarray(packed.hidden_states[-1][:, final_indices]),
        np.asarray(standalone.hidden_states[-1]),
        rtol=1e-5,
        atol=1e-5,
    )
    if apply_lm_head:
        np.testing.assert_allclose(np.asarray(packed.logits), np.asarray(standalone.logits), rtol=1e-5, atol=1e-5)
    else:
        assert packed.logits is None


@pytest.mark.parametrize("mask_layout", ["batched", "single_head", "multi_head"])
@pytest.mark.parametrize("embeddings_only", [False, True])
def test_sequence_classification_pools_pairwise_query_validity(mask_layout, embeddings_only):
    model = SeedOssForSequenceClassification(
        _config(residual_dropout=0.0, num_labels=3, pad_token_id=0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    model.eval()
    ids = jnp.array([[5, 6, 7, 8, 9], [5, 6, 7, 8, 9]], jnp.int32)
    inputs = {"inputs_embeds": model.model.embed_tokens(ids)} if embeddings_only else {"input_ids": ids}
    # Query and key validity deliberately differ: all keys are allowed, but the
    # last valid queries are positions 2 and 3. Non-pad suffix IDs cannot be pooled.
    query_valid = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0]], dtype=bool)
    pairwise = np.broadcast_to(query_valid[:, :, None], (2, 5, 5)).copy()
    if mask_layout != "batched":
        pairwise = pairwise[:, None]
    if mask_layout == "multi_head":
        pairwise = np.repeat(pairwise, 2, axis=1)
        # The final valid query participates only in head 1; validity must reduce
        # across all heads rather than selecting head 0 or requiring every head.
        pairwise[0, 0, 2] = False
        pairwise[1, 0, 3] = False
    mask = jnp.array(pairwise)
    output = model(**inputs, attention_mask=mask, output_hidden_states=True)
    reference = model.model(**inputs, attention_mask=mask).last_hidden_state
    expected = model.score(jnp.stack([reference[0, 2], reference[1, 3]]))
    assert output.logits.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(output.logits), np.asarray(expected), rtol=1e-5, atol=1e-5)
    # Pooling conversion must not replace the backbone's full pairwise mask.
    np.testing.assert_allclose(np.asarray(output.hidden_states[-1]), np.asarray(reference), rtol=1e-5, atol=1e-5)
