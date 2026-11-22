"""Exaone and Falcon model contracts; numerical tests require accelerator execution."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import HybridCacheView
from easydel.modules.dspark import DSparkConfig, DSparkModel
from easydel.modules.exaone import ExaoneConfig
from easydel.modules.exaone.modeling_exaone import ExaoneAttentionInner
from easydel.modules.falcon import FalconConfig, FalconModel
from easydel.modules.falcon_h1 import FalconH1Config
from easydel.modules.falcon_h1.modeling_falcon_h1 import FalconH1Mixer
from easydel.modules.falcon_mamba import FalconMambaForCausalLM, FalconMambaModel
from easydel.modules.falcon_mamba.modeling_falcon_mamba import FalconMambaMixer
from easydel.modules.gemma import GemmaConfig, GemmaForSequenceClassification, GemmaModel
from ejkernel.types import MaskInfo
from jax import numpy as jnp

from .test_falcon_state_boundaries import _mamba_config, _recurrent_cache

PRECISION = jax.lax.Precision.HIGHEST


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    # Keep independent JAX references and tied embeddings in true float32 on TPU.
    with jax.default_matmul_precision("highest"):
        yield


def _h1_config():
    return FalconH1Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        mamba_d_ssm=8,
        mamba_n_heads=2,
        mamba_d_head=4,
        mamba_d_state=2,
        mamba_d_conv=2,
        mamba_chunk_size=2,
    )


def _h1_cache(mixer, conv, state):
    return HybridCacheView(
        key=None,
        value=None,
        conv_state=conv,
        recurrent_state=state,
        positions=jnp.zeros((conv.shape[0],), dtype=jnp.int32),
        metadata=None,
        layer_index=0,
        layer_type="linear_attention",
    )


@pytest.mark.parametrize("family", ["mamba", "h1"])
@pytest.mark.parametrize("segmented", [False, True])
def test_cached_multi_token_chunks_match_token_steps(family, segmented):
    """Nonzero convolution and recurrent histories must both survive a chunk."""
    if family == "mamba":
        config = _mamba_config()
        mixer = FalconMambaMixer(
            config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0)
        )
        conv_shape = (1, config.intermediate_size, config.conv_kernel)
        state_shape = (1, config.intermediate_size, config.state_size)
    else:
        config = _h1_config()
        mixer = FalconH1Mixer(
            config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0)
        )
        conv_shape = (1, mixer.conv_dim, mixer.conv_kernel_size)
        state_shape = (1, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size)
    conv = jax.random.normal(jax.random.key(10), conv_shape) * 0.1
    state = jax.random.normal(jax.random.key(11), state_shape) * 0.1
    initial_cache = _recurrent_cache(conv, state) if family == "mamba" else _h1_cache(mixer, conv, state)
    inputs = jax.random.normal(jax.random.key(12), (1, 3, config.hidden_size))

    def forward(x, cache):
        segments = jnp.zeros(x.shape[:2], dtype=jnp.int32) if segmented else None
        if family == "mamba":
            return mixer(x, cache_params=cache, segment_ids=segments)
        mask = MaskInfo.from_segments(q_segment_ids=segments) if segmented else None
        return mixer(x, mask_info=mask, cache_view=cache)

    chunk_output, chunk_cache = forward(inputs, initial_cache)
    stepped_cache = initial_cache
    outputs = []
    for index in range(inputs.shape[1]):
        output, stepped_cache = forward(inputs[:, index : index + 1], stepped_cache)
        outputs.append(output)
    np.testing.assert_allclose(chunk_output, jnp.concatenate(outputs, axis=1), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(chunk_cache.conv_state, stepped_cache.conv_state, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(chunk_cache.recurrent_state, stepped_cache.recurrent_state, atol=2e-4, rtol=2e-4)


def test_h1_packed_slot_gather_matches_independent_sequences_and_keeps_idle_slot():
    config = _h1_config()
    mixer = FalconH1Mixer(
        config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0)
    )
    inputs = jax.random.normal(jax.random.key(13), (1, 5, config.hidden_size))
    conv = jax.random.normal(jax.random.key(14), (3, mixer.conv_dim, mixer.conv_kernel_size)) * 0.1
    state = jax.random.normal(jax.random.key(15), (3, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size)) * 0.1
    metadata = SimpleNamespace(query_start_loc=jnp.asarray([0, 2, 5, 5]), num_seqs=2)
    packed_output, packed_cache = mixer(
        inputs, mask_info=None, cache_view=_h1_cache(mixer, conv, state), cache_metadata=metadata
    )
    for slot, (start, end) in enumerate(((0, 2), (2, 5))):
        output, cache = mixer(
            inputs[:, start:end],
            mask_info=None,
            cache_view=_h1_cache(mixer, conv[slot : slot + 1], state[slot : slot + 1]),
        )
        np.testing.assert_allclose(packed_output[:, start:end], output, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(packed_cache.conv_state[slot], cache.conv_state[0], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(packed_cache.recurrent_state[slot], cache.recurrent_state[0], atol=2e-4, rtol=2e-4)
    np.testing.assert_array_equal(packed_cache.conv_state[2], conv[2])
    np.testing.assert_array_equal(packed_cache.recurrent_state[2], state[2])


@pytest.mark.cpu_ok
@pytest.mark.parametrize("width,factor,expected", [(19, None, 19), (19, 3, 19), (None, 3, 24), (None, None, 32)])
def test_falcon_width_precedence_preserves_explicit_size_and_legacy_factor(width, factor, expected):
    config = FalconConfig(hidden_size=8, num_attention_heads=2, ffn_hidden_size=width, ff_factor=factor)
    assert config.ffn_hidden_size == expected


@pytest.mark.parametrize("num_ln", [1, 2])
def test_falcon_new_decoder_takes_precedence_over_legacy_parallel_flag(num_ln):
    config = FalconConfig(
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_kv_heads=2,
        new_decoder_architecture=True,
        parallel_attn=False,
        num_ln_in_parallel_attn=num_ln,
        # Divisible by the four-chip mesh but not by hidden_size.
        ffn_hidden_size=20,
        attn_mechanism="vanilla",
    )
    model = FalconModel(config, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0))
    result = model(input_ids=jnp.asarray([[1, 2, 3]], dtype=jnp.int32))
    assert result.last_hidden_state.shape == (1, 3, 8)
    assert np.isfinite(result.last_hidden_state).all()
    mlp = model.h[0].mlp
    assert mlp.dense_h_to_4h.weight.value.shape == (8, 20)
    assert mlp.dense_4h_to_h.weight.value.shape == (20, 8)
    # Exact GELU-MLP reference additionally checks that all 20 channels execute.
    x = jnp.ones((1, 2, 8), dtype=jnp.float32)
    reference = jax.nn.gelu(x @ mlp.dense_h_to_4h.weight.value, approximate=False) @ mlp.dense_4h_to_h.weight.value
    np.testing.assert_allclose(mlp(x), reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("family", ["falcon", "mamba"])
def test_hidden_state_tuple_starts_with_embedding_and_then_first_decoder_output(family):
    if family == "falcon":
        config = FalconConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_kv_heads=2,
            attn_mechanism="vanilla",
        )
        model = FalconModel(config, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0))
        embedding = model.word_embeddings
    else:
        config = _mamba_config()
        config.num_hidden_layers = 2
        model = FalconMambaModel(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0)
        )
        embedding = model.embeddings
    ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    output = model(input_ids=ids, output_hidden_states=True)
    assert len(output.hidden_states) == 3
    np.testing.assert_allclose(output.hidden_states[0], embedding(ids), atol=1e-6, rtol=1e-5)
    # The public first decoder layer supplies the intermediate identity;
    # the last tuple entry alone includes the model's final normalization.
    layers = model.h if family == "falcon" else model.layers
    if family == "falcon":
        first = layers[0](
            embedding(ids),
            MaskInfo.from_segments(q_segment_ids=jnp.zeros_like(ids)),
            jnp.arange(3, dtype=jnp.int32)[None],
            "train",
            frequencies=model.frequencies,
        ).hidden_states
    else:
        first, _ = layers[0](hidden_states=embedding(ids))
    np.testing.assert_allclose(output.hidden_states[1], first, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(output.hidden_states[-1], output.last_hidden_state)


@pytest.mark.parametrize("bias", [False, True])
def test_exaone_output_projection_bias_matches_configured_parameter_contract(bias):
    config = ExaoneConfig(hidden_size=8, num_attention_heads=2, num_key_value_heads=2, attention_bias=bias)
    attention = ExaoneAttentionInner(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0), layer_idx=0
    )
    projection = attention.out_proj
    assert (projection.bias is not None) == bias
    x = jnp.ones((1, 2, 8), dtype=jnp.float32)
    expected = x @ projection.weight.value
    if bias:
        projection.bias.value = jnp.ones_like(projection.bias.value)
        expected += 1
    np.testing.assert_allclose(projection(x), expected, atol=1e-6, rtol=1e-5)


def test_dspark_sequence_mode_attends_past_draft_without_future_context_leakage():
    config = DSparkConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        target_hidden_size=8,
        target_layer_ids=[0],
        confidence_head_alpha=0.0,
    )
    model = DSparkModel(config, dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0))
    ids = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    context = jax.random.normal(jax.random.key(16), (1, 3, 8))
    output = model(ids, context).logits
    changed_past = model(ids.at[0, 0].set(4), context).logits
    assert float(jnp.max(jnp.abs(changed_past[:, -1] - output[:, -1]))) > 1e-7
    future = model(ids.at[0, 2].set(5), context.at[:, 2].add(10.0)).logits
    np.testing.assert_allclose(future[:, :2], output[:, :2], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("model_class", [FalconMambaModel, FalconMambaForCausalLM])
def test_mamba_sequential_init_restores_log_arange_spectrum(model_class):
    config = _mamba_config()
    config.state_size = 3
    model = model_class.sequential_init(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    backbone = model if model_class is FalconMambaModel else model.backbone
    expected = np.broadcast_to(np.log(np.arange(1, 4, dtype=np.float32)), (config.intermediate_size, 3))
    np.testing.assert_allclose(backbone.layers[0].mixer.A_log.value, expected, atol=1e-6, rtol=1e-6)


def _gemma_config(pad_token_id=0):
    return GemmaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_labels=2,
        pad_token_id=pad_token_id,
        attn_mechanism="vanilla",
    )


def test_gemma_norm_parameter_storage_independent_of_compute_dtype():
    model = GemmaModel(_gemma_config(), dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    for norm in (model.norm, model.layers[0].input_layernorm, model.layers[0].post_attention_layernorm):
        assert norm.weight.value.dtype == jnp.float32
        inputs = jnp.asarray([[[0.5, 0.25, -0.5, 1.0, -1.0, 0.0, 0.25, 0.5]]], dtype=jnp.bfloat16)
        result = norm(inputs)
        assert result.dtype == jnp.bfloat16
        expected = np.asarray(inputs, dtype=np.float32)
        expected /= np.sqrt(np.mean(expected**2, axis=-1, keepdims=True) + model.config.rms_norm_eps)
        np.testing.assert_array_equal(result, jnp.asarray(expected, dtype=jnp.bfloat16))


@pytest.mark.parametrize("embeddings_only", [False, True])
@pytest.mark.parametrize("pad_token_id", [0, None])
def test_gemma_classifier_pools_last_masked_valid_token(embeddings_only, pad_token_id):
    model = GemmaForSequenceClassification(
        _gemma_config(pad_token_id), dtype=jnp.float32, param_dtype=jnp.float32, precision=PRECISION, rngs=spx.Rngs(0)
    )
    ids = jnp.asarray([[0, 2, 3, 0], [1, 2, 0, 0]], dtype=jnp.int32)
    mask = jnp.asarray([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.int32)
    arguments = {"inputs_embeds": model.model.embed_tokens(ids)} if embeddings_only else {"input_ids": ids}
    output = model(**arguments, attention_mask=mask)
    hidden = model.model(**arguments, attention_mask=mask).last_hidden_state
    expected = jnp.stack((hidden[0, 2], hidden[1, 1])) @ model.score.weight.value
    np.testing.assert_allclose(output.logits, expected, atol=1e-6, rtol=1e-5)
