"""Audio length, chunk-local position, window isolation, and padding regressions."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention,
    Qwen3OmniMoeAudioEncoder,
)
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import Qwen3OmniMoeAudioConfig
from jax import numpy as jnp


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    with jax.default_matmul_precision("highest"):
        yield


def _encoder(n_window=8, n_window_infer=16):
    config = Qwen3OmniMoeAudioConfig(
        num_mel_bins=8,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        downsample_hidden_size=2,
        output_dim=8,
        n_window=n_window,
        n_window_infer=n_window_infer,
        conv_chunksize=1,
        attn_mechanism="vanilla",
        # Attention has an independent compute dtype; the model dtype alone
        # does not disable its default bf16 rounding in float32 HF parity tests.
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        scan_layers=False,
    )
    return Qwen3OmniMoeAudioEncoder(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(0)
    )


def test_audio_windows_match_independently_encoded_chunks():
    model = _encoder()
    features = jnp.asarray(np.random.default_rng(0).normal(size=(1, 8, 32)), dtype=jnp.float32)
    actual = model(features).last_hidden_state
    first = model(features[:, :, :16]).last_hidden_state
    second = model(features[:, :, 16:]).last_hidden_state
    expected = jnp.concatenate((first, second), axis=1)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)
    model.conv_chunksize = 3
    grouped = model(features).last_hidden_state
    np.testing.assert_allclose(np.asarray(actual), np.asarray(grouped), rtol=2e-5, atol=2e-5)


def test_audio_lengths_mask_padding_before_convolution_and_after_projection():
    model = _encoder()
    features = jnp.asarray(np.random.default_rng(1).normal(size=(2, 8, 32)), dtype=jnp.float32)
    lengths = jnp.array([32, 9], dtype=jnp.int32)
    actual = model(features, feature_lens=lengths).last_hidden_state
    changed = model(features.at[1, :, 9:].set(1000), feature_lens=lengths).last_hidden_state
    np.testing.assert_allclose(np.asarray(actual), np.asarray(changed), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(np.asarray(actual[1, 2:]), np.zeros((2, 8)))
    mask = jnp.arange(32)[None, :] < lengths[:, None]
    inferred = model(features, attention_mask=mask).last_hidden_state
    np.testing.assert_allclose(np.asarray(actual), np.asarray(inferred), rtol=2e-5, atol=2e-5)


def test_audio_inference_window_controls_cross_chunk_context():
    model = _encoder()
    features = jnp.asarray(np.random.default_rng(2).normal(size=(1, 8, 32)), dtype=jnp.float32)
    changed_features = features.at[:, :, 16:].add(5)
    local = model(features).last_hidden_state
    changed_local = model(changed_features).last_hidden_state
    np.testing.assert_allclose(np.asarray(local[:, :2]), np.asarray(changed_local[:, :2]), rtol=2e-5, atol=2e-5)
    model.n_window_infer = 32
    joined = model(features).last_hidden_state
    changed_joined = model(changed_features).last_hidden_state
    assert not np.allclose(np.asarray(joined[:, :2]), np.asarray(changed_joined[:, :2]), rtol=1e-5, atol=1e-5)


def test_audio_attention_excludes_padded_keys_from_valid_queries():
    config = _encoder().config
    attention = Qwen3OmniMoeAudioAttention(
        config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3)
    )
    states = jnp.asarray(np.random.default_rng(3).normal(size=(2, 4, 8)), dtype=jnp.float32)
    mask = jnp.array([[True, True, False, False], [True, True, True, True]])
    changed = states.at[0, 2:].add(10)
    baseline = attention(states, attention_mask=mask)
    actual = attention(changed, attention_mask=mask)
    np.testing.assert_allclose(np.asarray(baseline[0, :2]), np.asarray(actual[0, :2]), rtol=1e-5, atol=1e-5)
    assert not np.allclose(
        np.asarray(attention(states)[0, :2]),
        np.asarray(attention(changed)[0, :2]),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("inference_window", [100, 200])
@pytest.mark.parametrize("lengths", [(200, 9), (10,), (10, 9), (8, 3), (9, 7), (17, 16)])
@pytest.mark.parametrize("conv_chunksize", [1, 3])
def test_audio_encoder_matches_hf_chunking_windowing_and_padding(inference_window, lengths, conv_chunksize):
    torch = pytest.importorskip("torch")
    hf = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    # Installed HF hardcodes 13 CNN outputs per full chunk, so use its supported
    # 100-frame chunks (n_window=50), including batches with no full chunk at all.
    model = _encoder(n_window=50, n_window_infer=inference_window)
    model.conv_chunksize = conv_chunksize
    reference_config = hf.Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=8,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        downsample_hidden_size=2,
        output_dim=8,
        n_window=50,
        n_window_infer=inference_window,
        conv_chunksize=conv_chunksize,
        activation_function=model.config.activation_function,
        attention_dropout=0.0,
    )
    reference_config._attn_implementation = "eager"
    reference = hf.Qwen3OmniMoeAudioEncoder(reference_config).eval()
    random = np.random.default_rng(23)

    def copy_parameters(source, target, permutation=None):
        # Nontrivial weights/biases expose boundary leakage through successive
        # convolutions rather than hiding it behind near-zero final projections.
        if source.weight.value.ndim > 1:
            source.weight.value = jnp.asarray(random.normal(0, 0.2, source.weight.value.shape), dtype=jnp.float32)
        weight = np.asarray(source.weight.value)
        if permutation is not None:
            weight = weight.transpose(permutation)
        target.weight.copy_(torch.tensor(weight.copy()))
        if target.bias is not None:
            source.bias.value = jnp.asarray(random.normal(0, 0.1, source.bias.value.shape), dtype=jnp.float32)
            target.bias.copy_(torch.tensor(np.asarray(source.bias.value).copy()))

    with torch.no_grad():
        for name in ("conv2d1", "conv2d2", "conv2d3"):
            # EasyDeL convolves [time, frequency, channels], HF [channels, frequency, time].
            copy_parameters(getattr(model, name), getattr(reference, name), (3, 2, 1, 0))
        for name in ("conv_out", "proj1", "proj2"):
            copy_parameters(getattr(model, name), getattr(reference, name), (1, 0))
        copy_parameters(model.ln_post, reference.ln_post)
        for source_layer, target_layer in zip(model.layers, reference.layers, strict=True):
            for name in ("fc1", "fc2"):
                copy_parameters(getattr(source_layer, name), getattr(target_layer, name), (1, 0))
            for name in ("self_attn_layer_norm", "final_layer_norm"):
                copy_parameters(getattr(source_layer, name), getattr(target_layer, name))
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                copy_parameters(getattr(source_layer.self_attn, name), getattr(target_layer.self_attn, name), (1, 0))

        features = np.random.default_rng(4).normal(size=(len(lengths), 8, 200)).astype(np.float32)
        lengths = np.array(lengths, dtype=np.int32)
        # HF accepts mel frames packed across the batch rather than a padded batch.
        packed = np.concatenate([features[row, :, :length] for row, length in enumerate(lengths)], axis=1)
        # HF's chunk_lengths destination is torch.long; its indexed assignment
        # requires matching int64 lengths, independently of JAX's int32 inputs.
        expected = reference(
            torch.tensor(packed), feature_lens=torch.tensor(lengths, dtype=torch.long)
        ).last_hidden_state.numpy()
    actual = np.asarray(model(jnp.asarray(features), feature_lens=jnp.asarray(lengths)).last_hidden_state)
    output_lengths = (lengths // 100) * 13 + (lengths % 100 + 7) // 8
    valid = np.concatenate([actual[row, :length] for row, length in enumerate(output_lengths)], axis=0)
    # With fp32 attention and highest dot precision this also guards the CNN's
    # exact (not tanh-approximated) GELU against the independent HF reference.
    np.testing.assert_allclose(valid, expected, rtol=2e-5, atol=2e-6)
    for row, length in enumerate(output_lengths):
        np.testing.assert_array_equal(actual[row, length:], np.zeros_like(actual[row, length:]))


def test_audio_short_batch_convolution_boundaries_with_traced_lengths():
    model = _encoder()
    random = np.random.default_rng(17)
    for name in ("conv2d1", "conv2d2", "conv2d3", "conv_out", "proj1", "proj2"):
        layer = getattr(model, name)
        layer.weight.value = jnp.asarray(random.normal(0, 0.2, layer.weight.value.shape), dtype=jnp.float32)
        if layer.bias is not None:
            layer.bias.value = jnp.asarray(random.normal(0, 0.1, layer.bias.value.shape), dtype=jnp.float32)
    features = jnp.asarray(random.normal(size=(2, 8, 32)), dtype=jnp.float32)
    lengths = jnp.array([10, 9], dtype=jnp.int32)
    # The trimmed call uses the reference's actual longest-chunk width. Traced
    # lengths must reproduce that boundary through all three convolution stages.
    expected = np.asarray(model(features[:, :, :10], feature_lens=lengths).last_hidden_state)
    encode = jax.jit(lambda inputs, lens: model(inputs, feature_lens=lens).last_hidden_state)
    actual = np.asarray(encode(features, lengths))
    np.testing.assert_allclose(actual[:, :2], expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual[:, 2:], np.zeros_like(actual[:, 2:]))
