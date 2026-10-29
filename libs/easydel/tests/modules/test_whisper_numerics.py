"""Hardware-only Whisper regressions (accelerator only)."""

import copy
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.logits_process import ForceTokensLogitsProcessor, LogitsProcessorList
from easydel.modules._base import BaseConditionalGenerationModule
from easydel.modules.whisper.modeling_whisper import (
    WhisperDecoder,
    WhisperEncoder,
    WhisperForConditionalGeneration,
)
from easydel.modules.whisper.whisper_configuration import WhisperConfig
from jax import numpy as jnp
from transformers import GenerationConfig


def _config(**kwargs):
    return WhisperConfig(
        vocab_size=32,
        d_model=8,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        num_mel_bins=4,
        max_source_positions=4,
        max_target_positions=8,
        pad_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=1,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


def test_encoder_stem_keeps_parameter_storage_dtype():
    config = _config()
    with config.mesh:
        encoder = WhisperEncoder(config, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        encoder.eval()
        assert encoder.conv1.weight.value.dtype == jnp.float32
        assert encoder.conv2.weight.value.dtype == jnp.float32
        assert encoder.conv1.bias.value.dtype == jnp.float32
        assert encoder.conv2.bias.value.dtype == jnp.float32
        output = encoder(jnp.ones((1, 4, 8), dtype=jnp.bfloat16))
        assert output.last_hidden_state.shape == (1, 4, 8)
        assert output.last_hidden_state.dtype == jnp.bfloat16
        # The casts are transient operands, not mutations of checkpoint state.
        assert encoder.conv1.weight.value.dtype == jnp.float32
        assert encoder.conv2.weight.value.dtype == jnp.float32
        assert encoder.conv1.bias.value.dtype == jnp.float32
        assert encoder.conv2.bias.value.dtype == jnp.float32


def test_decode_mask_excludes_encoder_padding():
    config = _config()
    with config.mesh:
        model = WhisperForConditionalGeneration(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(2))
        model.eval()
        inputs = jnp.array([[1, 3, 4]], dtype=jnp.int32)
        encoder = jax.random.normal(jax.random.key(3), (1, 4, 8))
        changed = encoder.at[:, -1, :].set(jnp.arange(8) * 100.0)
        mask = jnp.array([[1, 1, 1, 0]], dtype=jnp.bool_)
        first = model.decode(inputs, (encoder,), encoder_attention_mask=mask).logits
        second = model.decode(inputs, (changed,), encoder_attention_mask=mask).logits
        np.testing.assert_allclose(np.asarray(first), np.asarray(second), atol=1e-5, rtol=1e-5)
        unmasked = model.decode(inputs, (changed,), encoder_attention_mask=jnp.ones_like(mask)).logits
        assert not np.allclose(np.asarray(first), np.asarray(unmasked), atol=1e-6, rtol=1e-6)
        with pytest.raises(ValueError, match="encoder_attention_mask must have shape"):
            model.decode(inputs, (encoder,), encoder_attention_mask=jnp.ones((1, 8)))


@pytest.mark.parametrize("stack", ["encoder", "decoder"])
def test_layerdrop_changes_across_compiled_training_calls(stack):
    config = _config(encoder_layerdrop=0.5, decoder_layerdrop=0.5)
    with config.mesh:
        cls = WhisperEncoder if stack == "encoder" else WhisperDecoder
        module = cls(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(7))
        module.train()
        inputs = (
            jnp.arange(32, dtype=jnp.float32).reshape(1, 4, 8) / 32 if stack == "encoder" else jnp.array([[1, 3, 4]])
        )

        @spx.jit(mutable="rng")
        def step(model, inputs):
            return model(inputs, output_hidden_states=True).hidden_states

        dropped = []
        for _ in range(16):
            states = step(module, inputs)
            dropped.append(np.array_equal(np.asarray(states[0]), np.asarray(states[1])))
        assert any(dropped)
        assert not all(dropped)


@pytest.mark.parametrize("prompt_length", [1, 4])
def test_normal_generate_forced_tokens_reach_shared_processor_builder(monkeypatch, prompt_length):
    def generate(self, inputs, generation_config, logits_processor=None, **kwargs):
        # Exercise the real shared normal-generation processor builder rather
        # than stopping at Whisper's prepared GenerationConfig.
        return self._get_logits_processor(generation_config, prompt_length, logits_processor)

    monkeypatch.setattr(BaseConditionalGenerationModule, "generate", generate)
    model = object.__new__(WhisperForConditionalGeneration)
    config = GenerationConfig(
        eos_token_id=2,
        is_multilingual=False,
        return_timestamps=False,
        no_timestamps_token_id=20,
        forced_decoder_ids=[(1, 7), (2, 8)],
    )
    object.__setattr__(model, "generation_config", config)
    processors = model.generate(None)
    forcing = [processor for processor in processors if isinstance(processor, ForceTokensLogitsProcessor)]
    assert len(forcing) == 1
    for offset, token in enumerate((7, 8, 20)):
        scores = forcing[0](jnp.zeros((1, prompt_length), dtype=jnp.int32), jnp.zeros((1, 32)), prompt_length + offset)
        assert int(jnp.argmax(scores[0])) == token
        assert int(jnp.sum(jnp.isfinite(scores[0]))) == 1
    assert config.forced_decoder_ids == [(1, 7), (2, 8)]


def test_force_generate_keeps_configs_and_caller_processors(monkeypatch):
    def generate(self, inputs, generation_config, logits_processor=None, **kwargs):
        return generation_config, logits_processor

    monkeypatch.setattr(BaseConditionalGenerationModule, "generate", generate)
    model = object.__new__(WhisperForConditionalGeneration)
    config = SimpleNamespace(forced_decoder_ids=[(1, 10)])
    object.__setattr__(model, "generation_config", config)
    original = copy.deepcopy(vars(config))
    user_processor = object()
    supplied = LogitsProcessorList([user_processor])
    result, processors = model._force_generate(None, jnp.array([[1, 3]]), logits_processor=supplied)
    assert vars(config) == original
    assert result.forced_decoder_ids is None
    assert processors[0] is user_processor
    assert len(supplied) == 1
    scores = processors[-1](jnp.array([[1, 0]]), jnp.zeros((1, 8)), 1)
    assert int(jnp.argmax(scores[0])) == 3
    assert int(jnp.sum(jnp.isfinite(scores[0]))) == 1
