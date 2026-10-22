"""Grok full/trace-safe/split-head logit parity (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import resolve_lm_logit_postprocess
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.grok_1 import Grok1Config, Grok1ForCausalLM
from jax import numpy as jnp


@pytest.mark.parametrize("clip_cap", [None, 0.25])
@pytest.mark.parametrize("multiplier", [0.5, 2.0])
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_grok_full_trace_safe_and_split_heads_scale_before_clipping(
    clip_cap, multiplier, tie_word_embeddings, chunk_size
):
    config = Grok1Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        output_multiplier_scale=multiplier,
        tie_word_embeddings=tie_word_embeddings,
        lmhead_chunksize=chunk_size,
    )
    with config.mesh:
        model = Grok1ForCausalLM(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
        )
        model._logit_cap_feature = None if clip_cap is None else LogitCapFeature(clip_cap)
        head_weight = model.get_embedding().weight if tie_word_embeddings else model.get_lm_head().weight
        head_weight.value = jnp.full_like(head_weight.value, 0.5)
        hidden = jnp.asarray(
            [[[0.1] * 8, [-0.1] * 8, [0.025] * 8, [-0.025] * 8]],
            dtype=jnp.float32,
        )
        raw = np.asarray(hidden) @ np.full((8, 16), 0.5, dtype=np.float32)
        scaled = raw * multiplier
        expected = scaled if clip_cap is None else np.clip(scaled, -clip_cap, clip_cap)
        np.testing.assert_allclose(model.apply_lm_head(hidden), scaled, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(model.compute_lm_logits(hidden), expected, atol=1e-6, rtol=1e-5)
        for stage in (None, 0):
            trace_safe_head = model.make_lm_head_fn(vocab_shard_stage=stage)
            np.testing.assert_allclose(jax.jit(trace_safe_head)(hidden), expected, atol=1e-6, rtol=1e-5)
        postprocess = resolve_lm_logit_postprocess(model)
        np.testing.assert_allclose(jax.jit(postprocess)(jnp.asarray(raw)), expected, atol=1e-6, rtol=1e-5)
