"""Falcon-H1 projection/scale/clip numerical regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base._features import LogitCapFeature
from easydel.modules.falcon_h1 import FalconH1Config, FalconH1ForCausalLM
from jax import numpy as jnp


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    # Tied Embed.attend uses jnp.dot without an explicit precision argument.
    # Pin its default too: the reference tests order-of-operations, not TPU's
    # reduced-precision default dot products on non-bf16-exact decimal inputs.
    with jax.default_matmul_precision("highest"):
        yield


@pytest.mark.parametrize("clip_cap", [None, 0.25])
@pytest.mark.parametrize("multiplier", [0.5, 2.0])
@pytest.mark.parametrize("tie_word_embeddings", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_falcon_h1_full_trace_safe_and_split_heads_scale_before_clipping(
    clip_cap, multiplier, tie_word_embeddings, chunk_size
):
    config = FalconH1Config(
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
        lm_head_multiplier=multiplier,
        tie_word_embeddings=tie_word_embeddings,
        lmhead_chunksize=chunk_size,
    )
    model = FalconH1ForCausalLM(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )
    # Falcon-H1 inherits the optional base clipping feature; its existing
    # constructor does not expose a separate cap argument.
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
    postprocess = model.get_lm_logit_postprocess()
    np.testing.assert_allclose(jax.jit(postprocess)(jnp.asarray(raw)), expected, atol=1e-6, rtol=1e-5)
