# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma-family norm and attention numerics against NumPy references (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.gemma2 import Gemma2Config
from easydel.modules.gemma2.modeling_gemma2 import Gemma2Attention
from easydel.modules.gemma3 import Gemma3TextConfig
from easydel.modules.gemma3.modeling_gemma3 import Gemma3Attention
from easydel.modules.gemma4 import Gemma4TextConfig
from easydel.modules.gemma4.modeling_gemma4 import Gemma4Attention, Gemma4RMSNorm
from easydel.modules.gemma4_assistant import Gemma4AssistantTextConfig
from easydel.modules.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantQOnlyAttention
from easydel.modules.gidd import GiddConfig
from easydel.modules.gidd.modeling_gidd import GiddRMSNorm
from jax import numpy as jnp
from spectrax import common_types


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("with_scale", [True, False])
def test_gemma4_norm_preserves_scale_and_output_dtype(dtype, with_scale):
    norm = Gemma4RMSNorm(dim=8, epsilon=1e-6, with_scale=with_scale, param_dtype=jnp.float32)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, 3, 8)), dtype)
    weight = np.linspace(0.5, 1.5, 8, dtype=np.float32)
    if with_scale:
        norm.weight.value = jnp.asarray(weight)
    ref_x = np.asarray(x, dtype=np.float32)
    expected = ref_x / np.sqrt(np.mean(ref_x**2, axis=-1, keepdims=True) + 1e-6)
    if with_scale:
        expected *= weight
    actual = norm(x)
    assert actual.dtype == jnp.float32
    assert actual.shape == x.shape
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gidd_norm_matches_offset_scale_reference(dtype):
    norm = GiddRMSNorm(GiddConfig(hidden_size=8), dtype=dtype, param_dtype=jnp.float32)
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 3, 8)), dtype)
    weight = np.linspace(-0.3, 0.3, 8, dtype=np.float32)
    norm.weight.value = jnp.asarray(weight)
    ref_x = np.asarray(x, dtype=np.float32)
    expected = (1.0 + weight) * ref_x / np.sqrt(np.mean(ref_x**2, axis=-1, keepdims=True) + 1e-6)
    actual = norm(x)
    assert actual.dtype == dtype
    expected = np.asarray(jnp.asarray(expected, dtype), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), expected, rtol=8e-3, atol=1e-5)


def _attention_reference(q, k, v, bias=None, scale=1.0, logits_soft_cap=None):
    # Independent NumPy QK/softmax/PV reference, including GQA's contiguous head grouping.
    repeats = q.shape[-2] // k.shape[-2]
    k = np.repeat(k, repeats, axis=-2)
    v = np.repeat(v, repeats, axis=-2)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) * scale
    if logits_soft_cap is not None and logits_soft_cap > 0:
        scores = logits_soft_cap * np.tanh(scores / logits_soft_cap)
    if bias is not None:
        scores += bias
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probabilities, v)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_assistant_registered_attention_matches_dense_masked_reference(kv_heads):
    cfg = Gemma4AssistantTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=kv_heads,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=16,
        layer_types=["full_attention"],
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    attn = Gemma4AssistantQOnlyAttention(
        cfg,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(2),
    )
    rng = np.random.default_rng(3)
    hidden = jnp.asarray(rng.normal(size=(1, 3, 16)), jnp.float32)
    k = rng.normal(size=(1, 5, kv_heads, 8)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    bias = np.zeros((1, 1, 3, 5), dtype=np.float32)
    bias[..., :2] = -1e10
    # All query positions are zero, so RoPE is the identity in this independent reference.
    q = np.asarray(attn.q_proj(hidden)).reshape(1, 3, 2, 8)
    q = q / np.sqrt(np.mean(q**2, axis=-1, keepdims=True) + cfg.rms_norm_eps)
    q *= np.asarray(attn.q_norm.weight.value)
    expected = _attention_reference(q, k, v, bias).reshape(1, 3, 16)
    expected = np.asarray(attn.o_proj(jnp.asarray(expected)))
    actual = attn(hidden, jnp.asarray(k), jnp.asarray(v), jnp.zeros((1, 3), jnp.int32), jnp.asarray(bias))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-4, atol=1e-5)


def test_gemma4_shared_kv_attention_matches_causal_reference():
    cfg = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        layer_types=["full_attention"] * 3,
        num_kv_shared_layers=1,
        max_position_embeddings=16,
        hidden_size_per_layer_input=0,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    attn = Gemma4Attention(
        cfg,
        layer_idx=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(4),
    )
    rng = np.random.default_rng(5)
    hidden = jnp.asarray(rng.normal(size=(1, 3, 16)), jnp.float32)
    k = rng.normal(size=(1, 3, 1, 8)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    q = np.asarray(attn.query_projection(hidden)).reshape(1, 3, 2, 8)
    q = q / np.sqrt(np.mean(q**2, axis=-1, keepdims=True) + cfg.rms_norm_eps)
    q *= np.asarray(attn.q_norm.weight.value)
    bias = np.where(np.tri(3, dtype=bool), 0.0, -1e10).astype(np.float32)[None, None]
    expected = _attention_reference(q, k, v, bias).reshape(1, 3, 16)
    expected = np.asarray(attn.output_projection(jnp.asarray(expected)))
    output = attn(
        hidden,
        mask_info=None,
        position_ids=jnp.zeros((1, 3), jnp.int32),
        mode=common_types.MODE_TRAIN,
        shared_key_value=(jnp.asarray(k), jnp.asarray(v)),
    )
    np.testing.assert_allclose(np.asarray(output.attention_output), expected, rtol=2e-4, atol=1e-5)


@pytest.mark.parametrize(
    "config_class, attention_class", [(Gemma2Config, Gemma2Attention), (Gemma3TextConfig, Gemma3Attention)]
)
@pytest.mark.parametrize("default_cap", [None, 5.0])
@pytest.mark.parametrize("call_cap", [None, 2.0, 0.0])
def test_gemma_models_shared_softcap_matches_reference(config_class, attention_class, default_cap, call_cap):
    cfg = config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        attn_logit_softcapping=default_cap,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    attention = attention_class(cfg, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(12))
    rng = np.random.default_rng(13)
    # The kernel scales Q before its contraction; the independent reference
    # scales QK afterwards. Half-integer inputs and the power-of-two scale
    # make both contraction orders exact in fp32, isolating soft-cap dispatch
    # rather than testing backend-specific rounding on large random logits.
    q = (np.round(20 * rng.normal(size=(1, 3, 2, 8))) / 2).astype(np.float32)
    k = (np.round(20 * rng.normal(size=(1, 5, 1, 8))) / 2).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    expected = _attention_reference(
        q.astype(np.float64),
        k.astype(np.float64),
        v.astype(np.float64),
        scale=cfg.query_pre_attn_scalar**-0.5,
        logits_soft_cap=default_cap if call_cap is None else call_cap,
    )
    for run in (attention.attention_performer.forward, attention.attention_performer):
        output = run(
            jnp.asarray(q),
            jnp.asarray(k),
            jnp.asarray(v),
            mode=common_types.MODE_TRAIN,
            causal=False,
            logits_soft_cap=call_cap,
            precision=jax.lax.Precision.HIGHEST,
        )
        np.testing.assert_allclose(np.asarray(output.attention_outputs), expected, rtol=2e-5, atol=1e-6)


def test_gemma2_low_precision_attention_retains_fp32_softmax_contract():
    cfg = Gemma2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.bfloat16,
        attn_logit_softcapping=None,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    attn = Gemma2Attention(cfg, layer_idx=0, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(6))
    rng = np.random.default_rng(7)
    q = rng.normal(size=(1, 3, 2, 8)).astype(np.float32)
    k = rng.normal(size=(1, 5, 1, 8)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    actual = attn.attention_performer(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        mode=common_types.MODE_TRAIN,
        causal=False,
        precision=jax.lax.Precision.HIGHEST,
    )
    expected = _attention_reference(q, k, v, scale=cfg.query_pre_attn_scalar**-0.5)
    np.testing.assert_allclose(np.asarray(actual.attention_outputs), expected, rtol=2e-5, atol=1e-6)
    assert cfg.attn_softmax_dtype == jnp.bfloat16
