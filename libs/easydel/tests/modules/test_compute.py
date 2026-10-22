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

"""Module computation regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.hunyuan_vl.modeling_hunyuan_vl import merge_multimodal_embeddings
from easydel.modules.internlm2 import InternLM2Config
from easydel.modules.internlm2.modeling_internlm2 import InternLM2Model
from easydel.modules.kimi_linear import KimiLinearConfig
from easydel.modules.kimi_linear.modeling_kimi_linear import (
    KimiAttentionResidual,
    KimiLinearModel,
    KimiMLAAttention,
    KimiRMSNorm,
)
from jax import numpy as jnp


def test_internlm2_model_accepts_inputs_embeds():
    config = InternLM2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    with config.mesh:
        model = InternLM2Model(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        from_ids = model(input_ids=ids).last_hidden_state
        embeds = model.tok_embeddings(ids)
        from_embeds = model(inputs_embeds=embeds).last_hidden_state
        with pytest.raises(ValueError, match="must specify either one"):
            model()
        with pytest.raises(ValueError, match="must specify either one"):
            model(input_ids=ids, inputs_embeds=embeds)
    np.testing.assert_allclose(np.asarray(from_ids), np.asarray(from_embeds), rtol=1e-5, atol=1e-5)


def _kimi_config(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=16,
        q_lora_rank=None,
        mla_use_nope=False,
        linear_attn_config={
            "full_attn_layers": [2, 4],
            "kda_layers": [1, 3],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
    )
    kwargs.update(overrides)
    return KimiLinearConfig(**kwargs)


def test_attention_residual_starts_as_uniform_average():
    config = _kimi_config()
    mixer = KimiAttentionResidual(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    candidates = jax.random.normal(jax.random.PRNGKey(0), (3, 1, 4, config.hidden_size), dtype=jnp.float32)
    mixed = mixer(candidates)
    np.testing.assert_allclose(np.asarray(mixed), np.asarray(candidates.mean(axis=0)), rtol=1e-5, atol=1e-5)


def test_kimi_model_runs_with_attention_residuals():
    config = _kimi_config(attn_res_block_size=2)
    with config.mesh:
        model = KimiLinearModel(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % config.vocab_size
        out = model(input_ids=ids, output_hidden_states=True)
    assert out.last_hidden_state.shape == (1, 8, config.hidden_size)
    assert bool(jnp.all(jnp.isfinite(out.last_hidden_state)))
    assert len(out.hidden_states) == config.num_hidden_layers + 1
    np.testing.assert_array_equal(np.asarray(out.hidden_states[-1]), np.asarray(out.last_hidden_state))


@pytest.mark.parametrize("block_size", [None, 1, 2, 3])
def test_kimi_residual_paths_accept_embeddings_and_jit(block_size):
    config = _kimi_config(attn_res_block_size=block_size)
    with config.mesh:
        model = KimiLinearModel(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        ids = jnp.arange(4, dtype=jnp.int32).reshape(1, 4)
        eager = model(input_ids=ids).last_hidden_state
        embeds = model.embed_tokens(ids)
        from_embeds = model(inputs_embeds=embeds).last_hidden_state
        compiled = spx.jit(lambda module, tokens: module(input_ids=tokens).last_hidden_state)(model, ids)
    np.testing.assert_allclose(np.asarray(from_embeds), np.asarray(eager), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_kimi_attention_residual_two_stage_matches_non_pipelined(block_size):
    """Cross a PP edge with completed candidates and with an open partial block."""
    if jax.device_count() < 2 or jax.device_count() % 2:
        pytest.skip("requires an even accelerator device count of at least two")
    kwargs = dict(
        attn_res_block_size=block_size,
        linear_attn_config={"kda_layers": [], "full_attn_layers": [1, 2, 3, 4]},
        max_position_embeddings=64,
        attn_mechanism="vanilla",
    )
    reference_config = _kimi_config(**kwargs, sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    pipeline_config = _kimi_config(**kwargs, sharding_axis_dims=(2, 1, -1, 1, 1, 1))
    ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    # Float32 storage alone does not request full-precision TPU dots. Eager and
    # fused MPMD contractions may otherwise select different reduced-precision
    # paths; isolate candidate transport from that numerical-policy difference.
    # The context also covers the residual mixer's two direct einsums.
    with jax.default_matmul_precision("highest"), reference_config.mesh:
        reference = KimiLinearModel(
            reference_config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        # Nonuniform mixing makes lost or misplaced candidates observable beyond the zero-query initialization.
        for index, layer in enumerate(reference.layers):
            for mixer in (layer.attn_res, layer.mlp_res):
                query = np.linspace(-0.2, 0.3, reference_config.hidden_size, dtype=np.float32) * (index + 1)
                mixer.weight.value = jax.device_put(query, mixer.weight.value.sharding)
        expected = reference(input_ids=ids, output_hidden_states=True)
        reference_leaves = {path: np.asarray(var.value) for path, var in spx.iter_variables(reference)}
    with jax.default_matmul_precision("highest"), pipeline_config.mesh:
        pipeline = KimiLinearModel(
            pipeline_config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
        )
        # Check constructor placement before importing weights: a stage stamp alone does
        # not place a raw ArrayParam on the stage's devices without a tensor layout.
        for layer in pipeline.layers:
            layer_devices = layer.input_layernorm.weight.value.sharding.device_set
            for mixer in (layer.attn_res, layer.mlp_res):
                assert mixer.weight.value.sharding.device_set == layer_devices
                assert mixer.weight.value.sharding.is_fully_replicated
        pipeline_leaves = dict(spx.iter_variables(pipeline))
        assert pipeline_leaves.keys() == reference_leaves.keys()
        for path, var in pipeline_leaves.items():
            var.value = jax.device_put(reference_leaves[path], var.value.sharding)
        stage_devices = [layer.input_layernorm.weight.value.sharding.device_set for layer in pipeline.layers]
        assert stage_devices[0].isdisjoint(stage_devices[2])
        eager = pipeline(input_ids=ids, output_hidden_states=True)
        compiled = spx.jit(
            lambda module, tokens: module(input_ids=tokens).last_hidden_state,
            mesh=pipeline_config.mesh,
        )(pipeline, ids)
    np.testing.assert_allclose(
        np.asarray(eager.last_hidden_state), np.asarray(expected.last_hidden_state), rtol=1e-4, atol=1e-5
    )
    for actual, wanted in zip(eager.hidden_states, expected.hidden_states, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(wanted), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(expected.last_hidden_state), rtol=1e-4, atol=1e-5)


def test_kimi_mla_projection_bias_follows_config():
    config = _kimi_config(attention_bias=True)
    attn = KimiMLAAttention(config, layer_idx=1, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert attn.kv_a_proj_with_mqa.bias is not None
    assert attn.o_proj.bias is not None

    plain = KimiMLAAttention(_kimi_config(), layer_idx=1, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    assert plain.kv_a_proj_with_mqa.bias is None
    assert plain.o_proj.bias is None


def test_attention_residual_matches_independent_softmax_reference():
    config = _kimi_config()
    mixer = KimiAttentionResidual(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    candidates = np.linspace(-2, 3, 3 * 2 * config.hidden_size, dtype=np.float32).reshape(3, 1, 2, config.hidden_size)
    query = np.linspace(-0.2, 0.3, config.hidden_size, dtype=np.float32)
    mixer.weight.value = jnp.asarray(query)
    keys = candidates / np.sqrt(np.mean(candidates**2, axis=-1, keepdims=True) + config.rms_norm_eps)
    scores = np.einsum("d,nbtd->nbt", query, keys)
    weights = np.exp(scores - scores.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    expected = np.einsum("nbt,nbtd->btd", weights, candidates)
    np.testing.assert_allclose(np.asarray(mixer(jnp.asarray(candidates))), expected, rtol=1e-5, atol=1e-5)


def test_kimi_norm_preserves_fp32_multiply_before_cast():
    norm = KimiRMSNorm(hidden_size=4, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    values = jnp.asarray([[0.125, -2.5, 0.3, 4.0]], dtype=jnp.bfloat16)
    scale = np.array([1.37, 0.19, 2.71, -0.43], dtype=np.float32)
    norm.weight.value = jnp.asarray(scale)
    host_values = np.asarray(values).astype(np.float32)
    expected = host_values / np.sqrt(np.mean(host_values**2, axis=-1, keepdims=True) + 1e-6)
    expected = np.asarray(jnp.asarray(expected * scale, dtype=jnp.bfloat16))
    result = norm(values)
    assert result.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_hunyuan_merge_validates_placeholder_count_and_preserves_text():
    ids = jnp.asarray([[1, 99, 2, 99]], dtype=jnp.int32)
    text = jnp.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
    image = jnp.asarray([[10.0, 11.0], [12.0, 13.0]])
    result = merge_multimodal_embeddings(ids, text, image, 99)
    np.testing.assert_array_equal(np.asarray(result), [[[1.0, 2.0], [10.0, 11.0], [5.0, 6.0], [12.0, 13.0]]])
    with pytest.raises(ValueError, match=r"(?i)placeholder|token|embedding"):
        merge_multimodal_embeddings(ids, text, image[:1], 99)
