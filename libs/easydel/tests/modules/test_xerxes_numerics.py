"""Hardware-only regressions for Xerxes sparse routing, attention and scanned RoPE."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.layers.moe import MoEMethods
from easydel.modules.xerxes.modeling_xerxes import XerxesAttention, XerxesModel, XerxesSparseMoeBlock
from easydel.modules.xerxes.xerxes_configuration import XerxesConfig
from easydel.modules.xerxes2.modeling_xerxes2 import Xerxes2Attention, Xerxes2ForCausalLM, Xerxes2Model
from easydel.modules.xerxes2.xerxes2_configuration import Xerxes2Config
from jax import numpy as jnp


def _config(num_experts_per_tok=2, **kwargs):
    return XerxesConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_local_experts=4,
        num_experts_per_tok=num_experts_per_tok,
        max_position_embeddings=16,
        xe_kvnorm=True,
        swish_run=False,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


@pytest.mark.parametrize("method", [MoEMethods.STANDARD_MOE, MoEMethods.FUSED_MOE, MoEMethods.DENSE_MOE])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("module_experts", [False, True])
def test_sparse_experts_match_dense_topk_reference_without_checkpoint_changes(method, top_k, module_experts):
    config = _config(
        moe_method=method,
        num_experts_per_tok=top_k,
        use_scan_mlp=module_experts,
        scan_mlp_chunk_size=1,
        initializer_range=0.2,
    )
    with config.mesh:
        # TPU's default f32 dot uses reduced-precision MXU operands. This is
        # a routing-math parity test against NumPy, so request full precision
        # explicitly rather than weakening the output/gradient tolerances.
        model = XerxesSparseMoeBlock(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
        )
        inputs = jax.random.normal(jax.random.key(2), (1, 3, 8))
        actual, logits = model(inputs)
        # Independent NumPy selected-logit softmax and dense expert reference.
        # Larger initialization makes probability-scaled top-1 outputs fail the
        # output tolerance rather than hiding the regression behind tiny MLPs.
        x = np.asarray(inputs)
        reference_logits = x @ np.asarray(model.gate.weight.value)
        selected = np.argsort(-reference_logits, axis=-1)[..., :top_k]
        top_logits = np.take_along_axis(reference_logits, selected, axis=-1)
        weights = np.exp(top_logits - np.max(top_logits, axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expert_outputs = []
        for expert in model.experts:
            gate_weight, up_weight = np.split(np.asarray(expert.gate_up_proj.weight.value), 2, axis=-1)
            gate = x @ gate_weight
            gelu = 0.5 * gate * (1 + np.tanh(np.sqrt(2 / np.pi) * (gate + 0.044715 * gate**3)))
            expert_outputs.append((gelu * (x @ up_weight)) @ np.asarray(expert.down_proj.weight.value))
        expert_outputs = np.stack(expert_outputs, axis=-2)
        selected_outputs = np.take_along_axis(expert_outputs, selected[..., None], axis=-2)
        expected = np.sum(weights[..., None] * selected_outputs, axis=-2)
        np.testing.assert_allclose(np.asarray(logits), reference_logits, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6, rtol=1e-4)

        # Analytic softmax derivative for sum(output), with expert outputs held
        # fixed: top-1 has exactly zero task gradient with respect to the router.
        scores = selected_outputs.sum(axis=-1)
        selected_grad = weights * (scores - np.sum(weights * scores, axis=-1, keepdims=True))
        logit_grad = np.zeros_like(reference_logits)
        for index in range(config.num_local_experts):
            logit_grad[..., index] = np.sum(np.where(selected == index, selected_grad, 0), axis=-1)
        expected_gate_grad = x.reshape(-1, 8).T @ logit_grad.reshape(-1, config.num_local_experts)
        grads = spx.grad(lambda block: jnp.sum(block(inputs)[0]))(model).flatten()
        np.testing.assert_allclose(np.asarray(grads["parameters/gate.weight"]), expected_gate_grad, atol=1e-7, rtol=2e-4)
        if top_k == 1:
            np.testing.assert_array_equal(weights, np.ones_like(weights))
        state = spx.export(model)[1].flatten()
        paths = {path.split("/", 1)[1] for path in state if "experts" in path}
        assert paths == {
            f"experts.{index}.{projection}.weight" for index in range(4) for projection in ("gate_up_proj", "down_proj")
        }


def test_xerxes_inputs_embeds_matches_token_input():
    config = _config(xe_moe=False)
    with config.mesh:
        model = XerxesModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(5))
        model.eval()
        inputs = jnp.array([[1, 2, 3]])
        direct = model(input_ids=inputs).last_hidden_state
        embedded = model(inputs_embeds=model.embed_tokens(inputs)).last_hidden_state
        np.testing.assert_allclose(np.asarray(direct), np.asarray(embedded), atol=1e-6, rtol=1e-5)
        with pytest.raises(ValueError):
            model(input_ids=inputs, inputs_embeds=model.embed_tokens(inputs))
        with pytest.raises(ValueError):
            model()


def test_noncausal_and_cross_attention_observe_future_and_encoder_states():
    config = _config()
    with config.mesh:
        inputs = jax.random.normal(jax.random.key(8), (1, 3, 8))
        changed = inputs.at[:, -1, :].add(jnp.arange(8, dtype=jnp.float32) * 4)
        positions = jnp.arange(3)[None, :]
        causal = XerxesAttention(config, 0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(9))
        noncausal = XerxesAttention(
            config, 0, dtype=jnp.float32, param_dtype=jnp.float32, causal=False, rngs=spx.Rngs(9)
        )
        for module, should_change in [(causal, False), (noncausal, True)]:
            module.eval()
            first = module(inputs, None, positions, "train").attention_output[:, 0]
            second = module(changed, None, positions, "train").attention_output[:, 0]
            assert bool(not np.allclose(np.asarray(first), np.asarray(second), atol=1e-6)) == should_change
        cross = XerxesAttention(
            config, 0, dtype=jnp.float32, param_dtype=jnp.float32, is_cross_attention=True, rngs=spx.Rngs(9)
        )
        cross.eval()
        first = cross(inputs[:, :1], None, positions[:, :1], "train", key_value_states=inputs).attention_output
        second = cross(inputs[:, :1], None, positions[:, :1], "train", key_value_states=changed).attention_output
        assert first.shape == (1, 1, 8)
        assert not np.allclose(np.asarray(first), np.asarray(second), atol=1e-6)


@pytest.mark.parametrize("q_lora_dim", [None, 4])
def test_xerxes2_shared_mla_preserves_centered_layernorm_and_shapes(q_lora_dim):
    config = Xerxes2Config(
        hidden_size=8,
        num_attention_heads=2,
        q_lora_dim=q_lora_dim,
        kv_lora_dim=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        vhead_dim=4,
        max_position_embeddings=16,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    with config.mesh:
        attention = Xerxes2Attention(config, 0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3))
        values = jnp.array([[1.0, 2.0, 4.0, 7.0]])
        normalized = attention.kv_a_layernorm(values)
        expected = (values - jnp.mean(values, axis=-1, keepdims=True)) / jnp.sqrt(
            jnp.var(values, axis=-1, keepdims=True) + attention.kv_a_layernorm.epsilon
        )
        np.testing.assert_allclose(np.asarray(normalized), np.asarray(expected), atol=1e-5, rtol=1e-5)
        assert attention.kv_a_layernorm.bias.value.shape == (4,)
        if q_lora_dim is not None:
            assert attention.q_a_layernorm.bias.value.shape == (4,)
        result = attention(jnp.ones((1, 3, 8)), None, jnp.arange(3)[None, :], "train")
        assert result.attention_output.shape == (1, 3, 8)


def test_xerxes2_scanned_frequencies_can_be_reused_after_first_forward():
    config = Xerxes2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        q_lora_dim=4,
        kv_lora_dim=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        vhead_dim=4,
        mlp_only_layers=[0, 1],
        num_experts=2,
        num_experts_per_tok=1,
        scan_layers=True,
        max_position_embeddings=16,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    with config.mesh:
        model = Xerxes2Model(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(4))
        model.eval()
        inputs = jnp.array([[1, 2, 3]])
        with jax.checking_leaks():
            first = model(inputs).last_hidden_state
            second = model(inputs).last_hidden_state
        np.testing.assert_allclose(np.asarray(first), np.asarray(second), atol=1e-6, rtol=1e-5)
        wrapper = object.__new__(Xerxes2ForCausalLM)
        object.__setattr__(wrapper, "config", config)
        cache = wrapper.create_transformer_cache_config(1, 8)
        assert cache.key_dim == 8
        assert cache.value_dim == 4
