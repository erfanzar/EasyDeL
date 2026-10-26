# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Mixtral / Mistral4 / MiniMax-M3-VL regressions that execute JAX numerics (accelerator test)."""

from types import MethodType, SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base._features import RouterAuxLossFeature
from easydel.modules.minimax_m3_vl.modeling_minimax_m3_vl import _apply_partial_rotary
from easydel.modules.mistral4 import Mistral4Config
from easydel.modules.mistral4.modeling_mistral4 import Mistral4ForCausalLM, Mistral4Model, Mistral4MoE
from easydel.modules.mixtral import MixtralConfig
from easydel.modules.mixtral.modeling_mixtral import MixtralModel
from jax import numpy as jnp


def _single_device(config):
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    return config


def _tiny_mistral4(**kwargs):
    return _single_device(
        Mistral4Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            kv_lora_rank=8,
            q_lora_rank=16,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            max_position_embeddings=64,
            **kwargs,
        )
    )


def test_mixtral_cached_frequencies_are_concrete_after_forward():
    config = _single_device(
        MixtralConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
        )
    )
    with config.mesh:
        model = MixtralModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % 64
        with jax.checking_leaks():
            out = model(input_ids=input_ids).last_hidden_state
    cached = model.frequencies.value
    assert isinstance(cached, jax.Array)
    assert not isinstance(cached, jax.core.Tracer)
    assert out.shape == (1, 8, config.hidden_size)


def test_mixtral_initializer_range_controls_expert_and_router_init():
    def _std(initializer_range):
        config = _single_device(
            MixtralConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_local_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=32,
                initializer_range=initializer_range,
            )
        )
        with config.mesh:
            model = MixtralModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
            moe = model.layers[0].block_sparse_moe
            return (
                float(jnp.std(moe.gate.weight.value)),
                float(jnp.std(moe.experts.gate_up_proj.kernel_view())),
                float(jnp.std(moe.experts.w2.kernel_view())),
            )

    small = _std(0.005)
    large = _std(0.2)
    for small_std, large_std in zip(small, large, strict=True):
        assert large_std > 10 * small_std


@pytest.mark.parametrize("initialize", [None, False, True])
@pytest.mark.parametrize("loading_only", [False, True])
def test_mixtral_moe_placeholders_require_explicit_loading_only_mode(initialize, loading_only):
    config = _single_device(
        MixtralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=16,
            moe_checkpoint_loading=loading_only,
            **({} if initialize is None else {"initialization_of_moe": initialize}),
        )
    )
    with config.mesh:
        model = MixtralModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    moe = model.layers[0].block_sparse_moe
    for kernel in (moe.gate.weight.value, moe.experts.gate_up_proj.kernel_view(), moe.experts.w2.kernel_view()):
        assert bool(jnp.any(kernel != 0)) == (bool(initialize) or not loading_only)


def test_default_mixtral_moe_has_nonzero_task_gradients():
    config = _single_device(
        MixtralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=2,
            num_experts_per_tok=2,
            max_position_embeddings=16,
            moe_method="dense_moe",
        )
    )
    with config.mesh:
        model = MixtralModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        moe = model.layers[0].block_sparse_moe
        inputs = jnp.asarray(np.random.default_rng(12).normal(size=(1, 4, 16)).astype(np.float32))

        def loss(block):
            output, _ = block(inputs)
            return jnp.sum(output**2)

        grads = spx.grad(loss)(moe)
    for part in ("gate", "gate_up_proj", "w2"):
        part_grads = [value for _collection, path, value in grads.items() if part in path.split(".")]
        assert part_grads, f"No gradients returned for {part}"
        assert all(bool(jnp.all(jnp.isfinite(value))) for value in part_grads)
        assert any(bool(jnp.any(value != 0)) for value in part_grads), f"Zero task gradient for {part}"


@pytest.mark.parametrize("override", [None, False, True])
def test_mistral4_model_honours_config_output_flags_and_explicit_overrides(override):
    config = _tiny_mistral4(output_attentions=True, output_hidden_states=True, output_router_logits=True)
    with config.mesh:
        model = Mistral4Model(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        out = model(
            input_ids=jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % 64,
            output_attentions=override,
            output_hidden_states=override,
            output_router_logits=override,
        )
    if override is False:
        assert out.hidden_states is None
        assert out.attentions is None
        assert out.router_logits is None
    else:
        assert out.hidden_states is not None and len(out.hidden_states) == config.num_hidden_layers + 1
        assert out.attentions is not None and len(out.attentions) == config.num_hidden_layers
        assert out.router_logits is not None and len(out.router_logits) == 1  # one MoE layer


@pytest.mark.parametrize("debug", [False, True])
def test_mistral4_tensor_debug_effects_require_explicit_opt_in(debug):
    config = _tiny_mistral4(moe_debug=debug, moe_method="standard_moe")
    with config.mesh:
        block = Mistral4MoE(config, layer_idx=7, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        inputs = jnp.ones((1, 2, config.hidden_size), dtype=jnp.float32)
        program = jax.make_jaxpr(lambda x: block(x)[0])(inputs)
    # Debug effects represent host callbacks in the compiled forward, not log-message formatting.
    has_debug_effect = any("Debug" in type(effect).__name__ for effect in program.effects)
    assert has_debug_effect is debug


@pytest.mark.parametrize("with_mask", [False, True])
def test_mistral4_aux_loss_accepts_per_layer_router_logits(with_mask):
    config = _tiny_mistral4(output_router_logits=True, router_aux_loss_coef=0.01)
    with config.mesh:
        model = Mistral4ForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        mask = jnp.asarray([[1, 1, 1, 1, 1, 0, 0, 0]]) if with_mask else None
        out = model(input_ids=jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % 64, attention_mask=mask)
    assert out.aux_loss is not None
    assert out.aux_loss.shape == ()
    assert bool(jnp.isfinite(out.aux_loss))


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("topk_group", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_mistral4_shared_selector_matches_reference_grouped_topk(k, topk_group, normalize):
    n_experts, n_group, scale = 8, 4, 2.5
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(16, n_experts)).astype(np.float32)
    scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)

    config = _tiny_mistral4(routed_scaling_factor=scale, norm_topk_prob=normalize)
    config.n_routed_experts = n_experts
    config.n_group = n_group
    config.topk_group = topk_group
    config.num_experts_per_tok = k
    with config.mesh:
        block = Mistral4MoE(config, layer_idx=1, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        weights, indices = block.moe_hooks.select_hook(jnp.asarray(scores), None, k)

    grouped = scores.reshape(16, n_group, n_experts // n_group)
    group_score = np.sort(grouped, axis=-1)[..., -2:].sum(-1)
    keep_groups = np.argsort(-group_score, axis=-1)[:, :topk_group]
    for row in range(16):
        allowed = {g * (n_experts // n_group) + j for g in keep_groups[row] for j in range(n_experts // n_group)}
        order = sorted(allowed, key=lambda e: -scores[row, e])[:k]
        row_weights, row_indices = np.asarray(weights[row]), np.asarray(indices[row])
        assert row_weights.shape == row_indices.shape == (k,)
        active = row_weights > 0
        assert sorted(row_indices[active].tolist()) == sorted(order)
        assert np.count_nonzero(~active) == k - len(order)
        expected = scores[row, order]
        if normalize:
            expected = expected / (expected.sum() + 1e-20)
        expected = np.pad(expected * scale, (0, k - len(order)))
        np.testing.assert_allclose(np.sort(row_weights), np.sort(expected), rtol=1e-5, atol=1e-6)
        # Excluded groups must contribute nothing to the combined expert output.
        expert_values = np.arange(n_experts, dtype=np.float32) + 1
        np.testing.assert_allclose(
            np.sum(row_weights * expert_values[row_indices]),
            np.sum(expected[: len(order)] * expert_values[order]),
            rtol=1e-5,
            atol=1e-6,
        )


def test_minimax_m3_partial_rotary_matches_shared_split_half():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(1, 5, 2, 16)).astype(np.float32))
    angles = jnp.asarray(rng.normal(size=(1, 5, 1, 8)).astype(np.float32))
    cos, sin = jnp.cos(angles), jnp.sin(angles)

    out = _apply_partial_rotary(x, cos, sin)
    leading = np.asarray(x)[..., :8]
    rotated = np.concatenate((-leading[..., 4:], leading[..., :4]), axis=-1)
    expected = leading * np.asarray(cos) + rotated * np.asarray(sin)
    np.testing.assert_allclose(np.asarray(out[..., :8]), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(out[..., 8:]), np.asarray(x[..., 8:]))


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("with_mask", [False, True])
def test_mistral4_moe_returns_raw_logits_and_aux_loss_matches_independent_reference(with_mask):
    config = _tiny_mistral4(moe_method="dense_moe", router_aux_loss_coef=0.03)
    inputs = np.random.default_rng(38).normal(size=(1, 4, config.hidden_size)).astype(np.float32)
    with config.mesh:
        block = Mistral4MoE(
            config,
            layer_idx=1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        # Nonuniform routing makes double-softmax distinguishable from the
        # reference objective, unlike an almost-uniform random initializer.
        weight = np.random.default_rng(9).normal(size=block.gate.weight.value.shape).astype(np.float32)
        block.gate.weight.value = jnp.asarray(weight)
        _, raw = block(jnp.asarray(inputs))
    expected_logits = inputs.reshape(-1, config.hidden_size) @ weight
    np.testing.assert_allclose(np.asarray(raw), expected_logits, rtol=1e-5, atol=1e-5)
    mask = np.asarray([[1, 1, 1, 0]], dtype=np.int32) if with_mask else None
    # Mistral4 uses the shared task-module router loss; drive the real methods.
    owner = SimpleNamespace(config=config, _router_aux_loss_feature=RouterAuxLossFeature(config.router_aux_loss_coef))
    for name in ("compute_router_aux_loss", "compute_indexer_aux_loss", "_router_top_k"):
        setattr(owner, name, MethodType(getattr(Mistral4ForCausalLM, name), owner))
    actual = Mistral4ForCausalLM.compute_aux_loss(
        owner,
        SimpleNamespace(router_logits=(raw,)),
        attention_mask=jnp.asarray(mask) if mask is not None else None,
    )
    probabilities = np.exp(expected_logits - expected_logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    selected = np.argsort(-probabilities, axis=-1)[:, : config.num_experts_per_tok]
    one_hot = np.eye(config.n_routed_experts)[selected]
    valid = (
        np.ones(len(inputs.reshape(-1, config.hidden_size)), dtype=bool) if mask is None else mask.ravel().astype(bool)
    )
    expected = config.n_routed_experts * np.sum(one_hot[valid].mean(0) * probabilities[valid].mean(0))
    expected *= config.router_aux_loss_coef
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)
