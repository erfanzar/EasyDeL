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

"""MoE router aux-loss behaviour of DeepSeek-V2 / V3 causal LMs (computation test: run on TPU/GPU)."""

import json

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.modeling_outputs import MoeModelOutput
from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2MoE
from easydel.modules.deepseek_v3.deepseek_configuration import DeepseekV3Config
from easydel.modules.deepseek_v3.modeling_deepseek import DeepseekV3ForCausalLM
from jax import numpy as jnp


def _ids():
    return jnp.arange(8, dtype=jnp.int32)[None, :] % 64


def _v2_model(**overrides):
    config = DeepseekV2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=None,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        first_k_dense_replace=0,
        max_position_embeddings=64,
        scan_layers=False,
        layer_types=["full_attention", "full_attention"],
        **overrides,
    )
    config.moe_force_xla_gmm = True
    return DeepseekV2ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _v3_model(router_aux_loss_coef=None, aux_loss_alpha=0.001, seq_aux=True):
    config = DeepseekV3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=None,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        first_k_dense_replace=0,
        num_nextn_predict_layers=0,
        aux_loss_alpha=aux_loss_alpha,
        seq_aux=seq_aux,
        max_position_embeddings=64,
        scan_layers=False,
        layer_types=["full_attention", "full_attention"],
    )
    config.moe_force_xla_gmm = True
    if router_aux_loss_coef is not None:
        config.router_aux_loss_coef = router_aux_loss_coef
    return DeepseekV3ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _v2_reference_loss(layer_logits, config, mask):
    """Independent DeepSeek assignment-frequency objective, reduced per layer."""
    losses = []
    experts, top_k = config.n_routed_experts, config.num_experts_per_tok
    for logits in layer_logits:
        logits = np.asarray(logits).reshape(-1, experts)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        eligible = probabilities.copy()
        if config.topk_method == "group_limited_greedy":
            groups = eligible.reshape(-1, config.n_group, experts // config.n_group)
            chosen = np.argsort(groups.max(axis=-1), axis=-1)[:, -config.topk_group :]
            allowed = np.zeros(groups.shape[:2], dtype=bool)
            np.put_along_axis(allowed, chosen, True, axis=-1)
            eligible = np.where(np.repeat(allowed, experts // config.n_group, axis=-1), eligible, -np.inf)
        selected = np.argsort(eligible, axis=-1)[:, -top_k:]
        counts = np.eye(experts)[selected].sum(axis=-2)
        loss_mask = mask if config.seq_aux else mask.reshape(1, -1)
        counts = counts.reshape(*loss_mask.shape, experts)
        probabilities = probabilities.reshape(*loss_mask.shape, experts)
        token_count = np.maximum(loss_mask.sum(axis=-1, keepdims=True), 1)
        frequency = (counts * loss_mask[..., None]).sum(axis=1) / (top_k * token_count)
        mean_probability = (probabilities * loss_mask[..., None]).sum(axis=1) / token_count
        losses.append(experts * (frequency * mean_probability).sum(axis=-1).mean())
    return np.mean(losses)


def _v2_routing_diagnostics(records, config):
    """Serialize actual selector intermediates only when the independent assertion fails."""
    diagnostics = []
    for logits, probabilities, topk_calls, weights, selected in records:
        host_logits = np.asarray(logits)
        host_probabilities = np.exp(host_logits - host_logits.max(axis=-1, keepdims=True))
        host_probabilities /= host_probabilities.sum(axis=-1, keepdims=True)
        diagnostic = {
            "logits_sharding": str(logits.sharding),
            "selection_sharding": str(selected.sharding),
            "logits": host_logits.tolist(),
            "fp32_probabilities": np.asarray(probabilities).tolist(),
            "numpy_probabilities": host_probabilities.tolist(),
            "selected_experts": np.asarray(selected).tolist(),
            "selected_weights": np.asarray(weights).tolist(),
            "assignment_counts": np.eye(config.n_routed_experts)[np.asarray(selected)].sum(axis=-2).tolist(),
        }
        if config.topk_method == "group_limited_greedy":
            group_scores, (_, selected_groups) = topk_calls[0]
            masked_scores, _ = topk_calls[1]
            host_groups = host_probabilities.reshape(-1, config.n_group, config.n_routed_experts // config.n_group).max(
                axis=-1
            )
            diagnostic.update(
                group_scores=np.asarray(group_scores).tolist(),
                selected_groups=np.asarray(selected_groups).tolist(),
                numpy_selected_groups=np.argsort(host_groups, axis=-1)[:, -config.topk_group :].tolist(),
                eligibility=np.isfinite(np.asarray(masked_scores)).tolist(),
                masked_scores=np.asarray(masked_scores).tolist(),
            )
        diagnostics.append(diagnostic)
    return json.dumps(diagnostics)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("seq_aux", [False, True])
@pytest.mark.parametrize("topk_method", ["greedy", "group_limited_greedy"])
def test_v2_router_logits_forward_uses_aux_loss_alpha(masked, seq_aux, topk_method, monkeypatch):
    """Use real routed assignments and the configured reduction, not the generic Switch objective."""
    records = []
    selector = DeepseekV2MoE._select_experts_static

    def record_selection(logits, *args, **kwargs):
        # Fused dispatch traces this function inside shard_map. Observe only
        # concrete calls (including the auxiliary objective); never leak tracers.
        if isinstance(logits, jax.core.Tracer):
            return selector(logits, *args, **kwargs)
        probabilities, topk_calls = [], []
        softmax, top_k, sort_key_val = jax.nn.softmax, jax.lax.top_k, jax.lax.sort_key_val

        def record_softmax(*inputs, **options):
            scores = softmax(*inputs, **options)
            probabilities.append(scores)
            return scores

        def record_top_k(operand, *inputs, **options):
            result = top_k(operand, *inputs, **options)
            topk_calls.append((operand, result))
            return result

        def record_group_sort(keys, values, *inputs, **options):
            result = sort_key_val(keys, values, *inputs, **options)
            groups = kwargs["topk_group"]
            topk_calls.append((-keys, (-result[0][:, :groups], result[1][:, :groups])))
            return result

        # These wrappers return the original operation results unchanged and
        # are restored even on failure. They observe the real group mask and
        # selection, rather than trying to reconstruct a device discrepancy.
        with monkeypatch.context() as observation:
            observation.setattr(jax.nn, "softmax", record_softmax)
            observation.setattr(jax.lax, "top_k", record_top_k)
            observation.setattr(jax.lax, "sort_key_val", record_group_sort)
            weights, selected = selector(logits, *args, **kwargs)
        records.append((logits, probabilities[0], topk_calls, weights, selected))
        return weights, selected

    monkeypatch.setattr(DeepseekV2MoE, "_select_experts_static", staticmethod(record_selection))
    model = _v2_model(seq_aux=seq_aux, topk_method=topk_method, n_group=2, topk_group=1)
    mask = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=np.float32)
    if not masked:
        mask = np.ones_like(mask)
    ids = jnp.concatenate([_ids(), _ids()[:, ::-1]] * 2, axis=0)
    out = model(input_ids=ids, attention_mask=jnp.asarray(mask) if masked else None, output_router_logits=True)
    assert out.router_logits is not None and len(out.router_logits) == 2
    expected = _v2_reference_loss(out.router_logits, model.config, mask) * model.config.aux_loss_alpha
    try:
        np.testing.assert_allclose(out.aux_loss, expected, rtol=1e-5, atol=1e-7)
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\nmask={mask.tolist()} seq_aux={seq_aux} "
            f"selector_observations={_v2_routing_diagnostics(records, model.config)}"
        ) from exc


@pytest.mark.parametrize("seq_aux", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
def test_v2_grouped_auxiliary_with_controlled_logits(seq_aux, masked, compiled, sharded):
    """Isolate group selection/reduction from transformer contractions and dispatch."""
    config = DeepseekV2Config(
        hidden_size=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        topk_method="group_limited_greedy",
        n_group=2,
        topk_group=1,
        seq_aux=seq_aux,
        aux_loss_alpha=1.0,
    )
    model = object.__new__(DeepseekV2ForCausalLM)
    object.__setattr__(model, "config", config)
    # Asymmetric assignment frequencies distinguish incorrect group choices
    # and per-sequence/global reduction. No winning-group ties occur.
    logits = np.asarray(
        [
            [[7, -4, 3, 2], [5, -2, 4, 3], [-1, 2, 3, 8], [3, 4, 5, 6]],
            [[2, 1, -3, 7], [5, 8, 3, 4], [1, 2, 8, -5], [0, 1, 6, 5]],
        ]
        * 2,
        dtype=np.float32,
    )
    mask = np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]] * 2, dtype=np.float32) if masked else np.ones((4, 4), np.float32)
    device_logits = jnp.asarray(logits)
    if sharded:
        device_logits = jax.device_put(
            device_logits,
            jax.sharding.NamedSharding(config.mesh.jax_mesh, jax.sharding.PartitionSpec("fsdp", None, None)),
        )
    outputs = MoeModelOutput(last_hidden_state=jnp.zeros((4, 4, 4)), router_logits=(device_logits,))
    compute = jax.jit(model._compute_aux_loss) if compiled else model._compute_aux_loss
    actual = compute(outputs, jnp.asarray(mask) if masked else None)
    expected = _v2_reference_loss((logits,), config, mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("experts_per_token", [2, 3])
@pytest.mark.parametrize("all_second_group", [False, True])
def test_v2_grouped_dispatch_preserves_per_token_group_eligibility(all_second_group, experts_per_token):
    """Both group membership and probability weights affect public MoE outputs."""
    config = DeepseekV2Config(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=4,
        n_shared_experts=None,
        n_routed_experts=4,
        num_experts_per_tok=experts_per_token,
        topk_method="group_limited_greedy",
        n_group=2,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        use_expert_tensor_mode=False,
    )
    config.moe_force_xla_gmm = True
    moe = DeepseekV2MoE(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(1),
    )
    logits = np.asarray([[7, -4, 3, 2], [5, -2, 4, 3], [-1, 2, 3, 8], [3, 4, 5, 6]], dtype=np.float32)
    if all_second_group:
        logits[:, :2] -= 20
    moe.gate.weight.value = jnp.asarray(logits.T)
    moe.experts.gate_up_proj.weight.value = jnp.ones_like(moe.experts.gate_up_proj.weight.value)
    coefficients = np.arange(1, 5, dtype=np.float32)
    moe.experts.down_proj.weight.value = jnp.asarray(
        np.broadcast_to(coefficients[:, None, None], moe.experts.down_proj.weight.value.shape).copy()
    )
    # Every token's hidden sum is one, making all expert gate/up projections
    # equal to one. Distinct down projections expose an incorrect expert group.
    hidden = np.broadcast_to(np.eye(4, dtype=np.float32)[None, :, :], (4, 4, 4)).copy()
    actual, actual_logits = moe(jnp.asarray(hidden))
    np.testing.assert_allclose(np.asarray(actual_logits).reshape(4, 4, 4), np.broadcast_to(logits, (4, 4, 4)))
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    groups = probabilities.reshape(4, 2, 2).max(axis=-1).argmax(axis=-1)
    selected = groups[:, None] * 2 + np.arange(2)[None, :]
    weights = np.take_along_axis(probabilities, selected, axis=-1)
    weights /= weights.sum(axis=-1, keepdims=True)
    expected = 4 / (1 + np.exp(-1.0)) * (weights * coefficients[selected]).sum(axis=-1)
    np.testing.assert_allclose(actual, np.broadcast_to(expected[None, :, None], (4, 4, 4)), rtol=2e-5, atol=2e-5)


def test_v2_zero_aux_loss_alpha_disables_aux_loss():
    out = _v2_model(aux_loss_alpha=0.0)(input_ids=_ids(), output_router_logits=True)
    assert out.aux_loss is None


def test_v2_explicit_zero_router_coefficient_takes_precedence():
    out = _v2_model(router_aux_loss_coef=0.0)(input_ids=_ids(), output_router_logits=True)
    assert out.aux_loss is None


@pytest.mark.parametrize("seq_aux", [True, False])
@pytest.mark.parametrize("coefficient", [None, 0.01])
def test_v3_aux_loss_uses_unbiased_sigmoid_scores_and_grouped_selections(seq_aux, coefficient):
    model = _v3_model(router_aux_loss_coef=coefficient, seq_aux=seq_aux)
    bias = np.asarray([0.9, -0.2, 0.4, 0.1, -0.3, 0.5, 0.2, -0.1], dtype=np.float32)
    for layer in model.model.layers:
        layer.mlp.gate.e_score_correction_bias.value = jnp.asarray(bias)
    ids = jnp.concatenate([_ids(), _ids()[:, ::-1]], axis=0)
    mask = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.float32)
    out = model(input_ids=ids, attention_mask=jnp.asarray(mask), output_router_logits=True)
    assert len(out.router_logits) == 2
    reference_losses = []
    for router_scores in out.router_logits:
        choices = np.asarray(router_scores).reshape(-1, 8)
        scores = choices - bias
        probabilities = scores / scores.sum(axis=-1, keepdims=True)
        groups = choices.reshape(-1, 2, 4)
        group_mass = np.sort(groups, axis=-1)[..., -2:].sum(axis=-1)
        best_group = group_mass.argmax(axis=-1)
        selected = np.stack([np.argsort(groups[t, g])[-2:] + 4 * g for t, g in enumerate(best_group)])
        counts = np.eye(8)[selected].sum(axis=-2)
        loss_mask = mask if seq_aux else mask.reshape(1, -1)
        counts = counts.reshape(*loss_mask.shape, 8)
        probabilities = probabilities.reshape(*loss_mask.shape, 8)
        token_count = loss_mask.sum(axis=-1, keepdims=True)
        frequency = (counts * loss_mask[..., None]).sum(axis=1) / (2 * token_count)
        mean_probability = (probabilities * loss_mask[..., None]).sum(axis=1) / token_count
        reference_losses.append(8 * (frequency * mean_probability).sum(axis=-1).mean())
    coef = model.config.aux_loss_alpha if coefficient is None else coefficient
    np.testing.assert_allclose(out.aux_loss, coef * np.mean(reference_losses), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("coefficient", [None, 0.0])
def test_v3_zero_coefficient_disables_aux_loss(coefficient):
    model = _v3_model(router_aux_loss_coef=coefficient, aux_loss_alpha=0.0 if coefficient is None else 0.001)
    out = model(input_ids=_ids(), output_router_logits=True)
    assert out.router_logits is not None
    assert out.aux_loss is None


@pytest.mark.parametrize("factory", [_v2_model, _v3_model])
def test_mla_cache_config_preserves_latent_rope_and_indexer_geometry(factory):
    model = factory()
    transformer = model.create_transformer_cache_config(batch_size=2, max_length=32)
    assert transformer.key_heads == 4
    assert transformer.value_heads == 4
    assert transformer.key_dim == 16
    assert transformer.value_dim == 16

    model.config.attn_mechanism = "multi_latent_ragged_page_attention_v1"
    model.config.mla_cache_rope_width = 128
    model.config.index_kpool = 2
    model.config.index_head_dim = 16
    paged = model.create_ragged_page_cache_config(max_length=32, page_size=128, max_cache_tokens=128)
    assert paged.k_headdim == 16
    assert paged.v_headdim == 128
    assert paged.num_kv_heads == 4
    assert paged.indexer_key_dim == 16
    assert paged.indexer_gate_dim == 16
    assert paged.indexer_kpool == 2
    assert paged.num_pages * paged.page_size <= 128
