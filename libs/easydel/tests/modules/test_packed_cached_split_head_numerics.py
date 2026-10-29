"""Accelerator regressions for public packed, cached and split-head execution.

Accelerator only (not cpu_ok).
"""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.caching import RecurrentCache, RecurrentCacheConfig
from easydel.infra.loss_utils import CausalLMLossStrategy, LossConfig, auxiliary_load_balancing_loss_func
from easydel.modules.deepseek_v3 import DeepseekV3Config
from easydel.modules.kimi_vl import KimiVLConfig, KimiVLForConditionalGeneration, MoonViTConfig
from easydel.modules.llama4 import Llama4ForCausalLM, Llama4TextConfig
from easydel.modules.mamba import MambaConfig, MambaForCausalLM
from easydel.modules.mamba2 import Mamba2Config, Mamba2ForCausalLM
from jax import numpy as jnp

from .test_llama4_router_loss import _capture_router_loss, _router_loss_diagnostics
from .test_vision_llama_forward import _model as _vision_model


def _mamba_model(family, dtype=jnp.float32):
    options = dict(
        vocab_size=16,
        hidden_size=8,
        state_size=2,
        expand=2,
        num_hidden_layers=1,
        conv_kernel=4,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    if family == "mamba":
        config = MambaConfig(time_step_rank=2, **options)
        model_class = MambaForCausalLM
    else:
        config = Mamba2Config(num_heads=8, head_dim=2, n_groups=1, chunk_size=4, **options)
        model_class = Mamba2ForCausalLM
    with config.mesh:
        return model_class(
            config, dtype=dtype, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(9)
        )


def _cache(model, family):
    config = model.config
    options = dict(
        num_hidden_layers=1,
        partition_axis=config.partition_axis,
        batch_size=4,
        intermediate_size=config.intermediate_size,
        conv_kernel_size=config.conv_kernel,
    )
    if family == "mamba":
        cache_config = RecurrentCacheConfig.create_for_mamba(ssm_state_size=config.state_size, **options)
    else:
        cache_config = RecurrentCacheConfig.create_for_mamba2(
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            state_size=config.state_size,
            n_groups=config.n_groups,
            **options,
        )
    return RecurrentCache.init_cache(cache_config, dtype=model.dtype)


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
def test_public_packed_sequence_matches_independent_documents(family):
    model = _mamba_model(family)
    ids = jnp.tile(jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32), (4, 1))
    segments = jnp.tile(jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32), (4, 1))
    packed = model(input_ids=ids, segment_ids=segments, apply_lm_head=False).last_hidden_state
    first = model(input_ids=ids[:, :4], apply_lm_head=False).last_hidden_state
    second = model(input_ids=ids[:, 4:], apply_lm_head=False).last_hidden_state
    expected = jnp.concatenate((first, second), axis=1)
    np.testing.assert_allclose(np.asarray(packed), np.asarray(expected), rtol=3e-4, atol=2e-5)


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("segmented", [False, True])
def test_public_multitoken_cache_continuation_matches_full_sequence(family, segmented):
    model = _mamba_model(family)
    ids = jnp.tile(jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=jnp.int32), (4, 1))
    segment_kwargs = dict(segment_ids=jnp.zeros_like(ids)) if segmented else {}
    full = model(input_ids=ids, **segment_kwargs).logits
    cache_name = "cache" if family == "mamba" else "cache_params"
    cache = _cache(model, family)
    outputs = []
    for start, end in ((0, 4), (4, 6), (6, 7), (7, 8)):
        kwargs = dict(segment_ids=jnp.zeros_like(ids[:, start:end])) if segmented else {}
        output = model(input_ids=ids[:, start:end], **{cache_name: cache}, **kwargs)
        outputs.append(output.logits)
        cache = getattr(output, cache_name)
    np.testing.assert_allclose(
        np.asarray(jnp.concatenate(outputs, axis=1)),
        np.asarray(full),
        rtol=5e-4,
        atol=2e-5,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_mamba_convolution_honors_storage_compute_and_precision(dtype):
    model = _mamba_model("mamba", dtype=dtype)
    conv = model.backbone.layers[0].mixer.conv1d
    assert conv.weight.value.dtype == jnp.float32
    assert conv.dtype == dtype
    assert conv.precision == jax.lax.Precision.HIGHEST


@pytest.mark.parametrize("family", ["mamba", "mamba2"])
def test_public_headless_chunked_training_matches_independent_cross_entropy(family):
    model = _mamba_model(family)
    ids = jnp.tile(jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32), (4, 1))
    dense = model(input_ids=ids)
    logits = np.asarray(dense.logits[:, :-1]).astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    expected = -np.take_along_axis(log_probs, np.asarray(ids[:, 1:, None]), axis=-1).mean()
    strategy, loss_config = CausalLMLossStrategy(), LossConfig(chunk_token_size=2)
    plan = strategy.plan_forward(module=model, labels=ids, loss_config=loss_config, batch={}, loss_kwargs={})
    output = model(input_ids=ids, **plan.forward_kwargs)
    assert output.logits is None
    assert output.last_hidden_state.shape == (4, 4, 8)
    metrics = strategy.compute(
        module=model,
        outputs=output,
        labels=ids,
        loss_config=loss_config,
        batch={},
        loss_kwargs={},
        paxis=None,
        forward_plan=plan,
    )
    np.testing.assert_allclose(np.asarray(metrics.loss), expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("text_tied", [False, True])
@pytest.mark.parametrize("vision_tied", [False, True])
def test_exported_joint_projection_has_no_backbone_and_matches_full_logits(text_tied, vision_tied):
    model = _vision_model(tie_vision_embeddings=vision_tied)
    model.config.tie_word_embeddings = text_tied
    model._tie_embeddings_feature.tie = text_tied
    hidden = jnp.arange(32, dtype=jnp.float32).reshape(2, 16) / 32
    projection = model.get_lm_head_projection()
    graphdef, state = spx.export(projection)
    restored = spx.bind(graphdef, state)
    expected = model.compute_lm_logits(hidden)
    np.testing.assert_allclose(np.asarray(restored(hidden)), np.asarray(expected), rtol=1e-5, atol=1e-6)
    assert not hasattr(restored, "model")
    assert not hasattr(restored, "layers")


def test_kimi_public_headless_output_supports_chunked_training():
    text = DeepseekV3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        moe_intermediate_size=16,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        attn_mechanism="vanilla",
    )
    config = KimiVLConfig(
        text_config=text,
        vision_config=MoonViTConfig(
            patch_size=2,
            init_pos_emb_height=4,
            init_pos_emb_width=4,
            num_attention_heads=2,
            num_hidden_layers=1,
            hidden_size=8,
            intermediate_size=16,
            merge_kernel_size=(2, 2),
        ),
        media_placeholder_token_id=15,
        pad_token_id=0,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    with config.mesh:
        model = KimiVLForConditionalGeneration(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    ids = jnp.tile(jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32), (4, 1))
    dense = model(input_ids=ids)
    logits = np.asarray(dense.logits[:, :-1]).astype(np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    expected = -np.take_along_axis(log_probs, np.asarray(ids[:, 1:, None]), axis=-1).mean()
    strategy, loss_config = CausalLMLossStrategy(), LossConfig(chunk_token_size=2)
    plan = strategy.plan_forward(module=model, labels=ids, loss_config=loss_config, batch={}, loss_kwargs={})
    output = model(input_ids=ids, **plan.forward_kwargs)
    assert output.logits is None
    assert output.last_hidden_state.shape == (4, 4, 16)
    metrics = strategy.compute(
        module=model,
        outputs=output,
        labels=ids,
        loss_config=loss_config,
        batch={},
        loss_kwargs={},
        paxis=None,
        forward_plan=plan,
    )
    np.testing.assert_allclose(np.asarray(metrics.loss), expected, rtol=1e-4, atol=1e-5)


def test_llama4_public_router_outputs_and_weighted_aux_loss():
    config = Llama4TextConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        intermediate_size_mlp=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=1,
        moe_layers=[1],
        no_rope_layers=[0, 1],
        output_router_logits=True,
        router_aux_loss_coef=0.025,
        attn_mechanism="vanilla",
        attention_chunk_size=4,
        max_position_embeddings=32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    with config.mesh:
        model = Llama4ForCausalLM(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    ids = jnp.tile(jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32), (4, 1))
    mask = jnp.tile(jnp.asarray([[1, 1, 1, 0]], dtype=jnp.int32), (4, 1))
    output = model(input_ids=ids, attention_mask=mask)
    assert len(output.router_logits) == 1
    raw = np.asarray(output.router_logits[0]).reshape(-1, 4).astype(np.float64)
    raw -= raw.max(axis=-1, keepdims=True)
    probabilities = np.exp(raw) / np.exp(raw).sum(axis=-1, keepdims=True)
    valid = np.asarray(mask).reshape(-1).astype(bool)
    selected = raw.argmax(axis=-1)
    routing = np.eye(4)[selected]
    expected = 4 * np.sum(routing[valid].mean(axis=0) * probabilities[valid].mean(axis=0))
    # Check the scalar loss before the per-operation diagnostics so a top-k
    # mismatch cannot hide the shared loss result.
    with _capture_router_loss() as observations:
        standalone = auxiliary_load_balancing_loss_func(
            output.router_logits, num_experts=4, top_k=1, attention_mask=mask
        )
    try:
        np.testing.assert_allclose(
            np.asarray(standalone),
            expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg="standalone shared masked router loss disagrees with independent softmax reference",
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{exc}\nactual_shared_loss_operations={_router_loss_diagnostics(observations)}\n"
            f"independent_selected={selected.tolist()}\n"
            f"independent_mask={valid.tolist()}\nindependent_probabilities={probabilities.tolist()}"
        ) from exc
    np.testing.assert_allclose(
        np.asarray(output.aux_loss),
        np.asarray(standalone) * 0.025,
        rtol=1e-4,
        atol=1e-6,
        err_msg="Llama4 task auxiliary loss disagrees with its exposed router logits and attention mask",
    )
    np.testing.assert_allclose(np.asarray(output.aux_loss), expected * 0.025, rtol=1e-4, atol=1e-6)
    # Operation-level checks observe the actual shared-loss call, not an unrelated
    # native top-k that may take a different path.
    assert len(observations) == 1
    actual = observations[0]
    device_probabilities = actual["routing_weights"]
    np.testing.assert_allclose(
        np.asarray(device_probabilities), probabilities, rtol=1e-5, atol=1e-7, err_msg="router softmax mismatch"
    )
    device_selected = actual["selected_experts"]
    np.testing.assert_array_equal(
        np.asarray(device_selected).reshape(-1), selected, err_msg="router top-k indices mismatch"
    )
    device_expert_mask = actual["expert_mask"]
    np.testing.assert_array_equal(
        np.asarray(device_expert_mask).reshape(-1, 4), routing, err_msg="router expert one-hot encoding mismatch"
    )
    disabled = model(input_ids=ids, output_router_logits=False)
    assert disabled.router_logits is None
    assert disabled.aux_loss is None
