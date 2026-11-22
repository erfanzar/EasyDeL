# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""GLM-5-Next pooled DSA warm-up: isolation, optimization, masks and opt-out.

These numerical tests require an accelerator, like the other model tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.modules.glm5_next import Glm5NextForCausalLM, Glm5NextTextConfig
from easydel.modules.glm5_next.modeling_glm5_next import Glm5NextIndexer

COEF = 0.5


def _config(coef=COEF, **overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=32,
        initializer_range=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        index_head_dim=4,
        index_n_heads=2,
        index_topk=4,
        index_kpool=2,
        indexer_aux_loss_coef=coef,
        layer_types=["linear_attention", "deepseek_sparse_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "sparse", "dense"],
        indexer_types=["full", "full", "full"],
        linear_attn_config={"head_dim": 8, "num_heads": 2, "short_conv_kernel_size": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        output_router_logits=False,
        sharding_axis_dims=(1, -1, 1, 1, 1, 1),
        dsa_sparse_training=False,
        attn_mechanism="vanilla",
        gradient_checkpointing="none",
    )
    kwargs.update(overrides)
    config = Glm5NextTextConfig(**kwargs)
    config.moe_force_xla_gmm = True
    config.fsdp_is_ep_bound = False
    config.attach_custom_arguments()
    return config


def _model(config):
    with config.mesh:
        model = Glm5NextForCausalLM(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        model.eval()
    return model


def _ids():
    batch = max(2, jax.device_count())
    return jnp.asarray(np.random.default_rng(17).integers(1, 64, size=(batch, 12)), dtype=jnp.int32)


def _arrays(state):
    return {
        path: np.asarray(leaf.value if hasattr(leaf, "value") else leaf)
        for collection, path, leaf in state.items()
        if collection == "parameters"
    }


def _loss(state, ids):
    return lambda graphstate: state.merge(graphstate)(input_ids=ids).indexer_aux_loss


def test_indexer_aux_coefficient_validation_and_roundtrip():
    """The opt-in coefficient is non-negative and survives config serialization."""
    assert Glm5NextTextConfig().indexer_aux_loss_coef == 0.0
    for invalid in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="indexer_aux_loss_coef"):
            Glm5NextTextConfig(indexer_aux_loss_coef=invalid)
    config = _config()
    assert Glm5NextTextConfig.from_dict(config.to_dict()).indexer_aux_loss_coef == COEF
    with pytest.raises(ValueError, match="indexer_aux_loss_coef"):
        Glm5NextTextConfig(text_config={"indexer_aux_loss_coef": -1.0})


def test_indexer_params_get_finite_nonzero_gradients_and_nothing_else_does():
    config = _config()
    model, ids = _model(config), _ids()
    with config.mesh:
        state = EasyDeLState.create(model=model)
        loss, grads = jax.value_and_grad(_loss(state, ids))(state.graphstate)
    assert np.isfinite(float(loss)) and float(loss) > 0
    params, grads = _arrays(state.graphstate), _arrays(grads)
    indexers = {name for name in params if ".indexer." in name}
    assert indexers and set(grads) == set(params)
    for name, grad in grads.items():
        assert np.isfinite(grad).all(), name
        if name in indexers:
            assert np.any(grad != 0), f"{name}: all-zero indexer gradient"
        else:
            np.testing.assert_array_equal(grad, np.zeros_like(grad), err_msg=name)


def test_aux_loss_decreases_updating_only_indexer_params():
    config = _config()
    model, ids = _model(config), _ids()
    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(2e-2), init_opt_state=True)
        before = _arrays(state.graphstate)

        @jax.jit
        def step(state):
            loss, grads = jax.value_and_grad(_loss(state, ids))(state.graphstate)
            return loss, state.apply_gradients(grads=grads)

        losses = []
        for _ in range(12):
            loss, state = step(state)
            losses.append(float(loss))
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.99 * losses[0], losses
    after = _arrays(state.graphstate)
    changed = {name for name in before if not np.array_equal(before[name], after[name])}
    assert changed
    assert all(".indexer." in name for name in changed), sorted(changed)


def test_coef_zero_logits_are_identical_and_aux_loss_is_unscaled():
    off_config, on_config = _config(0.0), _config(COEF)
    off_model, on_model, ids = _model(off_config), _model(on_config), _ids()
    with off_config.mesh:
        off = off_model(input_ids=ids)
    with on_config.mesh:
        on = on_model(input_ids=ids)
        backbone = on_model.model(input_ids=ids)
    assert off.indexer_aux_loss is None and off.aux_loss is None
    assert np.isfinite(float(on.indexer_aux_loss)) and float(on.indexer_aux_loss) > 0
    np.testing.assert_array_equal(off.logits, on.logits)
    np.testing.assert_allclose(on.indexer_aux_loss, backbone.indexer_aux_loss, rtol=1e-6)
    np.testing.assert_allclose(on.aux_loss, COEF * on.indexer_aux_loss, rtol=1e-6)


def test_cached_forward_does_not_compute_indexer_aux_loss():
    """The opt-in training objective is absent on cached prefill as well as decode."""
    config = _config()
    model, ids = _model(config), _ids()
    with config.mesh:
        cache = model.init_cache(batch_size=ids.shape[0], max_length=16)
        prefill = model(input_ids=ids, past_key_values=cache)
        decode = model(input_ids=ids[:, :1], past_key_values=prefill.past_key_values)
    for output in (prefill, decode):
        assert output.indexer_aux_loss is None and output.aux_loss is None
        assert np.isfinite(np.asarray(output.logits)).all()
    assert prefill.logits.shape == (*ids.shape, config.vocab_size)
    assert decode.logits.shape == (ids.shape[0], 1, config.vocab_size)


def test_shared_indexer_has_no_auxiliary_gradient():
    """Reused selections must not train the unused shared layer's indexer."""
    config = _config(indexer_types=["full", "full", "shared"])
    model, ids = _model(config), _ids()
    with config.mesh:
        state = EasyDeLState.create(model=model)
        loss, grads = jax.value_and_grad(_loss(state, ids))(state.graphstate)
    assert np.isfinite(float(loss)) and float(loss) > 0
    grads = _arrays(grads)
    shared = {name for name in grads if "layers.2.self_attn.indexer." in name}
    source = {name for name in grads if "layers.1.self_attn.indexer." in name}
    assert shared and source
    for name, grad in grads.items():
        assert np.isfinite(grad).all(), name
        if name in source:
            assert np.any(grad != 0), name
        else:
            np.testing.assert_array_equal(grad, np.zeros_like(grad), err_msg=name)


def _softmax(x, axis=-1):
    exp = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


def _reference_loss(indexer, hidden, residual, target_q, target_k, visible, kpool, scaling):
    """Independent NumPy projections, learned pools, head-summed targets and KL."""

    def array(param):
        return np.asarray(param.value, dtype=np.float64)

    queries = (residual @ array(indexer.wq_b.weight)).reshape(*hidden.shape[:2], indexer.n_heads, indexer.head_dim)
    keys = hidden @ array(indexer.wk.weight)
    keys = (keys - keys.mean(axis=-1, keepdims=True)) / np.sqrt(keys.var(axis=-1, keepdims=True) + 1e-6)
    keys = keys * array(indexer.k_norm.weight) + array(indexer.k_norm.bias)
    gates = hidden @ array(indexer.index_kpool_compress_gate).T
    weights = (hidden @ array(indexer.weights_proj.weight)) * indexer.n_heads**-0.5
    ape = array(indexer.index_kpool_compress_ape)
    losses = []
    for batch in range(hidden.shape[0]):
        for row in range(hidden.shape[1]):
            members = np.flatnonzero(visible[batch, row])
            members = members[: len(members) // kpool * kpool]
            if not len(members):
                continue
            groups = members.reshape(-1, kpool)
            pools = (_softmax(gates[batch, groups] + ape, axis=1) * keys[batch, groups]).sum(axis=1)
            dots = np.maximum(queries[batch, row] @ pools.T * indexer.head_dim**-0.5, 0)
            scores = weights[batch, row] @ dots
            dense = np.einsum("hd,thd->ht", target_q[batch, row], target_k[batch, members]) * scaling
            target = _softmax(dense).sum(axis=0).reshape(-1, kpool).sum(axis=-1)
            target /= target.sum()
            pred = _softmax(scores)
            losses.append(np.sum(target * (np.log(target) - np.log(pred))))
    return np.mean(losses)


def test_pooled_target_matches_numpy_with_padding_packed_segments_and_visibility_holes():
    """Physical padding/holes must not shift group mass or leak other documents."""
    config = _config()
    rng = np.random.default_rng(42)
    batch = max(2, jax.device_count())
    hidden = rng.normal(size=(batch, 14, config.hidden_size)).astype(np.float32)
    residual = rng.normal(size=(batch, 14, config.q_lora_rank)).astype(np.float32)
    target_q = rng.normal(size=(batch, 14, 2, 8)).astype(np.float32)
    target_k = rng.normal(size=(batch, 14, 2, 8)).astype(np.float32)
    segments = np.array([[-1, 0, 0, 0, -1, 0, 0, 0, 1, 1, 1, 1, 1, 1]] * batch, dtype=np.int32)
    causal = np.arange(14)[None, :] <= np.arange(14)[:, None]
    visible = causal[None] & (segments[:, :, None] == segments[:, None, :]) & (segments[:, :, None] >= 0)
    visible[:, 6:8, 2] = False  # query-local compaction rather than storage-wide pools
    scaling = 8**-0.5
    with config.mesh:
        indexer = Glm5NextIndexer(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(9),
        )
        actual = indexer.distillation_loss(
            jnp.asarray(hidden),
            jnp.asarray(residual),
            jnp.asarray(target_q),
            jnp.asarray(target_k),
            jnp.asarray(visible),
            jnp.asarray(segments),
            jnp.asarray(segments),
            scaling=scaling,
        )
    expected = _reference_loss(indexer, hidden, residual, target_q, target_k, visible, config.index_kpool, scaling)
    assert expected > 0
    np.testing.assert_allclose(float(actual), expected, rtol=2e-5, atol=2e-6)
