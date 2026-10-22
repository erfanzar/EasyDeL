# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""GLM-MoE-DSA warm-up: isolated indexer gradients, optimization and masking.

These computation tests are intended for accelerators. Direct CPU invocation is
only a logic check, not hardware validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.modules.glm_moe_dsa import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from ejkernel.types import MaskInfo

COEF = 0.5
IDS = (2, 7, 3, 11, 5, 13)


def _config(coef=COEF, **kwargs):
    config = GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=8,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        max_position_embeddings=32,
        initializer_range=0.15,
        mlp_layer_types=["dense", "dense"],
        indexer_aux_loss_coef=coef,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        **kwargs,
    )
    config.attn_dtype = jnp.float32
    config.dsa_sparse_training = False
    return config


def _model(coef=COEF, **kwargs):
    config = _config(coef, **kwargs)
    with config.mesh:
        model = GlmMoeDsaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(17),
        )
        model.eval()
    return model


def _arrays(state):
    return {path: np.asarray(getattr(leaf, "value", leaf)) for _, path, leaf in state.items()}


def _objective(state, ids):
    return lambda params: state.merge(params)(input_ids=ids).indexer_aux_loss


def test_indexer_params_get_finite_nonzero_gradients_and_nothing_else_does():
    model = _model()
    ids = jnp.asarray([IDS], jnp.int32)
    with model.config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        loss, grads = jax.value_and_grad(_objective(state, ids))(state.graphstate)
    assert np.isfinite(float(loss)) and float(loss) > 0
    named = _arrays(grads)
    assert set(named) == set(_arrays(state.graphstate))
    indexer = {name for name in named if ".indexer." in name}
    assert len(indexer) == 10  # wq_b, wk, norm weight/bias, head weights in each layer.
    for name, grad in named.items():
        assert np.isfinite(grad).all(), name
        if name in indexer:
            assert np.any(grad != 0), f"all-zero indexer gradient: {name}"
        else:
            assert np.all(grad == 0), f"aux gradient leaked into {name}"


def test_aux_loss_decreases_updating_only_indexer_params():
    model = _model()
    ids = jnp.asarray([IDS], jnp.int32)
    with model.config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        before = _arrays(state.graphstate)

        @jax.jit
        def step(state):
            loss, grads = jax.value_and_grad(_objective(state, ids))(state.graphstate)
            return loss, state.apply_gradients(grads=grads)

        losses = []
        for _ in range(12):
            loss, state = step(state)
            losses.append(float(loss))
        final_loss = float(_objective(state, ids)(state.graphstate))
    assert np.isfinite(losses).all() and np.isfinite(final_loss)
    assert final_loss < 0.9 * losses[0], (losses, final_loss)
    after = _arrays(state.graphstate)
    changed = {name for name in before if not np.array_equal(before[name], after[name])}
    assert changed and all(".indexer." in name for name in changed), changed


def test_coef_zero_logits_identical_and_aux_loss_scaled_once():
    off, on = _model(0.0), _model(COEF)
    ids = jnp.asarray([IDS], jnp.int32)
    with off.config.mesh:
        out_off = off(input_ids=ids)
    with on.config.mesh:
        out_on = on(input_ids=ids)
    assert out_off.indexer_aux_loss is None and out_off.aux_loss is None
    assert np.isfinite(float(out_on.indexer_aux_loss)) and float(out_on.indexer_aux_loss) > 0
    np.testing.assert_array_equal(out_off.logits, out_on.logits)
    np.testing.assert_allclose(out_on.aux_loss, COEF * out_on.indexer_aux_loss, rtol=1e-6)
    doubled = _model(2 * COEF)
    with doubled.config.mesh:
        out_double = doubled(input_ids=ids)
    np.testing.assert_array_equal(out_on.indexer_aux_loss, out_double.indexer_aux_loss)
    np.testing.assert_allclose(out_double.aux_loss, 2 * out_on.aux_loss, rtol=1e-6)


def test_padding_and_packed_segments_match_independent_documents():
    model = _model()
    ids = jnp.asarray([IDS], jnp.int32)
    with model.config.mesh:
        first = model(input_ids=ids[:, :3]).indexer_aux_loss
        second = model(input_ids=ids[:, 3:]).indexer_aux_loss
        segments = jnp.asarray([[0, 0, 0, 1, 1, 1, -1, -1]], jnp.int32)
        positions = jnp.asarray([[0, 1, 2, 0, 1, 2, 0, 0]], jnp.int32)
        padded = jnp.asarray([[*IDS, 19, 23]], jnp.int32)
        packed = model(
            input_ids=padded,
            position_ids=positions,
            mask_info=MaskInfo.from_segments(segments),
        )
        plain = model(input_ids=ids)
        pad = model(input_ids=padded, attention_mask=jnp.asarray([[1] * 6 + [0, 0]], jnp.bool_))
        changed_pad = model(
            input_ids=padded.at[:, 6:].set(29), attention_mask=jnp.asarray([[1] * 6 + [0, 0]], jnp.bool_)
        )
        all_pad = model(input_ids=padded, attention_mask=jnp.zeros_like(padded, dtype=jnp.bool_))
    np.testing.assert_allclose(packed.indexer_aux_loss, (first + second) / 2, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(plain.indexer_aux_loss, pad.indexer_aux_loss, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(pad.indexer_aux_loss, changed_pad.indexer_aux_loss)
    assert float(all_pad.indexer_aux_loss) == 0.0


def test_shared_indexer_layers_do_not_add_an_auxiliary_objective():
    model = _model(indexer_types=["full", "shared"])
    ids = jnp.asarray([IDS], jnp.int32)
    with model.config.mesh:
        out = model(input_ids=ids)
        layer = model.model.layers[0]
        hidden = model.model.embed_tokens(ids)
        first = layer(
            hidden_states=hidden,
            mask_info=MaskInfo.from_segments(jnp.zeros_like(ids)),
            position_ids=jnp.arange(ids.shape[1], dtype=jnp.int32)[None],
            mode="train",
            frequencies=model.model.frequencies,
        )
    assert float(out.indexer_aux_loss) > 0
    np.testing.assert_allclose(out.indexer_aux_loss, first.indexer_aux_loss, rtol=1e-6)


def test_config_default_and_negative_coefficient():
    assert GlmMoeDsaConfig().indexer_aux_loss_coef == 0.0
    with pytest.raises(ValueError, match="indexer_aux_loss_coef"):
        _config(-0.1)


def test_cached_forward_never_computes_aux_loss():
    model = _model()
    ids = jnp.asarray([IDS], jnp.int32)
    with model.config.mesh:
        cache = model.init_cache(batch_size=1, max_length=16)
        prefill = model(input_ids=ids, past_key_values=cache)
        decode = model(input_ids=ids[:, :1], past_key_values=prefill.past_key_values)
    assert prefill.indexer_aux_loss is None and prefill.aux_loss is None
    assert decode.indexer_aux_loss is None and decode.aux_loss is None


def _weight(module):
    return np.asarray(module.weight.value, dtype=np.float64)


def _rms(x, norm):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + norm.eps) * _weight(norm)


def _rotate(x, cos, sin, interleave=True):
    if interleave:
        a, b = x[..., ::2], x[..., 1::2]
        return np.stack((a * cos - b * sin, b * cos + a * sin), -1).reshape(x.shape)
    a, b = np.split(x, 2, axis=-1)
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), -1)


def _softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def test_aux_matches_numpy_dense_mla_target_and_weighted_relu_scores():
    """Independent full MLA/DSA projections, RoPE and KL catch scale/layout errors."""
    for interleave in (True, False):
        model = _model(indexer_rope_interleave=interleave)
        layer = model.model.layers[0].self_attn
        config = model.config
        hidden = np.random.default_rng(3).normal(size=(1, 6, config.hidden_size)).astype(np.float32)
        positions = jnp.arange(6, dtype=jnp.int32)[None]
        with config.mesh:
            frequencies = model.model.frequencies
            out = layer.forward_mla(
                jnp.asarray(hidden),
                MaskInfo.from_segments(jnp.zeros((1, 6), jnp.int32)),
                positions,
                mode="train",
                frequencies=frequencies,
            )
        x = hidden.astype(np.float64)
        q_resid = _rms(x @ _weight(layer.mla_q_a_proj), layer.mla_q_a_layernorm)
        q = (q_resid @ _weight(layer.mla_q_b_proj)).reshape(1, 6, 2, 8)
        latent = x @ _weight(layer.mla_kv_a_proj_with_mqa)
        kv = (_rms(latent[..., :8], layer.mla_kv_a_layernorm) @ _weight(layer.mla_kv_b_proj)).reshape(1, 6, 2, 12)
        freq = np.asarray(getattr(frequencies, "value", frequencies))[:6].astype(np.float64)
        cos, sin = np.split(freq[None, :, None], 2, axis=-1)
        q = np.concatenate((q[..., :4], _rotate(q[..., 4:], cos, sin)), -1)
        k_pe = _rotate(latent[..., 8:][:, :, None], cos, sin)
        k = np.concatenate((kv[..., :4], np.broadcast_to(k_pe, (1, 6, 2, 4))), -1)
        visible = np.tri(6, dtype=bool)[None]
        target_logits = np.einsum("bshd,bthd->bhst", q, k) * layer.attention_performer.softmax_scale
        p = _softmax(np.where(visible[:, None], target_logits, -1e30)).mean(axis=1)

        indexer = layer.indexer
        iq = (q_resid @ _weight(indexer.wq_b)).reshape(1, 6, 2, 8)
        ik = x @ _weight(indexer.wk)
        ik = (ik - ik.mean(-1, keepdims=True)) / np.sqrt(ik.var(-1, keepdims=True) + indexer.k_norm.eps)
        ik = ik * _weight(indexer.k_norm) + np.asarray(indexer.k_norm.bias.value)
        iq = np.concatenate((_rotate(iq[..., :4], cos, sin, interleave), iq[..., 4:]), -1)
        ik = np.concatenate((_rotate(ik[..., :4], cos[:, :, 0], sin[:, :, 0], interleave), ik[..., 4:]), -1)
        weights = (x @ _weight(indexer.kernels_proj)) / np.sqrt(2)
        per_head = np.maximum(np.einsum("bshd,btd->bsht", iq, ik) / np.sqrt(8), 0)
        scores = np.einsum("bsht,bsh->bst", per_head, weights)
        predicted = _softmax(np.where(visible, scores, -1e30))
        expected = np.sum(p * (np.log(np.maximum(p, 1e-30)) - np.log(np.maximum(predicted, 1e-30))), -1).mean()
        assert expected > 0
        np.testing.assert_allclose(out.indexer_aux_loss, expected, rtol=1e-5, atol=2e-7)


def test_scanned_layers_preserve_aux_loss_and_gradients():
    traced, scanned = _model(), _model(scan_layers=True)
    ids = jnp.asarray([IDS], jnp.int32)
    outputs, gradients = [], []
    for model in (traced, scanned):
        with model.config.mesh:
            state = EasyDeLState.create(model=model, tx=optax.sgd(1e-2), init_opt_state=True)
            loss, grads = jax.value_and_grad(_objective(state, ids))(state.graphstate)
            outputs.append(float(loss))
            gradients.append(_arrays(grads))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        np.testing.assert_allclose(gradients[0][name], gradients[1][name], rtol=2e-5, atol=1e-7, err_msg=name)
