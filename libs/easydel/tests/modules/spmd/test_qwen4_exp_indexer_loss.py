# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

"""QSA warm-up: isolated gradients, optimization, masks, and text/VLM aux plumbing.

Computation tests require an accelerator. The zero-coefficient path retains the original LM STE;
with warm-up enabled its input is detached but the STE still trains indexer parameters.
"""

import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.modules.auto import AutoEasyDeLModelForCausalLM, AutoEasyDeLModelForImageTextToText
from easydel.modules.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types

COEF = 0.5


def _config(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        layer_types=["qwen_sparse_attention", "qwen_sparse_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        full_attention_interval=1,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=2,
        indexer_compress_ratio=2,
        indexer_aux_loss_coef=COEF,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[],
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        eos_token_id=0,
        bos_token_id=1,
        pad_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        attn_dtype=jnp.float32,
        attn_mechanism="vanilla",
        mtp=None,
        mtp_num_hidden_layers=0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    kwargs.update(overrides)
    return Qwen4ExpTextConfig(**kwargs)


def _model(config):
    auto = AutoEasyDeLModelForImageTextToText if isinstance(config, Qwen4ExpConfig) else AutoEasyDeLModelForCausalLM
    with config.mesh:
        return auto.from_config(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


def _ids():
    return jnp.array([[1, 7, 13, 19, 23, 31, 37, 43, 11, 17, 29, 47]], jnp.int32)


def _named_arrays(state):
    return {path: np.asarray(leaf.value if hasattr(leaf, "value") else leaf) for _, path, leaf in state.items()}


def _aux_fn(state, ids):
    def loss_fn(graphstate):
        return state.merge(graphstate)(input_ids=ids, mode=common_types.MODE_TRAIN).indexer_aux_loss

    return loss_fn


def test_config_validation_and_nested_roundtrip():
    """The only non-computation test: default, validation, and nested serialization."""
    assert Qwen4ExpTextConfig().indexer_aux_loss_coef == 0.0
    with pytest.raises(ValueError, match="indexer_aux_loss_coef must be non-negative"):
        Qwen4ExpTextConfig(indexer_aux_loss_coef=-0.1)
    nested = Qwen4ExpConfig(text_config={"indexer_aux_loss_coef": 0.25})
    assert nested.text_config.indexer_aux_loss_coef == 0.25
    restored = Qwen4ExpConfig(**nested.to_dict())
    assert restored.text_config.indexer_aux_loss_coef == 0.25
    override = Qwen4ExpConfig(text_config={"indexer_aux_loss_coef": 0.25}, indexer_aux_loss_coef=0.75)
    assert override.text_config.indexer_aux_loss_coef == 0.75
    with pytest.raises(ValueError, match="indexer_aux_loss_coef must be non-negative"):
        Qwen4ExpConfig(indexer_aux_loss_coef=-0.1)


def test_indexer_params_get_finite_nonzero_gradients_and_nothing_else_does():
    config = _config()
    model = _model(config)
    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        loss, grads = jax.jit(jax.value_and_grad(_aux_fn(state, _ids())))(state.graphstate)
    assert np.isfinite(float(loss)) and float(loss) > 0
    named = _named_arrays(grads)
    indexer_names = {name for name in _named_arrays(state.graphstate) if ".indexer." in name}
    assert indexer_names
    assert indexer_names <= named.keys()
    for name, grad in named.items():
        assert np.isfinite(grad).all(), name
        if name in indexer_names:
            assert np.any(grad != 0), f"all-zero indexer gradient: {name}"
        else:
            assert not np.any(grad != 0), f"auxiliary gradient leaked into {name}"


def test_aux_loss_decreases_updating_only_indexer_params():
    config = _config()
    model = _model(config)
    with config.mesh:
        state = EasyDeLState.create(model=model, tx=optax.adam(1e-2), init_opt_state=True)
        before = _named_arrays(state.graphstate)
        ids = _ids()

        @jax.jit
        def step(state):
            loss, grads = jax.value_and_grad(_aux_fn(state, ids))(state.graphstate)
            return loss, state.apply_gradients(grads=grads)

        losses = []
        for _ in range(12):
            loss, state = step(state)
            losses.append(float(loss))
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.99 * losses[0], losses
    after = _named_arrays(state.graphstate)
    changed = {name for name in before if not np.array_equal(before[name], after[name])}
    assert changed
    assert all(".indexer." in name for name in changed), sorted(changed)


def test_coef_zero_logits_identical_aux_absent_and_ste_still_trains():
    off_config, on_config = _config(indexer_aux_loss_coef=0.0), _config()
    off_model, on_model = _model(off_config), _model(on_config)
    ids = _ids()
    with off_config.mesh:
        off = off_model(input_ids=ids)
        state = EasyDeLState.create(model=off_model)

        def lm_loss(params):
            out = state.merge(params)(input_ids=ids)
            return optax.softmax_cross_entropy_with_integer_labels(out.logits[:, :-1], ids[:, 1:]).mean()

        grads = _named_arrays(jax.jit(jax.grad(lm_loss))(state.graphstate))
    with on_config.mesh:
        on = on_model(input_ids=ids)
    assert off.indexer_aux_loss is None and off.aux_loss is None
    assert np.isfinite(float(on.indexer_aux_loss)) and float(on.indexer_aux_loss) > 0
    np.testing.assert_array_equal(np.asarray(off.logits), np.asarray(on.logits))
    np.testing.assert_allclose(float(on.aux_loss), COEF * float(on.indexer_aux_loss), rtol=1e-6)
    assert any(".indexer." in name and np.any(grad != 0) for name, grad in grads.items())


def _check_mtp_combination(vlm):
    text = _config(num_hidden_layers=1, layer_types=["qwen_sparse_attention"], mtp_num_hidden_layers=1)
    config = (
        Qwen4ExpConfig(
            text_config=text,
            language_model_only=True,
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        if vlm
        else text
    )
    model = _model(config)
    with config.mesh:
        out = model(input_ids=_ids(), mode=common_types.MODE_TRAIN)
    assert out.mtp_loss is not None and np.isfinite(float(out.mtp_loss))
    assert out.indexer_aux_loss is not None and float(out.indexer_aux_loss) > 0
    np.testing.assert_allclose(float(out.aux_loss), COEF * float(out.indexer_aux_loss) + float(out.mtp_loss), rtol=1e-6)


def test_text_aux_is_scaled_once_alongside_mtp():
    _check_mtp_combination(False)


def test_vlm_nested_aux_is_scaled_once_alongside_mtp():
    _check_mtp_combination(True)


def test_padding_and_multimodal_position_rows_preserve_aux():
    """Rebasing blocks must remove left/right padding, not rebase the mRoPE target."""
    config = _config()
    model = _model(config)
    ids = _ids()[:, :9]  # leave a partial tail block
    text_positions = jnp.arange(ids.shape[1], dtype=jnp.int32)[None, :]
    # Non-monotonic image-style rows exercise physical-index, rather than RoPE-row, causality.
    rows = jnp.stack([text_positions, text_positions % 3, text_positions // 3])
    with config.mesh:
        reference = model(input_ids=ids, position_ids=rows)
        padded_ids = jnp.pad(ids, ((0, 0), (3, 2)), constant_values=config.pad_token_id)
        mask = jnp.pad(jnp.ones_like(ids), ((0, 0), (3, 2)))
        padded = model(
            input_ids=padded_ids,
            attention_mask=mask,
            position_ids=jnp.pad(rows, ((0, 0), (0, 0), (3, 2))),
        )
        empty = model(input_ids=padded_ids, attention_mask=jnp.zeros_like(padded_ids))
    np.testing.assert_allclose(float(padded.indexer_aux_loss), float(reference.indexer_aux_loss), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(padded.logits[:, 3:-2]), np.asarray(reference.logits), rtol=2e-5, atol=2e-6)
    assert float(empty.indexer_aux_loss) == 0.0


def test_cached_forward_has_no_aux_and_packed_qsa_still_rejects():
    config = _config()
    model = _model(config)
    ids = _ids()[:, :6]
    with config.mesh:
        out = model(
            input_ids=ids,
            past_key_values=model.init_cache(batch_size=1, max_length=16),
            mode=common_types.MODE_PREFILL,
        )
        segments = jnp.array([[0, 0, 0, 1, 1, 1]], jnp.int32)
        with pytest.raises(NotImplementedError, match="packed-document QSA"):
            model(input_ids=ids, mask_info=MaskInfo.from_segments(segments))
    assert out.indexer_aux_loss is None and out.aux_loss is None


def test_block_target_matches_independent_numpy_attention_mass():
    """Numerical oracle for GQA, shifted block origins, score masks, and query padding."""
    config = _config(num_hidden_layers=1, layer_types=["qwen_sparse_attention"])
    model = _model(config)
    attention = model.model.layers[0].self_attn
    rng = np.random.default_rng(17)
    batch, seq, heads, dim = 2, 9, 2, 16
    queries = rng.normal(size=(batch, seq, heads, dim)).astype(np.float32)
    keys = rng.normal(size=(batch, seq, 1, dim)).astype(np.float32)
    visible = np.array([[0, 0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
    proxy = np.zeros((batch, 1, seq, seq), np.float32)
    losses = []
    for row in range(batch):
        first = np.flatnonzero(visible[row])[0]
        starts = [start for start in range(first, seq - 1, 2) if visible[row, start : start + 2].all()]
        scores = rng.normal(size=(seq, len(starts))).astype(np.float32)
        for query in range(seq):
            live = [i for i, start in enumerate(starts) if start + 1 <= query]
            for block in live:
                proxy[row, 0, query, starts[block] : starts[block] + 2] = scores[query, block]
            if not visible[row, query] or not live:
                continue
            tokens = [token for block in live for token in range(starts[block], starts[block] + 2)]
            logits = np.einsum("hd,td->ht", queries[row, query].astype(np.float64), keys[row, tokens, 0]) / np.sqrt(dim)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            target = probs.sum(axis=0).reshape(len(live), 2).sum(axis=-1) / heads
            student_logits = scores[query, live].astype(np.float64)
            log_student = student_logits - np.logaddexp.reduce(student_logits)
            losses.append(np.sum(target * (np.log(target) - log_student)))
    with config.mesh:
        actual = attention._indexer_distillation_loss(
            jnp.asarray(queries),
            jnp.asarray(keys),
            jnp.asarray(proxy),
            MaskInfo.dynamic_init(attention_mask=jnp.asarray(visible)),
        )
    np.testing.assert_allclose(float(actual), np.mean(losses), rtol=2e-5, atol=2e-6)
