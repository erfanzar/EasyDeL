"""Prefix-shared GRPO forward: parity with the repeated-prompt forward and fail-closed validation."""

from types import SimpleNamespace

import easydel as ed
import jax
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.group_relative_policy_optimization._fn import (
    get_per_token_logps,
    get_per_token_logps_prefix_shared,
    grpo_step,
    validate_prefix_shared_model,
)
from easydel.trainers.group_relative_policy_optimization.grpo_config import GRPOConfig
from jax import numpy as jnp
from jax.sharding import PartitionSpec

PROMPTS, PROMPT_LEN, COMPLETION_LEN = 2, 6, 5


def _config():
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        mlp_bias=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )


def _model(cfg):
    return ed.LlamaForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _batch(num_generations):
    rng = np.random.default_rng(num_generations)
    prompt_mask = np.ones((PROMPTS, PROMPT_LEN), np.int32)
    prompt_mask[0, :2] = 0  # left-padded ragged prompt
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 120, (PROMPTS, PROMPT_LEN)), 0).astype(np.int32)
    rows = PROMPTS * num_generations
    lengths = rng.integers(1, COMPLETION_LEN + 1, rows)
    completion_mask = (np.arange(COMPLETION_LEN)[None, :] < lengths[:, None]).astype(np.int32)
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 120, (rows, COMPLETION_LEN)), 0).astype(np.int32)
    advantages = rng.standard_normal(rows).astype(np.float32)
    return prompt_ids, prompt_mask, completion_ids, completion_mask, advantages


@pytest.mark.parametrize("num_generations", [1, 2, 4])
def test_shared_per_token_logps_match_repeated_prompt_forward(num_generations):
    cfg = _config()
    prompt_ids, prompt_mask, completion_ids, completion_mask, _ = _batch(num_generations)
    with jax.default_matmul_precision("highest"), cfg.mesh:
        model = _model(cfg)
        shared = get_per_token_logps_prefix_shared(
            model,
            jnp.asarray(prompt_ids),
            jnp.asarray(prompt_mask),
            jnp.asarray(completion_ids),
            jnp.asarray(completion_mask),
            num_generations,
        )
        input_ids = np.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
        attention_mask = np.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)
        reference = get_per_token_logps(model, jnp.asarray(input_ids), jnp.asarray(attention_mask), PROMPT_LEN)
    valid = completion_mask > 0
    error = np.max(np.abs(np.asarray(shared)[valid] - np.asarray(reference)[valid]))
    assert error <= 1e-5, error


def _grpo(state, batch, num_generations, prefix_shared, **kwargs):
    return grpo_step(
        state,
        batch,
        num_generations,
        0.0,
        partition_spec=PartitionSpec(),
        is_training=True,
        loss_type="dapo",
        prefix_shared_forward=prefix_shared,
        **kwargs,
    )


@pytest.mark.parametrize("num_generations", [1, 2, 4])
def test_grpo_step_loss_and_gradients_match_repeated_prompt_forward(num_generations):
    cfg = _config()
    prompt_ids, prompt_mask, completion_ids, completion_mask, advantages = _batch(num_generations)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask, jnp.float32),
        "advantages": jnp.asarray(advantages),
    }
    with jax.default_matmul_precision("highest"), cfg.mesh:
        state = EasyDeLState.create(model=_model(cfg), tx=optax.sgd(1.0), init_opt_state=True)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
        ref_state, ref_metrics = _grpo(state, batch, num_generations, False)
        shared_state, shared_metrics = _grpo(state, batch, num_generations, True)
        ref_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(ref_state.graphstate)]
        shared_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(shared_state.graphstate)]
    np.testing.assert_allclose(float(shared_metrics.loss), float(ref_metrics.loss), rtol=1e-4, atol=1e-6)
    # SGD at lr=1: parameter delta is minus the gradient.
    assert any(np.max(np.abs(b - a)) > 0 for b, a in zip(before, ref_after, strict=True))
    for b, ref, shared in zip(before, ref_after, shared_after, strict=True):
        np.testing.assert_allclose(b - shared, b - ref, rtol=1e-4, atol=1e-6)


def test_grpo_step_chunked_loss_raises_under_prefix_shared():
    cfg = _config()
    prompt_ids, prompt_mask, completion_ids, completion_mask, advantages = _batch(2)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask, jnp.float32),
        "advantages": jnp.asarray(advantages),
    }
    with cfg.mesh:
        state = EasyDeLState.create(model=_model(cfg), tx=optax.sgd(1.0), init_opt_state=True)
        with pytest.raises(NotImplementedError, match="chunked"):
            _grpo(state, batch, 2, True, completion_chunk_size=1)


def _stub(**overrides):
    values = dict(attn_mechanism="vanilla", kv_cache_quantization_config=None, sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    values.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**values))


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(attn_mechanism="flash_attn2"), "attn_mechanism"),
        (dict(attn_mechanism="sdpa"), "attn_mechanism"),
        (dict(kv_cache_quantization_config=object()), "quantized"),
        (dict(attention_dropout=0.1), "dropout"),
        (dict(sharding_axis_dims=(2, 1, -1, 1, 1, 1)), "pipeline"),
        (dict(use_sliding_window=True), "sliding"),
        (dict(layer_types=["linear_attention", "full_attention"]), "full-attention"),
    ],
)
def test_unsupported_models_are_rejected(overrides, match):
    with pytest.raises(NotImplementedError, match=match):
        validate_prefix_shared_model(_stub(**overrides))


def test_supported_dense_model_is_accepted():
    validate_prefix_shared_model(_stub(layer_types=["full_attention", "full_attention"]))


@pytest.mark.parametrize(
    "kwargs,match",
    [(dict(completion_chunk_size=4), "completion_chunk_size"), (dict(top_entropy_quantile=0.5), "top_entropy")],
)
def test_config_rejects_unsupported_combinations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GRPOConfig(prefix_shared_forward=True, **kwargs)


def test_config_default_keeps_prefix_sharing_off():
    assert GRPOConfig().prefix_shared_forward is False
