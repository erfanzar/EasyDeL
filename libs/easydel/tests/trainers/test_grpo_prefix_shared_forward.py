"""Prefix-shared GRPO forward: parity with the repeated-prompt forward and fail-closed validation."""

from types import SimpleNamespace

import easydel as ed
import easydel.caching.shared_prefix as shared_prefix
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
from ejkernel.modules.operations.configs import FlashAttentionConfig
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

PROMPTS, PROMPT_LEN, COMPLETION_LEN = 2, 6, 5


# flash_attn2 runs pinned to the XLA flash kernel with an explicit HIGHEST precision: the operation
# forwards an explicit precision (default DEFAULT = bf16 passes on TPU) that overrides the ambient
# ``jax.default_matmul_precision("highest")``, and the Pallas TPU kernel neither accepts HIGHEST nor
# runs these tiny shapes. See tests/operations/test_prefix_share_kernel_semantics.py.
_FLASH_OPERATION_CONFIGS = {"flash_attn2": FlashAttentionConfig(platform="xla", backend="any")}
MECHANISMS = ["vanilla", "sdpa", "flash_attn2"]


def _config(attn_mechanism="vanilla", max_position_embeddings=64):
    extra = {"operation_configs": _FLASH_OPERATION_CONFIGS} if attn_mechanism == "flash_attn2" else {}
    return ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=max_position_embeddings,
        mlp_bias=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism=attn_mechanism,
        attn_dtype=jnp.float32,
        **extra,
    )


def _model(cfg):
    precision = lax.Precision.HIGHEST if cfg.attn_mechanism == "flash_attn2" else None
    return ed.LlamaForCausalLM(
        config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32, precision=precision
    )


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


@pytest.mark.parametrize("attn_mechanism", MECHANISMS)
@pytest.mark.parametrize("num_generations", [1, 2, 4])
def test_shared_per_token_logps_match_repeated_prompt_forward(num_generations, attn_mechanism):
    cfg = _config(attn_mechanism)
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


@pytest.mark.parametrize("attn_mechanism", MECHANISMS)
@pytest.mark.parametrize("num_generations", [1, 2, 4])
def test_grpo_step_loss_and_gradients_match_repeated_prompt_forward(num_generations, attn_mechanism):
    cfg = _config(attn_mechanism)
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


@pytest.mark.parametrize("num_generations", [1, 2])
def test_blocksparse_grpo_step_matches_repeated_prompt_forward_at_aligned_lengths(num_generations):
    """512-token prompts and completions drive the TPU blocksparse kernel itself (short ones use its fallback)."""
    prompt_len = completion_len = 512
    cfg = _config("blocksparse", max_position_embeddings=prompt_len + completion_len)
    rng = np.random.default_rng(7 + num_generations)
    prompt_mask = np.ones((PROMPTS, prompt_len), np.int32)
    prompt_mask[0, :37] = 0  # left-padded ragged prompt
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 120, (PROMPTS, prompt_len)), 0).astype(np.int32)
    rows = PROMPTS * num_generations
    lengths = rng.integers(1, completion_len + 1, rows)
    completion_mask = (np.arange(completion_len)[None, :] < lengths[:, None]).astype(np.int32)
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 120, (rows, completion_len)), 0).astype(np.int32)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask, jnp.float32),
        "advantages": jnp.asarray(rng.standard_normal(rows).astype(np.float32)),
    }
    with jax.default_matmul_precision("highest"), cfg.mesh:
        state = EasyDeLState.create(model=_model(cfg), tx=optax.sgd(1.0), init_opt_state=True)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
        ref_state, ref_metrics = _grpo(state, batch, num_generations, False)
        shared_state, shared_metrics = _grpo(state, batch, num_generations, True)
        ref_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(ref_state.graphstate)]
        shared_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(shared_state.graphstate)]
    np.testing.assert_allclose(float(shared_metrics.loss), float(ref_metrics.loss), rtol=1e-4, atol=1e-6)
    assert any(np.max(np.abs(b - a)) > 0 for b, a in zip(before, ref_after, strict=True))
    for b, ref, shared in zip(before, ref_after, shared_after, strict=True):
        np.testing.assert_allclose(b - shared, b - ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("attn_mechanism", ["vanilla", "sdpa"])
def test_dense_mechanisms_take_the_grouped_fork_path_with_causal_query_blocks(monkeypatch, attn_mechanism):
    """vanilla/sdpa never materialize the per-completion prefix copy; 2-row query blocks cover block skipping."""

    def forbid_copy(self, **kwargs):
        raise AssertionError("the per-completion prefix copy must not run for dense mechanisms")

    monkeypatch.setattr(shared_prefix.PrefixForkView, "concatenate_to_cache", forbid_copy)
    monkeypatch.setattr(shared_prefix, "GROUPED_FORK_QUERY_BLOCK", 2)
    num_generations = 4
    cfg = _config(attn_mechanism)
    prompt_ids, prompt_mask, completion_ids, completion_mask, advantages = _batch(num_generations)
    batch = {
        "prompt_ids": jnp.asarray(prompt_ids),
        "prompt_mask": jnp.asarray(prompt_mask),
        "completion_ids": jnp.asarray(completion_ids),
        "completion_mask": jnp.asarray(completion_mask, jnp.float32),
        "advantages": jnp.asarray(advantages),
    }
    with jax.default_matmul_precision("highest"), cfg.mesh:
        model = _model(cfg)
        shared = get_per_token_logps_prefix_shared(
            model,
            batch["prompt_ids"],
            batch["prompt_mask"],
            batch["completion_ids"],
            jnp.asarray(completion_mask),
            num_generations,
        )
        input_ids = np.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
        attention_mask = np.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)
        reference = get_per_token_logps(model, jnp.asarray(input_ids), jnp.asarray(attention_mask), PROMPT_LEN)
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        before = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
        ref_state, ref_metrics = _grpo(state, batch, num_generations, False)
        shared_state, shared_metrics = _grpo(state, batch, num_generations, True)
        ref_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(ref_state.graphstate)]
        shared_after = [np.asarray(x) for x in jax.tree_util.tree_leaves(shared_state.graphstate)]
    valid = completion_mask > 0
    error = np.max(np.abs(np.asarray(shared)[valid] - np.asarray(reference)[valid]))
    print(
        f"MEASURE grouped-fork {attn_mechanism} logps {error:.3e} loss "
        f"{abs(float(shared_metrics.loss) - float(ref_metrics.loss)):.3e} update "
        f"{max(float(np.max(np.abs(s - r))) for s, r in zip(shared_after, ref_after, strict=True)):.3e}"
    )
    assert error <= 1e-5, error
    np.testing.assert_allclose(float(shared_metrics.loss), float(ref_metrics.loss), rtol=1e-4, atol=1e-6)
    for b, ref, shared_param in zip(before, ref_after, shared_after, strict=True):
        np.testing.assert_allclose(b - shared_param, b - ref, rtol=1e-4, atol=1e-6)


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
        (dict(attn_mechanism="ring"), "attn_mechanism"),
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
