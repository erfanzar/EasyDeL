"""Prefix-shared GRPO forward on hybrid qwen3_next (full attention + gated-delta linear attention)."""

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
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig
from jax import lax
from jax import numpy as jnp
from jax.sharding import PartitionSpec

PROMPTS, PROMPT_LEN, COMPLETION_LEN, D_CONV = 2, 6, 5, 4
LAYER_TYPES = ["linear_attention", "full_attention", "linear_attention"]

# The gated-delta rule runs pinned to the XLA kernel, whose matmuls use HIGHEST precision: the TPU Pallas
# kernel's matmuls run at DEFAULT precision (bf16 passes), which overrides the ambient
# ``jax.default_matmul_precision("highest")`` and puts both forwards about 2e-4 apart for any reordering.
_GDR_OPERATION_CONFIGS = {"gated_delta_rule": GatedDeltaRuleConfig(platform="xla", backend="any", chunk_size=16)}

# Measured on TPU v5p with the pinned XLA kernel: log-probs <= 4.8e-7, parameter updates <= 1.2e-7.
LOGPS_ATOL = 2e-6
GRAD_RTOL, GRAD_ATOL = 1e-4, 1e-6


def _config(moe=False, max_position_embeddings=64, pin_gdr=True):
    extra = {"operation_configs": _GDR_OPERATION_CONFIGS} if pin_gdr else {}
    return ed.Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=list(LAYER_TYPES),
        linear_conv_kernel_dim=D_CONV,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        mlp_only_layers=[] if moe else list(range(len(LAYER_TYPES))),
        max_position_embeddings=max_position_embeddings,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        **extra,
    )


def _model(cfg):
    return ed.Qwen3NextForCausalLM(
        config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32, precision=lax.Precision.HIGHEST
    )


def _batch(num_generations, prompt_len=PROMPT_LEN, extra_left_pad=0):
    rng = np.random.default_rng(num_generations)
    prompt_mask = np.ones((PROMPTS, PROMPT_LEN), np.int32)
    prompt_mask[0, :4] = 0  # ragged prompt of 2 tokens, shorter than d_conv
    prompt_mask[1, :1] = 0
    prompt_ids = np.where(prompt_mask > 0, rng.integers(1, 120, (PROMPTS, PROMPT_LEN)), 0).astype(np.int32)
    if extra_left_pad:
        prompt_mask = np.pad(prompt_mask, ((0, 0), (extra_left_pad, 0)))
        prompt_ids = np.pad(prompt_ids, ((0, 0), (extra_left_pad, 0)))
    rows = PROMPTS * num_generations
    lengths = rng.integers(1, COMPLETION_LEN + 1, rows)
    completion_mask = (np.arange(COMPLETION_LEN)[None, :] < lengths[:, None]).astype(np.int32)
    completion_ids = np.where(completion_mask > 0, rng.integers(1, 120, (rows, COMPLETION_LEN)), 0).astype(np.int32)
    advantages = rng.standard_normal(rows).astype(np.float32)
    return prompt_ids, prompt_mask, completion_ids, completion_mask, advantages


def _shared_and_reference(model, prompt_ids, prompt_mask, completion_ids, completion_mask, num_generations):
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
    reference = get_per_token_logps(model, jnp.asarray(input_ids), jnp.asarray(attention_mask), prompt_ids.shape[1])
    return np.asarray(shared), np.asarray(reference)


@pytest.mark.parametrize("moe,num_generations", [(False, 1), (False, 2), (False, 3), (True, 2)])
def test_hybrid_shared_per_token_logps_match_repeated_prompt_forward(moe, num_generations):
    cfg = _config(moe)
    prompt_ids, prompt_mask, completion_ids, completion_mask, _ = _batch(num_generations)
    with jax.default_matmul_precision("highest"), cfg.mesh:
        shared, reference = _shared_and_reference(
            _model(cfg), prompt_ids, prompt_mask, completion_ids, completion_mask, num_generations
        )
    valid = completion_mask > 0
    error = float(np.max(np.abs(shared[valid] - reference[valid])))
    print(f"[hybrid logps] moe={moe} G={num_generations} max_abs_err={error:.3e}")
    assert error <= LOGPS_ATOL, error


def _grpo(state, batch, num_generations, prefix_shared):
    return grpo_step(
        state,
        batch,
        num_generations,
        0.0,
        partition_spec=PartitionSpec(),
        is_training=True,
        loss_type="dapo",
        prefix_shared_forward=prefix_shared,
    )


@pytest.mark.parametrize("moe,num_generations", [(False, 1), (False, 2), (False, 3), (True, 2)])
def test_hybrid_grpo_step_loss_and_gradients_match_repeated_prompt_forward(moe, num_generations):
    cfg = _config(moe)
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
    loss_err = abs(float(shared_metrics.loss) - float(ref_metrics.loss))
    grad_err = max(
        float(np.max(np.abs((b - s) - (b - r)))) for b, r, s in zip(before, ref_after, shared_after, strict=True)
    )
    print(f"[hybrid grpo] moe={moe} G={num_generations} loss_err={loss_err:.3e} max_update_err={grad_err:.3e}")
    np.testing.assert_allclose(float(shared_metrics.loss), float(ref_metrics.loss), rtol=GRAD_RTOL, atol=GRAD_ATOL)
    # SGD at lr=1: parameter delta is minus the gradient.
    assert any(np.max(np.abs(b - a)) > 0 for b, a in zip(before, ref_after, strict=True))
    for b, ref, shared in zip(before, ref_after, shared_after, strict=True):
        np.testing.assert_allclose(b - shared, b - ref, rtol=GRAD_RTOL, atol=GRAD_ATOL)


def test_hybrid_shared_logps_are_invariant_to_extra_left_padding():
    """Padding rows contribute nothing to the captured conv window or recurrent state."""
    num_generations = 2
    cfg = _config()
    prompt_ids, prompt_mask, completion_ids, completion_mask, _ = _batch(num_generations)
    padded_ids, padded_mask, *_ = _batch(num_generations, extra_left_pad=5)
    with jax.default_matmul_precision("highest"), cfg.mesh:
        model = _model(cfg)
        args = (jnp.asarray(completion_ids), jnp.asarray(completion_mask), num_generations)
        base = get_per_token_logps_prefix_shared(model, jnp.asarray(prompt_ids), jnp.asarray(prompt_mask), *args)
        padded = get_per_token_logps_prefix_shared(model, jnp.asarray(padded_ids), jnp.asarray(padded_mask), *args)
    valid = completion_mask > 0
    error = float(np.max(np.abs(np.asarray(base)[valid] - np.asarray(padded)[valid])))
    print(f"[hybrid padding] max_abs_err={error:.3e}")
    assert error <= LOGPS_ATOL, error


def test_hybrid_rejects_single_token_completions():
    cfg = _config()
    prompt_ids, prompt_mask, completion_ids, completion_mask, _ = _batch(2)
    with cfg.mesh:
        model = _model(cfg)
        with pytest.raises(NotImplementedError, match=">= 2"):
            get_per_token_logps_prefix_shared(
                model,
                jnp.asarray(prompt_ids),
                jnp.asarray(prompt_mask),
                jnp.asarray(completion_ids[:, :1]),
                jnp.asarray(completion_mask[:, :1]),
                2,
            )


def test_recurrent_fork_view_gathers_group_state_and_sums_gradients():
    from easydel.caching.shared_prefix import PrefixRecurrentForkView

    b, g = 2, 3
    rng = np.random.default_rng(0)
    conv = rng.standard_normal((b, 8, D_CONV)).astype(np.float32)
    rec = rng.standard_normal((b, 4, 16, 16)).astype(np.float32)
    weights = rng.standard_normal((b * g, 4, 16, 16)).astype(np.float32)
    view = PrefixRecurrentForkView(
        prefix_conv_state=jnp.asarray(conv), prefix_recurrent_state=jnp.asarray(rec), num_generations=g
    )
    rows = np.arange(b * g) // g
    np.testing.assert_array_equal(np.asarray(view.conv_state), conv[rows])
    np.testing.assert_array_equal(np.asarray(view.recurrent_state), rec[rows])
    assert view.replace(conv_state=None, recurrent_state=None) is view
    leaves, treedef = jax.tree_util.tree_flatten(view)
    assert len(leaves) == 2 and jax.tree_util.tree_unflatten(treedef, leaves).num_generations == g

    def loss(state):
        fork = PrefixRecurrentForkView(
            prefix_conv_state=jnp.asarray(conv), prefix_recurrent_state=state, num_generations=g
        )
        return jnp.sum(fork.recurrent_state * weights)

    grad = jax.grad(loss)(jnp.asarray(rec))
    np.testing.assert_allclose(np.asarray(grad), weights.reshape(b, g, 4, 16, 16).sum(axis=1), rtol=1e-6)


def test_recurrent_capture_view_starts_empty_and_records_state():
    from easydel.caching.shared_prefix import PrefixRecurrentCaptureView

    view = PrefixRecurrentCaptureView()
    assert view.conv_state is None and view.recurrent_state is None
    state = jnp.ones((2, 4, 16, 16))
    captured = view.replace(conv_state=jnp.zeros((2, 8, D_CONV)), recurrent_state=state)
    assert isinstance(captured, PrefixRecurrentCaptureView)
    np.testing.assert_array_equal(np.asarray(captured.recurrent_state), np.asarray(state))


def _stub(**overrides):
    values = dict(attn_mechanism="vanilla", kv_cache_quantization_config=None, sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    values.update(overrides)
    return SimpleNamespace(config=SimpleNamespace(**values))


@pytest.mark.cpu_ok
def test_qwen3_next_hybrid_is_accepted():
    validate_prefix_shared_model(_stub(model_type="qwen3_next", layer_types=list(LAYER_TYPES)))


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "overrides,match",
    [
        # KDA families: kimi_linear names KDA layers explicitly, glm5_next calls them "linear_attention".
        (dict(model_type="kimi_linear", layer_types=["kda_linear_attention", "full_attention"]), "layer_types"),
        (dict(model_type="glm5_next_text", layer_types=["linear_attention", "full_attention"]), "glm5_next_text"),
        (dict(model_type="qwen3_next", layer_types=["linear_attention", "sliding_attention"]), "sliding_attention"),
        (dict(model_type="qwen3_next", layer_types=list(LAYER_TYPES), use_sliding_window=True), "sliding"),
        (dict(layer_types=["linear_attention", "full_attention"]), "full-attention"),
    ],
)
def test_unsupported_hybrid_models_are_rejected(overrides, match):
    with pytest.raises(NotImplementedError, match=match):
        validate_prefix_shared_model(_stub(**overrides))
