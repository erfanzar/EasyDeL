"""Numerical checks for the VAPO extensions of the PPO trainer (arXiv:2504.05118)."""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.proximal_policy_optimization_trainer import PPOConfig, PPOTrainer
from easydel.trainers.proximal_policy_optimization_trainer._fn import (
    _token_normalizer,
    clipped_policy_loss,
    compute_gae,
    length_adaptive_gae_lambda,
    ppo_step,
)
from easydel.trainers.vapo_trainer import VAPOConfig, VAPOTrainer
from easydel.utils import Registry
from jax import numpy as jnp
from jax.sharding import PartitionSpec

VOCAB, HIDDEN, PROMPT = 5, 3, 2


def _trainer(config):
    trainer = object.__new__(PPOTrainer)
    trainer.arguments = config
    return trainer


def test_compute_gae_limits_match_monte_carlo_and_td_residuals():
    rewards = np.array([[0.5, -0.2, 1.0, 0.0], [2.0, 0.3, 0.0, 0.0]], dtype=np.float32)
    values = np.array([[0.1, 0.4, -0.3, 9.0], [0.7, -0.5, 9.0, 9.0]], dtype=np.float32)
    mask = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)

    adv_mc, ret_mc = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    # gamma = lambda = 1: returns are reward-to-go, independent of the critic.
    np.testing.assert_allclose(ret_mc[0, :3], [1.3, 0.8, 1.0], atol=1e-6)
    np.testing.assert_allclose(ret_mc[1, :2], [2.3, 0.3], atol=1e-6)
    np.testing.assert_array_equal(np.asarray(adv_mc) * (1 - mask), 0.0)

    adv_td, _ = compute_gae(rewards, values, mask, gamma=0.9, lam=0.0)
    # lambda = 0: one-step TD residual; the last valid token bootstraps from 0.
    expected_row0 = [0.5 + 0.9 * 0.4 - 0.1, -0.2 + 0.9 * -0.3 - 0.4, 1.0 - (-0.3)]
    np.testing.assert_allclose(adv_td[0, :3], expected_row0, atol=1e-6)
    np.testing.assert_allclose(adv_td[1, :2], [2.0 + 0.9 * -0.5 - 0.7, 0.3 + 0.5], atol=1e-6)

    # A per-row lambda equals running each row with its own scalar lambda.
    per_row, _ = compute_gae(rewards, values, mask, gamma=1.0, lam=jnp.array([1.0, 0.0]))
    np.testing.assert_allclose(per_row[0], adv_mc[0], atol=1e-6)
    np.testing.assert_allclose(per_row[1], compute_gae(rewards, values, mask, gamma=1.0, lam=0.0)[0][1], atol=1e-6)


def test_length_adaptive_lambda_schedule():
    lambdas = length_adaptive_gae_lambda(jnp.array([0, 3, 20, 40, 1000]), alpha=0.05)
    np.testing.assert_allclose(lambdas, [0.0, 0.0, 0.0, 0.5, 0.98], atol=1e-6)
    with pytest.raises(ValueError, match="alpha"):
        length_adaptive_gae_lambda(jnp.array([4]), alpha=0.0)


def test_decoupled_length_adaptive_gae_uses_separate_policy_and_critic_lambdas():
    long_len, length = 40, 41
    rewards = np.zeros((2, length), dtype=np.float32)
    rewards[0, 2] = 1.0
    rewards[1, long_len - 1] = 3.0
    rewards[1, 0] = 0.25
    values = np.random.default_rng(0).normal(size=(2, length)).astype(np.float32)
    mask = np.zeros((2, length), dtype=np.float32)
    mask[0, :3] = 1.0
    mask[1, :long_len] = 1.0
    config = PPOConfig(gamma=1.0, lam=0.95, gae_lambda_critic=1.0, length_adaptive_gae=True)
    advantages, returns, lambdas = _trainer(config)._compute_advantages_and_returns(rewards, values, mask)

    np.testing.assert_allclose(lambdas, [0.0, 0.5], atol=1e-6)
    # Critic lambda = 1 with gamma = 1: reward-to-go targets, independent of values.
    np.testing.assert_allclose(returns[0, :3], [1.0, 1.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(returns[1, 0], 3.25, atol=1e-5)
    np.testing.assert_allclose(returns[1, 1:long_len], 3.0, atol=1e-5)
    # Short row: policy lambda clips to 0, so advantages are TD residuals.
    td_row0 = [values[0, 1] - values[0, 0], values[0, 2] - values[0, 1], 1.0 - values[0, 2]]
    np.testing.assert_allclose(advantages[0, :3], td_row0, atol=1e-5)
    # Long row: the first advantage obeys A_0 = delta_0 + 0.5 * A_1.
    delta0 = 0.25 + values[1, 1] - values[1, 0]
    np.testing.assert_allclose(advantages[1, 0], delta0 + 0.5 * advantages[1, 1], atol=1e-5)


def _numpy_gae(rewards, values, mask, gamma, lam):
    """Independent float64 GAE recursion over right-padded rows (zero bootstrap past the last valid token)."""
    rewards, values, mask = (np.asarray(x, dtype=np.float64) for x in (rewards, values, mask))
    advantages = np.zeros_like(rewards)
    for row in range(rewards.shape[0]):
        length = int(mask[row].sum())
        next_adv, next_value = 0.0, 0.0
        for t in reversed(range(length)):
            delta = rewards[row, t] + gamma * next_value - values[row, t]
            next_adv = delta + gamma * lam * next_adv
            next_value = values[row, t]
            advantages[row, t] = next_adv
    return advantages, advantages + values


def test_default_config_keeps_classic_gae():
    rng = np.random.default_rng(1)
    rewards = rng.normal(size=(3, 6)).astype(np.float32)
    values = rng.normal(size=(3, 6)).astype(np.float32)
    mask = np.array([[1] * 6, [1] * 4 + [0] * 2, [1] + [0] * 5], dtype=np.float32)
    trainer = _trainer(PPOConfig(gamma=0.97, lam=0.9))
    advantages, returns, lambdas = trainer._compute_advantages_and_returns(rewards, values, mask)
    expected_adv, expected_ret = _numpy_gae(rewards, values, mask, gamma=0.97, lam=0.9)
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5, atol=1e-5)
    valid = mask.astype(bool)
    np.testing.assert_allclose(np.asarray(returns)[valid], expected_ret[valid], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lambdas, 0.9)


def test_clip_higher_only_moves_the_upper_bound():
    old = jnp.zeros((1, 2))
    new = jnp.log(jnp.array([[1.25, 0.7]]))
    adv = jnp.array([[2.0, -1.0]])
    symmetric, _, clipped_sym = clipped_policy_loss(new, old, adv, cliprange=0.2)
    higher, ratio, clipped_high = clipped_policy_loss(new, old, adv, cliprange=0.2, cliprange_high=0.28)
    np.testing.assert_allclose(ratio, [[1.25, 0.7]], rtol=1e-6)
    np.testing.assert_allclose(symmetric, [[-2.0 * 1.2, 0.8]], rtol=1e-6)
    np.testing.assert_allclose(higher, [[-2.0 * 1.25, 0.8]], rtol=1e-6)
    np.testing.assert_array_equal(clipped_sym, [[1.0, 1.0]])
    np.testing.assert_array_equal(clipped_high, [[0.0, 1.0]])


def test_token_normalizer_global_count_compensates_accumulation():
    mask = jnp.array([[1.0, 1.0, 0.0]])
    assert float(_token_normalizer(mask, jnp.asarray(12.0), token_level_loss=False, gradient_accumulation_steps=4)) == 2
    assert float(_token_normalizer(mask, jnp.asarray(12.0), token_level_loss=True, gradient_accumulation_steps=4)) == 3
    assert float(_token_normalizer(mask, None, token_level_loss=True, gradient_accumulation_steps=4)) == 2
    assert float(_token_normalizer(jnp.zeros((1, 3)), None, token_level_loss=False, gradient_accumulation_steps=1)) == 1


class TinyValueHead(spx.Module):
    def __init__(self, weight):
        self.weight = spx.Parameter(jnp.asarray(weight, dtype=jnp.float32))

    def forward(self, hidden_states):
        return hidden_states @ self.weight.value


class TinyPolicy(spx.Module):
    def __init__(self, mesh, seed=0):
        rng = np.random.default_rng(seed)
        self.embed = spx.Parameter(jnp.asarray(rng.normal(size=(VOCAB, HIDDEN)), dtype=jnp.float32))
        self.head = spx.Parameter(jnp.asarray(rng.normal(size=(HIDDEN, VOCAB)), dtype=jnp.float32))
        self.value_head = TinyValueHead(rng.normal(size=(HIDDEN, 1)))
        self.mesh = mesh

    def forward(self, input_ids, attention_mask, output_hidden_states=False, apply_lm_head=True):
        hidden = self.embed.value[input_ids]
        logits = hidden @ self.head.value
        return SimpleNamespace(logits=logits if apply_lm_head else None, last_hidden_state=hidden)


def _numpy_forward(state, input_ids):
    model = state.model
    embed = np.asarray(model.embed.value, dtype=np.float64)
    head = np.asarray(model.head.value, dtype=np.float64)
    value_w = np.asarray(model.value_head.weight.value, dtype=np.float64)
    hidden = embed[np.asarray(input_ids)][:, PROMPT - 1 : -1]
    logits = hidden @ head
    log_probs = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    targets = np.asarray(input_ids)[:, PROMPT:]
    logps = np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    entropies = -(np.exp(log_probs) * log_probs).sum(-1)
    return logps, (hidden @ value_w)[..., 0], entropies


def _make_state(tx=None, seed=0):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("devices",))
    state = EasyDeLState.create(model=TinyPolicy(mesh, seed=seed))
    if tx is None:
        return state
    return state.replace(tx=tx, opt_state=tx.init(state.graphstate))


def _make_batch(state, ratios, positive):
    input_ids = jnp.array([[1, 2, 3, 4, 0], [0, 3, 1, 2, 4], [2, 2, 4, 1, 3], [4, 1, 0, 0, 2]], dtype=jnp.int32)
    completion_mask = np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.float32)
    logps, values, _ = _numpy_forward(state, input_ids)
    rng = np.random.default_rng(3)
    old_logps = (logps - np.log(ratios)).astype(np.float32)
    old_values = (values + rng.normal(scale=0.5, size=values.shape)).astype(np.float32)
    advantages = rng.normal(size=values.shape).astype(np.float32) * completion_mask
    returns = rng.normal(size=values.shape).astype(np.float32)
    positive = np.asarray(positive, dtype=np.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": jnp.ones_like(input_ids),
        "completion_mask": jnp.asarray(completion_mask),
        "old_logps": jnp.asarray(old_logps),
        "old_values": jnp.asarray(old_values),
        "advantages": jnp.asarray(advantages),
        "returns": jnp.asarray(returns),
        "positive_mask": jnp.asarray(positive),
        "num_items_in_batch": jnp.asarray(completion_mask.sum()),
        "num_positive_tokens": jnp.asarray((completion_mask * positive[:, None]).sum()),
    }


def _step(state, batch, *, is_training, gradient_accumulation_steps=1, **vapo):
    options = {
        "cliprange": 0.2,
        "vf_coef": 0.5,
        "cliprange_value": 0.2,
        "entropy_coef": 0.01,
        "cliprange_high": None,
        "token_level_loss": False,
        "positive_nll_coef": 0.0,
        "value_pretrain_steps": 0,
        **vapo,
    }
    return ppo_step(
        state,
        batch,
        PROMPT,
        options.pop("cliprange"),
        options.pop("vf_coef"),
        options.pop("cliprange_value"),
        options.pop("entropy_coef"),
        None,
        partition_spec=PartitionSpec(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_training=is_training,
        **options,
    )


def _other_metrics(metrics) -> dict[str, jax.Array]:
    other = metrics.other_metrics
    assert other is not None
    return other


def test_ppo_step_vapo_loss_matches_numpy_reference():
    state = _make_state()
    ratios = np.array([[1.25, 0.7, 1.0], [1.4, 1.0, 1.0], [0.9, 1.1, 1.0], [0.75, 1.3, 1.05]])
    batch = _make_batch(state, ratios, positive=[1, 0, 1, 0])
    # Compare against a float64 reference, so keep TPU f32 matmuls at full precision.
    with jax.default_matmul_precision("highest"):
        metrics = _step(
            state,
            batch,
            is_training=False,
            cliprange_high=0.28,
            token_level_loss=True,
            positive_nll_coef=0.1,
        )

    logps, values, entropies = _numpy_forward(state, batch["input_ids"])
    m = np.asarray(batch["completion_mask"], dtype=np.float64)
    adv = np.asarray(batch["advantages"], dtype=np.float64)
    ratio = np.exp(logps - np.asarray(batch["old_logps"], dtype=np.float64))
    pg = np.maximum(-adv * ratio, -adv * np.clip(ratio, 0.8, 1.28))
    pg_loss = (pg * m).sum() / m.sum()
    old_v = np.asarray(batch["old_values"], dtype=np.float64)
    ret = np.asarray(batch["returns"], dtype=np.float64)
    v_clip = old_v + np.clip(values - old_v, -0.2, 0.2)
    vf_loss = 0.5 * (np.maximum((values - ret) ** 2, (v_clip - ret) ** 2) * m).sum() / m.sum()
    entropy = (entropies * m).sum() / m.sum()
    pos = m * np.array([1, 0, 1, 0])[:, None]
    nll = -(logps * pos).sum() / pos.sum()
    expected = pg_loss + 0.5 * vf_loss - 0.01 * entropy + 0.1 * nll

    np.testing.assert_allclose(_other_metrics(metrics)["policy_loss"], pg_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(_other_metrics(metrics)["positive_nll"], nll, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(metrics.loss, expected, rtol=1e-5, atol=1e-6)


def _leaves_by_path(state):
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in jax.tree_util.tree_leaves_with_path(state)}


def test_token_level_loss_is_invariant_to_gradient_accumulation():
    ratios = np.array([[1.25, 0.7, 1.0], [1.4, 1.0, 1.0], [0.9, 1.1, 1.0], [0.75, 1.3, 1.05]])
    updated = {}
    for token_level in (True, False):
        for steps in (1, 2):
            state = _make_state(tx=optax.sgd(0.5))
            batch = _make_batch(state, ratios, positive=[1, 1, 0, 1])
            new_state, _ = _step(
                state,
                batch,
                is_training=True,
                gradient_accumulation_steps=steps,
                vf_coef=0.0,
                entropy_coef=0.0,
                cliprange_high=0.28,
                token_level_loss=token_level,
                positive_nll_coef=0.1,
            )
            updated[token_level, steps] = _leaves_by_path(new_state.graphstate)

    for key, full in updated[True, 1].items():
        np.testing.assert_allclose(updated[True, 2][key], full, rtol=1e-5, atol=1e-6)
    # Sequence-local means weight microbatches with unequal token counts differently.
    assert any(
        not np.allclose(updated[False, 2][key], full, rtol=1e-5, atol=1e-6) for key, full in updated[False, 1].items()
    )


def test_value_pretraining_updates_only_the_value_head():
    ratios = np.array([[1.25, 0.7, 1.0], [1.4, 1.0, 1.0], [0.9, 1.1, 1.0], [0.75, 1.3, 1.05]])
    state = _make_state(tx=optax.sgd(0.5))
    batch = _make_batch(state, ratios, positive=[1, 0, 1, 0])
    vapo = {"value_pretrain_steps": 2, "positive_nll_coef": 0.1, "cliprange_high": 0.28}
    before = _leaves_by_path(state.graphstate)

    eval_metrics = _step(state, batch, is_training=False, **vapo)
    np.testing.assert_allclose(eval_metrics.loss, 0.5 * _other_metrics(eval_metrics)["value_loss"], rtol=1e-6)
    assert float(_other_metrics(eval_metrics)["value_pretrain_active"]) == 1.0

    pretrained, _ = _step(state, batch, is_training=True, **vapo)
    after = _leaves_by_path(pretrained.graphstate)
    for key, value in before.items():
        if "value_head" in key:
            assert not np.allclose(after[key], value), key
        else:
            np.testing.assert_array_equal(after[key], value, err_msg=key)

    joint_state = state.replace(step=jnp.asarray(2, dtype=jnp.int32))
    joint_metrics = _step(joint_state, batch, is_training=False, **vapo)
    assert float(_other_metrics(joint_metrics)["value_pretrain_active"]) == 0.0
    joint, _ = _step(joint_state, batch, is_training=True, **vapo)
    joint_after = _leaves_by_path(joint.graphstate)
    assert any(not np.allclose(joint_after[key], before[key]) for key in before if "value_head" not in key)


def test_value_pretraining_freezes_policy_under_weight_decay():
    ratios = np.array([[1.25, 0.7, 1.0], [1.4, 1.0, 1.0], [0.9, 1.1, 1.0], [0.75, 1.3, 1.05]])
    state = _make_state(tx=optax.adamw(1e-2, weight_decay=0.1))
    batch = _make_batch(state, ratios, positive=[1, 0, 1, 0])
    vapo = {"value_pretrain_steps": 2}
    before = _leaves_by_path(state.graphstate)
    opt_before = _leaves_by_path(state.opt_state)

    pretrained, _ = _step(state, batch, is_training=True, **vapo)
    assert int(pretrained.step) == 1
    after = _leaves_by_path(pretrained.graphstate)
    for key, value in before.items():
        if "value_head" in key:
            assert not np.allclose(after[key], value), key
        else:
            np.testing.assert_array_equal(after[key], value, err_msg=key)
    opt_after = _leaves_by_path(pretrained.opt_state)
    value_head_moment_moved = False
    for key, value in opt_before.items():
        if "value_head" in key:
            value_head_moment_moved |= not np.array_equal(opt_after[key], value)
        elif ("embed" in key or "'head'" in key) and "value_head" not in key:
            np.testing.assert_array_equal(opt_after[key], value, err_msg=key)
    assert value_head_moment_moved

    # Once pretraining ends, decay and the policy gradient move the policy again.
    joint, _ = _step(pretrained.replace(step=jnp.asarray(2, dtype=jnp.int32)), batch, is_training=True, **vapo)
    joint_after = _leaves_by_path(joint.graphstate)
    for key in before:
        if "value_head" not in key:
            assert not np.allclose(joint_after[key], before[key]), key


def test_vapo_config_defaults_registry_and_static_args():
    config = VAPOConfig()
    assert config.trainer_prefix == "VAPO"
    assert config.gae_lambda_critic == 1.0
    assert config.length_adaptive_gae is True
    assert config.length_adaptive_gae_alpha == 0.05
    assert config.cliprange_high == 0.28
    assert config.token_level_loss is True
    assert config.positive_nll_coef == 0.1
    assert config.value_pretrain_steps == 0
    assert issubclass(VAPOTrainer, PPOTrainer)
    assert Registry.get_or_raise("trainer-arguments", "vapo", wakeup=False) is VAPOConfig
    assert Registry.get_or_raise("trainer", "vapo", wakeup=False) is VAPOTrainer
    # The trailing static args are spliced positionally after the 14 fixed ppo_step arguments, so
    # bind them to the real signature and check every value lands on the parameter it configures.
    fixed_args = tuple(f"fixed{i}" for i in range(14))
    signature = inspect.signature(ppo_step)
    for cfg, expected in (
        (config, {"cliprange_high": 0.28, "token_level_loss": True, "positive_nll_coef": 0.1}),
        (PPOConfig(), {"cliprange_high": None, "token_level_loss": False, "positive_nll_coef": 0.0}),
        (PPOConfig(value_pretrain_steps=7), {"value_pretrain_steps": 7}),
    ):
        bound = signature.bind(*fixed_args, *_trainer(cfg)._vapo_static_args()).arguments
        assert bound["straight_through_emulator"] == "fixed13"
        assert set(bound) == set(signature.parameters)
        for name, value in expected.items():
            assert bound[name] == value, name
            assert type(bound[name]) is type(value), name


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"gae_lambda_critic": 1.5}, "gae_lambda_critic"),
        ({"length_adaptive_gae_alpha": 0.0}, "length_adaptive_gae_alpha"),
        ({"cliprange_high": -0.1}, "cliprange_high"),
        ({"positive_nll_coef": -1.0}, "positive_nll_coef"),
        ({"value_pretrain_steps": -1}, "value_pretrain_steps"),
        ({"value_pretrain_steps": 3, "vf_coef": 0.0}, "vf_coef"),
    ],
)
def test_ppo_config_rejects_invalid_vapo_knobs(overrides, match):
    with pytest.raises(ValueError, match=match):
        PPOConfig(**overrides)
