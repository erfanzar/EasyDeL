"""Accelerator-only end-to-end numerical checks for ORPO, PPO and online DPO trainers."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.direct_preference_optimization_trainer import DPOTrainer
from easydel.trainers.odds_ratio_preference_optimization_trainer._fn import concatenated_forward
from easydel.trainers.on_policy_distillation_trainer._fn import on_policy_distillation_step
from easydel.trainers.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from easydel.trainers.proximal_policy_optimization_trainer import PPOConfig, PPOTrainer
from easydel.trainers.trainer import Trainer
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from .test_preprocessing import TextTokenizer
from .test_ray_orpo_numerics import FixedLogits, TinyReward


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_orpo_rejects_pairs_without_causally_scoreable_completions(chunk_size):
    batch = {
        "chosen_input_ids": jnp.array([[1, 2, 3, 0]]),
        "chosen_attention_mask": jnp.array([[1, 1, 1, 0]]),
        "chosen_labels": jnp.array([[-100, -100, -100, -100]]),
        "rejected_input_ids": jnp.array([[1, 2, 3, 0]]),
        "rejected_attention_mask": jnp.array([[1, 1, 1, 0]]),
        "rejected_labels": jnp.array([[-100, -100, 3, -100]]),
    }
    with pytest.raises(ValueError, match="causally scoreable completion"):
        concatenated_forward(FixedLogits(chunk_size), batch, False, -100, 0)


@pytest.mark.parametrize("custom", [None, "bound", "graphdef"])
def test_ppo_prebuilt_rewards_install_or_preserve_forward(monkeypatch, custom):
    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    monkeypatch.setattr(PPOTrainer, "requires_value_head", False)
    module = TinyReward()
    module.config = SimpleNamespace(pad_token_id=0)
    state = EasyDeLState.create(model=module)
    if custom == "bound":
        state = state.replace(apply_fn=lambda gs, go, batch: {"logits": batch["input_ids"] + 7})
    elif custom == "graphdef":
        state = state.replace(apply_fn=lambda gd, gs, go, batch: {"logits": batch["input_ids"] + 9})
    trainer = PPOTrainer(
        PPOConfig(kl_coef=0), state, state, processing_class=TextTokenizer(), reward_processing_classes=TextTokenizer()
    )
    reward = trainer.reward_funcs[0]
    output = reward.apply_fn(
        reward.graphdef,
        reward.graphstate,
        reward.graphother,
        {"input_ids": jnp.array([[1, 2]]), "attention_mask": jnp.ones((1, 2)), "unused": jnp.array([1])},
    )
    expected = [[8, 9]] if custom == "bound" else ([[10, 11]] if custom == "graphdef" else [[6]])
    np.testing.assert_allclose(output["logits"], expected)


def test_online_dpo_callbacks_receive_candidate_aligned_answers(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)

    def reward(*, completions, answer):
        return [float(completion == expected) for completion, expected in zip(completions, answer, strict=True)]

    config = OnlineDPOConfig()
    trainer = OnlineDPOTrainer(config, model=None, reward_funcs=reward)
    trainer.arguments = config
    rewards = trainer._score_online_dpo_rewards(
        prompts=["p0", "p0", "p1", "p1"],
        completions=["a0", "bad", "bad", "a1"],
        completion_ids=jnp.ones((4, 1), dtype=jnp.int32),
        batch={"prompt": ["p0", "p1"], "answer": ["a0", "a1"]},
    )
    np.testing.assert_array_equal(rewards, [1, 0, 0, 1])


class ConstantDistillModel(spx.Module):
    def __init__(self, logits, mesh):
        self.values = spx.Parameter(jnp.asarray(logits, dtype=jnp.float32))
        self.mesh = mesh

    def forward(self, input_ids, attention_mask, apply_lm_head=True):
        logits = jnp.broadcast_to(self.values.value, (*input_ids.shape, self.values.value.shape[0]))
        return SimpleNamespace(logits=logits if apply_lm_head else None, last_hidden_state=logits)

    def make_lm_head_fn(self):
        return lambda hidden: hidden


@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize("selector", ["forward", "reverse", "jsd", "cakld", "topk_tail"])
def test_on_policy_distillation_honors_objective_selectors(chunk_size, selector):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("devices",))
    student = EasyDeLState.create(model=ConstantDistillModel([0.3, -0.2, 1.1], mesh))
    teacher = EasyDeLState.create(model=ConstantDistillModel([1.2, 0.4, -0.1], mesh))
    batch = {
        "input_ids": jnp.ones((4, 3), dtype=jnp.int32),
        "attention_mask": jnp.ones((4, 3), dtype=jnp.int32),
        "completion_mask": jnp.array([[0, 1, 1]] * 4),
    }
    options = {
        "reverse": {"beta": 1.0},
        "jsd": {"beta": 0.5},
        "cakld": {"cakld_gamma": 0.3},
        "topk_tail": {"loss_top_k": 2, "loss_add_tail": True},
    }.get(selector, {})
    metrics = on_policy_distillation_step(
        student,
        batch,
        teacher,
        partition_spec=PartitionSpec(),
        is_training=False,
        temperature=1.0,
        alpha=1.0,
        logits_chunk_size=chunk_size,
        **options,
    )
    p = np.exp(np.array([1.2, 0.4, -0.1], dtype=np.float64))
    q = np.exp(np.array([0.3, -0.2, 1.1], dtype=np.float64))
    p /= p.sum()
    q /= q.sum()
    forward = np.sum(p * np.log(p / q))
    reverse = np.sum(q * np.log(q / p))
    if selector == "jsd":
        mixture = (p + q) / 2
        expected = (np.sum(p * np.log(p / mixture)) + np.sum(q * np.log(q / mixture))) / 2
    elif selector == "reverse":
        expected = reverse
    elif selector == "cakld":
        expected = 0.3 * reverse + 0.7 * forward
    else:
        # Top two teacher tokens plus the (single remaining) tail reproduce
        # the full three-token distribution exactly.
        expected = forward
    np.testing.assert_allclose(metrics.loss, expected, rtol=1e-5, atol=1e-6)
