"""Tiny accelerator-only regressions; never run as CPU validation."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.direct_preference_optimization_trainer import DPOTrainer
from easydel.trainers.odds_ratio_preference_optimization_trainer._fn import concatenated_forward
from easydel.trainers.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer
from easydel.trainers.prompt_utils import keep_arrays_map
from easydel.trainers.proximal_policy_optimization_trainer._fn import compute_per_token_logps
from easydel.trainers.ray_scaler.distributed_trainer import RayDistributedTrainer
from easydel.trainers.reward_protocol import RewardProtocol
from jax import numpy as jnp


def test_ray_train_stages_supplied_state_on_offload_backend():
    launcher = object.__new__(RayDistributedTrainer)
    launcher.offload_backend = "cpu"
    launcher.create_trainer = lambda **kwargs: SimpleNamespace(train=lambda: kwargs["state"])
    state = {"parameters": jnp.array([1.0, 2.0])}
    result = launcher.train(1, arguments=SimpleNamespace(), dataset_train=None, state=state)
    assert result["parameters"].device.platform == "cpu"
    np.testing.assert_array_equal(result["parameters"], [1.0, 2.0])


def test_ray_train_preserves_distributed_state_sharding():
    launcher = object.__new__(RayDistributedTrainer)
    launcher.offload_backend = "cpu"
    launcher.create_trainer = lambda **kwargs: SimpleNamespace(train=lambda: kwargs["state"])
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("workers",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("workers"))
    parameters = jax.make_array_from_callback((len(jax.devices()), 2), sharding, lambda index: np.ones((1, 2)))
    result = launcher.train(1, arguments=SimpleNamespace(), dataset_train=None, state={"parameters": parameters})
    # This also runs on multi-host hardware: it must not gather non-addressable
    # global arrays onto a single local offload device.
    assert result["parameters"] is parameters
    assert result["parameters"].sharding == sharding
    for shard in result["parameters"].addressable_shards:
        np.testing.assert_array_equal(shard.data, np.ones((1, 2)))


class FixedLogits:
    def __init__(self, chunk_size):
        self.config = SimpleNamespace(lmhead_chunksize=chunk_size)

    def __call__(self, input_ids, attention_mask, apply_lm_head=True):
        logits = jnp.broadcast_to(jnp.array([0.0, 1.0, 2.0, 3.0]), (*input_ids.shape, 4))
        logits = logits + 10.0 * jnp.arange(input_ids.shape[1])[None, :, None]
        return SimpleNamespace(logits=logits if apply_lm_head else None, last_hidden_state=logits)

    def compute_lm_logits(self, hidden_states):
        return hidden_states


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_orpo_scores_only_completion_labels(chunk_size):
    batch = {
        "chosen_input_ids": jnp.array([[0, 1, 3, 3]]),
        "chosen_attention_mask": jnp.ones((1, 4), dtype=jnp.int32),
        "chosen_labels": jnp.array([[-100, -100, 3, 3]]),
        "rejected_input_ids": jnp.array([[0, 1, 2, 0]]),
        "rejected_attention_mask": jnp.array([[1, 1, 1, 0]]),
        # Pretokenized sources need not pre-mask attention padding in labels.
        "rejected_labels": jnp.array([[-100, -100, 2, 0]]),
    }
    chosen, rejected, chosen_summary, rejected_summary, nll, accuracy = concatenated_forward(
        FixedLogits(chunk_size),
        batch,
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        padding_value=0,
    )
    log_z = np.log(np.exp([0.0, 1.0, 2.0, 3.0]).sum())
    np.testing.assert_allclose(chosen, [3.0 - log_z], rtol=1e-5)
    np.testing.assert_allclose(rejected, [2.0 - log_z], rtol=1e-5)
    np.testing.assert_allclose(chosen_summary, [66.0], rtol=1e-5)
    np.testing.assert_allclose(rejected_summary, [46.0], rtol=1e-5)
    np.testing.assert_allclose(nll, log_z - 3.0, rtol=1e-5)
    np.testing.assert_allclose(accuracy, 1.0)


def test_public_ppo_logps_drop_prompt_positions_without_causal_shift():
    logits = jnp.array([[[10.0, 0.0], [0.0, 2.0], [3.0, 0.0]]])
    result = compute_per_token_logps(logits, jnp.array([[0, 1, 0]]), prompt_length=1)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[-np.log1p(np.exp(-2.0)), -np.log1p(np.exp(-3.0))]], rtol=1e-5)


class TinyReward(spx.Module):
    mesh = None

    def __init__(self):
        self.scale = spx.Parameter(jnp.asarray(2.0))

    def __call__(self, input_ids, attention_mask):
        return {"logits": jnp.sum(input_ids * attention_mask, axis=1, keepdims=True) * self.scale.value}


class WeightedReward(RewardProtocol):
    def compute(self, **kwargs):
        return 4.0


@pytest.mark.parametrize("explicit_weights", [None, [0.5]])
def test_online_dpo_protocol_weights_affect_scored_rewards(monkeypatch, explicit_weights):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)
    config = OnlineDPOConfig(reward_weights=explicit_weights)
    trainer = OnlineDPOTrainer(config, model=None, reward_funcs=WeightedReward(weight=0.25))
    trainer.arguments = config
    rewards = trainer._score_online_dpo_rewards(
        prompts=["p"],
        completions=["a"],
        completion_ids=jnp.array([[1]]),
        batch={},
    )
    np.testing.assert_allclose(rewards, [2.0 if explicit_weights else 1.0])


def test_online_dpo_installs_forward_for_prebuilt_reward_state(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)
    reward_state = EasyDeLState.create(model=TinyReward())
    assert reward_state.apply_fn is None
    trainer = OnlineDPOTrainer(OnlineDPOConfig(), model=None, reward_funcs=reward_state)
    state = trainer.reward_funcs[0]
    output = state.apply_fn(
        state.graphdef,
        state.graphstate,
        state.graphother,
        {"input_ids": jnp.array([[1, 2]]), "attention_mask": jnp.array([[1, 1]]), "unused": jnp.array([0])},
    )
    np.testing.assert_allclose(output["logits"], [[6.0]])


@pytest.mark.parametrize("bound_graphdef", [False, True])
def test_online_dpo_preserves_custom_reward_forwards(monkeypatch, bound_graphdef):
    monkeypatch.setattr(DPOTrainer, "__init__", lambda self, **kwargs: None)
    state = EasyDeLState.create(model=TinyReward())

    def graphdef_forward(graphdef, graphstate, graphother, batch):
        return {"logits": batch["input_ids"] + 7}

    def bound_forward(graphstate, graphother, batch):
        return {"logits": batch["input_ids"] + 9}

    state = state.replace(apply_fn=bound_forward if bound_graphdef else graphdef_forward)
    trainer = OnlineDPOTrainer(OnlineDPOConfig(), model=None, reward_funcs=state)
    reward = trainer.reward_funcs[0]
    output = reward.apply_fn(reward.graphdef, reward.graphstate, reward.graphother, {"input_ids": jnp.array([[1]])})
    np.testing.assert_array_equal(output["logits"], [[10 if bound_graphdef else 8]])


def test_keep_arrays_explicit_drop_overrides_forced_keep():
    result = keep_arrays_map({"drop": [1, 2], "keep": 3}, array_fields=["drop", "keep"], drop_fields=["drop"])
    assert list(result) == ["keep"]
    np.testing.assert_array_equal(result["keep"], np.asarray(3))
