"""Accelerator-only GRPO reward-forward regression."""

from types import SimpleNamespace

import jax
import numpy as np
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.group_relative_policy_optimization import grpo_trainer
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec


class TinyReward(spx.Module):
    def __init__(self, mesh):
        self.mesh = mesh
        self.config = SimpleNamespace(pad_token_id=0)
        self.scale = spx.Parameter(jnp.asarray(2.0))

    def __call__(self, input_ids, attention_mask):
        return {"logits": jnp.sum(input_ids * attention_mask, axis=1, keepdims=True) * self.scale.value}


def test_grpo_installs_graphdef_first_forward_for_prebuilt_state(monkeypatch):
    monkeypatch.setattr(grpo_trainer.Trainer, "__init__", lambda self, **kwargs: None)
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    state = EasyDeLState.create(model=TinyReward(mesh))
    assert state.apply_fn is None
    config = grpo_trainer.GRPOConfig(
        beta=0.0,
        disable_dropout=False,
        log_completions=False,
        step_partition_spec=PartitionSpec(),
    )
    tokenizer = SimpleNamespace(pad_token_id=0)
    trainer = grpo_trainer.GRPOTrainer(
        config,
        model=state,
        reward_funcs=state,
        processing_class=tokenizer,
        reward_processing_classes=[tokenizer],
    )
    reward = trainer.reward_funcs[0]
    output = reward.apply_fn(
        reward.graphdef,
        reward.graphstate,
        reward.graphother,
        {"input_ids": jnp.array([[1, 2, 9]]), "attention_mask": jnp.array([[1, 1, 0]]), "unused": jnp.array([3])},
    )
    np.testing.assert_allclose(output["logits"], [[6.0]])
    assert state.apply_fn is None
