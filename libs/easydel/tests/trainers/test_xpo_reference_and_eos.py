"""XPO reference initialization and mask-aware preference regressions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.xpo_trainer.xpo_trainer import XPOTrainer
from jax import numpy as jnp


@dataclass
class _HostState:
    """Non-array state for exercising the reference-selection lifecycle on host."""

    graphstate: dict
    graphother: dict
    step: int = 0
    tx: object = None
    opt_state: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("beta, alpha", [(0.0, 1e-5), (0.0, 0.0), ([0.0, 0.1], [1e-5, 0.0])])
@pytest.mark.parametrize("explicit_reference", [False, True])
def test_xpo_reference_is_resolved_before_base_configuration(monkeypatch, beta, alpha, explicit_reference):
    policy = _HostState(graphstate={"weight": "policy"}, graphother={})
    reference = _HostState(graphstate={"weight": "explicit-reference"}, graphother={})

    def base_init(self, *, arguments, model, **kwargs):
        # Preserve GRPO's real reference-selection ordering without loading
        # models, allocating arrays, or compiling a train step.
        self.arguments = arguments
        self.ref_state = self._create_reference_state(model)
        self.reference_at_configuration = self.ref_state

    monkeypatch.setattr(GRPOTrainer, "__init__", base_init)
    reference_model = SimpleNamespace(to_state=lambda **kwargs: reference) if explicit_reference else None
    trainer = XPOTrainer(
        SimpleNamespace(beta=beta, alpha=alpha, loss_type="sigmoid", trainable_selector="parameters"),
        model=policy,
        reference_model=reference_model,
        reward_funcs=lambda **kwargs: [],
        processing_class=object(),
    )
    assert trainer.ref_state is trainer.reference_at_configuration
    assert trainer.ref_state.graphstate == {"weight": "explicit-reference" if explicit_reference else "policy"}
    assert trainer.ref_state is not policy
    if not explicit_reference:
        policy.graphstate["weight"] = "updated-policy"
        assert trainer.ref_state.graphstate == {"weight": "policy"}


@pytest.mark.parametrize("eos_token_id", [99, [98, 99]])
def test_xpo_padding_eos_does_not_suppress_penalty_or_change_preference(eos_token_id):
    trainer = object.__new__(XPOTrainer)
    trainer.arguments = SimpleNamespace(missing_eos_penalty=2.0)
    trainer._beta_schedule = 0.0
    trainer._alpha_schedule = 1e-5
    trainer.loss_type_id = 0
    trainer.ref_state = object()
    trainer.reward_funcs = []
    trainer.reward_processing_classes = None
    trainer.processing_class = SimpleNamespace(eos_token_id=eos_token_id)
    trainer._apply_user_data_collator = lambda batch: batch
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._log_training_generations_to_wandb = lambda **kwargs: None
    prompt_ids = jnp.array([[1], [2]], dtype=jnp.int32)
    prompt_mask = jnp.ones_like(prompt_ids)
    policy_ids = jnp.array([[7, 99], [7, 99]], dtype=jnp.int32)
    # Row 0 stops without EOS and gets EOS-valued padding. Row 1 emits a real EOS.
    policy_mask = jnp.array([[1, 0], [1, 1]], dtype=jnp.int32)
    ref_ids = jnp.array([[8, 99], [8, 99]], dtype=jnp.int32)
    ref_mask = jnp.ones_like(ref_ids)

    def rollout(**kwargs):
        ids, mask = (ref_ids, ref_mask) if kwargs["state"] is trainer.ref_state else (policy_ids, policy_mask)
        return SimpleNamespace(
            sequences=ids,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=ids,
            completion_mask=mask,
            text=None,
            raw_text=None,
            reasoning=None,
            tool_calls=None,
        )

    trainer.rollout = rollout
    scores = iter([jnp.array([1.0, 1.0]), jnp.array([0.0, 0.0])])
    trainer._score_rewards = lambda *args, **kwargs: (next(scores), {})
    processed, metrics = trainer._preprocess_batch_input(
        object(),
        {"input_ids": prompt_ids, "attention_mask": prompt_mask},
        is_train=True,
    )
    np.testing.assert_array_equal(np.asarray(processed["chosen_mask"]), [False, True])
    assert metrics["rewards/margins"] == 0.0
