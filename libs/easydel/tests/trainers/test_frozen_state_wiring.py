# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen-state wiring tests for the trainers that rebind after compilation.

XPO rebinds ``ref_state`` (an explicitly supplied reference, and the optional
``sync_ref_model`` refresh) and GKD self-distillation rebinds ``teacher_state``
*after* ``configure_functions`` has run. Both therefore have to resolve the state
handed to the compiled step per call: a tuple captured at compile time keeps
feeding the loss the original object while generation uses the current one.

The GKD case additionally pins the copy count -- self-distillation must borrow
the student state before the base initializer runs and take exactly one copy
afterwards.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import optax
import spectrax as spx
from easydel.trainers.generalized_knowledge_distillation_trainer import GKDConfig, GKDTrainer
from easydel.trainers.xpo_trainer import XPOTrainer
from jax import numpy as jnp


def _tiny_state(seed: int = 0, *, with_optimizer: bool = False):
    """Build a small llama state, optionally with initialized Adam slots."""
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=16,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(seed))
    state = model.to_state(trainable_selector="parameters")
    return state.init_tx(optax.adam(1e-3)) if with_optimizer else state


def _array_leaves(tree):
    """Return only the JAX array leaves of a pytree."""
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)]


def test_xpo_forwards_the_currently_bound_reference_state():
    trainer = XPOTrainer.__new__(XPOTrainer)
    initial, replacement = _tiny_state(0), _tiny_state(1)

    trainer.ref_state = initial
    assert trainer._train_shared_fn_extra_args == (initial,)
    assert trainer._eval_shared_fn_extra_args == (initial,)

    # A stale tuple left in the base trainer's backing field must not win: this is
    # exactly what compile-time capture used to leave behind.
    trainer._train_shared_fn_extra_args_ = (initial,)
    trainer._eval_shared_fn_extra_args_ = (initial,)
    trainer.ref_state = replacement

    assert trainer._train_shared_fn_extra_args == (replacement,)
    assert trainer._eval_shared_fn_extra_args == (replacement,)


def test_gkd_forwards_the_currently_bound_teacher_state():
    trainer = GKDTrainer.__new__(GKDTrainer)
    initial, replacement = _tiny_state(0), _tiny_state(1)

    trainer.teacher_state = initial
    assert trainer._train_shared_fn_extra_args == (initial,)

    trainer._train_shared_fn_extra_args_ = (initial,)
    trainer._eval_shared_fn_extra_args_ = (initial,)
    trainer.teacher_state = replacement

    assert trainer._train_shared_fn_extra_args == (replacement,)
    assert trainer._eval_shared_fn_extra_args == (replacement,)


def test_gkd_self_distillation_takes_exactly_one_teacher_copy(monkeypatch):
    from easydel.trainers.supervised_fine_tuning_trainer import SFTTrainer

    student_state = _tiny_state(0, with_optimizer=True)
    teacher_before_base_init = {}

    def fake_sft_init(self, **kwargs):
        """Stand in for the base initializer and record the borrowed teacher."""
        teacher_before_base_init["state"] = self.teacher_state
        self.model_state = kwargs["model"]
        self.arguments = kwargs["arguments"]
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(SFTTrainer, "__init__", fake_sft_init)

    trainer = GKDTrainer(
        arguments=GKDConfig(max_training_steps=1, disable_dropout=False),
        processing_class=None,
        model=student_state,
        teacher_model=None,
    )

    # Before the base initializer finalizes the student, the teacher borrows it
    # rather than copying: a pre-super copy used to stay resident for the whole
    # run alongside the real one.
    borrowed = teacher_before_base_init["state"]
    for student_leaf, borrowed_leaf in zip(
        _array_leaves(student_state.graphstate),
        _array_leaves(borrowed.graphstate),
        strict=True,
    ):
        assert borrowed_leaf is student_leaf

    # The final teacher is one params-only copy of the finalized student.
    assert trainer.teacher_state is not borrowed
    assert trainer.teacher_state.tx is None
    assert trainer.teacher_state.opt_state is None
    assert trainer.teacher_state.size < student_state.size
    for student_leaf, teacher_leaf in zip(
        _array_leaves(student_state.graphstate),
        _array_leaves(trainer.teacher_state.graphstate),
        strict=True,
    ):
        assert teacher_leaf is not student_leaf
        assert jnp.array_equal(teacher_leaf, student_leaf)


def test_gkd_supplied_teacher_state_loses_its_optimizer_slots(monkeypatch):
    from easydel.trainers.supervised_fine_tuning_trainer import SFTTrainer

    def fake_sft_init(self, **kwargs):
        """Stand in for the base initializer with the attributes GKD reads after it."""
        self.model_state = kwargs["model"]
        self.arguments = kwargs["arguments"]
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(SFTTrainer, "__init__", fake_sft_init)
    teacher_state = _tiny_state(1, with_optimizer=True)
    assert teacher_state.opt_state is not None

    trainer = GKDTrainer(
        arguments=GKDConfig(max_training_steps=1, disable_dropout=False),
        processing_class=None,
        model=_tiny_state(0),
        teacher_model=teacher_state,
    )

    assert trainer.teacher_state.opt_state is None
    assert trainer.teacher_state.tx is None
    # Parameters are shared, not copied: only the optimizer reference is dropped.
    for supplied, frozen in zip(
        _array_leaves(teacher_state.graphstate),
        _array_leaves(trainer.teacher_state.graphstate),
        strict=True,
    ):
        assert frozen is supplied


def test_dpo_reference_is_params_only_whether_derived_or_supplied(monkeypatch):
    from easydel.trainers.direct_preference_optimization_trainer import DPOConfig, DPOTrainer
    from easydel.trainers.trainer.trainer import Trainer

    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    policy = _tiny_state(0, with_optimizer=True)
    assert policy.opt_state is not None
    tokenizer = SimpleNamespace(pad_token_id=0)

    derived = DPOTrainer(
        arguments=DPOConfig(max_training_steps=1, disable_dropout=False),
        model=policy,
        processing_class=tokenizer,
    )
    # Derived reference: one parameter copy, no duplicated Adam slots.
    assert derived.reference_state.tx is None
    assert derived.reference_state.opt_state is None
    assert derived.reference_state.size < policy.size
    for policy_leaf, reference_leaf in zip(
        _array_leaves(policy.graphstate),
        _array_leaves(derived.reference_state.graphstate),
        strict=True,
    ):
        assert reference_leaf is not policy_leaf
        assert jnp.array_equal(reference_leaf, policy_leaf)

    supplied_reference = _tiny_state(1, with_optimizer=True)
    supplied = DPOTrainer(
        arguments=DPOConfig(max_training_steps=1, disable_dropout=False),
        model=_tiny_state(0),
        reference_model=supplied_reference,
        processing_class=tokenizer,
    )
    # Supplied reference: slots dropped, parameters shared rather than copied.
    assert supplied.reference_state.opt_state is None
    for supplied_leaf, frozen_leaf in zip(
        _array_leaves(supplied_reference.graphstate),
        _array_leaves(supplied.reference_state.graphstate),
        strict=True,
    ):
        assert frozen_leaf is supplied_leaf


def test_dpo_reference_drop_applies_with_dropout_disabled(monkeypatch):
    from easydel.trainers.direct_preference_optimization_trainer import DPOConfig, DPOTrainer
    from easydel.trainers.trainer.trainer import Trainer

    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    reference = _tiny_state(1, with_optimizer=True)

    trainer = DPOTrainer(
        arguments=DPOConfig(max_training_steps=1, disable_dropout=True),
        model=_tiny_state(0),
        reference_model=reference,
        processing_class=SimpleNamespace(pad_token_id=0),
    )

    assert trainer.reference_state.opt_state is None
    assert trainer.reference_state.tx is None
