# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from easydel.trainers.reward_trainer._fn import _reward_pair_loss, _reward_scores_from_logits
from easydel.trainers.reward_trainer.reward_config import RewardConfig
from easydel.trainers.reward_trainer.reward_trainer import RewardTrainer


def test_reward_logits_are_scalar_scores_before_margin_loss():
    chosen_logits = jnp.array([[2.0], [0.5]], dtype=jnp.float32)
    rejected_logits = jnp.array([[0.0], [0.25]], dtype=jnp.float32)
    margin = jnp.array([1.0, -0.5], dtype=jnp.float32)

    chosen_rewards = _reward_scores_from_logits(chosen_logits)
    rejected_rewards = _reward_scores_from_logits(rejected_logits)
    actual = _reward_pair_loss(chosen_rewards, rejected_rewards, margin, center_rewards_coefficient=None)

    expected_margins = jnp.array([1.0, 0.75], dtype=jnp.float32)
    expected = -jnp.mean(jax.nn.log_sigmoid(expected_margins))

    assert chosen_rewards.shape == (2,)
    assert rejected_rewards.shape == (2,)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_binary_reward_logits_use_positive_log_odds():
    logits = jnp.array([[0.25, 1.25], [2.0, -0.5]], dtype=jnp.float32)

    rewards = _reward_scores_from_logits(logits)

    np.testing.assert_allclose(np.asarray(rewards), np.asarray(jnp.array([1.0, -2.5])), rtol=1e-6, atol=1e-6)


def test_reward_config_matches_trl_data_knobs_and_validation():
    cfg = RewardConfig(
        pad_to_multiple_of=8,
        eos_token="<eos>",
        chat_template_path="template.jinja",
    )

    assert cfg.learning_rate == 1e-4
    assert cfg.pad_to_multiple_of == 8
    assert cfg.eos_token == "<eos>"
    assert cfg.chat_template_path == "template.jinja"
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        RewardConfig(pad_to_multiple_of=0)
    with pytest.warns(FutureWarning, match="pad_token"):
        RewardConfig(pad_token="<pad>")
    assert RewardConfig(activation_offloading=True).activation_offloading is True


def test_reward_tokenizer_overrides_and_chat_template_path(tmp_path):
    class Tokenizer:
        eos_token = None
        pad_token = None
        chat_template = None

    tokenizer = Tokenizer()
    template = tmp_path / "reward_template.jinja"
    template.write_text("{{ messages[0]['content'] }}", encoding="utf-8")

    RewardTrainer._apply_tokenizer_overrides(tokenizer, eos_token="<eos>", pad_token="<pad>")
    RewardTrainer._apply_chat_template_path(tokenizer, str(template))

    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.chat_template == "{{ messages[0]['content'] }}"


def test_reward_disable_dropout_puts_model_in_eval_mode():
    class Model:
        def __init__(self):
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1

    state = SimpleNamespace(model=Model())

    RewardTrainer._disable_state_dropout(state)

    assert state.model.eval_calls == 1
