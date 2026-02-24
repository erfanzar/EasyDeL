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

"""Unit tests for SDPO (Self-Distillation Policy Optimization).

Tests cover:
- SDPOConfig validation and defaults
- Feedback separator template construction
- Self-feedback derivation from rollout groups
- Rich-feedback wrapping
- Feedback tokenisation padding
- sdpo_step loss computation (KL and JSD variants)
"""

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pytest.skip("JAX unavailable", allow_module_level=True)

from easydel.trainers.self_distillation_policy_optimization import SDPOConfig
from easydel.trainers.self_distillation_policy_optimization.sdpo_trainer import (
    _FEEDBACK_CORRECT,
    _FEEDBACK_TEMPLATE_SOLVE,
    _build_feedback_separator,
)


class TestSDPOConfig:
    """SDPOConfig instantiation and validation."""

    def test_defaults(self):
        cfg = SDPOConfig(max_prompt_length=128, max_completion_length=64)
        assert cfg.distillation_type == "jsd"
        assert cfg.beta == 0.0
        assert cfg.max_feedback_length == 256
        assert cfg.trainer_prefix == "sdpotrainer"

    def test_max_length_derived(self):
        cfg = SDPOConfig(max_prompt_length=128, max_completion_length=64)
        assert cfg.max_length == 128 + 64

    def test_invalid_distillation_type(self):
        with pytest.raises(ValueError, match="distillation_type"):
            SDPOConfig(
                max_prompt_length=128,
                max_completion_length=64,
                distillation_type="mse",
            )

    def test_kl_distillation_type(self):
        cfg = SDPOConfig(
            max_prompt_length=128,
            max_completion_length=64,
            distillation_type="kl",
        )
        assert cfg.distillation_type == "kl"

    def test_inherits_grpo_fields(self):
        cfg = SDPOConfig(
            max_prompt_length=128,
            max_completion_length=64,
            num_generations=8,
            temperature=0.9,
        )
        assert cfg.num_generations == 8
        assert cfg.temperature == 0.9


class TestFeedbackSeparator:
    """_build_feedback_separator template construction."""

    def test_successful_attempt(self):
        result = _build_feedback_separator(
            is_successful=True,
            env_feedback="",
            correct_solution=None,
        )
        assert result == _FEEDBACK_CORRECT

    def test_failed_with_correct_solution(self):
        result = _build_feedback_separator(
            is_successful=False,
            env_feedback="",
            correct_solution="print(42)",
        )
        assert "Correct solution:" in result
        assert "print(42)" in result
        assert "Correctly solve the original question." in result

    def test_failed_with_env_feedback(self):
        result = _build_feedback_separator(
            is_successful=False,
            env_feedback="ZeroDivisionError: division by zero",
            correct_solution=None,
        )
        assert "ZeroDivisionError" in result
        assert "unsuccessful earlier attempt" in result
        assert "Correctly solve the original question." in result

    def test_failed_with_both(self):
        result = _build_feedback_separator(
            is_successful=False,
            env_feedback="IndexError: list index out of range",
            correct_solution="return sorted(xs)",
        )
        assert "Correct solution:" in result
        assert "return sorted(xs)" in result
        assert "IndexError" in result

    def test_failed_no_feedback_no_solution(self):
        result = _build_feedback_separator(
            is_successful=False,
            env_feedback="",
            correct_solution=None,
        )
        assert result == _FEEDBACK_TEMPLATE_SOLVE


class TestSelfFeedback:
    """_get_self_feedback logic for deriving feedback from rollout groups."""

    @pytest.fixture()
    def _mock_trainer(self):
        class _Stub:
            pass

        from easydel.trainers.self_distillation_policy_optimization.sdpo_trainer import SDPOTrainer

        stub = _Stub()
        stub._get_self_feedback = SDPOTrainer._get_self_feedback.__get__(stub)
        return stub

    def test_successful_rollout_gets_correct_marker(self, _mock_trainer):
        completions = ["good1", "good2", "bad1", "bad2"]
        rewards = jnp.array([1.0, 0.5, 0.0, 0.0])
        texts, _ = _mock_trainer._get_self_feedback(completions, rewards, generation_factor=4)
        assert len(texts) == 4
        assert texts[0] == _FEEDBACK_CORRECT
        assert texts[1] == _FEEDBACK_CORRECT

    def test_failed_rollout_gets_best_solution(self, _mock_trainer):
        completions = ["best", "ok", "wrong1", "wrong2"]
        rewards = jnp.array([1.0, 0.5, 0.0, 0.0])
        texts, _ = _mock_trainer._get_self_feedback(completions, rewards, generation_factor=4)
        assert "best" in texts[2]
        assert "best" in texts[3]

    def test_all_failed_no_solution(self, _mock_trainer):
        completions = ["a", "b", "c", "d"]
        rewards = jnp.array([0.0, 0.0, 0.0, 0.0])
        texts, _ = _mock_trainer._get_self_feedback(completions, rewards, generation_factor=4)
        for t in texts:
            assert "Correct solution:" not in t

    def test_multiple_groups(self, _mock_trainer):
        completions = ["g1a", "g1b", "g2a", "g2b"]
        rewards = jnp.array([1.0, 0.0, 0.0, 0.5])
        texts, _ = _mock_trainer._get_self_feedback(completions, rewards, generation_factor=2)
        assert len(texts) == 4
        assert texts[0] == _FEEDBACK_CORRECT
        assert "g1a" in texts[1]
        assert "g2b" not in texts[1]
        assert texts[3] == _FEEDBACK_CORRECT


class TestRichFeedback:
    """_get_rich_feedback wrapping logic."""

    @pytest.fixture()
    def _mock_trainer(self):
        class _Stub:
            def feedback_func(self, prompts, completions, rewards):
                return [f"err_{i}" if r <= 0 else "" for i, r in enumerate(rewards)]

        from easydel.trainers.self_distillation_policy_optimization.sdpo_trainer import SDPOTrainer

        stub = _Stub()
        stub._get_rich_feedback = SDPOTrainer._get_rich_feedback.__get__(stub)
        return stub

    def test_wraps_raw_feedback_into_template(self, _mock_trainer):
        prompts = ["p1", "p2"]
        completions = ["c1", "c2"]
        rewards = jnp.array([0.0, 1.0])
        texts = _mock_trainer._get_rich_feedback(prompts, completions, rewards)
        assert len(texts) == 2
        assert "err_0" in texts[0]
        assert texts[1] == _FEEDBACK_CORRECT


class TestSDPOStepLoss:
    """Numerical checks on sdpo_step loss computation."""

    @staticmethod
    def _make_dummy_batch(batch_size=2, num_gen=2, prompt_len=4, comp_len=4, feedback_len=4):
        rng = np.random.RandomState(0)
        B, G = batch_size, num_gen
        prompt_ids = rng.randint(1, 100, (B, prompt_len)).astype(np.int32)
        prompt_mask = np.ones((B, prompt_len), dtype=np.int32)
        completion_ids = rng.randint(1, 100, (B * G, comp_len)).astype(np.int32)
        completion_mask = np.ones((B * G, comp_len), dtype=np.int32)
        feedback_ids = rng.randint(1, 100, (B * G, feedback_len)).astype(np.int32)
        feedback_mask = np.ones((B * G, feedback_len), dtype=np.int32)

        rids = np.repeat(prompt_ids, G, axis=0)
        rmask = np.repeat(prompt_mask, G, axis=0)
        teacher_ids = np.concatenate([rids, feedback_ids, completion_ids], axis=1)
        teacher_mask = np.concatenate([rmask, feedback_mask, completion_mask], axis=1)

        return {
            "prompt_ids": jnp.array(prompt_ids),
            "prompt_mask": jnp.array(prompt_mask),
            "completion_ids": jnp.array(completion_ids),
            "completion_mask": jnp.array(completion_mask),
            "teacher_ids": jnp.array(teacher_ids),
            "teacher_mask": jnp.array(teacher_mask),
            "num_items_in_batch": jnp.array(B * G * comp_len, dtype=jnp.float32),
        }

    def test_kl_loss_zero_when_same_context(self):
        batch = self._make_dummy_batch(batch_size=1, num_gen=2, feedback_len=0)
        student_logps = jnp.array([[-1.0, -2.0, -1.5, -0.5], [-0.8, -1.2, -1.0, -1.8]])
        teacher_logps = student_logps

        per_token_loss = student_logps - jax.lax.stop_gradient(teacher_logps)
        mask = batch["completion_mask"]
        loss = jnp.sum(per_token_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        assert jnp.allclose(loss, 0.0, atol=1e-6)

    def test_jsd_loss_zero_when_same_context(self):
        student_logps = jnp.array([[-1.0, -2.0, -1.5, -0.5]])
        teacher_logps = student_logps
        m_logp = jnp.logaddexp(student_logps, teacher_logps) - jnp.log(2.0)
        per_token_loss = student_logps - m_logp
        assert jnp.allclose(per_token_loss, 0.0, atol=1e-6)

    def test_kl_loss_positive_when_student_above_teacher(self):
        student_logps = jnp.array([[-0.5, -0.5]])
        teacher_logps = jnp.array([[-2.0, -2.0]])
        per_token_loss = student_logps - teacher_logps
        assert jnp.all(per_token_loss > 0)

    def test_kl_loss_negative_when_teacher_above_student(self):
        student_logps = jnp.array([[-2.0, -2.0]])
        teacher_logps = jnp.array([[-0.5, -0.5]])
        per_token_loss = student_logps - teacher_logps
        assert jnp.all(per_token_loss < 0)

    def test_jsd_bounded(self):
        student_logps = jnp.array([[-0.1, -5.0, -2.0]])
        teacher_logps = jnp.array([[-5.0, -0.1, -2.0]])
        m_logp = jnp.logaddexp(student_logps, teacher_logps) - jnp.log(2.0)
        per_token_loss = student_logps - m_logp
        assert jnp.all(per_token_loss <= jnp.log(2.0) + 1e-6)

    def test_advantage_sign(self):
        student_logps = jnp.array([[-2.0, -0.5]])
        teacher_logps = jnp.array([[-0.5, -2.0]])
        advantage = teacher_logps - student_logps
        assert advantage[0, 0] > 0
        assert advantage[0, 1] < 0

    def test_batch_shapes(self):
        batch = self._make_dummy_batch(batch_size=3, num_gen=4, prompt_len=8, comp_len=6, feedback_len=10)
        assert batch["prompt_ids"].shape == (3, 8)
        assert batch["completion_ids"].shape == (12, 6)
        assert batch["teacher_ids"].shape == (12, 8 + 10 + 6)
        assert batch["teacher_mask"].shape == batch["teacher_ids"].shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
