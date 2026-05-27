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

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    pytest.skip("JAX unavailable", allow_module_level=True)

from easydel.trainers.self_distillation_policy_optimization import SDPOConfig
from easydel.trainers.self_distillation_policy_optimization import _fn as sdpo_fn
from easydel.trainers.self_distillation_policy_optimization.sdpo_trainer import (
    _FEEDBACK_CORRECT,
    _FEEDBACK_TEMPLATE_SOLVE,
    SDPOTrainer,
    _build_feedback_separator,
)


class TestSDPOConfig:
    """SDPOConfig instantiation and validation."""

    def test_defaults(self):
        cfg = SDPOConfig(max_prompt_length=128, max_completion_length=64)
        assert cfg.distillation_type == "jsd"
        assert cfg.beta == 0.0
        assert cfg.max_feedback_length == 256
        assert cfg.trainer_prefix == "SDPO"
        assert cfg.distillation_weight == 1.0
        assert cfg.use_successful_as_teacher is True
        assert cfg.include_environment_feedback is True
        assert cfg.teacher_regularization == "none"

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

    def test_distillation_alpha_aliases_supported_token_losses(self):
        assert (
            SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_alpha=0.5).distillation_type
            == "jsd"
        )
        assert (
            SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_alpha=1.0).distillation_type == "kl"
        )

    def test_unsupported_sdpo_modes_fail_loudly(self):
        with pytest.raises(ValueError, match="distillation_alpha"):
            SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_alpha=0.0)
        cfg = SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_topk=10)
        assert cfg.full_logit_distillation is True
        assert cfg.distillation_topk == 10
        assert SDPOConfig(max_prompt_length=128, max_completion_length=64, full_logit_distillation=True)
        assert SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_is_clip=2.0)
        with pytest.raises(ValueError, match="distillation_topk"):
            SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_topk=0)
        with pytest.raises(ValueError, match="distillation_add_tail"):
            SDPOConfig(max_prompt_length=128, max_completion_length=64, distillation_add_tail=True)
        with pytest.raises(ValueError, match="sdpo_policy_loss_mode"):
            SDPOConfig(max_prompt_length=128, max_completion_length=64, sdpo_policy_loss_mode="hybrid")
        ema_cfg = SDPOConfig(max_prompt_length=128, max_completion_length=64, teacher_regularization="ema")
        assert ema_cfg.sync_ref_model is True
        assert ema_cfg.ref_model_mixup_alpha == ema_cfg.teacher_update_rate


def test_completion_loss_token_skip_masks_prefix_tokens():
    completion_mask = jnp.asarray([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=jnp.int32)

    skipped = sdpo_fn._apply_completion_loss_token_skip(completion_mask, 2)

    assert skipped.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]
    assert sdpo_fn._apply_completion_loss_token_skip(completion_mask, 0).tolist() == completion_mask.tolist()


def test_full_vocab_sdpo_topk_tail_matches_bucketed_kl():
    student_logits = jnp.asarray([[[0.2, 0.8, -0.4, 0.1]]], dtype=jnp.float32)
    teacher_logits = jnp.asarray([[[1.0, 0.3, 0.2, -0.5]]], dtype=jnp.float32)
    completion_ids = jnp.asarray([[1]], dtype=jnp.int32)

    per_token, student_logps, teacher_logps = sdpo_fn._full_vocab_sdpo_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        completion_ids=completion_ids,
        distillation_type="kl",
        distillation_topk=2,
        distillation_add_tail=True,
        distillation_clip=None,
    )

    teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
    top_teacher_log_probs, top_indices = jax.lax.top_k(teacher_log_probs, 2)
    top_student_log_probs = jnp.take_along_axis(student_log_probs, top_indices, axis=-1)
    top_teacher_probs = jnp.exp(top_teacher_log_probs)
    top_student_probs = jnp.exp(top_student_log_probs)
    teacher_tail = 1.0 - jnp.sum(top_teacher_probs, axis=-1)
    student_tail = 1.0 - jnp.sum(top_student_probs, axis=-1)
    expected = jnp.sum(top_teacher_probs * (top_teacher_log_probs - top_student_log_probs), axis=-1) + teacher_tail * (
        jnp.log(teacher_tail) - jnp.log(student_tail)
    )

    assert jnp.allclose(per_token, expected, atol=1e-6)
    assert jnp.allclose(student_logps, jnp.take_along_axis(student_log_probs, completion_ids[..., None], -1).squeeze(-1))
    assert jnp.allclose(teacher_logps, jnp.take_along_axis(teacher_log_probs, completion_ids[..., None], -1).squeeze(-1))


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
        stub.arguments = SimpleNamespace()
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

    def test_threshold_and_self_success_skip_are_respected(self, _mock_trainer):
        _mock_trainer.arguments = SimpleNamespace(
            success_reward_threshold=0.75,
            use_successful_as_teacher=True,
            dont_reprompt_on_self_success=True,
        )
        completions = ["best", "weak", "bad"]
        rewards = jnp.array([1.0, 0.5, 0.0])
        texts, _ = _mock_trainer._get_self_feedback(completions, rewards, generation_factor=3)
        assert texts[0] == ""
        assert "best" in texts[1]
        assert "best" in texts[2]

    def test_custom_solution_template_and_thinking_strip(self, _mock_trainer):
        _mock_trainer.arguments = SimpleNamespace(
            solution_template="S:{successful_previous_attempt}|",
            reprompt_template="{solution}GO",
            remove_thinking_from_demonstration=True,
        )
        completions = ["<think>hidden</think>answer", "wrong"]
        rewards = jnp.array([1.0, 0.0])
        texts, _ = _mock_trainer._get_self_feedback(completions, rewards, generation_factor=2)
        assert texts[1] == "S:answer|GO"


class TestRichFeedback:
    """_get_rich_feedback wrapping logic."""

    @pytest.fixture()
    def _mock_trainer(self):
        class _Stub:
            def feedback_func(self, prompts, completions, rewards):
                return [f"err_{i}" if r <= 0 else "" for i, r in enumerate(rewards)]

        from easydel.trainers.self_distillation_policy_optimization.sdpo_trainer import SDPOTrainer

        stub = _Stub()
        stub.arguments = SimpleNamespace()
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

    def test_rich_feedback_can_be_disabled(self, _mock_trainer):
        _mock_trainer.arguments = SimpleNamespace(include_environment_feedback=False)
        prompts = ["p1"]
        completions = ["c1"]
        rewards = jnp.array([0.0])
        texts = _mock_trainer._get_rich_feedback(prompts, completions, rewards)
        assert "err_0" not in texts[0]
        assert texts[0] == _FEEDBACK_TEMPLATE_SOLVE


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

    def test_max_loss_completion_tokens_truncates_scoring_inputs(self, monkeypatch):
        captured_shapes = []

        class _DummyState:
            graphstate = object()

            def merge(self, tree):
                del tree
                return object()

        def _fake_get_per_token_logps(
            module,
            input_ids,
            attention_mask,
            prompt_length,
            model_kwargs=None,
            logprob_vocab_chunk_size=None,
        ):
            del module, attention_mask, model_kwargs, logprob_vocab_chunk_size
            captured_shapes.append((tuple(input_ids.shape), int(prompt_length)))
            return jnp.zeros((input_ids.shape[0], input_ids.shape[1] - prompt_length), dtype=jnp.float32)

        monkeypatch.setattr(sdpo_fn, "get_per_token_logps", _fake_get_per_token_logps)

        batch = self._make_dummy_batch(batch_size=1, num_gen=2, prompt_len=4, comp_len=6, feedback_len=3)
        metrics = sdpo_fn.sdpo_step(
            state=_DummyState(),
            batch=batch,
            num_generations=2,
            teacher_prompt_length=7,
            beta=0.0,
            distillation_type="jsd",
            distillation_weight=1.0,
            logprob_vocab_chunk_size=8,
            max_loss_completion_tokens=4,
            completion_chunk_size=None,
            loss_config=None,
            learning_rate_fn=None,
            partition_spec=None,
            gradient_accumulation_steps=1,
            is_training=False,
            straight_through_emulator=None,
        )

        assert captured_shapes == [((2, 8), 4), ((2, 11), 7)]
        assert jnp.allclose(metrics.loss, 0.0)

    def test_completion_chunk_size_splits_scoring_batch(self, monkeypatch):
        captured_shapes = []

        class _DummyState:
            graphstate = object()

            def merge(self, tree):
                del tree
                return object()

        def _fake_get_per_token_logps(
            module,
            input_ids,
            attention_mask,
            prompt_length,
            model_kwargs=None,
            logprob_vocab_chunk_size=None,
        ):
            del module, attention_mask, model_kwargs, logprob_vocab_chunk_size
            captured_shapes.append((tuple(input_ids.shape), int(prompt_length)))
            return jnp.zeros((input_ids.shape[0], input_ids.shape[1] - prompt_length), dtype=jnp.float32)

        monkeypatch.setattr(sdpo_fn, "get_per_token_logps", _fake_get_per_token_logps)

        batch = self._make_dummy_batch(batch_size=1, num_gen=2, prompt_len=4, comp_len=6, feedback_len=3)
        metrics = sdpo_fn.sdpo_step(
            state=_DummyState(),
            batch=batch,
            num_generations=2,
            teacher_prompt_length=7,
            beta=0.0,
            distillation_type="jsd",
            distillation_weight=1.0,
            logprob_vocab_chunk_size=8,
            max_loss_completion_tokens=4,
            completion_chunk_size=1,
            loss_config=None,
            learning_rate_fn=None,
            partition_spec=None,
            gradient_accumulation_steps=1,
            is_training=False,
            straight_through_emulator=None,
        )

        assert captured_shapes == [((1, 8), 4), ((1, 11), 7), ((1, 8), 4), ((1, 11), 7)]
        assert jnp.allclose(metrics.loss, 0.0)


class TestSDPOPromptGuard:
    """Tests for empty-prompt guard in SDPO preprocessing."""

    @pytest.fixture()
    def _mock_trainer(self):
        class _Stub:
            _eos_token_id = [2]  # noqa: RUF012
            _pad_token_id = 0

        stub = _Stub()
        stub._ensure_non_empty_prompts = SDPOTrainer._ensure_non_empty_prompts.__get__(stub)
        return stub

    def test_inserts_fallback_token_for_empty_rows(self, _mock_trainer):
        prompt_ids = jnp.array(
            [
                [0, 0, 0, 0],
                [10, 11, 0, 0],
            ],
            dtype=jnp.int32,
        )
        prompt_mask = jnp.array(
            [
                [0, 0, 0, 0],
                [1, 1, 0, 0],
            ],
            dtype=jnp.int32,
        )

        new_ids, new_mask, fixed = _mock_trainer._ensure_non_empty_prompts(prompt_ids, prompt_mask)

        assert fixed == 1
        assert int(jnp.sum(new_mask[0])) == 1
        assert int(new_ids[0, -1]) == 2
        assert int(jnp.sum(new_mask[1])) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
