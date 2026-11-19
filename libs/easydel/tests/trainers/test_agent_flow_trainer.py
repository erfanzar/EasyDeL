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
"""Tests for the AgentFlowTrainer multi-turn composition wrapper and AsyncGRPO environment rewards.

Tests marked ``cpu_ok`` do no array computation; the rest build JAX arrays and run on an accelerator.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.environments._base import AgenticEnvironment, ResetResult, StepResult
from easydel.infra.elarge.types import normalize_trainer_config
from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
from easydel.trainers.dppo_trainer import DPPOTrainer
from easydel.trainers.gdpo_trainer import GDPOConfig, GDPOTrainer
from easydel.trainers.generalized_knowledge_distillation_trainer import GKDTrainer
from easydel.trainers.gmpo_trainer import GMPOConfig, GMPOTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOConfig, GRPOTrainer
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step
from easydel.trainers.proximal_policy_optimization_trainer import PPOTrainer
from easydel.trainers.reinforce_plus_plus_trainer import ReinforcePlusPlusTrainer
from easydel.trainers.vapo_trainer import VAPOTrainer
from easydel.utils.registery import Registry
from jax import numpy as jnp


class _DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs):
        ids = [ord(c) for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) > 0)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + (
            "<assistant>" if add_generation_prompt else ""
        )


class _MultiTurnCountingEnv(AgenticEnvironment):
    def __init__(
        self,
        prompt: str = "start",
        target_turns: int = 2,
        step_reward: float = 0.5,
        fail_step: bool = False,
        raise_step: bool = False,
    ):
        self._prompt = prompt
        self._target_turns = target_turns
        self._step_reward = step_reward
        self._fail_step = fail_step
        self._raise_step = raise_step
        self._turn = 0

    def reset(self, seed: int | None = None) -> ResetResult:
        self._turn = 0
        return ResetResult(observation=self._prompt, info={})

    def step(self, action: str) -> StepResult:
        self._turn += 1
        if self._raise_step:
            raise RuntimeError("sandbox crashed")
        if self._fail_step:
            return StepResult(
                observation="infra fail",
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"env_error": True},
            )
        done = self._turn >= self._target_turns or "STOP" in action
        obs = "" if done else f"obs{self._turn}"
        return StepResult(
            observation=obs,
            reward=self._step_reward,
            terminated=done,
            truncated=False,
            info={"turn": self._turn},
        )


def _stub_trainer(cls, args=None):
    inst = object.__new__(cls)
    inst.arguments = args or SimpleNamespace(
        num_generations=2,
        scale_rewards="group",
        async_mode=False,
        max_prompt_length=24,
        max_completion_length=48,
        beta=0.04,
    )
    inst.processing_class = _DummyTokenizer()
    inst._preprocess_batch_input = lambda state, batch, is_train: ({}, {})
    return inst


@pytest.mark.cpu_ok
def test_agent_flow_registry_and_config_validation():
    assert Registry.get_or_raise("trainer", "agent_flow") is AgentFlowTrainer
    assert Registry.get_or_raise("trainer-arguments", "agent_flow") is AgentFlowConfig
    assert AgentFlowConfig().environment_backend == "local"
    assert normalize_trainer_config({"trainer_type": "agent-flow"})["trainer_type"] == "agent_flow"
    assert normalize_trainer_config({"trainer_type": "reinforce++"})["trainer_type"] == "reinforce_plus_plus"

    with pytest.raises(ValueError, match="max_turns"):
        AgentFlowConfig(max_turns=0)
    with pytest.raises(ValueError, match="context_mode"):
        AgentFlowConfig(context_mode="bad")
    with pytest.raises(ValueError, match="reward_mode"):
        AgentFlowConfig(reward_mode="bad")
    with pytest.raises(ValueError, match="recursive"):
        AgentFlowConfig(base_trainer_type="agent_flow")


@pytest.mark.parametrize("trainer_cls", [GRPOTrainer, GMPOTrainer, GDPOTrainer, ReinforcePlusPlusTrainer, DPPOTrainer])
def test_agent_flow_wraps_grpo_family_with_tito_ids_and_masks(trainer_cls):
    args = (
        GMPOConfig(num_generations=2, max_prompt_length=24, max_completion_length=48)
        if trainer_cls is GMPOTrainer
        else GDPOConfig(num_generations=2, max_prompt_length=24, max_completion_length=48)
        if trainer_cls is GDPOTrainer
        else GRPOConfig(num_generations=2, max_prompt_length=24, max_completion_length=48)
    )
    inner = _stub_trainer(trainer_cls, args=args)

    seen_states: list[object] = []
    seen_prompt_ids: list[list[int]] = []

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        seen_states.append(state)
        seen_prompt_ids.append(list(prompt_ids))
        text = f"a{turn_idx}" if turn_idx == 0 else "STOP"
        toks = [ord(c) for c in text]
        lps = [-0.1 * (i + 1) for i in range(len(toks))]
        return toks, text, lps

    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=4,
            reward_mode="sum",
            group_size=2,
            max_prompt_length=24,
            max_completion_length=48,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(
            prompt=kw.get("prompt", "hello"),
            target_turns=2,
            step_reward=0.5,
        ),
        generate_turn_fn=gen_turn,
    )

    sentinel_state = SimpleNamespace(step=7)
    out, metrics = inner._preprocess_batch_input(sentinel_state, {"prompts": ["hello"]}, True)

    assert all(s is sentinel_state for s in seen_states)
    assert out["prompt_ids"].shape == (2, 24)
    assert out["completion_ids"].shape == (2, 48)
    assert out["completion_mask"].shape == (2, 48)
    assert out["completion_attention_mask"].shape == (2, 48)
    assert "ref_per_token_logps" in out
    assert out["old_per_token_logps"].shape == (2, 48)

    # TITO check: turn 1 prompt_ids == turn 0 prompt_ids + "a0" + the template delta "<user>obs1<assistant>".
    obs_wrapped = "<user>obs1<assistant>"
    expected_turn1_prefix = [ord(c) for c in "<user>hello<assistant>a0" + obs_wrapped]
    assert expected_turn1_prefix in seen_prompt_ids

    expected_len = 2 + len(obs_wrapped) + 4
    act_row = np.asarray(out["completion_mask"][0]).tolist()
    attn_row = np.asarray(out["completion_attention_mask"][0]).tolist()
    assert act_row[:expected_len] == [1, 1] + [0] * len(obs_wrapped) + [1, 1, 1, 1]
    assert attn_row[:expected_len] == [1] * expected_len
    assert attn_row[expected_len:] == [0] * (48 - expected_len)
    assert np.allclose(np.asarray(out["environment_rewards"]), [1.0, 1.0])
    assert metrics["agent_flow/mean_turns"] == 2.0
    assert flow.trainer is inner
    if trainer_cls is GMPOTrainer:
        assert flow.arguments.base_trainer_type == "gmpo"
    elif trainer_cls is GDPOTrainer:
        assert flow.arguments.base_trainer_type == "gdpo"
    elif trainer_cls is ReinforcePlusPlusTrainer:
        assert flow.arguments.base_trainer_type == "reinforce_plus_plus"


@pytest.mark.parametrize("trainer_cls", [PPOTrainer, VAPOTrainer])
def test_agent_flow_adapts_ppo_and_vapo_with_nonzero_advantages(trainer_cls):
    inner = _stub_trainer(trainer_cls)
    inner.arguments.positive_reward_threshold = 0.0
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            reward_mode="discounted_sum",
            gamma=0.5,
            group_size=1,
            max_prompt_length=24,
            max_completion_length=32,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(target_turns=2, step_reward=1.0),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("x")], "x", [-0.2]),
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=1), {"prompts": ["hi"]}, True)
    assert {
        "input_ids",
        "attention_mask",
        "completion_mask",
        "rewards",
        "returns",
        "advantages",
        "old_logps",
        "positive_mask",
    }.issubset(out.keys())
    assert out["input_ids"].shape == (1, 72)
    assert out["completion_mask"].shape == (1, 48)
    assert np.allclose(np.asarray(out["rewards"]), [1.5])
    # Group size 1 must preserve non-zero advantage instead of centering to 0.
    assert np.allclose(np.asarray(out["advantages"]), 1.5)
    assert np.allclose(np.asarray(out["positive_mask"]), [1.0])
    assert flow.trainer is inner


def test_agent_flow_env_error_and_exception_mask_gradients():
    # Partial failure (1 good + 1 env_error in group_size=2): masks failed row, keeps good row.
    for env_kw in ({"fail_step": True}, {"raise_step": True}):
        inner = _stub_trainer(GRPOTrainer)
        inner.rollout = lambda **kw: SimpleNamespace(
            completion_ids=np.asarray([[ord("S"), ord("T"), ord("O"), ord("P")]]),
            completion_mask=np.asarray([[1, 1, 1, 1]]),
            sampling_per_token_logps=np.asarray([[-0.1, -0.2, -0.1, -0.2]]),
        )
        call_n = {"n": 0}

        def _make_env(_kw=env_kw, _cn=call_n, **kw):
            _cn["n"] += 1
            return _MultiTurnCountingEnv(**_kw) if _cn["n"] == 1 else _MultiTurnCountingEnv(target_turns=1)

        AgentFlowTrainer(
            arguments=AgentFlowConfig(
                max_turns=2,
                group_size=2,
                max_prompt_length=16,
                max_completion_length=16,
                skip_apply_chat_template=True,
            ),
            trainer=inner,
            environment_factory=_make_env,
        )
        out, metrics = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
        assert int(np.sum(out["completion_mask"][0])) == 0
        assert int(np.sum(out["completion_mask"][1])) > 0
        assert metrics["agent_flow/env_errors"] == 1.0

    # All-error batch must raise RuntimeError rather than training on all-zero masks.
    inner_all_err = _stub_trainer(GRPOTrainer)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=1,
            max_prompt_length=16,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=inner_all_err,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(fail_step=True),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("x")], "x", [-0.1]),
    )
    with pytest.raises(RuntimeError, match="All 1 AgentFlow episodes failed"):
        inner_all_err._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)

    # Constructor errors in env_factory must raise loudly (never swallowed).
    inner_bad = _stub_trainer(GRPOTrainer)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, skip_apply_chat_template=True),
        trainer=inner_bad,
        environment_factory=lambda **kw: (_ for _ in ()).throw(TypeError("bad ctor arg")),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("x")], "x", [-0.1]),
    )
    with pytest.raises(TypeError, match="bad ctor arg"):
        inner_bad._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)


def test_agent_flow_real_humaneval_rows_through_transform_to_local_code_judge():
    """Verify env.py-shaped rows (with task_id + test_code) survive transform + GRPODataCollatorGrain."""
    from easydel.environments import ContainerConfig
    from easydel.trainers.prompt_transforms import GRPOPreprocessTransform
    from easydel.trainers.utils import GRPODataCollatorGrain

    tok = _DummyTokenizer()
    transform = GRPOPreprocessTransform(tokenizer=tok, max_prompt_length=64, skip_apply_chat_template=True)
    row = transform(
        {
            "task_id": "HumanEval/0",
            "prompt": "Write add(a, b).",
            "test_code": "assert add(2, 3) == 5\n",
        }
    )
    collator = GRPODataCollatorGrain(max_prompt_length=64, pad_token_id=0)
    collated = collator(row)
    batch = {k: (v[None, ...] if isinstance(v, np.ndarray) else [v]) for k, v in collated.items()}

    inner = _stub_trainer(GRPOTrainer)
    attempt = {"i": 0}

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        attempt["i"] += 1
        code = (
            "```python\ndef add(a, b):\n    return a + b\n```"
            if attempt["i"] == 1
            else "```python\ndef add(a, b):\n    return 0\n```"
        )
        return [ord(c) for c in code], code, [-0.1] * len(code)

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=2,
            max_workers=1,
            max_prompt_length=64,
            max_completion_length=128,
            skip_apply_chat_template=True,
            environment_backend="local",
        ),
        trainer=inner,
        environment={
            "name": "code_judge",
            "max_steps": 1,
            "backend": "local",
            "container_config": ContainerConfig(use_jail="off"),
        },
        generate_turn_fn=gen_turn,
    )
    out, metrics = inner._preprocess_batch_input(SimpleNamespace(step=0), batch, True)
    assert np.allclose(np.asarray(out["environment_rewards"]), [1.0, 0.0])
    assert metrics["agent_flow/clipped_episodes"] == 0.0


def test_agent_flow_icepop_masks_out_of_band_ratios_and_handles_reset_error_and_obs_cap():
    # 1) IcePop masking under AgentFlow when policy logps differ from eSurge logps.
    cfg = GRPOConfig(
        num_generations=2,
        max_prompt_length=8,
        max_completion_length=16,
        esurge_importance_sampling_correction=True,
        esurge_importance_sampling_mode="token_icepop",
        esurge_importance_sampling_cap=2.0,
        esurge_importance_sampling_cap_low=0.5,
    )
    inner = _stub_trainer(GRPOTrainer, args=cfg)
    # Token 0: diff = 0.0 -> exp(0) = 1.0 (kept); Token 1: diff = +2.0 -> exp(2) ~ 7.39 > 2.0 (masked to 0.0);
    # Token 2: diff = -2.0 -> exp(-2) ~ 0.135 < 0.5 (masked to 0.0).
    inner.compute_state_logps = lambda *a, **kw: (
        jnp.zeros((2, 16), dtype=jnp.float32).at[:, :3].set(jnp.asarray([-0.1, 1.9, -2.1], dtype=jnp.float32))
    )
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=2,
            max_prompt_length=8,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(target_turns=1, step_reward=1.0),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("a"), ord("b"), ord("c")], "abc", [-0.1, -0.1, -0.1]),
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
    ratio = np.asarray(out["importance_sampling_ratio"][0, :3])
    assert np.isclose(ratio[0], 1.0, atol=1e-4)
    assert ratio[1] == 0.0
    assert ratio[2] == 0.0

    # 2) Reset-time env_error preserves old_per_token_logps for the batch instead of dropping it.
    class _ResetFailThenPassEnv(AgenticEnvironment):
        def __init__(self, fail: bool):
            self._fail = fail

        def reset(self, seed: int | None = None) -> ResetResult:
            return ResetResult(observation="p", info={"env_error": self._fail})

        def step(self, action: str) -> StepResult:
            return StepResult(observation="", reward=1.0, terminated=True, truncated=False, info={})

    idx = {"n": 0}

    def _make_reset_env(**kw):
        idx["n"] += 1
        return _ResetFailThenPassEnv(fail=(idx["n"] == 1))

    inner_reset = _stub_trainer(GRPOTrainer)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=2,
            max_workers=1,
            max_prompt_length=8,
            max_completion_length=16,
            skip_apply_chat_template=True,
        ),
        trainer=inner_reset,
        environment_factory=_make_reset_env,
        generate_turn_fn=lambda st, ids, txt, t: ([ord("x")], "x", [-0.2]),
    )
    out_reset, metrics_reset = inner_reset._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
    assert "old_per_token_logps" in out_reset
    assert metrics_reset["agent_flow/env_errors"] == 1.0

    # 3) Observation cap: 3 turns * 4 tokens + 2 observations fit in max_completion_length=20 without clipping!
    class _HugeObsEnv(AgenticEnvironment):
        def __init__(self):
            self._t = 0

        def reset(self, seed: int | None = None) -> ResetResult:
            self._t = 0
            return ResetResult(observation="P", info={})

        def step(self, action: str) -> StepResult:
            self._t += 1
            done = self._t >= 3
            return StepResult(observation=("X" * 200) if not done else "", reward=1.0, terminated=done, truncated=False)

    inner_cap = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=1, max_prompt_length=8, max_completion_length=20),
    )
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=3,
            per_turn_max_tokens=4,
            group_size=1,
            max_prompt_length=8,
            max_completion_length=20,
            skip_apply_chat_template=True,
        ),
        trainer=inner_cap,
        environment_factory=lambda **kw: _HugeObsEnv(),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("A")] * 4, "AAAA", [-0.1] * 4),
    )
    out_cap, metrics_cap = inner_cap._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)
    assert metrics_cap["agent_flow/clipped_episodes"] == 0.0
    # 3 turns * 4 action tokens = 12 unmasked action tokens (no final try cut off!).
    assert int(np.sum(out_cap["completion_mask"])) == 12


def test_agent_flow_chat_template_wraps_observation_as_user_turn_and_assistant_header():
    class _ChatTok(_DummyTokenizer):
        eos_token = "<E>"
        eos_token_id = 3

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = [f"<U>{m['content']}<E>" for m in messages]
            if add_generation_prompt:
                parts.append("<A>")
            return "".join(parts)

    inner = _stub_trainer(
        GRPOTrainer,
        args=GRPOConfig(num_generations=1, max_prompt_length=32, max_completion_length=64),
    )
    inner.processing_class = _ChatTok()

    class _TwoTurnEnv(AgenticEnvironment):
        def __init__(self):
            self._t = 0

        def reset(self, seed: int | None = None) -> ResetResult:
            self._t = 0
            return ResetResult(observation="P", info={})

        def step(self, action: str) -> StepResult:
            self._t += 1
            return StepResult(observation="OBS" if self._t == 1 else "", reward=1.0, terminated=self._t >= 2)

    turns = {"n": 0}

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        turns["n"] += 1
        if turn_idx == 1:
            assert "<U>OBS<E><A>" in prompt_text
        return [ord("a"), 3], "a", [-0.1, -0.1]

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            group_size=1,
            max_prompt_length=32,
            max_completion_length=64,
            skip_apply_chat_template=False,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _TwoTurnEnv(),
        generate_turn_fn=gen_turn,
    )
    out, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)
    # Turn 1 (2 action tokens) + Obs chat wrapper (action_mask=0) + Turn 2 (2 action tokens) => 4 active tokens.
    assert int(np.sum(out["completion_mask"])) == 4


def test_agent_flow_async_mode_train_executes_wrapped_env_factory():
    """Verify train() under async_mode=True routes rollouts through AgentFlowTrainer's wrapper."""
    from easydel.trainers.async_rollout_mixin import AsyncRolloutMixin

    env_calls: list[str] = []

    class _MinimalAsyncTrainer(AsyncRolloutMixin):
        supports_async_rollout = True

        def __init__(self):
            self.arguments = SimpleNamespace(
                async_mode=True,
                max_inflight_tasks=1,
                queue_maxsize=1,
                max_staleness=1,
                weight_sync_steps=1,
                request_timeout=5.0,
                first_rollout_timeout=5.0,
                heartbeat_stale_after_s=5.0,
                learning_rate=1e-4,
                max_length=32,
                max_prompt_length=16,
                max_completion_length=16,
                num_generations=1,
                scale_rewards="group",
                merge_lora_before_save=False,
            )
            self.processing_class = _DummyTokenizer()
            self.data_collator = None
            self.max_training_steps = 1
            self._async_grpo_train_compiled_once = False
            self.scheduler = None
            self._backward_flops_per_token = 0.0
            self._extra_backward_flops_per_token = 0.0
            self.training_batch_size = 1
            self.pruning_module = None
            import contextlib

            self.train_tracker = SimpleNamespace(trace_compilation=lambda: contextlib.nullcontext())

        def _preprocess_batch_input(self, state, batch, is_train):
            raise AssertionError("unwrapped _preprocess_batch_input should never be called")

        def _get_next_batch(self, iterator, dataset):
            return next(iterator), iterator

        def _take_rollout_snapshot(self, state, *, policy_step, **kwargs):
            return SimpleNamespace(
                state=SimpleNamespace(step=policy_step),
                policy_step=policy_step,
                nbytes=0,
                release=lambda: None,
            )

        def on_step_start(self, *, state, step):
            return state

        def _execute_preprocessed_train_step(self, **kwargs):
            from easydel.infra.loss_utils import LossMetrics

            return SimpleNamespace(step=1, esurge_cache_scope_key="t"), LossMetrics(loss=0.1, accuracy=0.5), None

        def _maybe_start_profiler(self, step):
            return None

        def _maybe_stop_profiler(self, step):
            return None

        def apply_training_hooks(self, *, metrics):
            return metrics

        def on_step_end(self, *, state, metrics, step):
            return state, metrics

        def _should_save_tpu_preemption_checkpoint(self, step):
            return False

        def _should_save_checkpoint(self, step):
            return False

        def _should_run_evaluation(self, step):
            return False

        def log_watchers(self, *args, **kwargs):
            return None

        def log_weight_distribution(self, *args, **kwargs):
            return None

        def _save_state(self, *args, **kwargs):
            return None

        def _save_checkpoint_for_step(self, *args, **kwargs):
            return None

        def maybe_generate(self, *args, **kwargs):
            return None

        def maybe_benchmark(self, *args, **kwargs):
            return None

        def _maybe_stop_step_trace(self, step):
            return None

        def log_metrics(self, *args, **kwargs):
            return None

        def train(self):
            state = SimpleNamespace(step=0, esurge_cache_scope_key="t")
            return self._train_epoch_async(
                state,
                None,
                iter([{"prompts": ["hi"]}]),
                SimpleNamespace(update=lambda **kw: (0.0, None)),
                SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
                None,
                0,
                epoch_start_step=0,
                epoch_end_step=1,
            )

    def _tracked_env(**kw):
        env_calls.append(str(kw.get("prompt", "")))
        return _MultiTurnCountingEnv(prompt=str(kw.get("prompt", "p")), target_turns=1, step_reward=1.0)

    inner = _MinimalAsyncTrainer()
    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=1,
            group_size=1,
            max_prompt_length=16,
            max_completion_length=16,
            skip_apply_chat_template=True,
            async_mode=True,
        ),
        trainer=inner,
        environment_factory=_tracked_env,
        generate_turn_fn=lambda st, ids, txt, t: ([ord("x")], "x", [-0.1]),
    )
    flow.train()
    assert env_calls == ["hi"]


def _numpy_grpo_reference(scale: float, *, advantages, action_rows, old_lp: float, ref_lp: float, beta: float):
    """Independent numpy GRPO ('grpo' loss, token-level clip 0.2) for the _TinyLM below."""
    vocab = 256
    log_z = np.log((vocab - 4) + 4 * np.exp(scale))
    action_lp = scale - log_z
    total = 0.0
    for adv, n_act in zip(advantages, action_rows, strict=True):
        coef = np.exp(action_lp - old_lp)
        surrogate = min(coef * adv, np.clip(coef, 0.8, 1.2) * adv)
        d = ref_lp - action_lp
        kl = np.exp(d) - d - 1.0
        per_token = -surrogate + beta * kl
        total += per_token * n_act / n_act
    return total / len(advantages)


def test_agent_flow_real_grpo_step_numerics_match_numpy_reference():
    inner = _stub_trainer(GRPOTrainer)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            group_size=2,
            max_prompt_length=16,
            max_completion_length=32,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(target_turns=2, step_reward=1.0),
        generate_turn_fn=lambda st, ids, txt, t: (
            [1, 2] if t == 0 else [3, 4],
            "a" if t == 0 else "STOP",
            [-0.5, -0.5],
        ),
    )
    batch, _ = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["p"]}, True)
    batch["advantages"] = jnp.asarray([1.0, -1.0], dtype=jnp.float32)
    # Observation tokens are excluded from the loss: exactly the 4 action tokens per row are active.
    assert np.asarray(batch["completion_mask"]).sum(axis=1).tolist() == [4, 4]

    class _TinyLM:
        mesh = None

        def __init__(self, scale):
            self.scale = scale

        def __call__(self, input_ids, attention_mask=None):
            b, s = input_ids.shape
            logits = jnp.zeros((b, s, 256), dtype=jnp.float32).at[:, :, 1:5].set(self.scale)
            return SimpleNamespace(logits=logits)

    def _loss(tree):
        metrics = grpo_step(
            state=SimpleNamespace(
                model=_TinyLM(tree["scale"]),
                graphdef=None,
                graphstate=tree,
                graphother=None,
                apply_loss_fn=None,
                merge=lambda t: _TinyLM(t["scale"]),
            ),
            batch=batch,
            num_generations=2,
            beta=0.04,
            loss_config=None,
            learning_rate_fn=lambda _: 1e-4,
            partition_spec=None,
            gradient_accumulation_steps=1,
            is_training=False,
            loss_type="grpo",
        )
        return metrics.loss

    scale = 0.5
    loss_val, grads = jax.value_and_grad(_loss)({"scale": jnp.array(scale, dtype=jnp.float32)})

    def ref(s):
        return _numpy_grpo_reference(s, advantages=[1.0, -1.0], action_rows=[4, 4], old_lp=-0.5, ref_lp=-0.5, beta=0.04)

    eps = 1e-4
    ref_grad = (ref(scale + eps) - ref(scale - eps)) / (2 * eps)
    np.testing.assert_allclose(float(loss_val), ref(scale), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(grads["scale"]), ref_grad, rtol=1e-3, atol=1e-6)


def test_agent_flow_adapts_distillation_and_sliding_window_context():
    inner = _stub_trainer(GKDTrainer)
    seen_prompts: list[str] = []

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        seen_prompts.append(prompt_text)
        return [ord("y")], "y", None

    flow = AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=3,
            context_mode="sliding_window",
            sliding_window_turns=1,
            reward_mode="terminal",
            group_size=1,
            max_prompt_length=16,
            max_completion_length=64,
            skip_apply_chat_template=True,
        ),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(prompt="P0", target_turns=3, step_reward=0.25),
        generate_turn_fn=gen_turn,
    )
    out, metrics = inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P0"]}, True)
    assert "action_mask" in out
    assert out["completion_mask"].shape == out["input_ids"].shape
    assert "obs1" in seen_prompts[1]
    assert "obs1" not in seen_prompts[2]
    assert "obs2" in seen_prompts[2]
    assert metrics["agent_flow/mean_reward"] == 0.25
    assert flow.arguments.context_mode == "sliding_window"


@pytest.mark.cpu_ok
def test_agent_flow_advantages_use_valid_rows_only():
    holder = SimpleNamespace(trainer=SimpleNamespace(arguments=SimpleNamespace(scale_rewards="group")))
    rewards = np.asarray([1.0, 3.0, 0.0, 5.0, 2.0, 0.0], dtype=np.float32)
    valid = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    adv = AgentFlowTrainer._compute_advantages(holder, rewards, valid, 3)
    # Group 0: valid rewards [1, 3] -> mean 2, std 1; failed row gets 0 and never enters the stats.
    expected = np.asarray([-1.0 / 1.0001, 1.0 / 1.0001, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # Group 1 has a single valid row -> zero advantage for the whole group.
    np.testing.assert_allclose(adv, expected, rtol=1e-6, atol=1e-7)

    holder.trainer.arguments.scale_rewards = "batch"
    adv_b = AgentFlowTrainer._compute_advantages(holder, rewards, valid, 3)
    vals = np.asarray([1.0, 3.0, 5.0])
    ref = (vals - vals.mean()) / (vals.std() + 1e-4)
    np.testing.assert_allclose(adv_b[[0, 1, 3]], ref, rtol=1e-6)
    assert adv_b[[2, 4, 5]].tolist() == [0.0, 0.0, 0.0]

    holder.trainer.arguments.scale_rewards = "none"
    adv_n = AgentFlowTrainer._compute_advantages(holder, rewards, valid, 3)
    np.testing.assert_allclose(adv_n, [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.cpu_ok
def test_agent_flow_rejects_online_dpo_inner_trainer():
    from easydel.trainers.online_dpo_trainer import OnlineDPOTrainer

    with pytest.raises(ValueError, match="OnlineDPO"):
        AgentFlowTrainer(
            arguments=AgentFlowConfig(max_turns=1, group_size=2),
            trainer=_stub_trainer(OnlineDPOTrainer),
            environment_factory=lambda **kw: _MultiTurnCountingEnv(),
        )


@pytest.mark.cpu_ok
def test_agent_flow_environment_only_grpo_gets_uncallable_placeholder_reward(monkeypatch):
    from easydel.trainers.agent_flow_trainer.agent_flow_trainer import agent_flow_supplied_reward

    captured: dict = {}

    class _CapturingGRPO(GRPOTrainer):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.arguments = kwargs["arguments"]
            self.processing_class = _DummyTokenizer()
            self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    original = Registry.get_or_raise

    def _get_or_raise(category, impl_name, wakeup=True):
        if category == "trainer" and impl_name == "grpo":
            return _CapturingGRPO
        return original(category, impl_name, wakeup)

    monkeypatch.setattr(Registry, "get_or_raise", _get_or_raise)
    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, base_config=GRPOConfig(num_generations=2)),
        environment_factory=lambda **kw: _MultiTurnCountingEnv(),
    )
    assert captured["reward_funcs"] is agent_flow_supplied_reward
    with pytest.raises(RuntimeError, match="supplies episode rewards"):
        agent_flow_supplied_reward(prompts=["p"], completions=["c"])


def test_agent_flow_ppo_without_sampling_logprobs_scores_old_logps_with_policy():
    inner = _stub_trainer(PPOTrainer)
    inner.arguments.positive_reward_threshold = 0.0
    seen: dict = {}

    def _score(state, input_ids, attention_mask):
        seen["state"] = state
        b, s = input_ids.shape
        comp = s - 24  # completion width = full width minus the stub's max_prompt_length
        lps = np.tile(-0.25 * np.arange(1, comp + 1, dtype=np.float32), (b, 1))
        return jnp.asarray(lps), jnp.full((b, comp), 0.75, dtype=jnp.float32)

    inner._score_rollout_policy_and_values = _score
    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, max_prompt_length=24, max_completion_length=32),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(target_turns=1, step_reward=1.0),
        generate_turn_fn=lambda st, ids, txt, t: ([ord("x")], "x", None),
    )
    state = SimpleNamespace(step=3)
    out, _ = inner._preprocess_batch_input(state, {"prompts": ["hi"]}, True)
    assert seen["state"] is state
    assert out["old_logps"].shape == out["completion_mask"].shape == (1, 48)
    np.testing.assert_allclose(np.asarray(out["old_logps"])[0], -0.25 * np.arange(1, 49, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(out["old_values"]), 0.75)


def test_agent_flow_tokenized_prompt_is_not_templated_twice():
    inner = _stub_trainer(
        GRPOTrainer, args=GRPOConfig(num_generations=1, max_prompt_length=32, max_completion_length=16)
    )
    seen: list[list[int]] = []

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        seen.append(list(prompt_ids))
        return [ord("y")], "y", [-0.1]

    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=1, group_size=1, max_prompt_length=32, max_completion_length=16),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(prompt="", target_turns=1),
        generate_turn_fn=gen_turn,
    )
    templated = "<user>hello<assistant>"
    ids = [0] * (32 - len(templated)) + [ord(c) for c in templated]
    mask = [0] * (32 - len(templated)) + [1] * len(templated)
    inner._preprocess_batch_input(
        SimpleNamespace(step=0),
        {"input_ids": np.asarray([ids], dtype=np.int32), "attention_mask": np.asarray([mask], dtype=np.int32)},
        True,
    )
    assert seen == [[ord(c) for c in templated]]


def test_agent_flow_observation_is_rendered_as_template_delta_without_system_reinjection():
    class _SystemTok(_DummyTokenizer):
        eos_token = "<E>"
        eos_token_id = 3

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = [] if messages and messages[0]["role"] == "system" else ["<system>default<E>"]
            parts += [f"<{m['role']}>{m['content']}<E>" for m in messages]
            if add_generation_prompt:
                parts.append("<assistant>")
            return "".join(parts)

    tok = _SystemTok()
    inner = _stub_trainer(
        GRPOTrainer, args=GRPOConfig(num_generations=1, max_prompt_length=64, max_completion_length=96)
    )
    inner.processing_class = tok
    seen: list[str] = []

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        seen.append("".join(chr(i) for i in prompt_ids))
        return [ord("a")], "a", [-0.1]

    AgentFlowTrainer(
        arguments=AgentFlowConfig(max_turns=2, group_size=1, max_prompt_length=64, max_completion_length=96),
        trainer=inner,
        environment_factory=lambda **kw: _MultiTurnCountingEnv(prompt="P", target_turns=2),
        generate_turn_fn=gen_turn,
    )
    inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompts": ["P"]}, True)
    expected_turn1 = tok.apply_chat_template(
        [
            {"role": "user", "content": "P"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "obs1"},
        ],
        add_generation_prompt=True,
    )
    assert seen[1] == expected_turn1
    assert seen[1].count("<system>") == 1


def test_agent_flow_conversational_prompt_is_rendered_once_and_seeds_chat_history():
    tok = _DummyTokenizer()
    inner = _stub_trainer(
        GRPOTrainer, args=GRPOConfig(num_generations=1, max_prompt_length=64, max_completion_length=96)
    )
    inner.processing_class = tok
    seen: list[str] = []

    def gen_turn(state, prompt_ids, prompt_text, turn_idx):
        seen.append("".join(chr(i) for i in prompt_ids))
        return [ord("a")], "a", [-0.1]

    AgentFlowTrainer(
        arguments=AgentFlowConfig(
            max_turns=2,
            group_size=1,
            max_prompt_length=64,
            max_completion_length=96,
            skip_apply_chat_template=False,
        ),
        trainer=inner,
        # Empty reset observation: the episode prompt comes from the batch's message list.
        environment_factory=lambda **kw: _MultiTurnCountingEnv(prompt="", target_turns=2),
        generate_turn_fn=gen_turn,
    )
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    inner._preprocess_batch_input(SimpleNamespace(step=0), {"prompt": [messages]}, True)

    assert seen[0] == tok.apply_chat_template(messages, add_generation_prompt=True)
    assert "'role'" not in seen[0]
    # The observation delta is rendered against a history seeded with the original messages.
    assert seen[1] == tok.apply_chat_template(
        [*messages, {"role": "assistant", "content": "a"}, {"role": "user", "content": "obs1"}],
        add_generation_prompt=True,
    )


def test_async_grpo_environment_only_counts_environment_reward_once(monkeypatch):
    from easydel.trainers.async_grpo_trainer import AsyncGRPOConfig, AsyncGRPOTrainer, environment_reward_placeholder
    from easydel.trainers.group_relative_policy_optimization.grpo_trainer import _compute_rewards_and_advantages
    from easydel.trainers.trainer.trainer import Trainer

    # Run the real AsyncGRPOTrainer -> GRPOTrainer.__init__ reward setup; only the generic Trainer base
    # (model compilation, dataloaders) is skipped.
    monkeypatch.setattr(Trainer, "__init__", lambda self, **kw: None)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("dp",))
    fake_state = SimpleNamespace(model=SimpleNamespace(mesh=mesh))
    fake_model = SimpleNamespace(to_state=lambda **kw: fake_state)
    args = AsyncGRPOConfig(num_generations=2, beta=0.0, disable_dropout=False)
    factory = lambda **kw: _MultiTurnCountingEnv()  # noqa: E731
    trainer = AsyncGRPOTrainer(
        arguments=args, model=fake_model, processing_class=_DummyTokenizer(), environment_factory=factory
    )

    assert trainer.reward_funcs == [environment_reward_placeholder]
    assert trainer.environment_factory is factory
    np.testing.assert_array_equal(np.asarray(trainer.reward_weights), np.asarray([0.0], dtype=np.float32))

    env_rewards = np.asarray([0.0, 1.0, 0.5, 2.0], dtype=np.float32)
    placeholder_col = np.asarray(trainer.reward_funcs[0](prompts=["p"] * 4, completions=["c"] * 4), dtype=np.float32)
    rewards_per_func = jnp.asarray(np.stack([placeholder_col, env_rewards], axis=1))
    # GRPO appends the environment column after the reward_funcs columns with weight 1.0.
    reward_weights = jnp.concatenate([trainer.reward_weights, jnp.ones((1,), dtype=trainer.reward_weights.dtype)])
    rewards, *_ = _compute_rewards_and_advantages(
        rewards_per_func=rewards_per_func,
        reward_weights=reward_weights,
        generation_factor=2,
        scale_rewards=trainer.scale_rewards,
        multi_objective_aggregation=trainer.multi_objective_aggregation,
        arguments=trainer.arguments,
        group_reduction=trainer._group_reduction,
    )
    np.testing.assert_allclose(np.asarray(rewards), env_rewards, rtol=1e-6)
