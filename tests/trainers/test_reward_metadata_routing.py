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

import json
from types import SimpleNamespace

import numpy as np
import pytest
from jax import numpy as jnp

from easydel.inference.openai_api_modules import FunctionCall, ToolCall
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.agentic_moshpit.agentic_moshpit_trainer import AgenticMoshPitTrainer
from easydel.trainers.agentic_moshpit.env_manager import (
    RolloutManager,
    TrajectoryResult,
    TurnRecord,
    turn_record_to_message,
)
from easydel.trainers.agentic_moshpit.environment import ToolEnvWrapper
from easydel.trainers.agentic_moshpit.tools import Tool
from easydel.trainers.base_trainer import GenerationResults
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.nash_md_trainer.nash_md_trainer import NashMDTrainer
from easydel.trainers.xpo_trainer.xpo_trainer import XPOTrainer


class _TokenizerStub:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        rendered = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        if tokenize:
            return {"input_ids": [[ord(ch) for ch in rendered]], "attention_mask": [[1] * len(rendered)]}
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


class _ToolAwareTokenizerStub(_TokenizerStub):
    def __init__(self):
        self.calls = []
        self.rendered_texts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        del kwargs
        self.calls.append((messages, tools))
        parts = []
        for message in messages:
            rendered = f"{message['role']}:{message['content']}"
            tool_calls = message.get("tool_calls")
            if tool_calls not in (None, []):
                rendered += f"<tool_calls>{json.dumps(tool_calls, sort_keys=True)}</tool_calls>"
            parts.append(rendered)
        if add_generation_prompt:
            parts.append("assistant:")
        rendered = "\n".join(parts)
        if tokenize:
            self.rendered_texts.append(rendered)
            return {"input_ids": [[ord(ch) for ch in rendered]], "attention_mask": [[1] * len(rendered)]}
        return rendered

    def __call__(
        self,
        texts,
        padding="max_length",
        padding_side="right",
        add_special_tokens=False,
        truncation=True,
        return_attention_mask=True,
        max_length=None,
        return_tensors="np",
    ):
        del padding, padding_side, add_special_tokens, truncation, return_attention_mask, max_length, return_tensors
        self.rendered_texts.extend(texts)
        batch_size = len(texts)
        return {
            "input_ids": np.ones((batch_size, 4), dtype=np.int32),
            "attention_mask": np.ones((batch_size, 4), dtype=np.int32),
        }


class _StrictToolAwareTokenizerStub(_ToolAwareTokenizerStub):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs):
        has_tool_calls = any(message.get("tool_calls") not in (None, []) for message in messages)
        if has_tool_calls and tools is None:
            raise RuntimeError("tools are required to render tool calls")
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **kwargs,
        )


class _TerminalEnv:
    def reset(self, seed=None):
        return SimpleNamespace(observation="solve", info={"seed": seed})

    def step(self, action):
        return SimpleNamespace(observation="", reward=1.0, terminated=True, truncated=False, info={})

    def close(self):
        return None


class _EchoTool(Tool):
    def call(self, value: str) -> str:
        return f"echo:{value}"


def test_agentic_generate_fn_keeps_raw_text_but_returns_post_strip_action_text():
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer._strip_thinking = lambda text: text.replace("<think>plan</think>", "")

    def fake_generate_unified(**kwargs):
        del kwargs
        return GenerationResults(
            generation_results=["legacy"],
            prompt_ids=np.zeros((1, 1), dtype=np.int32),
            prompt_mask=np.ones((1, 1), dtype=np.int32),
            sequences=np.zeros((1, 1), dtype=np.int32),
            completion_ids=np.zeros((1, 1), dtype=np.int32),
            completion_mask=np.ones((1, 1), dtype=np.int32),
            decoded_prompts=["prompt"],
            completion_prompts=["prompt"],
            text=["<think>plan</think>answer"],
            reasoning=["plan"],
            tool_calls=[None],
            raw_text=["<think>plan</think>answer"],
        )

    trainer.generate_unified = fake_generate_unified
    generate_fn = AgenticMoshPitTrainer._make_generate_fn(trainer, state=SimpleNamespace())

    responses = generate_fn(["prompt"], strip_thinking=True)

    assert responses == ["answer"]
    assert generate_fn.last_generation_metadata == [
        {
            "text": "answer",
            "raw_text": "<think>plan</think>answer",
            "reasoning": "plan",
            "tool_calls": None,
        }
    ]


def test_agentic_rollout_releases_esurge_after_rollout_failure():
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = SimpleNamespace(
        use_esurge_generation=True,
        group_size=1,
        num_env_groups=1,
    )
    trainer._rollout_step = 0
    trainer._strip_thinking = lambda text: text
    trainer.processing_class = None
    trainer.env_factory = lambda: _TerminalEnv()
    trainer._wrap_env_with_tools = lambda env: env

    class _RolloutManager:
        def run_grouped_episodes(self, **kwargs):
            del kwargs
            raise RuntimeError("rollout failed")

    trainer._rollout_manager = _RolloutManager()

    pause_calls: list[tuple[bool, bool]] = []

    class _Model:
        def pause_esurge(self, *, release_model_state=False, clear_compiled_cache=False):
            pause_calls.append((release_model_state, clear_compiled_cache))

    state = SimpleNamespace(model=_Model())

    with pytest.raises(RuntimeError, match="rollout failed"):
        AgenticMoshPitTrainer._preprocess_batch_input(
            trainer,
            state=state,
            batch={},
            is_train=True,
        )

    assert pause_calls == [(True, False)]


def test_rollout_manager_preserves_generation_metadata_on_turn_records():
    manager = RolloutManager(tokenizer=_TokenizerStub(), max_steps=2, max_seq_length=256)

    def generate_fn(prompts, **kwargs):
        generate_fn.last_generation_metadata = [
            {
                "text": "Visible answer",
                "raw_text": "<tool_call>{}</tool_call>",
                "reasoning": "plan",
                "tool_calls": [{"name": "lookup", "args": {}}],
            }
        ]
        return ["Visible answer"]

    trajectories = manager.run_grouped_episodes(
        env_factory=_TerminalEnv,
        generate_fn=generate_fn,
        group_size=1,
        num_groups=1,
    )

    assistant_turn = next(turn for turn in trajectories[0].turns if turn.role == "assistant")
    assert assistant_turn.content == "<tool_call>{}</tool_call>"
    assert assistant_turn.visible_content == "Visible answer"
    assert assistant_turn.raw_content == "<tool_call>{}</tool_call>"
    assert assistant_turn.reasoning == "plan"
    assert assistant_turn.tool_calls == [{"name": "lookup", "args": {}}]


def test_turn_record_to_message_normalizes_tool_call_models():
    message = turn_record_to_message(
        TurnRecord(
            role="assistant",
            content="<tool_call>{}</tool_call>",
            visible_content="Need tool",
            tool_calls=[ToolCall(function=FunctionCall(name="lookup", arguments='{"value":"ok"}'))],
        )
    )

    assert message["content"] == "Need tool"
    assert message["tool_calls"][0]["function"]["name"] == "lookup"
    assert message["tool_calls"][0]["function"]["arguments"] == {"value": "ok"}


def test_rollout_manager_replays_structured_tool_calls_in_follow_up_prompts_and_trajectory():
    tool = _EchoTool()
    tokenizer = _ToolAwareTokenizerStub()
    manager = RolloutManager(tokenizer=tokenizer, max_steps=3, max_seq_length=2048)
    prompts_seen = []

    def generate_fn(prompts, **kwargs):
        del kwargs
        prompts_seen.extend(prompts)
        if len(prompts_seen) == 1:
            generate_fn.last_generation_metadata = [
                {
                    "text": "Need tool",
                    "raw_text": f'<tool_call>{{"name":"{tool.name}","arguments":{{"value":"ok"}}}}</tool_call>',
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool.name,
                                "arguments": {"value": "ok"},
                            }
                        }
                    ],
                }
            ]
            return ["Need tool"]
        generate_fn.last_generation_metadata = [
            {
                "text": "Final answer",
                "raw_text": "Final answer",
                "tool_calls": None,
            }
        ]
        return ["Final answer"]

    trajectories = manager.run_grouped_episodes(
        env_factory=lambda: ToolEnvWrapper(
            env=_TerminalEnv(),
            tools=[tool],
            tool_call_parser=lambda action: [],
            max_tool_calls_per_step=1,
        ),
        generate_fn=generate_fn,
        group_size=1,
        num_groups=1,
    )

    assert len(prompts_seen) == 2
    assert "<tool_calls>" in prompts_seen[1]
    assert tool.name in prompts_seen[1]

    full_text = "".join(chr(int(token)) for token in trajectories[0].input_ids.tolist())
    assert "<tool_calls>" not in full_text
    assert f'<tool_call>{{"name":"{tool.name}","arguments":{{"value":"ok"}}}}</tool_call>' in full_text


def test_agentic_aux_rewards_update_final_step_and_receive_metadata():
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = SimpleNamespace(max_length=128, advantage_estimator="step_reinforce", tool_schemas=None)
    trainer.processing_class = _TokenizerStub()
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = np.asarray([1.0], dtype=np.float32)
    trainer.reward_func_names = ["capture_reward"]

    captured = {}

    def capture_reward(prompts, completions, raw_text, reasoning, tool_calls, trajectories, **kwargs):
        captured["prompts"] = prompts
        captured["completions"] = completions
        captured["raw_text"] = raw_text
        captured["reasoning"] = reasoning
        captured["tool_calls"] = tool_calls
        captured["trajectories"] = trajectories
        return [0.25]

    trainer.reward_funcs = [capture_reward]

    traj = TrajectoryResult(
        input_ids=np.array([1, 2, 3], dtype=np.int64),
        attention_mask=np.array([1, 1, 1], dtype=np.int32),
        prompt_mask=np.array([1, 1, 0], dtype=np.int32),
        response_mask=np.array([0, 0, 1], dtype=np.int32),
        episode_reward=1.0,
        step_rewards=[1.0],
        num_steps=1,
        turns=[
            TurnRecord(role="user", content="Question"),
            TurnRecord(
                role="assistant",
                content="Answer",
                is_response=True,
                visible_content="Answer",
                raw_content="<tool_call>{}</tool_call>",
                reasoning="Need tool",
                tool_calls=[{"name": "lookup", "args": {}}],
            ),
        ],
        info={},
    )

    aux_rewards, breakdown = AgenticMoshPitTrainer._apply_auxiliary_rewards_to_trajectories(trainer, [traj])

    assert aux_rewards.tolist() == pytest.approx([0.25])
    assert breakdown["capture_reward"].tolist() == pytest.approx([0.25])
    assert captured["raw_text"] == ["<tool_call>{}</tool_call>"]
    assert captured["reasoning"] == ["Need tool"]
    assert captured["tool_calls"] == [[{"name": "lookup", "args": {}}]]
    assert traj.episode_reward == pytest.approx(1.25)
    assert traj.step_rewards == pytest.approx([1.25])
    assert traj.info["env_reward"] == pytest.approx(1.0)
    assert traj.info["aux_reward"] == pytest.approx(0.25)
    assert traj.info["combined_reward"] == pytest.approx(1.25)


def test_agentic_callable_rewards_receive_structured_completion_messages():
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = SimpleNamespace(max_length=128, tool_schemas=None)
    trainer.processing_class = _TokenizerStub()
    trainer.reward_processing_classes = [None]
    trainer.reward_weights = np.asarray([1.0], dtype=np.float32)
    trainer.reward_func_names = ["capture_reward"]

    captured = {}

    def capture_reward(prompts, completions, raw_completions, **kwargs):
        captured["prompts"] = prompts
        captured["completions"] = completions
        captured["raw_completions"] = raw_completions
        return [0.0]

    trainer.reward_funcs = [capture_reward]

    traj = TrajectoryResult(
        input_ids=np.array([1, 2, 3], dtype=np.int64),
        attention_mask=np.array([1, 1, 1], dtype=np.int32),
        prompt_mask=np.array([1, 1, 0], dtype=np.int32),
        response_mask=np.array([0, 0, 1], dtype=np.int32),
        episode_reward=1.0,
        step_rewards=[1.0],
        num_steps=1,
        turns=[
            TurnRecord(role="user", content="Question"),
            TurnRecord(
                role="assistant",
                content="Answer",
                is_response=True,
                visible_content="Answer",
                raw_content="<tool_call>{}</tool_call>",
                tool_calls=[{"name": "lookup", "args": {}}],
            ),
        ],
        info={},
    )

    rewards, breakdown = AgenticMoshPitTrainer._score_auxiliary_rewards(trainer, [traj])

    assert rewards.tolist() == pytest.approx([0.0])
    assert breakdown["capture_reward"].tolist() == pytest.approx([0.0])
    assert captured["completions"] == [
        [{"role": "assistant", "content": "Answer", "tool_calls": [{"name": "lookup", "args": {}}]}]
    ]
    assert captured["raw_completions"] == [
        [
            {
                "role": "assistant",
                "content": "<tool_call>{}</tool_call>",
                "tool_calls": [{"name": "lookup", "args": {}}],
            }
        ]
    ]


def test_agentic_aux_reward_models_render_structured_tool_calls():
    tool = _EchoTool()
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer.arguments = SimpleNamespace(max_length=128, tool_schemas=[tool.chat_schema])
    trainer.processing_class = _ToolAwareTokenizerStub()
    reward_processor = _StrictToolAwareTokenizerStub()
    trainer.reward_processing_classes = [reward_processor]
    trainer.reward_weights = np.asarray([1.0], dtype=np.float32)
    trainer.reward_func_names = ["reward_model"]

    reward_model = EasyDeLState.__new__(EasyDeLState)
    object.__setattr__(reward_model, "graphdef", None)
    object.__setattr__(reward_model, "graphstate", None)
    object.__setattr__(reward_model, "graphother", None)

    def reward_apply_fn(graphdef, graphstate, graphother, reward_inputs):
        del graphdef, graphstate, graphother, reward_inputs
        return SimpleNamespace(logits=jnp.asarray([[1.0]], dtype=jnp.float32))

    object.__setattr__(reward_model, "apply_fn", reward_apply_fn)
    trainer.reward_funcs = [reward_model]

    traj = TrajectoryResult(
        input_ids=np.array([1, 2, 3], dtype=np.int64),
        attention_mask=np.array([1, 1, 1], dtype=np.int32),
        prompt_mask=np.array([1, 1, 0], dtype=np.int32),
        response_mask=np.array([0, 0, 1], dtype=np.int32),
        episode_reward=1.0,
        step_rewards=[0.0, 1.0],
        num_steps=2,
        turns=[
            TurnRecord(role="user", content="Question"),
            TurnRecord(
                role="assistant",
                content="Need tool",
                is_response=True,
                visible_content="Need tool",
                raw_content="Need tool",
                tool_calls=[
                    {
                        "function": {
                            "name": tool.name,
                            "arguments": {"value": "ok"},
                        }
                    }
                ],
            ),
            TurnRecord(role="tool", content=f"[{tool.name}]: echo:ok"),
            TurnRecord(
                role="assistant",
                content="Final answer",
                is_response=True,
                visible_content="Final answer",
                raw_content="Final answer",
            ),
        ],
        info={},
    )

    rewards, breakdown = AgenticMoshPitTrainer._score_auxiliary_rewards(trainer, [traj])

    assert rewards.tolist() == pytest.approx([1.0])
    assert breakdown["reward_model"].tolist() == pytest.approx([1.0])
    assert reward_processor.rendered_texts
    assert "<tool_calls>" in reward_processor.rendered_texts[0]
    assert tool.name in reward_processor.rendered_texts[0]
    assert reward_processor.calls[-1][1] == [tool.chat_schema]


def test_grpo_reward_models_render_structured_tool_calls():
    tool = _EchoTool()
    trainer = object.__new__(GRPOTrainer)
    trainer.arguments = SimpleNamespace(
        max_length=64,
        mask_truncated_completions=False,
        tool_schemas=[tool.chat_schema],
    )
    trainer.processing_class = _TokenizerStub()
    reward_processor = _StrictToolAwareTokenizerStub()
    trainer.reward_processing_classes = [reward_processor]
    trainer.reward_weights = jnp.asarray([1.0], dtype=jnp.float32)
    trainer.reward_func_names = ["reward_model"]
    trainer.ref_state = SimpleNamespace(
        model=SimpleNamespace(__call__=lambda *args, **kwargs: None),
        graphstate=None,
        graphother=None,
    )
    trainer.ref_logps_chunk_size = None
    trainer.scale_rewards = "none"
    trainer.train_is_conversational = True
    trainer.eval_is_conversational = True
    trainer.log_table = None
    trainer._pad_token_id = 0
    trainer._eos_token_id = [99]
    trainer._purify_batch = lambda batch: batch
    trainer._make_attn_mask = lambda ids: (ids != 0).astype(jnp.int32)
    trainer._all_gather = lambda value: value
    trainer.compute_refmodel_logps = lambda *args, **kwargs: jnp.zeros((1, 1), dtype=jnp.float32)

    reward_model = EasyDeLState.__new__(EasyDeLState)
    object.__setattr__(reward_model, "graphdef", None)
    object.__setattr__(reward_model, "graphstate", None)
    object.__setattr__(reward_model, "graphother", None)

    def reward_apply_fn(graphdef, graphstate, graphother, reward_inputs):
        del graphdef, graphstate, graphother, reward_inputs
        return SimpleNamespace(logits=jnp.asarray([[1.0]], dtype=jnp.float32))

    object.__setattr__(reward_model, "apply_fn", reward_apply_fn)
    trainer.reward_funcs = [reward_model]

    def fake_generate_unified(**kwargs):
        del kwargs
        prompt_messages = [[{"role": "user", "content": "Question"}]]
        return GenerationResults(
            generation_results=["Need tool"],
            prompt_ids=jnp.asarray([[1, 2]], dtype=jnp.int32),
            prompt_mask=jnp.asarray([[1, 1]], dtype=jnp.int32),
            sequences=jnp.asarray([[1, 2, 11]], dtype=jnp.int32),
            completion_ids=jnp.asarray([[11]], dtype=jnp.int32),
            completion_mask=jnp.asarray([[1]], dtype=jnp.int32),
            decoded_prompts=["Question"],
            completion_prompts=prompt_messages,
            text=["Need tool"],
            reasoning=["plan"],
            tool_calls=[
                [
                    {
                        "function": {
                            "name": tool.name,
                            "arguments": {"value": "ok"},
                        }
                    }
                ]
            ],
            raw_text=["<tool_call>{}</tool_call>"],
        )

    trainer.generate_unified = fake_generate_unified

    processed_batch, metrics = GRPOTrainer._preprocess_batch_input(
        trainer,
        state=SimpleNamespace(model=SimpleNamespace(__call__=lambda *args, **kwargs: None), step=jnp.asarray(0)),
        batch={
            "input_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1]], dtype=jnp.int32),
        },
        is_train=True,
    )

    assert "<tool_calls>" in reward_processor.rendered_texts[0]
    assert tool.name in reward_processor.rendered_texts[0]
    assert reward_processor.calls[-1][1] == [tool.chat_schema]
    assert processed_batch["advantages"].shape == (1,)
    assert metrics["reward_model"] == pytest.approx(1.0)


def test_tool_env_wrapper_resets_tool_limit_per_new_turn():
    tool = _EchoTool()
    wrapped = ToolEnvWrapper(
        env=_TerminalEnv(),
        tools=[tool],
        tool_call_parser=lambda action: [(tool.name, '{"value":"ok"}')],
        max_tool_calls_per_step=1,
    )

    first = wrapped.step("first")
    second = wrapped.step("second")

    assert first.info["tool_calls"][0]["name"] == tool.name
    assert second.info["tool_calls"][0]["name"] == tool.name
    assert second.observation == f"[{tool.name}]: echo:ok"


def test_tool_env_wrapper_executes_structured_tool_calls_without_raw_markup():
    tool = _EchoTool()
    wrapped = ToolEnvWrapper(
        env=_TerminalEnv(),
        tools=[tool],
        tool_call_parser=lambda action: [],
        max_tool_calls_per_step=1,
    )

    result = wrapped.step_with_tool_calls(
        "Visible answer",
        tool_calls=[
            {
                "function": {
                    "name": tool.name,
                    "arguments": {"value": "ok"},
                }
            }
        ],
    )

    assert result.info["tool_calls"] == [{"name": tool.name, "args": '{"value": "ok"}'}]
    assert result.observation == f"[{tool.name}]: echo:ok"


def test_xpo_reward_callables_receive_reasoning_and_tool_calls():
    trainer = object.__new__(XPOTrainer)
    trainer.arguments = SimpleNamespace(max_length=64)
    trainer.reward_processing_classes = [None]

    captured = {}

    def capture_reward(prompts, completions, raw_text, reasoning, tool_calls, **kwargs):
        captured["prompts"] = prompts
        captured["completions"] = completions
        captured["raw_text"] = raw_text
        captured["reasoning"] = reasoning
        captured["tool_calls"] = tool_calls
        return [0.5]

    trainer.reward_funcs = [capture_reward]

    rewards, breakdown = XPOTrainer._score_rewards(
        trainer,
        prompt_ids=np.zeros((1, 2), dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.int32),
        completion_ids=np.zeros((1, 2), dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.int32),
        prompt_texts=["Prompt"],
        completion_texts=["Answer"],
        raw_text=["<tool_call>{}</tool_call>"],
        reasoning=["Plan"],
        tool_calls=[[{"name": "lookup", "args": {}}]],
    )

    assert rewards.tolist() == pytest.approx([0.5])
    assert breakdown["capture_reward"].tolist() == pytest.approx([0.5])
    assert captured["reasoning"] == ["Plan"]
    assert captured["tool_calls"] == [[{"name": "lookup", "args": {}}]]


def test_nash_md_reward_callables_receive_reasoning_and_tool_calls():
    trainer = object.__new__(NashMDTrainer)
    trainer.arguments = SimpleNamespace(max_length=64)
    trainer.reward_processing_classes = [None]

    captured = {}

    def capture_reward(prompts, completions, raw_text, reasoning, tool_calls, batch, **kwargs):
        captured["prompts"] = prompts
        captured["completions"] = completions
        captured["raw_text"] = raw_text
        captured["reasoning"] = reasoning
        captured["tool_calls"] = tool_calls
        captured["batch"] = batch
        return [0.75]

    trainer.reward_funcs = [capture_reward]

    rewards, breakdown = NashMDTrainer._score_rewards(
        trainer,
        prompts=["Prompt"],
        completions=["Answer"],
        raw_text=["<tool_call>{}</tool_call>"],
        reasoning=["Plan"],
        tool_calls=[[{"name": "lookup", "args": {}}]],
        batch={"id": [1]},
    )

    assert rewards.tolist() == pytest.approx([0.75])
    assert breakdown["capture_reward"].tolist() == pytest.approx([0.75])
    assert captured["reasoning"] == ["Plan"]
    assert captured["tool_calls"] == [[{"name": "lookup", "args": {}}]]
    assert captured["batch"] == {"id": [1]}


def test_nash_md_mixture_rewards_follow_sampled_policy_metadata():
    class _PromptTokenizer:
        eos_token_id = None

        @staticmethod
        def batch_decode(ids, skip_special_tokens=True):
            del ids, skip_special_tokens
            return ["Prompt"]

    trainer = object.__new__(NashMDTrainer)
    trainer.arguments = SimpleNamespace(max_length=64)
    trainer.processing_class = _PromptTokenizer()
    trainer.reward_processing_classes = [None]
    trainer.reward_funcs = []
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._current_mixture_coef = lambda: 1.0
    trainer._current_beta_value = lambda: 0.2
    trainer.missing_eos_penalty = None
    trainer.ref_state = SimpleNamespace(graphstate=None, graphother=None)
    trainer.compute_refmodel_logps = lambda *args, **kwargs: jnp.zeros((1, 2), dtype=jnp.float32)

    captured_calls = []

    def capture_reward(prompts, completions, raw_text, reasoning, tool_calls, batch, **kwargs):
        del kwargs
        captured_calls.append(
            {
                "prompts": prompts,
                "completions": completions,
                "raw_text": raw_text,
                "reasoning": reasoning,
                "tool_calls": tool_calls,
                "batch": batch,
            }
        )
        return [0.0]

    trainer.reward_funcs = [capture_reward]

    def fake_generate_unified(*, state, **kwargs):
        del kwargs
        if state is trainer.ref_state:
            return GenerationResults(
                generation_results=["Reference answer"],
                prompt_ids=jnp.asarray([[1, 2]], dtype=jnp.int32),
                prompt_mask=jnp.asarray([[1, 1]], dtype=jnp.int32),
                sequences=jnp.asarray([[1, 2, 21, 22]], dtype=jnp.int32),
                completion_ids=jnp.asarray([[21, 22]], dtype=jnp.int32),
                completion_mask=jnp.asarray([[1, 1]], dtype=jnp.int32),
                decoded_prompts=["Prompt"],
                completion_prompts=["Prompt"],
                text=["Reference answer"],
                reasoning=["reference"],
                tool_calls=[[{"name": "reference-tool"}]],
                raw_text=["<think>reference</think>Reference answer"],
            )
        return GenerationResults(
            generation_results=["Policy answer"],
            prompt_ids=jnp.asarray([[1, 2]], dtype=jnp.int32),
            prompt_mask=jnp.asarray([[1, 1]], dtype=jnp.int32),
            sequences=jnp.asarray([[1, 2, 11, 12]], dtype=jnp.int32),
            completion_ids=jnp.asarray([[11, 12]], dtype=jnp.int32),
            completion_mask=jnp.asarray([[1, 1]], dtype=jnp.int32),
            decoded_prompts=["Prompt"],
            completion_prompts=["Prompt"],
            text=["Policy answer"],
            reasoning=["policy"],
            tool_calls=[[{"name": "policy-tool"}]],
            raw_text=["<think>policy</think>Policy answer"],
        )

    trainer.generate_unified = fake_generate_unified

    NashMDTrainer._preprocess_batch_input(
        trainer,
        state=SimpleNamespace(step=jnp.asarray(0, dtype=jnp.int32)),
        batch={
            "input_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1]], dtype=jnp.int32),
        },
        is_train=True,
    )

    assert len(captured_calls) == 2
    assert captured_calls[0]["completions"] == ["Policy answer"]
    assert captured_calls[1]["completions"] == ["Policy answer"]
    assert captured_calls[1]["raw_text"] == ["<think>policy</think>Policy answer"]
    assert captured_calls[1]["reasoning"] == ["policy"]
    assert captured_calls[1]["tool_calls"] == [[{"name": "policy-tool"}]]


def test_xpo_preprocess_supports_reward_models_without_text_decoding():
    trainer = object.__new__(XPOTrainer)
    trainer.arguments = SimpleNamespace(max_length=64, missing_eos_penalty=None)
    trainer.processing_class = SimpleNamespace(eos_token_id=None)
    trainer._purify_batch = lambda batch: batch
    trainer._all_gather = lambda value: value
    trainer._gather_scalar = lambda value, batch_size: jnp.full((batch_size,), value, dtype=jnp.float32)
    trainer._current_beta_value = lambda: 0.2
    trainer._current_alpha_value = lambda: 0.1
    trainer.ref_state = SimpleNamespace()
    trainer.loss_type_id = 0

    reward_state = EasyDeLState.__new__(EasyDeLState)
    object.__setattr__(reward_state, "graphdef", None)
    object.__setattr__(reward_state, "graphstate", None)
    object.__setattr__(reward_state, "graphother", None)
    reward_calls = []

    def reward_apply_fn(graphdef, graphstate, graphother, reward_inputs):
        del graphdef, graphstate, graphother
        reward_calls.append(reward_inputs)
        batch_size = reward_inputs["input_ids"].shape[0]
        return SimpleNamespace(logits=jnp.ones((batch_size, 1), dtype=jnp.float32))

    object.__setattr__(reward_state, "apply_fn", reward_apply_fn)
    trainer.reward_funcs = [reward_state]
    trainer._get_reward_processing_classes = lambda: [trainer.processing_class]

    def fake_generate_unified(*, state, **kwargs):
        del kwargs
        completion_token = 11 if state is not trainer.ref_state else 21
        completion_text = "policy" if state is not trainer.ref_state else "reference"
        return GenerationResults(
            generation_results=[completion_text],
            prompt_ids=jnp.asarray([[1, 2]], dtype=jnp.int32),
            prompt_mask=jnp.asarray([[1, 1]], dtype=jnp.int32),
            sequences=jnp.asarray([[1, 2, completion_token]], dtype=jnp.int32),
            completion_ids=jnp.asarray([[completion_token]], dtype=jnp.int32),
            completion_mask=jnp.asarray([[1]], dtype=jnp.int32),
            decoded_prompts=["Prompt"],
            completion_prompts=["Prompt"],
            text=[completion_text],
            reasoning=["plan"],
            tool_calls=[[{"name": "lookup"}]],
            raw_text=[f"<think>plan</think>{completion_text}"],
        )

    trainer.generate_unified = fake_generate_unified

    processed_batch, metrics = XPOTrainer._preprocess_batch_input(
        trainer,
        state=SimpleNamespace(),
        batch={
            "input_ids": jnp.asarray([[1, 2]], dtype=jnp.int32),
            "attention_mask": jnp.asarray([[1, 1]], dtype=jnp.int32),
        },
        is_train=True,
    )

    assert len(reward_calls) == 2
    assert processed_batch["chosen_mask"].shape == (1,)
    assert metrics["rewards/chosen"] == pytest.approx(1.0)
