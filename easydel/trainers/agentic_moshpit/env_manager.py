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

"""Rollout manager for multi-turn agentic trajectories.

This module implements the core multi-turn rollout loop for agentic MoshPit
training with **batched parallel generation**. All active environments
generate simultaneously at each turn via a single batched call to
``generate_unified``, then all environments are stepped in parallel.

The loop is:

1. Reset all ``N`` environments.
2. Batch-generate responses for all active environments in one call.
3. Step all active environments with their responses.
4. Drop terminated environments from the active set.
5. Repeat from (2) until all environments are done or ``max_steps`` hit.
6. Build trajectories for all ``N`` environments.

This avoids the sequential one-by-one generation that would otherwise
make eSurge reinitialize its KV cache ``N * max_steps`` times.
"""

from __future__ import annotations

import json
import math
import typing as tp
from dataclasses import dataclass, field

import numpy as np

from easydel.utils.helpers import get_logger

from .environment import AgenticEnvironment, ResetResult, ToolEnvWrapper
from .self_play import SelfPlayEnvironment

if tp.TYPE_CHECKING:
    from easydel.infra.utils import ProcessingClassType

logger = get_logger(__name__)


def _coerce_mapping_like(value: tp.Any) -> tp.Any:
    """Coerce JSON-string payloads into mapping-like objects when possible."""

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _normalize_tool_call_payloads(tool_calls: tp.Any) -> list[dict[str, tp.Any]]:
    """Normalize tool calls for HF/Jinja chat-template compatibility."""

    if not isinstance(tool_calls, list):
        return []

    normalized_calls: list[dict[str, tp.Any]] = []
    for raw_call in tool_calls:
        if isinstance(raw_call, dict):
            call = dict(raw_call)
        elif hasattr(raw_call, "model_dump"):
            try:
                call = dict(raw_call.model_dump(exclude_none=True))
            except Exception:
                continue
        else:
            function_payload = getattr(raw_call, "function", None)
            call = {}
            call_id = getattr(raw_call, "id", None)
            call_type = getattr(raw_call, "type", None)
            if call_id is not None:
                call["id"] = call_id
            if call_type is not None:
                call["type"] = call_type
            if function_payload is not None:
                if hasattr(function_payload, "model_dump"):
                    try:
                        call["function"] = dict(function_payload.model_dump(exclude_none=True))
                    except Exception:
                        pass
                else:
                    function_dict: dict[str, tp.Any] = {}
                    function_name = getattr(function_payload, "name", None)
                    function_arguments = getattr(function_payload, "arguments", None)
                    if function_name is not None:
                        function_dict["name"] = function_name
                    if function_arguments is not None:
                        function_dict["arguments"] = function_arguments
                    if function_dict:
                        call["function"] = function_dict
            if not call:
                continue

        function_payload = call.get("function")
        if isinstance(function_payload, dict):
            function_dict = dict(function_payload)
            arguments = _coerce_mapping_like(function_dict.get("arguments"))
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {"value": str(arguments)}
            function_dict["arguments"] = arguments
            call["function"] = function_dict
        elif isinstance(function_payload, str):
            coerced = _coerce_mapping_like(function_payload)
            if isinstance(coerced, dict):
                call["function"] = coerced

        normalized_calls.append(call)

    return normalized_calls


@dataclass
class TurnRecord:
    """Record of a single interaction turn.

    Attributes:
        role: Message role ("user", "assistant", "tool", "system").
        content: Canonical conversation content stored in the trajectory.
            For assistant turns this is the raw model output so future prompts
            and training masks reflect the actual generated tokens.
        token_ids: Token IDs for this turn's content.
        is_response: Whether this turn is a model response (trained on).
        reward: Step reward received after this turn (if any).
        visible_content: Parsed visible assistant content, if available.
        raw_content: Raw unsplit assistant content, if available.
        reasoning: Parsed reasoning content, if available.
        tool_calls: Parsed tool calls, if available.
    """

    role: str
    content: str
    token_ids: np.ndarray | None = None
    is_response: bool = False
    reward: float = 0.0
    visible_content: str | None = None
    raw_content: str | None = None
    reasoning: str | None = None
    tool_calls: tp.Any | None = None


def turn_record_to_message(turn: TurnRecord) -> dict[str, tp.Any]:
    """Convert a stored turn into a chat-template message payload.

    Assistant tool calls may be tracked separately from ``content`` when the
    generation backend emits structured tool-call metadata instead of raw text
    markup. In that case, rebuild the message from visible assistant content
    plus ``tool_calls`` so downstream chat templates can serialize the call
    instead of silently dropping it.
    """

    message: dict[str, tp.Any] = {"role": turn.role, "content": turn.content}
    if turn.role == "assistant" and turn.tool_calls not in (None, []):
        normalized_tool_calls = _normalize_tool_call_payloads(turn.tool_calls)
        if normalized_tool_calls:
            message["content"] = turn.visible_content if turn.visible_content is not None else turn.content
            message["tool_calls"] = normalized_tool_calls
    return message


@dataclass
class TrajectoryResult:
    """Complete trajectory from a single episode rollout.

    Attributes:
        input_ids: Concatenated token IDs for all turns.
        attention_mask: Valid token mask.
        prompt_mask: Mask for non-response tokens.
        response_mask: Mask for response tokens (gradient flows here).
        episode_reward: Total episode reward.
        step_rewards: List of per-step rewards.
        num_steps: Number of environment steps taken.
        turns: List of turn records for debugging/logging.
        info: Additional metadata from the environment.
    """

    input_ids: np.ndarray
    attention_mask: np.ndarray
    prompt_mask: np.ndarray
    response_mask: np.ndarray
    episode_reward: float
    step_rewards: list[float]
    num_steps: int
    turns: list[TurnRecord]
    info: dict[str, tp.Any] = field(default_factory=dict)


@dataclass
class _EpisodeState:
    """Mutable state for a single in-flight episode."""

    env: AgenticEnvironment
    turns: list[TurnRecord]
    step_rewards: list[float]
    episode_reward: float
    num_steps: int
    done: bool
    info: dict[str, tp.Any]


class RolloutManager:
    """Manages batched multi-turn rollouts for agentic MoshPit training.

    All active environments generate in a single batched call per turn,
    avoiding sequential eSurge reinitializations.

    The ``generate_fn`` passed to ``run_grouped_episodes`` must accept
    a **list of prompt strings** and return a **list of response strings**
    (one per prompt). The trainer wraps ``generate_unified`` to provide
    this batched interface.

    Args:
        tokenizer: Tokenizer for encoding/decoding text.
        max_steps: Maximum interaction steps per episode.
        max_seq_length: Maximum total sequence length for the trajectory.
        system_prompt: Optional system prompt prepended to conversations.
        tool_schemas: Optional tool schemas for chat template formatting.
    """

    def __init__(
        self,
        tokenizer: ProcessingClassType,
        max_steps: int = 10,
        max_seq_length: int = 4096,
        system_prompt: str | None = None,
        tool_schemas: list[dict[str, tp.Any]] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas

        self._pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def run_grouped_episodes(
        self,
        env_factory: tp.Callable[[], AgenticEnvironment],
        generate_fn: tp.Callable[[list[str]], list[str]],
        group_size: int,
        base_seed: int = 0,
        num_groups: int = 1,
    ) -> list[TrajectoryResult]:
        """Run all episodes with batched parallel generation.

        Creates ``num_groups * group_size`` environments, resets them
        all, then runs the multi-turn loop with batched generation:
        at each step, all active environments generate simultaneously
        in a single call. Terminated environments are removed from
        the active set but their trajectories are preserved.

        Args:
            env_factory: Factory function to create environment instances.
            generate_fn: Batched generation function. Accepts a list of
                prompt strings, returns a list of response strings (same
                length). This is one call to ``generate_unified``.
            group_size: Number of rollouts per seed group.
            base_seed: Starting seed for the first group.
            num_groups: Number of distinct seeds (prompts).

        Returns:
            List of TrajectoryResults, length ``num_groups * group_size``.
        """
        total = num_groups * group_size

        envs: list[AgenticEnvironment] = []
        seeds: list[int] = []
        for group_idx in range(num_groups):
            seed = base_seed + group_idx
            for _member in range(group_size):
                envs.append(env_factory())
                seeds.append(seed)

        def _unwrap(env: AgenticEnvironment) -> AgenticEnvironment:
            return env.env if isinstance(env, ToolEnvWrapper) else env

        inner_envs = [_unwrap(e) for e in envs]
        is_self_play = all(isinstance(e, SelfPlayEnvironment) for e in inner_envs)

        if is_self_play:
            sp_envs: list[SelfPlayEnvironment] = inner_envs  # type: ignore[assignment]
            generator = sp_envs[0]._generator
            for sp_env in sp_envs:
                sp_env._defer_verify = True
            topics = [e._topic for e in sp_envs]
            questions = generator.generate_batch(topics, seeds)
            for sp_env, q in zip(sp_envs, questions, strict=True):
                sp_env.reset_with_question(q)
            reset_results = [
                ResetResult(
                    observation=q.question,
                    info={"question": q.question, "topic": sp._topic, **q.metadata},
                )
                for sp, q in zip(sp_envs, questions, strict=True)
            ]
            for env in envs:
                if isinstance(env, ToolEnvWrapper):
                    env._tool_calls_this_step = 0
        else:
            reset_results = [e.reset(seed=s) for e, s in zip(envs, seeds, strict=True)]

        episodes: list[_EpisodeState] = []
        for env, reset_result in zip(envs, reset_results, strict=True):
            turns: list[TurnRecord] = []
            if self.system_prompt:
                turns.append(TurnRecord(role="system", content=self.system_prompt))
            turns.append(TurnRecord(role="user", content=reset_result.observation))
            episodes.append(
                _EpisodeState(
                    env=env,
                    turns=turns,
                    step_rewards=[],
                    episode_reward=0.0,
                    num_steps=0,
                    done=False,
                    info=reset_result.info,
                )
            )

        active_indices = list(range(total))

        for _step in range(self.max_steps):
            if not active_indices:
                break

            prompts = []
            for idx in active_indices:
                ep = episodes[idx]
                messages = [turn_record_to_message(t) for t in ep.turns]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=self.tool_schemas,
                )
                prompts.append(prompt_text)

            responses = generate_fn(prompts, num_return_sequences=1)
            generation_metadata = getattr(generate_fn, "last_generation_metadata", None)
            if not isinstance(generation_metadata, list):
                generation_metadata = [{} for _ in responses]
            elif len(generation_metadata) < len(responses):
                generation_metadata = generation_metadata + [
                    {} for _ in range(len(responses) - len(generation_metadata))
                ]

            next_active: list[int] = []
            pending_verify: list[tuple[int, str]] = []

            for i, idx in enumerate(active_indices):
                ep = episodes[idx]
                action_text = responses[i]
                metadata = generation_metadata[i] if i < len(generation_metadata) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                raw_text = metadata.get("raw_text")
                if not isinstance(raw_text, str):
                    raw_text = action_text
                visible_content = metadata.get("text")
                if not isinstance(visible_content, str):
                    visible_content = action_text
                reasoning = metadata.get("reasoning")
                if not isinstance(reasoning, str):
                    reasoning = None
                tool_calls = metadata.get("tool_calls")

                ep.turns.append(
                    TurnRecord(
                        role="assistant",
                        content=raw_text,
                        is_response=True,
                        visible_content=visible_content,
                        raw_content=raw_text,
                        reasoning=reasoning,
                        tool_calls=tool_calls,
                    )
                )

                if isinstance(ep.env, ToolEnvWrapper):
                    step_result = ep.env.step_with_tool_calls(action_text, tool_calls=tool_calls)
                else:
                    step_result = ep.env.step(action_text)
                is_deferred = math.isnan(step_result.reward)
                ep.step_rewards.append(0.0 if is_deferred else step_result.reward)
                if not is_deferred:
                    ep.episode_reward += step_result.reward
                ep.num_steps += 1

                if step_result.terminated or step_result.truncated:
                    if step_result.observation:
                        ep.turns.append(TurnRecord(role="user", content=step_result.observation))
                    ep.done = True
                    if is_deferred:
                        pending_verify.append((idx, sp_envs[idx]._final_answer or ""))
                    continue

                if step_result.observation:
                    is_tool = step_result.info.get("tool_calls") is not None
                    ep.turns.append(
                        TurnRecord(
                            role="tool" if is_tool else "user",
                            content=step_result.observation,
                        )
                    )
                next_active.append(idx)

            if pending_verify and is_self_play:
                verify_questions = [sp_envs[idx]._question for idx, _ in pending_verify]
                verify_answers = [ans for _, ans in pending_verify]
                verify_metas = [sp_envs[idx]._metadata for idx, _ in pending_verify]
                rewards = generator.verify_batch(verify_questions, verify_answers, verify_metas)
                raw_responses = getattr(generator, "_last_verify_responses", [""] * len(rewards))
                for i, ((idx, ans), reward) in enumerate(zip(pending_verify, rewards, strict=True)):
                    episodes[idx].episode_reward = reward
                    if episodes[idx].step_rewards:
                        episodes[idx].step_rewards[-1] = reward
                    info = sp_envs[idx]._make_info(reward, ans)
                    if i < len(raw_responses):
                        info["verifier_response"] = raw_responses[i]
                    episodes[idx].info.update(info)

            active_indices = next_active

        for ep in episodes:
            ep.env.close()

        return [self._build_trajectory(ep) for ep in episodes]

    def collate_trajectories(
        self,
        trajectories: list[TrajectoryResult],
        max_prompt_length: int,
        max_completion_length: int,
    ) -> dict[str, np.ndarray]:
        """Collate trajectories into batched arrays for training.

        Separates each trajectory into prompt and completion portions,
        then pads/truncates to uniform lengths. The split point is the
        first response token in each trajectory.

        Args:
            trajectories: List of TrajectoryResult from rollouts.
            max_prompt_length: Maximum prompt length (left-padded).
            max_completion_length: Maximum completion length (right-padded).

        Returns:
            Dict with ``prompt_ids``, ``prompt_mask``, ``completion_ids``,
            ``completion_mask``, ``rewards``, ``step_rewards_list``.
        """
        batch_prompt_ids = []
        batch_prompt_mask = []
        batch_completion_ids = []
        batch_completion_mask = []
        batch_rewards = []
        batch_step_rewards = []

        for traj in trajectories:
            first_response_idx = int(np.argmax(traj.response_mask))
            if traj.response_mask[first_response_idx] == 0:
                first_response_idx = len(traj.input_ids)

            prompt_ids = traj.input_ids[:first_response_idx]
            completion_ids = traj.input_ids[first_response_idx:]

            if len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]
            pad_len = max_prompt_length - len(prompt_ids)
            padded_prompt = np.concatenate(
                [
                    np.full(pad_len, self._pad_token_id, dtype=np.int64),
                    prompt_ids,
                ]
            )
            prompt_mask = np.concatenate(
                [
                    np.zeros(pad_len, dtype=np.int32),
                    np.ones(len(prompt_ids), dtype=np.int32),
                ]
            )

            if len(completion_ids) > max_completion_length:
                completion_ids = completion_ids[:max_completion_length]
            comp_pad_len = max_completion_length - len(completion_ids)
            padded_completion = np.concatenate(
                [
                    completion_ids,
                    np.full(comp_pad_len, self._pad_token_id, dtype=np.int64),
                ]
            )
            completion_mask = np.concatenate(
                [
                    np.ones(len(completion_ids), dtype=np.int32),
                    np.zeros(comp_pad_len, dtype=np.int32),
                ]
            )

            batch_prompt_ids.append(padded_prompt)
            batch_prompt_mask.append(prompt_mask)
            batch_completion_ids.append(padded_completion)
            batch_completion_mask.append(completion_mask)
            batch_rewards.append(traj.episode_reward)
            batch_step_rewards.append(traj.step_rewards)

        return {
            "prompt_ids": np.stack(batch_prompt_ids),
            "prompt_mask": np.stack(batch_prompt_mask),
            "completion_ids": np.stack(batch_completion_ids),
            "completion_mask": np.stack(batch_completion_mask),
            "rewards": np.array(batch_rewards, dtype=np.float32),
            "step_rewards_list": batch_step_rewards,
        }

    def _build_trajectory(self, ep: _EpisodeState) -> TrajectoryResult:
        """Tokenize turns and build a TrajectoryResult.

        Tokenizes each turn individually to track prompt vs response
        boundaries accurately, then concatenates into a single sequence.
        """
        messages = [{"role": t.role, "content": t.content} for t in ep.turns]
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=self.tool_schemas,
        )
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        response_positions: set[int] = set()
        for i, turn in enumerate(ep.turns):
            if turn.is_response and i > 0:
                prev_messages = messages[:i]
                prev_text = self.tokenizer.apply_chat_template(
                    prev_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=self.tool_schemas,
                )
                prev_ids = self.tokenizer.encode(prev_text, add_special_tokens=False)

                cur_messages = messages[: i + 1]
                cur_text = self.tokenizer.apply_chat_template(
                    cur_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=self.tool_schemas,
                )
                cur_ids = self.tokenizer.encode(cur_text, add_special_tokens=False)

                for pos in range(len(prev_ids), len(cur_ids)):
                    response_positions.add(pos)

        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[: self.max_seq_length]

        input_ids = np.array(full_ids, dtype=np.int64)
        attention_mask = np.ones(len(full_ids), dtype=np.int32)
        prompt_mask = np.array(
            [0 if i in response_positions else 1 for i in range(len(full_ids))],
            dtype=np.int32,
        )
        response_mask = 1 - prompt_mask

        return TrajectoryResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            response_mask=response_mask,
            episode_reward=ep.episode_reward,
            step_rewards=ep.step_rewards,
            num_steps=ep.num_steps,
            turns=ep.turns,
            info=ep.info,
        )
