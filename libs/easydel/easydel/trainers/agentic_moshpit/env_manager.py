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
``rollout``, then all environments are stepped in parallel.

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

from .environment import AgenticEnvironment, ToolEnvWrapper

if tp.TYPE_CHECKING:
    from easydel.infra.utils import ProcessingClassType

logger = get_logger(__name__)


def _coerce_mapping_like(value: tp.Any) -> tp.Any:
    """Coerce JSON-string payloads into mapping-like objects when possible.

    Args:
        value: Arbitrary payload; if it is a JSON-encoded string, the
            decoded object is returned, otherwise ``value`` passes through
            unchanged.

    Returns:
        The decoded JSON object when ``value`` is a parseable JSON string,
        else the original ``value``.
    """

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _normalize_tool_call_payloads(tool_calls: tp.Any) -> list[dict[str, tp.Any]]:
    """Normalize tool calls for HF/Jinja chat-template compatibility.

    Flattens dicts, ``model_dump``-able pydantic models, and objects with
    ``function`` attributes into a list of dicts with a guaranteed
    ``function.arguments`` mapping so chat templates can render them
    without crashing on string-encoded arguments.

    Args:
        tool_calls: Optional payload emitted by the generator; expected
            to be a list of tool-call descriptors but any non-list input
            is treated as empty.

    Returns:
        A list of normalized tool-call dicts. Empty when the input is not
        a recognisable list of tool calls.
    """

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


_TURN_BOUNDARY_SENTINEL = "__EASYDEL_AGENTIC_TURN_BOUNDARY_9F3A7C__"
_COMMON_EOT_MARKERS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|endoftext|>", "</s>")


@dataclass
class GenerationStepOutput:
    """Rich per-sample generation result preserving sampled token IDs and log-probs.

    Custom ``generate_fn`` callables passed to :class:`RolloutManager` may return
    either legacy ``list[str]`` outputs or a list of ``GenerationStepOutput`` /
    mapping objects carrying exact sampled ``token_ids`` and optional
    ``sampling_logps``.
    """

    text: str
    raw_text: str | None = None
    token_ids: np.ndarray | list[int] | None = None
    sampling_logps: np.ndarray | list[float] | None = None
    reasoning: str | None = None
    tool_calls: tp.Any | None = None


@dataclass
class TurnRecord:
    """Record of a single interaction turn.

    Attributes:
        role: Message role ("user", "assistant", "tool", "system").
        content: Canonical conversation content stored in the trajectory.
            For assistant turns this is the raw model output so future prompts
            and training masks reflect the actual generated tokens.
        token_ids: Exact sampled token IDs for this turn's content, if available.
        sampling_logps: Exact per-token sampling log-probabilities aligned with
            ``token_ids``, if available.
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
    sampling_logps: np.ndarray | None = None


def turn_record_to_message(turn: TurnRecord) -> dict[str, tp.Any]:
    """Convert a stored turn into a chat-template message payload.

    Assistant tool calls may be tracked separately from ``content`` when the
    generation backend emits structured tool-call metadata instead of raw text
    markup. In that case, rebuild the message from visible assistant content
    plus ``tool_calls`` so downstream chat templates can serialize the call
    instead of silently dropping it.

    Args:
        turn: The :class:`TurnRecord` to convert.

    Returns:
        A chat-template message dict with ``role`` / ``content`` and, for
        assistant turns with structured tool calls, an extra
        ``tool_calls`` list normalised via
        :func:`_normalize_tool_call_payloads`.
    """

    message: dict[str, tp.Any] = {"role": turn.role, "content": turn.content}
    if turn.role == "assistant" and turn.tool_calls not in (None, []):
        normalized_tool_calls = _normalize_tool_call_payloads(turn.tool_calls)
        if normalized_tool_calls:
            message["content"] = turn.visible_content if turn.visible_content is not None else turn.content
            message["tool_calls"] = normalized_tool_calls
    return message


def _canonical_turn_message(turn: TurnRecord) -> dict[str, tp.Any]:
    """Build the single canonical chat message used by BOTH rollout prompts and training.

    Prefers exact generated raw ``turn.content`` whenever present so rollout
    conditioning and training trajectories never diverge. Falls back to
    :func:`turn_record_to_message` only when ``turn.content`` is empty and
    structured ``tool_calls`` carry the assistant action.
    """
    if turn.content:
        return {"role": turn.role, "content": turn.content}
    return turn_record_to_message(turn)


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
        turn_ids: Explicit per-token assistant turn index (``0..T-1`` for
            assistant response tokens, ``-1`` for prompt/observation tokens).
        sampling_logps: Optional per-token sampling log-probabilities aligned
            with ``input_ids`` when returned by the generation backend.
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
    turn_ids: np.ndarray | None = None
    sampling_logps: np.ndarray | None = None


@dataclass
class _RecordedStep:
    """Incremental assistant turn + post-turn transition recorded during rollout."""

    turn: TurnRecord
    canonical_text: str
    transition_text: str


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
    initial_prompt_text: str = ""
    rendered_prompt: str = ""
    uses_sentinel_tracking: bool = False
    steps: list[_RecordedStep] = field(default_factory=list)
    running_token_ids: list[int] | None = None


class RolloutManager:
    """Manages batched multi-turn rollouts for agentic MoshPit training.

    All active environments generate in a single batched call per turn,
    avoiding sequential eSurge reinitializations.

    The ``generate_fn`` passed to ``run_grouped_episodes`` must accept
    a **list of prompt strings** and return a **list of response strings**
    (one per prompt). The trainer wraps ``rollout`` to provide
    this batched interface.

    Attributes:
        tokenizer: Tokenizer used for chat-template formatting and
            encoding/decoding text.
        max_steps: Maximum interaction steps per episode.
        max_seq_length: Maximum total sequence length for the trajectory
            (longer trajectories are truncated when tokenized).
        system_prompt: Optional system prompt prepended to conversations.
        tool_schemas: Optional list of tool schemas threaded through every
            ``apply_chat_template`` call.
    """

    def __init__(
        self,
        tokenizer: ProcessingClassType,
        max_steps: int = 10,
        max_seq_length: int = 4096,
        system_prompt: str | None = None,
        tool_schemas: list[dict[str, tp.Any]] | None = None,
    ):
        """Cache the tokenizer and rollout limits.

        Args:
            tokenizer: Tokenizer used to apply chat templates and
                tokenize observations.
            max_steps: Maximum number of agent/environment interaction
                steps per episode.
            max_seq_length: Maximum cumulative trajectory length in
                tokens (used to truncate state once exceeded).
            system_prompt: Optional system prompt injected at the start
                of every trajectory.
            tool_schemas: Optional list of tool function-call schemas
                forwarded to chat templates.
        """
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.max_seq_length = max_seq_length
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos_token_id, list | tuple):
                eos_token_id = eos_token_id[0] if eos_token_id else None
            pad_token_id = eos_token_id
        self._pad_token_id = int(pad_token_id) if pad_token_id is not None else 0

    def _special_eot_strings(self) -> tuple[str, ...]:
        """Collect non-whitespace special/EOS token strings sorted longest first."""
        candidates: list[str] = []
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if isinstance(eos_token, str) and eos_token.strip():
            candidates.append(eos_token)
        for attr in ("additional_special_tokens", "all_special_tokens"):
            val = getattr(self.tokenizer, attr, None)
            if isinstance(val, list | tuple | set):
                for tok in val:
                    if isinstance(tok, str) and tok.strip():
                        candidates.append(tok)
        for marker in _COMMON_EOT_MARKERS:
            candidates.append(marker)
        # Deduplicate and sort longest-first so multi-char markers match before substrings.
        unique = list(dict.fromkeys(candidates))
        unique.sort(key=len, reverse=True)
        return tuple(unique)

    def _dedup_transition_text(
        self,
        transition_text: str,
        *,
        assistant_text: str = "",
        token_ids: np.ndarray | None = None,
    ) -> str:
        """Strip a leading special EOT marker from ``transition_text`` if already emitted.

        Prevents duplicate ``<|im_end|>`` / EOS tokens when the model already generated
        the turn's closing special token, while never stripping ordinary whitespace such
        as ``\\n`` that might legitimately end an assistant message.
        """
        if not transition_text:
            return transition_text
        for sp in self._special_eot_strings():
            if not transition_text.startswith(sp):
                continue
            if token_ids is not None:
                encode = getattr(self.tokenizer, "encode", None)
                if callable(encode) and len(token_ids) > 0:
                    sp_ids = encode(sp, add_special_tokens=False)
                    if sp_ids and len(token_ids) >= len(sp_ids) and np.array_equal(token_ids[-len(sp_ids) :], sp_ids):
                        return transition_text[len(sp) :]
                if assistant_text.endswith(sp):
                    return transition_text[len(sp) :]
            elif assistant_text.endswith(sp):
                return transition_text[len(sp) :]
        return transition_text

    def _make_sentinel(self, *avoid_texts: str) -> str:
        """Pick a boundary sentinel guaranteed not to occur in any provided text."""
        sentinel = _TURN_BOUNDARY_SENTINEL
        counter = 0
        while any(sentinel in text for text in avoid_texts if text):
            counter += 1
            sentinel = f"{_TURN_BOUNDARY_SENTINEL}_{counter}"
        return sentinel

    def _render_canonical_assistant_text(self, turn: TurnRecord) -> str:
        """Return canonical assistant body text shared across rollout prompts and training."""
        if turn.content:
            return turn.content
        if turn.role == "assistant" and turn.tool_calls not in (None, []):
            sentinel = self._make_sentinel()
            try:
                with_sentinel = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "u"}, {"role": "assistant", "content": sentinel}],
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=self.tool_schemas,
                )
                with_turn = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "u"}, turn_record_to_message(turn)],
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=self.tool_schemas,
                )
            except Exception:
                return ""
            if sentinel in with_sentinel:
                idx = with_sentinel.rindex(sentinel)
                prefix = with_sentinel[:idx]
                suffix = with_sentinel[idx + len(sentinel) :]
                body = with_turn
                if prefix and body.startswith(prefix):
                    body = body[len(prefix) :]
                if suffix and body.endswith(suffix):
                    body = body[: len(body) - len(suffix)]
                return body
        return ""

    def _extract_transition_text(
        self,
        assistant_turn: TurnRecord,
        obs_turn: TurnRecord | None,
        *,
        add_generation_prompt: bool,
    ) -> tuple[str, bool]:
        """Extract the exact template transition text after an assistant turn.

        Uses a `<think>`-free sentinel assistant message so reasoning templates that
        strip `<think>` blocks from earlier turns or inject `<think>\\n` on
        ``add_generation_prompt=True`` never corrupt boundary offsets.
        """
        obs_content = obs_turn.content if obs_turn is not None else ""
        sentinel = self._make_sentinel(obs_content)
        probe: list[dict[str, tp.Any]] = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": sentinel},
        ]
        if obs_turn is not None:
            probe.append({"role": obs_turn.role, "content": obs_turn.content})
        try:
            rendered = self.tokenizer.apply_chat_template(
                probe,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                tools=self.tool_schemas,
            )
        except Exception:
            if obs_turn is not None and obs_turn.role == "tool":
                # Strict templates (e.g. Mistral) may require tool_calls on the
                # assistant turn immediately preceding a tool response.
                tool_calls = _normalize_tool_call_payloads(assistant_turn.tool_calls) or [
                    {"type": "function", "function": {"name": "tool", "arguments": {}}}
                ]
                probe[1] = {"role": "assistant", "content": sentinel, "tool_calls": tool_calls}
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        probe,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        tools=self.tool_schemas,
                    )
                    rendered_assistant_only = self.tokenizer.apply_chat_template(
                        probe[:2],
                        tokenize=False,
                        add_generation_prompt=False,
                        tools=self.tool_schemas,
                    )
                    if rendered.startswith(rendered_assistant_only):
                        # Extract end-of-assistant suffix + tool observation + next prompt.
                        with_plain = self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": "u"}, {"role": "assistant", "content": sentinel}],
                            tokenize=False,
                            add_generation_prompt=False,
                            tools=self.tool_schemas,
                        )
                        suffix = (
                            with_plain[with_plain.rindex(sentinel) + len(sentinel) :] if sentinel in with_plain else ""
                        )
                        return suffix + rendered[len(rendered_assistant_only) :], True
                except Exception:
                    return "", False
            return "", False

        if not isinstance(rendered, str) or sentinel not in rendered:
            return "", False
        return rendered[rendered.rindex(sentinel) + len(sentinel) :], True

    @staticmethod
    def _coerce_1d_int_array(value: tp.Any) -> np.ndarray | None:
        """Convert optional token-ID payload into a 1D int64 numpy array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
        return arr

    @staticmethod
    def _coerce_1d_float_array(value: tp.Any) -> np.ndarray | None:
        """Convert optional sampling-logprob payload into a 1D float32 numpy array."""
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        return arr

    def _unpack_generation_item(
        self,
        item: tp.Any,
        metadata: tp.Any,
        token_entry: tp.Any,
    ) -> tuple[str, str, str, str | None, tp.Any, np.ndarray | None, np.ndarray | None]:
        """Normalize legacy ``str`` or rich object/mapping generation outputs."""
        meta = metadata if isinstance(metadata, dict) else {}
        tok_meta = token_entry if isinstance(token_entry, dict) else {}

        def _first_present(*sources: tuple[tp.Mapping[str, tp.Any], tuple[str, ...]]) -> tp.Any:
            for mapping, keys in sources:
                for key in keys:
                    if key in mapping and mapping[key] is not None:
                        return mapping[key]
            return None

        if isinstance(item, str):
            action_text = item
            raw_text = meta.get("raw_text")
            if not isinstance(raw_text, str):
                raw_text = action_text
            visible_content = meta.get("text")
            if not isinstance(visible_content, str):
                visible_content = action_text
            reasoning = meta.get("reasoning")
            if not isinstance(reasoning, str):
                reasoning = None
            tool_calls = meta.get("tool_calls")
            raw_token_ids = _first_present(
                (tok_meta, ("token_ids", "completion_ids")),
                (meta, ("token_ids", "completion_ids")),
            )
            raw_logps = _first_present(
                (tok_meta, ("sampling_logps", "sampling_per_token_logps")),
                (meta, ("sampling_logps", "sampling_per_token_logps")),
            )
        elif isinstance(item, tp.Mapping):
            action_text = item.get("text")
            if not isinstance(action_text, str):
                action_text = str(item.get("raw_text", ""))
            raw_text = item.get("raw_text", meta.get("raw_text"))
            if not isinstance(raw_text, str):
                raw_text = action_text
            visible_content = item.get("text", meta.get("text"))
            if not isinstance(visible_content, str):
                visible_content = action_text
            reasoning = item.get("reasoning", meta.get("reasoning"))
            if not isinstance(reasoning, str):
                reasoning = None
            tool_calls = item.get("tool_calls", meta.get("tool_calls"))
            raw_token_ids = _first_present(
                (item, ("token_ids", "completion_ids")),
                (tok_meta, ("token_ids", "completion_ids")),
                (meta, ("token_ids", "completion_ids")),
            )
            raw_logps = _first_present(
                (item, ("sampling_logps", "sampling_per_token_logps")),
                (tok_meta, ("sampling_logps", "sampling_per_token_logps")),
                (meta, ("sampling_logps", "sampling_per_token_logps")),
            )
        else:
            action_text = getattr(item, "text", None)
            if not isinstance(action_text, str):
                action_text = str(getattr(item, "raw_text", "") or "")
            raw_text = getattr(item, "raw_text", None)
            if not isinstance(raw_text, str):
                raw_text = meta.get("raw_text") if isinstance(meta.get("raw_text"), str) else action_text
            visible_content = getattr(item, "text", None)
            if not isinstance(visible_content, str):
                visible_content = meta.get("text") if isinstance(meta.get("text"), str) else action_text
            reasoning = getattr(item, "reasoning", None)
            if not isinstance(reasoning, str):
                reasoning = meta.get("reasoning") if isinstance(meta.get("reasoning"), str) else None
            tool_calls = getattr(item, "tool_calls", None)
            if tool_calls is None:
                tool_calls = meta.get("tool_calls")
            raw_token_ids = getattr(item, "token_ids", None)
            if raw_token_ids is None:
                raw_token_ids = getattr(item, "completion_ids", None)
            if raw_token_ids is None:
                raw_token_ids = _first_present(
                    (tok_meta, ("token_ids", "completion_ids")),
                    (meta, ("token_ids", "completion_ids")),
                )
            raw_logps = getattr(item, "sampling_logps", None)
            if raw_logps is None:
                raw_logps = getattr(item, "sampling_per_token_logps", None)
            if raw_logps is None:
                raw_logps = _first_present(
                    (tok_meta, ("sampling_logps", "sampling_per_token_logps")),
                    (meta, ("sampling_logps", "sampling_per_token_logps")),
                )

        token_ids = self._coerce_1d_int_array(raw_token_ids)
        sampling_logps = self._coerce_1d_float_array(raw_logps)
        if token_ids is not None and sampling_logps is not None and len(sampling_logps) != len(token_ids):
            sampling_logps = None
        return action_text, raw_text, visible_content, reasoning, tool_calls, token_ids, sampling_logps

    def run_grouped_episodes(
        self,
        env_factory: tp.Callable[[], AgenticEnvironment],
        generate_fn: tp.Callable[..., list[tp.Any]],
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
                prompt strings, returns a list of response strings (or rich
                :class:`GenerationStepOutput` / mapping objects) of the same
                length.
            group_size: Number of rollouts per seed group.
            base_seed: Starting seed for the first group.
            num_groups: Number of distinct seeds (prompts).

        Returns:
            List of TrajectoryResults, length ``num_groups * group_size``.
        """
        total = num_groups * group_size
        if total <= 0:
            return []

        envs: list[AgenticEnvironment] = []
        seeds: list[int] = []
        encode_fn = getattr(self.tokenizer, "encode", None)
        pass_prompt_ids = bool(getattr(generate_fn, "_accepts_prompt_token_ids", False)) and callable(encode_fn)

        try:
            for group_idx in range(num_groups):
                seed = base_seed + group_idx
                for _member in range(group_size):
                    envs.append(env_factory())
                    seeds.append(seed)

            if not envs:
                return []

            reset_batch = getattr(envs[0], "reset_batch", None)
            reset_results = (
                reset_batch(envs, seeds)
                if callable(reset_batch)
                else [env.reset(seed=seed) for env, seed in zip(envs, seeds, strict=True)]
            )
            step_limits = [
                min(self.max_steps, limit) if (limit := getattr(env, "max_steps", None)) is not None else self.max_steps
                for env in envs
            ]

            episodes: list[_EpisodeState] = []
            for env, reset_result in zip(envs, reset_results, strict=True):
                turns: list[TurnRecord] = []
                if self.system_prompt:
                    turns.append(TurnRecord(role="system", content=self.system_prompt))
                turns.append(TurnRecord(role="user", content=reset_result.observation))
                initial_messages = [_canonical_turn_message(t) for t in turns]
                initial_prompt = self.tokenizer.apply_chat_template(
                    initial_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=self.tool_schemas,
                )
                running_ids = list(encode_fn(initial_prompt, add_special_tokens=False)) if pass_prompt_ids else None
                episodes.append(
                    _EpisodeState(
                        env=env,
                        turns=turns,
                        step_rewards=[],
                        episode_reward=0.0,
                        num_steps=0,
                        done=False,
                        info=dict(reset_result.info) if isinstance(reset_result.info, dict) else reset_result.info,
                        initial_prompt_text=initial_prompt,
                        rendered_prompt=initial_prompt,
                        uses_sentinel_tracking=True,
                        running_token_ids=running_ids,
                    )
                )

            active_indices = list(range(total))

            for _step in range(self.max_steps):
                if not active_indices:
                    break

                prompts: list[str] = []
                prompt_token_ids_batch: list[list[int]] = []
                for idx in active_indices:
                    ep = episodes[idx]
                    if ep.uses_sentinel_tracking:
                        prompt_text = ep.rendered_prompt
                    else:
                        messages = [_canonical_turn_message(t) for t in ep.turns]
                        prompt_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            tools=self.tool_schemas,
                        )
                    prompts.append(prompt_text)
                    if pass_prompt_ids and ep.running_token_ids is not None:
                        prompt_token_ids_batch.append(list(ep.running_token_ids))

                if pass_prompt_ids and len(prompt_token_ids_batch) == len(prompts):
                    responses = generate_fn(
                        prompts,
                        num_return_sequences=1,
                        prompt_token_ids=prompt_token_ids_batch,
                    )
                else:
                    responses = generate_fn(prompts, num_return_sequences=1)

                generation_metadata = getattr(generate_fn, "last_generation_metadata", None)
                if not isinstance(generation_metadata, list):
                    generation_metadata = [{} for _ in responses]
                elif len(generation_metadata) < len(responses):
                    generation_metadata = generation_metadata + [
                        {} for _ in range(len(responses) - len(generation_metadata))
                    ]

                generation_tokens = getattr(generate_fn, "last_generation_tokens", None)
                if not isinstance(generation_tokens, list):
                    generation_tokens = [{} for _ in responses]
                elif len(generation_tokens) < len(responses):
                    generation_tokens = generation_tokens + [
                        {} for _ in range(len(responses) - len(generation_tokens))
                    ]

                next_active: list[int] = []
                pending_verify: list[int] = []

                for i, idx in enumerate(active_indices):
                    ep = episodes[idx]
                    (
                        action_text,
                        raw_text,
                        visible_content,
                        reasoning,
                        tool_calls,
                        token_ids,
                        sampling_logps,
                    ) = self._unpack_generation_item(
                        responses[i],
                        generation_metadata[i] if i < len(generation_metadata) else {},
                        generation_tokens[i] if i < len(generation_tokens) else {},
                    )

                    assistant_turn = TurnRecord(
                        role="assistant",
                        content=raw_text,
                        token_ids=token_ids,
                        is_response=True,
                        visible_content=visible_content,
                        raw_content=raw_text,
                        reasoning=reasoning,
                        tool_calls=tool_calls,
                        sampling_logps=sampling_logps,
                    )
                    ep.turns.append(assistant_turn)

                    if isinstance(ep.env, ToolEnvWrapper):
                        step_result = ep.env.step_with_tool_calls(action_text, tool_calls=tool_calls)
                    else:
                        step_result = ep.env.step(action_text)
                    is_deferred = math.isnan(step_result.reward)
                    ep.step_rewards.append(0.0 if is_deferred else step_result.reward)
                    if not is_deferred:
                        ep.episode_reward += step_result.reward
                    ep.num_steps += 1

                    reached_limit = ep.num_steps >= step_limits[idx]
                    ep.info.update(step_result.info)

                    is_done = bool(step_result.terminated or step_result.truncated or reached_limit)
                    obs_turn: TurnRecord | None = None
                    if step_result.observation:
                        is_tool = step_result.info.get("tool_calls") is not None
                        obs_turn = TurnRecord(
                            role="tool" if is_tool else "user",
                            content=step_result.observation,
                        )
                        ep.turns.append(obs_turn)

                    canonical_text = self._render_canonical_assistant_text(assistant_turn)
                    transition_text, sentinel_ok = self._extract_transition_text(
                        assistant_turn,
                        obs_turn,
                        add_generation_prompt=not is_done,
                    )
                    ep.steps.append(
                        _RecordedStep(
                            turn=assistant_turn,
                            canonical_text=canonical_text,
                            transition_text=transition_text,
                        )
                    )
                    if sentinel_ok and ep.uses_sentinel_tracking:
                        prompt_transition = self._dedup_transition_text(
                            transition_text,
                            assistant_text=canonical_text,
                            token_ids=None,
                        )
                        ep.rendered_prompt = ep.rendered_prompt + canonical_text + prompt_transition
                    else:
                        ep.uses_sentinel_tracking = False
                        ep.running_token_ids = None

                    if pass_prompt_ids and ep.running_token_ids is not None and callable(encode_fn):
                        if token_ids is not None:
                            ep.running_token_ids.extend(token_ids.tolist())
                        elif canonical_text:
                            ep.running_token_ids.extend(encode_fn(canonical_text, add_special_tokens=False))
                        tok_transition = self._dedup_transition_text(
                            transition_text,
                            assistant_text=canonical_text,
                            token_ids=token_ids,
                        )
                        if tok_transition:
                            ep.running_token_ids.extend(encode_fn(tok_transition, add_special_tokens=False))

                    if is_done:
                        ep.done = True
                        if reached_limit and not step_result.terminated:
                            ep.info["truncated"] = True
                        if is_deferred or (reached_limit and not step_result.terminated):
                            pending_verify.append(idx)
                        continue

                    next_active.append(idx)

                if pending_verify:
                    pending_envs = [envs[idx] for idx in pending_verify]
                    finalize_batch = getattr(pending_envs[0], "finalize_batch", None)
                    final_results = (
                        finalize_batch(pending_envs) if callable(finalize_batch) else [None] * len(pending_envs)
                    )
                    for idx, result in zip(pending_verify, final_results, strict=True):
                        if result is None:
                            continue
                        episode = episodes[idx]
                        previous_reward = episode.step_rewards[-1]
                        episode.episode_reward += result.reward - previous_reward
                        episode.step_rewards[-1] = result.reward
                        episode.info.update(result.info)

                active_indices = next_active

            return [self._build_trajectory(ep) for ep in episodes]
        finally:
            for env in envs:
                try:
                    env.close()
                except Exception as exc:  # pragma: no cover - best-effort cleanup
                    logger.debug("Failed to close environment cleanly: %s", exc)

    def collate_trajectories(
        self,
        trajectories: list[TrajectoryResult],
        max_prompt_length: int,
        max_completion_length: int,
    ) -> dict[str, np.ndarray | list[list[float]]]:
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
            ``completion_attention_mask``, ``completion_mask``,
            ``completion_turn_ids``, ``rewards``, and ``step_rewards_list``,
            plus ``sampling_per_token_logps`` when every trajectory carries
            sampling log-probabilities.
        """
        if not trajectories:
            return {
                "prompt_ids": np.zeros((0, max_prompt_length), dtype=np.int64),
                "prompt_mask": np.zeros((0, max_prompt_length), dtype=np.int32),
                "completion_ids": np.zeros((0, max_completion_length), dtype=np.int64),
                "completion_attention_mask": np.zeros((0, max_completion_length), dtype=np.int32),
                "completion_mask": np.zeros((0, max_completion_length), dtype=np.int32),
                "completion_turn_ids": np.full((0, max_completion_length), -1, dtype=np.int32),
                "rewards": np.zeros((0,), dtype=np.float32),
                "step_rewards_list": [],
            }

        batch_prompt_ids = []
        batch_prompt_mask = []
        batch_completion_ids = []
        batch_completion_attention_mask = []
        batch_completion_mask = []
        batch_completion_turn_ids = []
        batch_sampling_logps = []
        batch_rewards = []
        batch_step_rewards = []
        all_have_logps = all(traj.sampling_logps is not None for traj in trajectories)

        for traj in trajectories:
            if traj.response_mask.size == 0:
                first_response_idx = len(traj.input_ids)
            else:
                first_response_idx = int(np.argmax(traj.response_mask))
                if traj.response_mask[first_response_idx] == 0:
                    first_response_idx = len(traj.input_ids)

            prompt_ids = traj.input_ids[:first_response_idx]
            completion_ids = traj.input_ids[first_response_idx:]
            completion_response_mask = traj.response_mask[first_response_idx:]
            if traj.turn_ids is not None:
                completion_turn_ids = np.asarray(traj.turn_ids[first_response_idx:], dtype=np.int32)
            else:
                completion_turn_ids = np.full(len(completion_response_mask), -1, dtype=np.int32)
                mask_bool = np.asarray(completion_response_mask, dtype=bool)
                if mask_bool.size:
                    starts = np.flatnonzero(mask_bool & ~np.r_[False, mask_bool[:-1]])
                    ends = np.flatnonzero(mask_bool & ~np.r_[mask_bool[1:], False]) + 1
                    for seg_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
                        completion_turn_ids[start:end] = seg_idx

            completion_logps = (
                np.asarray(traj.sampling_logps[first_response_idx:], dtype=np.float32) if all_have_logps else None
            )

            if len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]
            pad_len = max_prompt_length - len(prompt_ids)
            padded_prompt = np.concatenate(
                [
                    np.full(pad_len, self._pad_token_id, dtype=np.int64),
                    prompt_ids.astype(np.int64, copy=False),
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
                completion_response_mask = completion_response_mask[:max_completion_length]
                completion_turn_ids = completion_turn_ids[:max_completion_length]
                if completion_logps is not None:
                    completion_logps = completion_logps[:max_completion_length]
            comp_pad_len = max_completion_length - len(completion_ids)
            padded_completion = np.concatenate(
                [
                    completion_ids.astype(np.int64, copy=False),
                    np.full(comp_pad_len, self._pad_token_id, dtype=np.int64),
                ]
            )
            completion_attention_mask = np.concatenate(
                [
                    np.ones(len(completion_ids), dtype=np.int32),
                    np.zeros(comp_pad_len, dtype=np.int32),
                ]
            )
            completion_mask = np.concatenate(
                [
                    completion_response_mask.astype(np.int32),
                    np.zeros(comp_pad_len, dtype=np.int32),
                ]
            )
            padded_turn_ids = np.concatenate(
                [
                    completion_turn_ids.astype(np.int32),
                    np.full(comp_pad_len, -1, dtype=np.int32),
                ]
            )

            batch_prompt_ids.append(padded_prompt)
            batch_prompt_mask.append(prompt_mask)
            batch_completion_ids.append(padded_completion)
            batch_completion_attention_mask.append(completion_attention_mask)
            batch_completion_mask.append(completion_mask)
            batch_completion_turn_ids.append(padded_turn_ids)
            if completion_logps is not None:
                padded_logps = np.concatenate(
                    [
                        completion_logps,
                        np.zeros(comp_pad_len, dtype=np.float32),
                    ]
                )
                batch_sampling_logps.append(padded_logps)
            batch_rewards.append(traj.episode_reward)
            batch_step_rewards.append(traj.step_rewards)

        result: dict[str, np.ndarray | list[list[float]]] = {
            "prompt_ids": np.stack(batch_prompt_ids),
            "prompt_mask": np.stack(batch_prompt_mask),
            "completion_ids": np.stack(batch_completion_ids),
            "completion_attention_mask": np.stack(batch_completion_attention_mask),
            "completion_mask": np.stack(batch_completion_mask),
            "completion_turn_ids": np.stack(batch_completion_turn_ids),
            "rewards": np.array(batch_rewards, dtype=np.float32),
            "step_rewards_list": batch_step_rewards,
        }
        if all_have_logps:
            result["sampling_per_token_logps"] = np.stack(batch_sampling_logps)
        return result

    def _reconstruct_steps_from_turns(self, turns: list[TurnRecord]) -> tuple[str, list[_RecordedStep]]:
        """Reconstruct incremental prompt + steps when ``_build_trajectory`` is called directly."""
        first_resp_idx = next((i for i, turn in enumerate(turns) if turn.is_response), len(turns))
        init_turns = turns[:first_resp_idx]
        if init_turns:
            init_messages = [_canonical_turn_message(t) for t in init_turns]
            initial_prompt = self.tokenizer.apply_chat_template(
                init_messages,
                tokenize=False,
                add_generation_prompt=first_resp_idx < len(turns),
                tools=self.tool_schemas,
            )
        else:
            initial_prompt = ""

        steps: list[_RecordedStep] = []
        i = first_resp_idx
        while i < len(turns):
            turn = turns[i]
            if not turn.is_response:
                i += 1
                continue
            obs_turn: TurnRecord | None = None
            next_idx = i + 1
            if next_idx < len(turns) and not turns[next_idx].is_response:
                obs_turn = turns[next_idx]
                next_idx += 1
            has_more_responses = any(t.is_response for t in turns[next_idx:])
            canonical_text = self._render_canonical_assistant_text(turn)
            transition_text, _ = self._extract_transition_text(
                turn,
                obs_turn,
                add_generation_prompt=has_more_responses,
            )
            steps.append(_RecordedStep(turn=turn, canonical_text=canonical_text, transition_text=transition_text))
            i = next_idx
        return initial_prompt, steps

    def _build_trajectory(self, ep: _EpisodeState) -> TrajectoryResult:
        """Build a TrajectoryResult incrementally using exact per-turn token IDs.

        Avoids re-tokenizing full multi-turn conversations and subtracting
        independent prefix lengths (which breaks under reasoning templates that
        strip ``<think>`` blocks or inject ``<think>\\n`` generation prompts,
        and under BPE cross-boundary merges).
        """
        if ep.steps:
            initial_prompt = ep.initial_prompt_text
            steps = ep.steps
        else:
            initial_prompt, steps = self._reconstruct_steps_from_turns(ep.turns)

        has_all_logps = bool(steps) and any(step.turn.sampling_logps is not None for step in steps) and all(
            step.turn.sampling_logps is not None
            or (step.turn.token_ids is not None and len(step.turn.token_ids) == 0)
            for step in steps
        )

        id_chunks: list[np.ndarray] = []
        resp_chunks: list[np.ndarray] = []
        turn_id_chunks: list[np.ndarray] = []
        logp_chunks: list[np.ndarray] = []

        if initial_prompt:
            init_ids = np.asarray(self.tokenizer.encode(initial_prompt, add_special_tokens=False), dtype=np.int64)
            if init_ids.size:
                id_chunks.append(init_ids)
                resp_chunks.append(np.zeros(init_ids.size, dtype=np.int32))
                turn_id_chunks.append(np.full(init_ids.size, -1, dtype=np.int32))
                if has_all_logps:
                    logp_chunks.append(np.zeros(init_ids.size, dtype=np.float32))

        for turn_idx, step in enumerate(steps):
            turn = step.turn
            if turn.token_ids is not None:
                resp_ids = np.asarray(turn.token_ids, dtype=np.int64).reshape(-1)
            elif step.canonical_text:
                resp_ids = np.asarray(
                    self.tokenizer.encode(step.canonical_text, add_special_tokens=False),
                    dtype=np.int64,
                )
            else:
                resp_ids = np.zeros(0, dtype=np.int64)

            if resp_ids.size:
                id_chunks.append(resp_ids)
                resp_chunks.append(np.ones(resp_ids.size, dtype=np.int32))
                turn_id_chunks.append(np.full(resp_ids.size, turn_idx, dtype=np.int32))
                if has_all_logps:
                    if turn.sampling_logps is not None:
                        logp_chunks.append(np.asarray(turn.sampling_logps, dtype=np.float32).reshape(-1))
                    else:
                        logp_chunks.append(np.zeros(resp_ids.size, dtype=np.float32))

            deduped_transition = self._dedup_transition_text(
                step.transition_text,
                assistant_text=step.canonical_text,
                token_ids=turn.token_ids,
            )
            if deduped_transition:
                trans_ids = np.asarray(
                    self.tokenizer.encode(deduped_transition, add_special_tokens=False),
                    dtype=np.int64,
                )
                if trans_ids.size:
                    id_chunks.append(trans_ids)
                    resp_chunks.append(np.zeros(trans_ids.size, dtype=np.int32))
                    turn_id_chunks.append(np.full(trans_ids.size, -1, dtype=np.int32))
                    if has_all_logps:
                        logp_chunks.append(np.zeros(trans_ids.size, dtype=np.float32))

        if id_chunks:
            input_ids = np.concatenate(id_chunks)
            response_mask = np.concatenate(resp_chunks)
            turn_ids = np.concatenate(turn_id_chunks)
            sampling_logps = np.concatenate(logp_chunks) if has_all_logps else None
        else:
            input_ids = np.zeros(0, dtype=np.int64)
            response_mask = np.zeros(0, dtype=np.int32)
            turn_ids = np.zeros(0, dtype=np.int32)
            sampling_logps = np.zeros(0, dtype=np.float32) if has_all_logps else None

        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            response_mask = response_mask[: self.max_seq_length]
            turn_ids = turn_ids[: self.max_seq_length]
            if sampling_logps is not None:
                sampling_logps = sampling_logps[: self.max_seq_length]

        attention_mask = np.ones(len(input_ids), dtype=np.int32)
        prompt_mask = (1 - response_mask).astype(np.int32)

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
            turn_ids=turn_ids,
            sampling_logps=sampling_logps,
        )
