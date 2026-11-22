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

"""Shared, backend-independent schemas for agentic policy rollouts.

The schemas deliberately keep an assistant action and its resulting
observation in one :class:`AtomicInteractionStep`.  Consumers may render the
two messages separately, but cannot retain one half of a step during context
compaction.
"""

from __future__ import annotations

import hashlib
import json
import typing as tp
from dataclasses import dataclass, field
from enum import StrEnum


def _json_default(value: tp.Any) -> tp.Any:
    """Return a stable JSON representation for common schema-like objects."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _canonical_json(value: tp.Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def redacted_hash(value: tp.Any) -> str:
    """Hash sensitive data for telemetry without retaining its contents.

    Args:
        value: JSON-like value, text, or bytes to fingerprint.

    Returns:
        A ``sha256:``-prefixed hexadecimal digest.
    """
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True)
class NormalizedToolCall:
    """Canonical representation of a model-emitted tool call.

    ``arguments_json`` is canonical JSON and is excluded from ``repr`` to
    avoid accidentally putting credentials, file contents, or user data into
    logs.  Use :attr:`arguments` only at the execution boundary.

    Attributes:
        name: Tool dispatch name.
        arguments_json: Canonical JSON object containing tool arguments.
        call_id: Optional provider-assigned call identifier.
    """

    name: str
    arguments_json: str = field(default="{}", repr=False)
    call_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("tool call name must not be empty")
        try:
            arguments = json.loads(self.arguments_json)
        except json.JSONDecodeError as exc:
            raise ValueError("arguments_json must contain valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments_json", _canonical_json(arguments))

    @classmethod
    def create(
        cls,
        name: str,
        arguments: tp.Mapping[str, tp.Any] | str | None = None,
        *,
        call_id: str | None = None,
    ) -> NormalizedToolCall:
        """Create a normalized call from a mapping or JSON object string."""
        if arguments is None:
            parsed: tp.Any = {}
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("tool call arguments string must contain valid JSON") from exc
        else:
            parsed = dict(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("tool call arguments must be a JSON object")
        return cls(name=name, arguments_json=_canonical_json(parsed), call_id=call_id)

    @property
    def arguments(self) -> dict[str, tp.Any]:
        """Return a fresh mutable mapping for the tool executor."""
        return tp.cast(dict[str, tp.Any], json.loads(self.arguments_json))

    @property
    def fingerprint(self) -> str:
        """Return a redacted fingerprint of the dispatch-relevant payload."""
        return redacted_hash({"name": self.name, "arguments": json.loads(self.arguments_json)})

    @property
    def name_fingerprint(self) -> str:
        """Return a redacted fingerprint of the tool name."""
        return redacted_hash(self.name)

    def to_openai_dict(self) -> dict[str, tp.Any]:
        """Return the normalized OpenAI function-call representation."""
        function = {"name": self.name, "arguments": self.arguments_json}
        result: dict[str, tp.Any] = {"type": "function", "function": function}
        if self.call_id is not None:
            result["id"] = self.call_id
        return result


def _read_field(value: tp.Any, name: str, default: tp.Any = None) -> tp.Any:
    if isinstance(value, tp.Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def normalize_tool_call(call: tp.Any) -> NormalizedToolCall:
    """Normalize tuple, mapping, or object-style tool-call payloads.

    Supported inputs include ``(name, arguments)`` pairs, OpenAI-style
    ``{"function": ...}`` mappings, and objects exposing either
    ``name``/``arguments`` or ``function.name``/``function.arguments``.

    Args:
        call: Provider or parser-specific tool-call value.

    Returns:
        A canonical :class:`NormalizedToolCall`.

    Raises:
        TypeError: If the input shape is unsupported.
        ValueError: If the name or arguments are malformed.
    """
    if isinstance(call, NormalizedToolCall):
        return call

    if isinstance(call, tuple) and len(call) == 2:
        return NormalizedToolCall.create(str(call[0]), call[1])

    function = _read_field(call, "function")
    source = function if function is not None else call
    name = _read_field(source, "name")
    arguments = _read_field(source, "arguments", _read_field(source, "args", {}))
    call_id = _read_field(call, "id", _read_field(call, "call_id"))
    if name is None:
        raise TypeError(f"unsupported tool call shape: {type(call).__name__}")
    return NormalizedToolCall.create(str(name), arguments, call_id=None if call_id is None else str(call_id))


def normalize_tool_calls(calls: tp.Iterable[tp.Any] | None) -> tuple[NormalizedToolCall, ...]:
    """Normalize a sequence of tool calls while preserving order."""
    if calls is None:
        return ()
    return tuple(normalize_tool_call(call) for call in calls)


@dataclass(frozen=True, slots=True)
class RolloutMessage:
    """Message plus token-level training ownership.

    Attributes:
        role: Chat-template role such as ``system``, ``assistant``, or
            ``tool``.
        content: Message contents. Excluded from ``repr`` because it may
            contain secrets returned by a tool.
        token_ids: Tokens corresponding to this message when available.
        trainable_token_mask: One flag per token. Copied context uses all
            ``False`` while newly sampled policy tokens use all ``True``.
        name: Optional tool name or message name.
    """

    role: str
    content: str = field(repr=False)
    token_ids: tuple[int, ...] = field(default=(), repr=False)
    trainable_token_mask: tuple[bool, ...] = field(default=(), repr=False)
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("message role must not be empty")
        token_ids = tuple(int(token) for token in self.token_ids)
        mask = tuple(bool(value) for value in self.trainable_token_mask)
        if not mask and token_ids:
            mask = (False,) * len(token_ids)
        if len(mask) != len(token_ids):
            raise ValueError("trainable_token_mask must have one entry per token")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "trainable_token_mask", mask)

    def as_copied_context(self) -> RolloutMessage:
        """Return this message with every token marked non-trainable."""
        return RolloutMessage(
            role=self.role,
            content=self.content,
            token_ids=self.token_ids,
            trainable_token_mask=(False,) * len(self.token_ids),
            name=self.name,
        )


@dataclass(frozen=True, slots=True)
class AtomicInteractionStep:
    """One indivisible assistant-action and environment-observation pair."""

    assistant_action: str = field(repr=False)
    observation: str = field(repr=False)
    assistant_token_ids: tuple[int, ...] = field(default=(), repr=False)
    observation_token_ids: tuple[int, ...] = field(default=(), repr=False)
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    observation_role: str = "tool"
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: tp.Mapping[str, tp.Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assistant_token_ids", tuple(int(token) for token in self.assistant_token_ids))
        object.__setattr__(self, "observation_token_ids", tuple(int(token) for token in self.observation_token_ids))
        object.__setattr__(self, "tool_calls", normalize_tool_calls(self.tool_calls))
        if not self.observation_role:
            raise ValueError("observation_role must not be empty")

    def messages(self, *, train_assistant: bool = True) -> tuple[RolloutMessage, RolloutMessage]:
        """Render the atomic pair as two messages without losing ownership."""
        action_mask = (bool(train_assistant),) * len(self.assistant_token_ids)
        assistant = RolloutMessage(
            role="assistant",
            content=self.assistant_action,
            token_ids=self.assistant_token_ids,
            trainable_token_mask=action_mask,
        )
        observation = RolloutMessage(
            role=self.observation_role,
            content=self.observation,
            token_ids=self.observation_token_ids,
            trainable_token_mask=(False,) * len(self.observation_token_ids),
        )
        return assistant, observation


@dataclass(frozen=True, slots=True)
class GuardEvent:
    """Redacted audit event produced by the tool-call guard."""

    sequence: int
    step_index: int
    context_hash: str
    call_hash: str
    tool_name_hash: str
    decision: str
    stage: str
    reason_code: str
    confirmed: bool


@dataclass(slots=True)
class GuardContext:
    """Mutable per-episode state shared by guard and dispatcher.

    Raw episode intent and metadata are excluded from ``repr``. Guard events
    only contain hashes and bounded reason codes.
    """

    episode_id: str = field(repr=False)
    intent: str = field(default="", repr=False)
    metadata: dict[str, tp.Any] = field(default_factory=dict, repr=False)
    step_index: int = 0
    events: list[GuardEvent] = field(default_factory=list)

    @property
    def context_hash(self) -> str:
        """Return a redacted identifier for this episode."""
        return redacted_hash(self.episode_id)

    def append_event(
        self,
        call: NormalizedToolCall,
        *,
        decision: str,
        stage: str,
        reason_code: str,
        confirmed: bool,
    ) -> GuardEvent:
        """Append one privacy-preserving decision event."""
        event = GuardEvent(
            sequence=len(self.events),
            step_index=self.step_index,
            context_hash=self.context_hash,
            call_hash=call.fingerprint,
            tool_name_hash=call.name_fingerprint,
            decision=decision,
            stage=stage,
            reason_code=reason_code,
            confirmed=confirmed,
        )
        self.events.append(event)
        return event

    def advance_step(self) -> None:
        """Advance the shared step counter after an atomic interaction."""
        self.step_index += 1


class SegmentKind(StrEnum):
    """Kinds of trajectory segments emitted around a compaction boundary."""

    EXECUTION = "execution"
    SUMMARY = "summary"
    COPIED_TAIL = "copied_tail"


@dataclass(frozen=True, slots=True)
class TrainingSegment:
    """A sequence of messages with explicit token-level trainability."""

    segment_id: int
    kind: SegmentKind
    messages: tuple[RolloutMessage, ...]
    source_step_indices: tuple[int, ...] = ()

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Return all message tokens in segment order."""
        return tuple(token for message in self.messages for token in message.token_ids)

    @property
    def trainable_token_mask(self) -> tuple[bool, ...]:
        """Return the flattened token-ownership mask."""
        return tuple(value for message in self.messages for value in message.trainable_token_mask)


@dataclass(frozen=True, slots=True)
class PolicySummary:
    """Summary sampled from the same policy that owns the trajectory."""

    text: str = field(repr=False)
    token_ids: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("policy summary text must not be empty")
        if not self.token_ids:
            raise ValueError("policy summary must include generated token ids")
        object.__setattr__(self, "token_ids", tuple(int(token) for token in self.token_ids))


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    """Input passed to a same-policy summary callback.

    ``history_prefix_messages`` is the prefix of the causal context currently
    visible to the policy.  On the first compaction it normally contains the
    original system prompt and user instruction; after reconstruction it
    contains the original system prompt and the previous resume message.
    """

    compaction_index: int
    history_prefix_messages: tuple[RolloutMessage, ...]
    steps: tuple[AtomicInteractionStep, ...]
    summary_instruction: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.summary_instruction:
            raise ValueError("summary_instruction must not be empty")

    @property
    def messages(self) -> tuple[RolloutMessage, ...]:
        """Return copied history followed by a non-trainable summary request."""
        messages = [message.as_copied_context() for message in self.history_prefix_messages]
        for step in self.steps:
            messages.extend(message.as_copied_context() for message in step.messages(train_assistant=False))
        messages.append(
            RolloutMessage(
                role="user",
                content=self.summary_instruction,
            )
        )
        return tuple(messages)


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Result of one attempted context compaction."""

    applied: bool
    context_messages: tuple[RolloutMessage, ...]
    segments: tuple[TrainingSegment, ...] = ()
    retained_steps: tuple[AtomicInteractionStep, ...] = ()
    dropped_step_count: int = 0
    compaction_index: int | None = None
    failure_code: str | None = None
