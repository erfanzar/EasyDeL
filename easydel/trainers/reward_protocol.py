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
"""Class-based reward abstraction for the RL trainers.

``RewardProtocol`` is a subclassable alternative to passing a bare
``Callable[[prompts, completions], list[float]]`` to the policy-optimization
trainers (GRPO and its relatives, PPO, OnlineDPO, ...).  A subclass bundles the
reward's *name*, its *group reduction*, and its *scoring logic* in one place:

    class MyReward(RewardProtocol):
        reduction = "sum"

        def compute(self, *, prompt, completion, messages=None, **kwargs) -> float:
            return 1.0 if "the answer is" in completion.lower() else 0.0

Instances are plain reward callables: ``__call__`` returns ``list[float]`` and
declares ``**kwargs``, so the trainers' existing signature-filtering
(:func:`filter_kwargs_for_callable`) forwards the full batch payload they
already assemble.  This means a ``RewardProtocol`` drops into any trainer that
accepts ``reward_funcs`` with no change to the trainer's reward-call site.

Scoring is **per example** by default: :meth:`compute_batch` loops over the
completions and calls :meth:`compute` once per completion, passing only the
keyword arguments that ``compute``'s signature declares.  Reward *models* (or
any vectorized scorer) override :meth:`compute_batch` directly.

``reduction`` controls how a reward is reduced **over the ``num_generations``
group** when forming the group baseline in advantage estimation (GRPO family).
It is inert for trainers without a group baseline (PPO uses GAE; OnlineDPO uses
pairwise selection).
"""

from __future__ import annotations

import typing as tp

from .training_utils import filter_kwargs_for_callable

Reduction = tp.Literal["sum", "mean"]

# A single OpenAI-style chat message. The value type is intentionally broad:
# `content` may be a string or a list of multimodal parts, and tool/role fields
# carry their own shapes -- so this is the one place a heterogeneous leaf is honest.
ChatMessage = dict[str, tp.Any]
# A prompt or completion is either plain text or a list of chat messages.
PromptType = str | list[ChatMessage]
CompletionType = str | list[ChatMessage]
# A single structured tool call emitted by the model (name + arguments).
ToolCall = dict[str, tp.Any]

# Batch kwargs whose per-example element is exposed to compute() under a
# singular name (the plural batch field -> one item for this completion).
_PER_ITEM_RENAME = {
    "completions": "completion",
    "completion_texts": "completion_text",
    "raw_completions": "raw_completion",
    "prompts": "prompt",
    "prompt_texts": "prompt_text",
}


def as_messages(prompt: PromptType | None, completion: CompletionType) -> list[ChatMessage]:
    """Build an OpenAI chat-style message list from a prompt and completion.

    Handles both plain-text mode (``prompt``/``completion`` are strings) and
    conversational mode (``prompt`` is already a message list, ``completion`` a
    single-item assistant message list), so reward authors can rely on
    ``messages`` regardless of dataset format.

    Args:
        prompt: A prompt string, a list of ``{"role", "content"}`` messages, or
            ``None``.
        completion: A completion string or a list of assistant messages.

    Returns:
        A list of ``{"role", "content"}`` dicts ending in the assistant turn.
    """
    if isinstance(prompt, list):
        base = list(prompt)
    elif prompt is None:
        base = []
    else:
        base = [{"role": "user", "content": prompt}]
    if isinstance(completion, list):
        return base + list(completion)
    return [*base, {"role": "assistant", "content": completion}]


def text_of(value: PromptType | CompletionType | None) -> str:
    """Flatten a single prompt/completion to plain text.

    A single prompt/completion is a string in plain-text mode but a chat
    message list in conversational mode; this returns the text either way
    (joining message ``content`` fields), so per-example ``compute`` always
    receives a ``str``.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for message in value:
            content = message.get("content", "") if isinstance(message, dict) else message
            parts.append(content if isinstance(content, str) else str(content))
        return "\n".join(part for part in parts if part)
    return str(value)


class RewardProtocol:
    """Base class for class-based reward functions.

    Subclasses override :meth:`compute` (per example) or :meth:`compute_batch`
    (vectorized).  Class attributes configure naming and group reduction:

    Attributes:
        reduction: ``"sum"`` or ``"mean"`` -- how this reward is reduced over
            the ``num_generations`` group to form the advantage baseline in the
            GRPO family.  Defaults to ``"sum"``.  Ignored by trainers that do
            not use a group baseline (PPO, OnlineDPO).
        weight: Scalar multiplier for this reward in the (multi-reward) weighted
            sum.  The trainer builds its ``reward_weights`` from each reward's
            ``weight`` (defaults to ``1.0``), so weights live on the reward
            itself instead of a parallel config list.  An explicit
            ``reward_weights`` in the trainer config still overrides these.
        name: Optional display name used for logged metrics; falls back to the
            class name.
    """

    reduction: Reduction = "sum"
    weight: float = 1.0
    name: str | None = None

    def __init__(
        self,
        *,
        weight: float | None = None,
        reduction: Reduction | None = None,
        name: str | None = None,
    ) -> None:
        """Optionally override the class-level config per instance.

        Lets you write either ``class MyReward(RewardProtocol): weight = 0.3``
        or ``MyReward(weight=0.3)``.  ``None`` keeps the class default.

        Args:
            weight: Per-instance weight override for the weighted reward sum.
            reduction: Per-instance group-reduction override (``"sum"``/``"mean"``).
            name: Per-instance metric-name override.
        """
        if weight is not None:
            self.weight = float(weight)
        if reduction is not None:
            if reduction not in ("sum", "mean"):
                raise ValueError(f"reduction must be 'sum' or 'mean'; got {reduction!r}.")
            self.reduction = reduction
        if name is not None:
            self.name = name

    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        """Validate that each subclass is usable and correctly configured.

        Enforces, at class-definition time, that the subclass overrides at
        least one scoring method and that ``reduction`` is a recognized value.
        """
        super().__init_subclass__(**kwargs)
        if cls.compute is RewardProtocol.compute and cls.compute_batch is RewardProtocol.compute_batch:
            raise TypeError(f"{cls.__name__} must override `compute` or `compute_batch`.")
        if cls.reduction not in ("sum", "mean"):
            raise ValueError(f"{cls.__name__}.reduction must be 'sum' or 'mean'; got {cls.reduction!r}.")

    @property
    def __name__(self) -> str:
        """Return the metric-facing name (``name`` override or the class name)."""
        return self.name or type(self).__name__

    def compute(
        self,
        *,
        prompt: str | None = None,
        completion: str | None = None,
        messages: list[ChatMessage] | None = None,
        completion_text: str | None = None,
        prompt_text: str | None = None,
        raw_completion: str | None = None,
        raw_text: str | None = None,
        reasoning: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
        truncated: bool | None = None,
        completion_length: int | None = None,
        completion_ids: list[int] | None = None,
        max_length: int | None = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> float | None:
        """Score a single completion.

        Override this for the common, per-example case.  Every argument is
        optional -- declare only the ones you need; the rest are filtered out
        automatically.  ``**kwargs`` absorbs anything else forwarded by the
        trainer (e.g. ``environment_*`` fields or extra dataset columns).

        Args:
            prompt: The prompt for this example, as plain text (chat prompts are
                flattened to their message contents).
            completion: The generated completion, as plain text (same string as
                ``completion_text``).  Use ``messages`` / ``tool_calls`` /
                ``reasoning`` for the structured form.
            messages: The full ``prompt + completion`` as a chat message list.
            completion_text: Decoded completion text (special tokens stripped).
            prompt_text: Decoded prompt text, when the trainer provides it.
            raw_completion: Completion text before reasoning/tool separation.
            raw_text: Raw completion text without special-token stripping.
            reasoning: Extracted reasoning/thinking text, if the model emitted any.
            tool_calls: Structured tool calls emitted in this completion.
            finish_reason: Generation stop reason -- ``"stop"`` / ``"length"`` /
                ``"eos_token"`` / ``"abort"`` (``None`` if the backend omits it).
            truncated: ``True`` when generation hit the length cap
                (``finish_reason == "length"``); ``None`` if unknown.
            completion_length: Number of generated tokens.
            completion_ids: Generated token ids (padding trimmed).
            max_length: Max sequence length used during generation.
            batch: The full batch dict (for gold answers / side-channels).
            **kwargs: Any other forwarded fields (e.g. ``environment_*``).

        Returns:
            A float reward, or ``None`` to mark this example as unscored
            (treated as ``NaN`` and excluded from aggregation).
        """
        raise NotImplementedError("Subclasses must override `compute` or `compute_batch`.")

    def compute_batch(
        self,
        *,
        prompts: tp.Sequence[PromptType] | None,
        completions: tp.Sequence[CompletionType],
        completion_texts: tp.Sequence[str] | None = None,
        prompt_texts: tp.Sequence[str] | None = None,
        raw_completions: tp.Sequence[CompletionType] | None = None,
        raw_text: tp.Sequence[str] | None = None,
        reasoning: tp.Sequence[str | None] | None = None,
        tool_calls: tp.Sequence[list[ToolCall] | None] | None = None,
        finish_reason: tp.Sequence[str | None] | None = None,
        truncated: tp.Sequence[bool | None] | None = None,
        completion_length: tp.Sequence[int] | None = None,
        completion_ids: tp.Sequence[list[int]] | None = None,
        max_length: int | None = None,
        batch: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ) -> list[float | None]:
        """Score a whole batch of completions.

        The default implementation dispatches to :meth:`compute` once per
        completion (flattening prompt/completion to text and splitting every
        list-aligned field to its per-example value).  Override this directly
        for vectorized scorers (e.g. a reward model that runs a single batched
        forward pass).

        The arguments mirror :meth:`compute`, but each is the **batch** form: a
        length-``N`` sequence (or a scalar/dict for ``max_length`` / ``batch``).
        ``**kwargs`` absorbs any other forwarded fields (e.g. ``environment_*``).

        Args:
            prompts: Per-completion prompts (length ``N``), or ``None``.
            completions: Per-completion outputs (length ``N``).
            completion_texts: Decoded completion texts.
            prompt_texts: Decoded prompt texts.
            raw_completions: Completions before reasoning/tool separation.
            raw_text: Raw completion texts (special tokens kept).
            reasoning: Per-completion reasoning texts.
            tool_calls: Per-completion structured tool calls.
            finish_reason: Per-completion generation stop reasons.
            truncated: Per-completion length-cap flags.
            completion_length: Per-completion generated-token counts.
            completion_ids: Per-completion generated token ids.
            max_length: Max sequence length used during generation.
            batch: The full batch dict.
            **kwargs: Any other forwarded fields (e.g. ``environment_*``).

        Returns:
            A list of ``N`` rewards (floats or ``None``).
        """
        completions = list(completions)
        n = len(completions)
        prompts = list(prompts) if prompts is not None else [None] * n
        # Reassemble the per-example-indexable payload (named fields + overflow).
        batch_fields = {
            "completion_texts": completion_texts,
            "prompt_texts": prompt_texts,
            "raw_completions": raw_completions,
            "raw_text": raw_text,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "truncated": truncated,
            "completion_length": completion_length,
            "completion_ids": completion_ids,
            "max_length": max_length,
            "batch": batch,
            **kwargs,
        }
        results: list[float | None] = []
        for i in range(n):
            prompt = prompts[i] if i < len(prompts) else None
            completion = completions[i]
            per_item = self._index_batch_field(batch_fields, i, n)
            call_kwargs = {
                # `prompt`/`completion` are flattened to plain text for the
                # single-example view; `messages` keeps the structured chat form.
                "prompt": text_of(prompt),
                "completion": text_of(completion),
                "messages": as_messages(prompt, completion),
                **per_item,
            }
            results.append(self.compute(**filter_kwargs_for_callable(self.compute, call_kwargs)))
        return results

    @staticmethod
    def _index_batch_field(kwargs: tp.Mapping[str, tp.Any], i: int, n: int) -> dict[str, tp.Any]:
        """Project batch-level kwargs onto example ``i``.

        List/tuple values of length ``n`` are indexed to their ``i``-th element
        (renamed to a singular form where applicable); everything else (scalars,
        the ``batch`` dict, ``max_length``) is passed through unchanged.
        """
        out: dict[str, tp.Any] = {}
        for key, value in kwargs.items():
            name = _PER_ITEM_RENAME.get(key, key)
            if isinstance(value, (list, tuple)) and len(value) == n:
                out[name] = value[i]
            else:
                out[name] = value
        return out

    def __call__(self, **batch_kwargs: tp.Any) -> list[float | None]:
        """Invoke the reward as a trainer ``reward_func``.

        Forwards the (signature-filtered) batch payload to
        :meth:`compute_batch` and returns one reward per completion.
        """
        return self.compute_batch(**filter_kwargs_for_callable(self.compute_batch, batch_kwargs))
