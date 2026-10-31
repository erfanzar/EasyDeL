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

"""Context compaction controls for long-horizon agentic RL rollouts."""

from __future__ import annotations

import typing as tp
from enum import StrEnum

from .schema import (
    AtomicInteractionStep,
    CompactionResult,
    PolicySummary,
    RolloutMessage,
    SegmentKind,
    SummaryRequest,
    TrainingSegment,
)

DEFAULT_RESUME_TEMPLATE = (
    "Earlier context was compacted into the following policy-generated summary:\n\n"
    "{summary}\n\nContinue from the retained complete interactions below."
)
DEFAULT_SUMMARY_INSTRUCTION = (
    "Summarize the trajectory so far for a future continuation. Preserve the task, "
    "important observations, completed actions, constraints, and unresolved next steps."
)


class CompactionFailurePolicy(StrEnum):
    """Behavior when compaction cannot produce a valid fitting context."""

    KEEP_CONTEXT = "keep_context"
    RAISE = "raise"


class CompactionError(RuntimeError):
    """Raised when compaction fails under the ``raise`` policy."""


PolicySummarizer = tp.Callable[[SummaryRequest], PolicySummary]
TokenCounter = tp.Callable[[tp.Sequence[RolloutMessage]], int]


class CompactionController:
    """Stateful controller for policy-generated trajectory compaction.

    A trigger is intentionally strict: compaction starts only when
    ``context_budget - current_tokens < compaction_threshold``. The same
    policy that generates actions must be supplied as ``policy_summarizer``;
    its sampled token IDs become a trainable summary segment.

    Args:
        context_budget: Maximum number of tokens in reconstructed context.
        compaction_threshold: Minimum desired remaining context capacity.
        max_compactions: Maximum successful compactions per episode.
        recent_pairs: Initial number of complete action-observation pairs to
            retain after the summary.
        train_summary_tokens: Whether policy-generated summary tokens belong
            to the actor loss.
        summary_instruction: Non-trainable user request appended to the full
            trajectory before the same policy samples its summary.
        resume_template: Template containing exactly the ``{summary}``
            substitution used for reconstructed context.
        failure_policy: Whether to keep the original context or raise when
            summarization/fitting fails.
    """

    def __init__(
        self,
        *,
        context_budget: int,
        compaction_threshold: int,
        max_compactions: int = 3,
        recent_pairs: int = 2,
        train_summary_tokens: bool = True,
        summary_instruction: str = DEFAULT_SUMMARY_INSTRUCTION,
        resume_template: str = DEFAULT_RESUME_TEMPLATE,
        failure_policy: CompactionFailurePolicy = CompactionFailurePolicy.RAISE,
    ) -> None:
        if context_budget <= 0:
            raise ValueError("context_budget must be positive")
        if compaction_threshold < 0:
            raise ValueError("compaction_threshold must be non-negative")
        if max_compactions < 0:
            raise ValueError("max_compactions must be non-negative")
        if recent_pairs < 0:
            raise ValueError("recent_pairs must be non-negative")
        if not summary_instruction:
            raise ValueError("summary_instruction must not be empty")
        if "{summary}" not in resume_template:
            raise ValueError("resume_template must contain {summary}")
        self.context_budget = context_budget
        self.compaction_threshold = compaction_threshold
        self.max_compactions = max_compactions
        self.recent_pairs = recent_pairs
        self.train_summary_tokens = bool(train_summary_tokens)
        self.summary_instruction = summary_instruction
        self.resume_template = resume_template
        self.failure_policy = CompactionFailurePolicy(failure_policy)
        self.compactions_done = 0

    def should_compact(self, current_tokens: int) -> bool:
        """Return the strict budget-threshold trigger decision."""
        if current_tokens < 0:
            raise ValueError("current_tokens must be non-negative")
        return self.context_budget - current_tokens < self.compaction_threshold

    def maybe_compact(
        self,
        *,
        original_system_messages: tp.Sequence[RolloutMessage],
        history_prefix_messages: tp.Sequence[RolloutMessage],
        steps: tp.Sequence[AtomicInteractionStep],
        current_tokens: int,
        policy_summarizer: PolicySummarizer,
        token_counter: TokenCounter,
    ) -> CompactionResult | None:
        """Compact a trajectory when the strict trigger fires.

        ``history_prefix_messages`` must contain the exact current causal
        prefix before ``steps``.  It therefore includes the original user
        instruction on the first compaction and a prior resume message after
        reconstruction.  ``original_system_messages`` is retained separately
        because only the original system prompt is copied into the newly
        reconstructed context.

        Returns ``None`` when no compaction is needed. Under
        ``keep_context``, failures return ``applied=False`` with a bounded
        ``failure_code`` and the original complete context.
        """
        if not self.should_compact(current_tokens):
            return None

        original_system = tuple(message.as_copied_context() for message in original_system_messages)
        history_prefix = tuple(message.as_copied_context() for message in history_prefix_messages)
        atomic_steps = tuple(steps)
        original_context = self._original_context(history_prefix, atomic_steps)
        if self.compactions_done >= self.max_compactions:
            return self._fail("max_compactions", original_context)

        compaction_index = self.compactions_done
        try:
            summary = policy_summarizer(
                SummaryRequest(
                    compaction_index=compaction_index,
                    history_prefix_messages=history_prefix,
                    steps=atomic_steps,
                    summary_instruction=self.summary_instruction,
                )
            )
            if not isinstance(summary, PolicySummary):
                raise TypeError("policy_summarizer must return PolicySummary")
        except Exception as exc:
            return self._fail("summary_failure", original_context, cause=exc)

        retained_count = min(self.recent_pairs, len(atomic_steps))
        fitted: tuple[tuple[RolloutMessage, ...], tuple[AtomicInteractionStep, ...]] | None = None
        while retained_count >= 0:
            retained = atomic_steps[len(atomic_steps) - retained_count :] if retained_count else ()
            candidate = self._reconstructed_context(original_system, summary, retained)
            try:
                token_count = int(token_counter(candidate))
            except Exception as exc:
                return self._fail("token_count_failure", original_context, cause=exc)
            if token_count < 0:
                return self._fail(
                    "invalid_token_count",
                    original_context,
                    cause=ValueError("token_counter returned a negative value"),
                )
            if token_count <= self.context_budget:
                fitted = candidate, retained
                break
            retained_count -= 1

        if fitted is None:
            return self._fail("context_does_not_fit", original_context)

        context_messages, retained_steps = fitted
        summary_message = RolloutMessage(
            role="assistant",
            content=summary.text,
            token_ids=summary.token_ids,
            trainable_token_mask=(self.train_summary_tokens,) * len(summary.token_ids),
        )
        summary_segment = TrainingSegment(
            segment_id=2 * compaction_index,
            kind=SegmentKind.SUMMARY,
            messages=(summary_message,),
        )

        tail_messages: list[RolloutMessage] = []
        source_indices: list[int] = []
        first_retained_index = len(atomic_steps) - len(retained_steps)
        for offset, step in enumerate(retained_steps):
            tail_messages.extend(message.as_copied_context() for message in step.messages(train_assistant=False))
            source_indices.append(first_retained_index + offset)
        tail_segment = TrainingSegment(
            segment_id=2 * compaction_index + 1,
            kind=SegmentKind.COPIED_TAIL,
            messages=tuple(tail_messages),
            source_step_indices=tuple(source_indices),
        )

        self.compactions_done += 1
        return CompactionResult(
            applied=True,
            context_messages=context_messages,
            segments=(summary_segment, tail_segment),
            retained_steps=retained_steps,
            dropped_step_count=len(atomic_steps) - len(retained_steps),
            compaction_index=compaction_index,
        )

    def _reconstructed_context(
        self,
        original_system: tuple[RolloutMessage, ...],
        summary: PolicySummary,
        retained_steps: tuple[AtomicInteractionStep, ...],
    ) -> tuple[RolloutMessage, ...]:
        messages = list(original_system)
        messages.append(
            RolloutMessage(
                role="user",
                content=self.resume_template.format(summary=summary.text),
            )
        )
        for step in retained_steps:
            messages.extend(message.as_copied_context() for message in step.messages(train_assistant=False))
        return tuple(messages)

    @staticmethod
    def _original_context(
        history_prefix: tuple[RolloutMessage, ...],
        steps: tuple[AtomicInteractionStep, ...],
    ) -> tuple[RolloutMessage, ...]:
        messages = list(history_prefix)
        for step in steps:
            messages.extend(message.as_copied_context() for message in step.messages(train_assistant=False))
        return tuple(messages)

    def _fail(
        self,
        code: str,
        original_context: tuple[RolloutMessage, ...],
        *,
        cause: Exception | None = None,
    ) -> CompactionResult:
        if self.failure_policy is CompactionFailurePolicy.RAISE:
            if cause is None:
                raise CompactionError(code)
            raise CompactionError(code) from cause
        return CompactionResult(
            applied=False,
            context_messages=original_context,
            failure_code=code,
        )
