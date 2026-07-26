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

"""Two-stage tool-call guard for agentic reinforcement-learning rollouts.

The first stage is a cheap deterministic rule filter. Calls marked for review
are sent to an optional *batched* intent judge. This mirrors the online
anti-hacking control used in long-horizon agent training while keeping the
policy decision independent of any particular inference backend.
"""

from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass, field
from enum import StrEnum

from .schema import GuardContext, GuardEvent, NormalizedToolCall


class RuleAction(StrEnum):
    """Possible deterministic rule-filter outcomes."""

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class GuardAction(StrEnum):
    """Final guard outcomes."""

    ALLOW = "allow"
    BLOCK = "block"


class JudgeFailurePolicy(StrEnum):
    """Action to take when the intent judge is unavailable or malformed."""

    ALLOW = "allow"
    BLOCK = "block"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class RuleCheck:
    """Result returned by a deterministic tool-call rule."""

    action: RuleAction
    reason_code: str = "rule"


@dataclass(frozen=True, slots=True)
class JudgeCheck:
    """Result returned by an intent judge for one reviewed call."""

    action: GuardAction
    reason_code: str = "intent_judge"


@dataclass(frozen=True, slots=True)
class IntentReview:
    """Sensitive, ephemeral judge input.

    Raw intent and call arguments are intentionally excluded from ``repr`` and
    are never copied into :class:`~.schema.GuardEvent`.
    """

    call: NormalizedToolCall = field(repr=False)
    intent: str = field(repr=False)
    step_index: int


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Final decision and its redacted audit event."""

    action: GuardAction
    stage: str
    reason_code: str
    confirmed: bool
    event: GuardEvent

    @property
    def allowed(self) -> bool:
        """Whether the dispatcher may execute the call."""
        return self.action is GuardAction.ALLOW


class IntentJudgeError(RuntimeError):
    """Raised when judge failure policy is ``raise``."""


RuleFilter = tp.Callable[[NormalizedToolCall, GuardContext], RuleCheck]
IntentJudge = tp.Callable[[tp.Sequence[IntentReview]], tp.Sequence[JudgeCheck]]


def _reason_code(value: str, fallback: str) -> str:
    """Bound telemetry reason codes without retaining arbitrary text."""
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:64]
    return normalized or fallback


class RegexRuleFilter:
    """Mark calls matching configured regular expressions for review.

    Pattern contents and matched text are never retained in guard events;
    events only expose a stable ``pattern_<index>`` reason code.

    Args:
        patterns: Regular expressions matched against canonical
            ``"<tool-name>\\n<arguments-json>"`` text.
        match_action: Outcome for a match. ``review`` is the normal first
            stage for a two-stage guard; ``block`` supports unconditional
            deny rules.
        flags: Flags passed to :func:`re.compile`.
    """

    def __init__(
        self,
        patterns: tp.Sequence[str],
        *,
        match_action: RuleAction = RuleAction.REVIEW,
        flags: int = re.IGNORECASE,
    ) -> None:
        self._patterns = tuple(re.compile(pattern, flags) for pattern in patterns)
        self._match_action = RuleAction(match_action)

    def __call__(self, call: NormalizedToolCall, context: GuardContext) -> RuleCheck:
        del context
        candidate = f"{call.name}\n{call.arguments_json}"
        for index, pattern in enumerate(self._patterns):
            if pattern.search(candidate):
                return RuleCheck(self._match_action, f"pattern_{index}")
        return RuleCheck(RuleAction.ALLOW, "no_rule_match")


class TwoStageToolGuard:
    """Apply rule filtering followed by optional batched intent review."""

    def __init__(
        self,
        rule_filter: RuleFilter,
        *,
        intent_judge: IntentJudge | None = None,
        judge_failure_policy: JudgeFailurePolicy = JudgeFailurePolicy.BLOCK,
    ) -> None:
        self.rule_filter = rule_filter
        self.intent_judge = intent_judge
        self.judge_failure_policy = JudgeFailurePolicy(judge_failure_policy)

    def evaluate(
        self,
        call: NormalizedToolCall,
        context: GuardContext,
    ) -> GuardDecision:
        """Evaluate one call through the same batched code path."""
        return self.evaluate_many((call,), context)[0]

    def evaluate_many(
        self,
        calls: tp.Sequence[NormalizedToolCall],
        context: GuardContext,
    ) -> tuple[GuardDecision, ...]:
        """Evaluate calls and batch all second-stage reviews into one request."""
        if not calls:
            return ()

        provisional: list[tuple[GuardAction, str, str, bool] | None] = [None] * len(calls)
        review_indices: list[int] = []
        reviews: list[IntentReview] = []

        for index, call in enumerate(calls):
            check = self.rule_filter(call, context)
            if not isinstance(check, RuleCheck):
                raise TypeError("rule_filter must return RuleCheck")
            reason = _reason_code(check.reason_code, "rule")
            if check.action is RuleAction.ALLOW:
                provisional[index] = (GuardAction.ALLOW, "rule", reason, True)
            elif check.action is RuleAction.BLOCK:
                provisional[index] = (GuardAction.BLOCK, "rule", reason, True)
            else:
                review_indices.append(index)
                reviews.append(IntentReview(call=call, intent=context.intent, step_index=context.step_index))

        if reviews:
            judged = self._judge_reviews(reviews)
            for index, (action, stage, reason, confirmed) in zip(review_indices, judged, strict=True):
                provisional[index] = (action, stage, reason, confirmed)

        decisions: list[GuardDecision] = []
        for call, raw_decision in zip(calls, provisional, strict=True):
            if raw_decision is None:  # pragma: no cover - internal invariant
                raise AssertionError("missing guard decision")
            action, stage, reason, confirmed = raw_decision
            event = context.append_event(
                call,
                decision=action.value,
                stage=stage,
                reason_code=reason,
                confirmed=confirmed,
            )
            decisions.append(
                GuardDecision(
                    action=action,
                    stage=stage,
                    reason_code=reason,
                    confirmed=confirmed,
                    event=event,
                )
            )
        return tuple(decisions)

    def _judge_reviews(
        self,
        reviews: tp.Sequence[IntentReview],
    ) -> tuple[tuple[GuardAction, str, str, bool], ...]:
        if self.intent_judge is None:
            return self._handle_judge_failure(len(reviews), "judge_unavailable")

        try:
            checks = tuple(self.intent_judge(reviews))
            if len(checks) != len(reviews):
                raise ValueError("intent judge returned the wrong number of decisions")
            normalized = []
            for check in checks:
                if not isinstance(check, JudgeCheck):
                    raise TypeError("intent_judge must return JudgeCheck values")
                normalized.append(
                    (
                        GuardAction(check.action),
                        "intent_judge",
                        _reason_code(check.reason_code, "intent_judge"),
                        True,
                    )
                )
            return tuple(normalized)
        except Exception as exc:
            if self.judge_failure_policy is JudgeFailurePolicy.RAISE:
                raise IntentJudgeError("intent judge failed") from exc
            return self._handle_judge_failure(len(reviews), "judge_failure")

    def _handle_judge_failure(
        self,
        count: int,
        reason: str,
    ) -> tuple[tuple[GuardAction, str, str, bool], ...]:
        if self.judge_failure_policy is JudgeFailurePolicy.RAISE:
            raise IntentJudgeError(reason)
        action = GuardAction(self.judge_failure_policy.value)
        return tuple((action, "judge_failure_policy", reason, False) for _ in range(count))
