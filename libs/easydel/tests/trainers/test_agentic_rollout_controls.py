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

from types import SimpleNamespace

import pytest
from easydel.trainers.agentic_rollout import (
    BLOCKED_TOOL_OBSERVATION,
    AtomicInteractionStep,
    CompactionController,
    CompactionError,
    CompactionFailurePolicy,
    GuardAction,
    GuardContext,
    IntentJudgeError,
    JudgeCheck,
    JudgeFailurePolicy,
    NormalizedToolCall,
    PolicySummary,
    RegexRuleFilter,
    RolloutMessage,
    SegmentKind,
    ToolDispatcher,
    TwoStageToolGuard,
    normalize_tool_call,
)


def _step(index: int) -> AtomicInteractionStep:
    return AtomicInteractionStep(
        assistant_action=f"action-{index}",
        observation=f"observation-{index}",
        assistant_token_ids=(10 + index, 20 + index),
        observation_token_ids=(30 + index, 40 + index),
    )


def test_tool_calls_normalize_provider_shapes_and_canonicalize_arguments():
    tuple_call = normalize_tool_call(("lookup", '{"z":2,"a":1}'))
    mapping_call = normalize_tool_call(
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": {"a": 1, "z": 2}},
        }
    )
    object_call = normalize_tool_call(
        SimpleNamespace(
            id="call-2",
            function=SimpleNamespace(name="lookup", arguments='{"a":1,"z":2}'),
        )
    )

    assert tuple_call.arguments_json == '{"a":1,"z":2}'
    assert tuple_call.arguments == {"a": 1, "z": 2}
    assert mapping_call.arguments_json == tuple_call.arguments_json
    assert object_call.arguments_json == tuple_call.arguments_json
    assert mapping_call.to_openai_dict()["function"]["arguments"] == '{"a":1,"z":2}'
    assert tuple_call.fingerprint == object_call.fingerprint


def test_two_stage_guard_batches_reviews_and_dispatches_only_allowed_calls():
    executions: list[str] = []
    judge_batches: list[tuple[NormalizedToolCall, ...]] = []

    def lookup(value: str) -> str:
        executions.append(value)
        return f"result:{value}"

    def judge(reviews):
        judge_batches.append(tuple(review.call for review in reviews))
        return [
            JudgeCheck(
                GuardAction.BLOCK if review.call.arguments["value"] == "secret-token" else GuardAction.ALLOW,
                "intent_confirmed",
            )
            for review in reviews
        ]

    guard = TwoStageToolGuard(
        RegexRuleFilter((r"secret", r"review-me")),
        intent_judge=judge,
        judge_failure_policy=JudgeFailurePolicy.RAISE,
    )
    dispatcher = ToolDispatcher({"lookup": lookup}, guard=guard)
    context = GuardContext(episode_id="private-episode-id", intent="private user objective")

    results = dispatcher.dispatch_many(
        (
            ("lookup", {"value": "ordinary"}),
            ("lookup", {"value": "review-me-but-allowed"}),
            ("lookup", {"value": "secret-token"}),
        ),
        context,
    )

    assert len(judge_batches) == 1
    assert [call.arguments["value"] for call in judge_batches[0]] == [
        "review-me-but-allowed",
        "secret-token",
    ]
    assert executions == ["ordinary", "review-me-but-allowed"]
    assert [result.blocked for result in results] == [False, False, True]
    assert results[2].observation == BLOCKED_TOOL_OBSERVATION
    assert results[2].terminated is False
    assert results[2].truncated is False
    assert results[2].guard_decision is not None
    assert results[2].guard_decision.confirmed is True

    assert len(context.events) == 3
    assert context.events[0].stage == "rule"
    assert context.events[1].stage == "intent_judge"
    assert context.events[2].decision == "block"
    audit_text = repr(context.events) + repr(results) + repr(context)
    assert "secret-token" not in audit_text
    assert "private user objective" not in audit_text
    assert "private-episode-id" not in audit_text
    assert all(event.call_hash.startswith("sha256:") for event in context.events)
    assert all(event.tool_name_hash.startswith("sha256:") for event in context.events)


@pytest.mark.parametrize(
    ("failure_policy", "blocked", "executed"),
    (
        (JudgeFailurePolicy.BLOCK, True, False),
        (JudgeFailurePolicy.ALLOW, False, True),
    ),
)
def test_intent_judge_failure_policy_is_explicit(failure_policy, blocked, executed):
    calls: list[str] = []

    def broken_judge(reviews):
        del reviews
        raise RuntimeError("judge transport failed with sensitive payload")

    guard = TwoStageToolGuard(
        RegexRuleFilter((r".*",)),
        intent_judge=broken_judge,
        judge_failure_policy=failure_policy,
    )
    dispatcher = ToolDispatcher(
        {"lookup": lambda value: calls.append(value) or "ok"},
        guard=guard,
    )
    context = GuardContext(episode_id="episode")

    result = dispatcher.dispatch(("lookup", {"value": "credential"}), context)

    assert result.blocked is blocked
    assert bool(calls) is executed
    assert context.events[0].stage == "judge_failure_policy"
    assert context.events[0].confirmed is False
    assert "credential" not in repr(context.events)


def test_intent_judge_raise_policy_preserves_original_failure_as_cause():
    def broken_judge(reviews):
        del reviews
        raise RuntimeError("offline")

    dispatcher = ToolDispatcher(
        {"lookup": lambda value: value},
        guard=TwoStageToolGuard(
            RegexRuleFilter((r".*",)),
            intent_judge=broken_judge,
            judge_failure_policy=JudgeFailurePolicy.RAISE,
        ),
    )

    with pytest.raises(IntentJudgeError) as exc_info:
        dispatcher.dispatch(("lookup", {"value": "x"}), GuardContext("episode"))

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_dispatch_step_keeps_action_and_blocked_observation_atomic_and_continuing():
    executed = False

    def executor(**kwargs):
        nonlocal executed
        del kwargs
        executed = True
        return "should not run"

    dispatcher = ToolDispatcher(
        {"shell": executor},
        guard=TwoStageToolGuard(
            RegexRuleFilter((r".*",), match_action="block"),
        ),
    )
    context = GuardContext("episode")

    step = dispatcher.dispatch_step(
        "assistant action",
        (("shell", {"command": "sensitive"}),),
        context,
        assistant_token_ids=(1, 2),
        encode_observation=lambda text: (9,) * len(text.split()),
    )

    assert executed is False
    assert step.observation == BLOCKED_TOOL_OBSERVATION
    assert step.terminated is False
    assert context.step_index == 1
    assert len(context.events) == 1
    action_message, observation_message = step.messages()
    assert action_message.trainable_token_mask == (True, True)
    assert not any(observation_message.trainable_token_mask)


def test_compaction_trigger_is_strict_at_threshold_boundary():
    controller = CompactionController(
        context_budget=10,
        compaction_threshold=3,
    )

    assert controller.should_compact(7) is False
    assert controller.should_compact(8) is True


def test_compaction_reduces_complete_tail_pairs_and_marks_token_ownership():
    system = (RolloutMessage(role="system", content="system", token_ids=(1, 2)),)
    user = RolloutMessage(role="user", content="original task", token_ids=(3,))
    steps = tuple(_step(index) for index in range(3))
    requests = []
    summary_instruction = "Summarize all completed work and unresolved constraints."

    def summarize(request):
        requests.append(request)
        return PolicySummary("summary", (70, 71, 72))

    def count_tokens(messages):
        # system=2, resume=3, each complete pair=4. Two pairs do not fit
        # budget 9, while one complete pair fits exactly.
        total = 0
        for message in messages:
            if message.role == "user" and "policy-generated summary" in message.content:
                total += 3
            else:
                total += len(message.token_ids)
        return total

    controller = CompactionController(
        context_budget=9,
        compaction_threshold=3,
        recent_pairs=2,
        summary_instruction=summary_instruction,
    )
    result = controller.maybe_compact(
        original_system_messages=system,
        history_prefix_messages=(*system, user),
        steps=steps,
        current_tokens=7,
        policy_summarizer=summarize,
        token_counter=count_tokens,
    )

    assert result is not None and result.applied
    assert len(requests) == 1
    assert requests[0].steps == steps
    assert [message.role for message in requests[0].messages[:2]] == ["system", "user"]
    assert requests[0].messages[1].content == "original task"
    assert requests[0].messages[-1].role == "user"
    assert requests[0].messages[-1].content == summary_instruction
    assert requests[0].messages[-1].trainable_token_mask == ()
    assert all(not any(message.trainable_token_mask) for message in requests[0].messages)
    assert result.retained_steps == (steps[-1],)
    assert result.dropped_step_count == 2
    assert [message.role for message in result.context_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert result.context_messages[-2].content == "action-2"
    assert result.context_messages[-1].content == "observation-2"
    assert all(not any(message.trainable_token_mask) for message in result.context_messages)

    summary_segment, tail_segment = result.segments
    assert summary_segment.kind is SegmentKind.SUMMARY
    assert summary_segment.token_ids == (70, 71, 72)
    assert summary_segment.trainable_token_mask == (True, True, True)
    assert tail_segment.kind is SegmentKind.COPIED_TAIL
    assert tail_segment.source_step_indices == (2,)
    assert tail_segment.token_ids == (12, 22, 32, 42)
    assert tail_segment.trainable_token_mask == (False, False, False, False)
    assert controller.compactions_done == 1


def test_repeated_compaction_summarizes_current_prefix_but_reconstructs_original_system():
    system = (RolloutMessage(role="system", content="original system", token_ids=(1,)),)
    original_user = RolloutMessage(role="user", content="original task", token_ids=(2,))
    requests = []

    def summarize(request):
        requests.append(request)
        return PolicySummary(f"summary-{request.compaction_index}", (70 + request.compaction_index,))

    controller = CompactionController(
        context_budget=20,
        compaction_threshold=5,
        recent_pairs=0,
    )
    first = controller.maybe_compact(
        original_system_messages=system,
        history_prefix_messages=(*system, original_user),
        steps=(_step(0),),
        current_tokens=16,
        policy_summarizer=summarize,
        token_counter=lambda messages: 2,
    )
    assert first is not None and first.applied
    assert [message.content for message in requests[0].messages[:2]] == ["original system", "original task"]
    assert [message.role for message in first.context_messages] == ["system", "user"]
    assert "summary-0" in first.context_messages[1].content
    assert "original task" not in first.context_messages[1].content

    second = controller.maybe_compact(
        original_system_messages=system,
        history_prefix_messages=first.context_messages,
        steps=(_step(1),),
        current_tokens=16,
        policy_summarizer=summarize,
        token_counter=lambda messages: 2,
    )
    assert second is not None and second.applied
    assert [message.role for message in requests[1].messages[:2]] == ["system", "user"]
    assert "summary-0" in requests[1].messages[1].content
    assert all(message.content != "original task" for message in requests[1].messages)
    assert [message.role for message in second.context_messages] == ["system", "user"]
    assert second.context_messages[0].content == "original system"
    assert "summary-1" in second.context_messages[1].content


def test_compaction_maximum_and_summary_failure_follow_keep_context_policy():
    system = (RolloutMessage(role="system", content="system", token_ids=(1,)),)
    steps = (_step(0),)
    controller = CompactionController(
        context_budget=20,
        compaction_threshold=5,
        max_compactions=1,
        recent_pairs=0,
        failure_policy=CompactionFailurePolicy.KEEP_CONTEXT,
    )

    first = controller.maybe_compact(
        original_system_messages=system,
        history_prefix_messages=system,
        steps=steps,
        current_tokens=16,
        policy_summarizer=lambda request: PolicySummary("summary", (7,)),
        token_counter=lambda messages: 2,
    )
    assert first is not None and first.applied

    second = controller.maybe_compact(
        original_system_messages=system,
        history_prefix_messages=system,
        steps=steps,
        current_tokens=16,
        policy_summarizer=lambda request: pytest.fail("summarizer must not run past maximum"),
        token_counter=lambda messages: 2,
    )
    assert second is not None and not second.applied
    assert second.failure_code == "max_compactions"
    assert [message.role for message in second.context_messages] == ["system", "assistant", "tool"]

    failing_controller = CompactionController(
        context_budget=20,
        compaction_threshold=5,
        failure_policy=CompactionFailurePolicy.KEEP_CONTEXT,
    )
    failed = failing_controller.maybe_compact(
        original_system_messages=system,
        history_prefix_messages=system,
        steps=steps,
        current_tokens=16,
        policy_summarizer=lambda request: (_ for _ in ()).throw(RuntimeError("failed")),
        token_counter=lambda messages: 2,
    )
    assert failed is not None and not failed.applied
    assert failed.failure_code == "summary_failure"
    assert failing_controller.compactions_done == 0


def test_compaction_can_exclude_summary_tokens_for_the_published_ablation():
    controller = CompactionController(
        context_budget=20,
        compaction_threshold=5,
        recent_pairs=0,
        train_summary_tokens=False,
    )
    result = controller.maybe_compact(
        original_system_messages=(RolloutMessage(role="system", content="system", token_ids=(1,)),),
        history_prefix_messages=(
            RolloutMessage(role="system", content="system", token_ids=(1,)),
            RolloutMessage(role="user", content="task", token_ids=(2,)),
        ),
        steps=(_step(0),),
        current_tokens=16,
        policy_summarizer=lambda request: PolicySummary("summary", (7, 8)),
        token_counter=lambda messages: 3,
    )

    assert result is not None and result.applied
    assert result.segments[0].kind is SegmentKind.SUMMARY
    assert result.segments[0].trainable_token_mask == (False, False)


def test_compaction_raise_policy_rejects_context_that_cannot_fit():
    controller = CompactionController(
        context_budget=5,
        compaction_threshold=2,
        recent_pairs=2,
        failure_policy=CompactionFailurePolicy.RAISE,
    )

    with pytest.raises(CompactionError, match="context_does_not_fit"):
        controller.maybe_compact(
            original_system_messages=(RolloutMessage(role="system", content="system"),),
            history_prefix_messages=(
                RolloutMessage(role="system", content="system"),
                RolloutMessage(role="user", content="task"),
            ),
            steps=(_step(0), _step(1)),
            current_tokens=4,
            policy_summarizer=lambda request: PolicySummary("summary", (7,)),
            token_counter=lambda messages: 100,
        )
