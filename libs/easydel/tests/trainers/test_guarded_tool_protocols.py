"""Guarded tools preserve callable and serialized structural protocols (no JAX)."""

import json

import pytest
from easydel.trainers._shared.environment import AgenticEnvironment, ResetResult, StepResult, ToolEnvWrapper
from easydel.trainers._shared.tools import FunctionTool
from easydel.trainers.agentic_rollout.guards import RegexRuleFilter, RuleAction, TwoStageToolGuard
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import _CallableToolAdapter

pytestmark = pytest.mark.cpu_ok


class Environment(AgenticEnvironment):
    def reset(self, seed=None):
        return ResetResult("Use the echo tool")

    def step(self, action):
        return StepResult(action)


@pytest.mark.parametrize("arguments", ['{"text": "hello"}', '"hello"', "hello"])
@pytest.mark.parametrize("kind", ["function", "callable", "structural"])
@pytest.mark.parametrize("blocked", [False, True])
def test_guarded_wrapper_preserves_tool_argument_protocol(arguments, kind, blocked):
    executed = []

    def echo(text: str) -> str:
        executed.append(text)
        return f"echo={text}"

    if kind == "function":
        tool = FunctionTool(echo, tool_name="echo")
    elif kind == "callable":
        tool = _CallableToolAdapter(echo)
    else:

        class StructuralTool:
            name = "echo"

            @property
            def chat_schema(self):
                return {
                    "name": "echo",
                    "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
                }

            def execute(self, serialized):
                # Preserve bytes/formatting: external tools own this interface.
                assert serialized == arguments
                try:
                    value = json.loads(serialized)
                except json.JSONDecodeError:
                    value = serialized
                return echo(value["text"] if isinstance(value, dict) else value)

        tool = StructuralTool()

    guard = TwoStageToolGuard(RegexRuleFilter(["hello"], match_action=RuleAction.BLOCK)) if blocked else None
    wrapper = ToolEnvWrapper(Environment(), [tool], tool_guard=guard)
    wrapper.reset(seed=1)
    result = wrapper.step_with_tool_calls("", tool_calls=[("echo", arguments)])
    if blocked:
        assert executed == []
        assert "blocked by policy" in result.observation
        assert result.info["guard_events"]
    else:
        assert result.observation == "[echo]: echo=hello"
        assert executed == ["hello"]
    assert not result.terminated


def test_guarded_wrapper_keeps_duplicate_call_order_and_arguments():
    calls = []
    tool = FunctionTool(lambda text: calls.append(text) or text, tool_name="echo")
    wrapper = ToolEnvWrapper(Environment(), [tool])
    result = wrapper.step_with_tool_calls("", tool_calls=[("echo", "first"), ("echo", '"second"')])
    assert result.observation == "[echo]: first\n[echo]: second"
    assert calls == ["first", "second"]
