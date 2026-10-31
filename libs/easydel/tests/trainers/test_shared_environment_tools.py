"""Host-only contracts for cross-trainer shared environment and tool APIs."""

import json

import pytest
from easydel.trainers import _shared
from easydel.trainers._shared import environment, tools
from easydel.trainers._shared.policy_losses import direct_importance_sampling_mask
from easydel.trainers.agentic_moshpit import environment as legacy_environment
from easydel.trainers.agentic_moshpit import tools as legacy_tools
from easydel.trainers.sdft_trainer import _zero_reward_func

pytestmark = pytest.mark.cpu_ok


class _TextEnvironment(legacy_environment.AgenticEnvironment):
    def reset(self, seed=None):
        return legacy_environment.ResetResult(f"task:{seed}")

    def step(self, action):
        return legacy_environment.StepResult(f"answer:{action}", reward=1.0, terminated=True)


def test_legacy_registered_tools_execute_through_shared_wrapper():
    @legacy_tools.register_tool("shared_registry_echo")
    class Echo(legacy_tools.Tool):
        def call(self, text: str, suffix: str = "!") -> str:
            """Echo text.

            text: Text to echo.
            suffix: Optional suffix.
            """
            return text + suffix

    try:
        tool = tools.make_tool("shared_registry_echo")
        wrapper = environment.ToolEnvWrapper(_TextEnvironment(), [tool])
        reset = wrapper.reset(seed=7)
        assert reset.observation == "task:7"
        assert isinstance(reset, environment.ResetResult)
        result = wrapper.step('<tool_call>{"name":"shared_registry_echo","arguments":{"text":"hello"}}</tool_call>')
        assert result.observation == "[shared_registry_echo]: hello!"
        assert result.reward == 0.0
        assert isinstance(result, legacy_environment.StepResult)
        assert tool.chat_schema["parameters"]["required"] == ["text"]
        assert tool.chat_schema["parameters"]["properties"]["text"]["type"] == "string"
        final = wrapper.step("done")
        assert final.observation == "answer:done"
        assert final.reward == 1.0
        assert final.terminated
    finally:
        # Remove only the temporary registration this test created.
        tools._TOOL_REGISTRY.pop("shared_registry_echo", None)


def test_shared_registry_has_all_builtins_and_legacy_stateful_tools():
    assert set(tools.list_tools()) >= {
        "python_code",
        "calculator",
        "bash",
        "regex",
        "json_processor",
        "wikipedia",
        "web_fetch",
        "file_read",
        "unit_converter",
        "notepad",
    }
    notepad = _shared.make_tool("notepad")
    assert isinstance(notepad, legacy_tools.NotepadTool)
    assert notepad.execute('{"action":"write","text":"first"}') == "Notes saved."
    assert notepad.execute('{"action":"append","text":"second"}') == "Notes appended."
    assert notepad.execute('{"action":"read"}') == "first\nsecond"
    assert tools.make_tool("notepad").call("read") == "(notepad is empty)"
    assert json.loads(tools.make_tool("regex").call(r"\d+", "a12 b34")) == ["12", "34"]
    assert tools.make_tool("json_processor").call("query", '{"x":["value"]}', "x.0") == "value"
    with pytest.raises(KeyError, match="not found"):
        tools.make_tool("unregistered_tool")


def test_shared_and_legacy_regex_parsers_preserve_nested_arguments():
    parser = _shared.create_tool_call_parser(r"regex:<call>(.*?)</call>")
    legacy_parser = legacy_environment.create_tool_call_parser(r"regex:<call>(.*?)</call>")
    text = '<call>{"name":"f","arguments":{"nested":{"values":[1,2]}}}</call><call>invalid</call>'
    expected = [("f", '{"nested": {"values": [1, 2]}}')]
    assert parser(text) == expected
    assert legacy_parser(text) == expected
    assert parser("no calls") == []


def test_zero_reward_hook_preserves_exported_sdft_alias():
    for completions in ([], ["first"], ["first", "second"]):
        expected = [0.0] * len(completions)
        assert _shared.zero_reward_func(completions, prompts=["ignored"]) == expected
        assert _zero_reward_func(completions, rewards=[1.0]) == expected
    assert _shared.zero_reward_func is _zero_reward_func


@pytest.mark.parametrize("low,high", [(0, 0.2), (-0.1, 0.2), (1.1, 0.2), (0.2, 0), (0.2, -0.1)])
def test_invalid_dis_widths_fail_before_any_array_computation(low, high):
    with pytest.raises(ValueError):
        direct_importance_sampling_mask(None, None, None, epsilon_low=low, epsilon_high=high)


def test_preference_and_sao_compatibility_paths_share_implementations():
    from easydel.trainers._shared import policy_losses
    from easydel.trainers.contrastive_preference_optimization_trainer import _fn as cpo
    from easydel.trainers.direct_preference_optimization_trainer import _fn as dpo
    from easydel.trainers.single_rollout_asynchronous_optimization_trainer import _fn as sao

    assert dpo.concatenated_inputs is _shared.concatenated_inputs
    assert cpo.concatenated_inputs is _shared.concatenated_inputs
    assert sao.direct_importance_sampling_mask is policy_losses.direct_importance_sampling_mask
    assert sao.sao_policy_loss is policy_losses.sao_policy_loss


def test_grpo_uses_shared_environment_and_tools():
    from easydel.trainers.group_relative_policy_optimization import grpo_trainer

    assert grpo_trainer.ToolEnvWrapper is environment.ToolEnvWrapper
    assert grpo_trainer.create_tool_call_parser is environment.create_tool_call_parser
    assert grpo_trainer.make_tool is tools.make_tool
    assert grpo_trainer.function_to_json is tools.function_to_json
