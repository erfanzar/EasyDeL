"""Pure-Python regressions for model-controlled tools and self-play boundaries."""

import urllib.error
from types import SimpleNamespace

import pytest
from easydel.trainers.agentic_moshpit.env_manager import RolloutManager
from easydel.trainers.agentic_moshpit.environment import AgenticEnvironment, ResetResult, StepResult, ToolEnvWrapper
from easydel.trainers.agentic_moshpit.self_play import (
    CallableQuestionGenerator,
    OpenAIQuestionGenerator,
    SelfPlayEnvironment,
)
from easydel.trainers.agentic_moshpit.tools import CalculatorTool, FileReadTool, Tool
from easydel.trainers.agentic_rollout.guards import RegexRuleFilter, RuleAction, TwoStageToolGuard

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("operation", ["generate", "verify"])
def test_api_failures_are_not_questions_or_rewards(monkeypatch, operation):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("http://unused", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fail)
    generator = OpenAIQuestionGenerator("http://unused/v1", "key", "model", "make questions")
    with pytest.raises(urllib.error.HTTPError):
        if operation == "generate":
            generator.generate("math")
        else:
            generator.verify("question", "answer")


def test_file_directory_restrictions_resolve_boundaries_and_symlinks(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "ok.txt").write_text("public")
    sibling = tmp_path / "allowed_secret"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("private")
    (allowed / "escape").symlink_to(sibling, target_is_directory=True)
    tool = FileReadTool(allowed_dirs=[str(allowed)])
    assert tool.call(str(allowed / "ok.txt")) == "public"
    assert tool.call(str(sibling / "secret.txt")).startswith("Error:")
    assert tool.call(str(allowed / "escape" / "secret.txt")).startswith("Error:")


def test_calculator_preserves_math_and_bounds_resource_bombs():
    tool = CalculatorTool(timeout=0.3)
    assert tool.call("factorial(6) + sum(range(5))") == "730"
    assert tool.call("sqrt(81) + 2**3") == "17.0"
    assert tool.call("().__class__.__base__.__subclasses__()").startswith("Error:")
    assert tool.call("sum(range(10**12))").startswith("Error:")
    assert tool.call("10**10**10").startswith("Error:")


class _Environment(AgenticEnvironment):
    def reset(self, seed=None):
        return ResetResult("task")

    def step(self, action):
        return StepResult("continue")

    @property
    def max_steps(self):
        return 2


class _RecordingTool(Tool):
    name = "record"

    def __init__(self):
        self.values = []

    def call(self, value: str) -> str:
        self.values.append(value)
        return value


def test_wrapper_dispatches_through_two_stage_guard():
    tool = _RecordingTool()
    guard = TwoStageToolGuard(RegexRuleFilter(["secret"], match_action=RuleAction.BLOCK))
    env = ToolEnvWrapper(_Environment(), [tool], tool_guard=guard)
    env.reset(seed=1)
    blocked = env.step_with_tool_calls("", tool_calls=[("record", {"value": "secret"})])
    assert "blocked" in blocked.observation.lower()
    assert tool.values == []
    allowed = env.step_with_tool_calls("", tool_calls=[("record", {"value": "public"})])
    assert allowed.observation == "[record]: public"
    assert tool.values == ["public"]
    assert len(env.guard_context.events) == 2


@pytest.mark.parametrize(
    "arguments,value", [('"public"', "public"), ("public", "public"), ("17", 17), ("[1,2]", [1, 2])]
)
def test_wrapper_preserves_legacy_arguments_before_guarding(arguments, value):
    tool = _RecordingTool()
    guard = TwoStageToolGuard(RegexRuleFilter(["secret"], match_action=RuleAction.BLOCK))
    env = ToolEnvWrapper(_Environment(), [tool], tool_guard=guard)
    env.reset()
    env.step_with_tool_calls("", tool_calls=[("record", arguments)])
    assert tool.values == [value]
    env.step_with_tool_calls("", tool_calls=[("record", '"secret"')])
    env.step_with_tool_calls("", tool_calls=[("record", "secret")])
    assert tool.values == [value]
    assert len(env.guard_context.events) == 3


def test_batched_self_play_hooks_preserve_generator_ownership():
    first = CallableQuestionGenerator(lambda topic, seed: f"A:{topic}:{seed}", lambda q, a, m: 0.25)
    second = CallableQuestionGenerator(lambda topic, seed: f"B:{topic}:{seed}", lambda q, a, m: 0.75)
    envs = [
        SelfPlayEnvironment("math", first, max_steps_override=1),
        SelfPlayEnvironment("code", second, max_steps_override=1),
    ]
    wrappers = [ToolEnvWrapper(env, []) for env in envs]
    resets = wrappers[0].reset_batch(wrappers, [1, 2])
    assert [result.observation for result in resets] == ["A:math:1", "B:code:2"]
    for env in wrappers:
        env.step("answer")
    results = wrappers[0].finalize_batch(wrappers)
    assert [result.reward for result in results] == [0.25, 0.75]
    assert [result.info["answer"] for result in results] == ["answer", "answer"]


def test_rollout_honors_wrapped_environment_step_limit():
    class TextOnlyManager(RolloutManager):
        def _build_trajectory(self, episode):
            # Avoid tokenization/array numerics: this test covers rollout control.
            return SimpleNamespace(num_steps=episode.num_steps, info=episode.info)

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

    manager = TextOnlyManager(Tokenizer(), max_steps=8)
    results = manager.run_grouped_episodes(
        lambda: ToolEnvWrapper(_Environment(), []),
        lambda prompts, **kwargs: ["answer"] * len(prompts),
        group_size=2,
    )
    assert [result.num_steps for result in results] == [2, 2]
    assert all(result.info["truncated"] for result in results)
