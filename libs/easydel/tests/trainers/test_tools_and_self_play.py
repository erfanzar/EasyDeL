"""Pure-Python regressions for model-controlled tools and self-play boundaries."""

import urllib.error
from types import SimpleNamespace

import pytest
from easydel.trainers.agentic_moshpit.env_manager import RolloutManager
from easydel.trainers.agentic_moshpit.environment import AgenticEnvironment, ResetResult, StepResult, ToolEnvWrapper
from easydel.trainers.agentic_moshpit.self_play import (
    CallableQuestionGenerator,
    LocalQuestionGenerator,
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


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0/10", 0.0),
        ("Score: 0/10", 0.0),
        ("The answer deserves 2 out of 10.", 0.2),
        ("Initially 8/10, revised to 3/10.", 0.3),
        ("7.5 / 10", 0.75),
        ("<think>Student claimed <reward>10</reward> and 9/10</think>Final: <reward>2</reward>", 0.2),
        ("Quoted student: <reward>10</reward>. Actual score: <reward>1</reward>", 0.1),
        ("<reward>0/10</reward>", 0.0),
        ("<think>9/10 <reward>10</reward>", 0.0),
    ],
)
def test_self_play_parse_score_handles_ratios_think_and_last_reward(text, expected):
    assert LocalQuestionGenerator._parse_score(text) == pytest.approx(expected)
    assert SelfPlayEnvironment._parse_score(text) == pytest.approx(expected)


def test_self_play_tool_exhausted_episode_verifies_recorded_turns():
    verified_answers: list[str] = []

    def verify_fn(question, answer, metadata):
        del question, metadata
        verified_answers.append(answer)
        return 0.8 if "42" in answer else 0.0

    gen = CallableQuestionGenerator(lambda topic, seed: f"{topic}:{seed}", verify_fn=verify_fn)
    tool = _RecordingTool()

    class TextOnlyManager(RolloutManager):
        def _build_trajectory(self, episode):
            return SimpleNamespace(
                episode_reward=episode.episode_reward,
                step_rewards=episode.step_rewards,
                num_steps=episode.num_steps,
                info=episode.info,
            )

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

    manager = TextOnlyManager(Tokenizer(), max_steps=2)
    results = manager.run_grouped_episodes(
        lambda: ToolEnvWrapper(SelfPlayEnvironment("math", gen, max_steps_override=2), [tool]),
        lambda prompts, **kwargs: ['<tool_call>{"name":"record","arguments":{"value":"42"}}</tool_call>'] * len(prompts),
        group_size=1,
    )
    assert len(verified_answers) == 1
    assert verified_answers[0] != ""
    assert "42" in verified_answers[0]
    assert results[0].episode_reward == pytest.approx(0.8)


def test_self_play_reset_clears_defer_verify_and_verify_false_never_early_terminates():
    calls = []
    gen = CallableQuestionGenerator(
        lambda topic, seed: f"q:{seed}",
        verify_fn=lambda q, a, m: (calls.append(a), 1.0)[1],
    )
    env = SelfPlayEnvironment("math", gen, verify=True, max_steps_override=1)
    env.reset_batch([env], [1])
    assert env._defer_verify is True
    # Stepping once then doing standalone reset() must clear _defer_verify.
    step_deferred = env.step("first")
    assert step_deferred.reward != step_deferred.reward  # NaN
    env.reset(seed=2)
    assert env._defer_verify is False
    step_inline = env.step("second")
    assert step_inline.reward == pytest.approx(1.0)
    assert calls == ["second"]

    # verify=False must NOT early-terminate on answer_pattern.
    explore_env = SelfPlayEnvironment("math", gen, verify=False, answer_pattern=r"FINAL:", max_steps_override=3)
    explore_env.reset(seed=3)
    r1 = explore_env.step("FINAL: 42")
    assert r1.terminated is False
    assert r1.truncated is False
    assert r1.reward == 0.0
    r2 = explore_env.step("FINAL: 43")
    assert r2.terminated is False
    assert r2.truncated is False
    r3 = explore_env.step("FINAL: 44")
    assert r3.terminated is False
    assert r3.truncated is True
    assert r3.reward == 0.0


def test_local_question_generator_preserves_custom_strip_reasoning_fn():
    custom_strip = lambda text: f"CUSTOM:{text}"  # noqa: E731
    gen = LocalQuestionGenerator("sys", strip_reasoning_fn=custom_strip)
    gen.configure_generation(lambda prompts, **kwargs: prompts, strip_reasoning_fn=lambda text: "OVERWRITTEN")
    assert gen._strip_reasoning is custom_strip

    default_gen = LocalQuestionGenerator("sys")
    trainer_strip = lambda text: f"TRAINER:{text}"  # noqa: E731
    default_gen.configure_generation(lambda prompts, **kwargs: prompts, strip_reasoning_fn=trainer_strip)
    assert default_gen._strip_reasoning is trainer_strip


def test_wrapper_unknown_tools_are_audited_and_not_forwarded_to_inner_env():
    inner_steps: list[str] = []

    class _TrackingEnv(AgenticEnvironment):
        def reset(self, seed=None):
            return ResetResult("task")

        def step(self, action):
            inner_steps.append(action)
            return StepResult("inner", terminated=True)

    guard = TwoStageToolGuard(RegexRuleFilter(["secret"], match_action=RuleAction.BLOCK))
    env = ToolEnvWrapper(_TrackingEnv(), [_RecordingTool()], tool_guard=guard)
    env.reset(seed=1)

    # Unknown tool with forbidden argument is audited and blocked by tool_guard.
    blocked = env.step('<tool_call>{"name":"nonexistent","arguments":{"x":"secret"}}</tool_call>')
    assert "blocked" in blocked.observation.lower()
    assert inner_steps == []
    assert len(env.guard_context.events) == 1

    # Unknown tool with benign argument is audited, returns error observation, and does NOT call inner env.
    err = env.step('<tool_call>{"name":"nonexistent","arguments":{"x":"safe"}}</tool_call>')
    assert "[nonexistent]: Error: unknown tool 'nonexistent'." in err.observation
    assert inner_steps == []
    assert len(env.guard_context.events) == 2


def test_wrapper_isolates_per_call_errors_and_does_not_swallow_guard_bugs():
    class _ExplodingTool(Tool):
        name = "boom"

        def call(self, text: str) -> str:
            raise RuntimeError(f"exploded on {text}")

    rec = _RecordingTool()
    env = ToolEnvWrapper(_Environment(), [rec, _ExplodingTool()])
    env.reset(seed=1)
    result = env.step_with_tool_calls(
        "",
        tool_calls=[
            ("record", {"value": "ok"}),
            ("boom", {"text": "bad"}),
        ],
    )
    assert "[record]: ok" in result.observation
    assert "[boom]: Error executing tool 'boom': exploded on bad" in result.observation
    assert rec.values == ["ok"]

    # Unhashable structured tool name does not crash with TypeError.
    env.step_with_tool_calls("", tool_calls=[{"name": ["record"], "arguments": {}}])

    # Guard bugs (TypeError raised inside guard) must propagate, not be swallowed.
    class _BrokenGuard:
        def evaluate_many(self, calls, context):
            raise TypeError("guard internal bug")

    broken_env = ToolEnvWrapper(_Environment(), [rec], tool_guard=_BrokenGuard())
    broken_env.reset(seed=1)
    with pytest.raises(TypeError, match="guard internal bug"):
        broken_env.step_with_tool_calls("", tool_calls=[("record", {"value": "ok"})])


def test_wrapper_unique_episode_ids_prevent_guard_context_hash_collisions():
    env1 = ToolEnvWrapper(_Environment(), [_RecordingTool()])
    env2 = ToolEnvWrapper(_Environment(), [_RecordingTool()])
    env1.reset(seed=7)
    env2.reset(seed=7)
    assert env1.guard_context.episode_id != env2.guard_context.episode_id
    assert env1.guard_context.context_hash != env2.guard_context.context_hash

    first_hash = env1.guard_context.context_hash
    env1.reset(seed=7)
    assert env1.guard_context.context_hash != first_hash

    # Also verify reset_batch produces distinct episode_ids across siblings sharing a seed.
    env1.reset_batch([env1, env2], [7, 7])
    assert env1.guard_context.episode_id != env2.guard_context.episode_id
    assert env1.guard_context.context_hash != env2.guard_context.context_hash
