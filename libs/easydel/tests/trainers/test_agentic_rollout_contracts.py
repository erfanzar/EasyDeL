"""Host-only regression coverage for structured trajectories and episode ownership."""

import json
import time
import typing as tp
from types import SimpleNamespace

import pytest
from easydel.trainers._shared.tools import FunctionTool, NotepadTool, RegexTool, function_to_json
from easydel.trainers.agentic_moshpit.agentic_moshpit_trainer import AgenticMoshPitTrainer
from easydel.trainers.agentic_moshpit.env_manager import RolloutManager
from easydel.trainers.agentic_moshpit.environment import AgenticEnvironment, ResetResult, StepResult, ToolEnvWrapper
from easydel.trainers.agentic_moshpit.self_play import (
    CallableQuestionGenerator,
    GeneratedQuestion,
    LocalQuestionGenerator,
    OpenAIQuestionGenerator,
    QuestionGenerator,
    SelfPlayEnvironment,
)

pytestmark = pytest.mark.cpu_ok


class TextEnvironment(AgenticEnvironment):
    def reset(self, seed=None):
        return ResetResult("task")

    def step(self, action):
        return StepResult("", reward=1.0, terminated=True)


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = ""
        for message in messages:
            text += message["role"] + ":" + message["content"]
            for call in message.get("tool_calls", []):
                function = call["function"]
                text += (
                    "<invoke>" + function["name"] + ":" + json.dumps(function["arguments"], sort_keys=True) + "</invoke>"
                )
            text += "\n"
        return text + ("assistant:" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def test_structured_tool_rollout_training_history_and_credit_match_sampling():
    prompts_seen = []

    def generate(prompts, **kwargs):
        prompts_seen.extend(prompts)
        if len(prompts_seen) == 1:
            generate.last_generation_metadata = [
                {
                    "raw_text": "",
                    "text": "",
                    "tool_calls": [{"type": "function", "function": {"name": "echo", "arguments": '{"text":"hello"}'}}],
                }
            ]
            return [""]
        generate.last_generation_metadata = [{}]
        return ["done"]

    tool = FunctionTool(lambda text: text, tool_name="echo")
    manager = RolloutManager(CharacterTokenizer(), max_steps=2)
    trajectory = manager.run_grouped_episodes(lambda: ToolEnvWrapper(TextEnvironment(), [tool]), generate, group_size=1)[
        0
    ]
    text = "".join(chr(int(token)) for token in trajectory.input_ids)
    assert text.startswith(prompts_seen[1])
    invocation = '<invoke>echo:{"text": "hello"}</invoke>'
    assert invocation in prompts_seen[1]
    invocation_start = text.index(invocation)
    assert trajectory.response_mask[invocation_start : invocation_start + len(invocation)].tolist() == [1] * len(
        invocation
    )
    observation_start = text.index("tool:[echo]: hello")
    assert trajectory.response_mask[observation_start : observation_start + len("tool:[echo]: hello")].tolist() == [
        0
    ] * len("tool:[echo]: hello")
    assert text.endswith("assistant:done\n")
    assert trajectory.episode_reward == 1.0


@pytest.mark.parametrize("reasoning", ["", "<think>plan</think>"])
def test_raw_tool_rollout_trains_on_generated_text_not_structured_rendering(reasoning):
    prompts_seen = []
    raw_text = reasoning + '<tool_call>{"name":"echo","arguments":{"text":"hello"}}</tool_call>'
    tool_calls = [{"function": {"name": "echo", "arguments": {"text": "hello"}}}]

    def generate(prompts, **kwargs):
        prompts_seen.extend(prompts)
        if len(prompts_seen) == 1:
            generate.last_generation_metadata = [{"raw_text": raw_text, "text": "Need tool", "tool_calls": tool_calls}]
            return ["Need tool"]
        generate.last_generation_metadata = [{}]
        return ["done"]

    tool = FunctionTool(lambda text: text, tool_name="echo")
    manager = RolloutManager(CharacterTokenizer(), max_steps=2)
    trajectory = manager.run_grouped_episodes(lambda: ToolEnvWrapper(TextEnvironment(), [tool]), generate, group_size=1)[
        0
    ]
    text = "".join(chr(int(token)) for token in trajectory.input_ids)
    assert len(prompts_seen) == 2
    assert '<invoke>echo:{"text": "hello"}</invoke>' in prompts_seen[1]
    assert text == "user:task\nassistant:" + raw_text + "\ntool:[echo]: hello\nassistant:done\n"
    assert "<invoke>" not in text
    raw_start = text.index(raw_text)
    assert trajectory.response_mask[raw_start : raw_start + len(raw_text)].tolist() == [1] * len(raw_text)
    observation_start = text.index("tool:[echo]: hello")
    assert trajectory.response_mask[observation_start : observation_start + len("tool:[echo]: hello")].tolist() == [
        0
    ] * len("tool:[echo]: hello")
    assistant_turn = trajectory.turns[1]
    assert assistant_turn.content == assistant_turn.raw_content == raw_text
    assert assistant_turn.visible_content == "Need tool"
    assert assistant_turn.tool_calls == tool_calls
    assert trajectory.episode_reward == 1.0


@pytest.mark.parametrize("backend", ["local", "api", "callable"])
def test_self_play_shares_one_stochastic_question_per_group(backend):
    generated_prompts = []

    def sample_one(prompt):
        generated_prompts.append(prompt)
        return f"random question {len(generated_prompts)}"

    if backend == "local":
        generator = LocalQuestionGenerator("generate", tokenizer=CharacterTokenizer())
        generator.set_generate_fn(lambda prompts, **kwargs: [sample_one(prompt) for prompt in prompts])
    elif backend == "api":
        generator = OpenAIQuestionGenerator("http://unused/v1", "key", "model", "generate")
        generator._chat = lambda system, user: sample_one(user)
    else:
        generator = CallableQuestionGenerator(lambda topic, seed: sample_one((topic, seed)))
    environments = [SelfPlayEnvironment("math", generator) for _ in range(4)]
    results = environments[0].reset_batch(environments, [7, 7, 8, 8])
    assert [item.observation for item in results] == [
        "random question 1",
        "random question 1",
        "random question 2",
        "random question 2",
    ]
    assert len(generated_prompts) == 2
    results[0].info["only_first"] = True
    assert "only_first" not in results[1].info
    # Backend generation is fresh for the next batch, not cached forever.
    later = environments[0].reset_batch(environments, [7, 7, 8, 8])
    assert [item.observation for item in later] == [
        "random question 3",
        "random question 3",
        "random question 4",
        "random question 4",
    ]


def test_shared_questions_preserve_but_isolate_nested_episode_metadata():
    class MetadataGenerator(QuestionGenerator):
        def generate(self, topic, seed=None):
            return GeneratedQuestion("shared question", {"reference": {"answer": 42}, "notes": []})

    generator = MetadataGenerator()
    environments = [SelfPlayEnvironment("math", generator) for _ in range(2)]
    first, second = environments[0].reset_batch(environments, [7, 7])
    assert first.observation == second.observation == "shared question"
    assert first.info["reference"] == second.info["reference"] == {"answer": 42}
    first.info["notes"].append("first solver scratch work")
    first.info["reference"]["answer"] = -1
    assert second.info["notes"] == []
    assert second.info["reference"] == {"answer": 42}


def test_local_question_variation_does_not_depend_on_batch_position():
    generator = LocalQuestionGenerator("generate", tokenizer=CharacterTokenizer())
    generator.set_generate_fn(lambda prompts, **kwargs: prompts)
    together = generator.generate_batch(["math", "math"], [7, 7])
    alone = generator.generate("math", 7)
    assert together[0].question == together[1].question == alone.question
    assert together[0].metadata == together[1].metadata == alone.metadata


def test_trainer_tools_are_isolated_between_members_batches_and_resets():
    # Exercise the trainer's real environment ownership boundary without
    # constructing a numerical model, optimizer, or device runtime.
    template = NotepadTool()
    trainer = SimpleNamespace(
        _tools=[template],
        _tool_call_parser=None,
        arguments=SimpleNamespace(max_tool_calls_per_step=5),
    )

    def make_env():
        return AgenticMoshPitTrainer._wrap_env_with_tools(trainer, TextEnvironment())

    first, second = make_env(), make_env()
    first.reset(1)
    second.reset(1)
    first.step_with_tool_calls("", tool_calls=[("notepad", {"action": "write", "text": "first secret"})])
    assert (
        first.step_with_tool_calls("", tool_calls=[("notepad", {"action": "read"})]).observation
        == "[notepad]: first secret"
    )
    assert (
        second.step_with_tool_calls("", tool_calls=[("notepad", {"action": "read"})]).observation
        == "[notepad]: (notepad is empty)"
    )
    second.step_with_tool_calls("", tool_calls=[("notepad", {"action": "write", "text": "second secret"})])
    first.reset(2)
    assert first.tools["notepad"].call("read") == "(notepad is empty)"
    assert second.tools["notepad"].call("read") == "second secret"
    first.tools["notepad"].call("write", "new notes")
    second.reset_batch([second], [3])
    assert second.tools["notepad"].call("read") == "(notepad is empty)"
    assert first.tools["notepad"].call("read") == "new notes"
    later = make_env()
    assert later.tools["notepad"].call("read") == template.call("read") == "(notepad is empty)"


def test_function_schemas_validate_pep604_typing_union_and_nested_containers():
    from jsonschema import Draft202012Validator

    def custom(
        count: int | None = None,
        choice: tp.Union[int, str] = 1,  # noqa: UP007 -- exercise the legacy typing.Union representation too
        values: list[int | None] | None = None,
    ):
        return str((count, choice, values))

    schema = function_to_json(custom)["function"]["parameters"]
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for instance in ({"count": None}, {"count": 4, "choice": "x", "values": [1, None]}, {"values": None}):
        validator.validate(instance)
    for instance in ({"count": "wrong"}, {"choice": []}, {"values": ["wrong"]}):
        assert not validator.is_valid(instance)
    assert FunctionTool(custom).chat_schema["parameters"] == schema


def test_regex_preserves_python_matching_and_bounds_hostile_patterns():
    tool = RegexTool()
    assert json.loads(tool.call(r"(\w+)@(\w+\.\w+)", "alice@example.org")) == [["alice", "example.org"]]
    assert json.loads(tool.call(r"(?<=price: )\d+", "price: 17")) == ["17"]
    assert tool.call("unmatched", "abc") == "No matches found."
    assert tool.call("[", "text").startswith("Regex error:")
    started = time.monotonic()
    assert "timed out" in RegexTool(timeout=0.3).call("(a+)+$", "a" * 64 + "!")
    assert time.monotonic() - started < 5.0
    # Text is streamed through stdin, so matching also supports inputs larger
    # than the OS limit for a single command-line argument.
    assert tool.call("missing", "x" * 200000) == "No matches found."
    output = RegexTool(max_output_length=128).call(".", "x" * 10000)
    assert len(output) <= 128 + len("\n... (output truncated)")
    assert output.endswith("... (output truncated)")
