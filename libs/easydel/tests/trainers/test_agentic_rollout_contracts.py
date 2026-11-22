"""Host-only regression coverage for structured trajectories and episode ownership."""

import json
import re
import time
import typing as tp
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers._shared.tools import FunctionTool, NotepadTool, RegexTool, function_to_json
from easydel.trainers.agentic_moshpit.agentic_moshpit_config import AgenticMoshPitConfig
from easydel.trainers.agentic_moshpit.agentic_moshpit_trainer import AgenticMoshPitTrainer
from easydel.trainers.agentic_moshpit.env_manager import (
    GenerationStepOutput,
    RolloutManager,
    TrajectoryResult,
    TurnRecord,
)
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
    assert text.startswith(prompts_seen[1])
    assert raw_text in prompts_seen[1]
    assert "<invoke>" not in prompts_seen[1]
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


class _ThinkStrippingTokenizer:
    """Simulates Qwen3/DeepSeek-R1 templates that strip <think> from non-final turns
    and optionally inject <think>\\n on add_generation_prompt=True."""

    pad_token_id = None
    eos_token_id = 999
    eos_token = "<|im_end|>"

    def __init__(self, *, inject_think_gen_prompt: bool = False):
        self.inject_think_gen_prompt = inject_think_gen_prompt

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        del kwargs
        parts = []
        last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1)
        for idx, m in enumerate(messages):
            content = m["content"]
            if m["role"] == "assistant" and idx < last_assistant:
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            parts.append(f"<|im_start|>{m['role']}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n<think>\n" if self.inject_think_gen_prompt else "<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, **kwargs):
        del kwargs
        # Encode <|im_end|> as single token 999 and every other char as ord(c).
        tokens: list[int] = []
        i = 0
        while i < len(text):
            if text.startswith("<|im_end|>", i):
                tokens.append(999)
                i += len("<|im_end|>")
            else:
                tokens.append(ord(text[i]))
                i += 1
        return tokens


def test_reasoning_template_think_stripping_and_think_gen_prompt_preserve_exact_masking():
    # 1) Template strips <think> from non-final turns: old prefix-subtraction shifted mask onto tool obs!
    tok = _ThinkStrippingTokenizer(inject_think_gen_prompt=False)
    prompts_seen: list[str] = []
    turn0_raw = "<think>long reasoning block</think>call_tool<|im_end|>"
    turn0_ids = tok.encode(turn0_raw)
    turn0_logps = [-0.25] * len(turn0_ids)
    turn1_raw = "<think>second thought</think>final_answer<|im_end|>"
    turn1_ids = tok.encode(turn1_raw)
    turn1_logps = [-0.5] * len(turn1_ids)

    class _TwoStepEnv(AgenticEnvironment):
        def __init__(self):
            self.step_num = 0

        def reset(self, seed=None):
            return ResetResult("solve me")

        def step(self, action):
            self.step_num += 1
            if self.step_num == 1:
                return StepResult("TOOL_OBSERVATION_TEXT", reward=0.0, terminated=False, info={"tool_calls": [{"name": "t"}]})
            return StepResult("", reward=1.0, terminated=True)

    def generate_rich(prompts, **kwargs):
        prompts_seen.extend(prompts)
        if len(prompts_seen) == 1:
            return [GenerationStepOutput(text="call_tool", raw_text=turn0_raw, token_ids=turn0_ids, sampling_logps=turn0_logps)]
        return [GenerationStepOutput(text="final_answer", raw_text=turn1_raw, token_ids=turn1_ids, sampling_logps=turn1_logps)]

    manager = RolloutManager(tok, max_steps=3)
    assert manager._pad_token_id == 999  # falls back to eos_token_id, not 0

    traj = manager.run_grouped_episodes(_TwoStepEnv, generate_rich, group_size=1)[0]
    # No duplicate <|im_end|> (999) tokens after turn0 or turn1.
    ids_list = traj.input_ids.tolist()
    mask_list = traj.response_mask.tolist()
    turn_id_list = traj.turn_ids.tolist()

    assert [t for t, m in zip(ids_list, mask_list, strict=True) if m == 1] == turn0_ids + turn1_ids
    assert [t for t, tid in zip(ids_list, turn_id_list, strict=True) if tid == 0] == turn0_ids
    assert [t for t, tid in zip(ids_list, turn_id_list, strict=True) if tid == 1] == turn1_ids
    # Observation tokens must have response_mask == 0 and turn_id == -1.
    obs_ids = tok.encode("TOOL_OBSERVATION_TEXT")
    obs_pos = next(i for i in range(len(ids_list) - len(obs_ids) + 1) if ids_list[i : i + len(obs_ids)] == obs_ids)
    assert mask_list[obs_pos : obs_pos + len(obs_ids)] == [0] * len(obs_ids)
    assert turn_id_list[obs_pos : obs_pos + len(obs_ids)] == [-1] * len(obs_ids)
    # Turn 1 prompt must retain exact turn 0 raw output (canonical prefix parity).
    assert turn0_raw in prompts_seen[1]
    assert "TOOL_OBSERVATION_TEXT" in prompts_seen[1]
    # Sampling logps preserved 1-to-1 through TrajectoryResult and collate_trajectories.
    assert traj.sampling_logps is not None
    collated = manager.collate_trajectories([traj], max_prompt_length=64, max_completion_length=256)
    assert "sampling_per_token_logps" in collated
    valid_comp = collated["completion_mask"][0].astype(bool)
    np.testing.assert_allclose(collated["sampling_per_token_logps"][0][valid_comp], turn0_logps + turn1_logps)

    # 2) Template injecting <think>\n on add_generation_prompt=True must NOT overshoot to 0 masked tokens.
    tok_think_prompt = _ThinkStrippingTokenizer(inject_think_gen_prompt=True)
    manager_think = RolloutManager(tok_think_prompt, max_steps=1)
    resp_ids = tok_think_prompt.encode("step</think>42<|im_end|>")
    traj_think = manager_think.run_grouped_episodes(
        TextEnvironment,
        lambda prompts, **kwargs: [GenerationStepOutput(text="42", raw_text="step</think>42<|im_end|>", token_ids=resp_ids)],
        group_size=1,
    )[0]
    assert int(traj_think.response_mask.sum()) == len(resp_ids)
    assert traj_think.input_ids[traj_think.response_mask == 1].tolist() == resp_ids


def test_bpe_cross_boundary_merge_preserves_exact_sampled_completion_ids():
    class _MergingBPETokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
            del kwargs
            out = "".join(f"{m['role']}:{m['content']}\n" for m in messages)
            return out + ("assistant:" if add_generation_prompt else "")

        def encode(self, text, **kwargs):
            del kwargs
            # Merge ":h" across the prompt/completion boundary into token 9001 if encoded together.
            ids = []
            i = 0
            while i < len(text):
                if text[i : i + 2] == ":h":
                    ids.append(9001)
                    i += 2
                else:
                    ids.append(ord(text[i]))
                    i += 1
            return ids

    tok = _MergingBPETokenizer()
    sampled_ids = [5001, 5002, 5003]  # Non-canonical BPE split sampled by engine
    sampled_logps = [-0.1, -0.2, -0.3]

    def generate(prompts, **kwargs):
        return [GenerationStepOutput(text="hello", raw_text="hello", token_ids=sampled_ids, sampling_logps=sampled_logps)]

    manager = RolloutManager(tok, max_steps=1)
    traj = manager.run_grouped_episodes(TextEnvironment, generate, group_size=1)[0]
    # Exact sampled completion_ids preserved without BPE cross-boundary merge (9001 must NOT appear).
    assert 9001 not in traj.input_ids.tolist()
    assert traj.input_ids[traj.response_mask == 1].tolist() == sampled_ids
    np.testing.assert_allclose(traj.sampling_logps[traj.response_mask == 1], sampled_logps)


def test_rollout_manager_closes_envs_on_exception_empty_collate_and_terminal_tool_role():
    closed: list[int] = []

    class _ClosingEnv(AgenticEnvironment):
        def __init__(self, idx):
            self.idx = idx

        def reset(self, seed=None):
            return ResetResult("task")

        def step(self, action):
            raise RuntimeError("step boom")

        def close(self):
            closed.append(self.idx)

    counter = iter(range(10))
    manager = RolloutManager(CharacterTokenizer(), max_steps=2)
    with pytest.raises(RuntimeError, match="step boom"):
        manager.run_grouped_episodes(
            lambda: _ClosingEnv(next(counter)),
            lambda prompts, **kwargs: ["hi"] * len(prompts),
            group_size=3,
        )
    assert sorted(closed) == [0, 1, 2]

    # Empty trajectory and empty batch must not crash np.argmax / np.stack.
    empty_traj = TrajectoryResult(
        input_ids=np.zeros(0, dtype=np.int64),
        attention_mask=np.zeros(0, dtype=np.int32),
        prompt_mask=np.zeros(0, dtype=np.int32),
        response_mask=np.zeros(0, dtype=np.int32),
        episode_reward=0.0,
        step_rewards=[],
        num_steps=0,
        turns=[],
    )
    collated_empty_traj = manager.collate_trajectories([empty_traj], max_prompt_length=4, max_completion_length=4)
    assert collated_empty_traj["prompt_ids"].shape == (1, 4)
    assert collated_empty_traj["completion_ids"].shape == (1, 4)
    assert collated_empty_traj["completion_mask"].sum() == 0

    collated_none = manager.collate_trajectories([], max_prompt_length=4, max_completion_length=4)
    assert collated_none["prompt_ids"].shape == (0, 4)

    # Terminal tool-call observation uses role="tool" (not "user").
    tool = FunctionTool(lambda text: text, tool_name="echo")
    manager_1step = RolloutManager(CharacterTokenizer(), max_steps=1)
    traj_term = manager_1step.run_grouped_episodes(
        lambda: ToolEnvWrapper(TextEnvironment(), [tool]),
        lambda prompts, **kwargs: ['<tool_call>{"name":"echo","arguments":{"text":"hi"}}</tool_call>'],
        group_size=1,
    )[0]
    assert traj_term.turns[-1].role == "tool"
    assert "[echo]: hi" in traj_term.turns[-1].content

    # _pad_token_id fallback to list eos_token_id and then 0.
    assert RolloutManager(SimpleNamespace(pad_token_id=None, eos_token_id=[7, 8]))._pad_token_id == 7
    assert RolloutManager(SimpleNamespace(pad_token_id=None, eos_token_id=None))._pad_token_id == 0


def test_agentic_trainer_make_generate_fn_preserves_tokens_and_multihost_rpc_lockstep(monkeypatch):
    import jax
    from easydel.trainers.base_trainer import GenerationResults

    # 1) _make_generate_fn preserves exact completion_ids & sampling_per_token_logps while keeping last_generation_metadata intact.
    trainer = object.__new__(AgenticMoshPitTrainer)
    trainer._strip_thinking = lambda text: text
    trainer.arguments = SimpleNamespace(max_length=128)
    trainer._pad_token_id = 0
    seen_rollout_kwargs: list[dict] = []

    def fake_rollout(**kwargs):
        seen_rollout_kwargs.append(kwargs)
        return GenerationResults(
            generation_results=["ok"],
            prompt_ids=np.array([[1, 2]], dtype=np.int32),
            prompt_mask=np.array([[1, 1]], dtype=np.int32),
            sequences=np.array([[1, 2, 10, 11, 0]], dtype=np.int32),
            completion_ids=np.array([[10, 11, 0]], dtype=np.int32),
            completion_mask=np.array([[1, 1, 0]], dtype=np.int32),
            decoded_prompts=["p"],
            text=["ok"],
            raw_text=["ok"],
            reasoning=[None],
            tool_calls=[None],
            sampling_per_token_logps=np.array([[-0.3, -0.7, 0.0]], dtype=np.float32),
        )

    trainer.rollout = fake_rollout
    gen_fn = AgenticMoshPitTrainer._make_generate_fn(trainer, state=SimpleNamespace())
    out = gen_fn(["p"], prompt_token_ids=[[1, 2]])
    assert out == ["ok"]
    assert gen_fn.last_generation_metadata == [{"text": "ok", "raw_text": "ok", "reasoning": None, "tool_calls": None}]
    assert gen_fn.last_generation_tokens[0]["token_ids"].tolist() == [10, 11]
    np.testing.assert_allclose(gen_fn.last_generation_tokens[0]["sampling_logps"], [-0.3, -0.7])
    assert "input_ids" in seen_rollout_kwargs[0]
    assert seen_rollout_kwargs[0]["input_ids"].tolist() == [[1, 2]]

    # 2) Simulated 2-host lockstep coordination: rank 0 drives envs & broadcasts RPCs; rank 1 runs identical rollouts without calling env_factory.
    wire: list[tp.Any] = []
    rank0_rollouts: list[list[str]] = []
    rank1_rollouts: list[list[str]] = []
    rank1_env_calls = 0

    class _TwoTurnEnv(AgenticEnvironment):
        def __init__(self):
            self.n = 0

        def reset(self, seed=None):
            return ResetResult(f"seed:{seed}")

        def step(self, action):
            self.n += 1
            return StepResult("next" if self.n == 1 else "", reward=1.0 if self.n == 2 else 0.0, terminated=self.n == 2)

    def make_host_trainer(rollout_log, on_env_call):
        t = object.__new__(AgenticMoshPitTrainer)
        t.arguments = SimpleNamespace(group_size=1, num_env_groups=1, max_length=128)
        t._strip_thinking = lambda x: x
        t._tools = []
        t._tool_call_parser = None
        t.processing_class = CharacterTokenizer()
        t._pad_token_id = 0
        t.env_factory = on_env_call
        t._rollout_manager = RolloutManager(CharacterTokenizer(), max_steps=3)

        def host_rollout(**kwargs):
            rollout_log.append(list(kwargs["prompts"]))
            resp = "turn_resp"
            rids = np.array([[ord(c) for c in resp]], dtype=np.int32)
            return GenerationResults(
                generation_results=[resp],
                prompt_ids=np.zeros((1, 1), dtype=np.int32),
                prompt_mask=np.ones((1, 1), dtype=np.int32),
                sequences=rids,
                completion_ids=rids,
                completion_mask=np.ones_like(rids),
                decoded_prompts=kwargs["prompts"],
                text=[resp],
                raw_text=[resp],
                reasoning=[None],
                tool_calls=[None],
            )

        t.rollout = host_rollout
        return t

    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: 0)
    monkeypatch.setattr(AgenticMoshPitTrainer, "_broadcast_object_from_host0", classmethod(lambda cls, obj: (wire.append(obj), obj)[1]))
    t0 = make_host_trainer(rank0_rollouts, lambda: _TwoTurnEnv())
    trajs0 = t0._run_grouped_episodes_coordinated(state=SimpleNamespace(), base_seed=5)

    wire_iter = iter(wire)
    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(AgenticMoshPitTrainer, "_broadcast_object_from_host0", classmethod(lambda cls, obj: next(wire_iter)))

    def forbidden_env():
        nonlocal rank1_env_calls
        rank1_env_calls += 1
        raise AssertionError("Rank 1 must not instantiate environments")

    t1 = make_host_trainer(rank1_rollouts, forbidden_env)
    trajs1 = t1._run_grouped_episodes_coordinated(state=SimpleNamespace(), base_seed=5)

    assert rank1_env_calls == 0
    assert rank0_rollouts == rank1_rollouts
    assert len(rank0_rollouts) == 2
    assert trajs0[0].input_ids.tolist() == trajs1[0].input_ids.tolist()
    assert trajs0[0].episode_reward == trajs1[0].episode_reward == 1.0


def test_agentic_trainer_eval_honors_is_train_zero_std_and_aux_zero_info():
    # Setup lightweight trainer instance without full model/optimizer init.
    trainer = object.__new__(AgenticMoshPitTrainer)
    cfg = AgenticMoshPitConfig(
        max_steps=2,
        group_size=2,
        num_env_groups=2,
        max_prompt_length=16,
        max_completion_length=16,
        advantage_estimator="agentic_reinforce",
        esurge_importance_sampling_correction=False,
    )
    trainer.arguments = cfg
    trainer.scale_rewards = "none"
    trainer._rollout_step = 5
    trainer._eval_rollout_step = 0
    trainer._tools = []
    trainer._tool_call_parser = None
    trainer._tool_schemas = None
    trainer.processing_class = CharacterTokenizer()
    trainer.ref_state = None
    trainer.reward_funcs = [lambda prompts, completions, **kw: [0.0] * len(completions)]
    trainer.reward_func_names = ["zero_aux"]
    trainer.reward_weights = np.array([1.0], dtype=np.float32)
    trainer.reward_processing_classes = [None]
    trainer._all_gather = lambda x: x

    wandb_gen_calls: list[bool] = []
    wandb_traj_calls: list[bool] = []
    trainer._log_training_generations_to_wandb = lambda **kw: wandb_gen_calls.append(True)
    trainer._log_trajectories_to_wandb = lambda *a, **kw: wandb_traj_calls.append(True)

    # Group 0 has identical rewards [1.0, 1.0] (zero std); group 1 has [0.0, 1.0] (non-zero std) -> frac == 0.5!
    rewards_by_idx = [1.0, 1.0, 0.0, 1.0]

    def make_traj(reward: float) -> TrajectoryResult:
        return TrajectoryResult(
            input_ids=np.array([1, 2, 10, 11], dtype=np.int64),
            attention_mask=np.ones(4, dtype=np.int32),
            prompt_mask=np.array([1, 1, 0, 0], dtype=np.int32),
            response_mask=np.array([0, 0, 1, 1], dtype=np.int32),
            episode_reward=reward,
            step_rewards=[reward],
            num_steps=1,
            turns=[TurnRecord(role="user", content="q"), TurnRecord(role="assistant", content="a", is_response=True)],
            info={},
            turn_ids=np.array([-1, -1, 0, 0], dtype=np.int32),
        )

    captured_trajs: list[TrajectoryResult] = []
    trainer._rollout_manager = RolloutManager(CharacterTokenizer(), max_steps=2)

    def fake_coordinated(*, state, base_seed):
        del state, base_seed
        trajs = [make_traj(r) for r in rewards_by_idx]
        captured_trajs.clear()
        captured_trajs.extend(trajs)
        return trajs

    trainer._run_grouped_episodes_coordinated = fake_coordinated
    state = SimpleNamespace(model=SimpleNamespace(pause_esurge=lambda **kw: None))

    _, eval_metrics = AgenticMoshPitTrainer._preprocess_batch_input(trainer, state=state, batch={}, is_train=False)
    assert trainer._rollout_step == 5  # Must NOT be mutated during eval!
    assert trainer._eval_rollout_step == 1
    assert wandb_gen_calls == []
    assert wandb_traj_calls == []
    assert eval_metrics["frac_reward_zero_std"] == pytest.approx(0.5)
    # Auxiliary reward == 0.0 must still write traj.info!
    assert all("aux_reward" in t.info and t.info["aux_reward/zero_aux"] == 0.0 for t in captured_trajs)

    # Train step MUST increment _rollout_step and invoke training W&B loggers.
    _, train_metrics = AgenticMoshPitTrainer._preprocess_batch_input(trainer, state=state, batch={}, is_train=True)
    assert trainer._rollout_step == 6
    assert len(wandb_gen_calls) == 1
    assert len(wandb_traj_calls) == 1
    assert train_metrics["frac_reward_zero_std"] == pytest.approx(0.5)


def test_agentic_trainer_does_not_mutate_arguments_tool_schemas(monkeypatch):
    cfg = AgenticMoshPitConfig(max_steps=2, group_size=2, num_env_groups=1, tool_schemas=None)
    tool = FunctionTool(lambda x: str(x), tool_name="echo")

    def fake_grpo_init(self, **kwargs):
        self.arguments = kwargs["arguments"]
        self.processing_class = kwargs["processing_class"]

    monkeypatch.setattr("easydel.trainers.group_relative_policy_optimization.grpo_trainer.GRPOTrainer.__init__", fake_grpo_init)
    trainer = AgenticMoshPitTrainer(
        arguments=cfg,
        model=None,
        env_factory=TextEnvironment,
        tools=[tool],
        processing_class=CharacterTokenizer(),
    )
    assert cfg.tool_schemas is None
    assert trainer._tool_schemas == [tool.chat_schema]
    assert trainer._rollout_manager.tool_schemas == [tool.chat_schema]


def test_agentic_trainer_wires_esurge_is_correction_and_multihost_global_arrays(monkeypatch):
    import jax
    from jax.experimental import multihost_utils

    # 1) esurge_importance_sampling_correction=True wires compute_state_logps + importance_sampling_ratio.
    trainer = object.__new__(AgenticMoshPitTrainer)
    cfg = AgenticMoshPitConfig(
        max_steps=1,
        group_size=2,
        num_env_groups=1,
        max_prompt_length=4,
        max_completion_length=4,
        esurge_importance_sampling_correction=True,
    )
    trainer.arguments = cfg
    trainer.scale_rewards = "group"
    trainer._rollout_step = 0
    trainer._eval_rollout_step = 0
    trainer._tools = []
    trainer._tool_call_parser = None
    trainer._tool_schemas = None
    trainer.processing_class = CharacterTokenizer()
    trainer.ref_state = None
    trainer.reward_funcs = [AgenticMoshPitTrainer._env_reward_placeholder]
    trainer.reward_func_names = ["_env_reward_placeholder"]
    trainer.reward_weights = np.array([1.0], dtype=np.float32)
    trainer.reward_processing_classes = [None]
    trainer._all_gather = lambda x: x
    trainer._log_training_generations_to_wandb = lambda **kw: None
    trainer._log_trajectories_to_wandb = lambda *a, **kw: None
    trainer._rollout_manager = RolloutManager(CharacterTokenizer(), max_steps=1)

    state_logps_called = []

    def fake_compute_state_logps(state, sequences, seq_mask, kwargs, *, prompt_length, logprob_vocab_chunk_size):
        state_logps_called.append((sequences.shape, prompt_length))
        return np.full((2, 4), -0.5, dtype=np.float32)

    trainer.compute_state_logps = fake_compute_state_logps

    trajs = [
        TrajectoryResult(
            input_ids=np.array([1, 2, 10, 11], dtype=np.int64),
            attention_mask=np.ones(4, dtype=np.int32),
            prompt_mask=np.array([1, 1, 0, 0], dtype=np.int32),
            response_mask=np.array([0, 0, 1, 1], dtype=np.int32),
            episode_reward=r,
            step_rewards=[r],
            num_steps=1,
            turns=[TurnRecord(role="user", content="q"), TurnRecord(role="assistant", content="a", is_response=True)],
            info={},
            turn_ids=np.array([-1, -1, 0, 0], dtype=np.int32),
            sampling_logps=np.array([0.0, 0.0, -0.5, -0.5], dtype=np.float32),
        )
        for r in (0.0, 1.0)
    ]
    trainer._run_grouped_episodes_coordinated = lambda *, state, base_seed: trajs
    state = SimpleNamespace(model=SimpleNamespace(forward=lambda **kw: None, pause_esurge=lambda **kw: None))

    model_batch, _ = AgenticMoshPitTrainer._preprocess_batch_input(trainer, state=state, batch={}, is_train=True)
    assert state_logps_called == [((2, 8), 4)]
    assert "sampling_per_token_logps" in model_batch
    assert "old_per_token_logps" in model_batch
    assert "importance_sampling_ratio" in model_batch
    # Identical policy & sampling logps -> importance ratio == 1.0!
    np.testing.assert_allclose(np.asarray(model_batch["importance_sampling_ratio"]), np.ones((2, 1), dtype=np.float32), atol=1e-5)

    # 2) Multi-host _to_global_replicated_array routes local arrays through host_local_array_to_global_array instead of _all_gather.
    routed: list[tuple] = []
    fake_mesh = object()
    trainer._model = SimpleNamespace(mesh=fake_mesh)
    trainer._all_gather = lambda x: (_ for _ in ()).throw(AssertionError("_all_gather must not receive local array on multi-host"))
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        multihost_utils,
        "host_local_array_to_global_array",
        lambda arr, mesh, pspec: (routed.append((arr.shape, mesh, pspec)), arr)[1],
    )
    out_arr = trainer._to_global_replicated_array(np.ones((2, 3), dtype=np.int32))
    assert out_arr.shape == (2, 3)
    assert len(routed) == 1 and routed[0][1] is fake_mesh


def test_agentic_and_step_advantages_numpy_contracts():
    from easydel.trainers.agentic_moshpit.utils import compute_advantages_agentic, compute_advantages_step

    # 1) Two identical 100%-successful 5-step rollouts must receive 0.0 advantage (not -1.375!).
    five_step_rewards = [
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    ]
    five_step_mask = np.array(
        [
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_allclose(
        compute_advantages_agentic(five_step_rewards, five_step_mask, group_size=2, gamma=0.95),
        np.zeros_like(five_step_mask, dtype=np.float32),
        atol=1e-7,
    )

    # 2) Explicit turn_ids preserve attribution across 0-token turns and adjacent turns without 0-gap.
    rewards = [np.array([0.0, 4.0]), np.array([2.0, 0.0])]
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
    turn_ids = np.array([[1, 1, -1, -1], [0, 0, 1, 1]], dtype=np.int32)
    actual = compute_advantages_agentic(rewards, mask, group_size=2, gamma=0.5, turn_ids=turn_ids)
    step1_mag = 2.0 / (2.0 + 1e-4)
    np.testing.assert_allclose(actual[0], [step1_mag, step1_mag, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(actual[1], [0.0, 0.0, -step1_mag, -step1_mag], atol=1e-5)

    # 3) Empty / 0-length masks must not crash.
    assert compute_advantages_agentic(rewards, np.zeros((2, 0), dtype=np.int32), group_size=2, gamma=0.5).shape == (2, 0)
    assert compute_advantages_agentic([], np.zeros((0, 4), dtype=np.int32), group_size=2, gamma=0.5).shape == (0, 4)

    # 4) compute_advantages_step uses per-turn G_t and honors scale_rewards="none".
    step_rewards = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 2.0], dtype=np.float32)]
    step_mask = np.array([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=np.int32)
    step_tids = np.array([[0, 0, -1, 1, 1], [0, 0, -1, 1, 1]], dtype=np.int32)
    adv_none = compute_advantages_step(
        step_rewards, group_size=2, gamma=0.5, response_masks=step_mask, turn_ids=step_tids, scale_rewards="none"
    )
    np.testing.assert_allclose(adv_none[0], [0.0, 0.0, 0.0, -1.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(adv_none[1], [0.0, 0.0, 0.0, 1.0, 1.0], atol=1e-5)
