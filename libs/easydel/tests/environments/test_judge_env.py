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

"""CPU tests for the LLM-judge client, rubric/pairwise judges, and environment (fake transport, no network)."""

from __future__ import annotations

import json
import re
import threading

import pytest
from easydel.environments import (
    JudgeClient,
    JudgeError,
    LLMJudgeEnvironment,
    PairwiseJudge,
    RequestRateLimiter,
    RubricJudge,
    build_environment,
    make_environment_reward_fn,
    make_judge_reward_fn,
    parse_judge_json,
)

pytestmark = pytest.mark.cpu_ok


def _ok(content: str):
    return 200, {}, json.dumps({"choices": [{"message": {"content": content}}]}).encode()


def _user_text(body: bytes) -> str:
    return json.loads(body)["messages"][-1]["content"]


def _rubric(correctness=8, clarity=10, completeness=0, fatal=False, **extra):
    return {"correctness": correctness, "clarity": clarity, "completeness": completeness, "fatal_issue": fatal, **extra}


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class RecordingTransport:
    """Thread-safe fake transport driven by a handler(user_text, call_index)."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[str] = []
        self.headers: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, url, headers, body, timeout):
        with self._lock:
            index = len(self.calls)
            text = _user_text(body)
            self.calls.append(text)
            self.headers.append(dict(headers))
        return self.handler(text, index)


def _client(transport, **kw):
    kw.setdefault("sleep", lambda s: None)
    return JudgeClient("http://judge.local/v1", "judge-model", transport=transport, **kw)


def test_rate_limiter_reserves_evenly_spaced_slots():
    clock = FakeClock()
    limiter = RequestRateLimiter(120, clock=clock)  # 0.5 s interval
    assert [limiter.reserve() for _ in range(4)] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    shared = RequestRateLimiter(60, world_size=4, clock=clock)  # 15 req/min per host
    assert shared.interval == pytest.approx(4.0)


def test_client_rpm_limiter_spaces_calls_with_fake_clock():
    clock = FakeClock()
    times: list[float] = []

    def handler(text, index):
        times.append(clock.now)
        return _ok("hi")

    client = _client(RecordingTransport(handler), requests_per_minute=60, world_size=2, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        assert client.chat([{"role": "user", "content": "x"}]) == "hi"
    assert times == pytest.approx([0.0, 2.0, 4.0])


def test_retry_after_is_honoured_and_capped():
    def handler(text, index):
        if index == 0:
            return 429, {"Retry-After": "7"}, b""
        if index == 1:
            return 429, {"retry-after": "1000"}, b""
        return _ok("done")

    clock = FakeClock()
    client = _client(RecordingTransport(handler), retry_after_cap=60.0, sleep=clock.sleep, max_retries=3)
    assert client.chat([{"role": "user", "content": "x"}]) == "done"
    assert clock.sleeps == pytest.approx([7.0, 60.0])


def test_api_key_from_named_env_var_is_sent_but_not_in_repr(monkeypatch):
    monkeypatch.setenv("TEST_JUDGE_KEY_NAME", "sk-secret")
    transport = RecordingTransport(lambda text, i: _ok("x"))
    client = _client(transport, api_key_env="TEST_JUDGE_KEY_NAME")
    client.chat([{"role": "user", "content": "x"}])
    assert transport.headers[0]["Authorization"] == "Bearer sk-secret"
    assert "sk-secret" not in repr(client)
    monkeypatch.delenv("TEST_JUDGE_KEY_NAME")
    with pytest.raises(ValueError):
        _client(transport, api_key_env="TEST_JUDGE_KEY_NAME")


def test_non_retryable_status_raises_without_retry():
    transport = RecordingTransport(lambda text, i: (400, {}, b"bad"))
    with pytest.raises(JudgeError) as info:
        _client(transport, max_retries=3).chat([{"role": "user", "content": "x"}])
    assert info.value.status == 400 and not info.value.rate_limited
    assert len(transport.calls) == 1


def test_parse_judge_json_is_strict():
    assert parse_judge_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_judge_json('Result: {"a": 2}') == {"a": 2}
    for bad in ("no json here", "[1, 2]", "{not: json}"):
        with pytest.raises(ValueError):
            parse_judge_json(bad)


def test_rubric_score_blends_lexicographically_and_fatal_is_zero():
    judge = RubricJudge(_client(RecordingTransport(lambda t, i: _ok(json.dumps(_rubric())))))
    # correctness 8, secondaries (10, 0) -> mean 0.5 -> 8 + 0.5 * 0.5 = 8.25, normalized by 10.5.
    assert judge.score("q", "r") == pytest.approx(8.25 / 10.5)
    assert judge.score_from_judgment(_rubric(fatal=True)) == 0.0
    # One correctness point always beats perfect secondaries.
    assert judge.score_from_judgment(_rubric(9, 0, 0)) > judge.score_from_judgment(_rubric(8, 10, 10))
    with pytest.raises(ValueError):
        judge.score_from_judgment(_rubric(correctness=11))
    with pytest.raises(ValueError):
        judge.score_from_judgment({"correctness": 5, "clarity": 5, "completeness": 5})


def test_rubric_malformed_reply_is_unscorable():
    transport = RecordingTransport(lambda t, i: _ok("I think it is fine."))
    judge = RubricJudge(_client(transport), max_parse_retries=1)
    assert judge.score("q", "r") is None
    assert len(transport.calls) == 2


def test_batch_judging_returns_scores_in_candidate_order():
    def handler(text, index):
        entries = [{"index": 1, **_rubric(2, 0, 0)}, {"index": 0, **_rubric(10, 0, 0)}]
        return _ok(json.dumps({"scores": entries}))

    transport = RecordingTransport(handler)
    scores = RubricJudge(_client(transport)).score_batch("q", ["best", "worst"])
    assert scores == pytest.approx([10 / 10.5, 2 / 10.5])
    assert len(transport.calls) == 1


def test_batch_bijection_failure_retries_then_falls_back_per_candidate():
    def handler(text, index):
        if "[Candidate 0]" in text:  # batch call: duplicate index -> not a bijection
            entries = [{"index": 0, **_rubric()}, {"index": 0, **_rubric()}]
            return _ok(json.dumps({"scores": entries}))
        correctness = 10 if "alpha" in text else 4
        return _ok(json.dumps(_rubric(correctness, 0, 0)))

    transport = RecordingTransport(handler)
    scores = RubricJudge(_client(transport), max_batch_retries=1).score_batch("q", ["alpha", "beta"])
    assert scores == pytest.approx([10 / 10.5, 4 / 10.5])
    assert sum("[Candidate 0]" in c for c in transport.calls) == 2
    assert sum("[Candidate 0]" not in c for c in transport.calls) == 2


def test_persistent_429_does_not_fan_out():
    transport = RecordingTransport(lambda t, i: (429, {"Retry-After": "1"}, b""))
    judge = RubricJudge(_client(transport, max_retries=2))
    assert judge.score_batch("q", ["a", "b", "c"]) == [None, None, None]
    assert len(transport.calls) == 3  # only the batch call's attempts, no per-candidate calls


def test_one_failed_call_does_not_cancel_siblings():
    def handler(text, index):
        if "poison" in text:
            return 400, {}, b""
        return _ok(json.dumps(_rubric(5, 0, 0)))

    judge = RubricJudge(_client(RecordingTransport(handler), max_concurrency=2))
    assert judge.score_many("q", ["a", "poison", "b"]) == [pytest.approx(5 / 10.5), None, pytest.approx(5 / 10.5)]


_PAIR_RE = re.compile(r"\[Response A\]\n(.*?)\n\n\[Response B\]\n(.*?)\n\n", re.DOTALL)


def _longer_wins(text, index):
    a, b = _PAIR_RE.search(text).groups()
    winner = "tie" if len(a) == len(b) else ("A" if len(a) > len(b) else "B")
    return _ok(json.dumps({"winner": winner}))


def test_pairwise_win_rates_hand_computed():
    judge = PairwiseJudge(_client(RecordingTransport(_longer_wins)), swap_positions=True, seed=3)
    # (0,1): 0 wins; (0,2): tie; (1,2): 2 wins -> 0: 1.5/2, 1: 0/2, 2: 1.5/2.
    assert judge.score_group("q", ["aaa", "a", "aaa"]) == pytest.approx([0.75, 0.0, 0.75])


def test_pairwise_single_member_group_is_unscorable():
    transport = RecordingTransport(_longer_wins)
    assert PairwiseJudge(_client(transport)).score_group("q", ["only"]) == [None]
    assert transport.calls == []


def test_pairwise_round_robin_respects_max_pairs_and_balance():
    judge = PairwiseJudge(_client(RecordingTransport(_longer_wins)), max_pairs=4)
    schedule = judge.pairs(4)
    assert len(schedule) == 4 and len(set(schedule)) == 4
    counts = [sum(k in pair for pair in schedule) for k in range(4)]
    assert counts == [2, 2, 2, 2]
    assert len(PairwiseJudge(_client(RecordingTransport(_longer_wins))).pairs(5)) == 10


def test_reward_fn_judge_failure_gives_none_not_zero():
    transport = RecordingTransport(lambda t, i: (500, {}, b""))
    reward_fn = make_judge_reward_fn(RubricJudge(_client(transport, max_retries=1)), num_generations=2)
    assert reward_fn(prompts=["p", "p", "q", "q"], completions=["a", "b", "c", "d"]) == [None] * 4


def test_reward_fn_groups_by_prompt_for_pairwise():
    reward_fn = make_judge_reward_fn(PairwiseJudge(_client(RecordingTransport(_longer_wins))))
    scores = reward_fn(prompts=["p", "p", "solo"], completions=["long", "s", "x"])
    assert scores == [1.0, 0.0, None]


def test_llm_judge_environment_registered_and_marks_unscorable():
    good = RubricJudge(_client(RecordingTransport(lambda t, i: _ok(json.dumps(_rubric(10, 10, 10))))))
    env = build_environment("llm_judge", prompt="task", judge=good)
    assert isinstance(env, LLMJudgeEnvironment)
    assert env.reset().observation == "task"
    step = env.step("answer")
    assert step.terminated and step.reward == pytest.approx(1.0) and not step.info["env_error"]

    bad = RubricJudge(_client(RecordingTransport(lambda t, i: (400, {}, b""))))
    failed = LLMJudgeEnvironment("task", judge=bad).step("answer")
    assert failed.info["env_error"] and failed.info["unscorable"]


def _scoring_judge():
    """Rubric judge giving correctness 10 when the response contains the reference, else 0."""

    def handler(text, index):
        match = re.search(r"\[Reference answer\]\n(.*?)\n", text)
        response = text.split("[Response]\n", 1)[1].split("\n\nScore each", 1)[0]
        correct = match is not None and match.group(1) in response
        return _ok(json.dumps(_rubric(10 if correct else 0, 0, 0)))

    transport = RecordingTransport(handler)
    return RubricJudge(_client(transport)), transport


def test_environment_reward_fn_llm_judge_accepts_dataset_columns():
    judge, transport = _scoring_judge()
    reward_fn = make_environment_reward_fn("llm_judge", judge=judge)
    scores = reward_fn(
        prompts=["What is 2+2?", "Capital of France?"],
        completions=["It is 4", "London"],
        reference=["4", "Paris"],
        difficulty=["easy", "easy"],  # unrelated dataset column is dropped, not forwarded to the client
    )
    assert scores == pytest.approx([10 / 10.5, 0.0])
    assert "[Task]\nWhat is 2+2?" in transport.calls[0] or "[Task]\nWhat is 2+2?" in transport.calls[1]


def test_build_environment_with_client_kwargs_mapping_and_row_aliases():
    transport = RecordingTransport(lambda t, i: _ok(json.dumps(_rubric(10, 10, 10))))
    env = build_environment(
        "llm_judge",
        problem_statement="aliased task",
        base_url="http://judge.local/v1",
        model="judge-model",
        client_kwargs={"transport": transport, "max_retries": 0},
    )
    assert env.reset().observation == "aliased task"
    assert env.step("answer").reward == pytest.approx(1.0)
    assert "aliased task" in transport.calls[0]
    with pytest.raises(ValueError):
        build_environment("llm_judge", judge=env.judge, client_kwargs={"max_retries": 0})


def test_judge_reward_fn_reads_references_from_batch():
    judge, _ = _scoring_judge()
    reward_fn = make_judge_reward_fn(judge, num_generations=2, batch=False)
    scores = reward_fn(
        prompts=["q1", "q1", "q2", "q2"],
        completions=["the answer is 7", "no idea", "Paris", "Rome"],
        batch={"reference": ["7", "7", "Paris", "Paris"], "input_ids": object()},
    )
    assert scores == pytest.approx([10 / 10.5, 0.0, 10 / 10.5, 0.0])


def test_judge_reward_fn_without_references_still_scores():
    judge, transport = _scoring_judge()
    reward_fn = make_judge_reward_fn(judge, num_generations=1)
    assert reward_fn(prompts=["q"], completions=["7"], batch={"reference": ["7", "8"]}) == [0.0]
    assert "[Reference answer]" not in transport.calls[0]


def test_judge_reward_fn_opts_into_discarding_unscorable_groups():
    judge, _ = _scoring_judge()
    assert getattr(make_judge_reward_fn(judge), "discard_unscorable", False) is True
