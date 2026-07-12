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

"""Unit tests for the pre-built RLVR reward verifiers.

Pure-Python (no JAX / no model), so they run fast and validate the reward
math and extraction of each verifier at its public callable surface.
"""

from easydel.trainers.rlvr_trainer import (
    CompositeVerifier,
    ContainsVerifier,
    ExactMatchVerifier,
    JSONVerifier,
    MultipleChoiceVerifier,
    ReasoningFormatVerifier,
    RepetitionPenaltyVerifier,
)


def test_exact_match_normalized_and_extracted():
    # default is strict (case-insensitive, whitespace-normalized, punctuation kept)
    v = ExactMatchVerifier(answer_key="answer")
    batch = {"answer": ["Paris", "Berlin"]}
    assert v(completions=["  PARIS ", "London"], batch=batch) == [1.0, 0.0]

    # strip_punctuation=True tolerates trailing punctuation on prose answers
    v_punct = ExactMatchVerifier(answer_key="answer", strip_punctuation=True)
    assert v_punct(completions=["  paris. ", "London"], batch=batch) == [1.0, 0.0]

    # regex extraction of the answer span
    v2 = ExactMatchVerifier(answer_key="answer", extract_pattern=r"Answer:\s*(.+)")
    out = v2(completions=["reasoning... Answer: 42"], batch={"answer": ["42"]})
    assert out == [1.0]
    # missing gold -> 0.0
    assert v(completions=["paris"], batch=None) == [0.0]


def test_multiple_choice_extraction():
    v = MultipleChoiceVerifier(answer_key="answer")
    batch = {"answer": ["B", "C", "A", "D"]}
    comps = [
        "The answer is (B).",
        r"so \boxed{C}",
        "I think A is correct",  # lone-letter fallback
        "the answer is B",  # gold is D -> wrong
    ]
    assert v(completions=comps, batch=batch) == [1.0, 1.0, 1.0, 0.0]


def test_contains_required_forbidden_graded():
    v = ContainsVerifier(required=["import", "def"], mode="all")
    assert v(completions=["import os\ndef f(): ...", "def f(): ..."]) == [1.0, 0.0]

    v_any = ContainsVerifier(required=["cat", "dog"], mode="any")
    assert v_any(completions=["a dog", "a fish"]) == [1.0, 0.0]

    v_forbid = ContainsVerifier(required=["ok"], forbidden=["sorry"])
    assert v_forbid(completions=["ok sure", "sorry, ok"]) == [1.0, 0.0]

    v_graded = ContainsVerifier(required=["a", "b", "c", "d"], graded=True)
    assert v_graded(completions=["a b only"]) == [0.5]


def test_json_validity_and_required_keys():
    v = JSONVerifier()
    assert v(completions=['{"a": 1}', "not json", '```json\n{"x": 2}\n```']) == [1.0, 0.0, 1.0]

    v_keys = JSONVerifier(required_keys=["name", "age"])
    assert v_keys(completions=['{"name": "x", "age": 3}', '{"name": "x"}']) == [1.0, 0.0]

    v_graded = JSONVerifier(required_keys=["name", "age"], graded=True)
    assert v_graded(completions=['{"name": "x"}']) == [0.5]


def test_reasoning_format_grades():
    v = ReasoningFormatVerifier()
    full = "<think>step by step</think> the result is \\boxed{7}"
    think_only = "<think>reasoning here</think> done"
    answer_only = "no reasoning, \\boxed{7}"
    empty_think = "<think></think> \\boxed{7}"  # empty think -> only answer credit
    nothing = "just some text"
    out = v(completions=[full, think_only, answer_only, empty_think, nothing])
    assert out == [1.0, 0.5, 0.5, 0.5, 0.0]


def test_repetition_penalty_distinct_ngrams():
    v = RepetitionPenaltyVerifier(ngram=2)
    diverse = "the quick brown fox jumps"  # all bigrams unique -> 1.0
    looped = "go go go go go go"  # every bigram is ("go","go") -> 1/5
    out = v(completions=[diverse, looped, "short"])
    assert out[0] == 1.0
    assert 0.0 < out[1] < 0.5
    assert out[2] == 1.0  # shorter than one n-gram -> 1.0


def test_composite_weighted_sum():
    fmt = ReasoningFormatVerifier()
    contains = ContainsVerifier(required=["answer"])
    comp = CompositeVerifier([fmt, contains], weights=[1.0, 0.5])
    # full format (1.0*1.0) + contains "answer" (0.5*1.0) = 1.5
    out = comp(completions=["<think>x</think> the answer is \\boxed{1}"])
    assert out == [1.5]

    # default weights are 1.0 each
    comp2 = CompositeVerifier([contains, contains])
    assert comp2(completions=["answer"]) == [2.0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
