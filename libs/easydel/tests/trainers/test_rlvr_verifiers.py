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

import pytest
from easydel.trainers.rlvr_trainer import (
    CompositeVerifier,
    ContainsVerifier,
    ExactMatchVerifier,
    JSONVerifier,
    LengthPenaltyVerifier,
    MathVerifier,
    MultipleChoiceVerifier,
    ReasoningFormatVerifier,
    RepetitionPenaltyVerifier,
)
from easydel.trainers.rlvr_trainer.reward_verifiers import _get_column


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


def test_verifiers_repeat_prompt_answers_for_each_generation():
    # GRPO returns prompt-major completions; the dataloader provides one
    # answer per prompt, not one answer per generated completion.
    batch = {"answer": ["1", "2"]}
    completions = [r"\boxed{1}", r"\boxed{1}", r"\boxed{2}", r"\boxed{2}"]
    assert MathVerifier()(completions=completions, batch=batch) == [1.0] * 4
    assert ExactMatchVerifier()(completions=["1", "1", "2", "2"], batch=batch) == [1.0] * 4
    assert MultipleChoiceVerifier()(completions=["A", "A", "B", "B"], batch={"answer": ["A", "B"]}) == [1.0] * 4


def test_reward_metadata_that_does_not_divide_completions_is_rejected():
    # 3 answers for 4 completions cannot be aligned prompt-major; padding with None would silently
    # score the last completion against no gold answer.
    batch = {"answer": ["1", "2", "3"]}
    with pytest.raises(ValueError, match="does not divide"):
        MathVerifier()(completions=["1", "2", "3", "3"], batch=batch)
    # Dividing lengths, equal lengths and absent columns keep working.
    assert _get_column({"answer": ["1", "2"]}, "answer", 4) == ["1", "1", "2", "2"]
    assert _get_column({"answer": ["1", "2"]}, "answer", 2) == ["1", "2"]
    assert _get_column({}, "answer", 3) == [None, None, None]


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


def test_math_verifier_comma_nested_boxed_last_match_and_none_gold():
    v = MathVerifier()
    # Bare comma in trailing prose must not match as a number.
    assert v(completions=["total is 42. Yes, right."], batch={"answer": ["42"]}) == [1.0]
    # Balanced nested braces in \boxed{\frac{1}{2}} and signed fractions.
    assert v(
        completions=[r"Therefore \boxed{\frac{1}{2}}", r"Result: \boxed{-\frac{3}{4}}"],
        batch={"answer": ["0.5", r"\boxed{-\frac{3}{4}}"]},
    ) == [1.0, 1.0]
    # Last \boxed{...} wins over earlier intermediate boxes.
    assert v(completions=[r"First \boxed{10}, wait actually \boxed{42}."], batch={"answer": ["42"]}) == [1.0]
    # Empty/unparseable \boxed{} falls through to #### or numeric fallback.
    assert v(
        completions=[r"\boxed{} so #### 42", r"\boxed{invalid} hence 42"],
        batch={"answer": ["42", "42"]},
    ) == [1.0, 1.0]
    # None gold answers stay None (not converted to "None") and score 0.0.
    assert v._get_gold_answers({"answer": [None, "42"]}, 2) == [None, "42.0"]
    assert v(completions=["42", "42"], batch={"answer": [None, "42"]}) == [0.0, 1.0]


def test_multiple_choice_verifier_avoids_prose_false_positives_and_takes_last_match():
    v = MultipleChoiceVerifier(answer_key="answer")
    comps = [
        "The answer can be derived: B",
        "Answer: a simple equation gives B",
        "The answer is correct, so D",
        "First I thought the answer is A, actually the answer is C.",
        r"Wait \boxed{A}, no \boxed{E}",
        "the answer is b",
    ]
    golds = ["B", "B", "D", "C", "E", "B"]
    assert v(completions=comps, batch={"answer": golds}) == [1.0] * len(comps)


def test_length_penalty_verifier_configurable_max_length():
    # Default must not hardcode max_length=5000 and zero out long targets.
    v_long = LengthPenaltyVerifier(target_length=6000)
    assert v_long(completions=["a" * 6000]) == [1.0]

    # Explicit max_length caps completions above that bound to 0.0.
    v_capped = LengthPenaltyVerifier(target_length=500, min_length=10, max_length=600)
    assert v_capped(completions=["a" * 500, "a" * 600, "a" * 601]) == [1.0, 0.8, 0.0]


def test_exact_match_verifier_handles_multi_group_extract_pattern():
    v_alt = ExactMatchVerifier(answer_key="answer", extract_pattern=r"Answer:\s*(\w+)|Result:\s*(\w+)")
    assert v_alt(
        completions=["Reasoning... Answer: Paris", "Reasoning... Result: Berlin", "No match"],
        batch={"answer": ["paris", "berlin", "rome"]},
    ) == [1.0, 1.0, 0.0]

    v_prefix = ExactMatchVerifier(answer_key="answer", extract_pattern=r"(Final\s+)?Answer:\s*(.+)")
    assert v_prefix(completions=["Final Answer: 42"], batch={"answer": ["42"]}) == [1.0]


def test_json_verifier_prefers_explicit_json_fence_over_earlier_non_json_fence():
    v = JSONVerifier(required_keys=["name", "age"])
    comp_with_python_first = (
        "Helper code:\n"
        "```python\n"
        'd = {"not": "valid json"}\n'
        "```\n"
        "Final JSON:\n"
        "```json\n"
        '{"name": "alice", "age": 30}\n'
        "```"
    )
    comp_with_plain_fence_first = '```\n{"other": 1}\n```\nThen:\n```json\n{"name": "bob", "age": 25}\n```'
    assert v(completions=[comp_with_python_first, comp_with_plain_fence_first]) == [1.0, 1.0]


def test_composite_verifier_guards_none_scores():
    def partial_verifier(*, completions, **kwargs):
        del completions, kwargs
        return [1.0, None]

    def always_verifier(*, completions, **kwargs):
        del completions, kwargs
        return [0.5, 0.8]

    def none_verifier(*, completions, **kwargs):
        del completions, kwargs
        return None

    comp = CompositeVerifier([partial_verifier, always_verifier, none_verifier], weights=[1.0, 0.5, 2.0])
    assert comp(completions=["a", "b"]) == pytest.approx([1.25, 0.4])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
