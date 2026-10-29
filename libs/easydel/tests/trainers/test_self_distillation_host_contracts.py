"""Host-only regressions for rollout metadata and conversational targets."""

import numpy as np
import pytest
from easydel.trainers.rlvr_trainer.reward_verifiers import (
    CodeVerifier,
    ContainsVerifier,
    ExactMatchVerifier,
    FormatVerifier,
    JSONVerifier,
    LengthPenaltyVerifier,
    MathVerifier,
    MultipleChoiceVerifier,
    ReasoningFormatVerifier,
    RepetitionPenaltyVerifier,
)
from easydel.trainers.sdft_trainer import SDFTConfig, SDFTTrainer
from easydel.trainers.self_distillation_policy_optimization import SDPOConfig, SDPOTrainer
from easydel.trainers.ssd_trainer import SSDConfig
from easydel.trainers.tpo_trainer.tpo_preprocess import TPOPreprocessTransform
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("config_cls", [SDPOConfig, SDFTConfig, SSDConfig])
@pytest.mark.parametrize("feedback_length", [4, 8])
def test_teacher_budget_rejects_nonpositive_completion_length(config_cls, feedback_length):
    with pytest.raises(ValueError, match="positive completion budget"):
        config_cls(max_length=16, max_prompt_length=12, max_completion_length=4, max_feedback_length=feedback_length)


@pytest.mark.parametrize("config_cls", [SDPOConfig, SDFTConfig, SSDConfig])
def test_teacher_budget_preserves_positive_remaining_completion_length(config_cls):
    config = config_cls(max_length=16, max_prompt_length=8, max_completion_length=8, max_feedback_length=4)
    assert config.max_completion_length == 4
    assert config.max_prompt_length + config.max_feedback_length + config.max_completion_length == 16


@pytest.mark.parametrize("config_cls", [SDPOConfig, SDFTConfig, SSDConfig])
def test_default_teacher_budget_keeps_generation_positive(config_cls):
    config = config_cls()
    assert config.max_completion_length == 256
    assert config.max_length == config.max_prompt_length + config.max_feedback_length + config.max_completion_length


@pytest.mark.parametrize("config_cls", [SDPOConfig, SDFTConfig, SSDConfig])
def test_explicit_teacher_headroom_is_not_lost_in_parent_normalization(config_cls):
    config = config_cls(max_length=32, max_prompt_length=4, max_completion_length=3, max_feedback_length=8)
    assert config.max_length == 32
    assert config.max_completion_length == 3


@pytest.mark.parametrize("contexts", [["ctxA", "ctxB"], ["ctxA", "ctxA", "ctxB", "ctxB"]])
def test_sdft_captured_context_tracks_prompt_major_rollouts(monkeypatch, contexts):
    def initialize(self, *, arguments, feedback_func, **kwargs):
        self.arguments = arguments
        self.feedback_func = feedback_func

    def preprocess(self, *, state, batch, is_train):
        return self.feedback_func(["A", "A", "B", "B"], ["one", "two", "three", "four"], [0.0] * 4), {}

    monkeypatch.setattr(SDPOTrainer, "__init__", initialize)
    monkeypatch.setattr(SDPOTrainer, "_preprocess_batch_input", preprocess)
    trainer = SDFTTrainer(SDFTConfig(teacher_prompt_template="{prompt}|{privileged_context}"), model=None)
    result, _ = trainer._preprocess_batch_input(None, {"privileged_context": contexts}, is_train=True)
    assert result == ["A|ctxA", "A|ctxA", "B|ctxB", "B|ctxB"]
    # No previous batch context survives into a subsequent feedback invocation.
    assert trainer.feedback_func(["C"], ["answer"], [0.0]) == ["C|"]


def test_sdft_rejects_incompatible_context_lengths(monkeypatch):
    def initialize(self, *, arguments, feedback_func, **kwargs):
        self.arguments = arguments
        self.feedback_func = feedback_func

    monkeypatch.setattr(SDPOTrainer, "__init__", initialize)
    trainer = SDFTTrainer(SDFTConfig(teacher_prompt_template="{prompt}|{privileged_context}"), model=None)
    trainer._sdft_privileged_context = ["A", "B", "C"]
    with pytest.raises(ValueError, match="cannot align prompt-major"):
        trainer.feedback_func(["p"] * 4, ["answer"] * 4, [0.0] * 4)


@pytest.mark.parametrize(
    "verifier, text, batch",
    [
        (MathVerifier(), "#### 42", {"answer": [42]}),
        (ExactMatchVerifier(), "yes", {"answer": ["yes"]}),
        (MultipleChoiceVerifier(), "Answer: B", {"answer": ["B"]}),
        (JSONVerifier(), '{"ok": true}', None),
        (ContainsVerifier(required=["yes"]), "yes", None),
        (FormatVerifier(r"yes", require_full_match=True), "yes", None),
        (LengthPenaltyVerifier(target_length=3, min_length=0), "yes", None),
        (ReasoningFormatVerifier(), "<think>reason</think>#### 42", None),
        (RepetitionPenaltyVerifier(), "one two three four", None),
    ],
)
@pytest.mark.parametrize("content_blocks", [False, True])
def test_verifiers_score_message_content_not_python_repr(verifier, text, batch, content_blocks):
    content = [{"type": "text", "text": text}] if content_blocks else text
    chat = [[{"role": "assistant", "content": content}]]
    assert verifier(completions=[text], batch=batch) == [1.0]
    assert verifier(completions=chat, batch=batch) == [1.0]


def test_code_verifier_executes_identical_plain_and_chat_code():
    text = "```python\nx = 1\n```"
    verifier = CodeVerifier(timeout=2)
    batch = {"tests": ["assert x == 1"]}
    assert verifier(completions=[text], batch=batch) == [1.0]
    assert verifier(completions=[[{"role": "assistant", "content": text}]], batch=batch) == [1.0]


def test_json_null_is_valid_unless_object_keys_are_required():
    completions = ["null", "```json\nnull\n```", "false", "0", "{}", "invalid"]
    assert JSONVerifier()(completions=completions) == [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert JSONVerifier(required_keys=["ok"])(completions=completions) == [0.0] * 6


@pytest.mark.parametrize("answer", [42, 42.0, np.asarray(42)])
def test_reward_numeric_scalar_metadata_broadcasts(answer):
    assert MathVerifier()(completions=["42"] * 4, batch={"answer": answer}) == [1.0] * 4


def test_reward_sequence_metadata_stays_prompt_major_and_rejects_mismatch():
    verifier = MathVerifier()
    assert verifier(completions=["42", "42", "7", "7"], batch={"answer": [42, 7]}) == [1.0] * 4
    with pytest.raises(ValueError, match="does not divide"):
        verifier(completions=["42"] * 4, batch={"answer": [42, 7, 3]})


def _tokenizer():
    vocab = {
        token: index
        for index, token in enumerate(["[UNK]", "user", "question", "assistant", "good", "bad", "gold", "END", "[EOS]"])
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message.role }} {{ message.content }} END {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    return tokenizer


@pytest.mark.parametrize("explicit_prompt", [True, False])
def test_tpo_renders_all_three_conversational_branches(explicit_prompt):
    prompt = [{"role": "user", "content": "question"}]
    row = {
        branch: [{"role": "assistant", "content": text}]
        for branch, text in [("chosen", "good"), ("rejected", "bad"), ("reference", "gold")]
    }
    if explicit_prompt:
        row["prompt"] = prompt
    else:
        row = {branch: prompt + suffix for branch, suffix in row.items()}
    output = TPOPreprocessTransform(_tokenizer())(row)
    assert output["prompt_input_ids"] == [1, 2, 7, 3]
    for branch, token in [("chosen", 4), ("rejected", 5), ("reference", 6)]:
        assert output[f"{branch}_input_ids"] == [1, 2, 7, 3, token, 7, 8]
        assert output[f"{branch}_labels"] == [-100, -100, -100, -100, token, 7, 8]


def test_tpo_rejects_reference_with_different_implicit_conversation():
    prompt = [{"role": "user", "content": "question"}]
    row = {
        branch: [*prompt, {"role": "assistant", "content": text}]
        for branch, text in [("chosen", "good"), ("rejected", "bad"), ("reference", "gold")]
    }
    row["reference"][0] = {"role": "user", "content": "different question"}
    with pytest.raises(ValueError, match=r"reference.*conversation"):
        TPOPreprocessTransform(_tokenizer())(row)
