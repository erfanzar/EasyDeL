"""Host-only regressions for preference boundaries and reward payload contracts."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from datasets import Dataset
from easydel.trainers.odds_ratio_preference_optimization_trainer import ORPOConfig
from easydel.trainers.packer import pack_sequences
from easydel.trainers.process_encoder import (
    EncoderBinding,
    EncoderProcessor,
    PixelLayout,
    ProcessEncoderConfig,
    VisionSelection,
)
from easydel.trainers.prompt_transforms import ORPOPreprocessTransform, SFTPreprocessTransform
from easydel.trainers.prompt_utils import extract_prompt
from easydel.trainers.proximal_policy_optimization_trainer import PPOConfig, PPOTrainer
from easydel.trainers.reward_protocol import (
    RewardProtocol,
    align_reward_sidechannels,
    normalize_reward_state,
    render_reward_texts,
)

from .test_preprocessing import TextTokenizer

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("chosen", "rejected", "prompt"),
    [
        ("abc", "abcdef", "abc"),
        ("abc", "abc", "abc"),
        ("a ", "b ", ""),
        ("", "b", ""),
        ("hello good", "hello bad", "hello"),
        (
            [{"role": "user", "content": "q"}],
            [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            [{"role": "user", "content": "q"}],
        ),
    ],
)
def test_shared_prompt_preserves_complete_prefix_and_distinct_suffixes(chosen, rejected, prompt):
    result = extract_prompt({"chosen": chosen, "rejected": rejected})
    assert result["prompt"] == prompt
    assert result["prompt"] + result["chosen"] == chosen
    assert result["prompt"] + result["rejected"] == rejected


class TruncatingTokenizer(TextTokenizer):
    truncation_side = "right"
    padding_side = "right"
    eos_token = "!"

    def __call__(self, text, *, truncation=False, max_length=None, padding=False, **kwargs):
        ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            ids = ids[-max_length:] if self.truncation_side == "left" else ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pads = [0] * (max_length - len(ids))
            ids = pads + ids if self.padding_side == "left" else ids + pads
            attention = pads + attention if self.padding_side == "left" else attention + pads
        return {"input_ids": ids, "attention_mask": attention}


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize(
    ("prompt", "completion", "mode", "expected_text", "expected_mask"),
    [
        ("abcdef", "XY", "keep_end", "efXY", [0, 0, 1, 1]),
        ("abcdef", "WXYZ", "keep_end", "WXYZ", [1, 1, 1, 1]),
        ("a", "B", "keep_end", "aB", [0, 1]),
        ("abc", "XY", "keep_start", "abcX", [0, 0, 0, 1]),
    ],
)
def test_sft_completion_mask_follows_retained_tokens(
    batched, padding_side, prompt, completion, mode, expected_text, expected_mask
):
    tokenizer = TruncatingTokenizer()
    tokenizer.padding_side = padding_side
    transform = SFTPreprocessTransform(tokenizer, max_length=4, mask_prompt=True, add_eos=False, truncation_mode=mode)
    row = {"prompt": prompt, "completion": completion}
    result = transform.map_batch([row])[0] if batched else transform(row)
    active_ids = [token for token, active in zip(result["input_ids"], result["attention_mask"], strict=True) if active]
    active_mask = [
        mask for mask, active in zip(result["completion_mask"], result["attention_mask"], strict=True) if active
    ]
    assert active_ids == [ord(char) for char in expected_text]
    assert active_mask == expected_mask
    assert all(
        not mask for mask, active in zip(result["completion_mask"], result["attention_mask"], strict=True) if not active
    )
    assert tokenizer.truncation_side == "right"


@pytest.mark.parametrize("budget", [0, -1])
def test_orpo_rejects_nonpositive_completion_budgets(budget):
    with pytest.raises(ValueError, match="positive max_completion_length"):
        ORPOConfig(max_length=4, max_prompt_length=4, max_completion_length=budget)
    transform = ORPOPreprocessTransform(TextTokenizer(), max_prompt_length=4, max_completion_length=budget)
    with pytest.raises(ValueError, match="positive max_completion_length"):
        transform({"prompt": "abcd", "chosen": "x", "rejected": "y"})


def test_orpo_rejects_derived_empty_budget_and_ignored_pretokenized_pairs():
    with pytest.raises(ValueError, match="positive max_completion_length"):
        ORPOConfig(max_length=4, max_prompt_length=4)
    transform = ORPOPreprocessTransform(TextTokenizer())
    with pytest.raises(ValueError, match="no scoreable completion"):
        transform({"prompt_input_ids": [1], "chosen_labels": [-100], "chosen_attention_mask": [1]})


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("rows", [[[1, 2, 3, 4]], [[], [1]], [[1, 2, 3, 4, 5, 6, 7, 8, 9]], [[1], [2], [3, 4, 5, 6]]])
def test_packing_boundary_rows_have_fixed_capacity_and_preserve_tokens(reset, rows):
    dataset = Dataset.from_dict({"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]})
    result = pack_sequences(dataset, max_length=4, reset_position_ids=reset)
    recovered = []
    for row in result:
        assert len(row["input_ids"]) == len(row["attention_mask"]) == len(row["position_ids"]) == 4
        assert any(row["attention_mask"])
        recovered.extend(token for token, active in zip(row["input_ids"], row["attention_mask"], strict=True) if active)
    assert recovered == [token for row in rows for token in row]


def test_reward_metadata_alignment_reaches_per_example_protocol():
    class AnswerReward(RewardProtocol):
        def compute(self, *, completion, answer):
            return float(completion == answer)

    payload = {"answer": ["a0", "a1"], "already_aligned": [0, 1, 2, 3], "shared": "x"}
    aligned = align_reward_sidechannels(payload, prompt_count=2, completion_count=4)
    assert aligned["answer"] == ["a0", "a0", "a1", "a1"]
    assert aligned["already_aligned"] is payload["already_aligned"]
    assert payload["answer"] == ["a0", "a1"]
    assert AnswerReward()(prompts=["p0", "p0", "p1", "p1"], completions=["a0", "bad", "a1", "bad"], **aligned) == [
        1,
        0,
        1,
        0,
    ]
    with pytest.raises(ValueError, match="Reward field"):
        align_reward_sidechannels({"answer": ["a", "b", "c"]}, prompt_count=2, completion_count=4)


def test_reward_texts_render_conversations_and_keep_plain_text():
    tokenizer = TextTokenizer()
    assert render_reward_texts([[{"role": "user", "content": "question"}], "p"], ["answer", "c"], tokenizer) == [
        "questionanswer",
        "pc",
    ]


@pytest.mark.parametrize("bound", [False, True])
def test_reward_normalization_keeps_custom_forward_results(bound):
    @dataclass
    class State:
        apply_fn: object

        def replace(self, **kwargs):
            return replace(self, **kwargs)

    def bound_forward(gs, go, batch):
        return (gs, go, batch)

    def unbound_forward(gd, gs, go, batch):
        return (gd, gs, go, batch)

    original = bound_forward if bound else unbound_forward
    state = normalize_reward_state(State(original))
    result = state.apply_fn("definition", "parameters", "other", {"answer": 3})
    assert result == (
        ("parameters", "other", {"answer": 3}) if bound else ("definition", "parameters", "other", {"answer": 3})
    )


def test_ppo_rejects_scalar_processor_with_multiple_rewards(monkeypatch):
    monkeypatch.setattr(PPOTrainer, "requires_value_head", False)
    policy = SimpleNamespace(to_state=lambda **kwargs: SimpleNamespace())
    with pytest.raises(ValueError, match="number of reward processing classes"):
        PPOTrainer(
            PPOConfig(kl_coef=0),
            policy,
            [lambda: None, lambda: None],
            processing_class=TextTokenizer(),
            reward_processing_classes=TextTokenizer(),
        )


def test_structured_encoder_output_preserves_in_model_deepstack_path(monkeypatch):
    import easydel.trainers.process_encoder as encoder

    config = ProcessEncoderConfig(enabled=True)
    binding = EncoderBinding("visual", "image_embeds", "get_image_features", frozenset({"pixel_values"}))
    processor = EncoderProcessor(SimpleNamespace(), config, binding)
    selection = VisionSelection(SimpleNamespace(size=1), 1, 1, PixelLayout.IMAGE_MAJOR)
    monkeypatch.setattr(encoder, "resolve_pixel_layout", lambda *args: PixelLayout.IMAGE_MAJOR)
    monkeypatch.setattr(encoder, "select_vision_rows", lambda *args, **kwargs: selection)
    monkeypatch.setattr(encoder, "pack_vision_inputs", lambda batch, selection: batch)
    features = SimpleNamespace(shape=(2, 4))
    residual = SimpleNamespace(shape=(2, 4))
    monkeypatch.setattr(processor, "_encode", lambda *args: ((features,), [residual]))
    batch = {"pixel_values": object(), "image_grid_thw": [[1, 2, 4]], "input_ids": [[1, 1]]}
    assert processor.process(batch) is batch
    assert "image_embeds" not in batch


@pytest.mark.parametrize("generic_call", [False, True])
def test_default_reward_forward_filters_actual_entrypoint(monkeypatch, generic_call):
    import easydel.trainers.reward_protocol as rewards

    class DirectCall:
        def __call__(self, input_ids, attention_mask):
            return {"logits": input_ids * attention_mask}

        def forward(self, *args, **kwargs):
            raise AssertionError("unused generic forward must not be invoked")

    class WrappedForward:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, input_ids, attention_mask):
            return {"logits": input_ids * attention_mask}

    module = WrappedForward() if generic_call else DirectCall()
    monkeypatch.setattr(rewards.spx, "bind", lambda *args: module)
    state = SimpleNamespace(merge=lambda *args, **kwargs: {})
    result = rewards.reward_model_forward(None, state, None, {"input_ids": 3, "attention_mask": 2, "unused": 99})
    assert result == {"logits": 6}


def test_packer_empty_rows_and_invalid_budgets():
    dataset = Dataset.from_dict({"input_ids": [[]], "attention_mask": [[]]})
    assert len(pack_sequences(dataset, max_length=4)) == 0
    for budget in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            pack_sequences(dataset, max_length=budget)
