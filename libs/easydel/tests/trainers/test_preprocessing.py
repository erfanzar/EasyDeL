"""Pure-text regression coverage for trainer preprocessing contracts."""

from collections import UserDict
from types import SimpleNamespace

import pytest
from datasets import Dataset
from easydel.data.sources.hf_wrapper import HFDatasetShardedSource
from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationConfig
from easydel.trainers.prm_trainer import PRMTrainer
from easydel.trainers.prompt_transforms import (
    BCOPreprocessTransform,
    KTOPreprocessTransform,
    apply_chat_template_to_preference,
    preprocess_dataset_with_num_proc,
)
from easydel.trainers.prompt_utils import (
    is_conversational_from_value,
    keep_array_and_primitives,
    remove_none_values,
    reverse_openai_format,
)
from easydel.trainers.ray_scaler.distributed_trainer import RayDistributedTrainer
from easydel.trainers.reward_protocol import resolve_reward_weights

pytestmark = pytest.mark.cpu_ok


def test_reward_weight_resolution_honors_protocol_defaults_and_explicit_overrides():
    rewards = [SimpleNamespace(weight=0.3), object()]
    assert resolve_reward_weights(rewards) == [0.3, 1.0]
    assert resolve_reward_weights(rewards, [2.0, 0.0]) == [2.0, 0.0]
    with pytest.raises(ValueError, match="match"):
        resolve_reward_weights(rewards, [1.0])


def test_ray_created_config_keeps_attention_softmax_float32():
    launcher = object.__new__(RayDistributedTrainer)
    launcher.config_variables = dict(RayDistributedTrainer.CONFIG_VARIABLES)
    launcher.config_scaling_variables = {}
    launcher.model_class = SimpleNamespace(config_class=SimpleNamespace)
    config = launcher.create_config(1)
    assert config.attn_softmax_dtype.__name__ == "float32"


class TextTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        ids = [ord(char) for char in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def encode(self, text, **kwargs):
        return self(text)["input_ids"]

    def apply_chat_template(self, messages, **kwargs):
        return "".join(message["content"] for message in messages)


@pytest.mark.parametrize("explicit", [False, True])
def test_on_policy_sampling_defaults_preserve_user_values(explicit):
    overrides = {"generation_top_p": 0.8, "generation_top_k": 7} if explicit else {}
    config = OnPolicyDistillationConfig(top_p=0.91, top_k=23, **overrides)
    assert config.top_p == 0.91
    assert config.top_k == 23
    assert config.generation_top_p == (0.8 if explicit else 0.91)
    assert config.generation_top_k == (7 if explicit else 23)


def test_kto_and_bco_preserve_unpaired_tokenization_contract():
    options = dict(
        tokenizer=TextTokenizer(),
        max_prompt_length=3,
        max_completion_length=3,
        add_special_tokens=True,
        embedding_tokenizer=TextTokenizer(),
    )
    row = {"prompt": "abcd", "completion": "xy", "label": True}
    expected = {
        "prompt_input_ids": [98, 99, 100],
        "prompt_attention_mask": [1, 1, 1],
        "completion_input_ids": [98, 99, 100, 120, 121, 2],
        "completion_attention_mask": [1] * 6,
        "completion_labels": [-100, -100, -100, 120, 121, 2],
        "label": True,
        "embedding_input_ids": [97, 98, 99, 100],
        "embedding_attention_mask": [1] * 4,
    }
    assert KTOPreprocessTransform(**options)(row) == expected
    assert list(BCOPreprocessTransform(**options)(row)) == [expected]
    paired = {"prompt": "abcd", "chosen": "xy", "rejected": "z"}
    chosen, rejected = BCOPreprocessTransform(**options)(paired)
    assert chosen == expected
    assert rejected["completion_labels"] == [-100, -100, -100, 122, 2]
    assert rejected["label"] is False


def test_prm_sources_keep_all_eval_labels_and_only_last_training_label():
    trainer = object.__new__(PRMTrainer)
    trainer.processing_class = TextTokenizer()
    trainer.arguments = SimpleNamespace(
        step_separator="\n",
        max_length=8,
        max_completion_length=None,
        train_on_last_step_only=True,
        pad_to_multiple_of=None,
        sequence_packing=False,
    )
    rows = Dataset.from_list([{"prompt": "p", "completions": ["a", "b"], "labels": [0, 1]}])
    trainer._train_source = HFDatasetShardedSource(rows)
    trainer._eval_source = HFDatasetShardedSource(rows)
    trainer._apply_preprocess_transforms()
    train = next(trainer._train_source.open_shard(trainer._train_source.shard_names[0]))
    evaluation = next(trainer._eval_source.open_shard(trainer._eval_source.shard_names[0]))
    assert [value for value in train["labels"] if value != -100] == [1]
    assert [value for value in evaluation["labels"] if value != -100] == [0, 1]


def test_prm_completion_exhausts_length_budget_without_prompt():
    result = PRMTrainer.tokenize_row(
        {"prompt": "long", "completions": ["abcd"], "labels": [1]},
        tokenizer=TextTokenizer(),
        max_length=3,
    )
    assert result == {"input_ids": [97, 98, 99], "attention_mask": [1, 1, 1], "labels": [-100, -100, 1]}


def test_prm_string_model_rejected_before_loading():
    from easydel.trainers.prm_trainer import PRMConfig

    with pytest.raises(ValueError, match="do not load token-classification model from string ids"):
        PRMTrainer(PRMConfig(), model="not-a-local-model", processing_class=TextTokenizer())


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("pad_token", [None, "<pad>"])
def test_prm_initialization_applies_shared_tokenizer_overrides_and_pad_fallback(monkeypatch, wrapped, pad_token):
    from easydel.trainers.prm_trainer import PRMConfig
    from easydel.trainers.trainer import Trainer

    monkeypatch.setattr(Trainer, "__init__", lambda self, **kwargs: None)
    tokenizer = SimpleNamespace(eos_token="old-eos", pad_token=None)
    processor = SimpleNamespace(tokenizer=tokenizer) if wrapped else tokenizer
    state = object()
    model = SimpleNamespace(to_state=lambda **kwargs: state)
    arguments = PRMConfig(eos_token="<eos>", pad_token=pad_token, disable_dropout=False)
    trainer = PRMTrainer(arguments, model=model, processing_class=processor)
    assert trainer.processing_class is processor
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.pad_token == (pad_token if pad_token is not None else "<eos>")


def test_documented_prompt_utilities_remain_available():
    assert reverse_openai_format([{"role": "user", "content": [{"type": "text", "text": "hello"}]}]) == {
        "role": "user",
        "content": "hello",
    }
    assert remove_none_values([None, UserDict({"keep": 3, "drop": None}), {"nested": [None, 4]}]) == [
        {"keep": 3},
        {"nested": [4]},
    ]
    assert keep_array_and_primitives({"keep": 3, "drop": "text", "nested": [True, "drop"]}) == {
        "keep": 3,
        "nested": [True],
    }
    assert is_conversational_from_value({"conversations": [{"from": "human", "value": "hello"}]})
    assert not is_conversational_from_value({"conversations": [{"role": "user", "content": "hello"}]})
    assert not is_conversational_from_value({"conversations": []})
    assert apply_chat_template_to_preference({"prompt": "p", "chosen": "a", "rejected": "b"}, TextTokenizer()) == {
        "prompt": "p",
        "chosen": "a",
        "rejected": "b",
    }


def test_dataset_num_proc_preprocesses_all_rows():
    rows = Dataset.from_list(
        [
            {"prompt": "p", "completion": "a", "label": True},
            {"prompt": "q", "completion": "b", "label": False},
        ]
    )
    result = preprocess_dataset_with_num_proc(rows, KTOPreprocessTransform(TextTokenizer()), 2)
    assert result["completion_labels"] == [[-100, 97, 2], [-100, 98, 2]]
    assert "prompt" not in result.column_names


@pytest.mark.parametrize("count", [0, -1, True])
def test_dataset_num_proc_rejects_invalid_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        preprocess_dataset_with_num_proc(Dataset.from_list([{"value": 1}]), dict, count)
