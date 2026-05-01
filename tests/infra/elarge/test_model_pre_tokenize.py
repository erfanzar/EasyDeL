from __future__ import annotations

import json
from pathlib import Path

import pytest

from easydel.data.execution import pipeline as pipeline_mod
from easydel.infra.elarge.builders import _create_source_from_inform
from easydel.infra.elarge.model import eLargeModel


class SimpleTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

    def __call__(
        self,
        text,
        max_length=None,
        truncation=True,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
        return_attention_mask=True,
        **kwargs,
    ):
        if isinstance(text, list):
            rows = [self._encode_one(item, max_length=max_length, truncation=truncation) for item in text]
            return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}

        tokens = self._encode_one(text, max_length=max_length, truncation=truncation)
        result = {"input_ids": tokens}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(tokens)
        return result

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        return_attention_mask=True,
        **kwargs,
    ):
        text = "".join(message.get("content", "") for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        tokens = self._encode_one(text, max_length=kwargs.get("max_length"), truncation=kwargs.get("truncation", True))
        if tokenize:
            return tokens
        if return_dict:
            result = {"input_ids": tokens}
            if return_attention_mask:
                result["attention_mask"] = [1] * len(tokens)
            return result
        return text

    @staticmethod
    def _encode_one(text, max_length=None, truncation=True):
        tokens = [ord(char) % 100 for char in str(text)]
        if max_length and truncation:
            tokens = tokens[:max_length]
        return tokens


def _read_first_jsonl(stats):
    assert stats.output_paths
    with Path(stats.output_paths[0]).open() as handle:
        return json.loads(handle.readline())


def _elm_with_tokenizer(config):
    elm = eLargeModel(config)
    elm._tokenizer = SimpleTokenizer()
    return elm


def test_pre_tokenize_sft_streams_mixture_to_jsonl(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"source_text": "hello"}\n{"source_text": "world"}\n',
        encoding="utf-8",
    )

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [
                    {
                        "type": "jsonl",
                        "data_files": str(data_path),
                        "content_field": "source_text",
                    }
                ],
                "text_target_field": "text",
                "save": {"format": "json"},
            },
            "trainer": {"trainer_type": "sft", "max_length": 8},
        }
    )

    stats = elm.pre_tokenize(tmp_path / "sft-out", trainer_type="sft", compression=None, show_progress=False)

    assert stats.num_examples == 2
    assert str(tmp_path / "sft-out" / "sft-SimpleTokenizer-MXL8-PLNA-CLNA-train") in stats.output_paths[0]
    row = _read_first_jsonl(stats)
    assert set(row) == {"input_ids", "attention_mask"}
    assert len(row["input_ids"]) <= 8


def test_pre_tokenize_dpo_uses_preference_transform(tmp_path):
    data_path = tmp_path / "pref.jsonl"
    data_path.write_text(
        '{"prompt": "2+2=", "chosen": "4", "rejected": "5"}\n',
        encoding="utf-8",
    )

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [{"type": "jsonl", "data_files": str(data_path)}],
                "save": {"format": "jsonl", "num_shards": 1},
            },
            "trainer": {
                "trainer_type": "sft",
                "max_length": 32,
                "max_prompt_length": 6,
                "max_completion_length": 4,
            },
        }
    )

    stats = elm.pre_tokenize(tmp_path / "dpo-out", trainer_type="dpo", compression=None, show_progress=False)

    assert stats.num_examples == 1
    assert str(tmp_path / "dpo-out" / "dpo-SimpleTokenizer-MXL32-PL6-CL4-pref") in stats.output_paths[0]
    row = _read_first_jsonl(stats)
    assert {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"} <= set(row)
    assert len(row["prompt_input_ids"]) <= 6
    prompt_token_count = sum(row["prompt_attention_mask"])
    assert row["chosen_labels"][:prompt_token_count] == [-100] * prompt_token_count


def test_pre_tokenize_infers_trainer_type_and_output_path(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "distill me"}\n', encoding="utf-8")

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [{"type": "jsonl", "data_files": str(data_path), "content_field": "text"}],
                "save": {"format": "jsonl"},
            },
            "trainer": {
                "trainer_type": "distillation",
                "max_length": 16,
                "save_directory": str(tmp_path / "trainer-out"),
            },
        }
    )

    stats = elm.pre_tokenize(compression=None, show_progress=False)

    assert stats.num_examples == 1
    assert stats.output_paths
    expected_folder = tmp_path / "trainer-out" / "pretokenized" / "distillation-SimpleTokenizer-MXL16-PLNA-CLNA-train"
    assert str(expected_folder) in stats.output_paths[0]
    row = _read_first_jsonl(stats)
    assert set(row) == {"input_ids", "attention_mask"}


def test_pre_tokenize_prefers_mixture_save_output_path(tmp_path):
    data_path = tmp_path / "train.jsonl"
    mixture_output = tmp_path / "mixture-out"
    data_path.write_text('{"text": "save here"}\n', encoding="utf-8")

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [{"type": "jsonl", "data_files": str(data_path), "content_field": "text"}],
                "save": {"format": "jsonl", "output_path": str(mixture_output)},
            },
            "trainer": {
                "trainer_type": "sft",
                "max_length": 16,
                "save_directory": str(tmp_path / "trainer-out"),
            },
        }
    )

    stats = elm.pre_tokenize(compression=None, show_progress=False)

    assert stats.output_paths
    expected_folder = mixture_output / "sft-SimpleTokenizer-MXL16-PLNA-CLNA-train"
    assert str(expected_folder) in stats.output_paths[0]


def test_pre_tokenize_accepts_base_path_as_first_positional_arg(tmp_path):
    data_path = tmp_path / "train.jsonl"
    base_output = tmp_path / "elarge-data"
    data_path.write_text('{"text": "positional path"}\n', encoding="utf-8")

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [{"type": "jsonl", "data_files": str(data_path), "content_field": "text"}],
                "save": {"format": "jsonl"},
            },
            "trainer": {"trainer_type": "sft", "max_length": 16},
        }
    )

    stats = elm.pre_tokenize(str(base_output), compression=None, show_progress=False)

    assert stats.output_paths
    expected_folder = base_output / "sft-SimpleTokenizer-MXL16-PLNA-CLNA-train"
    assert str(expected_folder) in stats.output_paths[0]


def test_pre_tokenize_log_process_updates_tqdm(tmp_path, monkeypatch):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
    progress_events = []

    class FakeProgressBar:
        def update(self, count):
            progress_events.append(("update", count))

        def close(self):
            progress_events.append(("close", None))

    monkeypatch.setattr(pipeline_mod, "_make_pre_tokenize_progress_bar", FakeProgressBar)

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [{"type": "jsonl", "data_files": str(data_path), "content_field": "text"}],
                "save": {"format": "jsonl"},
            },
            "trainer": {"trainer_type": "sft", "max_length": 16},
        }
    )

    stats = elm.pre_tokenize(tmp_path / "logged-out", compression=None, show_progress=False, log_process=1)

    assert stats.num_examples == 2
    assert progress_events == [("update", 1), ("update", 1), ("close", None)]


def test_pre_tokenize_num_proc_parallel_path_writes_rows(tmp_path):
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(
        '{"text": "one"}\n{"text": "two"}\n{"text": "three"}\n{"text": "four"}\n',
        encoding="utf-8",
    )

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [{"type": "jsonl", "data_files": str(data_path), "content_field": "text"}],
                "save": {"format": "jsonl"},
            },
            "trainer": {"trainer_type": "sft", "max_length": 16},
        }
    )

    stats = elm.pre_tokenize(tmp_path / "parallel-out", compression=None, show_progress=False, num_proc=2)

    assert stats.num_examples == 4
    row = _read_first_jsonl(stats)
    assert set(row) == {"input_ids", "attention_mask"}


def test_parquet_inform_projects_declared_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data_path = tmp_path / "train.parquet"
    table = pa.table(
        {
            "messages": [["hello"], ["world"]],
            "tools": [["tool"], ["tool"]],
            "unused_blob": ["x" * 1024, "y" * 1024],
        }
    )
    pq.write_table(table, data_path)

    source = _create_source_from_inform(
        {
            "type": "parquet",
            "data_files": str(data_path),
            "content_field": "messages",
            "additional_fields": ["tools"],
        },
        {},
    )

    assert source._columns == ["messages", "tools"]


def test_pre_tokenize_generated_path_joins_named_mixture_datasets(tmp_path):
    data_a = tmp_path / "a.jsonl"
    data_b = tmp_path / "b.jsonl"
    data_a.write_text('{"text": "a"}\n', encoding="utf-8")
    data_b.write_text('{"text": "b"}\n', encoding="utf-8")

    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "mixture": {
                "informs": [
                    {"name": "agentic-behave-1", "type": "jsonl", "data_files": str(data_a), "content_field": "text"},
                    {
                        "name": "reasoning/and calling",
                        "type": "jsonl",
                        "data_files": str(data_b),
                        "content_field": "text",
                    },
                ],
                "save": {"format": "jsonl"},
            },
            "trainer": {
                "trainer_type": "dpo",
                "max_length": 48,
                "max_prompt_length": 12,
                "max_completion_length": 8,
            },
        }
    )

    trainer_cfg, arguments = elm._build_training_arguments_for_type()
    folder = elm._build_pre_tokenize_folder_name(trainer_cfg, arguments, elm.build_tokenizer())

    assert folder == "dpo-SimpleTokenizer-MXL48-PL12-CL8-agentic-behave-1_reasoning_and_calling"


@pytest.mark.parametrize(
    ("trainer_type", "expected_cls"),
    [
        ("sft", "SFTPreprocessTransform"),
        ("gkd", "SFTPreprocessTransform"),
        ("distillation", "SFTPreprocessTransform"),
        ("dpo", "DPOPreprocessTransform"),
        ("orpo", "ORPOPreprocessTransform"),
        ("cpo", "DPOPreprocessTransform"),
        ("kto", "KTOPreprocessTransform"),
        ("bco", "BCOPreprocessTransform"),
        ("reward", "RewardPreprocessTransform"),
        ("embedding", "EmbeddingPreprocessTransform"),
        ("grpo", "GRPOPreprocessTransform"),
        ("gfpo", "GRPOPreprocessTransform"),
        ("gspo", "GRPOPreprocessTransform"),
        ("sdpo", "GRPOPreprocessTransform"),
        ("ppo", "PPOPreprocessTransform"),
        ("xpo", "GRPOPreprocessTransform"),
        ("nash_md", "GRPOPreprocessTransform"),
        ("rlvr", "GRPOPreprocessTransform"),
        ("agentic_moshpit", "GRPOPreprocessTransform"),
        ("on_policy_distillation", "GRPOPreprocessTransform"),
        ("seq_kd", "GRPOPreprocessTransform"),
        ("sparse_distillation", "GRPOPreprocessTransform"),
    ],
)
def test_pre_tokenize_resolves_trainer_transforms(trainer_type, expected_cls):
    elm = _elm_with_tokenizer(
        {
            "model": {"name_or_path": "dummy-model"},
            "trainer": {"max_length": 4096, "max_prompt_length": 16},
        }
    )

    transform = elm._build_pre_tokenize_transform(trainer_type)

    assert transform.__class__.__name__ == expected_cls
