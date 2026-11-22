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

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

from easydel.data.core.protocols import ShardedDataSource
from easydel.data.transforms.pack import PackedShardedSource
from easydel.trainers.prompt_transforms import SFTPreprocessTransform


class _MockTokenizer:
    """Minimal tokenizer for testing SFTPreprocessTransform padding behavior."""

    def __init__(self):
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.pad_token = "<pad>"
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.pad_token_id = 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        ids = [ord(c) % 100 for c in text]
        max_length = kwargs.get("max_length")
        padding = kwargs.get("padding")
        if kwargs.get("add_special_tokens", True):
            ids.append(self.eos_token_id)
        if padding == "max_length" and max_length:
            pad_len = max(0, max_length - len(ids))
            ids = ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * (len(ids) - pad_len) + [0] * pad_len
        else:
            attention_mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": attention_mask}


class _ListShardedSource(ShardedDataSource[dict]):
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    @property
    def shard_names(self) -> list[str]:
        return ["shard_0"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        yield from self._rows


def test_sft_packing_mode_produces_unpadded_rows():
    """packing_mode=True should disable padding so rows can be packed."""
    tokenizer = _MockTokenizer()
    transform = SFTPreprocessTransform(
        tokenizer=tokenizer,
        max_length=32,
        packing_mode=True,
    )

    short_text = "hi"
    result = transform({"text": short_text})

    assert len(result["input_ids"]) < 32
    assert all(t != tokenizer.pad_token_id for t in result["input_ids"])


def test_sft_default_constructor_pads_to_max_length():
    """Default constructor should still pad for backward compatibility."""
    tokenizer = _MockTokenizer()
    transform = SFTPreprocessTransform(
        tokenizer=tokenizer,
        max_length=32,
    )

    short_text = "hi"
    result = transform({"text": short_text})

    assert len(result["input_ids"]) == 32
    assert result["input_ids"].count(tokenizer.pad_token_id) > 0


def test_sft_explicit_max_length_padding_warns():
    """Explicitly passing padding='max_length' should warn about packing inefficiency."""
    tokenizer = _MockTokenizer()
    with pytest.warns(UserWarning, match="packing_mode=True"):
        SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=32,
            padding="max_length",
        )


def test_sft_default_constructor_does_not_warn():
    """The default constructor should remain warning-free for backward compatibility."""
    tokenizer = _MockTokenizer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=32,
        )


def test_sft_packing_mode_overrides_padding():
    """packing_mode=True should override an explicit padding request and warn."""
    tokenizer = _MockTokenizer()
    with pytest.warns(UserWarning, match="packing_mode=True overrides padding"):
        transform = SFTPreprocessTransform(
            tokenizer=tokenizer,
            max_length=32,
            padding="max_length",
            packing_mode=True,
        )

    result = transform({"text": "hello"})
    assert len(result["input_ids"]) < 32


def test_packed_sharded_source_warns_on_pre_padded_input(caplog):
    """PackedShardedSource should warn when upstream rows look pre-padded."""
    seq_length = 32
    rows = [{"input_ids": [1] * seq_length} for _ in range(10)]
    source = _ListShardedSource(rows)
    packed = PackedShardedSource(
        source=source,
        seq_length=seq_length,
        eos_token_id=2,
        strategy="greedy",
        shuffle=False,
        warn_on_padded_input=True,
    )

    with caplog.at_level("WARNING", logger="easydel.data.transforms.pack"):
        for _ in packed.open_shard(packed.shard_names[0]):
            pass

    assert any("pre-padded" in record.message for record in caplog.records)


def test_packed_sharded_source_no_warning_when_rows_vary(caplog):
    """No warning when upstream rows have varying lengths."""
    seq_length = 32
    rows = [{"input_ids": [1] * seq_length}, {"input_ids": [1] * 5}]
    source = _ListShardedSource(rows)
    packed = PackedShardedSource(
        source=source,
        seq_length=seq_length,
        eos_token_id=2,
        strategy="greedy",
        shuffle=False,
        warn_on_padded_input=True,
    )

    with caplog.at_level("WARNING", logger="easydel.data.transforms.pack"):
        for _ in packed.open_shard(packed.shard_names[0]):
            pass

    assert not any("pre-padded" in record.message for record in caplog.records)
