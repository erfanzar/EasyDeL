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

"""Batched-vs-per-row parity for streaming SFT tokenization.

Trainers apply :class:`SFTPreprocessTransform` to their (bucket) data
sources via ``ShardedDataSource.transform`` →
:class:`TransformedShardedSource`, which routes transforms exposing
``map_batch`` through small row chunks so the tokenizer encodes many rows
per call instead of one (the per-row path GIL-starves the training loop's
dispatch thread from the dataloader prefetch worker).

These tests pin, against a real fast tokenizer:

1. The streamed (batched) rows are identical to applying the transform
   per row — same rows, fields, values, and order — for padded,
   padding-free (packing collator), and ``pad_to_multiple_of`` modes,
   with pretokenized rows and their extra columns passed through intact.
2. The tokenizer is actually invoked with more than one row per call on
   the streaming path (counted via a delegating wrapper).
"""

from __future__ import annotations

import os
import typing as tp
import warnings
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from easydel.data.core.protocols import ShardedDataSource
from easydel.trainers.prompt_transforms import SFTPreprocessTransform

TOKENIZER_ID = os.environ.get("EASYDEL_TEST_TOKENIZER", "Qwen/Qwen3.6-27B")
MAX_LENGTH = 64


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"tokenizer {TOKENIZER_ID!r} unavailable: {exc}")


class _ListSource(ShardedDataSource[dict]):
    """Minimal in-memory sharded source over explicit per-shard row lists."""

    def __init__(self, shards: dict[str, list[dict]]):
        self._shards = shards

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shards)

    def num_shards(self) -> int:
        return len(self._shards)

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        return (dict(row) for row in self._shards[shard_name])


class _CountingTokenizer:
    """Delegating tokenizer wrapper that records rows-per-encode-call.

    ``chat_template_batch_sizes`` records, for each ``apply_chat_template``
    call, how many conversations it carried; ``encode_batch_sizes`` does the
    same for direct tokenizer calls. Everything else (attributes, attribute
    writes such as ``truncation_side``) is forwarded to the real tokenizer.
    """

    _OWN: tp.ClassVar[frozenset[str]] = frozenset({"_inner", "chat_template_batch_sizes", "encode_batch_sizes"})

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "chat_template_batch_sizes", [])
        object.__setattr__(self, "encode_batch_sizes", [])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._inner, name, value)

    def apply_chat_template(self, conversation, *args, **kwargs):
        batched = bool(conversation) and isinstance(conversation[0], (list, tuple))
        self.chat_template_batch_sizes.append(len(conversation) if batched else 1)
        return self._inner.apply_chat_template(conversation, *args, **kwargs)

    def __call__(self, text, *args, **kwargs):
        self.encode_batch_sizes.append(len(text) if isinstance(text, (list, tuple)) else 1)
        return self._inner(text, *args, **kwargs)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look something up.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _conversation(user: str, assistant: str, **extra) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        **extra,
    }


def _make_rows() -> list[dict]:
    """Mixed rows: conversations (varied length / tools / stringified), text, pretokenized."""
    return [
        _conversation("Hello", "Hi there! How can I help you today?"),
        _conversation("What is 2+2? Please answer with one word only.", "Four."),
        _conversation("Use the lookup tool for this request.", "Sure.", tools=TOOLS),
        {
            "messages": [
                '{"role": "user", "content": "Stringified message row"}',
                '{"role": "assistant", "content": "Decoded and templated."}',
            ],
            "metadata": {"source": "unit-test"},
        },
        _conversation(
            "Write a haiku about training loops that never converge despite tuning.",
            "Gradients wander\nthe loss plateau stretches on\npatience runs out first",
        ),
        {"text": "Plain text row that takes the direct tokenizer path."},
        {
            "input_ids": [5, 6, 7, 8],
            "attention_mask": [1, 1, 1, 1],
            "custom_meta": "keep-me",
            "embed_n_tok": [2, 2],
        },
        _conversation("Final short row", "ok"),
    ]


def _source_row_order(source: _ListSource) -> list[dict]:
    return [row for shard in source.shard_names for row in source.open_shard(shard)]


def _assert_rows_identical(streamed: list[dict], expected: list[dict]) -> None:
    assert len(streamed) == len(expected)
    for index, (got, want) in enumerate(zip(streamed, expected, strict=True)):
        assert set(got) == set(want), f"row {index}: keys {sorted(got)} != {sorted(want)}"
        for key in want:
            got_value, want_value = got[key], want[key]
            if key in {"input_ids", "attention_mask", "completion_mask", "labels"}:
                assert np.array_equal(np.asarray(got_value), np.asarray(want_value)), (
                    f"row {index}: field {key!r} differs"
                )
            else:
                assert got_value == want_value, f"row {index}: field {key!r} differs"


@pytest.mark.parametrize("padding", [False, "max_length"], ids=["padding_free", "pad_max_length"])
def test_streamed_rows_match_per_row_transform(tokenizer, padding: bool | str):
    """The batched streaming path emits exactly the per-row transform's rows."""
    rows = _make_rows()
    source = _ListSource({"shard_0": rows[:5], "shard_1": rows[5:]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # explicit padding="max_length" warns by design
        transform = SFTPreprocessTransform(tokenizer=tokenizer, max_length=MAX_LENGTH, padding=padding)

    expected = [transform(dict(row)) for row in _source_row_order(source)]
    streamed = [row for shard in source.shard_names for row in source.transform(transform).open_shard(shard)]

    _assert_rows_identical(streamed, expected)

    # Pretokenized rows and their extra (additional_fields-style) columns
    # must pass through the batched path untouched.
    passthrough = [row for row in streamed if "custom_meta" in row]
    assert len(passthrough) == 1
    assert passthrough[0]["custom_meta"] == "keep-me"
    assert passthrough[0]["embed_n_tok"] == [2, 2]
    assert passthrough[0]["input_ids"] == [5, 6, 7, 8]

    # Tool schemas survive tokenization on both paths.
    tool_rows = [row for row in streamed if "tools" in row]
    assert any(row["tools"] == TOOLS for row in tool_rows)


def test_pad_to_multiple_of_matches_per_row(tokenizer):
    """Safety-net multiple-of padding matches per-row output on the batched path."""
    rows = [row for row in _make_rows() if "messages" in row or "text" in row]
    source = _ListSource({"shard_0": rows})
    transform = SFTPreprocessTransform(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        padding=False,
        pad_to_multiple_of=8,
    )
    expected = [transform(dict(row)) for row in _source_row_order(source)]
    streamed = list(source.transform(transform).open_shard("shard_0"))

    _assert_rows_identical(streamed, expected)
    for row in streamed:
        assert len(row["input_ids"]) % 8 == 0


def test_open_shard_at_row_resume_matches_per_row(tokenizer):
    """Resuming mid-shard keeps batched rows aligned with the per-row path."""
    rows = _make_rows()
    source = _ListSource({"shard_0": rows})
    transform = SFTPreprocessTransform(tokenizer=tokenizer, max_length=MAX_LENGTH, padding=False)

    expected = [transform(dict(row)) for row in rows[3:]]
    streamed = list(source.transform(transform).open_shard_at_row("shard_0", 3))

    _assert_rows_identical(streamed, expected)


def test_tokenizer_encodes_multiple_rows_per_call(tokenizer):
    """The public ``source.transform(...)`` path batches tokenizer calls.

    This is the exact call shape the trainers use for (bucket) data sources
    (``BaseTrainer._prepare_bucket_source`` / ``_apply_preprocess_transforms``):
    per-row encoding here is what GIL-starves the step-dispatch thread.
    """
    rows = [_conversation(f"Question number {i}, elaborated a bit for variety.", f"Answer {i}.") for i in range(10)]
    source = _ListSource({"shard_0": rows})

    counting = _CountingTokenizer(tokenizer)
    transform = SFTPreprocessTransform(tokenizer=counting, max_length=MAX_LENGTH, padding=False)
    streamed = list(source.transform(transform).open_shard("shard_0"))

    reference = SFTPreprocessTransform(tokenizer=tokenizer, max_length=MAX_LENGTH, padding=False)
    _assert_rows_identical(streamed, [reference(dict(row)) for row in rows])

    # All 10 rows were encoded, in strictly fewer calls than rows, with at
    # least one genuinely multi-row call.
    total_rows_encoded = sum(counting.chat_template_batch_sizes) + sum(counting.encode_batch_sizes)
    assert total_rows_encoded == len(rows)
    assert len(counting.chat_template_batch_sizes) + len(counting.encode_batch_sizes) < len(rows)
    assert max(counting.chat_template_batch_sizes, default=0) > 1


def test_distinct_per_row_tools_still_batch_encode(tokenizer):
    """Rows with row-specific tool schemas share one batched encode call.

    Agentic mixtures carry a distinct ``tools`` list per row; grouping by
    tool schema alone would degrade to one encode per row. Such singleton
    groups must be rendered per row but encoded together — with output rows
    identical to the per-row path.
    """
    rows = [
        _conversation(
            f"Use tool number {i} for this request.",
            f"Calling tool_{i}.",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": f"Tool number {i}.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        for i in range(6)
    ]
    source = _ListSource({"shard_0": rows})

    counting = _CountingTokenizer(tokenizer)
    transform = SFTPreprocessTransform(tokenizer=counting, max_length=MAX_LENGTH, padding=False)
    streamed = list(source.transform(transform).open_shard("shard_0"))

    reference = SFTPreprocessTransform(tokenizer=tokenizer, max_length=MAX_LENGTH, padding=False)
    _assert_rows_identical(streamed, [reference(dict(row)) for row in rows])

    # Every conversation renders individually (distinct tools), but all six
    # rendered texts are encoded in a single batched tokenizer call.
    assert counting.encode_batch_sizes == [len(rows)]
    assert all(size == 1 for size in counting.chat_template_batch_sizes)


def test_map_batch_size_chunks_preserve_row_order(tokenizer):
    """Explicit chunk sizes split the stream into ordered multi-row encodes."""
    from easydel.data.transforms.source import TransformedShardedSource

    rows = [_conversation(f"Question number {i}, elaborated a bit for variety.", f"Answer {i}.") for i in range(10)]
    source = _ListSource({"shard_0": rows})

    counting = _CountingTokenizer(tokenizer)
    transform = SFTPreprocessTransform(tokenizer=counting, max_length=MAX_LENGTH, padding=False)
    streamed = list(TransformedShardedSource(source, transform, map_batch_size=4).open_shard("shard_0"))

    reference = SFTPreprocessTransform(tokenizer=tokenizer, max_length=MAX_LENGTH, padding=False)
    _assert_rows_identical(streamed, [reference(dict(row)) for row in rows])
    # 10 rows in chunks of 4 -> encode calls of [4, 4, 2] conversations.
    assert counting.chat_template_batch_sizes == [4, 4, 2]
