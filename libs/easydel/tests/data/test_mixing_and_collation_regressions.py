"""Public data regressions: finite mixing, tool metadata and NumPy-only collation."""

from __future__ import annotations

import copy
import json
import signal
from contextlib import contextmanager

import ml_dtypes
import numpy as np
import pytest
from easydel.data.core.config import (
    DatasetConfig,
    LoadStageConfig,
    MixStageConfig,
    PipelineConfig,
    TokenizeStageConfig,
    WeightSchedulePoint,
)
from easydel.data.core.protocols import PipelineContext
from easydel.data.execution.loader import LoadStage
from easydel.data.transforms.chat_template import ChatTemplateTransform, ConvertToChatML, convert_to_chatml
from easydel.data.transforms.collators import collate_bucket_static, collate_embeds_pack, collate_packed_embeds
from easydel.data.transforms.field_ops import CombineFields
from easydel.data.transforms.filter_ops import FilterNonEmpty
from easydel.data.transforms.mixture import MixedShardedSource, MixStage, WeightScheduler
from easydel.data.transforms.tokenize import TokenizerManager, TokenizeStage

from ._common import FakeTokenizer, ListSource

pytestmark = pytest.mark.cpu_ok


@contextmanager
def _must_terminate():
    def timed_out(*_args):
        raise TimeoutError("finite mixer failed to make progress")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("weights", [None, {"a": 1.0, "b": 1e-12, "c": 1e-30}])
@pytest.mark.parametrize("block_size", [1, 2])
def test_all_exhausted_drains_positive_sources_even_when_block_rounding_excludes_them(weights, block_size):
    sources = {name: ListSource({"s": [{"v": f"{name}0"}, {"v": f"{name}1"}]}) for name in "abc"}
    mixer = MixedShardedSource(sources, weights=weights, block_size=block_size, seed=0, stop_strategy="all_exhausted")
    with _must_terminate():
        rows = list(mixer.open_shard("mixed_shard_0"))
        assert sorted(row["v"] for row in rows) == ["a0", "a1", "b0", "b1", "c0", "c1"]
        assert list(mixer.open_shard_at_row("mixed_shard_0", 3)) == rows[3:]
        state = mixer.resume_state(3)
        assert list(mixer.open_shard_from_state(state)) == rows[3:]


@pytest.mark.parametrize("strategy", ["all_exhausted", "restart"])
def test_zero_weight_sources_do_not_prevent_termination(strategy):
    mixer = MixedShardedSource(
        {"a": ListSource({"s": []}), "b": ListSource({"s": [{"v": "excluded"}]})},
        weights={"a": 1.0, "b": 0.0},
        block_size=1,
        stop_strategy=strategy,
    )
    with _must_terminate():
        assert list(mixer.open_shard("mixed_shard_0")) == []


def test_all_exhausted_does_not_spin_when_schedule_has_no_live_positive_sources():
    mixer = MixedShardedSource(
        {"a": ListSource({"s": [{"v": "a"}]}), "b": ListSource({"s": [{"v": "b"}]})},
        block_size=1,
        stop_strategy="all_exhausted",
        weight_scheduler=WeightScheduler(
            [
                WeightSchedulePoint(step=0, weights={"a": 1.0, "b": 0.0}),
                WeightSchedulePoint(step=10, weights={"a": 0.0, "b": 1.0}),
            ]
        ),
    )
    with _must_terminate():
        assert list(mixer.open_shard("mixed_shard_0")) == [{"v": "a", "__source__": "a"}]


def test_automatic_chat_conversion_preserves_tool_metadata_and_normalizes_mixed_formats():
    messages = [
        {"role": "system", "content": "use tools"},
        {"from": "human", "value": "weather?"},
        {
            "from": "gpt",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}],
        },
        {"from": "tool", "value": "sunny", "tool_call_id": "c1", "name": "weather"},
    ]
    original = copy.deepcopy(messages)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)

    expected = [
        {"role": "system", "content": "use tools"},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": original[2]["tool_calls"]},
        {"role": "tool", "content": "sunny", "tool_call_id": "c1", "name": "weather"},
    ]
    assert json.loads(ChatTemplateTransform(Tokenizer())({"messages": messages})["text"]) == expected
    unchanged = ChatTemplateTransform(Tokenizer(), convert_from_value_format=False)({"messages": messages})
    assert json.loads(unchanged["text"]) == original
    assert convert_to_chatml(messages) == expected
    assert ConvertToChatML()({"messages": messages})["messages"] == expected
    assert messages == original
    assert (
        ConvertToChatML(role_mapping={"human": "customer"})({"messages": messages})["messages"][1]["role"] == "customer"
    )
    assert ConvertToChatML(use_default_mapping=False)({"messages": messages})["messages"][1]["role"] == "human"


def _embed_row(ids, vectors=()):
    embeddings = np.asarray(vectors, dtype=ml_dtypes.bfloat16)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": ids,
        "embed_n_tok": [len(vectors)] if len(vectors) else None,
        "image_embeds": [embeddings.tobytes()] if len(vectors) else [],
        "image_grid_thw": [[1, 1, len(vectors)]] if len(vectors) else [],
    }


def _positions(input_ids, **kwargs):
    return np.tile(np.arange(input_ids.shape[1]), (3, 1, 1)), None


def _collate(kind, rows, embed_dim=None):
    common = {"image_token_id": 99, "embed_dim": embed_dim}
    if kind == "dynamic":
        return collate_embeds_pack(rows, 0, 4, **common)
    if kind == "static":
        return collate_bucket_static(rows, 0, 8, 4, **common)
    return collate_packed_embeds([rows], 0, 8, 4, 2, position_id_fn=_positions, **common)


@pytest.mark.parametrize("kind", ["dynamic", "static", "packed"])
@pytest.mark.parametrize("image_first", [False, True])
def test_mixed_text_image_batch_infers_embed_width_before_decoding_text(kind, image_first):
    rows = [_embed_row([1, 2]), _embed_row([99], [[10, 11]])]
    if image_first:
        rows.reverse()
    batch = _collate(kind, rows)
    assert batch["image_embeds"].shape == (4, 2)
    np.testing.assert_array_equal(batch["image_embeds"], [[10, 11], [0, 0], [0, 0], [0, 0]])
    expected_position = [[0, 0] if image_first else [0, 2]] if kind == "packed" else [[0, 0] if image_first else [1, 0]]
    assert batch["image_embed_positions"][:1].tolist() == expected_position
    assert batch["n_real_embeds"] == 1


@pytest.mark.parametrize("kind", ["dynamic", "static", "packed"])
def test_opposite_per_document_placeholder_mismatches_cannot_cancel(kind):
    rows = [_embed_row([99, 99], [[10, 11]]), _embed_row([1], [[20, 21]])]
    with pytest.raises(ValueError, match="placeholder/embed mismatch"):
        _collate(kind, rows, embed_dim=2)


@pytest.mark.parametrize("kind", ["dynamic", "static", "packed"])
def test_batch_width_resolution_checks_metadata_and_blobs(kind):
    image = _embed_row([99], [[10, 11]])
    text = {**_embed_row([1]), "embed_dim": 3}
    with pytest.raises(ValueError, match="mixed embed_dim"):
        _collate(kind, [image, text])
    with pytest.raises(ValueError, match="embed_dim is required"):
        _collate(kind, [_embed_row([1])])


@pytest.mark.parametrize("drop_sources", [False, True])
def test_combine_fields_can_augment_a_source_field_in_place(drop_sources):
    row = {"text": "hello", "suffix": "world", "id": 7}
    result = CombineFields(["text", "suffix"], "text", drop_sources=drop_sources)(row)
    assert result == (
        {"text": "hello world", "id": 7} if drop_sources else {"text": "hello world", "suffix": "world", "id": 7}
    )
    assert row == {"text": "hello", "suffix": "world", "id": 7}


@pytest.mark.parametrize(
    "value, keep",
    [
        (None, False),
        ("", False),
        ([], False),
        ({}, False),
        ((), False),
        (np.asarray([], dtype=np.int32), False),
        (np.asarray([1, 2], dtype=np.int32), True),
        (np.asarray(1), True),
        (0, True),
        (False, True),
        ("text", True),
        ([1], True),
    ],
)
def test_filter_nonempty_handles_numpy_arrays_without_elementwise_truth(value, keep):
    row = {"input_ids": value}
    result = FilterNonEmpty(["input_ids"])(row)
    assert (result is row) == keep


def test_tokenize_mix_load_preserves_heterogeneous_optional_metadata(monkeypatch):
    monkeypatch.setattr(TokenizerManager, "get_tokenizer", lambda self, cfg: FakeTokenizer())
    config = PipelineConfig(
        datasets=[
            DatasetConfig(name="a", data_files="a.jsonl", tokenizer="fake"),
            DatasetConfig(name="b", data_files="b.jsonl", tokenizer="fake"),
        ]
    )
    context = PipelineContext(config=config)
    sources = {
        "a": ListSource({"s": [{"text": "hi", "lang": "en"}]}),
        "b": ListSource({"s": [{"text": "hi", "score": 7}]}),
    }
    tokenized = TokenizeStage(TokenizeStageConfig(num_workers=1)).process(sources, context)
    mixed = MixStage(MixStageConfig(block_size=2, seed=0, stop_strategy="all_exhausted")).process(tokenized, context)
    loader = LoadStage(LoadStageConfig(batch_size=2, prefetch_enabled=False)).process(mixed, context)["mixed"]
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"].tolist() == [[104, 105], [104, 105]]
    rows = {name: index for index, name in enumerate(batch["__source__"].tolist())}
    assert batch["lang"].tolist()[rows["a"]] == "en"
    assert batch["lang"].tolist()[rows["b"]] is None
    assert batch["score"].tolist()[rows["a"]] is None
    assert batch["score"].tolist()[rows["b"]] == 7
    assert batch["text"].tolist() == ["hi", "hi"]
