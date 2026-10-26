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

"""TokenizeStageConfig batch_size / num_workers / cache_tokenized / remove_columns take effect."""

from __future__ import annotations

import json
import pickle

import pytest
from easydel.data.core.config import DatasetConfig, PipelineConfig, TokenizerConfig, TokenizeStageConfig
from easydel.data.core.protocols import PipelineContext
from easydel.data.execution.cache import TreeCacheManager
from easydel.data.sources.base import JsonShardedSource
from easydel.data.transforms.tokenize import (
    TokenizedShardedSource,
    TokenizeStage,
    batched_tokenize_iterator,
    tokenize_dataset_config,
)

from ._common import FakeTokenizer, ListSource

pytestmark = pytest.mark.cpu_ok

ROWS = [{"text": t, "id": i} for i, t in enumerate(["a", "bcd", "ef", "ghijk", "l", "mn", "opq"])]


def _cfg() -> TokenizerConfig:
    return TokenizerConfig(name_or_path="fake-tok", max_length=64)


def _collect(source, shard="s0"):
    return list(source.open_shard(shard))


def test_batched_tokenization_matches_per_row_with_fewer_calls():
    per_row_tok, batched_tok = FakeTokenizer(), FakeTokenizer()
    base = ListSource({"s0": ROWS})
    per_row = TokenizedShardedSource(base, per_row_tok, _cfg(), additional_fields=["id"])
    batched = TokenizedShardedSource(base, batched_tok, _cfg(), additional_fields=["id"], batch_size=3)

    expected = _collect(per_row)
    got = _collect(batched)

    assert got == expected
    assert per_row_tok.calls == len(ROWS)
    assert batched_tok.calls == 3  # ceil(7 / 3) batched calls


def test_num_workers_preserves_order():
    base = ListSource({"s0": ROWS})
    expected = _collect(TokenizedShardedSource(base, FakeTokenizer(), _cfg(), additional_fields=["id"]))
    threaded = TokenizedShardedSource(
        base, FakeTokenizer(), _cfg(), additional_fields=["id"], batch_size=2, num_workers=3
    )
    assert _collect(threaded) == expected


def test_remove_columns_drops_outputs_and_preserved_fields():
    base = ListSource({"s0": ROWS})
    source = TokenizedShardedSource(
        base, FakeTokenizer(), _cfg(), additional_fields=["id"], remove_columns=["attention_mask", "id"]
    )
    for row in _collect(source):
        assert set(row) == {"input_ids"}


def test_cache_reuses_chunks_and_invalidates_on_file_change(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    cache = TreeCacheManager(cache_dir=tmp_path / "cache")

    def build(tok):
        return TokenizedShardedSource(
            JsonShardedSource(str(path)),
            tok,
            _cfg(),
            additional_fields=["id"],
            batch_size=3,
            cache=cache,
        )

    first_tok = FakeTokenizer()
    first = _collect(build(first_tok), str(path))
    assert first_tok.calls > 0

    second_tok = FakeTokenizer()
    second = _collect(build(second_tok), str(path))
    assert second == first
    assert second_tok.calls == 0  # served entirely from the cache

    path.write_text("\n".join(json.dumps({"text": "zzzzzzzz", "id": 9}) for _ in range(2)) + "\n")
    third_tok = FakeTokenizer()
    third = _collect(build(third_tok), str(path))
    assert third_tok.calls > 0
    assert [r["id"] for r in third] == [9, 9]
    assert len(third[0]["input_ids"]) == 8


def test_no_cache_means_every_iteration_tokenizes(tmp_path):
    base = ListSource({"s0": ROWS})
    tok = FakeTokenizer()
    source = TokenizedShardedSource(base, tok, _cfg(), batch_size=4)
    _collect(source)
    calls_after_first = tok.calls
    _collect(source)
    assert tok.calls == 2 * calls_after_first


def test_tokenize_stage_wires_config_fields(monkeypatch):
    tok = FakeTokenizer()
    ds = DatasetConfig(data_files="unused.jsonl", name="ds", tokenizer="fake-tok", additional_fields=["id"])
    config = PipelineConfig(
        datasets=[ds],
        tokenize=TokenizeStageConfig(
            max_length=64, batch_size=4, num_workers=2, cache_tokenized=False, remove_columns=["attention_mask"]
        ),
    )
    stage = TokenizeStage(config.tokenize)
    monkeypatch.setattr(stage._tokenizer_manager, "get_tokenizer", lambda _cfg: tok)

    out = stage.process({"ds": ListSource({"s0": ROWS})}, PipelineContext(config=config))
    rows = _collect(out["ds"])

    assert [r["id"] for r in rows] == [r["id"] for r in ROWS]
    assert all("attention_mask" not in r for r in rows)
    assert [r["input_ids"] for r in rows] == [[ord(c) for c in r["text"]] for r in ROWS]


def test_tokenize_dataset_config_forwards_call_kwargs():
    tok = FakeTokenizer()
    ds = DatasetConfig(data_files="unused.jsonl", tokenizer="fake-tok", tokenizer_kwargs={"marker": 7})
    from easydel.data.transforms import tokenize as tokenize_module

    original = tokenize_module.TokenizerManager.get_tokenizer
    tokenize_module.TokenizerManager.get_tokenizer = lambda self, cfg: tok
    try:
        source = tokenize_dataset_config(ListSource({"s0": ROWS[:2]}), ds)
        _collect(source)
    finally:
        tokenize_module.TokenizerManager.get_tokenizer = original
    assert all(kw == {"marker": 7} for kw in tok.kwargs_seen)


def test_batch_padding_is_honored_and_resume_keeps_original_chunk_boundaries():
    cfg = TokenizerConfig(name_or_path="fake-tok", padding="longest")
    source = TokenizedShardedSource(ListSource({"s0": ROWS}), FakeTokenizer(), cfg, batch_size=3)
    rows = _collect(source)
    assert [len(row["input_ids"]) for row in rows] == [3, 3, 3, 5, 5, 5, 3]
    assert rows[0]["input_ids"] == [ord("a"), 0, 0]
    assert list(source.open_shard_at_row("s0", 1)) == rows[1:]


def test_stage_preserves_source_columns_except_explicit_removals(monkeypatch):
    from easydel.data.transforms.tokenize import TokenizerManager

    monkeypatch.setattr(TokenizerManager, "get_tokenizer", lambda self, cfg: FakeTokenizer())
    ds = DatasetConfig(data_files="unused.jsonl", name="ds", tokenizer="fake-tok")
    config = PipelineConfig(datasets=[ds])
    raw = ListSource({"s0": ROWS[:2]})
    rows = _collect(
        TokenizeStage(TokenizeStageConfig(num_workers=1)).process({"ds": raw}, PipelineContext(config=config))["ds"]
    )
    assert [row["text"] for row in rows] == [row["text"] for row in ROWS[:2]]
    assert [row["id"] for row in rows] == [0, 1]
    removed = _collect(
        TokenizeStage(TokenizeStageConfig(num_workers=1, remove_columns=["text"])).process(
            {"ds": raw}, PipelineContext(config=config)
        )["ds"]
    )
    assert all("text" not in row and "input_ids" in row and "id" in row for row in removed)


def test_tokenizer_call_kwargs_override_defaults_without_duplicate_keywords():
    source = TokenizedShardedSource(
        ListSource({"s0": ROWS}),
        FakeTokenizer(),
        _cfg(),
        batch_size=3,
        tokenizer_call_kwargs={"max_length": 1, "return_attention_mask": False},
    )
    assert _collect(source) == [{"input_ids": [ord(row["text"][0])]} for row in ROWS]


def test_batched_public_iterator_preserves_metadata_and_kwargs():
    rows = list(
        batched_tokenize_iterator(
            ListSource({"s0": ROWS}),
            FakeTokenizer(),
            _cfg(),
            batch_size=2,
            additional_fields=["id"],
            tokenizer_call_kwargs={"max_length": 1},
        )
    )
    assert rows == [{"input_ids": [ord(row["text"][0])], "attention_mask": [1], "id": row["id"]} for row in ROWS]


def test_stage_cache_switches_and_persistent_reuse(tmp_path, monkeypatch):
    from easydel.data.transforms.tokenize import TokenizerManager

    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in ROWS))
    ds = DatasetConfig(data_files=str(path), name="ds", tokenizer="fake-tok")
    config = PipelineConfig(datasets=[ds])
    config.cache.cache_dir = str(tmp_path / "cache")
    tokenizers = []

    def tokenizer(self, cfg):
        tok = FakeTokenizer()
        tokenizers.append(tok)
        return tok

    monkeypatch.setattr(TokenizerManager, "get_tokenizer", tokenizer)

    def collect(cache_tokenized=True):
        stage = TokenizeStage(TokenizeStageConfig(batch_size=3, num_workers=1, cache_tokenized=cache_tokenized))
        wrapped = stage.process({"ds": JsonShardedSource(str(path))}, PipelineContext(config=config))["ds"]
        return _collect(wrapped, str(path))

    first = collect()
    assert tokenizers[-1].calls == 3
    assert collect() == first
    assert tokenizers[-1].calls == 0
    assert collect(cache_tokenized=False) == first
    assert tokenizers[-1].calls == 3
    config.cache.enabled = False
    assert collect() == first
    assert tokenizers[-1].calls == 3


def test_stage_disabled_dataset_neither_reads_nor_writes_shared_cache(tmp_path, monkeypatch):
    from easydel.data.transforms.tokenize import TokenizerManager

    ds = DatasetConfig(data_files="unused.jsonl", name="ds", tokenizer="fake-tok")
    config = PipelineConfig(datasets=[ds])
    cache_dir = tmp_path / "cache"
    config.cache.cache_dir = str(cache_dir)
    tokenizers = []

    def tokenizer(self, cfg):
        tok = FakeTokenizer()
        tokenizers.append(tok)
        return tok

    monkeypatch.setattr(TokenizerManager, "get_tokenizer", tokenizer)

    def collect(rows):
        stage = TokenizeStage(TokenizeStageConfig(batch_size=3, num_workers=1))
        wrapped = stage.process({"ds": ListSource({"s0": rows})}, PipelineContext(config=config))["ds"]
        return _collect(wrapped)

    expected = collect(ROWS)
    persisted = {path.relative_to(cache_dir): path.read_bytes() for path in cache_dir.rglob("*") if path.is_file()}
    assert persisted
    ds.cache_enabled = False
    assert collect(ROWS) == expected
    assert tokenizers[-1].calls == 3  # Must not read the already-persisted matching chunks.
    changed_rows = [{**row, "text": row["text"] + "x"} for row in ROWS]
    assert [row["input_ids"] for row in collect(changed_rows)] == [
        [ord(char) for char in row["text"]] for row in changed_rows
    ]
    assert tokenizers[-1].calls == 3
    assert {
        path.relative_to(cache_dir): path.read_bytes() for path in cache_dir.rglob("*") if path.is_file()
    } == persisted


@pytest.mark.parametrize("stage_enabled,pipeline_enabled", [(True, True), (False, True), (True, False)])
def test_stage_dataset_cache_path_override_and_master_switches(tmp_path, monkeypatch, stage_enabled, pipeline_enabled):
    from easydel.data.transforms.tokenize import TokenizerManager

    override_dir = tmp_path / "dataset-cache"
    global_dir = tmp_path / "shared-cache"
    ds = DatasetConfig(data_files="unused.jsonl", name="ds", tokenizer="fake-tok", cache_path=str(override_dir))
    config = PipelineConfig(datasets=[ds])
    config.cache.cache_dir = str(global_dir)
    config.cache.enabled = pipeline_enabled
    tokenizers = []

    def tokenizer(self, cfg):
        tok = FakeTokenizer()
        tokenizers.append(tok)
        return tok

    monkeypatch.setattr(TokenizerManager, "get_tokenizer", tokenizer)

    def collect():
        stage = TokenizeStage(TokenizeStageConfig(batch_size=3, num_workers=1, cache_tokenized=stage_enabled))
        wrapped = stage.process({"ds": ListSource({"s0": ROWS})}, PipelineContext(config=config))["ds"]
        return _collect(wrapped)

    expected = collect()
    assert tokenizers[-1].calls == 3
    assert collect() == expected
    caching = stage_enabled and pipeline_enabled
    assert tokenizers[-1].calls == (0 if caching else 3)
    assert not global_dir.exists()
    if caching:
        assert any(path.is_file() for path in override_dir.rglob("*"))
    else:
        assert not override_dir.exists()


def test_content_cache_supports_in_memory_sources_and_mutable_callbacks(tmp_path):
    tok = FakeTokenizer()
    suffix = {"text": "x"}

    def format_row(row):
        return {**row, "text": row["text"] + suffix["text"]}

    wrapped = TokenizedShardedSource(
        ListSource({"s0": ROWS[:2]}),
        tok,
        _cfg(),
        batch_size=2,
        cache=TreeCacheManager(tmp_path / "cache"),
        format_callback=format_row,
    )
    first = _collect(wrapped)
    first[0]["input_ids"][0] = -999
    cached = _collect(wrapped)
    assert tok.calls == 1
    assert cached[0]["input_ids"] == [ord("a"), ord("x")]
    cached[0]["input_ids"][0] = -999
    assert _collect(wrapped)[0]["input_ids"][0] == ord("a")
    suffix["text"] = "y"
    updated = _collect(wrapped)
    assert tok.calls == 2
    assert updated[0]["input_ids"] == [ord("a"), ord("y")]


def test_threaded_real_fast_tokenizer_matches_serial():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    raw = ListSource({"s0": [{"text": "hello world"}, {"text": "hello"}] * 8})
    serial = TokenizedShardedSource(raw, tok, _cfg(), batch_size=2)
    threaded = TokenizedShardedSource(raw, tok, _cfg(), batch_size=2, num_workers=3)
    assert _collect(threaded) == _collect(serial)


def test_threaded_source_remains_serializable_for_distributed_data_workers():
    source = TokenizedShardedSource(
        ListSource({"s0": ROWS}),
        FakeTokenizer(),
        _cfg(),
        batch_size=2,
        num_workers=2,
    )
    restored = pickle.loads(pickle.dumps(source))
    assert _collect(restored) == _collect(source)


@pytest.mark.parametrize("fail", [False, True])
def test_threaded_tokenization_closes_source_on_cancellation_and_error(fail):
    class ClosingSource(ListSource):
        closed = False

        def open_shard(self, shard_name):
            try:
                yield from super().open_shard(shard_name)
            finally:
                self.closed = True

    class Tokenizer(FakeTokenizer):
        def __call__(self, *args, **kwargs):
            if fail:
                raise ValueError("tokenization failed")
            return super().__call__(*args, **kwargs)

    raw = ClosingSource({"s0": ROWS * 3})
    source = TokenizedShardedSource(raw, Tokenizer(), _cfg(), batch_size=2, num_workers=2)
    stream = source.open_shard("s0")
    if fail:
        with pytest.raises(ValueError, match="tokenization failed"):
            next(stream)
    else:
        assert "input_ids" in next(stream)
        stream.close()
    assert raw.closed
