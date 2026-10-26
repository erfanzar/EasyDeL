"""Cache strategies and wrapped sources honor pipeline cache contracts without array computation."""

from __future__ import annotations

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from easydel.data.core.config import TokenizerConfig
from easydel.data.execution.cache import TreeCacheManager
from easydel.data.sources.base import (
    JsonShardedSource,
    MaterializedShardedSource,
    ParquetShardedSource,
    RetryingShardedSource,
)
from easydel.data.transforms.source import LimitedShardedSource
from easydel.data.transforms.tokenize import TokenizedShardedSource

from ._common import FakeTokenizer

pytestmark = pytest.mark.cpu_ok
STRATEGIES = ("content", "path", "combined")


def _write(path, texts=("abc", "de", "fg")):
    path.write_text("\n".join(json.dumps({"text": text}) for text in texts))


def _tokenize(source, tokenizer, cache, **kwargs):
    return TokenizedShardedSource(
        source,
        tokenizer,
        TokenizerConfig(name_or_path="fake-tok", max_length=64),
        cache=cache,
        batch_size=2,
        **kwargs,
    )


def _collect(source):
    return [row for name in source.shard_names for row in source.open_shard(name)]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_hash_strategy_controls_cross_path_reuse(tmp_path, strategy):
    first_path, other_path = tmp_path / "first.jsonl", tmp_path / "other.jsonl"
    _write(first_path)
    _write(other_path)
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    first = _collect(_tokenize(JsonShardedSource(str(first_path)), FakeTokenizer(), cache))
    tokenizer = FakeTokenizer()
    assert _collect(_tokenize(JsonShardedSource(str(other_path)), tokenizer, cache)) == first
    assert tokenizer.calls == (0 if strategy == "content" else 2)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_hash_strategy_controls_version_only_invalidation(tmp_path, strategy):
    path = tmp_path / "data.jsonl"
    _write(path)
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    first = _collect(_tokenize(JsonShardedSource(str(path)), FakeTokenizer(), cache))
    old = path.stat()
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 10_000_000_000))
    tokenizer = FakeTokenizer()
    assert _collect(_tokenize(JsonShardedSource(str(path)), tokenizer, cache)) == first
    assert tokenizer.calls == (0 if strategy == "content" else 2)
    # Contents and the file version change: every strategy must invalidate.
    _write(path, ("changed",))
    tokenizer = FakeTokenizer()
    changed = _collect(_tokenize(JsonShardedSource(str(path)), tokenizer, cache))
    assert changed[0]["input_ids"] == [ord(c) for c in "changed"]
    assert tokenizer.calls == 1


@pytest.mark.parametrize("strategy", ["content", "combined"])
def test_content_sensitive_strategies_reject_stale_file_version_metadata(tmp_path, monkeypatch, strategy):
    from eformer.paths import LocalPath

    path = tmp_path / "data.jsonl"
    _write(path, ("first",))
    frozen_version = LocalPath(path).stat()
    monkeypatch.setattr(LocalPath, "stat", lambda self: dict(frozen_version))
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    _collect(_tokenize(JsonShardedSource(str(path)), FakeTokenizer(), cache))
    _write(path, ("other",))  # Same byte count and an intentionally stale version provider.
    tokenizer = FakeTokenizer()
    changed = _collect(_tokenize(JsonShardedSource(str(path)), tokenizer, cache))
    assert changed[0]["input_ids"] == [ord(c) for c in "other"]
    assert tokenizer.calls == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_keep_tokenizer_and_output_settings_in_identity(tmp_path, strategy):
    path = tmp_path / "data.jsonl"
    _write(path)
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    _collect(_tokenize(JsonShardedSource(str(path)), FakeTokenizer(), cache))
    tokenizer = FakeTokenizer()
    changed = _collect(
        _tokenize(
            JsonShardedSource(str(path)),
            tokenizer,
            cache,
            tokenizer_call_kwargs={"max_length": 1},
            preserve_source_columns=True,
            remove_columns=["attention_mask"],
        )
    )
    assert tokenizer.calls == 2
    assert changed == [{"text": text, "input_ids": [ord(text[0])]} for text in ("abc", "de", "fg")]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_mutable_format_callbacks_invalidate_even_with_path_keys(tmp_path, strategy):
    path = tmp_path / "data.jsonl"
    _write(path)
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    suffix = ["x"]
    tokenizer = FakeTokenizer()
    wrapped = _tokenize(
        JsonShardedSource(str(path)),
        tokenizer,
        cache,
        format_callback=lambda row: {"text": row["text"] + suffix[0]},
    )
    first = _collect(wrapped)
    assert _collect(wrapped) == first
    assert tokenizer.calls == 2
    suffix[0] = "y"
    assert _collect(wrapped)[0]["input_ids"] == [ord(c) for c in "abcy"]
    assert tokenizer.calls == 4


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("wrapper", ["retry", "limited", "materialized", "nested"])
def test_wrapped_sources_cache_resume_and_invalidate(tmp_path, strategy, wrapper):
    path = tmp_path / "data.jsonl"
    _write(path)
    cache_dir = tmp_path / "cache"

    def build(tokenizer):
        source = JsonShardedSource(str(path))
        if wrapper in ("retry", "nested"):
            source = RetryingShardedSource(source, max_retries=1, retry_delay=0)
        if wrapper in ("materialized", "nested"):
            source = MaterializedShardedSource(source)
        if wrapper in ("limited", "nested"):
            source = LimitedShardedSource(source, max_rows=2)
        # A fresh manager proves reuse is persisted, not a source-local memory cache.
        return _tokenize(source, tokenizer, TreeCacheManager(cache_dir, hash_fn=strategy))

    first = _collect(build(FakeTokenizer()))
    tokenizer = FakeTokenizer()
    second = build(tokenizer)
    assert _collect(second) == first
    assert list(second.open_shard_at_row(str(path), 1)) == first[1:]
    assert tokenizer.calls == 0
    _write(path, ("new first", "new second", "new third"))
    tokenizer = FakeTokenizer()
    changed = _collect(build(tokenizer))
    assert changed[0]["input_ids"] == [ord(c) for c in "new first"]
    assert tokenizer.calls > 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_materialized_snapshot_cannot_alias_current_backing_file(tmp_path, strategy):
    path = tmp_path / "data.jsonl"
    _write(path, ("original",))
    snapshot = MaterializedShardedSource(JsonShardedSource(str(path)))
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    _write(path, ("replacement",))
    live = _collect(_tokenize(JsonShardedSource(str(path)), FakeTokenizer(), cache))
    tokenizer = FakeTokenizer()
    old = _collect(_tokenize(snapshot, tokenizer, cache))
    assert old[0]["input_ids"] == [ord(c) for c in "original"]
    assert old != live
    assert tokenizer.calls == 1
    assert _collect(_tokenize(snapshot, tokenizer, cache)) == old
    assert tokenizer.calls == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_parquet_guard_and_projection_participate_in_reader_identity(tmp_path, strategy):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"text": ["first", "second"], "id": [1, 2]}), path)
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)
    unlimited = ParquetShardedSource(str(path), max_row_group_bytes=0)
    first = _collect(_tokenize(unlimited, FakeTokenizer(), cache))
    tokenizer = FakeTokenizer()
    guarded = ParquetShardedSource(str(path), max_row_group_bytes=100_000_000)
    assert _collect(_tokenize(guarded, tokenizer, cache)) == first
    assert tokenizer.calls == 1
    # The same settings must now hit, rather than disabling caching for Parquet.
    assert _collect(_tokenize(guarded, tokenizer, cache)) == first
    assert tokenizer.calls == 1
    projected = ParquetShardedSource(str(path), columns=["text"], max_row_group_bytes=100_000_000)
    assert _collect(_tokenize(projected, tokenizer, cache)) == first
    assert tokenizer.calls == 2
    assert _collect(_tokenize(ParquetShardedSource(str(path), max_row_group_bytes=1), tokenizer, cache)) == first
