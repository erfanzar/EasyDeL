"""Effective tokenizer and reader settings invalidate persistent caches (no array computation)."""

from __future__ import annotations

import hashlib

import pytest
from easydel.data.core.config import TokenizerConfig
from easydel.data.execution.cache import TreeCacheManager
from easydel.data.sources.base import TextShardedSource
from easydel.data.transforms.tokenize import TokenizedShardedSource, compute_tokenizer_hash
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ._common import FakeTokenizer, ListSource

pytestmark = pytest.mark.cpu_ok
STRATEGIES = ("content", "path", "combined")


class MutatingFastTokenizer(PreTrainedTokenizerFast):
    """Pickleable real tokenizer whose calls change persistent padding policy."""

    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        self.pad_token = "[PAD]" if getattr(self, "toggle_padding", False) and self.pad_token == "[ALT]" else "[ALT]"
        return result


def _fast_tokenizer(tokenizer_type=PreTrainedTokenizerFast):
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "[PAD]": 3, "[ALT]": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer_type(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


def _collect(tokenizer, cache_dir, strategy, config=None, rows=None, num_workers=1):
    source = TokenizedShardedSource(
        ListSource({"s0": rows or [{"text": "hello world"}, {"text": "hello"}]}),
        tokenizer,
        config or TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length"),
        cache=TreeCacheManager(cache_dir, hash_fn=strategy),
        batch_size=2,
        num_workers=num_workers,
    )
    return list(source.open_shard("s0"))


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize(
    "setting", ["vocab", "padding_side", "truncation_side", "special_token", "added_token", "normalizer", "processor"]
)
def test_configured_tokenizers_do_not_share_incompatible_cache_entries(tmp_path, strategy, setting):
    first = _fast_tokenizer()
    changed = _fast_tokenizer()
    rows = [{"text": "hello world hello world"}, {"text": "HELLO"}]
    if setting == "vocab":
        changed.backend_tokenizer.model = models.WordLevel(
            {"[UNK]": 0, "hello": 2, "world": 1, "[PAD]": 3, "[ALT]": 4}, unk_token="[UNK]"
        )
    elif setting == "padding_side":
        changed.padding_side = "left"
    elif setting == "truncation_side":
        changed.truncation_side = "left"
    elif setting == "special_token":
        changed.pad_token = "[ALT]"
    elif setting == "added_token":
        changed.add_tokens(["hello world"])
    elif setting == "normalizer":
        changed.backend_tokenizer.normalizer = normalizers.Lowercase()
    else:
        changed.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="$A [ALT]", special_tokens=[("[ALT]", 4)]
        )
    cache_dir = tmp_path / "cache"
    original = _collect(first, cache_dir, strategy, rows=rows)
    expected = _collect(changed, tmp_path / "independent-cache", strategy, rows=rows)
    assert expected != original  # These settings really do change the delivered public outputs.
    assert _collect(changed, cache_dir, strategy, rows=rows) == expected


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_replacing_local_tokenizer_at_same_path_invalidates_cache(tmp_path, strategy):
    tokenizer_dir = tmp_path / "tokenizer"
    first = _fast_tokenizer()
    first.save_pretrained(tokenizer_dir)
    config = TokenizerConfig(name_or_path=str(tokenizer_dir), add_special_tokens=False)
    cache_dir = tmp_path / "cache"
    rows = [{"text": "hello"}]
    original = _collect(
        AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True), cache_dir, strategy, config, rows
    )
    assert original[0]["input_ids"] == [1]
    replacement = _fast_tokenizer()
    replacement.backend_tokenizer.model = models.WordLevel(
        {"[UNK]": 0, "hello": 2, "world": 1, "[PAD]": 3, "[ALT]": 4}, unk_token="[UNK]"
    )
    replacement.save_pretrained(tokenizer_dir)
    loaded = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    assert loaded("hello", add_special_tokens=False)["input_ids"] == [2]
    assert _collect(loaded, cache_dir, strategy, config, rows)[0]["input_ids"] == [2]
    # The identifier-only public helper keeps its historical SHA-256 contract,
    # independently of the effective tokenizer contents stored at this path.
    expected_hash = hashlib.sha256(str(tokenizer_dir).encode()).hexdigest()[:16]
    assert compute_tokenizer_hash(config.name_or_path) == expected_hash
    assert compute_tokenizer_hash(config.name_or_path + "-other") != expected_hash


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("num_workers", [1, 2])
def test_equivalent_real_tokenizers_reuse_persistent_cache_after_calls(tmp_path, monkeypatch, strategy, num_workers):
    calls = []
    original_call = PreTrainedTokenizerFast.__call__

    def record_call(self, *args, **kwargs):
        calls.append(1)
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "__call__", record_call)
    tokenizer = _fast_tokenizer()
    first = _collect(tokenizer, tmp_path / "cache", strategy, num_workers=num_workers)
    assert len(calls) == 1
    # Transformers changes its backend padding/truncation state during __call__.
    # Reusing that instance or loading an equivalent fresh one must still hit.
    assert _collect(tokenizer, tmp_path / "cache", strategy, num_workers=num_workers) == first
    assert _collect(_fast_tokenizer(), tmp_path / "cache", strategy, num_workers=num_workers) == first
    assert len(calls) == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_unserializable_custom_tokenizer_never_uses_an_unsafe_cache_key(tmp_path, strategy):
    class LocalTokenizer(FakeTokenizer):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            return {**result, "input_ids": [[value + 1 for value in row] for row in result["input_ids"]]}

    first = _collect(FakeTokenizer(), tmp_path / "cache", strategy)
    tokenizer = LocalTokenizer()
    changed = _collect(tokenizer, tmp_path / "cache", strategy)
    assert changed != first
    assert _collect(tokenizer, tmp_path / "cache", strategy) == changed
    assert tokenizer.calls == 2


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("setting", ["tokenizer", "config", "kwargs", "schema"])
def test_existing_source_refreshes_mutable_settings_without_poisoning_other_sources(
    tmp_path, strategy, num_workers, setting
):
    tokenizer = _fast_tokenizer()
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")
    kwargs = {"add_special_tokens": True}
    fields = ["id"]
    rows = [{"text": "hello world hello world", "id": 1}, {"text": "hello", "id": 2}]
    cache_dir = tmp_path / "cache"

    def build(tok, cfg, call_kwargs, additional_fields, directory=cache_dir):
        return TokenizedShardedSource(
            ListSource({"s0": rows}),
            tok,
            cfg,
            tokenizer_call_kwargs=call_kwargs,
            additional_fields=additional_fields,
            batch_size=2,
            num_workers=num_workers,
            cache=TreeCacheManager(directory, hash_fn=strategy),
        )

    source = build(tokenizer, config, kwargs, fields)
    first = list(source.open_shard("s0"))
    if setting == "tokenizer":
        tokenizer.pad_token = "[ALT]"
    elif setting == "config":
        config.max_length = 2
    elif setting == "kwargs":
        kwargs["max_length"] = 1
    else:
        fields.append("text")
    expected = list(build(tokenizer, config, kwargs, fields, tmp_path / "reference").open_shard("s0"))
    assert expected != first
    assert list(source.open_shard("s0")) == expected
    assert list(source.open_shard_at_row("s0", 1)) == expected[1:]
    # A stale construction-time hash must not cause the changed iteration to write
    # new results under the original settings' key.
    original = build(
        _fast_tokenizer(),
        TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length"),
        {"add_special_tokens": True},
        ["id"],
    )
    assert list(original.open_shard("s0")) == first


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_mutation_before_first_iteration_does_not_poison_original_configuration(tmp_path, strategy):
    tokenizer = _fast_tokenizer()
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")
    source = TokenizedShardedSource(
        ListSource({"s0": [{"text": "hello"}]}),
        tokenizer,
        config,
        cache=TreeCacheManager(tmp_path / "cache", hash_fn=strategy),
    )
    tokenizer.pad_token = "[ALT]"
    changed = list(source.open_shard("s0"))
    assert changed[0]["input_ids"] == [1, 4, 4]
    original = TokenizedShardedSource(
        ListSource({"s0": [{"text": "hello"}]}),
        _fast_tokenizer(),
        config,
        cache=TreeCacheManager(tmp_path / "cache", hash_fn=strategy),
    )
    unchanged = list(original.open_shard("s0"))
    assert unchanged[0]["input_ids"] == [1, 3, 3]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_mutation_between_chunks_cannot_publish_under_old_tokenizer_identity(tmp_path, strategy):
    tokenizer = _fast_tokenizer()
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")
    rows = [{"text": "hello"}, {"text": "hello world"}]
    source = TokenizedShardedSource(
        ListSource({"s0": rows}),
        tokenizer,
        config,
        cache=TreeCacheManager(tmp_path / "cache", hash_fn=strategy),
    )
    stream = source.open_shard("s0")
    assert next(stream)["input_ids"] == [1, 3, 3]
    tokenizer.pad_token = "[ALT]"
    remaining = list(stream)
    assert remaining[0]["input_ids"] == [1, 2, 4]
    original = TokenizedShardedSource(
        ListSource({"s0": rows}),
        _fast_tokenizer(),
        config,
        cache=TreeCacheManager(tmp_path / "cache", hash_fn=strategy),
    )
    assert list(original.open_shard("s0"))[1]["input_ids"] == [1, 2, 3]


@pytest.mark.parametrize("num_workers", [1, 2])
def test_interleaved_iterations_keep_their_own_call_policies_and_cache_keys(tmp_path, num_workers):
    tokenizer = _fast_tokenizer()
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")
    kwargs = {"return_attention_mask": True}
    fields = ["id"]
    rows = [{"text": "hello", "id": i} for i in range(10)]
    source = TokenizedShardedSource(
        ListSource({"s0": rows}),
        tokenizer,
        config,
        tokenizer_call_kwargs=kwargs,
        additional_fields=fields,
        num_workers=num_workers,
        cache=TreeCacheManager(tmp_path / "cache"),
    )
    first_stream = source.open_shard("s0")
    first_row = next(first_stream)
    config.max_length = 2
    kwargs["return_attention_mask"] = False
    fields.append("text")
    changed = list(source.open_shard("s0"))
    assert changed == [{"input_ids": [1, 3], "id": i, "text": "hello"} for i in range(10)]
    assert [first_row, *first_stream] == [
        {"input_ids": [1, 3, 3], "attention_mask": [1, 0, 0], "id": i} for i in range(10)
    ]
    # A fresh reader with each original policy must see only that policy's rows.
    config.max_length = 3
    kwargs["return_attention_mask"] = True
    fields.pop()
    assert list(source.open_shard("s0")) == [
        {"input_ids": [1, 3, 3], "attention_mask": [1, 0, 0], "id": i} for i in range(10)
    ]


def test_tokenizer_mutation_inside_a_call_cannot_poison_the_cache(tmp_path, monkeypatch):
    tokenizer = _fast_tokenizer()
    original_call = PreTrainedTokenizerFast.__call__

    def change_padding_then_tokenize(self, *args, **kwargs):
        if self is tokenizer:
            self.pad_token = "[ALT]"
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerFast, "__call__", change_padding_then_tokenize)
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")

    def collect(tok):
        source = TokenizedShardedSource(
            ListSource({"s0": [{"text": "hello"}]}),
            tok,
            config,
            cache=TreeCacheManager(tmp_path / "cache"),
        )
        return list(source.open_shard("s0"))

    assert collect(tokenizer) == [{"input_ids": [1, 4, 4], "attention_mask": [1, 0, 0]}]
    assert collect(_fast_tokenizer()) == [{"input_ids": [1, 3, 3], "attention_mask": [1, 0, 0]}]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("toggle_padding", [False, True])
def test_threaded_worker_mutations_do_not_poison_resumed_cache_outputs(tmp_path, strategy, batch_size, toggle_padding):
    rows = [{"text": "hello"} for _ in range(24)]
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)

    def source(cache_layer):
        tokenizer = _fast_tokenizer(MutatingFastTokenizer)
        tokenizer.toggle_padding = toggle_padding
        return TokenizedShardedSource(
            ListSource({"s0": rows}),
            tokenizer,
            config,
            num_workers=2,
            batch_size=batch_size,
            cache=cache_layer,
        )

    warm = list(source(cache).open_shard("s0"))
    assert warm[0]["input_ids"] == [1, 3, 3]
    assert any(row["input_ids"] == [1, 4, 4] for row in warm)
    expected = list(source(None).open_shard_at_row("s0", 23))
    assert expected == [{"input_ids": [1, 3, 3], "attention_mask": [1, 0, 0]}]
    # A fresh worker resumes with [PAD], not the warmed worker's mutated [ALT].
    assert list(source(cache).open_shard_at_row("s0", 23)) == expected
    assert list(source(cache).open_shard_at_row("s0", 1))[:1] == list(source(None).open_shard_at_row("s0", 1))[:1]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_mutated_workers_do_not_read_entries_for_their_initial_identity(tmp_path, monkeypatch, strategy):
    rows = [{"text": "hello"} for _ in range(24)]
    config = TokenizerConfig(name_or_path="same-identifier", max_length=3, padding="max_length")
    cache = TreeCacheManager(tmp_path / "cache", hash_fn=strategy)

    def source(cache_layer):
        return TokenizedShardedSource(
            ListSource({"s0": rows}),
            _fast_tokenizer(MutatingFastTokenizer),
            config,
            num_workers=2,
            cache=cache_layer,
        )

    # Seed a valid [PAD] entry while calls do not mutate. Changing __call__ here
    # leaves the serialized instance/class-source identity unchanged, as custom
    # tokenizers with external side effects can do between iterations.
    with monkeypatch.context() as patch:
        patch.setattr(MutatingFastTokenizer, "__call__", PreTrainedTokenizerFast.__call__)
        seeded = list(source(cache).open_shard_at_row("s0", 23))
        assert seeded == [{"input_ids": [1, 3, 3], "attention_mask": [1, 0, 0]}]
    expected_last = list(source(None).open_shard("s0"))[-1]
    assert expected_last["input_ids"] == [1, 4, 4]
    assert list(source(cache).open_shard("s0"))[-1] == expected_last


def test_noncopyable_call_arguments_preserve_tokenization_but_bypass_cache(tmp_path):
    import threading

    tokenizer = FakeTokenizer()
    source = TokenizedShardedSource(
        ListSource({"s0": [{"text": "hello"}]}),
        tokenizer,
        TokenizerConfig(name_or_path="same-identifier"),
        tokenizer_call_kwargs={"lock": threading.Lock()},
        cache=TreeCacheManager(tmp_path / "cache"),
    )
    assert list(source.open_shard("s0")) == list(source.open_shard("s0"))
    assert tokenizer.calls == 2


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_storage_decoding_options_change_reader_cache_identity(tmp_path, strategy):
    path = tmp_path / "text.txt"
    path.write_bytes(b"\x80\n")
    cache_dir = tmp_path / "cache"

    def collect(encoding):
        source = TextShardedSource(str(path), storage_options={"encoding": encoding})
        wrapped = TokenizedShardedSource(
            source,
            FakeTokenizer(),
            TokenizerConfig(name_or_path="same-tokenizer"),
            cache=TreeCacheManager(cache_dir, hash_fn=strategy),
        )
        return list(wrapped.open_shard(str(path)))

    assert collect("cp1252")[0]["input_ids"] == [0x20AC]
    assert collect("latin1")[0]["input_ids"] == [0x80]
