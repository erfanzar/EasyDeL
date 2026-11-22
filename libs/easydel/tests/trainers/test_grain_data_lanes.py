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
"""Grain random-access data lanes: batches produced, per-dataset stamps, exact resume, ArrayRecord reads.

Pure host-side data tests on local temp parquet / ArrayRecord files (no network, no array computation).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

grain = pytest.importorskip("grain.python")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from easydel.data import ASSISTANT_ONLY_FIELD, DATASET_ID_FIELD  # noqa: E402
from easydel.trainers import data_lanes  # noqa: E402
from easydel.trainers.training_configurations import TrainingArguments  # noqa: E402

pytestmark = pytest.mark.cpu_ok


def _corpus(path, n, *, seed, lo=8, hi=64, offset=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n):
        ids = [int(offset + r)] + [int(v) for v in rng.integers(5, 30_000, size=int(rng.integers(lo, hi)) - 1)]
        rows.append({"input_ids": ids, "labels": ids})
    pq.write_table(pa.Table.from_pylist(rows), str(path), row_group_size=8)
    return str(path)


def _arguments(tmp_path, **overrides):
    kwargs = dict(
        use_wandb=False,
        performance_mode=False,
        save_directory=str(tmp_path / "ckpt"),
        grain_shard_index=0,
        grain_shard_count=1,
        arrayrecord_num_threads=1,
        arrayrecord_prefetch_buffer=2,
        shuffle_seed_train=17,
        total_batch_size=2,
        max_training_steps=40,
    )
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


def _trainer(arguments, batch_size):
    return types.SimpleNamespace(
        arguments=arguments,
        _user_data_collator=True,
        training_batch_size=batch_size,
        _get_processing_class=lambda: object(),
        _get_preprocess_transform=lambda: lambda row: row,
    )


def _ids(window_set):
    return [tuple(r["input_ids"]) for r in window_set]


# ---------------------------------------------------------------------------------------------- packed lane
def _packed(tmp_path, *, cursor=None, step_start_point=None, **extra):
    data = {
        "alpha": _corpus(tmp_path / "a.parquet", 48, seed=1, offset=0),
        "beta": _corpus(tmp_path / "b.parquet", 48, seed=2, offset=1000),
    }
    args = _arguments(
        tmp_path,
        parquetra_train_datasets=data,
        parquetra_assistant_only={"alpha": True},
        parquetra_pack_seq_len=128,
        parquetra_pack_misfit_patience=4,
        step_start_point=step_start_point,
        **extra,
    )
    trainer = _trainer(args, batch_size=2)
    trainer._data_lane_resume_cursor = cursor
    loader, steps = data_lanes.configure_parquetra_lane(trainer)
    return trainer, loader, steps


def test_packed_lane_fills_windows_mixes_datasets_and_stamps_rows(tmp_path):
    trainer, loader, steps = _packed(tmp_path)
    assert steps == 40 and len(loader) == 40
    assert trainer._dataset_id_names == {0: "alpha", 1: "beta"}
    seen_ids = set()
    for window_set in list(iter(loader))[:10]:
        # Re-pack first fit in window-major order: rows must fit 2 windows of 128 tokens.
        total = sum(len(r["input_ids"]) for r in window_set)
        assert 0 < total <= 2 * 128
        for row in window_set:
            ds = row[DATASET_ID_FIELD]
            seen_ids.add(ds)
            assert row[ASSISTANT_ONLY_FIELD] == (1 if ds == 0 else -1)
            assert (row["input_ids"][0] >= 1000) == (ds == 1)
            assert row["attention_mask"] == [1] * len(row["input_ids"])
    assert seen_ids == {0, 1}


def test_packed_lane_resume_from_checkpoint_cursor_is_exact(tmp_path):
    _, loader, _ = _packed(tmp_path)
    it = iter(loader)
    for _ in range(7):
        next(it)
    cursor = loader.cursor_at(7)
    assert cursor is not None and cursor.step == 7
    continued = [_ids(next(it)) for _ in range(5)]

    # Persist next to a checkpoint the way the trainer does, read it back, resume a fresh loader.
    path = data_lanes.write_lane_cursor(str(tmp_path / "ckpt" / "run-7"), 0, cursor)
    restored = data_lanes.read_lane_cursor(str(tmp_path / "ckpt" / "run-7"), 0)
    assert path.endswith("data_lane_state/host-0000.json") and restored == cursor
    _, fresh, _ = _packed(tmp_path, cursor=restored)
    resumed = fresh.iter_from_batch(7)
    assert [_ids(next(resumed)) for _ in range(5)] == continued


def test_packed_lane_without_cursor_seeks_to_the_seed_shifted_stream(tmp_path):
    _, loader, _ = _packed(tmp_path)
    seeked = loader.iter_from_batch(9)
    _, shifted, _ = _packed(tmp_path, step_start_point=9)
    fresh = iter(shifted)
    assert [_ids(next(seeked)) for _ in range(3)] == [_ids(next(fresh)) for _ in range(3)]


def test_packed_lane_rejects_rows_longer_than_the_window(tmp_path):
    data = {"long": _corpus(tmp_path / "long.parquet", 8, seed=3, lo=200, hi=210)}
    args = _arguments(tmp_path, parquetra_train_datasets=data, parquetra_pack_seq_len=128)
    loader, _ = data_lanes.configure_parquetra_lane(_trainer(args, batch_size=2))
    with pytest.raises(ValueError, match="exceeds the 128-token window"):
        next(iter(loader))


def test_trainer_saves_the_cursor_with_the_checkpoint(tmp_path):
    from easydel.trainers.trainer import Trainer

    _, loader, _ = _packed(tmp_path)
    it = iter(loader)
    for _ in range(3):
        next(it)
    trainer = object.__new__(Trainer)
    trainer.dataloader_train = loader
    ckpt = tmp_path / "ckpt" / "run-3"
    trainer._save_data_lane_state(str(ckpt), 3)
    assert data_lanes.read_lane_cursor(str(ckpt), 0) == loader.cursor_at(3)


# ------------------------------------------------------------------------------------------------- map lane
def test_parquetra_map_lane_batches_are_per_source_and_seek_exactly(tmp_path):
    data = {
        "alpha": _corpus(tmp_path / "a.parquet", 32, seed=1, offset=0),
        "beta": _corpus(tmp_path / "b.parquet", 32, seed=2, offset=1000),
    }
    args = _arguments(tmp_path, parquetra_train_datasets=data, parquetra_mixture_weights={"alpha": 1.0, "beta": 1.0})
    loader, steps = data_lanes.configure_parquetra_lane(_trainer(args, batch_size=4))
    assert steps == 40
    batches = list(iter(loader))
    assert len(batches) == 40 and all(len(b) == 4 for b in batches)
    assert all(len({r[DATASET_ID_FIELD] for r in b}) == 1 for b in batches)
    assert {b[0][DATASET_ID_FIELD] for b in batches} == {0, 1}
    assert [_ids(b) for b in loader.iter_from_batch(13)] == [_ids(b) for b in batches[13:]]


def test_lane_config_validation(tmp_path):
    data = {"alpha": _corpus(tmp_path / "a.parquet", 16, seed=1)}
    bad_key = _arguments(tmp_path, parquetra_train_datasets=data, parquetra_assistant_only={"typo": True})
    with pytest.raises(ValueError, match="unknown dataset"):
        data_lanes.configure_parquetra_lane(_trainer(bad_key, batch_size=2))
    no_collator = _trainer(_arguments(tmp_path, parquetra_train_datasets=data), batch_size=2)
    no_collator._user_data_collator = False
    with pytest.raises(ValueError, match="explicit `data_collator`"):
        data_lanes.configure_parquetra_lane(no_collator)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _arguments(tmp_path, parquetra_train_datasets=data, arrayrecord_train_files="x.array_record")


# ------------------------------------------------------------------------------------------- arrayrecord
def _write_arrayrecord(path, rows):
    msgpack = pytest.importorskip("msgpack")
    module = pytest.importorskip("array_record.python.array_record_module")
    writer = module.ArrayRecordWriter(str(path), "group_size:1")
    for row in rows:
        writer.write(msgpack.packb(row))
    writer.close()


def test_arrayrecord_lane_reads_msgpack_rows_with_manifest_and_rate_limit(tmp_path):
    root = tmp_path / "sets"
    (root / "chat").mkdir(parents=True)
    (root / "code").mkdir(parents=True)
    _write_arrayrecord(root / "chat" / "s0.array_record", [{"input_ids": [i, i + 1]} for i in range(12)])
    _write_arrayrecord(root / "code" / "s0.array_record", [{"input_ids": [100 + i]} for i in range(12)])
    nbytes = (root / "code" / "s0.array_record").stat().st_size
    (root / "code" / "_MANIFEST.tsv").write_text(f"code/s0.array_record\t{nbytes}\t12\n")
    args = _arguments(
        tmp_path,
        arrayrecord_train_datasets={
            "chat": str(root / "chat" / "*.array_record"),
            "code": str(root / "code" / "*.array_record"),
        },
        arrayrecord_assistant_only={"code": False},
        arrayrecord_read_rate_limit_mb_per_sec=1000.0,
        max_training_steps=10,
    )
    trainer = _trainer(args, batch_size=3)
    loader, steps = data_lanes.configure_arrayrecord_lane(trainer)
    assert steps == 10 and trainer._dataset_id_names == {0: "chat", 1: "code"}
    batches = list(iter(loader))
    for batch in batches:
        ds = batch[0][DATASET_ID_FIELD]
        for row in batch:
            assert row[DATASET_ID_FIELD] == ds
            assert row[ASSISTANT_ONLY_FIELD] == (0 if ds == 1 else -1)
            assert (row["input_ids"][0] >= 100) == (ds == 1)
    assert [_ids(b) for b in loader.iter_from_batch(4)] == [_ids(b) for b in batches[4:]]
    instructions, per_row = data_lanes.arrayrecord_manifest_instructions(
        str(root / "code" / "*.array_record"), source_name="code"
    )
    assert [i.take for i in instructions] == [12] and per_row == pytest.approx(nbytes / 12)


def test_read_rate_limiter_paces_reads():
    limiter = data_lanes.ReadRateLimiter(bytes_per_second=10_000.0, burst_seconds=0.01)
    assert limiter.acquire(0) == 0.0
    waited = limiter.acquire(1_000)  # bucket starts empty: 1000 B at 10 kB/s is ~0.1 s of debt
    assert 0.05 <= waited <= 0.2
    assert data_lanes.read_rate_bytes_per_second(0) is None
    with pytest.raises(ValueError):
        data_lanes.read_rate_bytes_per_second(-1)


# ----------------------------------------------------------------------------------------- tokenization
def _chat_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "<u>", "<a>", "<e>", *[f"w{i}" for i in range(10)]]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="<e>")
    tok.chat_template = (
        "{% for m in messages %}{% if m['role'] == 'assistant' %}<a> {{ m['content'] }} <e> "
        "{% else %}<u> {{ m['content'] }} {% endif %}{% endfor %}"
    )
    return tok


def _row_tokenize(offset=0, verify_rows=2):
    from easydel.trainers.prompt_transforms import SFTPreprocessTransform

    tok = _chat_tokenizer()

    def factory(hf_tokenizer):
        return lambda text: [t + offset for t in hf_tokenizer(text, add_special_tokens=False)["input_ids"]]

    transform = SFTPreprocessTransform(tok, max_length=32, mask_prompt=False, padding="max_length")
    return data_lanes.ParquetRowTokenize(
        transform, hf_tokenizer=tok, fast_encoder_factory=factory, max_length=32, verify_rows=verify_rows
    )


ROW = {"messages": [{"role": "user", "content": "w1 w2"}, {"role": "assistant", "content": "w3"}]}


def test_row_tokenize_strips_padding_and_certifies_the_fast_path():
    tokenize = _row_tokenize()
    out = tokenize(ROW)
    assert out["input_ids"] == [2, 6, 7, 3, 8, 4]  # <u> w1 w2 <a> w3 <e>, padding removed
    assert out["attention_mask"] == [1] * 6 and out["labels"] == out["input_ids"]
    tokenize(ROW)
    assert tokenize.fast_verified == 2
    assert tokenize(ROW)["input_ids"] == out["input_ids"]  # now served by the fast path alone


def test_row_tokenize_fast_path_divergence_raises():
    with pytest.raises(ValueError, match="does not reproduce"):
        _row_tokenize(offset=1)(ROW)
    with pytest.raises(ValueError, match="module:attr"):
        data_lanes.load_fast_encoder_factory("no_colon")
    assert data_lanes.load_fast_encoder_factory("os.path:join") is __import__("os").path.join


def test_cpu_only_worker_init_edits_only_the_calling_process(monkeypatch):
    import os

    monkeypatch.setenv("TPU_WORKER_ID", "3")
    init = data_lanes.CpuOnlyWorkerInit()
    import pickle

    clone = pickle.loads(pickle.dumps(init))  # must survive the spawn pickle
    assert clone.overrides == {"JAX_PLATFORMS": "cpu", "ENABLE_DISTRIBUTED_INIT": "0"}
    assert os.environ["TPU_WORKER_ID"] == "3"  # building/pickling the initializer touches nothing
