"""Public data helpers remain usable through the loader's documented callback seam."""

from __future__ import annotations

from functools import partial
from uuid import uuid4

import fsspec
import pytest
from easydel.data.core.config import DatasetConfig, LoadStageConfig, PipelineConfig
from easydel.data.core.protocols import PipelineContext
from easydel.data.execution.loader import LoadStage
from easydel.data.transforms import collate_bucket_static, collate_embeds_pack, collate_packed_embeds
from easydel.data.transforms.packer import EmbedsWindowPacker
from easydel.data.utils import get_cached_filesystem

from ._common import ListSource

pytestmark = pytest.mark.cpu_ok


def _rows():
    return [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [1, 2],
            "embed_n_tok": None,
            "image_embeds": [],
            "image_grid_thw": [],
        },
        {
            "input_ids": [3],
            "attention_mask": [1],
            "labels": [3],
            "embed_n_tok": None,
            "image_embeds": [],
            "image_grid_thw": [],
        },
    ]


@pytest.mark.parametrize("static", [False, True])
def test_embed_collators_work_through_load_stage(static):
    if static:
        collate = partial(collate_bucket_static, pad_id=0, seq_len=4, max_embed_rows=0, image_token_id=99, embed_dim=2)
    else:
        collate = partial(collate_embeds_pack, pad_id=0, max_total=0, image_token_id=99, embed_dim=2)
    config = PipelineConfig(datasets=[DatasetConfig(name="ds", data_files="unused.jsonl")])
    stage = LoadStage(LoadStageConfig(batch_size=2, prefetch_enabled=False, collate_fn=collate))
    loader = stage.process({"ds": ListSource({"s0": _rows()})}, PipelineContext(config=config))["ds"]
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert batch["input_ids"].tolist() == ([[1, 2, 0, 0], [3, 0, 0, 0]] if static else [[1, 2], [3, 0]])
    assert batch["labels"][1, 1] == -100
    assert batch["image_embeds"].shape == (0, 2)


def test_embeds_packer_to_materializer_preserves_rows_and_document_boundaries():
    packer = EmbedsWindowPacker(n_windows=2, seq_len=4, max_embed_rows=0)
    packer.push(_rows())
    batch = collate_packed_embeds(packer.emit(), 0, 4, 0, 2, image_token_id=99, embed_dim=2)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 0], [0, 0, 0, 0]]
    assert batch["segment_ids"].tolist() == [[0, 0, 1, -1], [-1, -1, -1, -1]]
    assert batch["position_ids"][:, 0, :].tolist() == [[0, 1, 0, 0]] * 3
    assert len(packer) == 0


def test_cached_filesystem_serves_repeated_reads_from_local_cache(tmp_path):
    # memory:// is in-process storage: this test never contacts a cloud service.
    path = f"/data-2-{uuid4().hex}/data.jsonl"
    remote = fsspec.filesystem("memory")
    remote.pipe(path, b"original rows")
    cached = get_cached_filesystem("memory", tmp_path, storage_options={"memory": {}})
    with cached.open(path, "rb") as handle:
        assert handle.read() == b"original rows"
    remote.pipe(path, b"replacement rows")
    with cached.open(path, "rb") as handle:
        assert handle.read() == b"original rows"


def test_cached_filesystem_keeps_local_reads_plain(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("local text")
    fs = get_cached_filesystem("file", tmp_path / "not-created")
    with fs.open(str(path), "r") as handle:
        assert handle.read() == "local text"
    assert not (tmp_path / "not-created").exists()
