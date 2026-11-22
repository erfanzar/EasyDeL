from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

from easydel.data.core.protocols import ShardedDataSource
from easydel.data.execution.pipeline import pretokenize


class MemoryShardedSource(ShardedDataSource[dict]):
    def __init__(self, rows: list[dict]):
        self._rows = rows

    @property
    def shard_names(self) -> Sequence[str]:
        return ["memory"]

    def num_shards(self) -> int:
        return 1

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        if shard_name != "memory":
            raise KeyError(shard_name)
        yield from self._rows


class MetadataHeavyTransform:
    def __call__(self, example: dict) -> dict:
        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "label": 1,
            "scores": [0.5, 1.0],
            "source": "raw-row-id",
            "metadata": {"dataset": "debug"},
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "messages": [{"role": "user", "content": "hello"}],
        }


def test_pretokenize_arrays_only_drops_metadata_before_save(tmp_path):
    stats = pretokenize(
        source=MemoryShardedSource([{"text": "hello"}]),
        transform=MetadataHeavyTransform(),
        output_path=str(tmp_path),
        output_format="jsonl",
        compression=None,
        show_progress=False,
        arrays_only=True,
    )

    with open(stats.output_paths[0], encoding="utf-8") as handle:
        row = json.loads(handle.readline())

    assert row == {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "label": 1,
        "scores": [0.5, 1.0],
    }
