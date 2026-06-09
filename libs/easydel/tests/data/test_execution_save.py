from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easydel.data.core.protocols import ShardedDataSource
from easydel.data.execution.save import save_dataset


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


def test_parquet_save_json_encodes_tools_with_mixed_nested_scalars(tmp_path):
    rows = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "tools": [{"function": {"parameters": {"properties": {"limit": {"default": 20}}}}}],
        },
        {
            "input_ids": [3, 4],
            "attention_mask": [1, 1],
            "tools": [{"function": {"parameters": {"properties": {"limit": {"default": "20"}}}}}],
        },
    ]

    stats = save_dataset(
        MemoryShardedSource(rows),
        str(tmp_path),
        format="parquet",
        max_shard_size="10MB",
        compression=None,
    )

    assert stats.num_examples == 2
    table = pq.read_table(stats.output_paths[0])
    assert table.schema.field("input_ids").type.value_type.bit_width == 32
    assert pa.types.is_string(table.schema.field("tools").type)

    tools = table.column("tools").to_pylist()
    assert json.loads(tools[0])[0]["function"]["parameters"]["properties"]["limit"]["default"] == 20
    assert json.loads(tools[1])[0]["function"]["parameters"]["properties"]["limit"]["default"] == "20"


def test_parquet_save_reports_scalar_column_name_on_arrow_conversion_error(tmp_path):
    rows = [
        {"input_ids": [1], "source_id": 20},
        {"input_ids": [2], "source_id": "20"},
    ]

    with pytest.raises(ValueError, match="source_id"):
        save_dataset(
            MemoryShardedSource(rows),
            str(tmp_path),
            format="parquet",
            max_shard_size="10MB",
            compression=None,
        )
