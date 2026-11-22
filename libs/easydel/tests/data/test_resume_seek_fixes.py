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

"""Resume seeks must land on the same delivered example a fresh iteration would (r11 S13-1, S13-3).

Pure-Python data readers: no array computation, so these run on CPU.
"""

from __future__ import annotations

import itertools
import json

import pytest
from easydel.data.sources.base import JsonShardedSource
from easydel.data.transforms.mixture import MixedShardedSource

pytestmark = pytest.mark.cpu_ok


def test_jsonl_seek_counts_delivered_records_not_blank_lines(tmp_path):
    """Blank lines are never delivered, so they must not advance the seek offset (r11 S13-3)."""
    path = tmp_path / "data.jsonl"
    path.write_text('\n{"id": 0}\n\n{"id": 1}\n   \n{"id": 2}\n')
    source = JsonShardedSource(str(path))
    shard = source.shard_names[0]
    full = [record["id"] for record in source.open_shard(shard)]
    assert full == [0, 1, 2]
    for row in range(len(full) + 1):
        assert [record["id"] for record in source.open_shard_at_row(shard, row)] == full[row:], row
    # Public entry point used by trainer resume.
    assert [record["id"] for record in source.iter_shards(start_row=1)] == [1, 2]


def test_restart_mixture_resume_without_metadata_keeps_cycle_position(tmp_path):
    """A metadata-less constituent must wrap to ``row % period`` when resumed past its end (r11 S13-1)."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"id": "a"}, {"id": "b"}, {"id": "c"}]))
    source = JsonShardedSource(str(path), jsonl=False)
    mixed = MixedShardedSource({"json": source}, block_size=1, stop_strategy="restart")
    total = 14
    fresh = [record["id"] for record in itertools.islice(mixed.open_shard("mixed_shard_0"), total)]
    assert fresh == (["a", "b", "c"] * 5)[:total]
    for row in (1, 2, 3, 4, 5, 7, 10):
        resumed = itertools.islice(mixed.open_shard_at_row("mixed_shard_0", row), total - row)
        assert [record["id"] for record in resumed] == fresh[row:], row
