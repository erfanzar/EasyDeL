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

"""A dropped gcsfs connection must reopen the shard, not kill the stream.

aiohttp's client errors subclass ``ClientError(Exception)``, not ``OSError``, so before they were
added to ``_PARQUET_TRANSIENT_IO_ERRORS`` a gcsfs read dying with ``ServerDisconnectedError``
skipped ``_open_shard_resumable``'s reopen handler. The test drives the real reopen loop and only
replaces the lowest layer (``_iter_shard_rows``).
"""

from __future__ import annotations

import pytest
from easydel.data.sources.base import ParquetShardedSource

pytestmark = pytest.mark.cpu_ok

aiohttp = pytest.importorskip("aiohttp", reason="aiohttp is a gcsfs dependency; absent here")


class _FlakyShardSource(ParquetShardedSource):
    """Raises a given exception once mid-stream, then streams cleanly on reopen."""

    def __init__(self, exc, rows=4):  # no parent init: the reopen loop needs none of its state
        self._exc = exc
        self._rows = rows
        self._max_retries = None
        self._retry_delay = 0.0
        self.calls = 0

    def _iter_shard_rows(self, shard_name, row):
        self.calls += 1
        if self.calls == 1:
            yield 0, {"i": 0}
            raise self._exc
        for i in range(int(row), self._rows):
            yield i, {"i": i}


def _drain(exc):
    src = _FlakyShardSource(exc)
    return [r["i"] for r in src._open_shard_resumable("shard-0.parquet", 0)], src


def test_aiohttp_client_error_is_treated_as_transient():
    rows, src = _drain(aiohttp.ServerDisconnectedError())
    assert src.calls == 2, f"the shard was not reopened after the aiohttp failure (calls={src.calls})"
    assert rows == [0, 1, 2, 3], f"rows lost or duplicated across the reopen: {rows}"


def test_oserror_still_retries_unchanged():
    rows, src = _drain(OSError("connection reset"))
    assert src.calls == 2
    assert rows == [0, 1, 2, 3]


def test_non_transient_error_still_propagates():
    with pytest.raises(ValueError):
        _drain(ValueError("corrupt footer"))
