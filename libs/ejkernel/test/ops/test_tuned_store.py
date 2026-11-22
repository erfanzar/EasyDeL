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

"""The shipped table of pre-tuned kernel configurations.

This replaces per-kernel hand-maintained Python literals, so the properties that
matter are the ones that make it safe to consult from a kernel's hot path and
safe to ship half-populated:

* a missing, corrupt or unpopulated table must never break a call — the caller
  just keeps its own default, which is what makes adding the table additive;
* a shape nobody measured must still resolve to the nearest measured neighbour,
  or a table would only ever answer questions it was already asked;
* the winning **platform** travels with the config, because which backend is
  faster is shape-dependent (XLA ragged_dot wins at MoE shapes, Pallas wins for
  paged attention) and was previously encoded in scattered call-site branches.
"""

import json
import sqlite3

import pytest
from ejkernel.ops.tuned import (
    TunedEntry,
    TunedStore,
    bucket,
    dtype_signature,
    merge,
    open_for_write,
    set_tuned_store,
    shape_signature,
    tuned_choice,
    upsert,
)

DT = dtype_signature(lhs="bfloat16", rhs="int4")


def _entry(m, platform="pallas", block_m=64, ms=0.2, runner_ms=0.4, kernel="grouped_matmul"):
    return TunedEntry(
        kernel=kernel,
        device="TPU v5p",
        dtypes=DT,
        shape_key=shape_signature(m=m, k=4096, n=2048, e=256),
        platform=platform,
        config={"block_m": block_m},
        ms=ms,
        runner_up={"platform": "xla", "config": {}, "ms": runner_ms},
    )


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "tuned.db"
    conn = open_for_write(path)
    upsert(
        conn,
        [_entry(12288, block_m=64, ms=0.230, runner_ms=0.408), _entry(192, block_m=8, ms=0.104, runner_ms=0.170)],
    )
    conn.close()
    return path


def test_exact_lookup_returns_platform_and_config(db):
    entry = TunedStore(db).lookup("grouped_matmul", "TPU v5p", DT, shape_signature(m=12288, k=4096, n=2048, e=256))
    assert entry is not None
    assert entry.platform == "pallas"
    assert entry.config == {"block_m": 64}
    assert entry.speedup_over_runner_up() == pytest.approx(0.408 / 0.230, rel=1e-6)


def test_unmeasured_shape_resolves_to_the_nearest_measured_one(db):
    """A table keyed on exact sizes only answers questions it was already asked."""
    store = TunedStore(db)
    # 12288 -> bucket 8192 and 192 -> bucket 128 are what is stored; 3000 -> 2048
    # and 100 -> 64 are buckets nobody measured, so both must fall to a neighbour.
    near_big = store.lookup("grouped_matmul", "TPU v5p", DT, shape_signature(m=3000, k=4096, n=2048, e=256))
    near_small = store.lookup("grouped_matmul", "TPU v5p", DT, shape_signature(m=100, k=4096, n=2048, e=256))
    assert near_big.config == {"block_m": 64}
    assert near_small.config == {"block_m": 8}


def test_nearest_can_be_refused(db):
    assert (
        TunedStore(db).lookup(
            "grouped_matmul",
            "TPU v5p",
            DT,
            shape_signature(m=3000, k=4096, n=2048, e=256),
            allow_nearest=False,
        )
        is None
    )


def test_a_different_device_or_dtype_never_borrows_an_entry(db):
    """Tuning is per hardware and per dtype; bleeding across is silent garbage."""
    store = TunedStore(db)
    shape = shape_signature(m=12288, k=4096, n=2048, e=256)
    assert store.lookup("grouped_matmul", "TPU v6e", DT, shape) is None
    assert store.lookup("grouped_matmul", "TPU v5p", dtype_signature(lhs="float32", rhs="float32"), shape) is None


def test_missing_database_degrades_quietly():
    """The table is an optimization; its absence must not be an error."""
    store = TunedStore("/nonexistent/definitely/not/here.db")
    assert store.available() is False
    assert store.lookup("k", "d", "", "s") is None
    assert store.entries() == []
    assert store.kernels() == []


def test_corrupt_database_degrades_quietly(tmp_path):
    """A truncated or garbage file must not take a kernel call down with it."""
    bad = tmp_path / "corrupt.db"
    bad.write_bytes(b"this is not a database" * 100)
    store = TunedStore(bad)
    assert store.lookup("grouped_matmul", "TPU v5p", DT, shape_signature(m=1, k=1, n=1, e=1)) is None
    assert store.entries() == []


def test_shape_buckets_are_powers_of_two_and_round_down():
    """Rounding DOWN is the safe direction: a config tuned for fewer rows
    under-fills its tiles rather than over-running them."""
    assert [bucket(v) for v in (0, 1, 7, 48, 192, 3000, 12288)] == [1, 1, 4, 32, 128, 2048, 8192]
    for v in (5, 63, 1025, 99999):
        b = bucket(v)
        assert b <= v and b & (b - 1) == 0


def test_signatures_are_order_independent():
    assert dtype_signature(q="bfloat16", kv="int8") == dtype_signature(kv="int8", q="bfloat16")
    assert shape_signature(m=8, k=16) == shape_signature(k=16, m=8)


def test_upsert_replaces_rather_than_duplicates(tmp_path):
    path = tmp_path / "t.db"
    conn = open_for_write(path)
    upsert(conn, [_entry(12288, block_m=64, ms=0.5)])
    upsert(conn, [_entry(12288, block_m=32, ms=0.3)])
    conn.close()
    entries = TunedStore(path).entries()
    assert len(entries) == 1
    assert entries[0].config == {"block_m": 32}


def test_merge_keeps_the_faster_measurement(tmp_path):
    """Sweeps land from different machines and dates, so conflict resolution
    has to be defined rather than last-writer-wins by accident."""
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    ca = open_for_write(a)
    upsert(ca, [_entry(12288, block_m=64, ms=0.230)])
    ca.close()
    cb = open_for_write(b)
    upsert(cb, [_entry(12288, block_m=32, ms=0.900)])
    cb.close()

    merge(a, b)
    kept = TunedStore(a).entries()[0]
    assert kept.config == {"block_m": 64}, "a slower measurement must not displace a faster one"

    cb2 = open_for_write(b)
    upsert(cb2, [_entry(12288, block_m=16, ms=0.100)])
    cb2.close()
    merge(a, b)
    assert TunedStore(a).entries()[0].config == {"block_m": 16}


def test_untimed_row_never_displaces_a_timed_one(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    ca = open_for_write(a)
    upsert(ca, [_entry(12288, block_m=64, ms=0.230)])
    ca.close()
    cb = open_for_write(b)
    upsert(
        cb,
        [
            TunedEntry(
                "grouped_matmul", "TPU v5p", DT, shape_signature(m=12288, k=4096, n=2048, e=256), "xla", {}, ms=None
            )
        ],
    )
    cb.close()
    merge(a, b)
    assert TunedStore(a).entries()[0].config == {"block_m": 64}


def test_tuned_choice_uses_the_active_store(db):
    """The kernel-facing entry point, with dtypes/shape given as plain mappings."""
    try:
        set_tuned_store(db)
        choice = tuned_choice(
            "grouped_matmul",
            dtypes={"lhs": "bfloat16", "rhs": "int4"},
            shape={"m": 12288, "k": 4096, "n": 2048, "e": 256},
            device="TPU v5p",
        )
        assert choice is not None and choice.platform == "pallas"
        assert tuned_choice("unswept_kernel", dtypes={"x": "bfloat16"}, shape={"m": 1}, device="TPU v5p") is None, (
            "an unswept kernel must report nothing so the caller keeps its default"
        )
    finally:
        set_tuned_store(None)


def test_stored_rows_are_readable_as_plain_sql(db):
    """It is a database on purpose: inspectable without importing ejkernel."""
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT kernel, platform, config FROM tuned ORDER BY shape_key").fetchall()
    assert {r[0] for r in rows} == {"grouped_matmul"}
    assert all(json.loads(r[2]) for r in rows)


def test_lookup_cost_does_not_grow_with_table_size(tmp_path):
    """Indexed lookup, not a linear scan -- this sits on a per-call trace path."""
    path = tmp_path / "big.db"
    conn = open_for_write(path)
    upsert(
        conn,
        [
            TunedEntry(
                f"kernel_{i % 40}",
                f"TPU v{5 + i % 3}p",
                DT,
                shape_signature(m=2 ** (i % 16), k=4096, n=2048, e=256),
                "pallas",
                {"block_m": 64},
                ms=0.2,
            )
            for i in range(5000)
        ],
    )
    conn.close()
    store = TunedStore(path)
    plan = None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as c:
        plan = c.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM tuned WHERE kernel=? AND device=? AND dtypes=? AND shape_key=?",
            ("kernel_1", "TPU v5p", DT, shape_signature(m=1024, k=4096, n=2048, e=256)),
        ).fetchall()
    assert any("USING" in " ".join(str(x) for x in row) for row in plan), f"expected an index scan, got {plan}"
    assert store.lookup("kernel_1", "TPU v5p", DT, shape_signature(m=1024, k=4096, n=2048, e=256)) is not None
