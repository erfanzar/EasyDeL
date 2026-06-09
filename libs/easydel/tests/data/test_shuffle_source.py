from __future__ import annotations

from collections.abc import Iterator, Sequence

from easydel.data import ShuffledShardedSource
from easydel.data.core.protocols import ShardedDataSource
from easydel.data.transforms.mixture import MixedShardedSource


class ListSource(ShardedDataSource[dict]):
    """Minimal in-memory source: ``nshards`` shards, round-robin row assignment."""

    def __init__(self, rows: list[dict], nshards: int = 1):
        self._rows = rows
        self._nshards = nshards

    @property
    def shard_names(self) -> Sequence[str]:
        return [f"s{i}" for i in range(self._nshards)]

    def num_shards(self) -> int:
        return self._nshards

    def open_shard(self, shard_name: str) -> Iterator[dict]:
        i = int(shard_name[1:])
        yield from self._rows[i :: self._nshards]

    def __len__(self) -> int:
        return len(self._rows)


def _drain(source: ShardedDataSource) -> list:
    out: list = []
    for name in source.shard_names:
        out.extend(source.open_shard(name))
    return out


def test_shuffle_preserves_all_rows_and_reorders():
    src = ListSource([{"id": i} for i in range(50)])
    shuffled = src.shuffle(buffer_size=16, seed=42)
    order = [r["id"] for r in _drain(shuffled)]
    assert sorted(order) == list(range(50))  # nothing lost / duplicated
    assert order != list(range(50))  # actually shuffled


def test_shuffle_is_deterministic_for_fixed_seed():
    src = ListSource([{"id": i} for i in range(50)])
    a = [r["id"] for r in ShuffledShardedSource(src, buffer_size=16, seed=7).open_shard("shuffled_shard_0")]
    b = [r["id"] for r in ShuffledShardedSource(src, buffer_size=16, seed=7).open_shard("shuffled_shard_0")]
    assert a == b


def test_shuffle_seed_changes_order():
    src = ListSource([{"id": i} for i in range(50)])
    a = [r["id"] for r in src.shuffle(buffer_size=16, seed=7).open_shard("shuffled_shard_0")]
    b = [r["id"] for r in src.shuffle(buffer_size=16, seed=8).open_shard("shuffled_shard_0")]
    assert a != b


def test_open_shard_at_row_resumes_from_tail():
    """Resume offset must reproduce the exact tail (fast-forward correctness)."""
    src = ListSource([{"id": i} for i in range(50)])
    sh = src.shuffle(buffer_size=16, seed=42)
    full = [r["id"] for r in sh.open_shard("shuffled_shard_0")]
    for k in (0, 5, 16, 33):
        resumed = [r["id"] for r in sh.open_shard_at_row("shuffled_shard_0", k)]
        assert resumed == full[k:]


def test_shuffle_collapses_to_single_shard():
    src = ListSource([{"id": i} for i in range(50)], nshards=4)
    sh = src.shuffle(buffer_size=8, seed=1)
    assert list(sh.shard_names) == ["shuffled_shard_0"]
    assert sh.num_shards() == 1
    assert sorted(r["id"] for r in _drain(sh)) == list(range(50))


def test_buffer_size_must_be_positive():
    src = ListSource([{"id": 0}])
    try:
        ShuffledShardedSource(src, buffer_size=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("buffer_size=0 should raise ValueError")


def test_mix_then_shuffle_preserves_weights_and_source_tags():
    a = ListSource([{"text": f"A{i}"} for i in range(200)])
    b = ListSource([{"text": f"B{i}"} for i in range(200)])
    mixed = MixedShardedSource(
        {"A": a, "B": b},
        weights={"A": 1, "B": 9},
        block_size=20,
        seed=7,
        stop_strategy="first_exhausted",
    )
    shuffled = mixed.shuffle(buffer_size=64, seed=7)
    rows = list(shuffled.open_shard("shuffled_shard_0"))

    # __source__ tagging from the mixer survives the shuffle.
    assert all("__source__" in r for r in rows)

    # Weighted mix is preserved (B drawn ~9x as often as A).
    n_a = sum(1 for r in rows if r["text"].startswith("A"))
    n_b = sum(1 for r in rows if r["text"].startswith("B"))
    assert n_b > n_a * 5

    # Output is decorrelated from file order (the bug being fixed: mixed-but-not-shuffled).
    texts = [r["text"] for r in rows]
    assert texts[:8] != [f"B{i}" for i in range(8)]
