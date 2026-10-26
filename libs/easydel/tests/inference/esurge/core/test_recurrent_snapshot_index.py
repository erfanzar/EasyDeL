"""Host-side bookkeeping of recurrent prefix snapshots (no array computation)."""

import pytest
from easydel.inference.esurge.core.recurrent_snapshots import RecurrentSnapshotIndex

pytestmark = pytest.mark.cpu_ok


def test_reserved_slot_is_only_visible_after_commit():
    index = RecurrentSnapshotIndex(2)
    slot = index.reserve("a")
    assert slot is not None
    assert index.contains("a") and not index.is_ready("a")
    assert index.lookup("a") is None
    assert index.reserve("a") is None
    index.commit(slot)
    assert index.is_ready("a")
    assert index.lookup("a") == slot


def test_cancel_returns_slot_to_free_list():
    index = RecurrentSnapshotIndex(1)
    slot = index.reserve("a")
    index.cancel(slot)
    assert not index.contains("a")
    assert index.reserve("b") == slot


def test_full_index_recycles_least_recently_used_ready_slot():
    index = RecurrentSnapshotIndex(2)
    sa = index.reserve("a")
    index.commit(sa)
    sb = index.reserve("b")
    index.commit(sb)
    index.lookup("a")
    sc = index.reserve("c")
    assert sc == sb
    assert not index.contains("b")
    assert index.lookup("a") == sa


def test_reserve_never_recycles_excluded_or_pending_slots():
    index = RecurrentSnapshotIndex(2)
    sa = index.reserve("a")
    index.commit(sa)
    index.reserve("b")
    assert index.reserve("c", exclude={sa}) is None
    assert index.lookup("a") == sa


def test_reset_forgets_everything():
    index = RecurrentSnapshotIndex(2)
    index.commit(index.reserve("a"))
    index.reset()
    assert not index.contains("a")
    assert index.reserve("x") is not None and index.reserve("y") is not None
