# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Quarantine entries must have an owner, tracking issue and short expiry."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import pytest_collection_modifyitems


def _item(metadata):
    marker = pytest.mark.quarantine(**metadata).mark
    return SimpleNamespace(
        path=Path("unrelated.py"),
        nodeid="unrelated.py::test_case",
        iter_markers=lambda name: iter([marker] if name == "quarantine" else []),
    )


def _valid():
    return {
        "issue": "https://github.com/example/project/issues/1",
        "owner": "maintainer",
        "until": (datetime.now(UTC).date() + timedelta(days=7)).isoformat(),
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("issue", ""),
        ("issue", "not-an-issue"),
        ("owner", " "),
        ("until", "2000-01-01"),
        ("until", "2099-01-01"),
        ("until", "not-a-date"),
    ],
)
def test_invalid_quarantine_metadata_fails_collection(field, value):
    metadata = _valid()
    metadata[field] = value
    with pytest.raises(pytest.UsageError):
        pytest_collection_modifyitems([_item(metadata)])


def test_a_tracked_short_lived_quarantine_is_not_silently_skipped():
    item = _item(_valid())
    # No add_marker method: silently injecting skip/xfail would fail this test.
    pytest_collection_modifyitems([item])
