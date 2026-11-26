# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistence reference contract for shared records and replacement rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from eyvan.records import CellRecord, ResourceVector
from eyvan.server.persistence.memory import InMemoryRepository


def _cell() -> CellRecord:
    return CellRecord(
        id="cell-1",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=1),
        allocatable=ResourceVector(cpu=1),
        labels={"phase": "committed"},
    )


def test_persistence_reference_documents_shared_record_caller_obligations() -> None:
    reference = (Path(__file__).resolve().parents[1] / "docs/reference/persistence.md").read_text(encoding="utf-8")
    section = reference.split("## Transactions and concurrency\n", 1)[1].split("\n## Tables and model families", 1)[0]

    assert "deep-copies the" not in section, (
        "The reference still promises isolation the state repository does not provide"
    )
    assert "structural copy" in section
    assert "shared" in section
    assert "read-only" in section
    assert "nested" in section
    assert 'labels = {**cell.labels, "phase": "updated"}' in section
    assert 'cell.model_copy(update={"labels": labels})' in section
    assert "await transaction.put_cell(updated)" in section
    assert "Provided callers follow this replacement-only contract" in section
    assert "verify_record_immutability" in section
    assert "disabled in production" in section
    assert "does not restore" in section


@pytest.mark.asyncio
async def test_reference_replacement_pattern_preserves_rollback_and_commits_new_values() -> None:
    repository = InMemoryRepository()
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell())

    async with repository.transaction() as transaction:
        original = await transaction.get_cell("cell-1")
    assert original is not None

    with pytest.raises(RuntimeError, match="abort"):
        async with repository.transaction() as transaction:
            cell = await transaction.get_cell("cell-1")
            assert cell is original
            labels = {**cell.labels, "phase": "updated"}
            updated = cell.model_copy(update={"labels": labels})
            await transaction.put_cell(updated)
            raise RuntimeError("abort")

    async with repository.transaction() as transaction:
        after_rollback = await transaction.get_cell("cell-1")
    assert after_rollback is original
    assert after_rollback.labels == {"phase": "committed"}

    async with repository.transaction() as transaction:
        cell = await transaction.get_cell("cell-1")
        assert cell is not None
        labels = {**cell.labels, "phase": "updated"}
        updated = cell.model_copy(update={"labels": labels})
        await transaction.put_cell(updated)

    async with repository.transaction() as transaction:
        committed = await transaction.get_cell("cell-1")
    assert committed is updated
    assert committed.labels == {"phase": "updated"}
    assert original.labels == {"phase": "committed"}


@pytest.mark.asyncio
async def test_unsupported_in_place_mutation_is_not_production_rollback_isolation() -> None:
    """Counterexample to the old prose, not a supported mutation API promise."""
    repository = InMemoryRepository()
    # The global test fixture enables diagnostics; reproduce the production
    # setting on this disposable instance without changing any other repository.
    repository.verify_record_immutability = False
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell())

    with pytest.raises(RuntimeError, match="abort"):
        async with repository.transaction() as transaction:
            cell = await transaction.get_cell("cell-1")
            assert cell is not None
            cell.labels["phase"] = "aborted"
            raise RuntimeError("abort")

    async with repository.transaction() as transaction:
        stored = await transaction.get_cell("cell-1")
    assert stored is not None
    assert stored.labels == {"phase": "aborted"}
