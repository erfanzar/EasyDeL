# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The capacity lifecycle logger's optional structured sink (register #28).

Log lines are the always-on record; the JSONL sink is the durable one. These
tests pin what lands in the file: one self-contained JSON object per event,
flat string-typed identifiers, and a broken sink that never breaks the
transition it audits.
"""

from __future__ import annotations

import json
import logging
import pathlib
import typing as tp

import pytest

from eyvan.capacity.audit import CapacityLifecycleEvent, CapacityLifecycleLogger
from eyvan.records import CapacityStatus, CellRecord, CellState, ResourceVector


def _cell(state: CellState = CellState.REGISTERING) -> CellRecord:
    return CellRecord(
        id="cell-a",
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-central2",
        zone="us-central2-b",
        capacity=ResourceVector(tpu=8),
        allocatable=ResourceVector(tpu=8),
        accelerator_type="v5p-8",
        endpoint="http://agent:8090",
        state=state,
        provider_phase=CapacityStatus.REQUESTING,
    )


def test_every_transition_lands_one_json_object_in_the_sink(tmp_path: pathlib.Path) -> None:
    sink = tmp_path / "audit.jsonl"
    logger = CapacityLifecycleLogger(logging.getLogger("eyvan.test.audit"), sink_path=str(sink))
    try:
        assert logger.transition(_cell(), _cell(CellState.UNAVAILABLE)) is True
    finally:
        logger.close()

    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record: dict[str, tp.Any] = json.loads(lines[0])
    assert record["event"] == CapacityLifecycleEvent.UNHEALTHY.value
    assert record["cell"] == "cell-a"
    assert record["fleet"] == "-"
    assert record["project"] == "gcp-project-a"
    assert record["location"] == "us-central2-b"
    assert record["previous_cell_state"] == "registering"
    assert record["message"].startswith("capacity lifecycle event=")


def test_a_broken_sink_never_breaks_the_transition_it_audits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FailingSink:
        def write(self, payload: str) -> int:
            raise OSError("disk went away mid-audit")

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    logger = CapacityLifecycleLogger(logging.getLogger("eyvan.test.audit"))
    logger._sink_handle = _FailingSink()
    try:
        caplog.set_level(logging.WARNING, logger="eyvan.test.audit")
        assert logger.transition(_cell(), _cell(CellState.UNAVAILABLE)) is True
        assert any("sink write failed" in record.getMessage() for record in caplog.records)
    finally:
        logger._sink_handle = None
        logger.close()


def test_without_a_sink_nothing_is_written_and_transitions_are_unchanged() -> None:
    logger = CapacityLifecycleLogger(logging.getLogger("eyvan.test.audit"))
    assert logger.transition(_cell(), _cell()) is False
    assert logger.transition(_cell(), _cell(CellState.UNAVAILABLE)) is True


def test_a_runtime_probe_failure_on_healthy_capacity_is_not_a_preemption() -> None:
    """Require provider evidence of capacity loss before classifying a runtime probe failure as preemption."""

    ready = _cell(CellState.READY).model_copy(update={"provider_phase": CapacityStatus.HEALTHY})
    probe_failed = ready.model_copy(update={"state": CellState.REGISTERING})
    reclaimed = probe_failed.model_copy(update={"provider_phase": CapacityStatus.PROVISIONING})

    assert CapacityLifecycleLogger.classify(ready, probe_failed) is CapacityLifecycleEvent.UNHEALTHY
    assert CapacityLifecycleLogger.classify(ready, reclaimed) is CapacityLifecycleEvent.PREEMPTED
