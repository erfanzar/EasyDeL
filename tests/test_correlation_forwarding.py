# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for correlation propagation at controller/agent boundaries."""

from eyvan.capacity.drivers.base import MutationContext
from eyvan.worker.agent.models import WorkloadMutationRequest
from eyvan.worker.agent.service import CellAgentService


def test_remote_predecessor_retirement_forwards_correlation() -> None:
    forwarded: list[WorkloadMutationRequest] = []
    service = object.__new__(CellAgentService)
    service._remote_retire = lambda _cell, _remote, request: forwarded.append(request)
    context = MutationContext(
        idempotency_key="submit",
        generation=4,
        fencing_token="new-fence",
        correlation_id="corr-remote",
    )

    service._retire_predecessor(
        "workload",
        "replacement-cell",
        context,
        {
            "cell_id": "old-cell",
            "remote_id": "old-remote",
            "generation": 3,
            "fencing_token": "old-fence",
        },
    )

    assert len(forwarded) == 1
    assert forwarded[0].mutation.correlation_id == "corr-remote"
    assert forwarded[0].mutation.generation == 3
    assert forwarded[0].mutation.fencing_token == "old-fence"


def test_legacy_gang_predecessor_retirement_forwards_incoming_correlation() -> None:
    calls: list[MutationContext] = []
    service = object.__new__(CellAgentService)
    service._retire_predecessor = lambda _workload, _cell, context, _predecessor: calls.append(context)

    service._retire_gang_predecessor(
        "workload",
        "coordinator",
        7,
        {"kind": "single", "cell_id": "old-cell", "remote_id": "old-remote", "generation": 6, "fencing_token": "f"},
        correlation_id="corr-gang",
    )

    assert len(calls) == 1
    assert calls[0].correlation_id == "corr-gang"


def test_mutation_without_correlation_keeps_legacy_wire_shape() -> None:
    request = WorkloadMutationRequest(mutation={"idempotency_key": "k", "generation": 1, "fencing_token": "f"})

    assert "correlation_id" not in request.model_dump(mode="json")


def test_mutation_with_correlation_is_serialized() -> None:
    request = WorkloadMutationRequest(
        mutation={
            "idempotency_key": "k",
            "generation": 1,
            "fencing_token": "f",
            "correlation_id": "corr",
        }
    )

    assert request.model_dump(mode="json")["mutation"]["correlation_id"] == "corr"


def test_legacy_gang_predecessor_default_correlation_is_none() -> None:
    calls: list[MutationContext] = []
    service = object.__new__(CellAgentService)
    service._retire_predecessor = lambda _workload, _cell, context, _predecessor: calls.append(context)

    service._retire_gang_predecessor(
        "workload",
        "coordinator",
        7,
        {"kind": "single", "cell_id": "old-cell", "remote_id": "old-remote", "generation": 6, "fencing_token": "f"},
    )

    assert calls[0].correlation_id is None
