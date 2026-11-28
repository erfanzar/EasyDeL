# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Longitudinal submission/replay/cancellation checks on one growing repository.

This is a repository soak, not a live-provider or accelerator proof. Nightly
CI increases the iteration count and requires the PostgreSQL fixture; ordinary
PR runs retain a small sample on the same code path.
"""

import pytest
from test_chaos import _principal, _spec
from test_group_lease_lifecycle import repository as repository

from eyvan.records import WorkloadState
from eyvan.server.invariants import check_fleet
from eyvan.server.service import ControlPlaneService


async def test_replays_and_cancellations_remain_consistent_as_history_grows(repository, request) -> None:
    iterations = request.config.getoption("--soak-iterations")
    if not 1 <= iterations <= 10000:
        raise pytest.UsageError("--soak-iterations must be between 1 and 10000")
    principal = _principal()
    identifiers = set()
    for iteration in range(iterations):
        # Reconstruct the service repeatedly: correctness must live in the
        # repository, not in a service object's transient caches.
        service = ControlPlaneService(repository)
        key = f"soak-submit-{iteration}"
        workload, replayed = await service.submit_workload(
            principal, org_id="acme", project_id="research", spec=_spec(), idempotency_key=key
        )
        assert not replayed
        assert workload.id not in identifiers
        identifiers.add(workload.id)
        recovered = ControlPlaneService(repository)
        duplicate, replayed = await recovered.submit_workload(
            principal, org_id="acme", project_id="research", spec=_spec(), idempotency_key=key
        )
        assert replayed and duplicate.id == workload.id
        cancel_key = f"soak-cancel-{iteration}"
        await recovered.cancel_workload(principal, workload.id, idempotency_key=cancel_key)
        _, replayed = await service.cancel_workload(principal, workload.id, idempotency_key=cancel_key)
        assert replayed
        async with repository.transaction() as transaction:
            stored = await transaction.get_workload(workload.id)
            assert stored.state is WorkloadState.CANCELLED
        if iteration % 25 == 0:
            assert await check_fleet(repository) == ()
    assert await check_fleet(repository) == ()
