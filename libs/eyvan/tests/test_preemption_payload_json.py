# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Preemption phase payloads must be JSON before repository serialization."""

import json
from contextlib import asynccontextmanager
from datetime import datetime

from test_group_lease_lifecycle import _preemption
from test_priority_preemption_properties import _history
from test_priority_preemption_properties import repository as _repository_fixture

from eyvan.records import LeaseState, WorkloadState

repository = _repository_fixture


class JsonPayloadRepository:
    """Check each durable intermediate write, not just the final decoded row."""

    def __init__(self, inner):
        self.inner = inner
        self.timestamps = {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    @asynccontextmanager
    async def transaction(self):
        async with self.inner.transaction() as tx:
            owner = self

            class Session:
                def __getattr__(self, name):
                    return getattr(tx, name)

                async def put_outbox(self, record):
                    if record.topic == "workload.preempt":
                        # No model serializer/default=str to hide invalid JSON values.
                        assert json.loads(json.dumps(record.payload)) == record.payload
                        for name in ("execution_started_at", "stopping_started_at", "execution_completed_at"):
                            if name in record.payload:
                                value = record.payload[name]
                                assert datetime.fromisoformat(value).tzinfo is not None
                                owner.timestamps[name] = value
                    await tx.put_outbox(record)

            yield Session()


async def test_preemption_phases_write_json_payloads(repository):
    checked = JsonPayloadRepository(repository)
    await _history(checked, ("RUNNING", "RUNNING"), 1)
    assert set(checked.timestamps) == {"execution_started_at", "stopping_started_at", "execution_completed_at"}


async def test_checkpoint_abort_writes_json_and_restores_victims(repository):
    checked = JsonPayloadRepository(repository)
    controller, command = await _preemption(checked)
    prepared = await controller._prepare_preemption_plan(command)
    assert prepared is not None
    assert await controller._abort_preemption_checkpoint_phase(prepared, "checkpoint failed")
    async with checked.transaction() as tx:
        saved = await tx.get_outbox(command.id)
        victim = await tx.get_workload("victim")
        leases = await tx.list_leases()
    assert saved.payload["execution_completed"] is True
    assert saved.payload["aborted_reason"] == "checkpoint failed"
    assert victim.state is WorkloadState.RUNNING and victim.preemption_id is None
    assert all(lease.state is LeaseState.ACTIVE and lease.revocation_reason is None for lease in leases)
    assert set(checked.timestamps) == {"execution_started_at", "execution_completed_at"}
