# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""An elastic pack runs one JAX world per cell, not one across all of them.

Separate GCP queued resources have no shared interconnect, so ranks in an
elastic pack must not be dispatched into one coordination group across cells.

The gang executor's default is still one gang-wide world: correct when the
member cells share a fabric, which is what a GPU gang and a pod group over
one allocation are. Elastic placement opts each rank into its own cell's
world instead -- coordinator, process count, and process id local to the
cell -- while the ``EYVAN_*`` variables keep describing the whole gang.
"""

from __future__ import annotations

import httpx

from eyvan.capacity.drivers.base import MutationContext
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember


def _executor(handler) -> NativeGangExecutor:
    endpoints = ("http://10.12.0.10:8100", "http://10.12.0.11:8100", "http://10.13.0.10:8100")
    clients = [httpx.Client(transport=httpx.MockTransport(handler), base_url=endpoint) for endpoint in endpoints]
    members = (
        NativeGangMember(
            cell_id="cell-a",
            endpoints=endpoints[:2],
            secret="secret-a",
            context=MutationContext(idempotency_key="submit-a", generation=7, fencing_token="fence-a"),
        ),
        NativeGangMember(
            cell_id="cell-b",
            endpoints=endpoints[2:],
            secret="secret-b",
            context=MutationContext(idempotency_key="submit-b", generation=7, fencing_token="fence-b"),
        ),
    )
    return NativeGangExecutor(members, clients=clients)


def _submit(spec: dict) -> list[dict[str, str]]:
    submitted: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        submitted.append(dict(payload["spec"]["environment"]))
        return httpx.Response(200, json={"remote_id": "gang-job"})

    executor = _executor(handler)
    try:
        assert executor.submit(spec) == "gang-job"
    finally:
        for worker in executor._workers:
            worker.executor.close()
    return submitted


def test_an_elastic_pack_runs_one_jax_world_per_cell() -> None:
    environments = _submit({"submission_id": "gang-job", "entrypoint": "python train.py", "placement": "elastic"})

    assert len(environments) == 3
    by_coordinator: dict[str, list[dict[str, str]]] = {}
    for environment in environments:
        by_coordinator.setdefault(environment["JAX_COORDINATOR_ADDRESS"], []).append(environment)
    assert set(by_coordinator) == {"10.12.0.10:12345", "10.13.0.10:12345"}

    world_a = sorted(by_coordinator["10.12.0.10:12345"], key=lambda env: int(env["JAX_PROCESS_ID"]))
    assert [env["JAX_PROCESS_ID"] for env in world_a] == ["0", "1"]
    assert all(env["JAX_PROCESS_COUNT"] == "2" for env in world_a)

    world_b = by_coordinator["10.13.0.10:12345"]
    assert world_b[0]["JAX_PROCESS_ID"] == "0"
    assert world_b[0]["JAX_PROCESS_COUNT"] == "1"

    # The gang itself stays one shape: every rank still knows the pack's width
    # and its global identity, so a callable can address worlds and ranks.
    for rank, environment in enumerate(sorted(environments, key=lambda env: int(env["EYVAN_DISTRIBUTED_PROCESS_ID"]))):
        assert environment["EYVAN_DISTRIBUTED_PROCESS_COUNT"] == "3"
        assert environment["EYVAN_DISTRIBUTED_PROCESS_ID"] == str(rank)
        assert environment["EYVAN_GANG_CELL_INDEX"] == ("0" if rank < 2 else "1")


def test_a_fabric_gang_still_runs_one_jax_world_across_cells() -> None:
    environments = _submit({"submission_id": "gang-job", "entrypoint": "python train.py"})

    assert len(environments) == 3
    for environment in environments:
        assert environment["JAX_COORDINATOR_ADDRESS"] == "10.12.0.10:12345"
        assert environment["JAX_PROCESS_COUNT"] == "3"
    assert sorted(env["JAX_PROCESS_ID"] for env in environments) == ["0", "1", "2"]
