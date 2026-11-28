# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Several Eyvan ranks on one GPU host, and what each of them is told it owns.

W7 assigned devices one rank per host and W9 made a real fleet reach that path.
Both stopped at the same wall: ``plan_gpu_devices`` handed every rank its host's
devices ``0..n-1``, which is exact only while one rank owns one host, so a gang
whose endpoints named the same machine twice was *declined* rather than given an
assignment that would double-book eight cards. Declining was right and it left a
multi-rank host unsupported.

This file covers the grouping that removes the wall: ranks that share a host
split that host's devices, ``LOCAL_RANK`` names which share a process holds, and
the gang-wide ordinals still cover every device exactly once. It also pins the
part that must not move -- a gang with one rank per host is dispatched the same
bytes it is dispatched today, down to the mapping.

**No claim here is hardware-qualified.** No GPU has executed an Eyvan workload.
Every assertion is about values Eyvan computes and HTTP payloads it emits, which
is as far as a host without a GPU can honestly check. In particular, nothing
below is evidence that two processes sharing a machine train, only that they are
told to use different cards.

**Device isolation between two *workloads* on one cell is not shipped.** The
grouping here is within one gang, whose ranks Eyvan places together. Two
independently scheduled workloads packed onto the same cell still receive the
same devices; :func:`test_two_gangs_on_one_cell_are_still_given_the_same_devices`
states that as a fact rather than leaving it to be discovered, and says what
does currently keep two workloads off one native cell.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan.capacity.drivers.base import MutationContext
from eyvan.capacity.drivers.manual import ManualCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.worker.distributed_environment import (
    DEFAULT_GPU_RANK_LAYOUT,
    GpuRankLayout,
    gpu_rank_placement_error,
    plan_gpu_devices,
)
from eyvan.worker.runtime import native_remote
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember
from eyvan.worker.runtime.registry import (
    GPU_RANK_LAYOUT_ENV,
    ExecutionDriverResolver,
    NativeGangCell,
    gpu_rank_layout_from_environment,
)

#: Captured before anything patches it, so two dispatches in one test cannot
#: nest their transports: ``native_remote.httpx`` *is* the httpx module, so a
#: second patch that reads ``httpx.Client`` would read the first patch.
_HTTPX_CLIENT = httpx.Client

#: An eight-device host and a four-device host, both shipped catalog rows.
BIG_GPU = "a3-highgpu-8g"
SMALL_GPU = "g2-standard-48"

#: A real GCP machine with no catalog row, which is how a cell reports an
#: unknown device count without anything being mocked.
UNCATALOGUED_GPU = "a2-ultragpu-8g"

#: Exactly what a two-cell, one-rank-per-host GPU gang is dispatched with
#: defaults, written out in full. Fatality ships on; the deadlines keep their
#: measured values.
#: A subset would let a new variable appear unnoticed, and the property being
#: defended is that *nothing* moved for the shape that ships.
TODAYS_DISPATCH: dict[str, dict[str, str]] = {
    "10.0.0.10:8100": {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "EYVAN_DISTRIBUTED_PROCESS_COUNT": "2",
        "EYVAN_DISTRIBUTED_PROCESS_ID": "0",
        "EYVAN_GANG_CELL_ID": "cell-a",
        "EYVAN_GANG_CELL_INDEX": "0",
        "EYVAN_GPU_GLOBAL_DEVICE_COUNT": "12",
        "EYVAN_GPU_GLOBAL_DEVICE_OFFSET": "0",
        "EYVAN_GPU_HOST_COUNT": "2",
        "EYVAN_GPU_HOST_INDEX": "0",
        "EYVAN_GPU_LOCAL_DEVICE_COUNT": "8",
        "EYVAN_GPU_MODEL": "nvidia-h100-80gb",
        "EYVAN_JAX_HEARTBEAT_TIMEOUT_S": "900",
        "EYVAN_JAX_INIT_FATAL": "1",
        "EYVAN_JAX_INIT_TIMEOUT_S": "600",
        "EYVAN_JAX_MULTIHOST": "1",
        "EYVAN_REPLICA_COUNT": "2",
        "EYVAN_REPLICA_INDEX": "0",
        "JAX_COORDINATOR_ADDRESS": "10.0.0.10:12345",
        "JAX_COORDINATOR_PORT": "12345",
        "JAX_PARTITION_INDEX": "0",
        "JAX_PROCESS_COUNT": "2",
        "JAX_PROCESS_ID": "0",
        "JAX_PROCESS_INDEX": "0",
        "LOCAL_WORLD_SIZE": "8",
        "MASTER_ADDR": "10.0.0.10",
        "MASTER_PORT": "12346",
        "NODE_RANK": "0",
        "NVIDIA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "PYTHONPATH": "/opt/eyvan-native/multihost",
        "USER_FLAG": "kept",
        "WORLD_SIZE": "12",
    },
    "10.0.0.20:8100": {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "EYVAN_DISTRIBUTED_PROCESS_COUNT": "2",
        "EYVAN_DISTRIBUTED_PROCESS_ID": "1",
        "EYVAN_GANG_CELL_ID": "cell-b",
        "EYVAN_GANG_CELL_INDEX": "1",
        "EYVAN_GPU_GLOBAL_DEVICE_COUNT": "12",
        "EYVAN_GPU_GLOBAL_DEVICE_OFFSET": "8",
        "EYVAN_GPU_HOST_COUNT": "2",
        "EYVAN_GPU_HOST_INDEX": "1",
        "EYVAN_GPU_LOCAL_DEVICE_COUNT": "4",
        "EYVAN_GPU_MODEL": "nvidia-l4",
        "EYVAN_JAX_HEARTBEAT_TIMEOUT_S": "900",
        "EYVAN_JAX_INIT_FATAL": "1",
        "EYVAN_JAX_INIT_TIMEOUT_S": "600",
        "EYVAN_JAX_MULTIHOST": "1",
        "EYVAN_REPLICA_COUNT": "2",
        "EYVAN_REPLICA_INDEX": "1",
        "JAX_COORDINATOR_ADDRESS": "10.0.0.10:12345",
        "JAX_COORDINATOR_PORT": "12345",
        "JAX_PARTITION_INDEX": "1",
        "JAX_PROCESS_COUNT": "2",
        "JAX_PROCESS_ID": "1",
        "JAX_PROCESS_INDEX": "1",
        "LOCAL_WORLD_SIZE": "4",
        "MASTER_ADDR": "10.0.0.10",
        "MASTER_PORT": "12346",
        "NODE_RANK": "1",
        "NVIDIA_VISIBLE_DEVICES": "0,1,2,3",
        "PYTHONPATH": "/opt/eyvan-native/multihost",
        "USER_FLAG": "kept",
        "WORLD_SIZE": "12",
    },
}


def _context(key: str) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=1, fencing_token="fence-1")


def _driver(tmp_path, name: str = "fleet") -> ManualCapacityDriver:
    return ManualCapacityDriver(FleetStateStore(LocalBackend(tmp_path / f"{name}.json")))


def _desired(accelerator_type: str, *hosts: str) -> JsonObject:
    return {
        "provider": "manual",
        "capacity_backend": "manual",
        "executor": "eyvan-native",
        "accelerator_type": accelerator_type,
        "capacity_class": "on_demand",
        "region": "on-prem",
        "hosts": list(hosts),
    }


def _cells(driver: ManualCapacityDriver, *declared: tuple[str, str, tuple[str, ...]]) -> list[NativeGangCell]:
    """Apply each cell for real, then re-observe it exactly as the agent does."""

    cells = []
    for cell_id, accelerator_type, hosts in declared:
        driver.apply_capacity(cell_id, _desired(accelerator_type, *hosts), _context(f"apply-{cell_id}"))
        cells.append(
            NativeGangCell(
                cell_id=cell_id,
                observation=driver.observe(cell_id),
                context=_context(f"submit-{cell_id}"),
            )
        )
    return cells


def _dispatch(
    monkeypatch: pytest.MonkeyPatch,
    cells: tp.Sequence[NativeGangCell],
    environment: JsonObject | None = None,
    *,
    layout: GpuRankLayout | None = None,
    submission_id: str = "gpu-job",
) -> dict[str, JsonObject]:
    """Submit through the resolved gang and return each endpoint's environment.

    Keyed by endpoint, not collected in order: which machine a rank lands on is
    the subject, so reading the result positionally would assume the answer. Two
    endpoints on one host differ by port, which is exactly the case under test.
    """

    dispatched: dict[str, JsonObject] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert is_json_object(payload)
        spec = payload["spec"]
        assert is_json_object(spec)
        spec_environment = spec["environment"]
        assert is_json_object(spec_environment)
        dispatched[f"{request.url.host}:{request.url.port}"] = spec_environment
        return httpx.Response(200, json={"remote_id": submission_id})

    opened: list[httpx.Client] = []

    def client_factory(*_args: object, **kwargs: object) -> httpx.Client:
        base_url = kwargs.get("base_url")
        assert isinstance(base_url, str)
        client = _HTTPX_CLIENT(base_url=base_url, transport=httpx.MockTransport(handler))
        opened.append(client)
        return client

    monkeypatch.setattr(native_remote.httpx, "Client", client_factory)
    resolver = ExecutionDriverResolver(
        backend="eyvan-native",
        native_worker_token="root-secret",
        gpu_rank_layout=layout,
    )
    executor = resolver.resolve_gang(tuple(cells))
    try:
        assert (
            executor.submit(
                {
                    "entrypoint": "python train.py",
                    "submission_id": submission_id,
                    "environment": dict(environment or {}),
                }
            )
            == submission_id
        )
    finally:
        for client in opened:
            client.close()
    return dispatched


def _owned_devices(dispatched: tp.Mapping[str, JsonObject]) -> dict[str, set[tuple[str, int]]]:
    """Return the (machine, host-local device) pairs each endpoint was given.

    The pair is the unit of ownership. A host-local index repeats across
    machines by construction, so ``0`` alone cannot say whether two ranks hold
    one card or two, and the machine is what disambiguates it.
    """

    owned: dict[str, set[tuple[str, int]]] = {}
    for endpoint, environment in dispatched.items():
        machine = endpoint.split(":")[0]
        visible = str(environment["CUDA_VISIBLE_DEVICES"])
        owned[endpoint] = {(machine, int(index)) for index in visible.split(",")}
    return owned


# ------------------------------------------------------------- the placement


def test_two_ranks_on_one_host_split_it_instead_of_each_claiming_all_of_it() -> None:
    """The defect this exists to fix, at the level where the assignment is made."""

    plan = plan_gpu_devices([8, 8], rank_hosts=["worker-a", "worker-a"])

    assert [assignment.host_devices for assignment in plan.assignments] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert [assignment.local_rank for assignment in plan.assignments] == [0, 1]
    assert [assignment.host_index for assignment in plan.assignments] == [0, 0]
    assert plan.host_count == 1, "two ranks on one machine are one machine"
    assert plan.rank_count == 2
    assert plan.device_count == 8, "the host has eight cards, not sixteen"


def test_four_ranks_over_two_hosts_own_every_device_exactly_once() -> None:
    """Exact and non-overlapping gang-wide, which is what a collective depends on."""

    plan = plan_gpu_devices([8, 8, 8, 8], rank_hosts=["worker-a", "worker-a", "worker-b", "worker-b"])

    held = [(assignment.host_index, device) for assignment in plan.assignments for device in assignment.host_devices]
    assert len(held) == len(set(held)) == 16, "one card, one owner"
    assert sorted(held) == [(host, device) for host in (0, 1) for device in range(8)]
    globals_seen = [index for assignment in plan.assignments for index in assignment.global_devices]
    assert globals_seen == list(range(16)), "the gang's devices are covered once each, in rank order"
    assert [assignment.global_device_offset for assignment in plan.assignments] == [0, 4, 8, 12]
    assert plan.host_count == 2
    assert plan.rank_count == 4


def test_naming_the_hosts_of_a_one_rank_per_host_gang_changes_nothing() -> None:
    """The rolling-upgrade property at its source: grouping is not a new placement.

    A gang whose ranks are alone on their machines must produce the identical
    plan whether or not the hosts were passed, or every gang that ships today
    would be re-placed by an upgrade.
    """

    assert plan_gpu_devices([8, 1, 4], rank_hosts=["a", "b", "c"]) == plan_gpu_devices([8, 1, 4])


def test_a_host_whose_devices_do_not_divide_among_its_ranks_is_refused() -> None:
    """Three ranks on an eight-card host is a split that does not exist.

    Handing out 3/3/2 makes one rank smaller than its peers, and a collective
    sized by the largest would read past the end of the smallest; leaving the
    two spare cards idle would mean the plan does not cover the host it claims.
    Neither is a placement, so neither is invented.
    """

    with pytest.raises(ValueError, match="8 GPUs, which 3 ranks cannot divide evenly"):
        plan_gpu_devices([8, 8, 8], rank_hosts=["worker-a", "worker-a", "worker-a"])


def test_one_machine_described_with_two_different_device_counts_is_refused() -> None:
    """Two cells claiming one host disagree about it, and nothing here can arbitrate."""

    reason = gpu_rank_placement_error([8, 4], ["worker-a", "worker-a"])

    assert reason is not None
    assert "both 8 and 4 GPUs" in reason
    with pytest.raises(ValueError, match="both 8 and 4 GPUs"):
        plan_gpu_devices([8, 4], rank_hosts=["worker-a", "worker-a"])


@pytest.mark.parametrize("host", ["", "   ", " worker-a"])
def test_a_rank_whose_host_is_not_a_host_is_refused(host: str) -> None:
    """Grouping is by host name, so an unusable name would group by accident."""

    reason = gpu_rank_placement_error([8, 8], ["worker-a", host])

    assert reason is not None
    assert "non-empty trimmed host" in reason


def test_placing_ranks_on_distinct_hosts_is_never_an_error() -> None:
    """The shape that ships must not acquire a new way to be refused."""

    assert gpu_rank_placement_error([8, 1, 4], ["a", "b", "c"]) is None


# ------------------------------------------------------------- what a rank reads


def test_a_rank_sharing_a_host_is_told_which_of_them_it_is() -> None:
    """``LOCAL_RANK`` is a fact about Eyvan's own dispatch once a host holds two.

    It was left unset while a rank meant a host-local launcher, because it then
    named a worker inside a launcher Eyvan does not run. Under the layout where
    the rank *is* the process Eyvan started, it names that process, and
    ``WORLD_SIZE`` counts processes with it -- a ``RANK`` that does not index
    into ``WORLD_SIZE`` is how a collective hangs instead of failing.
    """

    plan = plan_gpu_devices([8, 8], rank_hosts=["worker-a", "worker-a"])
    first, second = (
        plan.rank_environment(rank, rendezvous_host="10.0.0.10", layout=GpuRankLayout.RANK_PROCESS) for rank in (0, 1)
    )

    assert (first["LOCAL_RANK"], second["LOCAL_RANK"]) == ("0", "1")
    assert (first["RANK"], second["RANK"]) == ("0", "1")
    assert (first["CUDA_VISIBLE_DEVICES"], second["CUDA_VISIBLE_DEVICES"]) == ("0,1,2,3", "4,5,6,7")
    assert first["LOCAL_WORLD_SIZE"] == second["LOCAL_WORLD_SIZE"] == "2", "processes on this machine"
    assert first["WORLD_SIZE"] == second["WORLD_SIZE"] == "2"
    assert first["NODE_RANK"] == second["NODE_RANK"] == "0", "one machine is one node"
    assert first["GROUP_RANK"] == second["GROUP_RANK"] == "0"
    assert first["EYVAN_GPU_LOCAL_DEVICE_COUNT"] == second["EYVAN_GPU_LOCAL_DEVICE_COUNT"] == "4"
    assert first["EYVAN_GPU_GLOBAL_DEVICE_COUNT"] == second["EYVAN_GPU_GLOBAL_DEVICE_COUNT"] == "8"


def test_the_launcher_layout_refuses_to_describe_a_shared_host_at_all() -> None:
    """It has one name for the machine and no name for the second process on it.

    Rendering it anyway is the original defect: both ranks would read
    ``NODE_RANK=0`` and be handed the same cards.
    """

    plan = plan_gpu_devices([8, 8], rank_hosts=["worker-a", "worker-a"])

    with pytest.raises(ValueError, match="has no name for a second rank on one host"):
        plan.rank_environment(0, rendezvous_host="10.0.0.10", layout=GpuRankLayout.HOST_LAUNCHER)


def test_the_layout_a_rank_environment_defaults_to_is_the_one_that_ships() -> None:
    """An upgraded agent nobody reconfigured must keep sending what it sends."""

    plan = plan_gpu_devices([8, 4])

    assert DEFAULT_GPU_RANK_LAYOUT is GpuRankLayout.HOST_LAUNCHER
    assert plan.rank_environment(0, rendezvous_host="10.0.0.10") == plan.rank_environment(
        0, rendezvous_host="10.0.0.10", layout=GpuRankLayout.HOST_LAUNCHER
    )
    assert "RANK" not in plan.rank_environment(0, rendezvous_host="10.0.0.10")


def test_the_two_layouts_disagree_about_processes_and_agree_about_hardware() -> None:
    """Opting in changes what a rank *is*, and must not change what it holds.

    An explicit difference set distinguishes process-counted ``WORLD_SIZE``
    from the hardware allocation that both layouts share.
    """

    plan = plan_gpu_devices([8, 4], rank_hosts=["worker-a", "worker-b"])
    launcher = plan.rank_environment(0, rendezvous_host="10.0.0.10", layout=GpuRankLayout.HOST_LAUNCHER)
    process = plan.rank_environment(0, rendezvous_host="10.0.0.10", layout=GpuRankLayout.RANK_PROCESS)

    assert set(process) - set(launcher) == {"RANK", "LOCAL_RANK", "GROUP_RANK"}
    assert not set(launcher) - set(process)
    assert {name for name in launcher if launcher[name] != process[name]} == {"WORLD_SIZE", "LOCAL_WORLD_SIZE"}
    assert (launcher["WORLD_SIZE"], process["WORLD_SIZE"]) == ("12", "2")
    assert (launcher["LOCAL_WORLD_SIZE"], process["LOCAL_WORLD_SIZE"]) == ("8", "1")
    for shared in ("CUDA_VISIBLE_DEVICES", "EYVAN_GPU_LOCAL_DEVICE_COUNT", "EYVAN_GPU_GLOBAL_DEVICE_COUNT"):
        assert launcher[shared] == process[shared], "the hardware a rank holds is not a rendering choice"


# ---------------------------------------------------------------- the wire


def test_a_one_rank_per_host_gang_dispatches_exactly_what_it_dispatches_today(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole mapping, both ranks, compared against a captured baseline.

    Not a subset: a subset would let a new variable appear unnoticed, and what
    is being defended is that an upgrade changes nothing at all for the shape
    every GPU-coordinated gang has today.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100",)),
        ("cell-b", SMALL_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, {"USER_FLAG": "kept"})

    assert dispatched == TODAYS_DISPATCH


def test_two_ranks_on_one_cell_host_are_dispatched_disjoint_devices(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap closed, end to end: one machine, two native workers, eight cards.

    Both endpoints resolve to ``10.0.0.10``. Before grouping this gang was
    declined outright, because both ranks would have been told they owned all
    eight.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, layout=GpuRankLayout.RANK_PROCESS)

    assert set(dispatched) == {"10.0.0.10:8100", "10.0.0.10:8101", "10.0.0.20:8100"}
    first, second = dispatched["10.0.0.10:8100"], dispatched["10.0.0.10:8101"]
    assert (first["CUDA_VISIBLE_DEVICES"], second["CUDA_VISIBLE_DEVICES"]) == ("0,1,2,3", "4,5,6,7")
    assert (first["LOCAL_RANK"], second["LOCAL_RANK"]) == ("0", "1")
    assert first["NODE_RANK"] == second["NODE_RANK"] == "0", "they are on one machine"
    assert dispatched["10.0.0.20:8100"]["LOCAL_RANK"] == "0"
    assert dispatched["10.0.0.20:8100"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    owned = _owned_devices(dispatched)
    assert not owned["10.0.0.10:8100"] & owned["10.0.0.10:8101"], "no card has two owners"
    assert len(set().union(*owned.values())) == 16
    assert {str(environment["EYVAN_GPU_GLOBAL_DEVICE_COUNT"]) for environment in dispatched.values()} == {"16"}
    assert {str(environment["EYVAN_GPU_HOST_COUNT"]) for environment in dispatched.values()} == {"2"}


def test_four_ranks_over_two_hosts_are_dispatched_exact_and_non_overlapping(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two machines, two native workers each, sixteen cards, sixteen owners of one."""

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100", "http://10.0.0.20:8101")),
    )

    dispatched = _dispatch(monkeypatch, cells, layout=GpuRankLayout.RANK_PROCESS)

    owned = _owned_devices(dispatched)
    held = [card for cards in owned.values() for card in cards]
    assert len(held) == len(set(held)) == 16, "one card, one owner, gang-wide"
    assert sorted(set(held)) == [(machine, device) for machine in ("10.0.0.10", "10.0.0.20") for device in range(8)]
    ordinals: list[int] = []
    for environment in dispatched.values():
        offset = int(str(environment["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"]))
        ordinals.extend(range(offset, offset + int(str(environment["EYVAN_GPU_LOCAL_DEVICE_COUNT"]))))
    assert sorted(ordinals) == list(range(16))
    assert sorted(str(environment["RANK"]) for environment in dispatched.values()) == ["0", "1", "2", "3"]
    assert {str(environment["WORLD_SIZE"]) for environment in dispatched.values()} == {"4"}
    assert sorted(
        (str(environment["NODE_RANK"]), str(environment["LOCAL_RANK"])) for environment in dispatched.values()
    ) == [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")]


def test_a_submission_cannot_spoof_its_local_rank_or_its_device_visibility(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property that has to hold whatever the caller sends, on the shared-host path.

    A submission that set ``LOCAL_RANK=0`` on both ranks of a machine would put
    both processes on the same four cards while the fleet reported a healthy
    gang; one that set ``CUDA_VISIBLE_DEVICES`` would do it directly.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )
    spoofed = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
        "LOCAL_RANK": "0",
        "RANK": "0",
        "GROUP_RANK": "7",
        "NODE_RANK": "7",
        "LOCAL_WORLD_SIZE": "64",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "attacker.invalid",
        "MASTER_PORT": "1",
        "NCCL_COMM_ID": "attacker.invalid:1",
        "NCCL_DEBUG": "INFO",
        "EYVAN_GPU_GLOBAL_DEVICE_OFFSET": "0",
        "USER_FLAG": "kept",
    }

    dispatched = _dispatch(monkeypatch, cells, spoofed, layout=GpuRankLayout.RANK_PROCESS)

    first, second = dispatched["10.0.0.10:8100"], dispatched["10.0.0.10:8101"]
    assert (first["LOCAL_RANK"], second["LOCAL_RANK"]) == ("0", "1")
    assert (first["CUDA_VISIBLE_DEVICES"], second["CUDA_VISIBLE_DEVICES"]) == ("0,1,2,3", "4,5,6,7")
    assert (first["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"], second["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"]) == ("0", "4")
    owned = _owned_devices(dispatched)
    assert not owned["10.0.0.10:8100"] & owned["10.0.0.10:8101"]
    for environment in dispatched.values():
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["NVIDIA_VISIBLE_DEVICES"] == environment["CUDA_VISIBLE_DEVICES"]
        assert environment["MASTER_ADDR"] == "10.0.0.10"
        assert environment["MASTER_PORT"] != "1"
        assert environment["WORLD_SIZE"] == "3"
        assert environment["LOCAL_WORLD_SIZE"] != "64"
        assert "HIP_VISIBLE_DEVICES" not in environment
        assert "NCCL_COMM_ID" not in environment
        assert environment["NCCL_DEBUG"] == "INFO", "an operator's NCCL tuning is not a topology claim"
        assert environment["USER_FLAG"] == "kept"


def test_a_cell_whose_device_count_is_unknown_still_declines_to_coordinate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouping does not make an unknown host guessable.

    ``a2-ultragpu-8g`` is a real machine with no catalog row, so nothing here
    knows what it carries -- and a gang holding one is dispatched the caller's
    own values, exactly as it is today, rather than a share of a number nobody
    read.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", UNCATALOGUED_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, {"CUDA_VISIBLE_DEVICES": "3"}, layout=GpuRankLayout.RANK_PROCESS)

    assert set(dispatched) == {"10.0.0.10:8100", "10.0.0.10:8101", "10.0.0.20:8100"}
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert "LOCAL_RANK" not in environment
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_a_host_whose_ranks_cannot_divide_it_is_declined_rather_than_dispatched(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three native workers on one four-card host is not a placement.

    The resolver downgrades rather than failing the dispatch, for the reason it
    downgrades a half-known gang: its inputs are observations of a fleet, and a
    workload that runs today must not start failing because its packing became
    expressible-looking.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", SMALL_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101", "http://10.0.0.10:8102")),
        ("cell-b", SMALL_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, {"CUDA_VISIBLE_DEVICES": "3"}, layout=GpuRankLayout.RANK_PROCESS)

    assert len(dispatched) == 4
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_the_shipped_layout_still_declines_two_workers_on_one_host(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinating a shared host is opt-in, so an upgrade alone changes nothing.

    This is the same gang the previous wave declined, dispatched by an agent
    that was upgraded and not reconfigured. It must still be declined: the
    caller's own device visibility survives, which is the behaviour a fleet has
    today.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, {"CUDA_VISIBLE_DEVICES": "3"})

    assert len(dispatched) == 3
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_a_gpu_gang_dispatched_on_shared_hosts_still_carries_no_hardware_claim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Everything dispatched is a placement, and none of it may read as a run.

    No GPU has executed an Eyvan workload, so nothing reports an observed
    device, a driver version, or a successful initialization -- and this file
    must not become readable as evidence that one has.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", SMALL_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, layout=GpuRankLayout.RANK_PROCESS)

    for environment in dispatched.values():
        assert {name for name in environment if name.startswith("EYVAN_GPU_")} == {
            "EYVAN_GPU_MODEL",
            "EYVAN_GPU_HOST_COUNT",
            "EYVAN_GPU_HOST_INDEX",
            "EYVAN_GPU_LOCAL_DEVICE_COUNT",
            "EYVAN_GPU_GLOBAL_DEVICE_COUNT",
            "EYVAN_GPU_GLOBAL_DEVICE_OFFSET",
        }


def test_two_gangs_on_one_cell_are_still_given_the_same_devices(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap this work does **not** close, stated rather than left to be found.

    Per-host grouping divides the devices of a host between the ranks of *one*
    gang, which Eyvan places together. Two independently scheduled workloads are
    two gangs, nothing shared decides between them, and both are told they own
    the same cards. Isolating them needs a device range recorded against the
    lease -- ``LeaseRecord`` has no such field, and the allocator that would fill
    it would live in the control plane; neither is touched here.

    What keeps that from being a live defect today is elsewhere and is not a
    device range: ``to_core_cell`` marks every ``eyvan-native`` cell
    ``exclusive_workloads``, so the scheduler refuses to place a second workload
    on a cell that already holds one. The isolation gap opens the moment that
    flag does, which is exactly why this dispatch-level fact is pinned rather
    than assumed away.

    This test fails the day that changes, which is the point: the claim it pins
    is a limitation, and a limitation that silently disappears is a claim nobody
    reviewed.
    """

    driver = _driver(tmp_path)
    cells = _cells(
        driver,
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100",)),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )

    first = _dispatch(monkeypatch, cells, layout=GpuRankLayout.RANK_PROCESS, submission_id="job-one")
    second = _dispatch(monkeypatch, cells, layout=GpuRankLayout.RANK_PROCESS, submission_id="job-two")

    assert first["10.0.0.10:8100"]["CUDA_VISIBLE_DEVICES"] == second["10.0.0.10:8100"]["CUDA_VISIBLE_DEVICES"]
    assert _owned_devices(first) == _owned_devices(second)


# ------------------------------------------------------- who chooses the layout


def test_an_agent_that_was_not_reconfigured_uses_the_layout_that_ships() -> None:
    """Absence is the historical behaviour, not an unset value to be filled in."""

    assert gpu_rank_layout_from_environment({}) is GpuRankLayout.HOST_LAUNCHER
    assert gpu_rank_layout_from_environment({GPU_RANK_LAYOUT_ENV: "  "}) is GpuRankLayout.HOST_LAUNCHER


def test_an_operator_can_choose_the_process_layout_on_the_agent() -> None:
    """The switch has to be reachable, or the capability ships turned off forever."""

    assert gpu_rank_layout_from_environment({GPU_RANK_LAYOUT_ENV: "rank-process"}) is GpuRankLayout.RANK_PROCESS


def test_a_misspelled_layout_is_refused_rather_than_read_as_the_default() -> None:
    """A typo that silently reads as "off" is a decision nobody made and nobody sees."""

    with pytest.raises(ValueError, match="must be one of: host-launcher, rank-process"):
        gpu_rank_layout_from_environment({GPU_RANK_LAYOUT_ENV: "rank_process"})


def test_the_resolver_reads_the_layout_from_the_agent_environment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end from the variable an operator sets to the bytes a rank reads."""

    monkeypatch.setenv(GPU_RANK_LAYOUT_ENV, "rank-process")
    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells)

    assert dispatched["10.0.0.10:8100"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert dispatched["10.0.0.10:8101"]["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"


# ------------------------------------------------- a hand-built gang is refused


def _member(cell_id: str, *endpoints: str, devices: int | None = 8) -> NativeGangMember:
    return NativeGangMember(
        cell_id=cell_id,
        endpoints=endpoints,
        secret=f"worker-secret-{cell_id}",
        context=_context(f"submit-{cell_id}"),
        gpu_devices_per_host=devices,
        gpu_model="nvidia-h100-80gb" if devices is not None else None,
    )


def test_a_hand_built_gpu_gang_cannot_double_book_a_host() -> None:
    """The refusal the resolver used to be the only holder of.

    ``resolve_gang`` declined this shape, so nothing shipped a double-booked
    assignment -- but the executor would happily have built one, and an executor
    that can be constructed wrong is one somebody eventually constructs wrong.
    """

    with pytest.raises(ValueError, match=r"place several ranks on host\(s\) 10\.0\.0\.10"):
        NativeGangExecutor(
            (
                _member("cell-a", "http://10.0.0.10:8100", "http://10.0.0.10:8101"),
                _member("cell-b", "http://10.0.0.20:8100"),
            )
        )


def test_a_gang_that_is_not_gpu_coordinated_may_still_put_two_workers_on_one_host() -> None:
    """The refusal is about a device plan, and a gang without one keeps its behaviour.

    A TPU pair, or any gang whose device counts are unknown, is dispatched
    exactly as before -- including on a machine that runs two native workers.
    """

    executor = NativeGangExecutor(
        (
            _member("cell-a", "http://10.0.0.10:8100", "http://10.0.0.10:8101", devices=None),
            _member("cell-b", "http://10.0.0.20:8100", devices=None),
        )
    )
    try:
        assert executor.gpu_plan is None
        assert executor.endpoints == ("http://10.0.0.10:8100", "http://10.0.0.10:8101", "http://10.0.0.20:8100")
    finally:
        executor.close()


def test_a_process_layout_gang_whose_endpoint_names_no_host_is_refused() -> None:
    """Unknown is not "alone on a machine".

    An endpoint nothing can parse a host out of cannot be grouped, and treating
    it as its own machine is precisely the inference that would hand two ranks
    the same cards.
    """

    with pytest.raises(ValueError, match="every worker endpoint must name one"):
        NativeGangExecutor(
            (
                _member("cell-a", "http://10.0.0.10:8100"),
                _member("cell-b", "10.0.0.20:8100"),
            ),
            gpu_rank_layout=GpuRankLayout.RANK_PROCESS,
        )


def test_a_hand_built_gang_whose_host_cannot_be_divided_is_refused() -> None:
    """The author of a gang stated something impossible, so it is not dispatched.

    The resolver downgrades the same shape instead, because its inputs are
    observations of a fleet rather than a statement; both refuse to invent a
    split, and only one of them can afford to fail the dispatch.
    """

    with pytest.raises(ValueError, match="cannot divide evenly"):
        NativeGangExecutor(
            (
                _member("cell-a", "http://10.0.0.10:8100", "http://10.0.0.10:8101", "http://10.0.0.10:8102"),
                _member("cell-b", "http://10.0.0.20:8100"),
            ),
            gpu_rank_layout=GpuRankLayout.RANK_PROCESS,
        )


def test_a_gang_layout_that_is_not_a_layout_is_refused() -> None:
    """A string that looks like a layout is not one, and would read as the default."""

    with pytest.raises(TypeError, match="must be a GpuRankLayout"):
        NativeGangExecutor(
            (_member("cell-a", "http://10.0.0.10:8100"), _member("cell-b", "http://10.0.0.20:8100")),
            gpu_rank_layout="rank-process",  # type: ignore[arg-type]
        )
