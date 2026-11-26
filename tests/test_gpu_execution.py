# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GPU cell profiles, per-rank device assignment, and who owns that assignment.

GPUs were modelled in the catalog and were placeable, and nothing had ever run
on one. This file covers the part of that gap which is host-testable: a GPU
shape has to state its host before it can be stored, a rank has to be given an
exact device set nobody else holds, and a submission has to be unable to say
which devices a rank gets.

**No claim here is hardware-qualified.** No GPU has executed an Eyvan workload.
Every assertion below is about values Eyvan computes and the bytes it dispatches,
which is exactly as far as a host can check.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest
import yaml

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.drivers.base import MutationContext
from eyvan.capacity.hardware_catalog import (
    CATALOG_PATH,
    AcceleratorShape,
    HardwareCatalogError,
    hardware_catalog,
    load_hardware_catalog,
)
from eyvan.worker.distributed_environment import (
    GPU_TOPOLOGY_ENVIRONMENT_NAMES,
    plan_gpu_devices,
    without_gpu_topology_overrides,
)
from eyvan.worker.runtime.native_gang import NativeGangExecutor, NativeGangMember

#: Provider-published GPU host shapes, written as independent literals rather
#: than read back out of the file under test. Google's accelerator-optimized
#: machine families page establishes vCPU, instance memory and attached GPU
#: count; the GPU machine types page establishes the accelerator names.
EXPECTED_GPU = {
    "a3-highgpu-8g": {"devices": 8, "cpu": 208, "memory_gib": 1872, "model": "nvidia-h100-80gb"},
    "a2-highgpu-1g": {"devices": 1, "cpu": 12, "memory_gib": 85, "model": "nvidia-tesla-a100"},
    "g2-standard-48": {"devices": 4, "cpu": 48, "memory_gib": 192, "model": "nvidia-l4"},
}


def _catalog_document() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))


def _write(tmp_path, document: dict):
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _gpu_entry(**overrides) -> dict:
    entry = {
        "accelerator_type": "a3-highgpu-8g",
        "kind": "gpu",
        "generation": "a3-high",
        "chips": 8,
        "quota_units": 8,
        "hosts": 1,
        "chips_per_host": 8,
        "host_cpu": 208,
        "host_memory_gib": 1872,
        "capacity_backend": "gcp-compute",
        "multislice_capable": False,
        "gpu_model": "nvidia-h100-80gb",
        "machine_type": "a3-highgpu-8g",
        "source_url": "https://cloud.google.com/compute/docs/accelerator-optimized-machines",
        "source_version": "Last updated 2026-08-07 UTC",
    }
    entry.update(overrides)
    return entry


# ------------------------------------------------------------------ the profile


@pytest.mark.parametrize("accelerator_type", sorted(EXPECTED_GPU))
def test_the_catalog_states_the_gpu_host_the_fleet_would_actually_get(accelerator_type: str) -> None:
    """Literals, not a comparison between two things that could both be wrong."""

    expected = EXPECTED_GPU[accelerator_type]
    shape = hardware_catalog().require(accelerator_type)

    assert shape.kind == "gpu"
    assert shape.schedulable_gpu == expected["devices"]
    assert shape.schedulable_tpu == 0, "a GPU host offers nothing in the TPU dimension"
    assert shape.devices_per_host == expected["devices"]
    assert shape.host_cpu == expected["cpu"]
    assert shape.host_memory_gib == expected["memory_gib"]
    assert shape.gpu_model == expected["model"]
    assert shape.machine_type == accelerator_type
    assert shape.source_url == "https://cloud.google.com/compute/docs/accelerator-optimized-machines"


def test_a_memory_less_gpu_shape_is_refused_rather_than_defaulted_to_zero(tmp_path) -> None:
    """The rule TPU shapes already obey, applied where it had never been tested.

    A zero-memory GPU pool is worse than no pool: it loads, it is schedulable,
    and every workload that asks for a gigabyte is refused by a cell that looks
    healthy. So the refusal has to happen when the shape is built, not when a
    workload fails to fit.
    """

    document = _catalog_document()
    entry = _gpu_entry()
    entry.pop("host_memory_gib")
    document["shapes"].append(entry)

    with pytest.raises(HardwareCatalogError, match="host_memory_gib"):
        load_hardware_catalog(_write(tmp_path, document))


@pytest.mark.parametrize("value", [0, -1, float("inf")])
def test_a_gpu_shape_cannot_state_an_unusable_memory_figure(value: float) -> None:
    """Present-but-meaningless is the same defect as absent, and refused the same way."""

    with pytest.raises(HardwareCatalogError, match="host_memory_gib must be a positive finite number"):
        AcceleratorShape(**_gpu_entry(host_memory_gib=value))


def test_a_cpu_less_gpu_shape_is_refused(tmp_path) -> None:
    """Both admission dimensions, not just the one that happened to be tested."""

    document = _catalog_document()
    entry = _gpu_entry()
    entry.pop("host_cpu")
    document["shapes"].append(entry)

    with pytest.raises(HardwareCatalogError, match="host_cpu"):
        load_hardware_catalog(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hosts": None, "chips_per_host": None}, "must record hosts and chips_per_host"),
        ({"hosts": 2, "chips_per_host": 4}, "describes one VM host, not 2"),
        ({"quota_units": 4}, "one quota unit per device"),
        ({"multislice_capable": True}, "multislice is a Cloud TPU concept"),
        ({"capacity_backend": "gcp-tpu"}, "provisioned as a VM"),
        ({"gpu_model": None}, "must name a non-empty trimmed gpu_model"),
        ({"machine_type": None}, "must name a non-empty trimmed machine_type"),
        ({"tpu_type": "ct5p"}, "cannot name a tpu_type"),
    ],
)
def test_a_gpu_shape_that_does_not_describe_one_vm_host_is_refused(overrides: dict, message: str) -> None:
    """Each of these values is read later by something that cannot ask again.

    ``chips_per_host`` becomes a rank's device list, ``machine_type`` is what a
    provider is asked to create, and ``gpu_model`` is the only surviving record
    of which accelerator a pool holds once the fleet document is compiled away.
    """

    with pytest.raises(HardwareCatalogError, match=message):
        AcceleratorShape(**_gpu_entry(**overrides))


def test_a_shape_of_an_unknown_kind_is_refused() -> None:
    """``kind`` selects the dimension a shape is counted in; an unknown one counts as nothing."""

    with pytest.raises(HardwareCatalogError, match="is not one of gpu, tpu"):
        AcceleratorShape(**_gpu_entry(kind="fpga"))


def test_a_tpu_shape_cannot_claim_a_gpu_model() -> None:
    """One row describes one kind of host, or the resource vector it fills is a guess."""

    tpu_entry = {
        "accelerator_type": "v5p-16",
        "kind": "tpu",
        "generation": "v5p",
        "chips": 8,
        "quota_units": 16,
        "host_cpu": 208,
        "host_memory_gib": 448,
        "capacity_backend": "gcp-tpu",
        "multislice_capable": True,
        "source_url": "https://cloud.google.com/tpu/docs/v5p",
        "source_version": "Last updated 2026-08-04 UTC",
        "gpu_model": "nvidia-l4",
    }

    with pytest.raises(HardwareCatalogError, match="cannot name a gpu_model"):
        AcceleratorShape(**tpu_entry)


def test_adding_gpu_rows_did_not_reinterpret_any_name_a_pool_already_uses() -> None:
    """The rolling-upgrade property: no live pool's accelerator_type changed meaning.

    GPU rows are keyed by machine type precisely so that the GPU model names
    deployments already write into ``accelerator_type`` stay uncatalogued, and
    therefore keep going down the documented complete-manual-vector path they
    go down today.
    """

    from eyvan.capacity.hardware_catalog import shape_for

    for model in ("nvidia-l4", "nvidia-tesla-a100", "nvidia-h100-80gb", "gpu"):
        assert shape_for(model) is None, model


def test_a_gpu_pool_write_is_held_to_the_catalog_device_count() -> None:
    """A cataloged shape fixes device count at the write boundary, as it does for TPU."""

    from pydantic import ValidationError

    from eyvan.server.api.schemas import CapacityPoolUpsertRequest

    payload = {
        "id": "research-a3",
        "fleet_name": "research",
        "node_type": "a3-highgpu-8g",
        "provider": "gcp",
        "capacity_backend": "gcp-compute",
        "provider_project": "gcp-project",
        "region": "us-central1",
        "zone": "us-central1-a",
        "accelerator_type": "a3-highgpu-8g",
        "agent_endpoint": "http://agent:8090",
        "capacity": {"cpu": 208, "memory_gib": 1872, "gpu": 8},
        "allocatable": {"cpu": 200, "memory_gib": 1800, "gpu": 8},
    }

    request = CapacityPoolUpsertRequest.model_validate(payload)
    assert request.capacity.gpu == request.allocatable.gpu == 8

    payload["capacity"] = {"cpu": 208, "memory_gib": 1872, "gpu": 4}
    with pytest.raises(ValidationError, match=r"capacity\.gpu.*catalog count 8"):
        CapacityPoolUpsertRequest.model_validate(payload)


# --------------------------------------------------------------- the assignment


def test_every_rank_gets_an_exact_non_overlapping_slice_of_the_gang_devices() -> None:
    """The property a collective depends on: one device, one owner.

    Two ranks on different hosts both see local index ``0``, so local indices
    alone cannot answer whether that is one device or two. The gang-wide
    ordinals can, and they are what is checked.
    """

    plan = plan_gpu_devices([8, 1, 4])

    assert [assignment.host_devices for assignment in plan.assignments] == [
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0,),
        (0, 1, 2, 3),
    ]
    globals_seen = [index for assignment in plan.assignments for index in assignment.global_devices]
    assert globals_seen == list(range(13)), "the gang's devices are covered once each, in rank order"
    assert len(globals_seen) == len(set(globals_seen)) == plan.device_count == 13
    assert [assignment.global_device_offset for assignment in plan.assignments] == [0, 8, 9]
    assert plan.host_count == 3


@pytest.mark.parametrize("devices_per_rank", [[], [0], [8, 0], [8, -1], [8, None], [8, True], [8, 2.0]])
def test_a_host_whose_device_count_is_not_a_positive_integer_is_refused(devices_per_rank) -> None:
    """An unknown device count is unknown; it is not evidence of an empty host.

    Zero would be delivered as an empty ``CUDA_VISIBLE_DEVICES``, which is how a
    rank ends up on the CPU while every peer waits for it at a collective.
    """

    with pytest.raises(ValueError):
        plan_gpu_devices(devices_per_rank)


def test_a_single_gpu_host_and_a_multi_gpu_host_differ_only_in_their_devices() -> None:
    """The two shapes must not diverge anywhere else, or one of them is a special case."""

    plan = plan_gpu_devices([1, 8])
    single = plan.rank_environment(0, rendezvous_host="10.0.0.1", gpu_model="nvidia-tesla-a100")
    multi = plan.rank_environment(1, rendezvous_host="10.0.0.1", gpu_model="nvidia-h100-80gb")

    assert single.keys() == multi.keys()
    assert single["CUDA_VISIBLE_DEVICES"] == "0"
    assert multi["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert (single["LOCAL_WORLD_SIZE"], multi["LOCAL_WORLD_SIZE"]) == ("1", "8")
    assert (single["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"], multi["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"]) == ("0", "1")
    differing = {name for name in single if single[name] != multi[name]}
    assert differing == {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LOCAL_WORLD_SIZE",
        "NODE_RANK",
        "EYVAN_GPU_HOST_INDEX",
        "EYVAN_GPU_LOCAL_DEVICE_COUNT",
        "EYVAN_GPU_GLOBAL_DEVICE_OFFSET",
        "EYVAN_GPU_MODEL",
    }, "a one-GPU host must not be a different kind of rank"
    for shared in ("WORLD_SIZE", "EYVAN_GPU_GLOBAL_DEVICE_COUNT", "EYVAN_GPU_HOST_COUNT"):
        assert single[shared] == multi[shared] == ("9" if shared != "EYVAN_GPU_HOST_COUNT" else "2")
    assert single["MASTER_ADDR"] == multi["MASTER_ADDR"] == "10.0.0.1"


def test_the_rendezvous_port_is_not_the_jax_coordinator_port() -> None:
    """A workload may initialize JAX and a process group; two services, two ports."""

    from eyvan.worker.runtime import native_gang

    environment = plan_gpu_devices([1, 1]).rank_environment(0, rendezvous_host="10.0.0.1")

    assert environment["MASTER_PORT"] != str(native_gang._COORDINATOR_PORT)


def test_a_rank_is_never_told_which_process_it_is_inside_its_own_host() -> None:
    """``RANK`` and ``LOCAL_RANK`` name a process in a launcher Eyvan does not run.

    Filling them with the host index would be right for the first local process
    and wrong for every other one, which is worse than leaving a launcher to set
    what it already sets.
    """

    environment = plan_gpu_devices([8, 8]).rank_environment(1, rendezvous_host="10.0.0.1")

    assert "RANK" not in environment
    assert "LOCAL_RANK" not in environment
    assert environment["NODE_RANK"] == "1"


def test_a_rendezvous_host_that_is_not_a_host_is_refused() -> None:
    """An empty MASTER_ADDR would be a rendezvous every rank silently fails to find."""

    with pytest.raises(ValueError, match="rendezvous host"):
        plan_gpu_devices([1, 1]).rank_environment(0, rendezvous_host=" ")


def test_every_standard_gpu_name_eyvan_writes_is_a_name_it_also_strips() -> None:
    """A variable written but not stripped is one a submission could pre-seed.

    The ``EYVAN_`` names are excluded because they are not stripped by name --
    they are Eyvan's namespace, unconditionally rewritten on every dispatch --
    and the executor test below asserts that rewrite on the wire. Every other
    name Eyvan writes belongs to a runtime, so it has to be in the stripped set
    or a caller's value could survive to a rank.
    """

    written = set(plan_gpu_devices([2, 2]).rank_environment(0, rendezvous_host="10.0.0.1", gpu_model="nvidia-l4"))
    eyvan_namespace = {name for name in written if name.startswith("EYVAN_")}

    assert eyvan_namespace, "the Eyvan-namespace facts a rank reads back are missing"
    assert written - eyvan_namespace <= GPU_TOPOLOGY_ENVIRONMENT_NAMES
    assert without_gpu_topology_overrides({name: "spoofed" for name in written}) == {
        name: "spoofed" for name in eyvan_namespace
    }


# ------------------------------------------------------------------ the gang


def _context(*, key: str) -> MutationContext:
    return MutationContext(idempotency_key=key, generation=7, fencing_token=f"fence-{key}")


def _client(endpoint: str, handler: tp.Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(base_url=endpoint, transport=httpx.MockTransport(handler))


def _json_object(value: JsonValue) -> JsonObject:
    assert is_json_object(value)
    return value


def _gpu_members(
    *,
    left_devices: int | None = 8,
    right_devices: int | None = 1,
) -> tuple[NativeGangMember, ...]:
    return (
        NativeGangMember(
            cell_id="cell-a",
            endpoints=("http://10.12.0.10:8100",),
            secret="worker-secret-a",
            context=_context(key="submit-a"),
            gpu_devices_per_host=left_devices,
            gpu_model="nvidia-h100-80gb" if left_devices is not None else None,
        ),
        NativeGangMember(
            cell_id="cell-b",
            endpoints=("http://10.12.0.20:8100",),
            secret="worker-secret-b",
            context=_context(key="submit-b"),
            gpu_devices_per_host=right_devices,
            gpu_model="nvidia-tesla-a100" if right_devices is not None else None,
        ),
    )


def _submit(members: tp.Sequence[NativeGangMember], environment: dict[str, str]) -> list[JsonObject]:
    dispatched: list[JsonObject] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert is_json_object(payload)
        dispatched.append(_json_object(_json_object(payload["spec"])["environment"]))
        return httpx.Response(200, json={"remote_id": "gpu-job"})

    endpoints = [endpoint for member in sorted(members, key=lambda item: item.cell_id) for endpoint in member.endpoints]
    clients = [_client(endpoint, handler) for endpoint in endpoints]
    try:
        executor = NativeGangExecutor(members, clients=clients)
        assert (
            executor.submit(
                {
                    "entrypoint": "python train.py",
                    "submission_id": "gpu-job",
                    "environment": dict(environment),
                }
            )
            == "gpu-job"
        )
    finally:
        for client in clients:
            client.close()
    return dispatched


def test_a_gpu_gang_dispatches_a_disjoint_device_set_to_every_rank() -> None:
    """The assignment has to reach the wire, not merely exist in a plan object."""

    dispatched = _submit(_gpu_members(), {"USER_FLAG": "kept"})

    assert [environment["CUDA_VISIBLE_DEVICES"] for environment in dispatched] == ["0,1,2,3,4,5,6,7", "0"]
    assert [environment["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"] for environment in dispatched] == ["0", "8"]
    assert [environment["EYVAN_GPU_MODEL"] for environment in dispatched] == [
        "nvidia-h100-80gb",
        "nvidia-tesla-a100",
    ]
    assert {environment["EYVAN_GPU_GLOBAL_DEVICE_COUNT"] for environment in dispatched} == {"9"}
    assert {environment["MASTER_ADDR"] for environment in dispatched} == {"10.12.0.10"}
    assert all(environment["USER_FLAG"] == "kept" for environment in dispatched)


def test_a_submission_cannot_spoof_its_device_visibility_or_its_rank() -> None:
    """The one property that has to hold whatever the caller sends.

    A submission that pinned every rank to device zero would put a gang's whole
    run on one GPU per host; one that pinned every rank's ``NODE_RANK`` to zero
    would make the ranks disagree about who they are while the fleet reported a
    healthy gang.
    """

    spoofed = {
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
        "LOCAL_WORLD_SIZE": "64",
        "NODE_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "attacker.invalid",
        "MASTER_PORT": "1",
        "NCCL_COMM_ID": "attacker.invalid:1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "NCCL_DEBUG": "INFO",
        # Not stripped by name -- the whole ``EYVAN_`` namespace is rewritten on
        # every dispatch instead -- so a caller's value must lose to Eyvan's.
        "EYVAN_GPU_LOCAL_DEVICE_COUNT": "64",
        "EYVAN_GPU_GLOBAL_DEVICE_OFFSET": "0",
    }

    dispatched = _submit(_gpu_members(), spoofed)

    for rank, environment in enumerate(dispatched):
        assert environment["CUDA_VISIBLE_DEVICES"] == ("0,1,2,3,4,5,6,7" if rank == 0 else "0")
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["NODE_RANK"] == str(rank)
        assert environment["MASTER_ADDR"] == "10.12.0.10"
        assert environment["MASTER_PORT"] != "1"
        assert environment["WORLD_SIZE"] == "9"
        assert environment["LOCAL_WORLD_SIZE"] == ("8" if rank == 0 else "1")
        assert "NCCL_COMM_ID" not in environment
        assert "RANK" not in environment
        assert "LOCAL_RANK" not in environment
        assert "HIP_VISIBLE_DEVICES" not in environment
        assert environment["NCCL_DEBUG"] == "INFO", "an operator's NCCL tuning is not a topology claim"
        # The rank identity the JAX path already owned must still be Eyvan's.
        assert environment["JAX_PROCESS_ID"] == str(rank)
        assert environment["EYVAN_DISTRIBUTED_PROCESS_ID"] == str(rank)
        assert environment["EYVAN_GPU_LOCAL_DEVICE_COUNT"] == ("8" if rank == 0 else "1")
        assert environment["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"] == ("0" if rank == 0 else "8")


def test_a_gang_that_declares_no_gpu_devices_dispatches_exactly_what_it_dispatches_today() -> None:
    """The shipped path must not change: no GPU variable written, none stripped."""

    members = _gpu_members(left_devices=None, right_devices=None)
    spoofed = {"CUDA_VISIBLE_DEVICES": "3", "MASTER_ADDR": "caller.example", "USER_FLAG": "kept"}

    dispatched = _submit(members, spoofed)

    for environment in dispatched:
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert environment["MASTER_ADDR"] == "caller.example"
        assert environment["USER_FLAG"] == "kept"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_a_gang_where_only_some_members_declare_devices_is_refused() -> None:
    """Half a plan would pin the declared ranks and leave the rest to the submission.

    That gang would look coordinated and would not be, which is the failure mode
    worth refusing outright rather than reporting later.
    """

    with pytest.raises(ValueError, match="every member to declare its device count"):
        NativeGangExecutor(_gpu_members(right_devices=None))


@pytest.mark.parametrize("devices", [0, -1, True])
def test_a_member_cannot_declare_a_device_count_that_is_not_a_count(devices) -> None:
    """The refusal belongs on the member, before a plan is ever built."""

    with pytest.raises(ValueError, match="positive integer or absent"):
        NativeGangMember(
            cell_id="cell-a",
            endpoints=("http://10.12.0.10:8100",),
            secret="worker-secret-a",
            context=_context(key="submit-a"),
            gpu_devices_per_host=devices,
        )


def test_a_member_naming_a_gpu_model_must_say_how_many_of_them_there_are() -> None:
    """A model with no count is a cell that looks GPU-coordinated and is not."""

    with pytest.raises(ValueError, match="must also state its device count"):
        NativeGangMember(
            cell_id="cell-a",
            endpoints=("http://10.12.0.10:8100",),
            secret="worker-secret-a",
            context=_context(key="submit-a"),
            gpu_model="nvidia-l4",
        )


def test_a_catalog_shape_is_enough_to_build_a_gpu_member() -> None:
    """The catalog and the gang have to join, or the device count is retyped by hand."""

    shape = hardware_catalog().require("g2-standard-48")
    member = NativeGangMember(
        cell_id="cell-a",
        endpoints=("http://10.12.0.10:8100",),
        secret="worker-secret-a",
        context=_context(key="submit-a"),
        gpu_devices_per_host=shape.devices_per_host,
        gpu_model=shape.gpu_model,
    )

    assert member.gpu_devices_per_host == 4
    assert member.gpu_model == "nvidia-l4"


def test_every_host_of_a_multi_host_gpu_member_gets_its_own_devices() -> None:
    """A member's device count is per host, so a two-host member contributes twice."""

    members = (
        NativeGangMember(
            cell_id="cell-a",
            endpoints=("http://10.12.0.10:8100", "http://10.12.0.11:8100"),
            secret="worker-secret-a",
            context=_context(key="submit-a"),
            gpu_devices_per_host=4,
        ),
        NativeGangMember(
            cell_id="cell-b",
            endpoints=("http://10.12.0.20:8100",),
            secret="worker-secret-b",
            context=_context(key="submit-b"),
            gpu_devices_per_host=2,
        ),
    )

    dispatched = _submit(members, {})

    assert [environment["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"] for environment in dispatched] == ["0", "4", "8"]
    assert {environment["EYVAN_GPU_GLOBAL_DEVICE_COUNT"] for environment in dispatched} == {"10"}
    assert [environment["EYVAN_GPU_HOST_COUNT"] for environment in dispatched] == ["3", "3", "3"]
