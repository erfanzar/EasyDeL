# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The chain from one observed cell to the bytes a GPU rank is dispatched.

W7 built ``plan_gpu_devices`` and proved a hand-built GPU gang dispatches exact,
non-overlapping devices. Nothing built such a gang: ``resolve_gang`` never set
``gpu_devices_per_host``, because the cell's accelerator was not in
``CellObservation.metadata``. This file covers the join -- a real capacity driver
publishes what the hardware catalog says its hosts carry, and the resolver turns
that into a coordinated gang or, when it cannot, into exactly the dispatch the
fleet does today.

**No claim here is hardware-qualified.** No GPU has executed an Eyvan workload.
Every assertion is about metadata Eyvan derives and HTTP payloads it emits, which
is as far as a host without a GPU can honestly check.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.drivers.base import (
    CELL_ACCELERATOR_METADATA_KEY,
    CapacityStatus,
    MutationContext,
    describe_cell_accelerator,
    read_cell_accelerator,
)
from eyvan.capacity.drivers.gcp import GcpComputeCapacityDriver, GcpTpuCapacityDriver
from eyvan.capacity.drivers.local import LocalCapacityDriver
from eyvan.capacity.drivers.manual import ManualCapacityDriver
from eyvan.capacity.fleet_state import FleetStateStore, LocalBackend
from eyvan.capacity.hardware_catalog import hardware_catalog
from eyvan.capacity.providers.gcp import ProviderObservation
from eyvan.worker.runtime import native_remote
from eyvan.worker.runtime.registry import ExecutionDriverResolver, NativeGangCell

#: An eight-device GPU host and a four-device GPU host, so the heterogeneous case
#: is covered by shipped catalog data rather than by a fixture.
BIG_GPU = "a3-highgpu-8g"
SMALL_GPU = "g2-standard-48"


def _context(key: str, *, generation: int = 1, token: str = "fence-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


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


def _resolver() -> ExecutionDriverResolver:
    return ExecutionDriverResolver(backend="eyvan-native", native_worker_token="root-secret")


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
) -> dict[str, JsonObject]:
    """Submit through the resolved gang and return each host's ranked environment.

    Keyed by endpoint rather than collected in order: the mapping from a rank to
    the machine it lands on is the thing under test, so reading it back
    positionally would assume the answer.
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
        return httpx.Response(200, json={"remote_id": "gpu-job"})

    real_client = httpx.Client
    opened: list[httpx.Client] = []

    def client_factory(*_args: object, **kwargs: object) -> httpx.Client:
        base_url = kwargs.get("base_url")
        assert isinstance(base_url, str)
        client = real_client(base_url=base_url, transport=httpx.MockTransport(handler))
        opened.append(client)
        return client

    monkeypatch.setattr(native_remote.httpx, "Client", client_factory)
    executor = _resolver().resolve_gang(tuple(cells))
    try:
        remote_id = executor.submit(
            {
                "entrypoint": "python train.py",
                "submission_id": "gpu-job",
                "environment": dict(environment or {}),
            }
        )
        assert remote_id == "gpu-job"
    finally:
        for client in opened:
            client.close()
    return dispatched


# ------------------------------------------------------- what a driver reports


def test_a_gpu_cell_reports_the_catalog_row_its_hosts_actually_are(tmp_path) -> None:
    """The prerequisite: the accelerator has to reach the observation at all."""

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(SMALL_GPU, "http://10.0.0.20:8100"), _context("apply"))

    metadata = driver.observe("cell-a").metadata[CELL_ACCELERATOR_METADATA_KEY]

    assert metadata == {
        "kind": "gpu",
        "accelerator_type": SMALL_GPU,
        "catalog_revision": hardware_catalog().revision,
        "devices_per_host": 4,
        "gpu_model": "nvidia-l4",
    }


def test_the_device_count_is_the_catalog_row_and_not_the_digits_in_the_name(tmp_path) -> None:
    """``g2-standard-48`` carries four L4s, not forty-eight of anything.

    The one number that becomes a rank's device list must come from a reviewed
    row. Parsing it out of the type name is the failure this asserts against.
    """

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(SMALL_GPU, "http://10.0.0.20:8100"), _context("apply"))

    accelerator = read_cell_accelerator(driver.observe("cell-a").metadata)

    assert accelerator is not None
    assert accelerator.gpu_devices_per_host == hardware_catalog().require(SMALL_GPU).devices_per_host == 4
    assert "48" not in str(accelerator.gpu_devices_per_host)


def test_a_cell_whose_type_has_no_catalog_row_reports_no_accelerator_at_all(tmp_path) -> None:
    """Silence, not a guess. ``a2-ultragpu-8g`` is a real machine with no row here."""

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired("a2-ultragpu-8g", "http://10.0.0.20:8100"), _context("apply"))

    metadata = driver.observe("cell-a").metadata

    assert CELL_ACCELERATOR_METADATA_KEY not in metadata
    assert read_cell_accelerator(metadata) is None


def test_a_tpu_cell_reports_its_kind_and_offers_the_gpu_path_nothing(tmp_path) -> None:
    """Known-not-GPU and unknown must give the same answer where it matters."""

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired("v5p-16", "http://10.0.0.30:8100"), _context("apply"))

    accelerator = read_cell_accelerator(driver.observe("cell-a").metadata)

    assert accelerator is not None
    assert accelerator.kind == "tpu"
    assert accelerator.devices_per_host == 4
    assert accelerator.gpu_devices_per_host is None
    assert accelerator.gpu_model is None


def test_the_development_driver_declines_to_describe_the_operators_machine(tmp_path) -> None:
    """A dev cell named after an eight-H100 host is not eight H100s.

    The local driver makes no isolation claim, so its accelerator stays unknown
    and its cells stay on the path they are on today.
    """

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = LocalCapacityDriver(state)
    driver.apply_capacity(
        "cell-a",
        {
            "provider": "local",
            "capacity_backend": "local",
            "executor": "native-local",
            "accelerator_type": BIG_GPU,
            "capacity_class": "on_demand",
            "region": "local",
        },
        _context("apply"),
    )

    metadata = driver.observe("cell-a").metadata

    assert CELL_ACCELERATOR_METADATA_KEY not in metadata
    assert read_cell_accelerator(metadata) is None


def test_a_persisted_accelerator_this_build_cannot_state_is_not_left_standing(tmp_path) -> None:
    """A downgrade must not keep dispatching devices the running build cannot confirm.

    Nothing persists this key today, and the point is that nothing has to: the
    observation derives it either way, so a copy written by some other build --
    or by a hand-edited fleet document -- cannot outlive this build's ability to
    describe the cell.
    """

    state = FleetStateStore(LocalBackend(tmp_path / "fleet.json"))
    driver = ManualCapacityDriver(state)
    driver.apply_capacity("cell-a", _desired("a2-ultragpu-8g", "http://10.0.0.20:8100"), _context("apply"))

    def forge(record: object) -> None:
        assert hasattr(record, "metadata")
        record.metadata[CELL_ACCELERATOR_METADATA_KEY] = {
            "kind": "gpu",
            "accelerator_type": BIG_GPU,
            "catalog_revision": "forged",
            "devices_per_host": 8,
            "gpu_model": "nvidia-h100-80gb",
        }

    state.mutate_record("cell-a", forge, generation=1)

    metadata = driver.observe("cell-a").metadata

    assert CELL_ACCELERATOR_METADATA_KEY not in metadata
    assert read_cell_accelerator(metadata) is None


# ------------------------------------------------------ unknown is not "no GPU"


@pytest.mark.parametrize(
    ("block", "why"),
    [
        (None, "a cell that says nothing"),
        ({"kind": "gpu", "accelerator_type": BIG_GPU, "catalog_revision": "r"}, "a GPU shape with no device count"),
        (
            {"kind": "gpu", "accelerator_type": BIG_GPU, "catalog_revision": "r", "devices_per_host": 0},
            "a device count of zero",
        ),
        (
            {"kind": "fpga", "accelerator_type": "x", "catalog_revision": "r", "devices_per_host": 2},
            "a kind this build has never heard of",
        ),
        ("a3-highgpu-8g", "a block that is not a block"),
    ],
)
def test_metadata_this_build_cannot_read_is_unknown_and_never_zero_devices(block: JsonValue, why: str) -> None:
    """UNKNOWN and "no GPU" are different answers and only one of them is safe.

    Zero devices reaches a rank as an empty ``CUDA_VISIBLE_DEVICES``, which puts
    that process on the CPU while every peer waits for it at a collective. So
    every unreadable input has to land on "not GPU-coordinated" instead.
    """

    metadata: JsonObject = {} if block is None else {CELL_ACCELERATOR_METADATA_KEY: block}

    accelerator = read_cell_accelerator(metadata)

    assert accelerator is None or accelerator.gpu_devices_per_host is None, why


def test_a_readable_description_round_trips_through_the_wire_shape() -> None:
    """Writer and reader have to agree, or the staged rollout ships two schemas."""

    described = describe_cell_accelerator(accelerator_type="nvidia-l4", machine_type=SMALL_GPU)
    assert described is not None

    assert read_cell_accelerator({CELL_ACCELERATOR_METADATA_KEY: described.as_metadata()}) == described


# ------------------------------------------------------------- what is dispatched


def test_an_agent_that_reports_no_accelerator_dispatches_exactly_what_it_does_today(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rolling-upgrade property, asserted on the wire rather than argued.

    An agent older than the accelerator metadata reports none, and its gang must
    keep the caller's own values: nothing written, nothing stripped.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", "a2-ultragpu-8g", ("http://10.0.0.10:8100",)),
        ("cell-b", "a2-ultragpu-8g", ("http://10.0.0.20:8100",)),
    )
    caller = {"CUDA_VISIBLE_DEVICES": "3", "MASTER_ADDR": "caller.example", "WORLD_SIZE": "1", "USER_FLAG": "kept"}

    dispatched = _dispatch(monkeypatch, cells, caller)

    assert set(dispatched) == {"10.0.0.10:8100", "10.0.0.20:8100"}
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert environment["MASTER_ADDR"] == "caller.example"
        assert environment["WORLD_SIZE"] == "1"
        assert environment["USER_FLAG"] == "kept"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_a_gpu_fleet_dispatches_a_gang_whose_ranks_hold_disjoint_devices(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole chain: two applied cells, one observation each, three ranked payloads.

    Twenty devices over three hosts, each device owned exactly once. Host-local
    indices repeat across hosts by construction -- every rank's first device is
    its own ``0`` -- so the gang-wide ordinals are what answers "one device, one
    owner", and they are what is checked.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.11:8100")),
        ("cell-b", SMALL_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells)

    assert [dispatched[host]["CUDA_VISIBLE_DEVICES"] for host in sorted(dispatched)] == [
        "0,1,2,3,4,5,6,7",
        "0,1,2,3,4,5,6,7",
        "0,1,2,3",
    ]
    owned: list[int] = []
    for host in sorted(dispatched):
        environment = dispatched[host]
        offset = int(str(environment["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"]))
        owned.extend(range(offset, offset + int(str(environment["EYVAN_GPU_LOCAL_DEVICE_COUNT"]))))
    assert sorted(owned) == list(range(20)) == owned
    assert len(owned) == len(set(owned))
    assert {str(environment["EYVAN_GPU_GLOBAL_DEVICE_COUNT"]) for environment in dispatched.values()} == {"20"}
    assert {str(environment["EYVAN_GPU_HOST_COUNT"]) for environment in dispatched.values()} == {"3"}
    assert [dispatched[host]["EYVAN_GPU_MODEL"] for host in sorted(dispatched)] == [
        "nvidia-h100-80gb",
        "nvidia-h100-80gb",
        "nvidia-l4",
    ]
    assert {str(environment["MASTER_ADDR"]) for environment in dispatched.values()} == {"10.0.0.10"}


def test_two_ranks_of_one_cell_are_told_they_own_different_devices(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cell, two hosts, two ranks, sixteen devices split down the middle.

    The pair a single-cell gang would get wrong: both ranks share a cell, a
    fence and a device count, and both see local indices ``0..7``. Only their
    gang-wide ordinals distinguish the sixteen devices from eight double-booked
    ones.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.11:8100")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100", "http://10.0.0.21:8100")),
    )

    dispatched = _dispatch(monkeypatch, cells)

    first, second = dispatched["10.0.0.10:8100"], dispatched["10.0.0.11:8100"]
    assert first["CUDA_VISIBLE_DEVICES"] == second["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert (first["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"], second["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"]) == ("0", "8")
    assert (first["NODE_RANK"], second["NODE_RANK"]) == ("0", "1")
    offsets = [int(str(environment["EYVAN_GPU_GLOBAL_DEVICE_OFFSET"])) for environment in dispatched.values()]
    assert sorted(offsets) == [0, 8, 16, 24]


def test_two_workers_on_one_host_are_left_uncoordinated_rather_than_double_booked(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eyvan runs one process per host, so two endpoints on one host is not modelled.

    Both ranks would be handed that host's devices ``0..7`` -- the same eight
    cards, twice -- which is a claim Eyvan cannot keep. It declines to make it
    and dispatches today's bytes instead of an exact-looking assignment that is
    wrong.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100", "http://10.0.0.10:8101")),
        ("cell-b", BIG_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, {"CUDA_VISIBLE_DEVICES": "3"})

    assert set(dispatched) == {"10.0.0.10:8100", "10.0.0.10:8101", "10.0.0.20:8100"}
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_a_gang_a_tpu_cell_belongs_to_is_dispatched_exactly_as_it_is_today(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v5p-16 cell reports an accelerator, and it is still not a GPU one."""

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", "v5p-16", ("http://10.0.0.10:8100", "http://10.0.0.11:8100")),
        ("cell-b", "v5p-16", ("http://10.0.0.20:8100", "http://10.0.0.21:8100")),
    )

    dispatched = _dispatch(monkeypatch, cells, {"CUDA_VISIBLE_DEVICES": "3"})

    assert len(dispatched) == 4
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)
        # The JAX coordination the TPU path already owned is untouched.
        assert environment["EYVAN_JAX_MULTIHOST"] == "1"
        assert environment["JAX_PROCESS_COUNT"] == "4"


def test_a_gang_only_half_of_which_is_a_known_gpu_shape_coordinates_neither_half(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half a plan pins half a gang and leaves the rest to the submission.

    That gang looks coordinated and is not. The executor refuses to be built
    that way; the resolver, whose inputs are observations rather than a
    hand-built gang, downgrades instead -- an agent mid-upgrade and an
    uncatalogued machine both land here, and failing the dispatch would break a
    fleet that runs today.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100",)),
        ("cell-b", "a2-ultragpu-8g", ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells, {"CUDA_VISIBLE_DEVICES": "3"})

    assert set(dispatched) == {"10.0.0.10:8100", "10.0.0.20:8100"}
    for environment in dispatched.values():
        assert environment["CUDA_VISIBLE_DEVICES"] == "3"
        assert not any(name.startswith("EYVAN_GPU_") for name in environment)


def test_a_gpu_gang_still_owns_the_topology_its_submission_tried_to_set(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reaching the path must not lose the property that made it safe.

    W7 proved a hand-built GPU gang strips a caller's device visibility. The
    same has to hold once the gang is built from observations, or a submission
    could pin every rank of a real fleet to device zero.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100",)),
        ("cell-b", SMALL_GPU, ("http://10.0.0.20:8100",)),
    )
    spoofed = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
        "MASTER_ADDR": "attacker.invalid",
        "NODE_RANK": "0",
        "WORLD_SIZE": "1",
        "RANK": "0",
        "NCCL_DEBUG": "INFO",
    }

    dispatched = _dispatch(monkeypatch, cells, spoofed)

    assert dispatched["10.0.0.10:8100"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert dispatched["10.0.0.20:8100"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    for host, rank in (("10.0.0.10:8100", "0"), ("10.0.0.20:8100", "1")):
        environment = dispatched[host]
        assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert environment["MASTER_ADDR"] == "10.0.0.10"
        assert environment["NODE_RANK"] == rank
        assert environment["WORLD_SIZE"] == "12"
        assert "RANK" not in environment
        assert environment["NCCL_DEBUG"] == "INFO", "an operator's NCCL tuning is not a topology claim"


def test_a_single_gpu_cell_is_dispatched_without_a_topology_it_has_no_peers_for(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single GPU cell is not a gang, and Eyvan does not start pretending it is.

    One process per endpoint means a one-cell workload has no peers to be exact
    against, and the single-cell path has never written a topology. The cell now
    reports its accelerator, and that must still change nothing it sends.
    """

    driver = _driver(tmp_path)
    driver.apply_capacity("cell-a", _desired(BIG_GPU, "http://10.0.0.10:8100"), _context("apply"))
    observation = driver.observe("cell-a")
    assert CELL_ACCELERATOR_METADATA_KEY in observation.metadata
    dispatched: dict[str, JsonObject] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert is_json_object(payload)
        spec = payload["spec"]
        assert is_json_object(spec)
        spec_environment = spec["environment"]
        assert is_json_object(spec_environment)
        dispatched[f"{request.url.host}:{request.url.port}"] = spec_environment
        return httpx.Response(200, json={"remote_id": "gpu-job"})

    real_client = httpx.Client
    monkeypatch.setattr(
        native_remote.httpx,
        "Client",
        lambda *_a, **kwargs: real_client(base_url=str(kwargs["base_url"]), transport=httpx.MockTransport(handler)),
    )
    executor = _resolver().resolve("cell-a", observation)
    remote_id = executor.submit(
        {
            "entrypoint": "python train.py",
            "submission_id": "gpu-job",
            "environment": {"CUDA_VISIBLE_DEVICES": "3"},
        },
        _context("submit-a"),
    )

    assert remote_id == "gpu-job"
    environment = dispatched["10.0.0.10:8100"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert not any(name.startswith("EYVAN_GPU_") for name in environment)
    assert "MASTER_ADDR" not in environment


def test_the_gang_dispatched_for_a_gpu_fleet_carries_no_hardware_claim(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What Eyvan states is a placement, and it must not read as a run.

    Everything dispatched is a device *assignment*. Nothing reports an observed
    device, a driver version, or a successful initialization, because no GPU has
    executed an Eyvan workload and this file must not be able to be read as
    evidence that one has.
    """

    cells = _cells(
        _driver(tmp_path),
        ("cell-a", BIG_GPU, ("http://10.0.0.10:8100",)),
        ("cell-b", SMALL_GPU, ("http://10.0.0.20:8100",)),
    )

    dispatched = _dispatch(monkeypatch, cells)

    for environment in dispatched.values():
        eyvan_gpu = {name for name in environment if name.startswith("EYVAN_GPU_")}
        assert eyvan_gpu == {
            "EYVAN_GPU_MODEL",
            "EYVAN_GPU_HOST_COUNT",
            "EYVAN_GPU_HOST_INDEX",
            "EYVAN_GPU_LOCAL_DEVICE_COUNT",
            "EYVAN_GPU_GLOBAL_DEVICE_COUNT",
            "EYVAN_GPU_GLOBAL_DEVICE_OFFSET",
        }


# ------------------------------------------------- the same join on a GCP cell


def _compute_driver(tmp_path) -> GcpComputeCapacityDriver:
    return GcpComputeCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "gcp.json")),
        provider=_StubProvider(),
        execution_backend="native-local",
    )


class _StubProvider:
    """A provider that reports one healthy cell and touches no cloud."""

    def ensure(self, record: object) -> ProviderObservation:
        return ProviderObservation(CapacityStatus.HEALTHY, "instance running", head_ip="10.0.0.10")

    def delete(self, resource_id: str, *, project: str, zone: str, management_uid: str | None = None) -> None:
        raise AssertionError("this test never releases capacity")


def _compute_desired(machine_type: str, **options: JsonValue) -> JsonObject:
    provider_options: JsonObject = {"machine_type": machine_type, **options}
    return {
        "capacity_backend": "gcp-compute",
        "project": "project-a",
        "zone": "us-central1-a",
        "accelerator_type": "nvidia-l4" if "gpu_type" in options else machine_type,
        "capacity_class": "on-demand",
        "executor": "native-local",
        "provider_options": provider_options,
    }


def test_a_gcp_gpu_cell_is_found_through_its_machine_type_not_its_gpu_model(tmp_path) -> None:
    """A GCP GPU pool writes the model into ``accelerator_type``, and the model says nothing.

    ``g2-standard-4`` and ``g2-standard-96`` both carry an L4 and differ by 8x in
    attached devices, which is why the catalog is keyed by machine type -- and
    why the observation has to look the cell up by the type it was created as.
    """

    driver = _compute_driver(tmp_path)
    driver.apply_capacity("gpu-cell", _compute_desired(SMALL_GPU, gpu_type="nvidia-l4", gpu_count=4), _context("apply"))

    accelerator = read_cell_accelerator(driver.observe("gpu-cell").metadata)

    assert accelerator is not None
    assert accelerator.accelerator_type == SMALL_GPU
    assert accelerator.gpu_devices_per_host == 4
    assert accelerator.gpu_model == "nvidia-l4"


def test_a_gcp_host_whose_devices_come_with_the_machine_type_is_described_too(tmp_path) -> None:
    """An A3 host attaches its eight H100s by machine family, with no accelerator option.

    Reading the device count only from a configured ``gpu_count`` would leave
    exactly the biggest hosts undescribed.
    """

    driver = _compute_driver(tmp_path)
    driver.apply_capacity("gpu-cell", _compute_desired(BIG_GPU), _context("apply"))

    accelerator = read_cell_accelerator(driver.observe("gpu-cell").metadata)

    assert accelerator is not None
    assert accelerator.gpu_devices_per_host == 8
    assert accelerator.gpu_model == "nvidia-h100-80gb"


def test_a_gcp_cell_whose_two_sources_disagree_is_reported_as_unknown(tmp_path) -> None:
    """Catalog says four, the instance was built with one, and nothing here can arbitrate.

    Picking a side would hand a rank an exact device list built on a coin flip.
    The cell stays unknown, which is the path it is on today.
    """

    driver = _compute_driver(tmp_path)
    driver.apply_capacity("gpu-cell", _compute_desired(SMALL_GPU, gpu_type="nvidia-l4", gpu_count=1), _context("apply"))

    metadata = driver.observe("gpu-cell").metadata

    assert CELL_ACCELERATOR_METADATA_KEY not in metadata
    assert read_cell_accelerator(metadata) is None


def test_a_gcp_cell_on_an_uncatalogued_machine_type_reports_nothing(tmp_path) -> None:
    """``g2-standard-8`` is a real host with no row here, so Eyvan describes none."""

    driver = _compute_driver(tmp_path)
    driver.apply_capacity(
        "gpu-cell",
        _compute_desired("g2-standard-8", gpu_type="nvidia-l4", gpu_count=1),
        _context("apply"),
    )

    assert CELL_ACCELERATOR_METADATA_KEY not in driver.observe("gpu-cell").metadata


def test_a_gcp_cpu_cell_reports_no_accelerator(tmp_path) -> None:
    """A machine with nothing attached must not acquire a description of one."""

    driver = _compute_driver(tmp_path)
    driver.apply_capacity("cpu-cell", _compute_desired("c4-standard-8"), _context("apply"))

    assert CELL_ACCELERATOR_METADATA_KEY not in driver.observe("cpu-cell").metadata


def test_a_gcp_tpu_cell_reports_a_tpu_and_stays_off_the_gpu_path(tmp_path) -> None:
    """The queued-resource TPU driver publishes the same block for a different kind."""

    driver = GcpTpuCapacityDriver(
        FleetStateStore(LocalBackend(tmp_path / "gcp.json")),
        provider=_StubProvider(),
        execution_backend="native-local",
    )
    driver.apply_capacity(
        "tpu-cell",
        {
            "capacity_backend": "gcp-tpu",
            "project": "project-a",
            "zone": "us-east5-a",
            "accelerator_type": "v5p-16",
            "capacity_class": "on-demand",
            "executor": "native-local",
        },
        _context("apply"),
    )

    accelerator = read_cell_accelerator(driver.observe("tpu-cell").metadata)

    assert accelerator is not None
    assert accelerator.kind == "tpu"
    assert accelerator.gpu_devices_per_host is None
