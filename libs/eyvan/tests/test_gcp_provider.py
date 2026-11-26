# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import subprocess
from dataclasses import replace

import pytest

from eyvan._internal.json_types import JsonValue
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import CapacityProviderFailureCode
from eyvan.capacity.fleet_state import FleetCellRecord
from eyvan.capacity.providers.gcp.queued_resources import (
    GcloudError,
    GcloudQueueAdmissionDeniedError,
    GcloudTpuProvider,
    QueuedResourceSpec,
    normalize_tpu_config,
)

UID = "0123456789abcdef0123456789abcdef"


def result(returncode: int = 0, *, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["gcloud"], returncode, stdout, stderr)


def qr_payload(
    *,
    state: str = "WAITING_FOR_RESOURCES",
    uid: str = UID,
    accelerator_type: str = "v6e-8",
    node_id: str = "cell-a",
) -> dict:
    return {
        "name": "projects/project-a/locations/us-central2-b/queuedResources/cell-a-r1",
        "state": {"state": state},
        "labels": {"eyvan-cell": "cell-a", "eyvan-uid": uid},
        "tpu": {"nodeSpec": [{"nodeId": node_id, "node": {"acceleratorType": accelerator_type}}]},
    }


def record() -> FleetCellRecord:
    return FleetCellRecord(
        name="cell-a",
        project="project-a",
        zone="us-central2-b",
        accelerator_type="v6e-8",
        qr_id="cell-a-r1",
        metadata={"management_uid": UID},
    )


def spec() -> QueuedResourceSpec:
    return QueuedResourceSpec(
        name="cell-a",
        accelerator_type="v6e-8",
        zone="us-central2-b",
        project="project-a",
        capacity_class="spot",
        labels={"eyvan-cell": "cell-a", "eyvan-uid": UID},
        metadata={"eyvan-uid": UID},
    )


def test_queued_resource_args_are_deterministic_and_eyvan_owned() -> None:
    args = spec().create_args(qr_id="cell-a-r1")

    assert args[:5] == ["compute", "tpus", "queued-resources", "create", "cell-a-r1"]
    assert "--spot" in args
    assert args[args.index("--labels") + 1] == f"eyvan-cell=cell-a,eyvan-uid={UID}"
    assert args[args.index("--metadata") + 1] == f"eyvan-uid={UID}"
    assert "--internal-ips" not in args
    assert not any("eray" in value for value in args)


def test_queued_resource_describe_rejects_invalid_gcloud_json() -> None:
    provider = GcloudTpuProvider(runner=lambda _args: result(stdout="not-json"))

    with pytest.raises(GcloudError, match="returned invalid JSON"):
        provider.describe_qr("cell-a-r1", project="project-a", zone="us-central2-b")


@pytest.mark.parametrize(
    ("state", "message"),
    [(None, "did not report a state"), ("FUTURE_STATE", "unsupported state")],
)
def test_queued_resource_describe_rejects_missing_or_unsupported_states(
    state: str | None,
    message: str,
) -> None:
    payload = qr_payload()
    payload["state"] = {} if state is None else {"state": state}
    provider = GcloudTpuProvider(runner=lambda _args: result(stdout=json.dumps(payload)))

    with pytest.raises(GcloudError, match=message):
        provider.describe_qr("cell-a-r1", project="project-a", zone="us-central2-b")


def test_tpu_provider_options_normalize_and_render_exact_gcloud_flags() -> None:
    config = normalize_tpu_config(
        {
            "provider_options": {
                "runtime_version": "v2-alpha-tpuv5",
                "network": "projects/network-host/global/networks/research",
                "subnetwork": "projects/network-host/regions/us-east5/subnetworks/tpu",
                "internal_ips": True,
                "service_account": "tpu-runtime@project-a.iam.gserviceaccount.com",
                "scopes": ["cloud-platform", "storage-ro"],
                "tags": ["eyvan-worker", "research"],
                "metadata": {"enable-oslogin": "TRUE"},
            }
        }
    )

    assert config == {
        "runtime_version": "v2-alpha-tpuv5",
        "network": "projects/network-host/global/networks/research",
        "subnetwork": "projects/network-host/regions/us-east5/subnetworks/tpu",
        "internal_ips": True,
        "service_account": "tpu-runtime@project-a.iam.gserviceaccount.com",
        "scopes": ["cloud-platform", "storage-ro"],
        "tags": ["eyvan-worker", "research"],
        "metadata": {"enable-oslogin": "TRUE"},
    }
    configured = {**config, "metadata": {**config["metadata"], "eyvan-uid": UID}}
    args = replace(spec(), **configured).create_args(qr_id="cell-a-r1")
    assert args == [
        "compute",
        "tpus",
        "queued-resources",
        "create",
        "cell-a-r1",
        "--zone",
        "us-central2-b",
        "--project",
        "project-a",
        "--accelerator-type",
        "v6e-8",
        "--runtime-version",
        "v2-alpha-tpuv5",
        "--node-id",
        "cell-a",
        "--spot",
        "--labels",
        f"eyvan-cell=cell-a,eyvan-uid={UID}",
        "--metadata",
        f"enable-oslogin=TRUE,eyvan-uid={UID}",
        "--network",
        "projects/network-host/global/networks/research",
        "--subnetwork",
        "projects/network-host/regions/us-east5/subnetworks/tpu",
        "--internal-ips",
        "--service-account",
        "tpu-runtime@project-a.iam.gserviceaccount.com",
        "--scopes",
        "cloud-platform,storage-ro",
        "--tags",
        "eyvan-worker,research",
    ]


def test_multislice_queued_resource_renders_atomic_node_count_and_prefix() -> None:
    configured = normalize_tpu_config(
        {
            "provider_options": {
                "node_count": 2,
                "node_prefix": "cell-a-r1-node",
                "startup_script": "#!/bin/bash\necho native worker\n",
                "bootstrap_egress": "private-google-access",
            }
        }
    )
    multi = replace(
        spec(),
        accelerator_type="v5p-16",
        runtime_version="v2-alpha-tpuv5",
        node_count=2,
        node_prefix="cell-a-r1-node",
        startup_script=configured["startup_script"],
        bootstrap_egress=configured["bootstrap_egress"],
    )

    args = multi.create_args(qr_id="cell-a-r1", startup_script_path="/tmp/startup.sh")

    assert "--node-id" not in args
    assert args[args.index("--node-count") + 1] == "2"
    assert args[args.index("--node-prefix") + 1] == "cell-a-r1-node"
    assert args[args.index("--metadata-from-file") + 1] == "startup-script=/tmp/startup.sh"


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ("not-a-mapping", "must be a mapping"),
        ({"unknown": "value"}, "unknown TPU provider option"),
        ({"internal_ips": "true"}, "must be a boolean"),
        ({"scopes": "cloud-platform"}, "must be a list of strings"),
        ({"tags": [1]}, "non-empty trimmed string"),
        ({"metadata": []}, "must be a string mapping"),
        ({"metadata": {"eyvan-uid": "foreign"}}, "reserved by Eyvan"),
        ({"metadata": {"EYVAN-cell": "foreign"}}, "reserved by Eyvan"),
    ],
)
def test_tpu_provider_options_reject_unknown_invalid_and_reserved_values(options, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_tpu_config({"provider_options": options})


def test_provider_builds_spec_from_durable_config_and_preserves_ownership() -> None:
    value = record()
    value.runtime_version = "v2-alpha-tpuv5"
    value.metadata["tpu_config"] = normalize_tpu_config(
        {
            "runtime_version": value.runtime_version,
            "network": "research",
            "internal_ips": False,
            "metadata": {"enable-oslogin": "TRUE"},
        }
    )

    configured = GcloudTpuProvider._spec(value)

    assert configured.runtime_version == "v2-alpha-tpuv5"
    assert configured.network == "research"
    assert configured.internal_ips is False
    assert configured.metadata == {"enable-oslogin": "TRUE", "eyvan-uid": UID}
    assert configured.labels == {"eyvan-cell": "cell-a", "eyvan-uid": UID}


def test_default_provider_transport_bounds_gcloud_calls(monkeypatch) -> None:
    calls: list[dict] = []

    def run(command, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 1, "", "NOT_FOUND")

    monkeypatch.delenv("EYVAN_GCLOUD_TPU_TIMEOUT_S", raising=False)
    monkeypatch.setattr(subprocess, "run", run)

    assert GcloudTpuProvider().describe_qr("cell-a-r1", project="project-a", zone="us-central2-b") is None
    assert calls[0]["timeout"] == 600


@pytest.mark.parametrize(
    ("queued_resource_state", "expected_capacity_state"),
    (
        ("WAITING_FOR_RESOURCES", CapacityStatus.WAITING),
        ("PROVISIONING", CapacityStatus.PROVISIONING),
        ("ACTIVE", CapacityStatus.ACTIVE),
    ),
)
def test_provider_reports_the_first_provider_confirmed_queued_resource_state(
    queued_resource_state: str,
    expected_capacity_state: CapacityStatus,
) -> None:
    created = False

    def runner(args):
        nonlocal created
        if "tpu-vm" in args:
            return result(1, stderr="NOT_FOUND")
        if "create" in args:
            created = True
            return result()
        if "queued-resources" in args:
            if not created:
                return result(1, stderr="NOT_FOUND")
            return result(stdout=json.dumps(qr_payload(state=queued_resource_state)))
        raise AssertionError(tuple(args))

    observation = GcloudTpuProvider(runner=runner).ensure(record())

    assert observation.state is expected_capacity_state
    assert observation.detail.startswith("queued resource was created")


def test_provider_observes_only_an_owned_ready_tpu() -> None:
    calls: list[tuple[str, ...]] = []

    def runner(args):
        calls.append(tuple(args))
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        assert "tpu-vm" in args
        return result(
            stdout=json.dumps(
                {
                    "name": "projects/project-a/locations/us-central2-b/nodes/cell-a",
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "10.2.0.4"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(record())

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.head_ip == "10.2.0.4"
    assert len(calls) == 2


def _multislice_record(accelerator_type: str = "v5p-16") -> FleetCellRecord:
    value = record()
    value.accelerator_type = accelerator_type
    value.metadata.update(
        {
            "execution_backend": "eyvan-native",
            "native_worker_port": 8123,
            "tpu_config": normalize_tpu_config({"provider_options": {"node_count": 2}}),
        }
    )
    return value


def _multislice_qr(accelerator_type: str = "v5p-16") -> dict:
    return {
        "name": "projects/project-a/locations/us-central2-b/queuedResources/cell-a-r1",
        "state": {"state": "ACTIVE"},
        "tpu": {
            "nodeSpec": [
                {
                    "nodeId": "cell-a-r1-node-0",
                    "node": {
                        "acceleratorType": accelerator_type,
                        "labels": {"eyvan-cell": "cell-a", "eyvan-uid": UID},
                    },
                },
                {
                    "nodeId": "cell-a-r1-node-1",
                    "node": {
                        "acceleratorType": accelerator_type,
                        "labels": {"eyvan-cell": "cell-a", "eyvan-uid": UID},
                    },
                },
            ]
        },
    }


def test_provider_observes_every_multislice_tpu_vm_before_native_execution() -> None:
    nodes = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5"),
        "cell-a-r1-node-1": ("10.2.0.6", "10.2.0.7"),
    }

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in nodes[node_id]],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.head_ip == "10.2.0.4"
    assert observation.worker_endpoints == (
        "http://10.2.0.4:8123",
        "http://10.2.0.5:8123",
        "http://10.2.0.6:8123",
        "http://10.2.0.7:8123",
    )
    assert observation.observed_tpu_vms == 4
    assert observation.runtime_process_count == 4


def test_provider_reports_only_gcp_observed_v6e_tpu_vm_count() -> None:
    nodes = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5", "10.2.0.6", "10.2.0.7"),
        "cell-a-r1-node-1": ("10.2.0.8", "10.2.0.9", "10.2.0.10", "10.2.0.11"),
    }

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr("v6e-16")))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in nodes[node_id]],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record("v6e-16"))

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.observed_tpu_vms == 8
    assert observation.runtime_process_count == 8
    assert observation.detail == "2 TPU worker allocations and 8 TPU VMs are ready"


def test_provider_observes_every_v5p_32_tpu_host_before_native_execution() -> None:
    value = record()
    value.accelerator_type = "v5p-32"
    value.metadata.update({"execution_backend": "eyvan-native", "native_worker_port": 8123})
    worker_ips = ("10.2.0.4", "10.2.0.5", "10.2.0.6", "10.2.0.7")

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE", accelerator_type="v5p-32")))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v5p-32",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in worker_ips],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.worker_endpoints == tuple(f"http://{ip}:8123" for ip in worker_ips)


def test_provider_uses_every_endpoint_returned_for_a_ready_tpu_node() -> None:
    """Every endpoint the provider returns reaches the workload, in order.

    Two hosts per node because that is what a v5p-16 slice is. The fixture used
    to give the second node a single IP, which made the allocation three hosts
    where the shape says four -- an arrangement the provider cannot produce, and
    one the hosts-per-slice check now correctly reports as still provisioning.
    """

    nodes = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5"),
        "cell-a-r1-node-1": ("10.2.0.6", "10.2.0.7"),
    }

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in nodes[node_id]],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.worker_endpoints == (
        "http://10.2.0.4:8123",
        "http://10.2.0.5:8123",
        "http://10.2.0.6:8123",
        "http://10.2.0.7:8123",
    )


@pytest.mark.parametrize("missing_node", ["cell-a-r1-node-0", "cell-a-r1-node-1"])
def test_provider_keeps_multislice_provisioning_until_every_ready_node_has_endpoints(missing_node: str) -> None:
    nodes = {
        "cell-a-r1-node-0": ("10.2.0.4", "10.2.0.5"),
        "cell-a-r1-node-1": ("10.2.0.6", "10.2.0.7"),
    }
    nodes[missing_node] = ()

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in nodes[node_id]],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.execution_endpoint is None
    assert observation.worker_endpoints == ()
    assert observation.observed_tpu_vms is None


@pytest.mark.parametrize(
    "network_endpoints",
    [
        "not-a-list",
        ["not-an-object"],
        [{"ipAddress": ""}],
        [{"ipAddress": " 10.2.0.4 "}],
        [{"ipAddress": "10.2.0.999"}],
        [{"ipAddress": "127.0.0.1@attacker.example"}],
        [{"ipAddress": "worker.example.com"}],
    ],
)
def test_provider_rejects_malformed_tpu_worker_endpoints(network_endpoints: JsonValue) -> None:
    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": network_endpoints,
                }
            )
        )

    with pytest.raises(GcloudError, match=r"invalid network endpoint|valid IP address"):
        GcloudTpuProvider(runner=runner).ensure(record())


def test_provider_keeps_ready_tpu_provisioning_while_gcp_endpoint_ip_is_pending() -> None:
    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [
                        {"ipAddress": "10.2.0.4"},
                        {"accessConfig": {"externalIp": "34.1.2.3"}},
                    ],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(record())

    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.worker_endpoints == ()


def test_profile_managed_public_tpu_workers_publish_external_not_internal_endpoints() -> None:
    value = record()
    value.metadata.update(
        {
            "execution_backend": "eyvan-native",
            "native_worker_port": 8123,
            "native_worker_endpoint_mode": "profile-managed-public",
        }
    )

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [
                        {
                            "ipAddress": "10.2.0.4",
                            "accessConfig": {"externalIp": "34.1.2.3"},
                        }
                    ],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.head_ip == "34.1.2.3"
    assert observation.execution_endpoint == "http://34.1.2.3:8123"
    assert observation.worker_endpoints == ("http://34.1.2.3:8123",)


def test_profile_managed_public_tpu_waits_for_external_ips_without_private_fallback() -> None:
    value = record()
    value.metadata.update(
        {
            "execution_backend": "eyvan-native",
            "native_worker_port": 8123,
            "native_worker_endpoint_mode": "profile-managed-public",
        }
    )

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "10.2.0.4"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.head_ip is None
    assert observation.execution_endpoint is None
    assert observation.worker_endpoints == ()


def test_provider_formats_valid_ipv6_worker_endpoints_as_urls() -> None:
    value = record()
    value.metadata.update({"execution_backend": "eyvan-native", "native_worker_port": 8123})

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "2001:0db8:0:0:0:0:0:1"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.worker_endpoints == ("http://[2001:db8::1]:8123",)


def _ready_slice(accelerator_type: str, worker_ips: tuple[str, ...]):
    """A runner whose queued resource is ACTIVE and whose node reports ``worker_ips``."""

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE", accelerator_type=accelerator_type)))
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": accelerator_type,
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in worker_ips],
                }
            )
        )

    return runner


def test_provider_observes_a_ready_shape_whose_host_count_eyvan_does_not_know() -> None:
    """v6e states its per-VM resources and deliberately not its endpoint count.

    Google's table distinguishes physical hosts from VMs there and nothing in
    this repository establishes which one the provider reports, so Eyvan takes
    the provider's endpoints as given rather than holding a slice to a number
    it would be guessing.
    """

    value = record()
    value.accelerator_type = "v6e-64"
    value.metadata.update({"execution_backend": "eyvan-native", "native_worker_port": 8123})
    worker_ips = tuple(f"10.2.0.{index}" for index in range(4, 12))

    observation = GcloudTpuProvider(runner=_ready_slice("v6e-64", worker_ips)).ensure(value)

    assert observation.state is CapacityStatus.HEALTHY
    assert observation.worker_endpoints == tuple(f"http://{ip}:8123" for ip in worker_ips)


def test_a_slice_short_of_its_known_host_count_is_still_provisioning() -> None:
    """v5p does state its endpoint count, so a partial slice is not ready.

    A real v5p-512 is 256 chips across sixty-four ct5p-hightpu-4t VMs. Calling
    it healthy at eight would start a workload on an eighth of the hardware it
    asked for, which fails later and blames the job.
    """

    value = record()
    value.accelerator_type = "v5p-512"
    value.metadata.update({"execution_backend": "eyvan-native", "native_worker_port": 8123})
    worker_ips = tuple(f"10.2.0.{index}" for index in range(4, 12))

    observation = GcloudTpuProvider(runner=_ready_slice("v5p-512", worker_ips)).ensure(value)

    assert observation.state is CapacityStatus.PROVISIONING
    assert "8" in observation.detail and "v5p-512" in observation.detail


@pytest.mark.parametrize(
    ("node_state", "worker_ips"),
    [
        ("CREATING", ("10.2.0.4",)),
        ("READY", ()),
    ],
)
def test_provider_keeps_arbitrary_tpu_shape_provisioning_until_ready_with_endpoints(
    node_state: str,
    worker_ips: tuple[str, ...],
) -> None:
    value = record()
    value.accelerator_type = "v5p-512"
    value.metadata.update({"execution_backend": "eyvan-native", "native_worker_port": 8123})

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE", accelerator_type="v5p-512")))
        return result(
            stdout=json.dumps(
                {
                    "state": node_state,
                    "acceleratorType": "v5p-512",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": ip} for ip in worker_ips],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.PROVISIONING
    assert observation.worker_endpoints == ()


def test_provider_preserves_the_official_gcp_unknown_node_state_as_unavailable() -> None:
    value = record()

    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state="ACTIVE")))
        return result(
            stdout=json.dumps(
                {
                    "state": "UNKNOWN",
                    "acceleratorType": "v6e-8",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(value)

    assert observation.state is CapacityStatus.UNAVAILABLE
    assert "UNKNOWN" in observation.detail


def test_provider_recreates_when_a_spot_multislice_node_is_preempted() -> None:
    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        state = "PREEMPTED" if node_id.endswith("-1") else "READY"
        return result(
            stdout=json.dumps(
                {
                    "state": state,
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "10.2.0.4" if node_id.endswith("-0") else "10.2.0.5"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert "PREEMPTED" in observation.detail


def test_provider_recreates_when_an_active_spot_multislice_node_disappears() -> None:
    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(_multislice_qr()))
        node_id = args[args.index("describe") + 1]
        if node_id.endswith("-1"):
            return result(1, stderr="NOT_FOUND: TPU node does not exist")
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v5p-16",
                    "metadata": {"eyvan-uid": UID},
                    "networkEndpoints": [{"ipAddress": "10.2.0.4"}],
                }
            )
        )

    observation = GcloudTpuProvider(runner=runner).ensure(_multislice_record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert "missing TPU worker allocations" in observation.detail


@pytest.mark.parametrize("state", ["FAILED", "SUSPENDED"])
def test_nested_terminal_qr_states_require_recreation(state: str) -> None:
    def runner(args):
        if "queued-resources" in args:
            return result(stdout=json.dumps(qr_payload(state=state)))
        return result(1, stderr="NOT_FOUND: TPU node does not exist")

    observation = GcloudTpuProvider(runner=runner).ensure(record())

    assert observation.state is CapacityStatus.NEEDS_RECREATE
    assert state in observation.detail


def test_foreign_qr_and_same_named_node_are_never_adopted() -> None:
    foreign_qr = GcloudTpuProvider._parse_qr(qr_payload(uid="foreign", accelerator_type="v4-8"))
    with pytest.raises(GcloudError, match="not owned"):
        GcloudTpuProvider._validate_qr(foreign_qr, spec())

    def runner(args):
        if "queued-resources" in args:
            return result(1, stderr="NOT_FOUND: queued resource does not exist")
        return result(
            stdout=json.dumps(
                {
                    "state": "READY",
                    "acceleratorType": "v6e-8",
                    "networkEndpoints": [{"ipAddress": "10.2.0.9"}],
                }
            )
        )

    with pytest.raises(GcloudError, match="without its owned queued resource"):
        GcloudTpuProvider(runner=runner).ensure(record())


def test_provider_rejects_partially_owned_node_labeled_multislice_qr() -> None:
    foreign = _multislice_qr()
    foreign["tpu"]["nodeSpec"][1]["node"]["labels"]["eyvan-uid"] = "foreign"

    with pytest.raises(GcloudError, match="not owned"):
        GcloudTpuProvider._validate_qr(GcloudTpuProvider._parse_qr(foreign), spec())


def test_delete_accepts_owned_node_labeled_qr() -> None:
    descriptions = 0

    def runner(args):
        nonlocal descriptions
        if "describe" in args:
            descriptions += 1
            if descriptions == 1:
                return result(stdout=json.dumps(_multislice_qr()))
            return result(1, stderr="NOT_FOUND: queued resource does not exist")
        if "delete" in args:
            return result()
        raise AssertionError(tuple(args))

    GcloudTpuProvider(runner=runner, delete_wait_timeout_s=0).delete(
        "cell-a-r1",
        project="project-a",
        zone="us-central2-b",
        management_uid=UID,
    )

    assert descriptions == 2


def test_create_error_recovers_only_when_owned_resource_landed() -> None:
    descriptions = 0

    def runner(args):
        nonlocal descriptions
        if "describe" in args:
            descriptions += 1
            if descriptions == 1:
                return result(1, stderr="NOT_FOUND")
            return result(stdout=json.dumps(qr_payload()))
        if "create" in args:
            return result(
                1,
                stderr=(
                    "PERMISSION_DENIED: User does not have permission to submit requests into this queue "
                    'for accelerator type "v6e-16" in location us-central1-c.'
                ),
            )
        raise AssertionError(tuple(args))

    created = GcloudTpuProvider(runner=runner).create(spec(), qr_id="cell-a-r1")

    assert created.qr_id == "cell-a-r1"
    assert descriptions == 2


def test_create_classifies_explicit_gcp_queue_admission_permission_denial() -> None:
    def runner(args):
        if "describe" in args:
            return result(1, stderr="NOT_FOUND: queued resource does not exist")
        if "create" in args:
            return result(
                1,
                stderr=(
                    "PERMISSION_DENIED: User does not have permission to submit requests into this queue "
                    'for accelerator type "v6e-16" in location us-central1-c.'
                ),
            )
        raise AssertionError(tuple(args))

    with pytest.raises(GcloudQueueAdmissionDeniedError, match="submit requests into this queue") as error:
        GcloudTpuProvider(runner=runner).create(spec(), qr_id="cell-a-r1")

    assert error.value.code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED


def test_create_classifies_a_zone_without_a_queue_for_the_type_as_denied() -> None:
    """GCP's INVALID_ARGUMENT for a zone with no queue for this type is as final
    as a permission denial: no retry of the same request can be admitted."""

    def runner(args):
        if "describe" in args:
            return result(1, stderr="NOT_FOUND: queued resource does not exist")
        if "create" in args:
            return result(
                1,
                stderr=(
                    "INVALID_ARGUMENT: Cloud TPU received an invalid argument. Queueing is not supported "
                    'for accelerator type "v5litepod-8" in location europe-west3-c.'
                ),
            )
        raise AssertionError(tuple(args))

    with pytest.raises(GcloudQueueAdmissionDeniedError, match="Queueing is not supported") as error:
        GcloudTpuProvider(runner=runner).create(spec(), qr_id="cell-a-r1")

    assert error.value.code is CapacityProviderFailureCode.QUEUE_ADMISSION_DENIED


def test_create_keeps_generic_permission_denial_as_provider_error() -> None:
    def runner(args):
        if "describe" in args:
            return result(1, stderr="NOT_FOUND: queued resource does not exist")
        if "create" in args:
            return result(1, stderr="PERMISSION_DENIED: missing tpu.nodes.create")
        raise AssertionError(tuple(args))

    with pytest.raises(GcloudError, match=r"missing tpu\.nodes\.create") as error:
        GcloudTpuProvider(runner=runner).create(spec(), qr_id="cell-a-r1")

    assert type(error.value) is GcloudError
    assert error.value.code is CapacityProviderFailureCode.PROVIDER_ERROR


def test_delete_is_idempotent_when_absent_and_refuses_foreign_replacement() -> None:
    absent_calls: list[tuple[str, ...]] = []

    def absent(args):
        absent_calls.append(tuple(args))
        return result(1, stderr="NOT_FOUND: queued resource does not exist")

    GcloudTpuProvider(runner=absent).delete(
        "cell-a-r1",
        project="project-a",
        zone="us-central2-b",
        management_uid=UID,
    )
    assert len(absent_calls) == 1

    def foreign(_args):
        return result(stdout=json.dumps(qr_payload(uid="foreign")))

    with pytest.raises(GcloudError, match="refusing to delete"):
        GcloudTpuProvider(runner=foreign).delete(
            "cell-a-r1",
            project="project-a",
            zone="us-central2-b",
            management_uid=UID,
        )


def test_successful_delete_rechecks_absence_even_with_zero_wait() -> None:
    descriptions = 0

    def runner(args):
        nonlocal descriptions
        if "describe" in args:
            descriptions += 1
            if descriptions == 1:
                return result(stdout=json.dumps(qr_payload()))
            return result(1, stderr="NOT_FOUND")
        if "delete" in args:
            return result()
        raise AssertionError(tuple(args))

    GcloudTpuProvider(runner=runner, delete_wait_timeout_s=0).delete(
        "cell-a-r1",
        project="project-a",
        zone="us-central2-b",
        management_uid=UID,
    )

    assert descriptions == 2


@pytest.mark.parametrize("present_after_error", [False, True])
def test_delete_ambiguous_failure_rechecks_presence(present_after_error: bool) -> None:
    descriptions = 0

    def runner(args):
        nonlocal descriptions
        if "describe" in args:
            descriptions += 1
            if descriptions == 1 or present_after_error:
                return result(stdout=json.dumps(qr_payload()))
            return result(1, stderr="NOT_FOUND")
        if "delete" in args:
            return result(1, stderr="connection closed before response")
        raise AssertionError(tuple(args))

    provider = GcloudTpuProvider(runner=runner, delete_wait_timeout_s=0)
    if present_after_error:
        with pytest.raises(GcloudError, match="connection closed"):
            provider.delete(
                "cell-a-r1",
                project="project-a",
                zone="us-central2-b",
                management_uid=UID,
            )
    else:
        provider.delete(
            "cell-a-r1",
            project="project-a",
            zone="us-central2-b",
            management_uid=UID,
        )


def test_delete_wait_timeout_is_a_typed_provider_failure() -> None:
    def runner(args):
        if "describe" in args:
            return result(stdout=json.dumps(qr_payload()))
        if "delete" in args:
            return result()
        raise AssertionError(tuple(args))

    with pytest.raises(GcloudError, match="still exists after delete") as error:
        GcloudTpuProvider(runner=runner, delete_wait_timeout_s=0).delete(
            "cell-a-r1",
            project="project-a",
            zone="us-central2-b",
            management_uid=UID,
        )

    assert type(error.value) is GcloudError
    assert error.value.code is CapacityProviderFailureCode.PROVIDER_ERROR
    assert error.value.diagnostic.startswith("queued resource 'cell-a-r1' still exists after delete")


def test_gcloud_subprocess_timeout_is_a_typed_provider_failure() -> None:
    def runner(args):
        raise subprocess.TimeoutExpired(cmd=["gcloud", *args], timeout=5)

    provider = GcloudTpuProvider(runner=runner)
    with pytest.raises(GcloudError, match="timed out after 5 seconds") as error:
        provider.describe_qr("cell-a-r1", project="project-a", zone="us-central2-b")

    assert type(error.value) is GcloudError
    assert error.value.code is CapacityProviderFailureCode.PROVIDER_ERROR


def test_create_recovers_when_a_timed_out_create_actually_landed() -> None:
    descriptions = 0

    def runner(args):
        nonlocal descriptions
        if "describe" in args:
            descriptions += 1
            if descriptions == 1:
                return result(1, stderr="NOT_FOUND")
            return result(stdout=json.dumps(qr_payload()))
        if "create" in args:
            raise subprocess.TimeoutExpired(cmd=["gcloud", *args], timeout=5)
        raise AssertionError(tuple(args))

    created = GcloudTpuProvider(runner=runner).create(spec(), qr_id="cell-a-r1")

    assert created.qr_id == "cell-a-r1"
    assert descriptions == 2


def test_create_surfaces_typed_failure_when_a_timed_out_create_never_landed() -> None:
    def runner(args):
        if "describe" in args:
            return result(1, stderr="NOT_FOUND")
        if "create" in args:
            raise subprocess.TimeoutExpired(cmd=["gcloud", *args], timeout=5)
        raise AssertionError(tuple(args))

    with pytest.raises(GcloudError, match="timed out after 5 seconds") as error:
        GcloudTpuProvider(runner=runner).create(spec(), qr_id="cell-a-r1")

    assert type(error.value) is GcloudError
    assert error.value.code is CapacityProviderFailureCode.PROVIDER_ERROR


@pytest.mark.parametrize("present_after_timeout", [False, True])
def test_delete_command_timeout_rechecks_presence(present_after_timeout: bool) -> None:
    descriptions = 0

    def runner(args):
        nonlocal descriptions
        if "describe" in args:
            descriptions += 1
            if descriptions == 1 or present_after_timeout:
                return result(stdout=json.dumps(qr_payload()))
            return result(1, stderr="NOT_FOUND")
        if "delete" in args:
            raise subprocess.TimeoutExpired(cmd=["gcloud", *args], timeout=5)
        raise AssertionError(tuple(args))

    provider = GcloudTpuProvider(runner=runner, delete_wait_timeout_s=0)
    if present_after_timeout:
        with pytest.raises(GcloudError, match="timed out after 5 seconds") as error:
            provider.delete(
                "cell-a-r1",
                project="project-a",
                zone="us-central2-b",
                management_uid=UID,
            )
        assert error.value.code is CapacityProviderFailureCode.PROVIDER_ERROR
    else:
        provider.delete(
            "cell-a-r1",
            project="project-a",
            zone="us-central2-b",
            management_uid=UID,
        )
        assert descriptions == 2
