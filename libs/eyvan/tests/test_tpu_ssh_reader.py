# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reading a TPU node through one of its own hosts, with a recorded ssh runner."""

from __future__ import annotations

import json
import subprocess

import pytest

from eyvan._internal.ssh import SshDestination, SshOptions
from eyvan.capacity.providers.gcp.tpu_nodes import TpuApiError, TpuNodeParseError
from eyvan.capacity.providers.gcp.tpu_ssh import SshTpuNodeReader, parse_host_identity

# Representative v5p-128 worker metadata with neutral identifiers.
_TPU_ENV = """project=other-project
preemptible=FALSE
ACCELERATOR_TYPE: 'v5p-128'
CHIPS_PER_HOST_BOUNDS: '2,2,1'
NODE_ID: 'shared_v5p'
PREEMPTIBLE: 'false'
WORKER_ID: '3'
ZONE: 'europe-west4-b'
"""

_DOCUMENT = {
    "name": "projects/other-project/locations/europe-west4-b/nodes/shared_v5p",
    "id": "999",
    "acceleratorType": "v5p-128",
    "state": "READY",
    "health": "HEALTHY",
    "schedulingConfig": {"reserved": True},
    "networkEndpoints": [{"ipAddress": "10.1.0.2", "accessConfig": {"externalIp": "198.51.100.2"}}],
}


def test_identity_parses_tpu_env():
    identity = parse_host_identity(_TPU_ENV)
    assert identity is not None
    assert (identity.project, identity.zone, identity.name) == ("other-project", "europe-west4-b", "shared_v5p")
    assert identity.accelerator_type == "v5p-128" and identity.worker_id == 3 and identity.preemptible is False


def test_a_host_without_tpu_env_is_not_a_tpu():
    assert parse_host_identity("project=p\npreemptible=FALSE\n") is None


class Runner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args, *, stdin_text, timeout):
        assert stdin_text is None and timeout > 0
        self.calls.append(tuple(args))
        code, out, err = self.results.pop(0)
        return subprocess.CompletedProcess(args, code, out, err)


def _reader(runner, **kwargs):
    return SshTpuNodeReader(
        SshDestination.parse("me@198.51.100.2"), options=SshOptions(), runner=runner, clock=lambda: 0.0, **kwargs
    )


def test_get_node_uses_the_hosts_gcloud_and_caches():
    runner = Runner((0, json.dumps(_DOCUMENT), ""))
    reader = _reader(runner)
    node = reader.get_node("other-project", "europe-west4-b", "shared_v5p")
    assert node is not None and node.accelerator_type == "v5p-128" and node.capacity_class.value == "reserved"
    assert reader.get_node("other-project", "europe-west4-b", "shared_v5p") == node
    [argv] = runner.calls
    assert argv[0] == "ssh" and argv[-2] == "me@198.51.100.2"
    assert "gcloud compute tpus tpu-vm describe shared_v5p" in argv[-1] and "--format=json" in argv[-1]


def test_not_found_is_none_and_other_errors_are_classified():
    assert _reader(Runner((1, "", "ERROR: (gcloud) NOT_FOUND: Resource not found"))).get_node("p", "z-a-b", "n") is None
    with pytest.raises(TpuApiError) as unreachable:
        _reader(Runner((255, "", "ssh: connect to host port 22: Operation timed out"))).get_node("p", "z-a-b", "n")
    assert unreachable.value.retryable
    with pytest.raises(TpuApiError) as denied:
        _reader(Runner((1, "", "ERROR: (gcloud) PERMISSION_DENIED"))).get_node("p", "z-a-b", "n")
    assert not denied.value.retryable
    with pytest.raises(TpuApiError, match="not installed"):
        _reader(Runner((127, "", "gcloud: command not found"))).get_node("p", "z-a-b", "n")
    with pytest.raises(TpuNodeParseError):
        _reader(Runner((0, "not json", ""))).get_node("p", "z-a-b", "n")


def test_identity_failure_paths():
    with pytest.raises(TpuApiError) as unreachable:
        _reader(Runner((255, "", "Permission denied (publickey)."))).identity()
    assert unreachable.value.retryable
    assert _reader(Runner((0, "project=p\n", ""))).identity() is None
