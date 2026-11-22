# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for eray.provision.qr — queued-resource argv contract and lifecycle."""

from unittest import mock

import eray.provision.qr as qr_module
import pytest
from click.testing import CliRunner
from eray.cli.main import cli
from eray.provision.qr import (
    QrSpec,
    default_runtime_version,
    describe_queued_resource,
    qr_create_args,
    wait_for_active,
)


def make_spec(**overrides) -> QrSpec:
    defaults = dict(name="my-tpu", accelerator_type="v5p-8", zone="us-central1-a", project="proj")
    defaults.update(overrides)
    return QrSpec(**defaults)


class TestRuntimeVersionMap:
    def test_generation_defaults(self):
        assert default_runtime_version("v5p-64") == "v2-alpha-tpuv5"
        assert default_runtime_version("v5litepod-16") == "v2-alpha-tpuv5-lite"
        assert default_runtime_version("v4-8") == "tpu-ubuntu2204-base"
        assert default_runtime_version("v6e-16") == "v2-alpha-tpuv6e"

    def test_unknown_family_requires_explicit(self):
        with pytest.raises(ValueError, match="runtime_version explicitly"):
            default_runtime_version("v9x-8")

    def test_explicit_wins(self):
        spec = make_spec(runtime_version="custom-rv")
        assert spec.resolved_runtime_version() == "custom-rv"


class TestCreateArgv:
    """The exact gcloud argv is the contract — validated against
    `gcloud compute tpus queued-resources create --help` (see qr.py)."""

    def test_spot_default(self):
        args = qr_create_args(make_spec())
        assert args[:5] == ["compute", "tpus", "queued-resources", "create", "my-tpu"]
        assert "--spot" in args
        assert args[args.index("--zone") + 1] == "us-central1-a"
        assert args[args.index("--project") + 1] == "proj"
        assert args[args.index("--accelerator-type") + 1] == "v5p-8"
        assert args[args.index("--runtime-version") + 1] == "v2-alpha-tpuv5"
        # single-node: stable node id == name
        assert args[args.index("--node-id") + 1] == "my-tpu"

    def test_capacity_tiers(self):
        assert "--reserved" in qr_create_args(make_spec(capacity="reserved"))
        assert "--guaranteed" in qr_create_args(make_spec(capacity="guaranteed"))
        on_demand = qr_create_args(make_spec(capacity="on-demand"))
        assert not ({"--spot", "--reserved", "--guaranteed"} & set(on_demand))

    def test_qr_id_override_keeps_node_id_stable(self):
        # Watchers create {name}-r{gen} QRs but the node keeps the plain name,
        # so `eray tpu connect -n NAME` works across regenerations.
        args = qr_create_args(make_spec(), qr_id="my-tpu-r3")
        assert args[4] == "my-tpu-r3"
        assert args[args.index("--node-id") + 1] == "my-tpu"

    def test_node_count_excludes_node_id(self):
        args = qr_create_args(make_spec(node_count=4))
        assert args[args.index("--node-count") + 1] == "4"
        assert "--node-id" not in args

    def test_labels_metadata_and_expiry(self):
        args = qr_create_args(
            make_spec(
                labels={"eray-owner": "erfan", "eray-cluster": "my-tpu"},
                metadata={"k": "v"},
                valid_until_duration="72h",
            )
        )
        assert args[args.index("--labels") + 1] == "eray-cluster=my-tpu,eray-owner=erfan"
        assert args[args.index("--metadata") + 1] == "k=v"
        assert (
            args[
                args.index("--valid-until") + 1 if "--valid-until" in args else args.index("--valid-until-duration") + 1
            ]
            == "72h"
        )


def qr_payload(state: str, qr_id: str = "my-tpu", node_id: str = "my-tpu", acc: str = "v5p-8") -> dict:
    return {
        "name": f"projects/proj/locations/us-central1-a/queuedResources/{qr_id}",
        "state": {"state": state},
        "tpu": {"nodeSpec": [{"nodeId": node_id, "node": {"acceleratorType": acc}}]},
    }


class TestDescribeParsing:
    def test_parses_nested_state_and_nodes(self, monkeypatch):
        monkeypatch.setattr(qr_module, "gcloud_json", lambda a, **k: qr_payload("ACTIVE"))
        qr = describe_queued_resource("my-tpu", project="proj", zone="us-central1-a")
        assert qr.state == "ACTIVE" and qr.is_active and not qr.is_terminal
        assert qr.qr_id == "my-tpu"
        assert qr.node_ids == ("my-tpu",)
        assert qr.accelerator_type == "v5p-8"

    def test_not_found_returns_none(self, monkeypatch):
        import subprocess

        def boom(args, **k):
            raise subprocess.CalledProcessError(1, "gcloud", stderr="ERROR: NOT_FOUND: ...")

        monkeypatch.setattr(qr_module, "gcloud_json", boom)
        assert describe_queued_resource("gone", project="proj", zone="z") is None

    def test_other_errors_propagate(self, monkeypatch):
        import subprocess

        def boom(args, **k):
            raise subprocess.CalledProcessError(1, "gcloud", stderr="ERROR: PERMISSION_DENIED")

        monkeypatch.setattr(qr_module, "gcloud_json", boom)
        with pytest.raises(subprocess.CalledProcessError):
            describe_queued_resource("x", project="proj", zone="z")

    def test_flat_state_accepted(self, monkeypatch):
        payload = qr_payload("ACTIVE")
        payload["state"] = "ACTIVE"
        monkeypatch.setattr(qr_module, "gcloud_json", lambda a, **k: payload)
        qr = describe_queued_resource("my-tpu", project="proj", zone="us-central1-a")
        assert qr.state == "ACTIVE"


class TestWaitForActive:
    def _sequence(self, monkeypatch, states):
        seq = iter(states)

        def fake_describe(qr_id, *, project, zone):
            state = next(seq)
            if state is None:
                return None
            return qr_module._parse_qr(qr_payload(state), project=project, zone=zone)

        monkeypatch.setattr(qr_module, "describe_queued_resource", fake_describe)
        monkeypatch.setattr(qr_module.time, "sleep", lambda s: None)

    def test_happy_path_reports_state_changes(self, monkeypatch):
        self._sequence(monkeypatch, ["WAITING_FOR_RESOURCES", "WAITING_FOR_RESOURCES", "PROVISIONING", "ACTIVE"])
        seen = []
        qr = wait_for_active("my-tpu", project="p", zone="z", on_state=seen.append)
        assert qr.is_active
        assert seen == ["WAITING_FOR_RESOURCES", "PROVISIONING", "ACTIVE"]  # deduped changes only

    def test_terminal_state_raises(self, monkeypatch):
        self._sequence(monkeypatch, ["WAITING_FOR_RESOURCES", "FAILED"])
        with pytest.raises(RuntimeError, match="terminal state FAILED"):
            wait_for_active("my-tpu", project="p", zone="z")

    def test_suspended_raises(self, monkeypatch):
        self._sequence(monkeypatch, ["SUSPENDED"])
        with pytest.raises(RuntimeError, match="SUSPENDED"):
            wait_for_active("my-tpu", project="p", zone="z")

    def test_disappearance_raises(self, monkeypatch):
        self._sequence(monkeypatch, ["WAITING_FOR_RESOURCES", None])
        with pytest.raises(RuntimeError, match="disappeared"):
            wait_for_active("my-tpu", project="p", zone="z")

    def test_timeout_raises(self, monkeypatch):
        self._sequence(monkeypatch, ["WAITING_FOR_RESOURCES"] * 50)
        clock = iter(range(0, 5000, 100))
        monkeypatch.setattr(qr_module.time, "monotonic", lambda: next(clock))
        with pytest.raises(RuntimeError, match="not ACTIVE after"):
            wait_for_active("my-tpu", project="p", zone="z", timeout=300)


class TestQrCli:
    def test_group_registered(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["qr", "--help"])
        assert result.exit_code == 0
        for cmd in ("create", "list", "status", "delete", "wait"):
            assert cmd in result.output

    def test_create_invokes_gcloud_with_golden_argv(self):
        captured = {}

        def fake_gcloud(args, **k):
            captured["args"] = args
            return ""

        runner = CliRunner()
        with (
            mock.patch.object(qr_module, "gcloud", fake_gcloud),
            mock.patch.object(qr_module, "gcloud_json", lambda a, **k: qr_payload("WAITING_FOR_RESOURCES")),
        ):
            result = runner.invoke(
                cli,
                ["qr", "create", "my-tpu", "--type", "v5p-8", "--zone", "us-central1-a", "--project", "proj"],
            )
        assert result.exit_code == 0, result.output
        assert "--spot" in captured["args"]
        assert captured["args"][4] == "my-tpu"

    def test_status_not_found_fails(self):
        runner = CliRunner()
        with mock.patch("eray.cli.qr.describe_queued_resource", return_value=None):
            result = runner.invoke(cli, ["qr", "status", "gone", "--zone", "z", "--project", "p"])
        assert result.exit_code != 0
        assert "not found" in result.output
