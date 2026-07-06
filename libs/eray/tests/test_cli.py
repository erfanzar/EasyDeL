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

"""Tests for eray.cli — CLI entry point, click commands, and TPU logic."""

import json
from unittest import mock

import pytest
from click.testing import CliRunner
from eray.cli.main import _resolve_tpu, cli
from eray.cli.tpu import ConnectResult
from eray.cli.utils import (
    TpuInfo,
    build_ray_resource_flags,
    detect_local_tpu,
    list_tpus_in_project,
    list_tpus_in_zone,
    run_on_host,
)

# ── TpuInfo unit tests ───────────────────────────────────────────


class TestTpuInfo:
    def make_tpu(self, acc_type: str = "v4-32", num_hosts: int = 4) -> TpuInfo:
        return TpuInfo(
            name="test-tpu",
            project="test-project",
            zone="us-central2-b",
            accelerator_type=acc_type,
            internal_ips=[f"10.0.0.{i}" for i in range(1, num_hosts + 1)],
            num_hosts=num_hosts,
            state="READY",
        )

    def test_tpu_version(self):
        tpu = self.make_tpu("v4-32")
        assert tpu.tpu_version == "v4"

    def test_slice_size(self):
        tpu = self.make_tpu("v4-32")
        assert tpu.slice_size == "32"

    def test_chips_per_host(self):
        tpu = self.make_tpu("v4-32", num_hosts=4)
        assert tpu.chips_per_host == 8

    def test_is_gcloud_managed_true(self):
        tpu = self.make_tpu()
        assert tpu.is_gcloud_managed is True

    def test_is_gcloud_managed_false_for_from_ips(self):
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v4-16")
        assert tpu.is_gcloud_managed is False

    def test_from_ips(self):
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2", "10.0.0.3"], "v5p-256")
        assert tpu.name is None
        assert tpu.project is None
        assert tpu.zone is None
        assert tpu.accelerator_type == "v5p-256"
        assert tpu.num_hosts == 3
        assert tpu.internal_ips == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]

    def test_from_ips_resource_calculation(self):
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v4-16")
        assert tpu.chips_per_host == 8  # 16 / 2
        assert tpu.tpu_version == "v4"
        assert tpu.slice_size == "16"


class TestBuildRayResourceFlags:
    def test_head_resources(self):
        tpu = TpuInfo(
            name="t",
            project="p",
            zone="z",
            accelerator_type="v4-32",
            internal_ips=["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            num_hosts=4,
            state="READY",
        )
        flags = build_ray_resource_flags(tpu, is_head=True)
        resources = json.loads(flags)
        assert resources["head-node"] == 1
        assert resources["TPU"] == 8
        assert "TPU-v4-32-head" in resources
        assert "accelerator_type:TPU-V4" in resources

    def test_worker_resources_direct_ip(self):
        """Resource flags work identically for direct-IP TpuInfo."""
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v4-16")
        flags = build_ray_resource_flags(tpu, is_head=False)
        resources = json.loads(flags)
        assert "head-node" not in resources
        assert resources["TPU"] == 8  # 16 / 2

    def test_head_has_more_resources_than_worker(self):
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v4-16")
        head = json.loads(build_ray_resource_flags(tpu, is_head=True))
        worker = json.loads(build_ray_resource_flags(tpu, is_head=False))
        assert len(head) > len(worker)


# ─_resolve_tpu tests ─────────────────────────────────────────


class TestResolveTpu:
    def test_no_args_off_tpu_raises(self):
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            with pytest.raises(Exception, match="auto-detected"):
                _resolve_tpu(None, None, None, None, None, None, None)

    def test_no_args_autodetects_on_tpu_vm(self):
        detected = TpuInfo(
            name="my-spot",
            project="proj",
            zone="us-central1-a",
            accelerator_type="v5p-8",
            internal_ips=["10.0.0.9"],
            num_hosts=1,
            state="READY",
        )
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=detected):
            tpu, user, key = _resolve_tpu(None, None, None, None, None, None, None)
        assert tpu is detected
        assert user is None and key is None

    def test_explicit_flags_bypass_autodetection(self):
        # --ips must not consult the metadata server at all.
        with mock.patch("eray.cli.main.detect_local_tpu", side_effect=AssertionError("must not be called")):
            tpu, _, _ = _resolve_tpu(None, None, None, "10.0.0.1", "v4-8", None, None)
        assert tpu.num_hosts == 1

    def test_both_tpu_name_and_ips_raises(self):
        with pytest.raises(Exception, match="mutually exclusive"):
            _resolve_tpu("my-tpu", "p", "z", "10.0.0.1", None, None, None)

    def test_tpu_name_without_project_raises(self):
        with pytest.raises(Exception, match="project"):
            _resolve_tpu("my-tpu", None, "z", None, None, None, None)

    def test_ips_without_type_raises(self):
        with pytest.raises(Exception, match="tpu-type"):
            _resolve_tpu(None, None, None, "10.0.0.1", None, None, None)

    def test_ips_returns_direct_tpuinfo(self):
        tpu, user, key = _resolve_tpu(
            None,
            None,
            None,
            "10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
            "v4-32",
            None,
            None,
        )
        assert tpu.is_gcloud_managed is False
        assert tpu.num_hosts == 4
        assert tpu.accelerator_type == "v4-32"
        assert user is None
        assert key is None

    def test_ips_with_ssh_user(self):
        _tpu, user, key = _resolve_tpu(
            None,
            None,
            None,
            "10.0.0.1",
            "v4-8",
            "myuser",
            "/home/user/.ssh/id_rsa",
        )
        assert user == "myuser"
        assert key == "/home/user/.ssh/id_rsa"


class TestDetectLocalTpu:
    """Metadata parsing for zero-flag `eray tpu connect` on a TPU VM.

    Attribute values mirror a live v5p-8 worker's real metadata."""

    def _detect(self, values):
        with mock.patch("eray.cli.utils._gce_metadata", side_effect=lambda p, timeout=2.0: values.get(p)):
            return detect_local_tpu()

    def test_parses_single_host_tpu_vm(self):
        tpu = self._detect(
            {
                "instance/attributes/accelerator-type": "v5p-8",
                "instance/attributes/worker-network-endpoints": "unknown:unknown:10.128.0.122",
                "instance/attributes/instance-id": "n_server_spot_m",
                "project/project-id": "my-proj",
                "instance/zone": "projects/1056288039276/zones/us-central1-a",
            }
        )
        assert tpu.name == "n_server_spot_m"
        assert tpu.project == "my-proj"
        assert tpu.zone == "us-central1-a"
        assert tpu.accelerator_type == "v5p-8"
        assert tpu.internal_ips == ["10.128.0.122"]
        assert tpu.num_hosts == 1
        assert tpu.is_gcloud_managed

    def test_parses_multi_host_endpoints(self):
        tpu = self._detect(
            {
                "instance/attributes/accelerator-type": "v5p-16",
                "instance/attributes/worker-network-endpoints": ("a:b:10.0.0.1, c:d:10.0.0.2"),
                "instance/attributes/instance-id": "pod",
                "project/project-id": "p",
                "instance/zone": "projects/1/zones/us-east5-a",
            }
        )
        assert tpu.internal_ips == ["10.0.0.1", "10.0.0.2"]
        assert tpu.num_hosts == 2

    def test_not_a_tpu_vm_returns_none(self):
        assert self._detect({}) is None

    def test_missing_endpoints_returns_none(self):
        assert self._detect({"instance/attributes/accelerator-type": "v5p-8"}) is None

    def test_partial_identity_falls_back_to_direct_mode(self):
        tpu = self._detect(
            {
                "instance/attributes/accelerator-type": "v5p-8",
                "instance/attributes/worker-network-endpoints": "x:y:10.0.0.5",
            }
        )
        assert tpu is not None
        assert tpu.is_gcloud_managed is False
        assert tpu.internal_ips == ["10.0.0.5"]


class TestRunOnHostLocalExec:
    def test_local_target_runs_without_ssh(self):
        tpu = TpuInfo.from_ips(["127.0.0.1"], "v5p-8")
        result = run_on_host(tpu, 0, "echo local-$((6 * 7))")
        assert result.returncode == 0
        assert "local-42" in result.stdout


# ── CLI command tests ────────────────────────────────────────────


class TestCliEntryPoint:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "TPU" in result.output
        assert "tpu" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "eray" in result.output.lower()

    def test_tpu_subcommand_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "--help"])
        assert result.exit_code == 0
        assert "connect" in result.output
        assert "disconnect" in result.output


class TestTpuConnect:
    def test_connect_no_args_fails_off_tpu(self):
        # Off a TPU VM (no metadata), zero-arg connect must fail with guidance.
        runner = CliRunner()
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            result = runner.invoke(cli, ["tpu", "connect"])
        assert result.exit_code != 0

    def test_connect_help_shows_both_modes(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "connect", "--help"])
        assert result.exit_code == 0
        assert "--tpu-name" in result.output
        assert "--ips" in result.output
        assert "--tpu-type" in result.output
        assert "--user" in result.output
        assert "--ssh-key" in result.output

    def test_connect_both_modes_rejected(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tpu",
                "connect",
                "--tpu-name",
                "x",
                "--project",
                "p",
                "--zone",
                "z",
                "--ips",
                "10.0.0.1",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_connect_ips_without_type_rejected(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tpu",
                "connect",
                "--ips",
                "10.0.0.1,10.0.0.2",
            ],
        )
        assert result.exit_code != 0
        assert "tpu-type" in result.output.lower()

    def test_connect_ips_without_type_rejected_message(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "tpu",
                "connect",
                "--ips",
                "10.0.0.1",
            ],
        )
        assert result.exit_code != 0


class TestTpuDisconnect:
    def test_disconnect_no_args_fails_off_tpu(self):
        runner = CliRunner()
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            result = runner.invoke(cli, ["tpu", "disconnect"])
        assert result.exit_code != 0

    def test_disconnect_help_shows_both_modes(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "disconnect", "--help"])
        assert result.exit_code == 0
        assert "--tpu-name" in result.output
        assert "--ips" in result.output


class TestTpuStatus:
    def test_status_requires_address_off_tpu(self):
        # Off a TPU VM there is nothing to auto-detect; must fail fast, not
        # attempt a connection.
        runner = CliRunner()
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            result = runner.invoke(cli, ["tpu", "status"])
        assert result.exit_code != 0

    def test_status_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "status", "--help"])
        assert result.exit_code == 0
        assert "--address" in result.output


class TestTpuHealth:
    def test_health_requires_address_off_tpu(self):
        runner = CliRunner()
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            result = runner.invoke(cli, ["tpu", "health"])
        assert result.exit_code != 0

    def test_health_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "health", "--help"])
        assert result.exit_code == 0
        assert "--address" in result.output


class TestTpuList:
    def test_list_help_shows_all_zones_as_default(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["tpu", "list", "--help"])
        assert result.exit_code == 0
        assert "--project" in result.output
        assert "--zone" in result.output
        # all-zones is now the default behavior, documented in help
        assert "all zones" in result.output.lower()

    def test_list_no_args_fails_without_gcloud(self):
        """Without gcloud, it should fail cleanly."""
        runner = CliRunner()
        with mock.patch("eray.cli.main.check_gcloud", return_value=False):
            result = runner.invoke(cli, ["tpu", "list"])
            assert result.exit_code != 0


class TestListTpuHelpers:
    def test_list_tpus_in_zone_is_importable(self):
        assert callable(list_tpus_in_zone)

    def test_list_tpus_in_project_is_importable(self):
        assert callable(list_tpus_in_project)

    def test_list_tpus_in_project_enriches_zone(self):
        """Zone should be extracted from the gcloud name path."""
        import eray.cli.utils as utils

        original_gcloud_json = utils.gcloud_json

        def mock_gcloud_json(args, **kwargs):
            return [
                {
                    "name": "projects/my-project/locations/us-central2-b/tpus/my-tpu-1",
                    "state": "READY",
                    "acceleratorType": "projects/my-project/locations/us-central2-b/acceleratorTypes/v4-32",
                    "health": "HEALTHY",
                    "networkEndpoints": [{"ipAddress": "10.0.0.1"}, {"ipAddress": "10.0.0.2"}],
                },
                {
                    "name": "projects/my-project/locations/europe-west4-a/tpus/my-tpu-2",
                    "state": "CREATING",
                    "acceleratorType": "v5p-128",
                    "health": "UNHEALTHY",
                },
            ]

        utils.gcloud_json = mock_gcloud_json
        try:
            result = list_tpus_in_project("my-project")
            assert len(result) == 2
            assert result[0]["zone"] == "us-central2-b"
            assert result[1]["zone"] == "europe-west4-a"
        finally:
            utils.gcloud_json = original_gcloud_json


# ── ConnectResult dataclass ──────────────────────────────────────


class TestConnectResult:
    def test_construction(self):
        tpu = TpuInfo.from_ips(["10.0.0.1"], "v4-8")
        result = ConnectResult(
            tpu=tpu,
            head_ip="10.0.0.1",
            ray_address="10.0.0.1:6379",
            dashboard_url="http://10.0.0.1:8265",
            num_hosts=1,
        )
        assert result.ray_address == "10.0.0.1:6379"
        assert result.num_hosts == 1
