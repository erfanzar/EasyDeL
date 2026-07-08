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
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
from click.testing import CliRunner
from eray.cli.main import _resolve_address, _resolve_tpu, cli
from eray.cli.tpu import ConnectResult, _ray_bin_preamble, resource_usage
from eray.cli.utils import (
    TpuInfo,
    build_ray_resource_flags,
    detect_local_tpu,
    format_ago,
    format_duration,
    list_tpus_in_project,
    list_tpus_in_zone,
    parse_gcp_timestamp,
    print_table,
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

    def test_chips_per_host_counts_chips_not_cores(self):
        # v4 slice suffixes count TensorCores (2 per chip): v4-32 = 16 chips
        # over 4 hosts. Advertising cores as "TPU" makes Ray assign
        # nonexistent chip ids to fractional-TPU tasks.
        tpu = self.make_tpu("v4-32", num_hosts=4)
        assert tpu.chips_per_host == 4

    def test_chips_per_host_v5p_matches_ray_native_detection(self):
        # Live-anchored: a v5p-8 host exposes 4 chips (/dev/vfio/0-3) and
        # Ray-native detection reports TPU=4.
        tpu = self.make_tpu("v5p-8", num_hosts=1)
        assert tpu.chips_per_host == 4

    def test_chips_per_host_single_core_generations(self):
        # v5e/v6e suffixes already count chips (one TensorCore per chip).
        assert self.make_tpu("v5litepod-8", num_hosts=1).chips_per_host == 8
        assert self.make_tpu("v6e-16", num_hosts=4).chips_per_host == 4

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
        assert tpu.chips_per_host == 4  # 16 cores / 2 cores-per-chip / 2 hosts
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
        assert resources["TPU"] == 4  # v4-32 = 16 chips over 4 hosts
        assert "TPU-v4-32-head" in resources
        assert "accelerator_type:TPU-V4" in resources

    def test_worker_resources_direct_ip(self):
        """Resource flags work identically for direct-IP TpuInfo."""
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v4-16")
        flags = build_ray_resource_flags(tpu, is_head=False)
        resources = json.loads(flags)
        assert "head-node" not in resources
        assert resources["TPU"] == 4  # 16 cores / 2 cores-per-chip / 2 hosts

    def test_head_has_more_resources_than_worker(self):
        tpu = TpuInfo.from_ips(["10.0.0.1", "10.0.0.2"], "v4-16")
        head = json.loads(build_ray_resource_flags(tpu, is_head=True))
        worker = json.loads(build_ray_resource_flags(tpu, is_head=False))
        assert len(head) > len(worker)


class TestFormatHelpers:
    def test_format_duration_coarsest_two_units(self):
        assert format_duration(45) == "45s"
        assert format_duration(90) == "1m"
        assert format_duration(3661) == "1h1m"
        assert format_duration(90000) == "1d1h"

    def test_format_duration_clamps_negative(self):
        assert format_duration(-5) == "0s"

    def test_format_ago_none_passthrough(self):
        assert format_ago(None) is None

    def test_format_ago_renders_a_duration(self):
        import time as time_module

        assert format_ago(time_module.time() - 90) == "1m ago"

    def test_parse_gcp_timestamp_none_and_empty(self):
        assert parse_gcp_timestamp(None) is None
        assert parse_gcp_timestamp("") is None

    def test_parse_gcp_timestamp_unparsable(self):
        assert parse_gcp_timestamp("not-a-timestamp") is None

    def test_parse_gcp_timestamp_parses_rfc3339(self):
        ts = parse_gcp_timestamp("2026-07-07T18:24:38.123-07:00")
        assert ts is not None
        assert ts > 0


class TestPrintTable:
    def test_empty_rows_is_a_noop(self, capsys):
        print_table([])
        assert capsys.readouterr().out == ""

    def test_renders_header_and_aligned_rows(self, capsys):
        print_table([{"name": "a", "status": "UP"}, {"name": "bb", "status": "DOWN"}])
        out = capsys.readouterr().out.splitlines()
        assert out[0].split() == ["NAME", "STATUS"]
        assert "a" in out[1] and "UP" in out[1]
        assert "bb" in out[2] and "DOWN" in out[2]


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


class TestRayBinPreamble:
    """The preamble is a shell contract executed on remote hosts through
    non-interactive SSH (where venv PATH entries are absent), so these tests
    run the generated snippet under bash with a controlled PATH/HOME and
    assert what $RAY_BIN actually resolves to."""

    def _run(self, preamble: str, *, path: str, home: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["/bin/bash", "-c", f'{preamble} && echo "resolved:$RAY_BIN"'],
            capture_output=True,
            text=True,
            env={"PATH": path, "HOME": home},
            timeout=10,
        )

    def _fake_ray(self, directory) -> str:
        directory.mkdir(parents=True, exist_ok=True)
        ray = directory / "ray"
        ray.write_text("#!/bin/sh\n")
        ray.chmod(0o755)
        return str(ray)

    def test_default_prefers_driver_venv_ray(self, tmp_path):
        venv_ray = self._fake_ray(tmp_path / "venv-bin")
        with mock.patch.object(sys, "executable", str(tmp_path / "venv-bin" / "python")):
            preamble = _ray_bin_preamble("ray", strict=True)
        result = self._run(preamble, path=str(tmp_path / "empty"), home=str(tmp_path / "home"))
        assert result.returncode == 0
        assert f"resolved:{venv_ray}" in result.stdout

    def test_default_falls_back_to_path_lookup(self, tmp_path):
        path_ray = self._fake_ray(tmp_path / "on-path")
        with mock.patch.object(sys, "executable", str(tmp_path / "no-venv" / "python")):
            preamble = _ray_bin_preamble("ray", strict=True)
        result = self._run(preamble, path=str(tmp_path / "on-path"), home=str(tmp_path / "home"))
        assert result.returncode == 0
        assert f"resolved:{path_ray}" in result.stdout

    def test_default_falls_back_to_user_local_bin(self, tmp_path):
        home = tmp_path / "home"
        local_ray = self._fake_ray(home / ".local" / "bin")
        with mock.patch.object(sys, "executable", str(tmp_path / "no-venv" / "python")):
            preamble = _ray_bin_preamble("ray", strict=True)
        result = self._run(preamble, path=str(tmp_path / "empty"), home=str(home))
        assert result.returncode == 0
        assert f"resolved:{local_ray}" in result.stdout

    def test_strict_fails_with_diagnostic_when_unresolvable(self, tmp_path):
        # The v5p-1024 regression: bare `ray` in a PATH-less SSH shell must
        # produce an actionable error, not `ray: command not found`.
        with mock.patch.object(sys, "executable", str(tmp_path / "no-venv" / "python")):
            preamble = _ray_bin_preamble("ray", strict=True)
        result = self._run(preamble, path=str(tmp_path / "empty"), home=str(tmp_path / "home"))
        assert result.returncode == 127
        assert "ray executable not found" in result.stderr
        assert "--ray-bin" in result.stderr

    def test_soft_mode_falls_back_without_failing(self, tmp_path):
        with mock.patch.object(sys, "executable", str(tmp_path / "no-venv" / "python")):
            preamble = _ray_bin_preamble("ray", strict=False)
        result = self._run(preamble, path=str(tmp_path / "empty"), home=str(tmp_path / "home"))
        assert result.returncode == 0
        assert "resolved:ray" in result.stdout

    def test_explicit_ray_bin_is_sole_candidate(self, tmp_path):
        explicit = self._fake_ray(tmp_path / "custom")
        decoy = self._fake_ray(tmp_path / "on-path")
        preamble = _ray_bin_preamble(explicit, strict=True)
        result = self._run(preamble, path=str(tmp_path / "on-path"), home=str(tmp_path / "home"))
        assert result.returncode == 0
        assert f"resolved:{explicit}" in result.stdout
        assert decoy not in result.stdout


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


class TestResolveAddress:
    """`_resolve_address` (for `eray resources`/`tpu status`/`tpu health`)
    auto-detects the GCS address from an open eray tunnel, so a laptop
    needs no `-a`. The tunnel store is isolated per-test by conftest."""

    def test_explicit_address_wins(self):
        assert _resolve_address("10.0.0.1:6379") == "10.0.0.1:6379"

    def test_tpu_metadata_beats_a_tunnel(self):
        import eray.provision.tunnel as tunnel_module

        tpu = TpuInfo.from_ips(["10.0.0.9"], "v5p-8")
        tunnel_module.open_tunnel(
            "c1", [sys.executable, "-c", "import time; time.sleep(30)"], kind="qr", local_port=51000, remote_port=6379
        )
        try:
            with mock.patch("eray.cli.main.detect_local_tpu", return_value=tpu):
                assert _resolve_address(None) == "10.0.0.9:6379"
        finally:
            tunnel_module.stop_tunnel("c1")

    def test_auto_detects_a_lone_gcs_tunnel(self):
        import eray.provision.tunnel as tunnel_module

        # local port isn't 6379 (6379 was busy) — must use the actual port
        tunnel_module.open_tunnel(
            "c1-gcs",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            kind="qr",
            local_port=51000,
            remote_port=6379,
        )
        try:
            with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
                assert _resolve_address(None) == "127.0.0.1:51000"
        finally:
            tunnel_module.stop_tunnel("c1-gcs")

    def test_a_dashboard_tunnel_is_not_a_gcs_tunnel(self):
        import eray.provision.tunnel as tunnel_module

        # only a dashboard tunnel (8265) open — no GCS address to resolve
        tunnel_module.open_tunnel(
            "c1", [sys.executable, "-c", "import time; time.sleep(30)"], kind="qr", local_port=55084, remote_port=8265
        )
        try:
            with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
                with pytest.raises(Exception, match="tunnel"):
                    _resolve_address(None)
        finally:
            tunnel_module.stop_tunnel("c1")

    def test_multiple_gcs_tunnels_are_ambiguous(self):
        import eray.provision.tunnel as tunnel_module

        for name, port in (("a-gcs", 51000), ("b-gcs", 51001)):
            tunnel_module.open_tunnel(
                name,
                [sys.executable, "-c", "import time; time.sleep(30)"],
                kind="qr",
                local_port=port,
                remote_port=6379,
            )
        try:
            with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
                with pytest.raises(Exception, match="multiple open GCS tunnels"):
                    _resolve_address(None)
        finally:
            tunnel_module.stop_tunnel("a-gcs")
            tunnel_module.stop_tunnel("b-gcs")

    def test_no_tunnel_off_tpu_errors_with_guidance(self):
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            with pytest.raises(Exception, match="open a tunnel first"):
                _resolve_address(None)


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


class TestRequireReachableHead:
    def test_dead_head_fails_fast_with_guidance(self):
        # Regression: ray.init against a dead head burns ~107s in GCS
        # retries; the read-only commands must fail in ~2s instead.
        import time

        from eray.cli.tpu import _require_reachable_head

        start = time.monotonic()
        with pytest.raises(RuntimeError, match="eray tpu connect"):
            _require_reachable_head("127.0.0.1:1")
        assert time.monotonic() - start < 5

    def test_listening_head_passes(self):
        import socket

        from eray.cli.tpu import _require_reachable_head

        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]
            _require_reachable_head(f"127.0.0.1:{port}")  # no raise


def _ru(name, total, used):
    """A fake autoscaler ResourceUsage."""
    return SimpleNamespace(resource_name=name, total=total, used=used)


def _node(ip, node_id, usages):
    """A fake autoscaler NodeInfo with a resource_usage.usage list."""
    return SimpleNamespace(ip_address=ip, node_id=node_id, resource_usage=SimpleNamespace(usage=usages))


def _cluster_status(cluster_usage, *, active=None, idle=None, pending=None):
    """A fake autoscaler ClusterStatus."""
    return SimpleNamespace(
        cluster_resource_usage=cluster_usage,
        active_nodes=active or [],
        idle_nodes=idle or [],
        pending_nodes=pending or [],
    )


class TestResourceUsage:
    """`resource_usage` reads the autoscaler status (used/total per resource)
    over GCS — no `ray.init` driver handshake (which can't complete over a
    single tunneled port). Falls back to the dashboard state API (totals
    only) on non-autoscaler clusters."""

    def test_used_and_available_from_autoscaler(self):
        status = _cluster_status([_ru("CPU", 100.0, 40.0), _ru("TPU", 8.0, 4.0)])
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=status),
        ):
            result = resource_usage("10.0.0.1:6379")
        assert result["resources"]["CPU"] == {"total": 100.0, "available": 60.0, "used": 40.0}
        assert result["resources"]["TPU"]["used"] == 4.0

    def test_fully_consumed_resource_reads_zero_available(self):
        status = _cluster_status([_ru("TPU", 8.0, 8.0)])
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=status),
        ):
            result = resource_usage("10.0.0.1:6379")
        assert result["resources"]["TPU"] == {"total": 8.0, "available": 0.0, "used": 8.0}

    def test_per_node_breakdown_excludes_markers_and_sorts_by_ip(self):
        status = _cluster_status(
            [_ru("TPU", 8.0, 4.0)],
            active=[
                _node("10.0.0.10", "aa", [_ru("TPU", 4.0, 4.0), _ru("CPU", 8.0, 0.0), _ru("node:10.0.0.10", 1.0, 0.0)])
            ],
            idle=[_node("10.0.0.2", "bb", [_ru("TPU", 4.0, 0.0), _ru("CPU", 8.0, 0.0), _ru("node:10.0.0.2", 1.0, 0.0)])],
        )
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=status),
        ):
            result = resource_usage("10.0.0.1:6379", per_node=True)
        ips = [n["ip"] for n in result["nodes"]]
        assert ips == ["10.0.0.2", "10.0.0.10"]  # numeric IP order (active + idle both alive)
        busy = next(n for n in result["nodes"] if n["ip"] == "10.0.0.10")
        assert busy["resources"]["TPU"]["used"] == 4.0
        assert not any(k.startswith("node:") for n in result["nodes"] for k in n["resources"])

    def test_dashboard_fallback_when_no_autoscaler(self):
        # A connect-mode/QR cluster has no autoscaler; fall back to the
        # dashboard state API, which has totals + liveness but no usage.
        rows = [
            {
                "state": "ALIVE",
                "node_ip": "10.0.0.1",
                "node_id": "aa",
                "resources_total": {"TPU": 4.0, "node:10.0.0.1": 1.0},
            },
            {"state": "DEAD", "node_ip": "10.0.0.9", "node_id": "zz", "resources_total": {"TPU": 4.0}},
        ]
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", side_effect=RuntimeError("no autoscaler")),
            mock.patch("eray.cli.tpu._dashboard_nodes", return_value=rows),
        ):
            result = resource_usage("10.0.0.1:6379", per_node=True)
        assert result["resources"]["TPU"] == {"total": 4.0, "available": None, "used": None}
        assert [n["ip"] for n in result["nodes"]] == ["10.0.0.1"]  # dead node dropped

    def test_empty_autoscaler_status_falls_back_to_dashboard(self):
        # A non-autoscaler cluster can answer the GCS read with an empty
        # status instead of raising; that must not shadow the dashboard
        # fallback and leave `eray resources` showing an empty table.
        rows = [{"state": "ALIVE", "node_ip": "10.0.0.1", "node_id": "aa", "resources_total": {"TPU": 4.0}}]
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=_cluster_status([])),
            mock.patch("eray.cli.tpu._dashboard_nodes", return_value=rows),
        ):
            result = resource_usage("10.0.0.1:6379")
        assert result["resources"]["TPU"]["total"] == 4.0

    def test_both_sources_fail_raises_a_readable_error(self):
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", side_effect=RuntimeError("no autoscaler")),
            mock.patch("eray.cli.tpu._dashboard_nodes", side_effect=RuntimeError("dashboard down")),
        ):
            with pytest.raises(RuntimeError, match=r"could not read resource usage.*autoscaler.*dashboard"):
                resource_usage("10.0.0.1:6379")


class TestClusterStatus:
    def test_summary_from_autoscaler(self):
        from eray.cli.tpu import cluster_status

        status = _cluster_status(
            [_ru("TPU", 8.0, 4.0), _ru("memory", 100.0, 10.0)],
            active=[_node("10.0.0.1", "aa", [])],
            idle=[_node("10.0.0.2", "bb", [])],
        )
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=status),
        ):
            result = cluster_status("10.0.0.1:6379")
        assert result["alive_nodes"] == 2
        assert result["resources"] == {"TPU": 8.0, "memory": 100.0}
        assert set(result["node_ips"]) == {"10.0.0.1", "10.0.0.2"}
        assert result["dashboard_url"] == "http://10.0.0.1:8265"

    def test_total_counts_pending_and_failed_nodes(self):
        # A crashed node stays in the total (via failed_nodes) so `Alive <
        # Total` still flags the loss, matching the dashboard fallback's count.
        from eray.cli.tpu import cluster_status

        status = _cluster_status(
            [_ru("TPU", 8.0, 4.0)], active=[_node("10.0.0.1", "aa", [])], pending=[SimpleNamespace()]
        )
        status.failed_nodes = [SimpleNamespace()]
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=status),
        ):
            result = cluster_status("10.0.0.1:6379")
        assert result["alive_nodes"] == 1
        assert result["total_nodes"] == 3  # 1 alive + 1 pending + 1 failed

    def test_dashboard_fallback(self):
        from eray.cli.tpu import cluster_status

        rows = [
            {"state": "ALIVE", "node_ip": "10.0.0.1", "resources_total": {"TPU": 4.0}},
            {"state": "DEAD", "node_ip": "10.0.0.9", "resources_total": {"TPU": 4.0}},
        ]
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", side_effect=RuntimeError("no autoscaler")),
            mock.patch("eray.cli.tpu._dashboard_nodes", return_value=rows),
        ):
            result = cluster_status("10.0.0.1:6379")
        assert result["alive_nodes"] == 1
        assert result["resources"] == {"TPU": 4.0}
        assert result["node_ips"] == ["10.0.0.1"]

    def test_empty_autoscaler_status_falls_back_to_dashboard(self):
        from eray.cli.tpu import cluster_status

        rows = [{"state": "ALIVE", "node_ip": "10.0.0.1", "resources_total": {"TPU": 4.0}}]
        with (
            mock.patch("eray.cli.tpu._require_reachable_head", lambda a, **k: None),
            mock.patch("eray.cli.tpu._autoscaler_cluster_status", return_value=_cluster_status([])),
            mock.patch("eray.cli.tpu._dashboard_nodes", return_value=rows),
        ):
            result = cluster_status("10.0.0.1:6379")
        assert result["alive_nodes"] == 1
        assert result["resources"] == {"TPU": 4.0}


class TestDashboardPortResolution:
    """The dashboard state-API fallback must query the dashboard tunnel's
    actual local port, not a hardcoded 8265 (which is wrong when 8265 was
    busy and the tunnel landed elsewhere)."""

    def test_loopback_resolves_open_tunnel_local_port(self):
        from eray.cli.tpu import _dashboard_port_for

        fake = SimpleNamespace(local_port=18265)
        with mock.patch("eray.provision.tunnel.tunnels_for_remote_port", return_value=[fake]):
            assert _dashboard_port_for("127.0.0.1") == 18265

    def test_loopback_with_ambiguous_tunnels_uses_default(self):
        from eray.cli.tpu import RAY_DASHBOARD_PORT, _dashboard_port_for

        two = [SimpleNamespace(local_port=18265), SimpleNamespace(local_port=28265)]
        with mock.patch("eray.provision.tunnel.tunnels_for_remote_port", return_value=two):
            assert _dashboard_port_for("127.0.0.1") == RAY_DASHBOARD_PORT

    def test_direct_ip_uses_default_port(self):
        from eray.cli.tpu import RAY_DASHBOARD_PORT, _dashboard_port_for

        assert _dashboard_port_for("10.0.0.5") == RAY_DASHBOARD_PORT


class TestExcStr:
    def test_empty_exception_str_falls_back_to_type_name(self):
        from eray.cli.tpu import _exc_str

        assert _exc_str(ConnectionError()) == "ConnectionError"
        assert _exc_str(RuntimeError("boom")) == "boom"


class TestResourcesCommand:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["resources", "--help"])
        assert result.exit_code == 0
        assert "--per-node" in result.output
        assert "--address" in result.output

    def test_requires_address_off_tpu(self):
        runner = CliRunner()
        with mock.patch("eray.cli.main.detect_local_tpu", return_value=None):
            result = runner.invoke(cli, ["resources"])
        assert result.exit_code != 0

    def test_table_shows_usage_and_hides_node_markers(self):
        usage = {
            "resources": {
                "CPU": {"total": 200.0, "available": 150.0, "used": 50.0},
                "TPU": {"total": 8.0, "available": 0.0, "used": 8.0},
                "memory": {"total": 8.0 * 1024**3, "available": 6.0 * 1024**3, "used": 2.0 * 1024**3},
                "node:10.0.0.1": {"total": 1.0, "available": 1.0, "used": 0.0},
            }
        }
        runner = CliRunner()
        with mock.patch("eray.cli.main.resource_usage", return_value=usage):
            result = runner.invoke(cli, ["resources", "-a", "10.0.0.1:6379"])
        assert result.exit_code == 0
        assert "node:10.0.0.1" not in result.output
        cpu_line = next(line for line in result.output.splitlines() if line.startswith("CPU"))
        assert "50" in cpu_line and "200" in cpu_line and "25.0%" in cpu_line
        tpu_line = next(line for line in result.output.splitlines() if line.startswith("TPU"))
        assert "100.0%" in tpu_line
        mem_line = next(line for line in result.output.splitlines() if line.startswith("memory"))
        assert "2.0GiB" in mem_line and "8.0GiB" in mem_line

    def test_json_output_roundtrips(self):
        usage = {"resources": {"TPU": {"total": 8.0, "available": 4.0, "used": 4.0}}}
        runner = CliRunner()
        with mock.patch("eray.cli.main.resource_usage", return_value=usage):
            result = runner.invoke(cli, ["resources", "-a", "10.0.0.1:6379", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output)["resources"]["TPU"]["used"] == 4.0

    def test_per_node_table(self):
        usage = {
            "resources": {"TPU": {"total": 8.0, "available": 4.0, "used": 4.0}},
            "nodes": [
                {
                    "ip": "10.0.0.1",
                    "node_id": "aa",
                    "resources": {
                        "TPU": {"total": 4.0, "available": 0.0, "used": 4.0},
                        "CPU": {"total": 8.0, "available": 8.0, "used": 0.0},
                    },
                },
                {
                    "ip": "10.0.0.2",
                    "node_id": "bb",
                    "resources": {
                        "TPU": {"total": 4.0, "available": 4.0, "used": 0.0},
                        "CPU": {"total": 8.0, "available": 8.0, "used": 0.0},
                    },
                },
            ],
        }
        runner = CliRunner()
        with mock.patch("eray.cli.main.resource_usage", return_value=usage):
            result = runner.invoke(cli, ["resources", "-a", "10.0.0.1:6379", "--per-node"])
        assert result.exit_code == 0
        node_line = next(line for line in result.output.splitlines() if line.startswith("10.0.0.1"))
        assert "4/4" in node_line
        idle_line = next(line for line in result.output.splitlines() if line.startswith("10.0.0.2"))
        assert "0/4" in idle_line


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


class TestWaitForCluster:
    """_wait_for_cluster polls the dashboard HTTP state API — never ray.init().

    A driver handshake would require the operator's Python/Ray versions to
    match the cluster's, which fails whenever the watcher box connects a
    node it bootstrapped with the image's system Python.
    """

    def _serve(self, payload_rows, port_holder):
        import http.server
        import json
        import threading

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"data": {"result": {"result": payload_rows}}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        port_holder.append(server.server_port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def test_alive_nodes_reaching_expected_returns_true(self):
        from eray.cli.tpu import _wait_for_cluster

        ports = []
        rows = [
            {"state": "ALIVE", "resources_total": {"TPU": 4.0}},
            {"state": "ALIVE", "resources_total": {"TPU": 4.0}},
            {"state": "DEAD", "resources_total": {"TPU": 4.0}},
        ]
        server = self._serve(rows, ports)
        try:
            assert _wait_for_cluster("127.0.0.1:6379", 2, timeout=10, dashboard_port=ports[0]) is True
        finally:
            server.shutdown()

    def test_dead_nodes_do_not_count(self, monkeypatch):
        import eray.cli.tpu as tpu_module
        from eray.cli.tpu import _wait_for_cluster

        monkeypatch.setattr(tpu_module, "RAY_READINESS_POLL_S", 0.05)
        ports = []
        rows = [{"state": "DEAD", "resources_total": {"TPU": 4.0}}]
        server = self._serve(rows, ports)
        try:
            assert _wait_for_cluster("127.0.0.1:6379", 1, timeout=0.5, dashboard_port=ports[0]) is False
        finally:
            server.shutdown()

    def test_unreachable_dashboard_times_out_false(self, monkeypatch):
        import eray.cli.tpu as tpu_module
        from eray.cli.tpu import _wait_for_cluster

        monkeypatch.setattr(tpu_module, "RAY_READINESS_POLL_S", 0.05)
        # Port from a server we immediately shut down — nothing listens.
        ports = []
        server = self._serve([], ports)
        server.shutdown()
        server.server_close()
        assert _wait_for_cluster("127.0.0.1:6379", 1, timeout=0.5, dashboard_port=ports[0]) is False


class TestDiscoverTpuNoIps:
    """Booting nodes (no networkEndpoints yet) are observable, not errors."""

    def _fake_describe(self, monkeypatch, raw):
        import eray.cli.utils as utils_module

        monkeypatch.setattr(utils_module, "gcloud_json", lambda args, **kw: raw)

    def test_booting_node_raises_by_default(self, monkeypatch):
        from eray.cli.utils import discover_tpu

        self._fake_describe(monkeypatch, {"acceleratorType": "v5p-8", "state": "CREATING"})
        with pytest.raises(ValueError, match="No internal IPs"):
            discover_tpu("boot", "proj", "zone")

    def test_allow_no_ips_returns_state_for_booting_node(self, monkeypatch):
        from eray.cli.utils import discover_tpu

        self._fake_describe(monkeypatch, {"acceleratorType": "v5p-8", "state": "CREATING"})
        tpu = discover_tpu("boot", "proj", "zone", allow_no_ips=True)
        assert tpu.state == "CREATING"
        assert tpu.internal_ips == []
        assert tpu.num_hosts == 0

    def test_describe_node_tolerates_booting_node(self, monkeypatch):
        import eray.cli.utils as utils_module
        from eray.provision.fleet import describe_node

        monkeypatch.setattr(
            utils_module, "gcloud_json", lambda args, **kw: {"acceleratorType": "v5p-8", "state": "STARTING"}
        )
        node = describe_node("boot", project="proj", zone="zone")
        assert node is not None
        assert node.state == "STARTING"
