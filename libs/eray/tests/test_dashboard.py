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

"""Tests for `eray dashboard`: unified fleet+autoscale status and tracked tunnels."""

from __future__ import annotations

import json
import os
import sys
import time
from unittest import mock

import eray.provision.fleet as fleet_module
import eray.provision.launcher as launcher_module
import eray.provision.tunnel as tunnel_module
import pytest
from click.testing import CliRunner
from eray.cli.main import cli
from eray.provision.registry import ClusterRecord, ClusterRegistry, LocalBackend

SLEEP_ARGV = [sys.executable, "-c", "import time; time.sleep(30)"]

# A realistic spawn timestamp for fixtures that fake a *live* tunnel on this
# test process's own PID: tunnel liveness now also checks that the recorded
# start time matches the process's real start (PID-reuse guard), so a stale
# placeholder like 0.0 would be (correctly) rejected as not-our-process.
_LIVE_TS = tunnel_module._proc_start_ts(os.getpid()) or time.time()


def make_record(**overrides) -> ClusterRecord:
    defaults = dict(name="trainer1", kind="qr", project="proj", zone="us-east5-a")
    defaults.update(overrides)
    return ClusterRecord(**defaults)


@pytest.fixture
def local_registry(tmp_path, monkeypatch):
    reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
    monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
    return reg


@pytest.fixture(autouse=True)
def isolated_autoscale_profiles_dir(tmp_path, monkeypatch):
    # _load_records() scans DEFAULT_OUTPUT_DIR for unregistered profiles;
    # point it at an empty tmp dir by default so these tests never pick up
    # whatever real profiles happen to exist under the actual ~/.eray on
    # the machine running the suite. Tests exercising the merge behavior
    # override this via the `profiles_dir` fixture below.
    monkeypatch.setattr(launcher_module, "DEFAULT_OUTPUT_DIR", tmp_path / "empty-profiles")


@pytest.fixture
def profiles_dir(tmp_path, monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    monkeypatch.setattr(launcher_module, "DEFAULT_OUTPUT_DIR", d)
    return d


class TestDashboardCliWiring:
    def test_help_lists_subcommands(self):
        result = CliRunner().invoke(cli, ["dashboard", "--help"])
        assert result.exit_code == 0, result.output
        for subcmd in ("ls", "open", "stop"):
            assert subcmd in result.output


class TestDashboardLs:
    def test_no_clusters_registered(self, local_registry):
        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert "eray fleet add" in result.output

    def test_sees_a_generated_but_never_registered_profile(self, local_registry, profiles_dir, monkeypatch):
        # A cluster brought up by hand (`ray up`), or by an eray version
        # before `up` started writing to the registry, has a generated YAML
        # but no ClusterRecord at all. `eray dashboard` must still surface
        # it (mirrors eray autoscale status's on-disk-plus-registry merge)
        # instead of claiming "no clusters registered".
        (profiles_dir / "easydel-us-east5-a.yaml").write_text(
            "cluster_name: easydel-us-east5-a\nprovider:\n  project_id: proj\n  availability_zone: us-east5-a\n"
        )
        monkeypatch.setattr(launcher_module, "gce_instances_for_cluster", lambda *a, **k: [])

        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert "easydel-us-east5-a" in result.output
        assert "UNREGISTERED" in result.output

    def test_registered_record_wins_over_the_on_disk_profile(self, local_registry, profiles_dir, monkeypatch):
        # Once a profile IS registered, it must not also show up a second
        # time (or get its real state clobbered back to UNREGISTERED) just
        # because the YAML is still sitting in the profiles dir.
        (profiles_dir / "l1.yaml").write_text("cluster_name: l1\n")
        local_registry.upsert(
            make_record(name="l1", kind="launcher", project="proj", zone="z", state="HEALTHY", head_ip="10.0.0.1")
        )
        monkeypatch.setattr(launcher_module, "gce_instances_for_cluster", lambda *a, **k: [])

        result = CliRunner().invoke(cli, ["dashboard", "ls", "--json"])
        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert len(rows) == 1
        assert rows[0]["name"] == "l1"
        assert rows[0]["status"] != "UNREGISTERED"

    def test_open_works_on_an_unregistered_profile(self, local_registry, profiles_dir):
        # The whole point: you shouldn't need `eray autoscale up` again
        # just to open a dashboard for a cluster that's already running.
        config_path = profiles_dir / "easydel-us-east5-a.yaml"
        config_path.write_text("cluster_name: easydel-us-east5-a\n")
        captured = {}

        def fake_open_tunnel(name, argv, **kwargs):
            captured["argv"] = argv
            return tunnel_module.TunnelSession(
                name=name,
                kind=kwargs["kind"],
                pid=1,
                local_port=kwargs["local_port"],
                remote_port=kwargs["remote_port"],
                started_ts=_LIVE_TS,
            )

        with mock.patch.object(tunnel_module, "open_tunnel", side_effect=fake_open_tunnel):
            result = CliRunner().invoke(cli, ["dashboard", "open", "easydel-us-east5-a"])
        assert result.exit_code == 0, result.output
        assert captured["argv"][0] == sys.executable
        assert "exec_cluster" in captured["argv"][2]
        assert str(config_path) in captured["argv"]

    def test_lists_qr_and_launcher_clusters(self, local_registry, monkeypatch):
        local_registry.upsert(make_record(name="q1", kind="qr", head_ip="10.0.0.1"))
        local_registry.upsert(
            make_record(name="l1", kind="launcher", project="proj", zone="z", config_path="/tmp/l1.yaml")
        )
        monkeypatch.setattr(fleet_module, "head_reachable", lambda ip: True)
        monkeypatch.setattr(launcher_module, "gce_instances_for_cluster", lambda *a, **k: [])

        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert "q1" in result.output
        assert "RUNNING" in result.output
        assert "l1" in result.output
        assert "launcher" in result.output

    def test_qr_with_unreachable_head_is_not_reported_as_its_stale_state(self, local_registry, monkeypatch):
        # A preempted spot pod keeps its last-persisted state (HEALTHY) and a
        # recorded head_ip, but the head no longer answers. dashboard must
        # say UNREACHABLE, not echo the stale HEALTHY (which would contradict
        # the launcher branch and `eray fleet status`).
        local_registry.upsert(make_record(name="q1", kind="qr", state="HEALTHY", head_ip="10.0.0.1"))
        monkeypatch.setattr(fleet_module, "head_reachable", lambda ip: False)

        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert "UNREACHABLE" in result.output
        assert "HEALTHY" not in result.output

    def test_dir_option_discovers_profiles_from_a_custom_dir(self, local_registry, tmp_path, monkeypatch):
        # dashboard must honor --dir like `eray autoscale --dir`, or a cluster
        # generated to a non-default dir shows in one view and not the other.
        custom = tmp_path / "custom-profiles"
        custom.mkdir()
        (custom / "easydel-eu.yaml").write_text(
            "cluster_name: easydel-eu\nprovider:\n  project_id: proj\n  availability_zone: eu\n"
        )
        monkeypatch.setattr(launcher_module, "gce_instances_for_cluster", lambda *a, **k: [])

        result = CliRunner().invoke(cli, ["dashboard", "ls", "--dir", str(custom)])
        assert result.exit_code == 0, result.output
        assert "easydel-eu" in result.output

    def test_launcher_status_probes_the_yaml_cluster_name_not_the_registry_key(
        self, local_registry, tmp_path, monkeypatch
    ):
        # The registry key (record.name, a launcher profile's file stem) and
        # the YAML's own `cluster_name:` field can diverge (e.g. a renamed
        # profile); the live GCE probe must key on the latter, matching
        # `eray autoscale status`'s own `profile.cluster_name` usage —
        # otherwise the two views can disagree about whether a cluster is up.
        config_path = tmp_path / "l1.yaml"
        config_path.write_text("cluster_name: the-real-cluster-name\n")
        local_registry.upsert(
            make_record(
                name="l1",
                kind="launcher",
                project="proj",
                zone="z",
                config_path=str(config_path),
                state="HEALTHY",
            )
        )
        captured = {}

        def fake_gce_instances(cluster_name, **kwargs):
            captured["cluster_name"] = cluster_name
            return []

        monkeypatch.setattr(launcher_module, "gce_instances_for_cluster", fake_gce_instances)

        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert captured["cluster_name"] == "the-real-cluster-name"

    def test_json_output(self, local_registry, monkeypatch):
        local_registry.upsert(make_record(name="q1", kind="qr"))
        monkeypatch.setattr(fleet_module, "head_reachable", lambda ip: False)

        result = CliRunner().invoke(cli, ["dashboard", "ls", "--json"])
        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert rows[0]["name"] == "q1"
        assert rows[0]["kind"] == "qr"

    def test_tunnel_column_shows_active_port(self, local_registry):
        local_registry.upsert(make_record(name="q1", kind="qr"))
        session = tunnel_module.TunnelSession(
            name="q1", kind="qr", pid=os.getpid(), local_port=9999, remote_port=8265, started_ts=_LIVE_TS
        )
        tunnel_module._store().update(lambda doc: doc.__setitem__("q1", session.to_dict()))

        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert "9999" in result.output


class TestDashboardOpen:
    def test_no_clusters_registered_errors(self, local_registry):
        result = CliRunner().invoke(cli, ["dashboard", "open"])
        assert result.exit_code != 0
        assert "eray fleet add" in result.output

    def test_unknown_name_errors(self, local_registry):
        local_registry.upsert(make_record(name="q1"))
        result = CliRunner().invoke(cli, ["dashboard", "open", "ghost"])
        assert result.exit_code != 0
        assert "unknown cluster" in result.output

    def test_opens_launcher_cluster_via_ray_dashboard(self, local_registry, tmp_path):
        config_path = tmp_path / "l1.yaml"
        config_path.write_text("cluster_name: l1\n")
        local_registry.upsert(
            make_record(name="l1", kind="launcher", project="proj", zone="z", config_path=str(config_path))
        )
        captured = {}

        def fake_open_tunnel(name, argv, **kwargs):
            captured["name"] = name
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return tunnel_module.TunnelSession(
                name=name,
                kind=kwargs["kind"],
                pid=1,
                local_port=kwargs["local_port"],
                remote_port=kwargs["remote_port"],
                started_ts=_LIVE_TS,
            )

        with mock.patch.object(tunnel_module, "open_tunnel", side_effect=fake_open_tunnel):
            # --no-gcs keeps this focused on the dashboard forward; GCS-by-default
            # is covered by TestDashboardGcs.
            result = CliRunner().invoke(cli, ["dashboard", "open", "l1", "--no-gcs"])
        assert result.exit_code == 0, result.output
        # Launcher forward goes through Ray's own exec_cluster (one process,
        # so several ports can share one SSH), not `ray dashboard`.
        assert captured["argv"][0] == sys.executable
        assert "exec_cluster" in captured["argv"][2]
        assert str(config_path) in captured["argv"]
        assert "8265" in captured["argv"]  # dashboard remote port forwarded
        assert captured["kwargs"]["kind"] == "launcher"
        assert captured["kwargs"]["remote_port"] == 8265

    def test_launcher_cluster_without_config_path_errors(self, local_registry):
        local_registry.upsert(make_record(name="l1", kind="launcher", config_path=None))
        result = CliRunner().invoke(cli, ["dashboard", "open", "l1"])
        assert result.exit_code != 0
        assert "no config_path" in result.output

    def test_opens_qr_cluster_via_gcloud_ssh(self, local_registry):
        local_registry.upsert(make_record(name="q1", kind="qr", project="proj", zone="z"))
        captured = {}

        def fake_open_tunnel(name, argv, **kwargs):
            captured["argv"] = argv
            return tunnel_module.TunnelSession(
                name=name,
                kind=kwargs["kind"],
                pid=1,
                local_port=kwargs["local_port"],
                remote_port=kwargs["remote_port"],
                started_ts=_LIVE_TS,
            )

        with mock.patch.object(tunnel_module, "open_tunnel", side_effect=fake_open_tunnel):
            result = CliRunner().invoke(cli, ["dashboard", "open", "q1"])
        assert result.exit_code == 0, result.output
        argv = captured["argv"]
        assert argv[:6] == ["gcloud", "compute", "tpus", "tpu-vm", "ssh", "q1"]
        assert "--worker" in argv and argv[argv.index("--worker") + 1] == "0"

    def test_reuses_an_already_open_tunnel(self, local_registry, tmp_path):
        config_path = tmp_path / "l1.yaml"
        config_path.write_text("cluster_name: l1\n")
        local_registry.upsert(
            make_record(name="l1", kind="launcher", project="proj", zone="z", config_path=str(config_path))
        )
        # This test process's own pid is always alive, standing in for a
        # tunnel opened by an earlier `eray dashboard open` invocation.
        session = tunnel_module.TunnelSession(
            name="l1", kind="launcher", pid=os.getpid(), local_port=12345, remote_port=8265, started_ts=_LIVE_TS
        )
        tunnel_module._store().update(lambda doc: doc.__setitem__("l1", session.to_dict()))

        with mock.patch.object(tunnel_module, "open_tunnel") as mock_open:
            # --no-gcs so the reuse check is about the dashboard tunnel only
            # (with GCS on, the not-yet-open GCS tunnel would be spawned).
            result = CliRunner().invoke(cli, ["dashboard", "open", "l1", "--no-gcs"])
        assert result.exit_code == 0, result.output
        assert "already open" in result.output
        assert "12345" in result.output
        mock_open.assert_not_called()

    def test_no_name_multiple_clusters_prompts_by_index(self, local_registry):
        local_registry.upsert(make_record(name="a-cluster", kind="qr"))
        local_registry.upsert(make_record(name="b-cluster", kind="qr"))
        captured = {}

        def fake_open_tunnel(name, argv, **kwargs):
            captured["name"] = name
            return tunnel_module.TunnelSession(
                name=name,
                kind=kwargs["kind"],
                pid=1,
                local_port=kwargs["local_port"],
                remote_port=kwargs["remote_port"],
                started_ts=_LIVE_TS,
            )

        with mock.patch.object(tunnel_module, "open_tunnel", side_effect=fake_open_tunnel):
            result = CliRunner().invoke(cli, ["dashboard", "open", "--no-gcs"], input="2\n")
        assert result.exit_code == 0, result.output
        assert captured["name"] == "b-cluster"

    def test_no_selection_non_interactive_errors_cleanly(self, local_registry):
        local_registry.upsert(make_record(name="a-cluster", kind="qr"))
        local_registry.upsert(make_record(name="b-cluster", kind="qr"))
        result = CliRunner().invoke(cli, ["dashboard", "open"], input="")
        assert result.exit_code != 0
        assert "no cluster selected" in result.output


def _fake_open_tunnel(store):
    """An open_tunnel stand-in recording (name, argv, kwargs) into `store`."""

    def _impl(name, argv, **kwargs):
        store.append((name, argv, kwargs))
        return tunnel_module.TunnelSession(
            name=name,
            kind=kwargs["kind"],
            pid=42,
            local_port=kwargs["local_port"],
            remote_port=kwargs["remote_port"],
            started_ts=_LIVE_TS,
        )

    return _impl


class TestDashboardGcs:
    """Opening a dashboard forwards the GCS port too **by default**, over the
    SAME single SSH process as the dashboard port (two concurrent processes
    would collide on a launcher cluster's shared SSH ControlMaster). Each
    forwarded port gets its own tracked entry (sharing the forwarder's pid),
    so `ray status`/`eray resources` work with no `-a`. `--no-gcs` opts out."""

    def test_launcher_forwards_both_ports_in_one_process(self, local_registry, tmp_path):
        config_path = tmp_path / "l1.yaml"
        config_path.write_text("cluster_name: l1\n")
        local_registry.upsert(
            make_record(name="l1", kind="launcher", project="proj", zone="z", config_path=str(config_path))
        )
        opened, aliased = [], []

        with (
            mock.patch.object(tunnel_module, "open_tunnel", side_effect=_fake_open_tunnel(opened)),
            mock.patch.object(
                tunnel_module, "register_alias", side_effect=lambda name, **kw: aliased.append((name, kw))
            ),
        ):
            # no --gcs flag: GCS is the default
            result = CliRunner().invoke(cli, ["dashboard", "open", "l1"])
        assert result.exit_code == 0, result.output

        # exactly ONE process spawned, forwarding BOTH remote ports (8265 + 6379)
        assert [name for name, _, _ in opened] == ["l1"]
        argv = opened[0][1]
        assert argv[0] == sys.executable and "exec_cluster" in argv[2]
        assert "8265" in argv and "6379" in argv

        # the GCS port is tracked as an alias sharing the forwarder's pid
        assert [name for name, _ in aliased] == ["l1-gcs"]
        assert aliased[0][1]["pid"] == 42
        assert aliased[0][1]["remote_port"] == 6379

    def test_qr_forwards_both_ports_in_one_ssh(self, local_registry):
        local_registry.upsert(make_record(name="q1", kind="qr", project="proj", zone="z"))
        opened, aliased = [], []

        with (
            mock.patch.object(tunnel_module, "open_tunnel", side_effect=_fake_open_tunnel(opened)),
            mock.patch.object(
                tunnel_module, "register_alias", side_effect=lambda name, **kw: aliased.append((name, kw))
            ),
        ):
            result = CliRunner().invoke(cli, ["dashboard", "open", "q1"])
        assert result.exit_code == 0, result.output
        assert [name for name, _, _ in opened] == ["q1"]
        argv = opened[0][1]
        # one gcloud ssh with two -L forwards (dashboard + GCS)
        assert argv[:5] == ["gcloud", "compute", "tpus", "tpu-vm", "ssh"]
        assert argv.count("-L") == 2
        assert any("8265" in a for a in argv) and any("6379" in a for a in argv)
        assert [name for name, _ in aliased] == ["q1-gcs"]

    def test_no_gcs_flag_forwards_only_the_dashboard_port(self, local_registry):
        local_registry.upsert(make_record(name="q1", kind="qr", project="proj", zone="z"))
        opened = []

        with (
            mock.patch.object(tunnel_module, "open_tunnel", side_effect=_fake_open_tunnel(opened)),
            mock.patch.object(tunnel_module, "register_alias") as mock_alias,
        ):
            result = CliRunner().invoke(cli, ["dashboard", "open", "q1", "--no-gcs"])
        assert result.exit_code == 0, result.output
        assert [name for name, _, _ in opened] == ["q1"]
        argv = opened[0][1]
        assert argv.count("-L") == 1  # dashboard port only
        assert not any("6379" in a for a in argv)
        mock_alias.assert_not_called()

    def test_reuses_an_already_open_gcs_tunnel(self, local_registry):
        local_registry.upsert(make_record(name="q1", kind="qr", project="proj", zone="z"))
        # Dashboard tunnel already open too, so only the reuse path for
        # both should be exercised — open_tunnel must not be called at all.
        for name, port in (("q1", 8265), ("q1-gcs", 6379)):
            session = tunnel_module.TunnelSession(
                name=name, kind="qr", pid=os.getpid(), local_port=port, remote_port=port, started_ts=_LIVE_TS
            )
            tunnel_module._store().update(lambda doc, s=session: doc.__setitem__(s.name, s.to_dict()))

        with mock.patch.object(tunnel_module, "open_tunnel") as mock_open:
            result = CliRunner().invoke(cli, ["dashboard", "open", "q1", "--gcs"])
        assert result.exit_code == 0, result.output
        assert "already open" in result.output
        mock_open.assert_not_called()

    def test_ls_shows_both_tunnels(self, local_registry):
        local_registry.upsert(make_record(name="q1", kind="qr"))
        for name, port in (("q1", 55084), ("q1-gcs", 6379)):
            session = tunnel_module.TunnelSession(
                name=name, kind="qr", pid=os.getpid(), local_port=port, remote_port=port, started_ts=_LIVE_TS
            )
            tunnel_module._store().update(lambda doc, s=session: doc.__setitem__(s.name, s.to_dict()))

        result = CliRunner().invoke(cli, ["dashboard", "ls"])
        assert result.exit_code == 0, result.output
        assert "55084" in result.output
        assert "6379" in result.output


class TestDashboardStop:
    def test_stop_by_name(self, local_registry):
        local_registry.upsert(make_record(name="l1", kind="launcher"))
        port = tunnel_module.find_free_port(59401)
        tunnel_module.open_tunnel("l1", SLEEP_ARGV, kind="launcher", local_port=port)

        result = CliRunner().invoke(cli, ["dashboard", "stop", "l1"])
        assert result.exit_code == 0, result.output
        assert "stopped l1" in result.output
        assert tunnel_module.get_tunnel("l1") is None

    def test_stop_no_open_tunnel(self, local_registry):
        result = CliRunner().invoke(cli, ["dashboard", "stop", "ghost"])
        assert result.exit_code == 0, result.output
        assert "no open tunnel" in result.output

    def test_stop_by_name_also_closes_the_gcs_companion(self, local_registry):
        local_registry.upsert(make_record(name="l1", kind="launcher"))
        port = tunnel_module.find_free_port(59404)
        tunnel_module.open_tunnel("l1", SLEEP_ARGV, kind="launcher", local_port=port)
        gcs_port = tunnel_module.find_free_port(59405)
        tunnel_module.open_tunnel("l1-gcs", SLEEP_ARGV, kind="launcher", local_port=gcs_port)

        result = CliRunner().invoke(cli, ["dashboard", "stop", "l1"])
        assert result.exit_code == 0, result.output
        assert "stopped l1" in result.output
        assert "GCS tunnel" in result.output
        assert tunnel_module.get_tunnel("l1") is None
        assert tunnel_module.get_tunnel("l1-gcs") is None

    def test_stop_requires_name_or_all(self, local_registry):
        result = CliRunner().invoke(cli, ["dashboard", "stop"])
        assert result.exit_code != 0
        assert "--all" in result.output

    def test_stop_all(self, local_registry):
        port_a = tunnel_module.find_free_port(59402)
        tunnel_module.open_tunnel("a", SLEEP_ARGV, kind="launcher", local_port=port_a)
        port_b = tunnel_module.find_free_port(59403)
        tunnel_module.open_tunnel("b", SLEEP_ARGV, kind="qr", local_port=port_b)

        result = CliRunner().invoke(cli, ["dashboard", "stop", "--all"])
        assert result.exit_code == 0, result.output
        assert "stopped a" in result.output
        assert "stopped b" in result.output
        assert tunnel_module.list_tunnels() == {}

    def test_stop_all_no_open_tunnels(self, local_registry):
        result = CliRunner().invoke(cli, ["dashboard", "stop", "--all"])
        assert result.exit_code == 0, result.output
        assert "no open tunnels" in result.output


class TestDashboardBare:
    def test_bare_invocation_lists_without_opening(self, local_registry, monkeypatch):
        local_registry.upsert(make_record(name="q1", kind="qr"))
        monkeypatch.setattr(fleet_module, "head_reachable", lambda ip: False)

        with mock.patch.object(tunnel_module, "open_tunnel") as mock_open:
            result = CliRunner().invoke(cli, ["dashboard"])
        assert result.exit_code == 0, result.output
        assert "q1" in result.output
        mock_open.assert_not_called()

    def test_bare_invocation_no_clusters(self, local_registry):
        result = CliRunner().invoke(cli, ["dashboard"])
        assert result.exit_code == 0, result.output
        assert "eray fleet add" in result.output
