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

"""Tests for the cluster-launcher generator: labels, host math, golden config."""

import json
import subprocess

import eray.provision.launcher as launcher_module
import pytest
import yaml
from click.testing import CliRunner
from eray.cli.main import cli
from eray.cli.utils import TpuInfo, build_ray_resource_flags, tpu_resource_labels
from eray.provision.launcher import (
    gce_instances_for_cluster,
    generate_configs,
    generate_zone_config,
    list_profiles,
    load_profile,
    make_node_type,
    resolve_profile,
    slice_hosts,
)
from eray.provision.registry import ClusterRegistry, LocalBackend


class TestSliceHosts:
    def test_v5p_hosts(self):
        assert slice_hosts("v5p-8") == 1  # 4 chips, 4 per host
        assert slice_hosts("v5p-64") == 8  # 32 chips
        assert slice_hosts("v5p-1024") == 128  # 512 chips

    def test_v4_hosts(self):
        assert slice_hosts("v4-32") == 4  # 16 chips

    def test_single_core_generations(self):
        assert slice_hosts("v5litepod-8") == 1  # 8 chips, 8 per host
        assert slice_hosts("v5litepod-16") == 2
        assert slice_hosts("v6e-16") == 4  # 16 chips, 4 per host


class TestResourceLabelDrift:
    """The generator and connect-mode must advertise identical labels."""

    def test_generator_matches_build_ray_resource_flags(self):
        for acc_type in ("v5p-8", "v5p-64", "v4-32", "v5litepod-16", "v6e-16"):
            hosts = slice_hosts(acc_type)
            node = make_node_type(acc_type, spot=True)
            connect_mode = json.loads(
                build_ray_resource_flags(
                    TpuInfo.from_ips(["10.0.0.1"] * hosts, acc_type),
                    is_head=True,
                )
            )
            generator_labels = {k: v for k, v in node["resources"].items() if k != "CPU"}
            assert generator_labels == connect_mode, acc_type

    def test_head_label_casing_matches_pool_scheduler(self):
        # SlicePoolManager schedules on TPU-{type}-head; casing is load-bearing.
        node = make_node_type("v5p-64", spot=True)
        assert "TPU-v5p-64-head" in node["resources"]
        assert not any(k.startswith("tpu-") for k in node["resources"])

    def test_physical_chip_counts(self):
        assert make_node_type("v5p-64", spot=True)["resources"]["TPU"] == 4
        assert make_node_type("v5litepod-16", spot=True)["resources"]["TPU"] == 8

    def test_labels_helper_consistency(self):
        labels = tpu_resource_labels("v5p-8", 1, is_head=True)
        assert labels["TPU"] == 4
        assert labels["TPU-v5p-8-head"] == 1
        worker = tpu_resource_labels("v5p-8", 1, is_head=False)
        assert "TPU-v5p-8-head" not in worker


class TestNodeType:
    def test_spot_flag_drives_scheduling_config(self):
        assert make_node_type("v5p-8", spot=True)["node_config"]["schedulingConfig"] == {"preemptible": True}
        assert make_node_type("v5p-8", spot=False)["node_config"]["schedulingConfig"] == {"preemptible": False}

    def test_runtime_version_from_family_map(self):
        assert make_node_type("v5p-8", spot=True)["node_config"]["runtimeVersion"] == "v2-alpha-tpuv5"
        assert make_node_type("v6e-16", spot=True)["node_config"]["runtimeVersion"] == "v2-alpha-tpuv6e"


class TestGenerateZoneConfig:
    TYPES = ("v5p-8", "v5p-64", "v6e-16", "v2-512")

    def test_golden_config(self):
        config = generate_zone_config(
            "proj",
            "us-central1-a",
            families=("v5p", "v6e"),
            image="my-image:tag",
            available_types=self.TYPES,
        )
        assert config["cluster_name"] == "easydel-us-central1-a"
        assert config["provider"]["availability_zone"] == "us-central1-a"
        assert config["provider"]["region"] == "us-central1"
        assert config["provider"]["project_id"] == "proj"
        assert config["docker"]["image"] == "my-image:tag"
        node_types = config["available_node_types"]
        assert "head_default" in node_types  # template head survives the merge
        assert set(node_types) == {"head_default", "tpu_slice_v5p_8", "tpu_slice_v5p_64", "tpu_slice_v6e_16"}
        v5p64 = node_types["tpu_slice_v5p_64"]
        assert v5p64["resources"]["TPU-v5p-64-head"] == 1
        assert v5p64["resources"]["TPU"] == 4
        assert v5p64["node_config"]["schedulingConfig"]["preemptible"] is True
        # v2-512 excluded (family not requested)
        assert not any("v2" in k for k in node_types)

    def test_zone_without_families_returns_none(self):
        assert generate_zone_config("proj", "z", families=("v5p",), available_types=["v2-8"]) is None

    def test_yaml_roundtrip_is_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher_module, "list_tpu_zones", lambda project: ["us-central1-a"])
        monkeypatch.setattr(launcher_module, "list_zone_accelerator_types", lambda project, zone: ["v5p-8"])
        written = generate_configs("proj", output_dir=tmp_path)
        assert len(written) == 1
        loaded = yaml.safe_load(written[0].read_text())
        assert loaded["available_node_types"]["tpu_slice_v5p_8"]["resources"]["TPU-v5p-8-head"] == 1
        assert loaded["max_workers"] == 1024


class TestAutoscaleCli:
    def test_group_registered(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["autoscale", "--help"])
        assert result.exit_code == 0
        for cmd in ("generate", "up", "down", "status"):
            assert cmd in result.output

    def test_generate_writes_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher_module, "list_zone_accelerator_types", lambda project, zone: ["v5p-8"])
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "autoscale",
                "generate",
                "--project",
                "proj",
                "--zones",
                "us-central1-a",
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "easydel-us-central1-a.yaml").exists()

    def test_template_ships_with_the_package(self):
        from importlib import resources

        text = (resources.files("eray.provision") / "templates" / "cluster-template.yaml").read_text()
        assert "available_node_types" in text
        assert "{{IMAGE}}" in text


class TestListProfiles:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert list_profiles(tmp_path / "does-not-exist") == []

    def test_discovers_valid_profiles(self, tmp_path):
        (tmp_path / "easydel-us-east5-a.yaml").write_text(
            "cluster_name: easydel-us-east5-a\nprovider:\n  project_id: proj\n  availability_zone: us-east5-a\n"
        )
        profiles = list_profiles(tmp_path)
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.name == "easydel-us-east5-a"
        assert profile.cluster_name == "easydel-us-east5-a"
        assert profile.project == "proj"
        assert profile.zone == "us-east5-a"
        assert profile.path == tmp_path / "easydel-us-east5-a.yaml"

    def test_skips_unparsable_and_nameless_files(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("cluster_name: [unterminated\n")
        (tmp_path / "no-name.yaml").write_text("provider:\n  project_id: proj\n")
        (tmp_path / "ok.yaml").write_text("cluster_name: ok\n")
        assert [p.name for p in list_profiles(tmp_path)] == ["ok"]

    def test_sorted_by_name(self, tmp_path):
        (tmp_path / "b-zone.yaml").write_text("cluster_name: b-zone\n")
        (tmp_path / "a-zone.yaml").write_text("cluster_name: a-zone\n")
        assert [p.name for p in list_profiles(tmp_path)] == ["a-zone", "b-zone"]


class TestLoadProfile:
    def test_missing_file_is_skipped_not_raised(self, tmp_path):
        assert load_profile(tmp_path / "ghost.yaml") is None

    def test_parses_a_profile(self, tmp_path):
        path = tmp_path / "x.yaml"
        path.write_text("cluster_name: x\nprovider:\n  project_id: p\n  availability_zone: z\n")
        profile = load_profile(path)
        assert profile.name == "x"
        assert profile.project == "p"
        assert profile.zone == "z"


class TestResolveProfile:
    def test_resolves_an_existing_path(self, tmp_path):
        path = tmp_path / "x.yaml"
        path.write_text("cluster_name: x\n")
        assert resolve_profile(str(path), tmp_path) == path

    def test_resolves_a_bare_profile_name(self, tmp_path):
        path = tmp_path / "easydel-us-east5-a.yaml"
        path.write_text("cluster_name: easydel-us-east5-a\n")
        assert resolve_profile("easydel-us-east5-a", tmp_path) == path

    def test_unknown_name_lists_known_profiles(self, tmp_path):
        (tmp_path / "a.yaml").write_text("cluster_name: a\n")
        with pytest.raises(FileNotFoundError, match="known profiles: a"):
            resolve_profile("ghost", tmp_path)

    def test_unknown_name_no_profiles_says_none(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="known profiles: none"):
            resolve_profile("ghost", tmp_path)


class TestGceInstancesForCluster:
    def test_builds_the_cluster_label_filter(self, monkeypatch):
        captured = {}

        def fake_gcloud_json(args, **kwargs):
            captured["args"] = args
            return [{"name": "head-0", "status": "RUNNING"}]

        monkeypatch.setattr(launcher_module, "gcloud_json", fake_gcloud_json)
        result = gce_instances_for_cluster("easydel-us-east5-a", project="proj", zone="us-east5-a")
        assert result == [{"name": "head-0", "status": "RUNNING"}]
        args = captured["args"]
        assert args[args.index("--filter") + 1] == "labels.ray-cluster-name=easydel-us-east5-a"
        assert args[args.index("--zones") + 1] == "us-east5-a"
        assert args[args.index("--project") + 1] == "proj"

    def test_gcloud_failure_returns_empty(self, monkeypatch):
        def raise_err(args, **kwargs):
            raise subprocess.CalledProcessError(1, "gcloud")

        monkeypatch.setattr(launcher_module, "gcloud_json", raise_err)
        assert gce_instances_for_cluster("x", project="p", zone="z") == []


class TestAutoscaleUpDownStatusCli:
    """`eray autoscale up/down/status` with no CONFIG: pickers + registry wiring."""

    @pytest.fixture
    def local_registry(self, tmp_path, monkeypatch):
        reg = ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))
        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
        return reg

    @pytest.fixture
    def profile_dir(self, tmp_path):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "easydel-us-east5-a.yaml").write_text(
            "cluster_name: easydel-us-east5-a\nprovider:\n  project_id: proj\n  availability_zone: us-east5-a\n"
        )
        return profiles_dir

    def test_up_auto_selects_the_only_profile_and_registers_it(self, local_registry, profile_dir, monkeypatch):
        monkeypatch.setattr(launcher_module, "ray_up", lambda path, yes=True: None)
        monkeypatch.setattr(launcher_module, "launcher_head_ip", lambda path: "10.0.0.9")
        result = CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(profile_dir), "-y"])
        assert result.exit_code == 0, result.output
        assert "one profile found" in result.output
        record = local_registry.get("easydel-us-east5-a")
        assert record is not None
        assert record.kind == "launcher"
        assert record.state == "HEALTHY"
        assert record.head_ip == "10.0.0.9"
        assert record.project == "proj"
        assert record.zone == "us-east5-a"
        assert record.last_up_ts is not None

    def test_up_multiple_profiles_prompts_and_uses_the_chosen_one(self, local_registry, tmp_path, monkeypatch):
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "a-zone.yaml").write_text("cluster_name: a-zone\n")
        (profiles_dir / "b-zone.yaml").write_text("cluster_name: b-zone\n")
        called = {}
        monkeypatch.setattr(launcher_module, "ray_up", lambda path, yes=True: called.setdefault("path", path))
        monkeypatch.setattr(launcher_module, "launcher_head_ip", lambda path: "10.0.0.9")
        result = CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(profiles_dir), "-y"], input="2\n")
        assert result.exit_code == 0, result.output
        assert str(called["path"]).endswith("b-zone.yaml")
        assert local_registry.get("b-zone") is not None
        assert local_registry.get("a-zone") is None

    def test_up_no_profiles_errors_pointing_at_generate(self, local_registry, tmp_path):
        result = CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(tmp_path / "empty"), "-y"])
        assert result.exit_code != 0
        assert "eray autoscale generate" in result.output

    def test_up_by_bare_profile_name_skips_the_picker(self, local_registry, profile_dir, monkeypatch):
        monkeypatch.setattr(launcher_module, "ray_up", lambda path, yes=True: None)
        monkeypatch.setattr(launcher_module, "launcher_head_ip", lambda path: "10.0.0.9")
        result = CliRunner().invoke(cli, ["autoscale", "up", "easydel-us-east5-a", "--dir", str(profile_dir), "-y"])
        assert result.exit_code == 0, result.output
        assert "Multiple profiles" not in result.output

    def test_down_stamps_down_state_and_timestamp(self, local_registry, profile_dir, monkeypatch):
        monkeypatch.setattr(launcher_module, "ray_up", lambda path, yes=True: None)
        monkeypatch.setattr(launcher_module, "launcher_head_ip", lambda path: "10.0.0.9")
        up = CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(profile_dir), "-y"])
        assert up.exit_code == 0, up.output

        monkeypatch.setattr(launcher_module, "ray_down", lambda path, yes=True: None)
        down = CliRunner().invoke(cli, ["autoscale", "down", "--dir", str(profile_dir), "-y"])
        assert down.exit_code == 0, down.output

        record = local_registry.get("easydel-us-east5-a")
        assert record.state == "DOWN"
        assert record.last_down_ts is not None

    def test_up_after_down_clears_stale_last_down_ts(self, local_registry, profile_dir, monkeypatch):
        # A fresh `up` must invalidate any earlier "died at T" history, or a
        # live cluster can flash DOWN/"died ... ago" if the live GCE probe
        # runs in the window before the new head instance is discoverable
        # (launcher_row_status falls back to last_down_ts when the probe
        # finds no head instance yet).
        monkeypatch.setattr(launcher_module, "ray_up", lambda path, yes=True: None)
        monkeypatch.setattr(launcher_module, "launcher_head_ip", lambda path: "10.0.0.9")
        monkeypatch.setattr(launcher_module, "ray_down", lambda path, yes=True: None)

        CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(profile_dir), "-y"])
        CliRunner().invoke(cli, ["autoscale", "down", "--dir", str(profile_dir), "-y"])
        assert local_registry.get("easydel-us-east5-a").last_down_ts is not None

        up_again = CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(profile_dir), "-y"])
        assert up_again.exit_code == 0, up_again.output

        record = local_registry.get("easydel-us-east5-a")
        assert record.state == "HEALTHY"
        assert record.last_down_ts is None

    def test_status_no_config_shows_unregistered_profile(self, local_registry, profile_dir):
        result = CliRunner().invoke(cli, ["autoscale", "status", "--dir", str(profile_dir)])
        assert result.exit_code == 0, result.output
        assert "easydel-us-east5-a" in result.output
        assert "UNREGISTERED" in result.output

    def test_status_no_config_json_after_down(self, local_registry, profile_dir, monkeypatch):
        monkeypatch.setattr(launcher_module, "ray_up", lambda path, yes=True: None)
        monkeypatch.setattr(launcher_module, "launcher_head_ip", lambda path: "10.0.0.9")
        CliRunner().invoke(cli, ["autoscale", "up", "--dir", str(profile_dir), "-y"])
        monkeypatch.setattr(launcher_module, "ray_down", lambda path, yes=True: None)
        CliRunner().invoke(cli, ["autoscale", "down", "--dir", str(profile_dir), "-y"])

        result = CliRunner().invoke(cli, ["autoscale", "status", "--dir", str(profile_dir), "--json"])
        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert rows[0]["status"] == "DOWN"
        assert rows[0]["since"].startswith("died")

    def test_status_no_profiles_no_registry_says_so(self, local_registry, tmp_path):
        result = CliRunner().invoke(cli, ["autoscale", "status", "--dir", str(tmp_path / "empty")])
        assert result.exit_code == 0, result.output
        assert "eray autoscale generate" in result.output
