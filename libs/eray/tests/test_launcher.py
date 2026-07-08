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

import eray.provision.launcher as launcher_module
import yaml
from click.testing import CliRunner
from eray.cli.main import cli
from eray.cli.utils import TpuInfo, build_ray_resource_flags, tpu_resource_labels
from eray.provision.launcher import (
    generate_configs,
    generate_zone_config,
    make_node_type,
    slice_hosts,
)


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
