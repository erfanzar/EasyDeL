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

"""Tests for eray.resources — hardware types, Ray resources, accelerator configs."""

import dataclasses

import pytest

from eray.resources.configs import (
    CpuAcceleratorConfig,
    GpuAcceleratorConfig,
    TpuAcceleratorConfig,
)
from eray.resources.hardware import HardwareType
from eray.resources.ray_resources import RayResources, available_cpu_cores


class TestHardwareType:
    def test_nvidia_constants(self):
        assert hasattr(HardwareType, "NVIDIA_A100")
        assert hasattr(HardwareType, "NVIDIA_A100_80G")
        assert hasattr(HardwareType, "NVIDIA_H100")
        assert hasattr(HardwareType, "NVIDIA_H200")
        assert hasattr(HardwareType, "NVIDIA_L4")

    def test_tpu_constants(self):
        assert hasattr(HardwareType, "GOOGLE_TPU_V4")
        assert hasattr(HardwareType, "GOOGLE_TPU_V5P")
        assert hasattr(HardwareType, "GOOGLE_TPU_V6E")

    def test_amd_constants(self):
        assert hasattr(HardwareType, "AMD_INSTINCT_MI300x")

    def test_intel_constants(self):
        assert hasattr(HardwareType, "INTEL_MAX_1550")
        assert hasattr(HardwareType, "INTEL_GAUDI")


class TestRayResources:
    def test_to_kwargs_basic(self):
        rr = RayResources(num_cpus=4, num_gpus=1)
        kwargs = rr.to_kwargs()
        assert kwargs["num_cpus"] == 4
        assert kwargs["num_gpus"] == 1

    def test_to_kwargs_empty(self):
        rr = RayResources()
        kwargs = rr.to_kwargs()
        assert "num_cpus" in kwargs

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(RayResources)

    def test_available_cpu_cores(self):
        cores = available_cpu_cores()
        assert isinstance(cores, int)
        assert cores > 0


class TestAcceleratorConfigs:
    def test_cpu_config(self):
        cfg = CpuAcceleratorConfig()
        opts = cfg.get_remote_options()
        assert "num_cpus" in opts

    def test_gpu_config(self):
        cfg = GpuAcceleratorConfig(device_count=2, gpu_model="A100")
        assert cfg.device_count == 2
        assert cfg.gpu_model == "A100"

    def test_tpu_config(self):
        cfg = TpuAcceleratorConfig(tpu_version="v4-8", pod_count=1)
        assert cfg.tpu_version == "v4-8"
        assert cfg.pod_count == 1

    def test_gpu_frozen(self):
        cfg = GpuAcceleratorConfig(device_count=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.device_count = 99

    def test_tpu_frozen(self):
        cfg = TpuAcceleratorConfig(tpu_version="v4-8")
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.tpu_version = "v5"

    def test_cpu_frozen(self):
        cfg = CpuAcceleratorConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.cpu_count = 999

    def test_all_are_frozen_dataclasses(self):
        for cls in [CpuAcceleratorConfig, GpuAcceleratorConfig, TpuAcceleratorConfig]:
            assert dataclasses.is_dataclass(cls)
            params = getattr(cls, "__dataclass_params__", None)
            assert params is not None and params.frozen
