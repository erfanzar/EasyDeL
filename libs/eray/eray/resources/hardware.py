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


"""Hardware type constants for accelerators."""


class HardwareType:
    """Constants representing known accelerator and hardware types.

    This class provides standardized identifiers for various hardware accelerators
    and compute devices that can be requested in Ray resource configurations.
    The constants ensure consistent naming across the application and provide
    a centralized reference for supported hardware types.

    The identifiers correspond to actual hardware accelerator names and models
    that may be available in cloud platforms, data centers, or local systems.
    They are used in resource configurations to specify hardware requirements
    for compute-intensive tasks.

    Categories:
        - NVIDIA GPUs: Tesla series, A-series, H-series (V100, A100, H100, etc.)
        - Intel: GPU Max series and Gaudi accelerators
        - AMD: Instinct series and Radeon GPUs
        - Google TPUs: Various TPU versions (V2, V3, V4, V5, V6)
        - AWS: Neuron cores for machine learning
        - Huawei: NPU accelerators (Ascend series)

    Example:
        >>> config = GpuAcceleratorConfig(
        ...     gpu_model=HardwareType.NVIDIA_A100,
        ...     device_count=2
        ... )
        >>> tpu_config = TpuAcceleratorConfig(
        ...     tpu_version=HardwareType.GOOGLE_TPU_V4
        ... )
    """

    NVIDIA_TESLA_V100 = "V100"
    NVIDIA_TESLA_P100 = "P100"
    NVIDIA_TESLA_T4 = "T4"
    NVIDIA_TESLA_P4 = "P4"
    NVIDIA_TESLA_K80 = "K80"
    NVIDIA_TESLA_A10G = "A10G"
    NVIDIA_L4 = "L4"
    NVIDIA_L40S = "L40S"
    NVIDIA_A100 = "A100"
    NVIDIA_H100 = "H100"
    NVIDIA_H200 = "H200"
    NVIDIA_H20 = "H20"
    INTEL_MAX_1550 = "Intel-GPU-Max-1550"
    INTEL_MAX_1100 = "Intel-GPU-Max-1100"
    INTEL_GAUDI = "Intel-GAUDI"
    AMD_INSTINCT_MI100 = "AMD-Instinct-MI100"
    AMD_INSTINCT_MI250x = "AMD-Instinct-MI250X"
    AMD_INSTINCT_MI250 = "AMD-Instinct-MI250X-MI250"
    AMD_INSTINCT_MI210 = "AMD-Instinct-MI210"
    AMD_INSTINCT_MI300x = "AMD-Instinct-MI300X-OAM"
    AMD_RADEON_R9_200_HD_7900 = "AMD-Radeon-R9-200-HD-7900"
    AMD_RADEON_HD_7900 = "AMD-Radeon-HD-7900"
    AWS_NEURON_CORE = "aws-neuron-core"
    GOOGLE_TPU_V2 = "TPU-V2"
    GOOGLE_TPU_V3 = "TPU-V3"
    GOOGLE_TPU_V4 = "TPU-V4"
    GOOGLE_TPU_V5P = "TPU-V5P"
    GOOGLE_TPU_V5LITEPOD = "TPU-V5LITEPOD"
    GOOGLE_TPU_V6E = "TPU-V6E"
    HUAWEI_NPU_910B = "Ascend910B"
    HUAWEI_NPU_910B4 = "Ascend910B4"
    NVIDIA_A100_40G = "A100-40G"
    NVIDIA_A100_80G = "A100-80G"
