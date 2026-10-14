# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Registry checks for TPU Pallas fused loss kernels."""

from __future__ import annotations

from ejkernel.kernels._pallas.tpu.fused_cross_entropy import _interface as ce_interface
from ejkernel.kernels._pallas.tpu.fused_kl_divergence import _interface as kl_interface
from ejkernel.kernels._registry import Backend, Platform, kernel_registry


def test_fused_cross_entropy_pallas_tpu_registration():
    """Ensure CE's TPU Pallas entrypoint is registered with the kernel registry."""
    impl = kernel_registry.get("fused_cross_entropy", platform=Platform.PALLAS, backend=Backend.TPU)
    assert impl is ce_interface.fused_cross_entropy
    assert kernel_registry.validate_signatures("fused_cross_entropy")


def test_fused_kl_divergence_pallas_tpu_registration():
    """Ensure KL's TPU Pallas entrypoint is registered with the kernel registry."""
    impl = kernel_registry.get("fused_kl_divergence", platform=Platform.PALLAS, backend=Backend.TPU)
    assert impl is kl_interface.fused_kl_divergence
    assert kernel_registry.validate_signatures("fused_kl_divergence")
