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

"""Parity tests for grouped_matmulv3 interfaces."""

from __future__ import annotations

import inspect

import pytest

from ejkernel.kernels import _xla as xla_kernels
from ejkernel.kernels._pallas import tpu as pallas_tpu_kernels
from ejkernel.kernels._pallas.tpu.grouped_matmulv3 import _interface as pallas_interface
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.grouped_matmulv3 import _interface as xla_interface


class TestGroupedMatmulV3Parity:
    """Validate API parity and registry wiring between TPU/Pallas and XLA."""

    def test_function_exists_in_both_interfaces(self):
        assert hasattr(xla_interface, "grouped_matmulv3")
        assert hasattr(pallas_interface, "grouped_matmulv3")

    def test_function_is_exported_from_backend_init_files(self):
        assert hasattr(xla_kernels, "grouped_matmulv3")
        assert hasattr(pallas_tpu_kernels, "grouped_matmulv3")

    def test_signature_and_defaults_match_exactly(self):
        xla_sig = inspect.signature(xla_interface.grouped_matmulv3)
        pallas_sig = inspect.signature(pallas_interface.grouped_matmulv3)
        assert list(xla_sig.parameters) == list(pallas_sig.parameters)
        for name in xla_sig.parameters:
            assert xla_sig.parameters[name].default == pallas_sig.parameters[name].default
            assert xla_sig.parameters[name].kind == pallas_sig.parameters[name].kind

    def test_registry_has_both_backends(self):
        xla_impl = kernel_registry.get("grouped_matmulv3", platform=Platform.XLA, backend=Backend.ANY)
        pallas_impl = kernel_registry.get("grouped_matmulv3", platform=Platform.PALLAS, backend=Backend.TPU)
        assert callable(xla_impl)
        assert callable(pallas_impl)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
