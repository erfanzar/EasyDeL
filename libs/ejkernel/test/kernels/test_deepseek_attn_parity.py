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

"""Parity tests for deepseek_attn interfaces."""

from __future__ import annotations

import inspect

import pytest

from ejkernel import modules
from ejkernel.kernels import _xla as xla_kernels
from ejkernel.kernels._pallas import tpu as pallas_tpu_kernels
from ejkernel.kernels._pallas.tpu.deepseek_attn import _interface as pallas_interface
from ejkernel.kernels._pallas.tpu.deepseek_attn._pallas_impl_bwd import _deepseek_attn_bwd_impl
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.deepseek_attn import _interface as xla_interface
from ejkernel.kernels._xla.deepseek_attn._xla_impl_bwd import deepseek_attn_backward
from ejkernel.modules import operations
from ejkernel.modules.operations import DeepSeekAttention
from ejkernel.modules.operations import deepseek_attn as op_functional


class TestDeepSeekAttnParity:
    """Validate API parity and registry wiring between TPU/Pallas and XLA."""

    def test_function_exists_in_both_interfaces(self):
        """Verify the public kernel entrypoint exists on both backends."""
        assert hasattr(xla_interface, "deepseek_attn")
        assert hasattr(pallas_interface, "deepseek_attn")

    def test_function_is_exported_from_backend_init_files(self):
        """Verify the public kernel entrypoint is re-exported from backend packages."""
        assert hasattr(xla_kernels, "deepseek_attn")
        assert hasattr(pallas_tpu_kernels, "deepseek_attn")

    def test_signature_and_defaults_match_exactly(self):
        """Verify XLA and Pallas interfaces expose identical call signatures."""
        xla_sig = inspect.signature(xla_interface.deepseek_attn)
        pallas_sig = inspect.signature(pallas_interface.deepseek_attn)

        canonical = [
            "query",
            "key_value",
            "w_kc",
            "w_vc",
            "query_index",
            "key_index",
            "index_weights",
            "index_topk",
            "softmax_scale",
            "index_softmax_scale",
            "b_q",
            "b_k",
            "causal",
        ]

        assert list(xla_sig.parameters) == canonical
        assert list(pallas_sig.parameters) == canonical

        for name in canonical:
            assert xla_sig.parameters[name].default == pallas_sig.parameters[name].default
            assert xla_sig.parameters[name].kind == pallas_sig.parameters[name].kind

        assert kernel_registry.validate_signatures("deepseek_attn")

    def test_registry_has_both_backends(self):
        """Verify XLA and Pallas implementations are both registered."""
        xla_impl = kernel_registry.get(
            "deepseek_attn",
            platform=Platform.XLA,
            backend=Backend.ANY,
        )
        pallas_impl = kernel_registry.get(
            "deepseek_attn",
            platform=Platform.PALLAS,
            backend=Backend.TPU,
        )
        assert callable(xla_impl)
        assert callable(pallas_impl)

    def test_backend_backward_helpers_are_importable(self):
        """Verify explicit XLA and Pallas backward helpers are importable."""
        assert callable(deepseek_attn_backward)
        assert callable(_deepseek_attn_bwd_impl)

    def test_operation_and_module_exports_exist(self):
        """Verify DeepSeek operation exports are available from both public modules."""
        assert hasattr(operations, "deepseek_attn")
        assert hasattr(operations, "DeepSeekAttention")
        assert hasattr(operations, "DeepSeekAttentionConfig")
        assert hasattr(modules, "deepseek_attn")
        assert hasattr(modules, "DeepSeekAttention")
        assert hasattr(modules, "DeepSeekAttentionConfig")

    def test_operation_api_is_callable(self):
        """Verify the public operation objects are wired up."""
        assert callable(op_functional)
        assert DeepSeekAttention is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
