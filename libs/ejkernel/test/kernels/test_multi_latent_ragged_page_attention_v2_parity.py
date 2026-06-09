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

"""Parity tests for multi_latent_ragged_page_attention_v2 interfaces."""

from __future__ import annotations

import inspect

import pytest

from ejkernel.kernels import _xla as xla_kernels
from ejkernel.kernels._pallas import tpu as pallas_tpu_kernels
from ejkernel.kernels._pallas.tpu.multi_latent_ragged_page_attention_v2 import (
    _interface as pallas_interface,
)
from ejkernel.kernels._pallas.tpu.multi_latent_ragged_page_attention_v2._pallas_impl_fwd import (
    mla_ragged_paged_attention_v2 as pallas_impl_entrypoint,
)
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.multi_latent_ragged_page_attention_v2 import _interface as xla_interface
from ejkernel.kernels._xla.multi_latent_ragged_page_attention_v2._xla_impl_fwd import (
    multi_latent_ragged_page_attention_v2_impl as xla_impl_entrypoint,
)
from ejkernel.modules.operations import MultiLatentRaggedPageAttentionV2
from ejkernel.modules.operations import multi_latent_ragged_page_attention_v2 as op_functional


class TestMultiLatentRaggedPageAttentionV2Parity:
    """Validate API parity and registry wiring between TPU/Pallas and XLA."""

    def test_function_exists_in_both_interfaces(self):
        """Verify function is exposed from both XLA and Pallas interfaces."""
        assert hasattr(xla_interface, "multi_latent_ragged_page_attention_v2")
        assert hasattr(pallas_interface, "multi_latent_ragged_page_attention_v2")

    def test_function_is_exported_from_backend_init_files(self):
        """Verify function is re-exported from backend __init__ modules."""
        assert hasattr(xla_kernels, "multi_latent_ragged_page_attention_v2")
        assert hasattr(pallas_tpu_kernels, "multi_latent_ragged_page_attention_v2")

    def test_signature_and_defaults_match_exactly(self):
        """Verify XLA and Pallas signatures have identical parameters and defaults."""
        xla_sig = inspect.signature(xla_interface.multi_latent_ragged_page_attention_v2)
        pallas_sig = inspect.signature(pallas_interface.multi_latent_ragged_page_attention_v2)

        canonical = [
            "queries_nope",
            "queries_pe",
            "keys_values",
            "keys_pe",
            "kv_cache",
            "kv_lens",
            "block_tables",
            "query_start_loc",
            "distribution",
            "softmax_scale",
            "sliding_window",
            "logits_soft_cap",
            "mask_value",
            "q_scale",
            "k_scale",
            "v_scale",
            "chunk_prefill_size",
            "num_kv_pages_per_block",
            "num_queries_per_block",
            "vmem_limit_bytes",
            "debug_mode",
        ]

        assert list(xla_sig.parameters) == canonical
        assert list(pallas_sig.parameters) == canonical

        for name in canonical:
            assert xla_sig.parameters[name].default == pallas_sig.parameters[name].default
            assert xla_sig.parameters[name].kind == pallas_sig.parameters[name].kind

        assert kernel_registry.validate_signatures("multi_latent_ragged_page_attention_v2")

    def test_registry_has_both_backends(self):
        """Verify both XLA and Pallas backends are registered in the kernel registry."""
        xla_impl = kernel_registry.get(
            "multi_latent_ragged_page_attention_v2",
            platform=Platform.XLA,
            backend=Backend.ANY,
        )
        pallas_impl = kernel_registry.get(
            "multi_latent_ragged_page_attention_v2",
            platform=Platform.PALLAS,
            backend=Backend.TPU,
        )
        assert callable(xla_impl)
        assert callable(pallas_impl)

    def test_backend_impl_entrypoints_are_importable(self):
        """Verify the lower-level XLA and Pallas entrypoints are importable."""
        assert callable(xla_impl_entrypoint)
        assert callable(pallas_impl_entrypoint)

    def test_operation_run_does_not_expose_impl_params(self):
        """Verify Kernel.run signature hides implementation-only parameters."""
        run_sig = inspect.signature(MultiLatentRaggedPageAttentionV2.run)
        run_params = set(run_sig.parameters)
        for impl_only in (
            "chunk_prefill_size",
            "num_kv_pages_per_block",
            "num_queries_per_block",
            "vmem_limit_bytes",
            "debug_mode",
        ):
            assert impl_only not in run_params

    def test_operation_functional_does_not_expose_impl_params(self):
        """Verify functional API signature hides implementation-only parameters."""
        func_sig = inspect.signature(op_functional)
        func_params = set(func_sig.parameters)
        for impl_only in (
            "chunk_prefill_size",
            "num_kv_pages_per_block",
            "num_queries_per_block",
            "vmem_limit_bytes",
            "debug_mode",
        ):
            assert impl_only not in func_params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
