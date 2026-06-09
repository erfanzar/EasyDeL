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


"""Test feature parity between XLA and Pallas ring attention implementations."""

import inspect

import pytest

from ejkernel.kernels._pallas.tpu.ring_attention import _interface as pallas_interface
from ejkernel.kernels._registry import kernel_registry
from ejkernel.kernels._xla.ring_attention import _interface as xla_interface


class TestFeatureParity:
    """Test that XLA and Pallas implementations have feature parity."""

    def test_function_exists_in_both(self):
        """Test that ring_attention function exists in both implementations."""
        assert hasattr(xla_interface, "ring_attention")
        assert hasattr(pallas_interface, "ring_attention")

    def test_default_values_consistency(self):
        """Test that signatures match (name/order/defaults/annotations)."""
        xla_sig = inspect.signature(xla_interface.ring_attention)
        pallas_sig = inspect.signature(pallas_interface.ring_attention)

        canonical = [
            "query",
            "key",
            "value",
            "q_segment_ids",
            "kv_segment_ids",
            "q_position_ids",
            "kv_position_ids",
            "softmax_aux",
            "bias",
            "mask_builder",
            "sliding_window",
            "chunk_size",
            "causal",
            "logits_soft_cap",
            "softmax_scale",
            "axis_name",
            "fwd_params",
            "bwd_params",
            "fused_backward",
        ]

        assert list(xla_sig.parameters) == canonical
        assert list(pallas_sig.parameters) == canonical

        assert xla_sig.parameters["q_segment_ids"].default is None
        assert xla_sig.parameters["kv_segment_ids"].default is None
        assert xla_sig.parameters["q_position_ids"].default is None
        assert xla_sig.parameters["kv_position_ids"].default is None
        assert xla_sig.parameters["softmax_aux"].default is None
        assert xla_sig.parameters["bias"].default is None
        assert xla_sig.parameters["mask_builder"].default is None
        assert xla_sig.parameters["sliding_window"].default is None
        assert xla_sig.parameters["chunk_size"].default is None
        assert xla_sig.parameters["causal"].default is False
        assert xla_sig.parameters["logits_soft_cap"].default is None
        assert xla_sig.parameters["softmax_scale"].default is None
        assert xla_sig.parameters["axis_name"].default is None
        assert xla_sig.parameters["fwd_params"].default is None
        assert xla_sig.parameters["bwd_params"].default is None
        assert xla_sig.parameters["fused_backward"].default is False

        assert kernel_registry.validate_signatures("ring_attention")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
