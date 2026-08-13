# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Quantization seam of the stacked-expert MoE linears.

Covers the conversion contract (``quantization_supported`` /
``to_quantized``) with the config's optional fields left at their defaults,
the standalone int4 forward, and preservation of the dense twin's
``(EP, FSDP)`` at-rest expert sharding when quantize-at-load runs outside the
parameter-init scope.
"""

import numpy as np
import spectrax as spx
from easydel.infra.sharding import moe_expert_param_layout_scope
from easydel.layers.linears import ColumnParallelMoELinear
from easydel.layers.quantization import QuantizationConfig, QuantizationType
from jax import numpy as jnp
from spectrax import common_types

EP = common_types.EP
FSDP = common_types.FSDP


def _dense(num_experts=4, in_features=32, out_features=64):
    return ColumnParallelMoELinear(
        num_experts=num_experts,
        in_features=in_features,
        out_features=out_features,
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )


class TestConfigDefaults:
    """The CHANNELWISE config with default (None) fields must convert."""

    def test_quantization_supported_with_default_bits(self):
        """``bits=None`` is the CHANNELWISE default and resolves to 8 — the
        support probe must not TypeError on every quantizer walk."""
        layer = _dense()
        config = QuantizationConfig(dtype=QuantizationType.CHANNELWISE)
        assert config.bits is None, "test targets the config default"
        assert layer.quantization_supported(config) is True

    def test_to_quantized_with_default_bits_yields_int8(self):
        layer = _dense()
        config = QuantizationConfig(dtype=QuantizationType.CHANNELWISE)
        qlayer = layer.to_quantized(config)
        assert qlayer.quant_kernel.value.dtype == jnp.int8
        assert qlayer.quant_scales.value.shape == (4, 1, 64)


class TestStandaloneForward:
    """The standalone (non-fused) forward must serve both bit widths."""

    def _reference(self, qlayer, x, group_sizes):
        codes = np.asarray(qlayer.quant_kernel.value, np.float32)
        scales = np.asarray(qlayer.quant_scales.value, np.float32)
        dense = codes * scales  # [E, K, N]
        out = np.zeros((x.shape[0], dense.shape[-1]), np.float32)
        start = 0
        for expert, size in enumerate(np.asarray(group_sizes)):
            out[start : start + size] = np.asarray(x[start : start + size], np.float32) @ dense[expert]
            start += size
        return out

    def _run(self, bits):
        layer = _dense()
        config = QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=bits)
        qlayer = layer.to_quantized(config)
        expected_dtype = jnp.int8 if bits == 8 else jnp.int4
        assert qlayer.quant_kernel.value.dtype == expected_dtype

        rng = np.random.default_rng(0)
        group_sizes = jnp.asarray([3, 0, 5, 8], jnp.int32)
        x = jnp.asarray(rng.normal(size=(16, 32)), jnp.float32)
        out = np.asarray(qlayer(x, group_sizes), np.float32)
        want = self._reference(qlayer, x, group_sizes)
        rel = np.abs(out - want).max() / (np.abs(want).max() + 1e-9)
        # The only approximation is the runtime int8 activation quantization.
        assert rel < 5e-2, f"bits={bits} standalone forward relerr {rel}"

    def test_int8_forward_matches_dequant_reference(self):
        self._run(bits=8)

    def test_int4_forward_matches_dequant_reference(self):
        """int4 codes must route through a working path (int8-upcast w8a8)
        instead of crashing the strictly-Int8 grouped matmul."""
        self._run(bits=4)


class TestAtRestShardingPreserved:
    """to_quantized outside the init scope must not drop (EP, FSDP)."""

    def test_quant_leaves_keep_ep_fsdp_expert_dim(self):
        with moe_expert_param_layout_scope(fsdp_shard_expert_weights=True):
            layer = _dense()
        dense_expert_entry = layer.weight.sharding.axis_names[0]
        assert tuple(dense_expert_entry) == (EP, FSDP), "dense twin must carry the knob's layout"

        # Quantize-at-load runs OUTSIDE the parameter-init scope.
        config = QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=8)
        qlayer = layer.to_quantized(config)

        quant_expert_entry = qlayer.quant_kernel.sharding.axis_names[0]
        scales_expert_entry = qlayer.quant_scales.sharding.axis_names[0]
        assert tuple(quant_expert_entry) == (EP, FSDP)
        assert tuple(scales_expert_entry) == (EP, FSDP)

    def test_default_layout_stays_ep_only(self):
        layer = _dense()
        config = QuantizationConfig(dtype=QuantizationType.CHANNELWISE, bits=8)
        qlayer = layer.to_quantized(config)
        assert qlayer.quant_kernel.sharding.axis_names[0] == EP
        assert qlayer.quant_scales.sharding.axis_names[0] == EP
