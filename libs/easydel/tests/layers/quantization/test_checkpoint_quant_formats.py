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

"""Numerical parity for every supported quantized checkpoint format.

Each format is encoded here with an independent NumPy reference — packing the
bits by hand rather than calling the production packer — then decoded through
the adapter and compared against the reference dequantization. That is the
only way these tests can fail for the right reason: comparing the adapter with
the code it wraps would pass no matter how wrong both were.

Tolerances are per-format because they are dominated by *requantization*, not
by decoding. Decoding is exact for every format; the error is introduced when
the dense weight is repacked into EasyDeL's runtime grouping, which runs along
a different axis than any checkpoint's. See
``TestGroupingAxisLimitation`` — that gap is the single largest accuracy item
in this subsystem and these tolerances pin its current cost.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.quantization import QuantizationType
from easydel.layers.quantization.checkpoint import (
    ActivationQuantKind,
    CheckpointQuantScheme,
    available_quant_methods,
)
from ejkernel.quantization import dequantize, prepack_quantized_weights

E2M1_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
AWQ_LANE_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _scheme(**quant_config) -> CheckpointQuantScheme:
    """Build a scheme from a raw ``quantization_config`` body.

    Args:
        **quant_config: Keys of the quantization block, including
            ``quant_method``.

    Returns:
        The resolved scheme.
    """
    scheme = CheckpointQuantScheme.from_hf_config({"quantization_config": dict(quant_config)})
    assert scheme is not None
    return scheme


def _dequantize_canonical(weight) -> np.ndarray:
    """Dequantize a canonical weight back to dense, applying any output scale.

    Args:
        weight: The :class:`CanonicalQuantizedWeight` to expand.

    Returns:
        The dense ``[in, out]`` array as NumPy.
    """
    mode, group_size, bits, _ = weight.spec.resolved
    dense = np.asarray(
        dequantize(
            weight.quant_kernel,
            weight.quant_scales,
            weight.quant_biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis="col",
        )
    )
    if weight.output_scale is not None:
        dense = dense * float(weight.output_scale)
    return dense


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Max absolute deviation normalized by the reference's magnitude.

    Args:
        actual: Reconstructed weight.
        expected: Reference weight.

    Returns:
        The relative error.
    """
    return float(np.abs(actual - expected).max() / np.abs(expected).max())


def _pack_u4_awq(values: np.ndarray) -> np.ndarray:
    """Pack uint4 values along the last axis into AWQ-ordered uint32 lanes.

    Independent reference implementation of AWQ's packing, used to build
    inputs the adapter must be able to read.

    Args:
        values: Integer array whose last axis is a multiple of 8.

    Returns:
        ``uint32`` array with the last axis divided by 8.
    """
    grouped = values.reshape(*values.shape[:-1], -1, 8)
    packed = np.zeros(grouped.shape[:-1], np.uint32)
    for slot, lane in enumerate(AWQ_LANE_ORDER):
        packed |= (grouped[..., lane].astype(np.uint32) & 0xF) << np.uint32(4 * slot)
    return packed


def _pack_u4_sequential(values: np.ndarray, axis: int) -> np.ndarray:
    """Pack uint4 values sequentially into uint32 lanes along ``axis``.

    This is GPTQ's and compressed-tensors' packing, which differs from AWQ's
    only in lane order — a difference that silently scrambles weights if
    confused, so both are exercised.

    Args:
        values: Integer array whose ``axis`` is a multiple of 8.
        axis: Axis to pack along.

    Returns:
        ``uint32`` array with ``axis`` divided by 8.
    """
    moved = np.moveaxis(values, axis, -1)
    grouped = moved.reshape(*moved.shape[:-1], -1, 8)
    packed = np.zeros(grouped.shape[:-1], np.uint32)
    for slot in range(8):
        packed |= (grouped[..., slot].astype(np.uint32) & 0xF) << np.uint32(4 * slot)
    return np.moveaxis(packed, -1, axis)


def _pack_e2m1(codes: np.ndarray) -> np.ndarray:
    """Pack E2M1 nibble codes two-per-byte along the last axis.

    Args:
        codes: Integer codes in ``[0, 16)``; last axis must be even.

    Returns:
        ``uint8`` array with the last axis halved.
    """
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


class TestRegistration:
    """Every published format resolves to an adapter."""

    @pytest.mark.parametrize(
        "quant_method",
        ["fp8", "awq", "auto_awq", "gptq", "mxfp4", "gpt_oss_mxfp4", "modelopt_fp4", "compressed-tensors"],
    )
    def test_method_is_registered(self, quant_method):
        """A checkpoint naming this method must find its adapter."""
        assert quant_method in available_quant_methods()


class TestFp8:
    """FP8 checkpoints across all three scale granularities."""

    def _encode(self, rng, shape, scale_shape):
        """Encode a random weight as fp8 elements plus a scale grid.

        The scale grid may be coarser than the weight along either axis
        (per-tensor, per-channel or blocked), so it is expanded by repetition
        before use rather than relying on NumPy broadcasting, which only
        handles the degenerate-axis case.
        """
        dense = rng.normal(size=shape).astype(np.float32)
        scale = np.abs(dense).max() / 448.0 * np.ones(scale_shape, np.float32)
        expanded = scale
        for axis, (scale_dim, target_dim) in enumerate(zip(scale_shape, shape, strict=True)):
            if scale_dim != target_dim:
                expanded = np.repeat(expanded, -(-target_dim // scale_dim), axis=axis)[
                    tuple(slice(0, dim) for dim in shape)
                ]
        codes = jnp.asarray(dense / expanded).astype(jnp.float8_e4m3fn)
        reference = (np.asarray(codes.astype(jnp.float32)) * expanded).T
        return codes, scale, reference

    def test_per_channel_round_trips(self):
        """Per-channel fp8 decodes and repacks within int8 requant error."""
        rng = np.random.default_rng(0)
        codes, scale, reference = self._encode(rng, (128, 256), (128, 1))
        resolved = _scheme(quant_method="fp8", activation_scheme="dynamic").for_path("l")
        weight = resolved.adapter.to_canonical(
            {"weight": codes, "weight_scale": jnp.asarray(scale)},
            source=resolved.source,
            target=resolved.target,
        )
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.01

    def test_block_scaled_uses_weight_scale_inv(self):
        """DeepSeek-style block fp8 names its scale tensor differently.

        Declaring the wrong source name makes the fusion silently never fire,
        because a rule only triggers when every source key is present.
        """
        resolved = _scheme(quant_method="fp8", weight_block_size=[128, 128]).for_path("l")
        assert resolved.source_suffixes == ("weight", "weight_scale_inv")

        rng = np.random.default_rng(1)
        codes, scale, reference = self._encode(rng, (128, 256), (1, 2))
        weight = resolved.adapter.to_canonical(
            {"weight": codes, "weight_scale_inv": jnp.asarray(scale)},
            source=resolved.source,
            target=resolved.target,
        )
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.01

    def test_block_scaled_non_divisible_dim_uses_declared_block(self):
        """Scale boundaries sit at the checkpoint's block size, not at ceil.

        A 576-row weight with 128-row blocks (DeepSeek-V3 shapes) carries 5
        scale rows. Expanding each by ``ceil(576 / 5) = 116`` puts every
        boundary in the wrong place and mis-scales 120 rows; the declared
        ``weight_block_size`` must drive the expansion instead.
        """
        from easydel.layers.quantization.checkpoint._codecs import decode_scaled_elements

        rng = np.random.default_rng(9)
        out_features, in_features, block = 576, 128, 128
        dense = rng.normal(size=(out_features, in_features)).astype(np.float32)
        scale = rng.random((-(-out_features // block), 1)).astype(np.float32) + 0.5
        expanded = np.repeat(scale, block, axis=0)[:out_features]
        expanded = np.repeat(expanded, block, axis=1)[:, :in_features]
        codes = jnp.asarray(dense / expanded).astype(jnp.float8_e4m3fn)
        reference = (np.asarray(codes.astype(jnp.float32)) * expanded).T

        resolved = _scheme(quant_method="fp8", weight_block_size=[block, block]).for_path("l")
        decoded = np.asarray(decode_scaled_elements(codes, jnp.asarray(scale), block_shape=resolved.source.block_shape))
        np.testing.assert_allclose(decoded, reference, rtol=1e-6, atol=1e-6)

        weight = resolved.adapter.to_canonical(
            {"weight": codes, "weight_scale_inv": jnp.asarray(scale)},
            source=resolved.source,
            target=resolved.target,
        )
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.01

    def test_static_activation_adds_input_scale(self):
        """A static activation scheme consumes and carries its calibration."""
        resolved = _scheme(quant_method="fp8", activation_scheme="static").for_path("l")
        assert "input_scale" in resolved.source_suffixes
        assert resolved.target.activation.kind is ActivationQuantKind.STATIC

    def test_dynamic_activation_needs_no_extra_tensor(self):
        """A dynamic scheme computes scales at run time, so ships none."""
        resolved = _scheme(quant_method="fp8", activation_scheme="dynamic").for_path("l")
        assert "input_scale" not in resolved.source_suffixes
        assert resolved.target.activation.kind is ActivationQuantKind.DYNAMIC


class TestAwq:
    """AWQ's interleaved uint4 packing."""

    def test_round_trips_with_zero_points(self):
        """Packed codes, scales and zero-points reconstruct the weight."""
        rng = np.random.default_rng(2)
        in_features, out_features, group = 256, 128, 32
        codes = rng.integers(0, 16, size=(in_features, out_features)).astype(np.uint32)
        zeros = rng.integers(0, 16, size=(in_features // group, out_features)).astype(np.uint32)
        scales = rng.random((in_features // group, out_features)).astype(np.float32) + 0.1
        reference = (codes.astype(np.float32) - np.repeat(zeros, group, 0).astype(np.float32)) * np.repeat(
            scales, group, 0
        )

        resolved = _scheme(quant_method="awq", group_size=group, zero_point=True, bits=4).for_path("l")
        weight = resolved.adapter.to_canonical(
            {
                "qweight": jnp.asarray(_pack_u4_awq(codes)),
                "scales": jnp.asarray(scales),
                "qzeros": jnp.asarray(_pack_u4_awq(zeros)),
            },
            source=resolved.source,
            target=resolved.target,
        )
        assert weight.spec.dtype is QuantizationType.AFFINE
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.10

    def test_group_size_follows_the_checkpoint(self):
        """The runtime group size is the checkpoint's, not a default."""
        resolved = _scheme(quant_method="awq", group_size=64).for_path("l")
        assert resolved.target.group_size == 64


class TestGptq:
    """GPTQ packs along the input axis, unlike AWQ."""

    def _build(self, rng, in_features, out_features, group, *, desc_act=False):
        """Build a GPTQ-encoded weight and its reference dequantization."""
        codes = rng.integers(0, 16, size=(in_features, out_features)).astype(np.uint32)
        zeros = rng.integers(0, 15, size=(in_features // group, out_features)).astype(np.uint32)
        scales = rng.random((in_features // group, out_features)).astype(np.float32) + 0.1
        tensors = {
            "qweight": jnp.asarray(_pack_u4_sequential(codes, axis=0)),
            "scales": jnp.asarray(scales),
            "qzeros": jnp.asarray(_pack_u4_sequential(zeros, axis=-1)),
        }
        reference = (codes.astype(np.float32) - (np.repeat(zeros, group, 0).astype(np.float32) + 1.0)) * np.repeat(
            scales, group, 0
        )
        return tensors, reference

    def test_round_trips(self):
        """Sequential input-axis packing decodes correctly."""
        rng = np.random.default_rng(3)
        tensors, reference = self._build(rng, 256, 128, 32)
        resolved = _scheme(quant_method="gptq", group_size=32, bits=4).for_path("l")
        weight = resolved.adapter.to_canonical(tensors, source=resolved.source, target=resolved.target)
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.10

    def test_act_order_requests_the_group_index(self):
        """``desc_act`` needs ``g_idx``; without it scales gather wrongly."""
        resolved = _scheme(quant_method="gptq", group_size=32, desc_act=True).for_path("l")
        assert "g_idx" in resolved.source_suffixes

    def test_no_act_order_omits_the_group_index(self):
        """Declaring an absent tensor would stop the rule from firing."""
        resolved = _scheme(quant_method="gptq", group_size=32, desc_act=False).for_path("l")
        assert "g_idx" not in resolved.source_suffixes


class TestMxfp4:
    """MXFP4's E2M1 elements and E8M0 shared exponents."""

    def test_round_trips(self):
        """Packed nibbles and exponent codes reconstruct the weight."""
        rng = np.random.default_rng(4)
        out_features, in_features = 128, 256
        codes = rng.integers(0, 16, size=(out_features, in_features)).astype(np.uint8)
        exponents = rng.integers(120, 135, size=(out_features, in_features // 32)).astype(np.uint8)
        reference = (E2M1_VALUES[codes] * np.repeat(np.exp2(exponents.astype(np.float32) - 127), 32, axis=1)).T

        resolved = _scheme(quant_method="mxfp4").for_path("l")
        weight = resolved.adapter.to_canonical(
            {"weight": jnp.asarray(_pack_e2m1(codes)), "weight_scale": jnp.asarray(exponents)},
            source=resolved.source,
            target=resolved.target,
        )
        assert weight.spec.dtype is QuantizationType.MXFP4
        assert weight.spec.group_size == 32
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.10


class TestNvfp4:
    """NVFP4's two scale levels."""

    def test_global_scale_is_carried_not_folded(self):
        """The per-tensor scale rides separately and is applied after."""
        rng = np.random.default_rng(5)
        out_features, in_features = 128, 256
        codes = rng.integers(0, 16, size=(out_features, in_features)).astype(np.uint8)
        # Avoid E4M3 code 127/255, which are NaN.
        scale_codes = rng.integers(100, 126, size=(out_features, in_features // 16)).astype(np.uint8)
        block_scales = np.asarray(jnp.asarray(scale_codes).view(jnp.float8_e4m3fn).astype(jnp.float32))
        global_scale = np.float32(0.037)
        reference = (E2M1_VALUES[codes] * np.repeat(block_scales, 16, axis=1) * global_scale).T

        resolved = _scheme(quant_method="modelopt_fp4", group_size=16).for_path("l")
        weight = resolved.adapter.to_canonical(
            {
                "weight": jnp.asarray(_pack_e2m1(codes)),
                "weight_scale": jnp.asarray(scale_codes),
                "weight_scale_2": jnp.asarray(global_scale),
            },
            source=resolved.source,
            target=resolved.target,
        )
        assert weight.output_scale is not None
        assert float(weight.output_scale) == pytest.approx(float(global_scale))
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.25

    def test_output_scale_is_declared_as_a_parameter(self):
        """The global scale must reach the layer as its own parameter."""
        from easydel.layers.quantization.checkpoint import canonical_param_names

        resolved = _scheme(quant_method="modelopt_fp4").for_path("l")
        assert "output_scale" in canonical_param_names(resolved.target, resolved.source)

    def test_rule_generation_refuses_the_unheld_output_scale(self):
        """No layer registers ``output_scale``, so no rule may declare it.

        Folding it into the weight is not a substitute: ejkernel's nvfp4 mode
        stores per-group scales as E4M3 codes, and the global factor is what
        keeps those inside E4M3's normal range. The seam must therefore fail
        where the missing surface is still nameable rather than emit a split
        that writes to a parameter the module tree does not have.
        """
        from easydel.layers.quantization.checkpoint import checkpoint_quant_reform_param

        resolved = _scheme(quant_method="modelopt_fp4").for_path("l")
        with pytest.raises(NotImplementedError, match="output_scale"):
            checkpoint_quant_reform_param(resolved, target_prefix="l")


class TestCompressedTensors:
    """One checkpoint, potentially several schemes."""

    def _config(self, weights, activations=None, ignore=(), targets=("Linear",)):
        """Build a compressed-tensors config body."""
        return {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {"targets": list(targets), "weights": weights, "input_activations": activations}
            },
            "ignore": list(ignore),
        }

    def test_int8_channel_scheme_round_trips(self):
        """W8A8 int8 decodes through the shared scaled-element codec."""
        rng = np.random.default_rng(6)
        out_features, in_features = 128, 256
        codes = rng.integers(-127, 128, size=(out_features, in_features)).astype(np.int8)
        scales = rng.random((out_features, 1)).astype(np.float32) * 0.01 + 0.001
        reference = (codes.astype(np.float32) * scales).T

        config = self._config(
            {"num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True},
            {"num_bits": 8, "type": "int", "dynamic": True},
        )
        resolved = _scheme(**config).for_path("model.layers.0.q_proj")
        assert resolved.target.activation.kind is ActivationQuantKind.DYNAMIC
        assert resolved.target.activation.dtype == "int8"

        weight = resolved.adapter.to_canonical(
            {"weight": jnp.asarray(codes), "weight_scale": jnp.asarray(scales)},
            source=resolved.source,
            target=resolved.target,
        )
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.01

    def test_fp8_group_reuses_the_fp8_target(self):
        """A float8 group must land on the same target as a plain fp8 file."""
        config = self._config(
            {"num_bits": 8, "type": "float", "strategy": "channel"},
            {"num_bits": 8, "type": "float", "dynamic": True},
        )
        resolved = _scheme(**config).for_path("x")
        assert resolved.target.activation.dtype == "fp8_e4m3"
        assert resolved.source_suffixes == ("weight", "weight_scale")

    def test_ignore_list_excludes_layers(self):
        """``ignore`` entries stay unquantized."""
        config = self._config({"num_bits": 8, "type": "int"}, ignore=["lm_head"])
        scheme = _scheme(**config)
        assert scheme.for_path("lm_head") is None
        assert scheme.for_path("model.layers.0.q_proj") is not None

    def test_regex_ignore_entries_are_honored(self):
        """``re:`` prefixed entries are patterns, not literal names."""
        config = self._config({"num_bits": 8, "type": "int"}, ignore=["re:.*mlp.*"])
        scheme = _scheme(**config)
        assert scheme.for_path("model.layers.0.mlp.down_proj") is None
        assert scheme.for_path("model.layers.0.self_attn.q_proj") is not None

    def test_int4_pack_quantized_round_trips(self):
        """CT ``pack-quantized``: ``weight_packed`` is ``[out, in // 8]``,
        packed sequentially along the INPUT axis with a +8 storage offset —
        the transpose of GPTQ's ``[in // 8, out]`` packing. Decoding with the
        GPTQ codec produces a wrong-shaped, wrong-valued weight."""
        from easydel.layers.quantization.checkpoint._codecs import decode_compressed_int4

        rng = np.random.default_rng(8)
        out_features, in_features, group = 128, 64, 32
        signed = rng.integers(-8, 8, size=(out_features, in_features))
        scales = rng.random((out_features, in_features // group)).astype(np.float32) + 0.1
        reference = (signed.astype(np.float32) * np.repeat(scales, group, axis=1)).T

        packed = _pack_u4_sequential((signed + 8).astype(np.uint32), axis=-1)
        assert packed.shape == (out_features, in_features // 8)

        # The codec decodes exactly: right axis, [in, out] orientation,
        # signed values via the -8 offset.
        dense = np.asarray(decode_compressed_int4(jnp.asarray(packed.astype(np.int32)), jnp.asarray(scales)))
        assert dense.shape == (in_features, out_features)
        np.testing.assert_allclose(dense, reference, rtol=1e-6, atol=1e-6)

        config = self._config({"num_bits": 4, "type": "int", "strategy": "group", "group_size": group})
        resolved = _scheme(**config).for_path("model.layers.0.q_proj")
        assert resolved.source_suffixes == ("weight_packed", "weight_scale")
        weight = resolved.adapter.to_canonical(
            {"weight_packed": jnp.asarray(packed.astype(np.int32)), "weight_scale": jnp.asarray(scales)},
            source=resolved.source,
            target=resolved.target,
        )
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.10

    def test_targeted_group_overrides_the_default(self):
        """A named target selects its own scheme over the catch-all."""
        config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "default": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "int"}},
                "narrow": {"targets": ["q_proj"], "weights": {"num_bits": 4, "type": "int", "group_size": 32}},
            },
        }
        scheme = _scheme(**config)
        assert scheme.for_path("model.layers.0.q_proj").target.bits == 4
        assert scheme.for_path("model.layers.0.k_proj").target.bits == 8

    def test_group_listing_class_and_module_targets_keeps_both(self):
        """A group may name a class AND specific modules; neither may be lost.

        Recording the catch-all and skipping the rest of the group drops the
        module targets, so those layers silently fall back to whichever
        catch-all was registered last — a different bit width here.
        """
        config = {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "narrow": {
                    "targets": ["QKVParallelLinear", "model.layers.0.mlp.down_proj"],
                    "weights": {"num_bits": 4, "type": "int", "group_size": 32},
                },
                "wide": {"targets": ["Linear"], "weights": {"num_bits": 8, "type": "int"}},
            },
        }
        scheme = _scheme(**config)

        assert scheme.for_path("model.layers.0.mlp.down_proj").target.bits == 4
        assert scheme.for_path("model.layers.0.self_attn.k_proj").target.bits == 8

    def test_nvfp4_global_scale_is_inverted(self):
        """compressed-tensors stores the NVFP4 global scale as a divisor.

        llm-compressor folds ``(FP8_MAX * FP4_MAX) / amax`` into the e4m3
        block scales, so dequantization divides by ``weight_global_scale``.
        ModelOpt's ``weight_scale_2`` is the reciprocal and multiplies.
        Carrying this one through unchanged is off by ``global**2``.
        """
        rng = np.random.default_rng(11)
        out_features, in_features = 128, 256
        codes = rng.integers(0, 16, size=(out_features, in_features)).astype(np.uint8)
        scale_codes = rng.integers(100, 126, size=(out_features, in_features // 16)).astype(np.uint8)
        block_scales = np.asarray(jnp.asarray(scale_codes).view(jnp.float8_e4m3fn).astype(jnp.float32))
        global_scale = np.float32(27.0)
        reference = (E2M1_VALUES[codes] * np.repeat(block_scales, 16, axis=1) / global_scale).T

        config = self._config({"num_bits": 4, "type": "float", "group_size": 16})
        resolved = _scheme(**config).for_path("model.layers.0.q_proj")
        assert resolved.source_suffixes == ("weight_packed", "weight_scale", "weight_global_scale")

        weight = resolved.adapter.to_canonical(
            {
                "weight_packed": jnp.asarray(_pack_e2m1(codes)),
                "weight_scale": jnp.asarray(scale_codes),
                "weight_global_scale": jnp.asarray(global_scale),
            },
            source=resolved.source,
            target=resolved.target,
        )

        assert float(weight.output_scale) == pytest.approx(1.0 / float(global_scale), rel=1e-6)
        assert _relative_error(_dequantize_canonical(weight), reference) < 0.25

    def test_asymmetric_scheme_is_refused(self):
        """``symmetric: false`` ships a zero-point this adapter cannot read.

        Parsing the flag and then decoding as symmetric offsets every group by
        a full scale, which loads cleanly and produces garbage.
        """
        config = self._config({"num_bits": 4, "type": "int", "group_size": 32, "symmetric": False})
        with pytest.raises(NotImplementedError, match="weight_zero_point"):
            _scheme(**config)

    def test_unsupported_scheme_raises(self):
        """A bit width with no decoder must fail loudly at resolution."""
        config = self._config({"num_bits": 3, "type": "int"})
        with pytest.raises(NotImplementedError, match="no decoder"):
            _scheme(**config).for_path("x")


class TestPerChannelScaleAxis:
    """A 1-D scale must land on the axis its own codec's layout uses.

    The codecs hand :func:`expand_scale` two orientations — fp8/MXFP4/NVFP4
    and compressed-tensors decode in HF ``[out, in]``, AWQ and GPTQ decode in
    ``[in, out]`` — so no fixed tie-break can be right for both. On a square
    weight both axes match a per-channel scale's length, and guessing
    transposes the scales: the load succeeds and the logits are garbage.
    """

    def test_square_weight_without_a_declared_axis_is_rejected(self):
        """Length alone cannot disambiguate; refuse rather than guess."""
        from easydel.layers.quantization.checkpoint import expand_scale

        with pytest.raises(ValueError, match="ambiguous"):
            expand_scale(jnp.arange(4, dtype=jnp.float32), (4, 4))

    def test_declared_axis_places_the_scale(self):
        """``channel_axis`` picks the axis outright, square or not."""
        from easydel.layers.quantization.checkpoint import expand_scale

        scale = jnp.asarray([1.0, 2.0, 3.0, 4.0])

        rows = np.asarray(expand_scale(scale, (4, 4), channel_axis=0))
        cols = np.asarray(expand_scale(scale, (4, 4), channel_axis=1))

        np.testing.assert_allclose(rows, np.tile(np.arange(1, 5)[:, None], (1, 4)))
        np.testing.assert_allclose(cols, np.tile(np.arange(1, 5)[None, :], (4, 1)))

    def test_fp8_square_weight_scales_per_output_channel(self):
        """The fp8 codec must scale rows of the HF ``[out, in]`` weight.

        Resolving the tie to the input axis instead transposes the scales,
        which is invisible in shape and fatal in value.
        """
        from easydel.layers.quantization.checkpoint import decode_scaled_elements

        size = 8
        codes = np.ones((size, size), np.float32)
        scale = np.arange(1, size + 1, dtype=np.float32)
        # HF [out, in] -> EasyDeL [in, out]: scaling output channels means the
        # ramp lands on the COLUMNS of the returned array.
        expected = np.tile(scale[None, :], (size, 1))

        dense = np.asarray(decode_scaled_elements(jnp.asarray(codes), jnp.asarray(scale)))

        np.testing.assert_allclose(dense, expected)


class TestGroupingAxisLimitation:
    """Pins the cost of EasyDeL's grouping axis versus the checkpoints'.

    Every checkpoint format groups scales along **input** features. EasyDeL's
    quantized linear packs ``[in, out]`` with ``transpose=False``, which groups
    along **output** features, so a checkpoint's calibration cannot be
    preserved by repacking — the groups do not correspond.

    For 8-bit targets the cost is negligible (256 levels absorb it). For 4-bit
    it is the dominant error term. These tests measure the gap directly so
    that whoever changes the layer's grouping axis has a number to beat, and
    so the improvement cannot land unnoticed.
    """

    def _exact_mxfp4_grid(self, rng, in_features, out_features):
        """Build data lying exactly on an MXFP4 grid grouped along inputs."""
        codes = rng.integers(0, 16, size=(in_features, out_features))
        exponents = rng.integers(120, 135, size=(in_features // 32, out_features)).astype(np.float32)
        return E2M1_VALUES[codes] * np.repeat(np.exp2(exponents - 127), 32, axis=0)

    def _round_trip(self, dense, axis):
        """Pack and unpack an array with a given grouping axis."""
        packed = prepack_quantized_weights(
            jnp.asarray(dense), group_size=32, bits=4, mode="mxfp4", axis=axis, transpose=False
        )
        restored = np.asarray(dequantize(packed[0], packed[1], None, group_size=32, bits=4, mode="mxfp4", axis=axis))
        return restored.T if restored.shape != dense.shape else restored

    def test_input_axis_grouping_would_be_lossless(self):
        """Grouped along inputs, an MXFP4 checkpoint round-trips exactly."""
        dense = self._exact_mxfp4_grid(np.random.default_rng(7), 256, 128)
        assert _relative_error(self._round_trip(dense, "row"), dense) == pytest.approx(0.0)

    def test_output_axis_grouping_costs_accuracy(self):
        """The axis EasyDeL actually uses loses what the checkpoint encoded.

        This asserts a *limitation*, not a desired behaviour: it should start
        failing the day the layer's grouping axis is fixed, which is exactly
        when someone should come back and delete it.
        """
        dense = self._exact_mxfp4_grid(np.random.default_rng(7), 256, 128)
        assert _relative_error(self._round_trip(dense, "col"), dense) > 0.01
