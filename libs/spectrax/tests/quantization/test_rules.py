# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the rule engine: matching, presets, and MaxText ``intmp`` parsing."""

from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import pytest
from spectrax.quantization import QuantProvider, QuantRule, resolve_qtype


def test_first_matching_rule_wins():
    """Rules are precedence-ordered, so a specific pattern shadows a catch-all."""
    provider = QuantProvider(
        [
            QuantRule(module_path="layers/0/.*", weight_qtype="int4"),
            QuantRule(module_path=".*", weight_qtype="int8"),
        ]
    )
    assert provider.rule_for_path("layers.0.mlp", "dot_general").weight_qtype == jnp.int4
    assert provider.rule_for_path("layers.1.mlp", "dot_general").weight_qtype == jnp.int8


def test_module_path_must_match_in_full():
    """Matching is a full match, so a prefix pattern does not accidentally claim children."""
    provider = QuantProvider([QuantRule(module_path="layers", weight_qtype="int8")])
    assert provider.rule_for_path("layers", "dot_general") is not None
    assert provider.rule_for_path("layers.0", "dot_general") is None


def test_slash_separators_match_spectrax_dot_paths():
    """MaxText writes ``.*/wo``; spectrax paths are dot-joined. Both must work."""
    provider = QuantProvider([QuantRule(module_path=".*/wo", weight_qtype="int8")])
    assert provider.rule_for_path("decoder.layers.0.wo", "dot_general") is not None
    assert provider.rule_for_path("decoder/layers/0/wo", "dot_general") is not None


def test_slash_inside_a_character_class_is_left_alone():
    """Separator rewriting must not corrupt a pattern that uses ``/`` deliberately."""
    provider = QuantProvider([QuantRule(module_path="a[/]b", weight_qtype="int8")])
    assert provider.rule_for_path("a/b", "dot_general") is not None


def test_op_names_filter_which_ops_a_rule_claims():
    """A rule scoped to one op must not leak into another."""
    provider = QuantProvider([QuantRule(module_path=".*", weight_qtype="int8", op_names=("ragged_dot",))])
    assert provider.rule_for_path("anything", "ragged_dot") is not None
    assert provider.rule_for_path("anything", "dot_general") is None


def test_empty_op_names_claims_every_op():
    """The default empty tuple means "all ops", matching Qwix's semantics."""
    provider = QuantProvider([QuantRule(module_path=".*", weight_qtype="int8")])
    for op in ("dot_general", "einsum", "ragged_dot"):
        assert provider.rule_for_path("anything", op) is not None


def test_plan_only_includes_ops_that_are_actually_quantized():
    """A rule with no weight type contributes nothing to a plan."""
    provider = QuantProvider([QuantRule(module_path=".*")])
    assert not provider.plan_for_path("anything")


def test_provider_rejects_non_rules():
    """Passing a dict instead of a rule fails at construction, not at trace time."""
    with pytest.raises(TypeError, match="QuantRule"):
        QuantProvider([{"module_path": ".*"}])


@pytest.mark.parametrize(
    ("spec", "expected"),
    [(4, jnp.int4), (8, jnp.int8), (3, "int3"), ("int4", jnp.int4), ("fp8", jnp.float8_e4m3fn), ("nf4", "nf4")],
)
def test_resolve_qtype_accepts_what_configs_actually_write(spec, expected):
    """Bit counts, short names and dtypes all normalize to the same types."""
    assert resolve_qtype(spec) == expected


def test_resolve_qtype_rejects_an_unsupported_width():
    """A 16-bit "quantization" is a configuration error, not a wide integer type."""
    with pytest.raises(ValueError, match="2-8 bits"):
        resolve_qtype(16)


def test_resolve_qtype_rejects_an_unknown_name():
    """An unknown name fails loudly rather than silently disabling quantization."""
    with pytest.raises(ValueError, match="Unknown quantized type"):
        resolve_qtype("int4x")


@pytest.mark.parametrize(
    ("preset", "weight", "act", "bwd"),
    [
        ("int4", jnp.int4, jnp.int4, jnp.int4),
        ("int8", jnp.int8, jnp.int8, jnp.int8),
        ("fp8", jnp.float8_e4m3fn, jnp.float8_e4m3fn, jnp.float8_e5m2),
        ("fp8_e5m2", jnp.float8_e5m2, jnp.float8_e5m2, jnp.float8_e5m2),
        ("fp4", jnp.float4_e2m1fn, jnp.float4_e2m1fn, jnp.float4_e2m1fn),
        ("w4a16", jnp.int4, None, None),
        ("w8a16", jnp.int8, None, None),
        ("nf4", "nf4", None, None),
    ],
)
def test_presets(preset, weight, act, bwd):
    """Named presets reproduce MaxText's numeric regimes."""
    rule = QuantProvider.from_preset(preset).rules[0]
    assert rule.weight_qtype == weight
    assert rule.act_qtype == act
    assert rule.bwd_qtype == bwd


def test_fp8_preset_uses_a_wider_exponent_for_gradients():
    """Gradients have a larger dynamic range than activations, so e5m2 carries them."""
    rule = QuantProvider.from_preset("fp8").rules[0]
    assert rule.act_qtype == jnp.float8_e4m3fn
    assert rule.bwd_qtype == jnp.float8_e5m2


def test_preset_can_leave_the_backward_pass_alone():
    """``quantize_backward=False`` keeps gradients in full precision."""
    assert QuantProvider.from_preset("int8", quantize_backward=False).rules[0].bwd_qtype is None


def test_unknown_preset_lists_the_known_ones():
    """A typo in a config value should say what was expected."""
    with pytest.raises(ValueError, match="Known presets"):
        QuantProvider.from_preset("int5weightonly")


def test_intmp_config_from_maxtext(tmp_path):
    """MaxText's mixed-precision JSON is read verbatim, including the catch-all."""
    config = {
        "__default__": {"w_bits": 8, "a_bits": 8},
        ".*/query": {"w_bits": 4, "tile_size": 128},
        ".*/key": {"w_bits": 4, "tile_size": 256},
        ".*/value": {"w_bits": 4, "w_scale": 0.8},
        ".*/wo": {"w_bits": 4, "tile_size": -1},
    }
    path = tmp_path / "intmp.json"
    path.write_text(json.dumps(config))
    provider = QuantProvider.from_intmp(str(path))

    # The catch-all is moved last regardless of where it appeared.
    assert provider.rules[-1].module_path == ".*"

    query = provider.rule_for_path("decoder.layers.0.query", "dot_general")
    assert query.weight_qtype == jnp.int4 and query.tile_size == 128
    assert query.is_weight_only, "omitting a_bits must mean weight-only"

    key = provider.rule_for_path("decoder.layers.0.key", "dot_general")
    assert key.tile_size == 256

    value = provider.rule_for_path("decoder.layers.0.value", "dot_general")
    assert value.weight_calibration_method == "absmax,0.8"

    wo = provider.rule_for_path("decoder.layers.0.wo", "dot_general")
    assert wo.tile_size is None, "tile_size of -1 means no subchannel"

    fallback = provider.rule_for_path("decoder.layers.0.other", "dot_general")
    assert fallback.weight_qtype == jnp.int8 and fallback.act_qtype == jnp.int8


def test_intmp_rejects_unknown_keys():
    """A misspelled key would silently disable part of the config, so it raises."""
    with pytest.raises(ValueError, match="Unknown key"):
        QuantProvider.from_intmp({"__default__": {"w_bit": 8}})


def test_tile_size_that_defeats_quantization_is_rejected():
    """One bf16 scale per four int4 values costs as much as int8; refuse it."""
    with pytest.raises(ValueError, match="effective bits"):
        QuantRule(weight_qtype="int4", tile_size=4)


def test_a_small_but_usable_tile_size_only_warns():
    """Between "wasteful" and "pointless" is a real trade, so it warns rather than raises."""
    with pytest.warns(UserWarning, match="overhead"):
        QuantRule(weight_qtype="int4", tile_size=8)


def test_a_power_of_two_scale_makes_a_small_tile_affordable():
    """MXFP4 specifies a 32-element block; that must not read as a mistake.

    A power-of-two scale is an eight-bit exponent rather than a
    sixteen-bit float, so the same tile costs half as much. Keying the
    warning on raw tile size would flag the format's own block size.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mx = QuantRule(weight_qtype="fp4", tile_size=32, power_of_two_scale=True)
    assert mx.effective_bits == pytest.approx(4.25), "an E8M0 scale costs 8 bits per block, not 16"

    free = QuantRule(weight_qtype="fp4", tile_size=32, power_of_two_scale=False)
    assert free.effective_bits == pytest.approx(4.5)

    # At a tile small enough to matter, the same block warns with a float
    # scale and is comfortable with a power-of-two one.
    with pytest.warns(UserWarning, match="overhead"):
        QuantRule(weight_qtype="fp4", tile_size=8, power_of_two_scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QuantRule(weight_qtype="fp4", tile_size=8, power_of_two_scale=True)


def test_mxfp4_preset_matches_the_published_format():
    """MXFP4 is E2M1 values with a power-of-two scale over 32-element blocks.

    This is the format DeepSeek-V4 applies to its mixture-of-experts
    weights, quantizing FP32 masters to FP4 and dequantizing to FP8 for
    the matmul, so the rule is weight-only by construction.
    """
    rule = QuantProvider.from_preset("mxfp4").rules[0]
    assert rule.weight_qtype == jnp.float4_e2m1fn
    assert rule.tile_size == 32
    assert rule.power_of_two_scale
    assert rule.is_weight_only


@pytest.mark.parametrize(
    ("qtype", "tile_size", "expected"),
    [("int4", 128, 4.125), ("int4", 256, 4.0625), ("int8", 128, 8.125), ("int4", None, 4.0)],
)
def test_effective_bits(qtype, tile_size, expected):
    """Effective width counts the subchannel scale against the payload."""
    assert QuantRule(weight_qtype=qtype, tile_size=tile_size).effective_bits == pytest.approx(expected)


@pytest.mark.parametrize("field", ["tile_size", "bwd_weight_grad_tile_size"])
@pytest.mark.parametrize("value", [-1.0, 0.0, 2.0, -8])
def test_a_tile_size_that_cannot_describe_a_tiling_is_rejected(field, value):
    """A reciprocal tile count must lie in (0, 1]; a literal count must be positive.

    The float case is the one that bites in practice. The natural way to
    derive it is ``1 / shard_count``, and a sentinel shard count of ``-1``
    turns into ``-1.0`` — which is not "no tiling", it is a request for a
    negative number of elements. MaxText's ``quantization_local_shard_count``
    defaults to exactly that sentinel, so this is a live trap rather than a
    hypothetical one.
    """
    with pytest.raises(ValueError, match=field):
        QuantRule(weight_qtype="int8", **{field: value})


@pytest.mark.parametrize("value", [1.0, 0.5, 0.125])
def test_valid_reciprocal_tile_counts_are_accepted(value):
    """A float in (0, 1] is a legitimate reciprocal tile count."""
    assert QuantRule(weight_qtype="int8", tile_size=value).tile_size == value


def test_none_disables_tiling():
    """``None`` is how tiling is switched off, not a zero or a negative."""
    rule = QuantRule(weight_qtype="int8", tile_size=None, bwd_weight_grad_tile_size=None)
    assert rule.tile_size is None
    assert rule.bwd_weight_grad_tile_size is None


def test_stochastic_rounding_mode_is_validated():
    """Only uniform noise is implemented; anything else is a typo."""
    with pytest.raises(ValueError, match="must be 'uniform' or None"):
        QuantRule(weight_qtype="int8", bwd_stochastic_rounding="gaussian")


def test_rules_are_hashable():
    """Rules ride the GraphDef as static metadata, so they must hash."""
    rule = QuantRule(weight_qtype="int4", tile_size=128, op_names=("dot_general",))
    assert hash(rule) == hash(QuantRule(weight_qtype="int4", tile_size=128, op_names=("dot_general",)))
    assert {rule: 1}[rule] == 1
