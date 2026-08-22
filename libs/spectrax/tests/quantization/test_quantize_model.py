# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :func:`spectrax.quantization.quantize_model` end to end.

The property that matters most here is that a stamped rule *survives
tracing*. A plan that is visible to an eager call but lost inside
``spx.jit`` or ``spx.grad`` produces a model that reports itself as
quantized while training at full precision — silently, and only
detectable by comparing gradients. Several tests below exist purely to
close off that failure mode.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from spectrax.quantization import (
    PLAN_ATTRIBUTE,
    QuantProvider,
    QuantRule,
    quantize_model,
    rule_for,
    unquantize_model,
)


class _Block(spx.Module):
    """A two-layer MLP block used as the quantization target."""

    def __init__(self, width: int, rngs: spx.Rngs):
        """Build the up and down projections.

        Args:
            width: Model width; the hidden layer is four times wider.
            rngs: RNG container for parameter initialization.
        """
        super().__init__()
        self.up = spx.nn.Linear(width, 4 * width, rngs=rngs, use_bias=False)
        self.down = spx.nn.Linear(4 * width, width, rngs=rngs, use_bias=False)

    def forward(self, x):
        """Apply up-projection, GELU and down-projection.

        Args:
            x: Input of shape ``(..., width)``.

        Returns:
            Output of shape ``(..., width)``.
        """
        return self.down(jax.nn.gelu(self.up(x)))


class _Net(spx.Module):
    """Embedding, a stack of blocks, and a head — enough to test path selectivity."""

    def __init__(self, width: int, depth: int, rngs: spx.Rngs):
        """Build the network.

        Args:
            width: Model width.
            depth: Number of blocks.
            rngs: RNG container for parameter initialization.
        """
        super().__init__()
        self.embed = spx.nn.Linear(width, width, rngs=rngs, use_bias=False)
        self.layers = spx.nn.Sequential(*[_Block(width, rngs) for _ in range(depth)])
        self.head = spx.nn.Linear(width, width, rngs=rngs, use_bias=False)

    def forward(self, x):
        """Run embed, blocks and head in sequence.

        Args:
            x: Input of shape ``(..., width)``.

        Returns:
            Output of shape ``(..., width)``.
        """
        return self.head(self.layers(self.embed(x)))


@pytest.fixture
def net():
    """Return a fresh, deterministically-initialized network.

    Returns:
        A ``_Net`` with width 16 and three blocks.
    """
    return _Net(16, 3, spx.Rngs(0))


@pytest.fixture
def inputs():
    """Return a deterministic input batch.

    Returns:
        A ``(2, 16)`` float32 array.
    """
    return jax.random.normal(jax.random.key(1), (2, 16), jnp.float32)


def _blocks_only():
    """Return a provider that quantizes the block linears and nothing else.

    Returns:
        A single-rule :class:`QuantProvider`.
    """
    return QuantProvider(
        [QuantRule(module_path="layers/.*", weight_qtype="int8", act_qtype="int8", op_names=("dot_general",))]
    )


def test_only_matching_modules_are_stamped(net):
    """Path selectivity is the point: embed and head must stay full precision."""
    quantize_model(net, _blocks_only())
    stamped = [path for path, module in spx.iter_modules(net) if getattr(module, PLAN_ATTRIBUTE, None)]
    assert stamped and all(path.startswith("layers.") for path in stamped)
    assert rule_for(net.embed, "dot_general") is None
    assert rule_for(net.head, "dot_general") is None
    assert rule_for(net.layers[0].up, "dot_general") is not None


def test_quantization_changes_the_output(net, inputs):
    """A stamped model must actually compute differently."""
    baseline = net(inputs)
    quantize_model(net, _blocks_only())
    assert not jnp.array_equal(net(inputs), baseline)


def test_int8_output_stays_close_to_the_baseline(net, inputs):
    """int8 is a perturbation, not a different function."""
    baseline = net(inputs)
    quantize_model(net, _blocks_only())
    assert jnp.allclose(net(inputs), baseline, rtol=5e-2, atol=5e-2)


def test_plan_survives_export_and_bind(net, inputs):
    """The plan rides the GraphDef, which is what makes it survive tracing."""
    quantize_model(net, _blocks_only())
    expected = net(inputs)

    graphdef, state = spx.export(net)
    rebound = spx.bind(graphdef, state)

    assert rule_for(rebound.layers[0].up, "dot_general") is not None
    assert jnp.array_equal(rebound(inputs), expected)


def test_structure_hash_changes_when_quantized(net):
    """Compile caches key on the GraphDef, so they must see the numeric regime change."""
    before = net.structure_hash()
    quantize_model(net, _blocks_only())
    assert net.structure_hash() != before


def test_rule_is_visible_inside_a_traced_function(net, inputs):
    """The lookup a layer performs must succeed while tracing, not only eagerly.

    Asserted directly rather than inferred from output values: this is the
    property that stops a quantized model from silently falling back to
    full precision the moment it enters ``spx.jit``.
    """
    quantize_model(net, _blocks_only())
    observed = []

    @spx.jit
    def run(module, x):
        """Record the rule seen during tracing, then run the module.

        Args:
            module: The model.
            x: Input batch.

        Returns:
            The model output.
        """
        observed.append(rule_for(module.layers[0].up, "dot_general"))
        return module(x)

    run(net, inputs)
    assert observed and observed[0] is not None
    assert observed[0].weight_qtype == jnp.int8


def test_quantization_is_applied_under_spx_jit(net, inputs):
    """The jitted forward must carry the same quantization the eager one does.

    The comparison is deliberately loose against the eager result and
    tight against the unquantized one. Rounding is discontinuous, so the
    ~1e-7 float reassociation that ``jit`` introduces can flip a
    borderline value by a full quantization step; what must hold is that
    the jitted output is quantized *at all*, which shows up as a
    difference from the full-precision baseline that is orders of
    magnitude larger than that reassociation noise.
    """
    unquantized = _Net(16, 3, spx.Rngs(0))
    baseline = unquantized(inputs)

    quantize_model(net, _blocks_only())
    eager = net(inputs)

    @spx.jit
    def run(module, x):
        """Call the module under a spectrax jit.

        Args:
            module: The model.
            x: Input batch.

        Returns:
            The model output.
        """
        return module(x)

    jitted = run(net, inputs)
    scale = float(jnp.abs(baseline).max())
    assert float(jnp.abs(jitted - baseline).max()) / scale > 1e-3, "jit path is not quantized"
    assert float(jnp.abs(jitted - eager).max()) / scale < 5e-2, "jit and eager disagree beyond a quantization step"


def test_quantization_is_visible_under_spx_grad(net, inputs):
    """Gradients must reflect the quantized forward, not the full-precision one."""
    unquantized = _Net(16, 3, spx.Rngs(0))
    quantized = _Net(16, 3, spx.Rngs(0))
    quantize_model(quantized, _blocks_only())

    def loss(module):
        """Return a scalar objective for gradient comparison.

        Args:
            module: The model.

        Returns:
            Mean squared activation.
        """
        return (module(inputs) ** 2).mean()

    reference = {path: value for _collection, path, value in spx.grad(loss)(unquantized).items()}
    actual = {path: value for _collection, path, value in spx.grad(loss)(quantized).items()}

    assert set(reference) == set(actual)
    assert any(not jnp.array_equal(reference[path], actual[path]) for path in reference)


def test_gradients_stay_full_precision_and_finite(net, inputs):
    """Master weights are never quantized; only the forward sees the narrow type."""
    quantize_model(net, _blocks_only())
    grads = spx.grad(lambda module: (module(inputs) ** 2).mean())(net)
    for _collection, _path, value in grads.items():
        assert value.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(value)))


def test_restamping_is_idempotent(net, inputs):
    """Applying the same provider twice must not compound."""
    quantize_model(net, _blocks_only())
    once = net(inputs)
    quantize_model(net, _blocks_only())
    assert jnp.array_equal(net(inputs), once)


def test_restamping_replaces_a_previous_plan(net, inputs):
    """A second provider fully supersedes the first, including where it no longer matches."""
    quantize_model(net, _blocks_only())
    quantize_model(net, QuantProvider([QuantRule(module_path="head", weight_qtype="int8")]))
    assert rule_for(net.layers[0].up, "dot_general") is None
    assert rule_for(net.head, "dot_general") is not None


def test_unquantize_restores_the_baseline_exactly(net, inputs):
    """Removing the plans must return the original function bit for bit."""
    baseline = net(inputs)
    quantize_model(net, _blocks_only())
    unquantize_model(net)
    assert jnp.array_equal(net(inputs), baseline)


def test_strict_mode_rejects_a_provider_that_matches_nothing(net):
    """Silently matching nothing is the most expensive way for this to fail."""
    provider = QuantProvider([QuantRule(module_path="does_not_exist.*", weight_qtype="int8")])
    with pytest.raises(ValueError, match="matched no module"):
        quantize_model(net, provider)


def test_non_strict_mode_warns_instead(net):
    """Callers that expect an optional match can opt out of the error."""
    provider = QuantProvider([QuantRule(module_path="does_not_exist.*", weight_qtype="int8")])
    with pytest.warns(UserWarning, match="matched no module"):
        quantize_model(net, provider, strict=False)


def test_weight_only_preset_perturbs_the_output(net, inputs):
    """A16W4 quantizes only the weight but still changes the forward pass."""
    baseline = net(inputs)
    quantize_model(net, QuantProvider.from_preset("w4a16", module_path="layers/.*"))
    assert not jnp.array_equal(net(inputs), baseline)


def test_subchannel_rule_runs_on_a_wide_model():
    """Subchannel tiling needs a contracted axis long enough to tile."""
    wide = _Net(256, 1, spx.Rngs(2))
    x = jax.random.normal(jax.random.key(3), (2, 256), jnp.float32)
    baseline = wide(x)
    quantize_model(
        wide,
        QuantProvider(
            [
                QuantRule(
                    module_path="layers/.*",
                    weight_qtype="int4",
                    act_qtype="int8",
                    tile_size=128,
                    op_names=("dot_general",),
                )
            ]
        ),
    )
    output = wide(x)
    assert bool(jnp.all(jnp.isfinite(output)))
    assert not jnp.array_equal(output, baseline)


def test_dense_general_and_einsum_layers_honour_rules():
    """``DenseGeneral`` and ``Einsum`` consult rules too, not just ``Linear``."""
    rngs = spx.Rngs(0)
    dense = spx.nn.DenseGeneral(8, in_shape=(4,), rngs=rngs, use_bias=False)
    einsum = spx.nn.Einsum("...i,io->...o", (4, 8), rngs=rngs, use_bias=False)
    x = jax.random.normal(jax.random.key(4), (3, 4), jnp.float32)

    dense_baseline, einsum_baseline = dense(x), einsum(x)

    quantize_model(dense, QuantProvider.from_preset("int8", op_names=("dot_general",)))
    quantize_model(einsum, QuantProvider.from_preset("int8", op_names=("einsum",)))

    assert not jnp.array_equal(dense(x), dense_baseline)
    assert jnp.allclose(dense(x), dense_baseline, rtol=5e-2, atol=5e-2)
    assert not jnp.array_equal(einsum(x), einsum_baseline)
    assert jnp.allclose(einsum(x), einsum_baseline, rtol=5e-2, atol=5e-2)


def test_unquantized_layers_are_bit_identical_to_before_the_feature():
    """A model never passed through ``quantize_model`` must be untouched."""
    rngs = spx.Rngs(0)
    layer = spx.nn.Linear(8, 16, rngs=rngs)
    x = jax.random.normal(jax.random.key(5), (3, 8), jnp.float32)
    expected = jax.lax.dot_general(x, layer.weight.value, (((1,), (0,)), ((), ()))) + layer.bias.value
    assert jnp.array_equal(layer(x), expected)
