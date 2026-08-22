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

"""Tests for model-level quantization-aware training on real EasyDeL models.

The claim being tested is that quantization is *model-agnostic*: no model
family declares anything, and quantization is applied by matching module
paths against the live graph. So the tests build actual registered models
and assert on which of their modules were stamped and what that did to
their outputs and gradients — a synthetic two-layer stand-in would not
exercise fused projections, the mixture-of-experts grouped-matmul path, or
the default exclusions.
"""

from __future__ import annotations

import json

import easydel as ed
import jax
import jax.numpy as jnp
import pytest
import spectrax as spx

_MESH = (1, 1, 1, 1, 1, 1)


def _llama():
    """Build a tiny dense Llama model.

    Returns:
        A ``LlamaForCausalLM`` small enough to run eagerly on CPU.
    """
    config = ed.LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attn_mechanism="vanilla",
        sharding_axis_dims=_MESH,
    )
    return ed.LlamaForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def _qwen3_moe():
    """Build a tiny mixture-of-experts model.

    Returns:
        A ``Qwen3MoeForCausalLM`` with four experts.
    """
    config = ed.Qwen3MoeConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        sharding_axis_dims=_MESH,
    )
    return ed.Qwen3MoeForCausalLM(config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def _ids(length=8):
    """Return a deterministic token-id batch.

    Args:
        length: Sequence length.

    Returns:
        An ``(1, length)`` int32 array.
    """
    return jnp.ones((1, length), dtype=jnp.int32)


def _stamped_paths(model):
    """List the canonical paths of every module carrying a quantization plan.

    Args:
        model: The model to inspect.

    Returns:
        A list of module paths.
    """
    return [
        path
        for path, module in spx.iter_modules(model)
        if getattr(module, spx.quantization.PLAN_ATTRIBUTE, None) is not None
    ]


@pytest.fixture(scope="module")
def dense_baseline():
    """Return an unquantized model and its logits, shared across tests.

    Returns:
        A ``(model, logits)`` pair.
    """
    model = _llama()
    return model, model(input_ids=_ids()).logits


def test_projections_are_quantized_and_sensitive_layers_are_not():
    """The default pattern must reach the projections and skip what quantizes badly."""
    model = _llama()
    ed.apply_quantization_rules(model, "int8")
    paths = _stamped_paths(model)

    assert paths, "no module matched the default pattern"
    assert any("proj" in path for path in paths), f"no projection was stamped: {paths}"
    assert not any("lm_head" in path for path in paths)
    assert not any("embed" in path for path in paths)
    assert not any("norm" in path for path in paths)


def test_quantization_changes_the_logits(dense_baseline):
    """A stamped model must compute differently, and not wildly so."""
    _baseline_model, baseline = dense_baseline
    model = _llama()
    ed.apply_quantization_rules(model, "int8")
    logits = model(input_ids=_ids()).logits

    assert bool(jnp.all(jnp.isfinite(logits)))
    assert not jnp.array_equal(logits, baseline)
    relative = float(jnp.abs(logits - baseline).max() / jnp.abs(baseline).max())
    assert relative < 0.1, f"int8 moved the logits by {relative:.3f}, which is more than a perturbation"


@pytest.mark.parametrize("preset", ["int8", "int4", "fp8", "fp8_e5m2", "fp4", "w4a16", "w8a16", "nf4"])
def test_every_preset_runs_end_to_end_on_a_real_model(preset, dense_baseline):
    """Each named regime must produce a finite, altered forward and finite gradients.

    A matrix rather than one representative case: the presets differ in
    which operands they quantize, whether the backward is quantized, and
    whether the type is affine or a code book, and each of those picks a
    different branch inside the ops.
    """
    _baseline_model, baseline = dense_baseline
    model = _llama()
    ed.apply_quantization_rules(model, preset)

    logits = model(input_ids=_ids()).logits
    assert bool(jnp.all(jnp.isfinite(logits))), f"{preset} produced non-finite logits"
    assert not jnp.array_equal(logits, baseline), f"{preset} did not change the forward pass"

    grads = spx.grad(lambda module: (module(input_ids=_ids()).logits ** 2).mean())(model)
    for _collection, path, value in grads.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"{preset} produced non-finite gradients at {path}"


def test_an_unquantized_model_is_untouched(dense_baseline):
    """Building a model without applying rules must be bit-identical to before this feature."""
    _baseline_model, baseline = dense_baseline
    assert jnp.array_equal(_llama()(input_ids=_ids()).logits, baseline)


def test_rules_survive_export_and_bind():
    """The plan rides the GraphDef, so it must survive a checkpoint round trip."""
    model = _llama()
    ed.apply_quantization_rules(model, "int8")
    expected = model(input_ids=_ids()).logits

    graphdef, state = spx.export(model)
    rebound = spx.bind(graphdef, state)

    assert _stamped_paths(rebound)
    assert jnp.array_equal(rebound(input_ids=_ids()).logits, expected)


def test_gradients_are_finite_and_full_precision():
    """Master weights stay float32; only the forward sees the narrow type."""
    model = _llama()
    ed.apply_quantization_rules(model, "int8")
    grads = spx.grad(lambda module: (module(input_ids=_ids()).logits ** 2).mean())(model)

    leaves = list(grads.items())
    assert leaves
    for _collection, _path, value in leaves:
        assert value.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(value)))


def test_rules_survive_the_easydel_state_round_trip(dense_baseline):
    """Every trainer reaches the model through ``EasyDeLState``, so the rules must survive it.

    This is the claim that lets quantization-aware training work with no
    trainer-side wiring at all: stamp the model, build the state, and the
    rules ride the ``GraphDef`` into the compiled step.
    """
    _baseline_model, baseline = dense_baseline
    model = _llama()
    ed.apply_quantization_rules(model, "int8")
    expected = model(input_ids=_ids()).logits

    state = model.to_state()
    merged = state.merge(state.graphstate)

    assert _stamped_paths(merged)
    assert jnp.array_equal(merged(input_ids=_ids()).logits, expected)
    assert not jnp.array_equal(merged(input_ids=_ids()).logits, baseline)


def test_quantization_reaches_a_jitted_train_step():
    """A rule that is visible eagerly but lost under ``jit`` would train at full precision."""

    @spx.jit
    def step(module, batch):
        """Take the gradient of a squared-logit objective.

        Args:
            module: The model.
            batch: Token ids.

        Returns:
            The gradient state.
        """
        return spx.grad(lambda m: (m(input_ids=batch).logits ** 2).mean())(module)

    quantized = _llama()
    ed.apply_quantization_rules(quantized, "int8")
    unquantized = _llama()

    quantized_grads = {path: value for _collection, path, value in step(quantized, _ids()).items()}
    reference_grads = {path: value for _collection, path, value in step(unquantized, _ids()).items()}

    assert set(quantized_grads) == set(reference_grads)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in quantized_grads.values())
    assert any(
        not jnp.array_equal(quantized_grads[path], reference_grads[path]) for path in quantized_grads
    ), "gradients through the jitted step are identical to the unquantized model"


def test_weight_only_preset_with_subchannel_tiling(dense_baseline):
    """A16W4 with a 64-wide tile is the memory-oriented regime; it must run and perturb."""
    _baseline_model, baseline = dense_baseline
    model = _llama()
    ed.apply_quantization_rules(model, "w4a16", tile_size=64)
    logits = model(input_ids=_ids()).logits

    assert bool(jnp.all(jnp.isfinite(logits)))
    assert not jnp.array_equal(logits, baseline)


def test_mixed_precision_config_applies_per_module_overrides(tmp_path):
    """A MaxText-style ``intmp`` config must reach the module it names."""
    config = tmp_path / "intmp.json"
    config.write_text(
        json.dumps({"__default__": {"w_bits": 8}, ".*/qkv_proj": {"w_bits": 4}, ".*/q_proj": {"w_bits": 4}})
    )

    model = _llama()
    ed.apply_quantization_rules(model, "intmp", quant_cfg_path=str(config))

    int4_paths = [
        path
        for path, module in spx.iter_modules(model)
        if (rule := spx.quantization.rule_for(module, "dot_general")) is not None
        and rule.weight_qtype == jnp.int4
    ]
    int8_paths = [
        path
        for path, module in spx.iter_modules(model)
        if (rule := spx.quantization.rule_for(module, "dot_general")) is not None
        and rule.weight_qtype == jnp.int8
    ]

    assert int4_paths and all("proj" in path for path in int4_paths), int4_paths
    assert int8_paths, "the catch-all rule reached nothing"
    assert bool(jnp.all(jnp.isfinite(model(input_ids=_ids()).logits)))


def test_a_weight_projection_written_as_an_einsum_still_honours_rules():
    """A projection spelled as an einsum is still a weight matmul.

    ``DeepseekV4GroupedLinear`` is the one layer in the model zoo that
    contracts its own parameter rather than delegating to
    :class:`ParallelLinear`. Left alone it would be stamped like every
    other module and silently ignore its plan, so it looks the rule up
    itself; this pins that it does.
    """
    from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear

    layer = DeepseekV4GroupedLinear(16, 8, 4, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 3, 4, 16), jnp.float32)
    baseline = layer(x)

    spx.quantization.quantize_model(
        layer, spx.quantization.QuantProvider.from_preset("int8", op_names=("dot_general",))
    )
    quantized = layer(x)

    assert quantized.shape == baseline.shape
    assert not jnp.array_equal(quantized, baseline)
    assert jnp.allclose(quantized, baseline, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("direction", ["column", "row"])
def test_standalone_expert_linear_honours_a_ragged_dot_rule(direction):
    """The non-fused expert linear must quantize too, bias and all.

    The fused mixture-of-experts path bypasses this layer, but the layer is
    still used directly, so it has its own lookup. The bias is added after
    the quantized contraction, which is what the fused path does too.
    """
    from easydel.layers.linears import ColumnParallelMoELinear, RowParallelMoELinear

    layer_class = ColumnParallelMoELinear if direction == "column" else RowParallelMoELinear
    layer = layer_class(
        in_features=32,
        out_features=16,
        num_experts=4,
        use_bias=True,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    rows = jax.random.normal(jax.random.key(1), (8, 32), jnp.float32)
    group_sizes = jnp.array([2, 2, 2, 2], jnp.int32)
    sorted_experts = jnp.repeat(jnp.arange(4), 2)

    baseline = layer(rows, group_sizes, sorted_experts)
    spx.quantization.quantize_model(
        layer, spx.quantization.QuantProvider.from_preset("int8", op_names=("ragged_dot",))
    )
    quantized = layer(rows, group_sizes, sorted_experts)

    assert quantized.shape == baseline.shape
    assert not jnp.array_equal(quantized, baseline)
    assert jnp.allclose(quantized, baseline, rtol=0.05, atol=0.05)


def test_every_declared_op_name_is_consulted_by_some_layer():
    """No op name may be offered that nothing reads.

    spectrax declares the vocabulary but ships no grouped-matmul layer, so
    ``ragged_dot`` is only real because easydel's mixture-of-experts path
    consults it. This is the side that can see both packages, so it is the
    side that can check the vocabulary is fully covered.
    """
    import pathlib

    import easydel

    roots = [pathlib.Path(spx.__file__).parent, pathlib.Path(easydel.__file__).parent]
    consulted = set()
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for op in spx.quantization.DEFAULT_OP_NAMES:
                if f'rule_for(module, "{op}")' in text or f'rule_for(self, "{op}")' in text:
                    consulted.add(op)

    inert = set(spx.quantization.DEFAULT_OP_NAMES) - consulted
    assert not inert, (
        f"{sorted(inert)} can be named in a quantization rule but no layer in spectrax or easydel "
        f"consults it, so such a rule would be stamped and silently do nothing."
    )


def test_elarge_config_enables_quantization_aware_training(dense_baseline):
    """A ``quantization.training`` key must reach the model the same way the API does.

    Applied where the model is built rather than where the trainer runs,
    because the rules travel with the ``GraphDef`` and
    :attr:`EasyDeLState.model` rebuilds from it on every access.
    """
    from easydel.infra.elarge.builders import apply_configured_quantization_rules

    _baseline_model, baseline = dense_baseline

    untouched = apply_configured_quantization_rules(_llama(), {"quantization": {"method": "nf4"}})
    assert jnp.array_equal(untouched(input_ids=_ids()).logits, baseline), (
        "a post-training-only quantization section must not enable quantized training"
    )
    assert jnp.array_equal(
        apply_configured_quantization_rules(_llama(), {})(input_ids=_ids()).logits, baseline
    )

    quantized = apply_configured_quantization_rules(_llama(), {"quantization": {"training": "int8"}})
    assert _stamped_paths(quantized)
    assert not jnp.array_equal(quantized(input_ids=_ids()).logits, baseline)


def test_elarge_config_narrows_and_tiles():
    """The optional ``training_*`` keys must reach the provider, not be dropped."""
    from easydel.infra.elarge.builders import apply_configured_quantization_rules

    model = apply_configured_quantization_rules(
        _llama(),
        {"quantization": {"training": "w4a16", "training_module_path": ".*mlp.*", "training_tile_size": 64}},
    )

    paths = _stamped_paths(model)
    assert paths and all("mlp" in path for path in paths), paths

    rule = spx.quantization.rule_for(model.model.layers[0].mlp.gate_up_proj, "dot_general")
    assert rule.tile_size == 64
    assert rule.is_weight_only


def test_elarge_intmp_config_path(tmp_path):
    """``training: intmp`` must read the mixed-precision file the config points at."""
    from easydel.infra.elarge.builders import apply_configured_quantization_rules

    config = tmp_path / "intmp.json"
    config.write_text(json.dumps({"__default__": {"w_bits": 8}, ".*/qkv_proj": {"w_bits": 4}}))

    model = apply_configured_quantization_rules(
        _llama(), {"quantization": {"training": "intmp", "training_config_path": str(config)}}
    )

    int4 = [
        path
        for path, module in spx.iter_modules(model)
        if (rule := spx.quantization.rule_for(module, "dot_general")) is not None
        and rule.weight_qtype == jnp.int4
    ]
    assert int4 and all("qkv_proj" in path for path in int4), int4


def test_a_pattern_that_matches_nothing_is_an_error():
    """Silently training at full precision while reporting as quantized is the worst outcome."""
    model = _llama()
    with pytest.raises(ValueError, match="matched no module"):
        ed.apply_quantization_rules(model, "int8", module_path="no_such_module.*")


def test_quantization_config_round_trips_into_a_rule():
    """One configuration should describe both the training and the serving discretization."""
    config = ed.QuantizationConfig(dtype=ed.QuantizationType.CHANNELWISE, bits=4)
    rule = ed.layers.quantization.quantization_config_to_rule(config)
    assert rule.weight_qtype == jnp.int4
    assert rule.tile_size is None, "channelwise means one scale per output channel, not a subchannel tile"
    assert rule.is_weight_only


@pytest.mark.parametrize(("shard_count", "expected"), [(None, None), (-1, None), (0, None), (1, None), (8, 0.125)])
def test_weight_gradient_tile_size_rejects_sentinels(shard_count, expected):
    """A shard count that does not describe a real split must mean "no tiling".

    MaxText computes this as ``1 / quantization_local_shard_count`` with the
    config defaulting to ``-1``, which yields ``-1.0`` — a negative number
    of elements rather than a disabled tiling.
    """
    from easydel.layers.quantization._quantized_training import weight_gradient_tile_size

    assert weight_gradient_tile_size(shard_count) == expected


def test_weight_gradient_shard_count_reaches_the_rule():
    """The derived tile size must actually land on the generated rules."""
    tiled = ed.build_quantization_provider("int8", weight_gradient_shard_count=8)
    assert tiled.rules[0].bwd_weight_grad_tile_size == 0.125

    untiled = ed.build_quantization_provider("int8", weight_gradient_shard_count=-1)
    assert untiled.rules[0].bwd_weight_grad_tile_size is None


def test_a_post_training_only_format_is_refused():
    """Microscaling formats have no rule-schema equivalent; say so rather than silently ignoring."""
    config = ed.QuantizationConfig(dtype=ed.QuantizationType.MXFP4)
    with pytest.raises(ValueError, match="no quantized-training equivalent"):
        ed.layers.quantization.quantization_config_to_rule(config)


@pytest.mark.slow
@pytest.mark.parametrize(
    "preset",
    [
        pytest.param("int8", id="narrow_type_contraction"),
        pytest.param("w4a16", id="discretized_kernels"),
    ],
)
def test_mixture_of_experts_stacked_kernels_are_quantized(preset):
    """The experts are the majority of a MoE model's parameters and bypass ``ParallelLinear``.

    The fused path reads the stacked kernels directly and runs the grouped
    matmul itself, so the rule has to be stamped on the MoE module rather
    than on the expert linears. Verified by both the output changing and
    the expert kernels still receiving gradient.

    Both routes are exercised. ``int8`` quantizes activations, so the whole
    contraction is handed to the quantized ragged dot; ``w4a16`` is
    weight-only, so the kernels are discretized in place and the existing
    grouped matmul runs unchanged. They are separate code paths in
    ``_sparse_moe_call`` and a bug in either would otherwise hide behind
    the other.
    """
    baseline_model = _qwen3_moe()
    baseline = baseline_model(input_ids=_ids(4)).logits

    model = _qwen3_moe()
    ed.apply_quantization_rules(model, preset)
    paths = _stamped_paths(model)

    assert any(path.endswith("mlp") for path in paths), f"the MoE module itself was not stamped: {paths}"
    assert not any(path.endswith(".gate") for path in paths), "the router must stay full precision"

    logits = model(input_ids=_ids(4)).logits
    assert bool(jnp.all(jnp.isfinite(logits)))
    assert not jnp.array_equal(logits, baseline)

    grads = spx.grad(lambda module: (module(input_ids=_ids(4)).logits ** 2).mean())(model)
    expert_grads = [value for _collection, path, value in grads.items() if "experts" in path]
    assert expert_grads, "no expert kernel appeared in the gradient tree"
    for value in expert_grads:
        assert bool(jnp.all(jnp.isfinite(value)))
        assert float(jnp.abs(value).sum()) > 0, "an expert kernel received a zero gradient"
