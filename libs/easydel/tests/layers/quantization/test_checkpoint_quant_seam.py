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

"""The checkpoint-quantization seam, exercised through the real converter.

These tests drive ``StateDictConverter.huggingface_to_easydel`` — the same
entry point ``from_pretrained`` uses — rather than a stand-in, so they cover
the property that actually matters: several checkpoint tensors collapse into
one canonical value and expand into the quantized layer's parameters *with
their packed dtypes intact*. A packed ``uint32`` kernel silently cast to the
model's ``bfloat16`` parameter dtype is destroyed, and nothing downstream
would report it.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from easydel.layers.quantization import QuantizationType
from easydel.layers.quantization.checkpoint import (
    ActivationPolicy,
    CanonicalQuantizedWeight,
    CheckpointQuantAdapter,
    CheckpointQuantScheme,
    ConfigParse,
    IgnoreRules,
    QuantSpec,
    SourceFormat,
    checkpoint_quant_reform_param,
    get_adapter,
    register_adapter,
)
from easydel.utils.parameters_transformation import StateDictConverter

QUANT_METHOD = "easydel-test-int8"


@register_adapter(QUANT_METHOD, overwrite=True)
class _StubInt8Adapter(CheckpointQuantAdapter):
    """Minimal adapter standing in for a real format.

    Consumes ``weight`` + ``weight_scale`` and emits packed ``uint32`` codes
    with ``uint8`` scales, which is the dtype shape every real format lands
    on. The arithmetic is deliberately trivial — these tests are about the
    seam's plumbing, not about quantization numerics.
    """

    @classmethod
    def parse_config(cls, quant_config):
        """Read the stub config's ignore lists and group size."""
        return ConfigParse(
            default=SourceFormat(
                quant_method=QUANT_METHOD,
                weight_dtype="int8",
                group_size=int(quant_config.get("group_size", 64)),
            ),
            ignored=IgnoreRules(
                exact=frozenset(quant_config.get("ignored_layers", ())),
                prefixes=tuple(quant_config.get("modules_to_not_convert", ())),
            ),
        )

    @classmethod
    def target_spec(cls, source, *, expert_dim=False):
        """Map the stub source onto an affine int8 runtime target."""
        return QuantSpec(
            dtype=QuantizationType.INT8,
            group_size=source.group_size or 64,
            bits=8,
            expert_dim=expert_dim,
        )

    @classmethod
    def source_suffixes(cls, source):
        """Declare the two checkpoint tensors this stub consumes."""
        return ("weight", "weight_scale")

    @classmethod
    def to_canonical(cls, tensors, *, source, target):
        """Build a canonical weight with packed/8-bit dtypes."""
        weight = tensors["weight"]
        scale = tensors["weight_scale"]
        return CanonicalQuantizedWeight(
            quant_kernel=weight.astype(jnp.uint32),
            quant_scales=scale.astype(jnp.uint8),
            quant_biases=jnp.zeros(scale.shape, dtype=jnp.float32),
            spec=target,
        )


def _leaf(params: dict, *path: str):
    """Walk the converter's nested parameter tree.

    Args:
        params: Nested tree returned by the converter.
        *path: Successive keys to descend.

    Returns:
        The leaf array, or ``None`` when any segment is absent.
    """
    node = params
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _scheme(**quant_config) -> CheckpointQuantScheme:
    """Build a scheme from a stub ``quantization_config``.

    Args:
        **quant_config: Extra keys merged into the stub config block.

    Returns:
        The resolved :class:`CheckpointQuantScheme`.
    """
    config = {"quantization_config": {"quant_method": QUANT_METHOD, **quant_config}}
    scheme = CheckpointQuantScheme.from_hf_config(config)
    assert scheme is not None
    return scheme


class TestReformRuleThroughConverter:
    """The generated rule must load through the production converter."""

    def test_expands_into_quantized_params_preserving_packed_dtypes(self):
        """N checkpoint tensors become M params without dtype destruction."""
        resolved = _scheme().for_path("layer")
        assert resolved is not None
        rules = checkpoint_quant_reform_param(resolved, target_prefix="layer")

        state_dict = {
            "layer.weight": torch.arange(8, dtype=torch.int32).reshape(4, 2),
            "layer.weight_scale": torch.ones(2, dtype=torch.uint8),
            "other.weight": torch.ones(4, 2, dtype=torch.float32),
        }

        params = StateDictConverter.huggingface_to_easydel(
            state_dict,
            reform_param=rules,
            dtype=jnp.bfloat16,
            verbose=False,
        )

        assert _leaf(params, "layer", "quant_kernel") is not None
        assert _leaf(params, "layer", "quant_scales") is not None
        assert _leaf(params, "layer", "quant_biases") is not None

        # The model dtype is bfloat16; packed leaves must ignore it entirely.
        assert _leaf(params, "layer", "quant_kernel").dtype == jnp.uint32
        assert _leaf(params, "layer", "quant_scales").dtype == jnp.uint8

        # Sources are consumed, not left behind as stray dense leaves.
        assert _leaf(params, "layer", "weight") is None
        assert _leaf(params, "layer", "weight_scale") is None

    def test_values_survive_the_round_trip(self):
        """The packed codes reaching the params are the ones on disk."""
        resolved = _scheme().for_path("layer")
        rules = checkpoint_quant_reform_param(resolved, target_prefix="layer")
        weight = torch.arange(8, dtype=torch.int32).reshape(4, 2)

        params = StateDictConverter.huggingface_to_easydel(
            {"layer.weight": weight, "layer.weight_scale": torch.full((2,), 3, dtype=torch.uint8)},
            reform_param=rules,
            dtype=jnp.bfloat16,
            verbose=False,
        )

        np.testing.assert_array_equal(
            np.asarray(_leaf(params, "layer", "quant_kernel")),
            weight.numpy().astype(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(_leaf(params, "layer", "quant_scales")),
            np.full((2,), 3, np.uint8),
        )

    def test_unquantized_weights_are_still_cast_and_transposed(self):
        """``preserve_dtype`` must not leak onto ordinary dense weights."""
        resolved = _scheme().for_path("layer")
        rules = checkpoint_quant_reform_param(resolved, target_prefix="layer")

        params = StateDictConverter.huggingface_to_easydel(
            {
                "layer.weight": torch.ones(4, 2, dtype=torch.int32),
                "layer.weight_scale": torch.ones(2, dtype=torch.uint8),
                "dense.weight": torch.ones(4, 2, dtype=torch.float32),
            },
            reform_param=rules,
            dtype=jnp.bfloat16,
            verbose=False,
        )

        assert _leaf(params, "dense", "weight").dtype == jnp.bfloat16
        # Dense weights keep the standard [out, in] -> [in, out] transpose.
        assert _leaf(params, "dense", "weight").shape == (2, 4)


class TestRuleShape:
    """Structural guarantees the converter relies on."""

    def test_rule_key_does_not_collide_with_its_sources(self):
        """A key equal to a source would make the fusion silently never fire."""
        resolved = _scheme().for_path("layer")
        rules = checkpoint_quant_reform_param(resolved, target_prefix="layer")
        (key,) = rules
        assert key not in rules[key]["sources"]
        assert key == "layer.quant_kernel"

    def test_sibling_hosting_resolves_source_names(self):
        """A parent-hosted rule names sources relative to the parent."""
        resolved = _scheme().for_path("mlp.gate_up_proj")
        rules = checkpoint_quant_reform_param(resolved, target_prefix="gate_up_proj")
        (key,) = rules
        assert key == "gate_up_proj.quant_kernel"
        assert rules[key]["sources"] == ("gate_up_proj.weight", "gate_up_proj.weight_scale")

    def test_schema_is_accepted_by_the_converter(self):
        """The generated rule must satisfy the converter's own validator."""
        resolved = _scheme().for_path("layer")
        rules = checkpoint_quant_reform_param(resolved, target_prefix="layer")
        StateDictConverter.validate_reform_param_schema(rules)


class TestIgnoreSemantics:
    """``ignored_layers`` and ``modules_to_not_convert`` are not the same rule."""

    def test_exact_names_do_not_match_by_prefix(self):
        """An exact entry must not silently capture longer sibling names."""
        scheme = _scheme(ignored_layers=["model.layers.0.q_proj"])
        assert scheme.for_path("model.layers.0.q_proj") is None
        assert scheme.for_path("model.layers.0.q_proj_extra") is not None
        assert scheme.for_path("model.layers.10.q_proj") is not None

    def test_container_names_match_everything_beneath(self):
        """A container entry captures itself and every nested module.

        Matching follows HF's ``should_convert_module`` (start-anchored
        regex / suffix), so a same-prefix sibling like
        ``model.vision_tower_adapter`` is ALSO excluded — HF checkpoints are
        authored against exactly that behavior.
        """
        scheme = _scheme(modules_to_not_convert=["model.vision_tower"])
        assert scheme.for_path("model.vision_tower") is None
        assert scheme.for_path("model.vision_tower.encoder.0.q_proj") is None
        assert scheme.for_path("model.vision_tower_adapter") is None
        assert scheme.for_path("model.language_model.q_proj") is not None

    def test_bare_leaf_name_excludes_by_suffix(self):
        """Mixtral-AWQ ships ``modules_to_not_convert=["gate"]`` to protect
        its routers: the entry must exclude every ``block_sparse_moe.gate``
        leaf without also swallowing ``gate_proj``."""
        scheme = _scheme(modules_to_not_convert=["gate"])
        assert scheme.for_path("model.layers.0.block_sparse_moe.gate") is None
        assert scheme.for_path("model.layers.7.block_sparse_moe.gate") is None
        assert scheme.for_path("model.layers.0.mlp.gate_proj") is not None
        assert scheme.for_path("model.layers.0.self_attn.q_proj") is not None

    def test_glob_entries_exclude_their_subtrees(self):
        """gpt-oss ships glob-style entries (``model.layers.*.self_attn``);
        every projection under a matching container must stay dense."""
        scheme = _scheme(modules_to_not_convert=["model.layers.*.self_attn"])
        assert scheme.for_path("model.layers.0.self_attn.q_proj") is None
        assert scheme.for_path("model.layers.11.self_attn.o_proj") is None
        assert scheme.for_path("model.layers.0.self_attn") is None
        assert scheme.for_path("model.layers.0.mlp.down_proj") is not None


class TestFusedShardAgreement:
    """Mixed precision inside one packed weight is not representable."""

    def test_disagreeing_shards_raise(self):
        """One ignored shard must fail loudly, not load a corrupt weight."""
        scheme = _scheme(ignored_layers=["attn.k_proj"])
        with pytest.raises(ValueError, match="disagreeing quantization schemes"):
            scheme.for_fused("attn.qkv_proj", ["attn.q_proj", "attn.k_proj", "attn.v_proj"])

    def test_agreeing_shards_resolve(self):
        """Uniformly quantized shards produce one scheme for the fused module."""
        resolved = _scheme().for_fused("attn.qkv_proj", ["attn.q_proj", "attn.k_proj", "attn.v_proj"])
        assert resolved is not None
        assert resolved.path == "attn.qkv_proj"

    def test_uniformly_ignored_shards_stay_dense(self):
        """When every shard is ignored the fused module is not quantized."""
        scheme = _scheme(modules_to_not_convert=["attn"])
        assert scheme.for_fused("attn.qkv_proj", ["attn.q_proj", "attn.k_proj", "attn.v_proj"]) is None


class TestSchemeDiscovery:
    """Locating ``quantization_config`` across checkpoint layouts."""

    def test_absent_config_yields_no_scheme(self):
        """An unquantized checkpoint resolves to ``None``, not an error."""
        assert CheckpointQuantScheme.from_hf_config({"hidden_size": 8}) is None

    def test_nested_text_config_is_found(self):
        """Multimodal checkpoints nest the block under the text tower."""
        scheme = CheckpointQuantScheme.from_hf_config(
            {"text_config": {"quantization_config": {"quant_method": QUANT_METHOD}}}
        )
        assert scheme is not None
        assert scheme.quant_method == QUANT_METHOD

    def test_missing_quant_method_raises(self):
        """A quantization block with no method is malformed, not ignorable."""
        with pytest.raises(ValueError, match="quant_method"):
            CheckpointQuantScheme.from_hf_config({"quantization_config": {"bits": 4}})

    def test_unregistered_method_lists_alternatives(self):
        """An unsupported format must say what is supported."""
        with pytest.raises(NotImplementedError, match="Available"):
            get_adapter("definitely-not-a-real-format")


class TestQuantSpecValidation:
    """The spec must reject targets the kernels cannot execute."""

    def test_invalid_group_size_for_mxfp4_is_rejected(self):
        """MXFP4 is pinned to group 32; anything else is unrunnable."""
        with pytest.raises(ValueError):
            QuantSpec(dtype=QuantizationType.MXFP4, group_size=64, bits=4)

    def test_valid_target_exposes_kernel_parameters(self):
        """A valid spec resolves to the ejkernel mode tuple."""
        spec = QuantSpec(dtype=QuantizationType.MXFP4, group_size=32, bits=4)
        assert spec.mode == "mxfp4"
        assert spec.needs_biases is False

    def test_affine_targets_require_biases(self):
        """Affine schemes carry zero-points, and the spec knows it."""
        assert QuantSpec(dtype=QuantizationType.INT8, group_size=64, bits=8).needs_biases is True


class TestCanonicalInvariants:
    """Adapter bugs must surface at conversion time, not inside a kernel."""

    def _spec(self, dtype=QuantizationType.MXFP4, group_size=32, bits=4, **kwargs):
        """Build a spec for invariant tests."""
        return QuantSpec(dtype=dtype, group_size=group_size, bits=bits, **kwargs)

    def test_missing_biases_for_affine_target_raises(self):
        """An affine target without zero-points is incomplete."""
        with pytest.raises(ValueError, match="requires quant_biases"):
            CanonicalQuantizedWeight(
                quant_kernel=jnp.zeros((4, 2), jnp.uint32),
                quant_scales=jnp.zeros((2,), jnp.uint8),
                quant_biases=None,
                spec=self._spec(dtype=QuantizationType.INT8, group_size=64, bits=8),
            )

    def test_unexpected_biases_for_non_affine_target_raises(self):
        """Biases on a scheme that has none signals a mismatched adapter."""
        with pytest.raises(ValueError, match="does not use quant_biases"):
            CanonicalQuantizedWeight(
                quant_kernel=jnp.zeros((4, 2), jnp.uint32),
                quant_scales=jnp.zeros((2,), jnp.uint8),
                quant_biases=jnp.zeros((2,), jnp.float32),
                spec=self._spec(),
            )

    def test_static_activation_requires_its_scale(self):
        """A calibrated policy without its scale would silently skip scaling."""
        with pytest.raises(ValueError, match="static activation policy requires"):
            CanonicalQuantizedWeight(
                quant_kernel=jnp.zeros((4, 2), jnp.uint32),
                quant_scales=jnp.zeros((2,), jnp.uint8),
                quant_biases=None,
                spec=self._spec(activation=ActivationPolicy.static()),
            )

    def test_activation_scale_without_static_policy_raises(self):
        """A stray scale means the policy and the payload disagree."""
        with pytest.raises(ValueError, match="only valid for a static"):
            CanonicalQuantizedWeight(
                quant_kernel=jnp.zeros((4, 2), jnp.uint32),
                quant_scales=jnp.zeros((2,), jnp.uint8),
                quant_biases=None,
                spec=self._spec(),
                activation_scale=jnp.ones((), jnp.float32),
            )

    def test_dtype_policy_without_quantization_raises(self):
        """Requesting an activation dtype while not quantizing is incoherent."""
        with pytest.raises(ValueError, match="cannot carry dtype"):
            ActivationPolicy(dtype="int8")


class TestOptionalParams:
    """Optional payload members appear only when the scheme calls for them."""

    def test_global_scale_source_adds_an_output_scale_split(self):
        """A two-level source carries its per-tensor scale as its own param."""
        source = SourceFormat(quant_method=QUANT_METHOD, has_global_scale=True)
        target = QuantSpec(dtype=QuantizationType.NVFP4, group_size=16, bits=4)
        from easydel.layers.quantization.checkpoint import canonical_param_names

        assert "output_scale" in canonical_param_names(target, source)

    def test_plain_source_has_no_output_scale(self):
        """Single-level sources must not declare a split that never arrives."""
        source = SourceFormat(quant_method=QUANT_METHOD)
        target = QuantSpec(dtype=QuantizationType.NVFP4, group_size=16, bits=4)
        from easydel.layers.quantization.checkpoint import canonical_param_names

        assert "output_scale" not in canonical_param_names(target, source)
