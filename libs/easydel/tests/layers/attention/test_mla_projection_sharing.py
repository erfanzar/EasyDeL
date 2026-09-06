# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""The MLA projection stack is built once, in ``UnifiedAttention``.

``deepseek_v2``, ``deepseek_v3``, ``glm4_moe_lite`` and ``mistral4`` each
carried a ``define_network`` whose bodies were byte-identical once the
family-name tokens were normalised away (``glm_moe_dsa`` matched at 0.967,
differing only by its extra indexer construction). They now delegate to
:meth:`UnifiedAttention._build_mla_projections`.

The shared builder is driven entirely by each class's ``projection_mapping``,
so the HF attribute names — and therefore checkpoint compatibility — are an
input rather than something the shared code decides.
"""

import importlib

import pytest
import spectrax as spx
from easydel.layers.attention._unified import UnifiedAttention
from jax import numpy as jnp

FAMILIES = {
    "deepseek_v2": ("modeling_deepseek", "DeepseekV2Attention"),
    "deepseek_v3": ("modeling_deepseek", "DeepseekV3Attention"),
    "glm4_moe_lite": ("modeling_glm4_moe_lite", "Glm4MoeLiteAttention"),
    "mistral4": ("modeling_mistral4", "Mistral4Attention"),
}

# The HF attribute names the shared builder must still produce.
MLA_KEYS = (
    "mla_q_a_proj",
    "mla_q_a_layernorm",
    "mla_q_b_proj",
    "mla_kv_a_proj_with_mqa",
    "mla_kv_a_layernorm",
    "mla_kv_b_proj",
    "output_projection",
)


def _attention_cls(family):
    module_name, cls_name = FAMILIES[family]
    module = importlib.import_module(f"easydel.modules.{family}.{module_name}")
    return getattr(module, cls_name)


def test_shared_builder_exists_on_unified_attention():
    assert hasattr(UnifiedAttention, "_build_mla_projections")


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_delegates_instead_of_restating(family):
    """Each family keeps a thin ``define_network`` that calls the shared builder."""
    cls = _attention_cls(family)
    own = vars(cls).get("define_network")
    assert own is not None, f"{family} must keep its own typed define_network"

    import inspect

    source = inspect.getsource(own)
    assert "_build_mla_projections" in source
    # A re-fork would bring the projection construction back inline.
    assert "ColumnParallelLinear(" not in source


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_projection_mapping_still_covers_every_mla_attribute(family):
    """Checkpoint compatibility rests on this mapping staying complete."""
    mapping = _attention_cls(family).projection_mapping
    for key in MLA_KEYS:
        assert key in mapping, f"{family} lost projection_mapping[{key!r}]"
        assert isinstance(mapping[key], str) and mapping[key]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_builder_materialises_the_mapped_attributes(family):
    """The shared builder must attach every projection under its mapped name."""
    import easydel as ed

    config_cls = {
        "deepseek_v2": ed.DeepseekV2Config,
        "deepseek_v3": ed.DeepseekV3Config,
        "glm4_moe_lite": ed.Glm4MoeLiteConfig,
        "mistral4": ed.Mistral4Config,
    }[family]
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    config = config_cls(**kwargs)
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()

    cls = _attention_cls(family)
    with config.mesh:
        layer = cls(
            config=config,
            layer_idx=0,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=None,
            rngs=spx.Rngs(0),
        )

    mapping = cls.projection_mapping
    # q_proj vs the q_a/q_b LoRA chain is config-dependent; the rest is not.
    for key in ("mla_kv_a_proj_with_mqa", "mla_kv_b_proj", "output_projection"):
        assert hasattr(layer, mapping[key]), f"{family}: missing {mapping[key]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
