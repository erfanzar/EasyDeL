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

"""Regression tests for GLM-5-Next composite-config flattening (no array compute).

The published `zai-org/GLM-5.3-Flash` checkpoint stores the text fields under a
nested ``text_config`` dict in a composite ``model_type: "glm5_next"`` config.
Keys arriving through that dict land in ``**kwargs`` even when they collide with
named ``Glm5NextTextConfig.__init__`` parameters; the schedule derivations must
still see the checkpoint's values, not the signature defaults.
"""

import pytest

pytestmark = pytest.mark.cpu_ok


def _composite(text_overrides: dict) -> dict:
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": text_overrides,
        "vision_config": {"depth": 2},
    }


def test_nested_non_default_depth_derives_matching_schedules():
    """A nested ``num_hidden_layers`` must drive every derived schedule.

    Regression: the derivations previously ran on the default 45 layers while
    the base class splatted the checkpoint's depth afterwards, so any
    non-default-depth composite config crashed ``validate_layer_type``.
    """
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(**_composite({"num_hidden_layers": 8, "hidden_size": 64}))
    assert cfg.num_hidden_layers == 8
    assert cfg.hidden_size == 64
    assert len(cfg.mlp_layer_types) == 8
    assert len(cfg.layer_types) == 8
    assert len(cfg.indexer_types) == 8
    assert cfg.mlp_layer_types[:3] == ["dense"] * 3
    assert cfg.mlp_layer_types[3:] == ["sparse"] * 5


def test_nested_explicit_schedules_are_respected():
    """Checkpoint-provided ``layer_types`` / ``indexer_types`` must not be replaced by derivations."""
    import easydel as ed

    layer_types = ["deepseek_sparse_attention"] * 4
    indexer_types = ["shared", "full", "full", "shared"]
    cfg = ed.Glm5NextTextConfig(
        **_composite(
            {
                "num_hidden_layers": 4,
                "layer_types": layer_types,
                "indexer_types": indexer_types,
                "mlp_layer_types": ["dense"] * 4,
            }
        )
    )
    assert cfg.layer_types == layer_types
    assert cfg.indexer_types == indexer_types


def test_nested_full_attention_alias_maps_to_dsa():
    """``full_attention`` entries keep their HF BC mapping to DSA through the nested path."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(
        **_composite({"num_hidden_layers": 2, "layer_types": ["full_attention", "linear_attention"]})
    )
    assert cfg.layer_types == ["deepseek_sparse_attention", "linear_attention"]


def test_nested_forget_gate_geometry_feeds_kda_cache_dict():
    """The synthesized ``linear_attn_config`` cache geometry tracks the nested values."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(**_composite({"num_hidden_layers": 4, "linear_head_dim": 32, "linear_num_heads": 4}))
    assert cfg.linear_attn_config == {"num_heads": 4, "head_k_dim": 32, "head_v_dim": 32, "d_conv": 4}


def test_nested_token_ids_and_tying_are_honored():
    """Token ids and tying arriving through the nested dict must reach the config."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(
        **_composite({"num_hidden_layers": 4, "pad_token_id": 7, "eos_token_id": [9, 10], "tie_word_embeddings": True})
    )
    assert cfg.pad_token_id == 7
    assert cfg.eos_token_id == [9, 10]
    assert cfg.tie_word_embeddings is True


def test_flat_construction_unchanged():
    """Direct (flat) construction behaves exactly as before the nested-merge repair."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(num_hidden_layers=8, hidden_size=64, pad_token_id=3)
    assert cfg.num_hidden_layers == 8
    assert cfg.hidden_size == 64
    assert cfg.pad_token_id == 3
    assert len(cfg.layer_types) == 8
    assert cfg.linear_attn_config == {"num_heads": 64, "head_k_dim": 128, "head_v_dim": 128, "d_conv": 4}


def test_default_composite_matches_flat_defaults():
    """The default-depth composite path (the published checkpoint shape) is unchanged."""
    import easydel as ed

    cfg = ed.Glm5NextTextConfig(**_composite({}))
    assert cfg.num_hidden_layers == 45
    assert len(cfg.mlp_layer_types) == 45
    assert cfg.layer_types[3] == "deepseek_sparse_attention"
    assert cfg.layer_types[0] == "linear_attention"
    assert set(cfg.indexer_types) == {"full"}
