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

"""Qwen3-VL-MoE reuses the Qwen3-VL vision tower instead of restating it.

The MoE family used to carry its own copy of the whole vision stack — patch
embed, patch merger, MLP, attention, block, tower — plus the text attention and
the multimodal-merge helpers. Every one of those compared code-identical to the
dense family once docstrings were stripped, so they now come from
``modeling_qwen3_vl`` directly.

These tests pin the three things that made the collapse safe, so a future edit
that re-forks the tower fails here rather than silently drifting again.
"""

import numpy as np
import pytest
from easydel.modules.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VisionTransformerPretrainedModel,
    Qwen3VLTextAttention,
    apply_rotary_pos_emb_vision,
    rotate_half,
)
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeTextAttention,
    Qwen3VLMoeVisionTransformerPretrainedModel,
)
from easydel.modules.qwen3_vl_moe.qwen3_vl_moe_configuration import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from jax import numpy as jnp


def test_moe_vision_tower_reuses_dense_tower():
    """The MoE tower must be the dense tower, not a second copy of it."""
    assert issubclass(Qwen3VLMoeVisionTransformerPretrainedModel, Qwen3VisionTransformerPretrainedModel)
    # A re-fork would give the subclass its own body; inheriting leaves it empty.
    assert "__init__" not in vars(Qwen3VLMoeVisionTransformerPretrainedModel)


def test_moe_text_attention_is_the_dense_one():
    """The MoE text attention was code-identical, so it is the same object."""
    assert Qwen3VLMoeTextAttention is Qwen3VLTextAttention


def _reference_full_width_rope(q, k, cos, sin):
    """The pre-refactor implementation: rotate the entire trailing axis."""
    oq, ok = q.dtype, k.dtype
    q, k = q.astype("f4"), k.astype("f4")
    cos = jnp.expand_dims(cos, -2).astype("f4")
    sin = jnp.expand_dims(sin, -2).astype("f4")
    return (((q * cos) + (rotate_half(q) * sin)).astype(oq), ((k * cos) + (rotate_half(k) * sin)).astype(ok))


@pytest.mark.parametrize("head_dim", [32, 64])
def test_vision_rope_matches_full_width_reference(head_dim):
    """Partial-rotary form is bit-identical on the tower's actual table layout.

    Both towers build their tables as ``concatenate([freqs, freqs], -1)``, so
    ``cos`` spans the full ``head_dim`` and the pass-through tail is empty. The
    generalised implementation must therefore reproduce the old full-width
    rotation exactly — not approximately.
    """
    rng = np.random.default_rng(head_dim)
    seq, heads, ro = 7, 3, head_dim // 2
    q = jnp.asarray(rng.standard_normal((seq, heads, head_dim)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((seq, heads, head_dim)), jnp.float32)

    half = jnp.asarray(rng.standard_normal((seq, ro)), jnp.float32)
    table = jnp.concatenate([half, half], axis=-1)
    cos, sin = jnp.cos(table), jnp.sin(table)

    got_q, got_k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    want_q, want_k = _reference_full_width_rope(q, k, cos, sin)

    assert jnp.array_equal(got_q, want_q)
    assert jnp.array_equal(got_k, want_k)


def test_vision_rope_leaves_tail_untouched_when_table_is_half_width():
    """A half-width table must rotate only the leading channels.

    This is the behaviour the full-width form could not express at all — it
    raised a broadcasting error — and it is why the generalised version is the
    one kept for both families.
    """
    rng = np.random.default_rng(0)
    seq, heads, head_dim, ro = 5, 2, 64, 32
    q = jnp.asarray(rng.standard_normal((seq, heads, head_dim)), jnp.float32)
    k = jnp.asarray(rng.standard_normal((seq, heads, head_dim)), jnp.float32)
    table = jnp.asarray(rng.standard_normal((seq, ro)), jnp.float32)

    got_q, got_k = apply_rotary_pos_emb_vision(q, k, jnp.cos(table), jnp.sin(table))

    assert jnp.array_equal(got_q[..., ro:], q[..., ro:])
    assert jnp.array_equal(got_k[..., ro:], k[..., ro:])
    assert not jnp.array_equal(got_q[..., :ro], q[..., :ro])


def test_composite_config_accepts_prebuilt_sub_configs():
    """``vision_config``/``text_config`` accept objects, as the signature says.

    Only ``dict`` and ``None`` were handled; an already-built config fell
    through both branches, so the attribute was never assigned and the next
    read raised ``AttributeError``.
    """
    vision = Qwen3VLMoeVisionConfig(depth=2, hidden_size=64)
    text = Qwen3VLMoeTextConfig(hidden_size=64, num_hidden_layers=2)
    config = Qwen3VLMoeConfig(vision_config=vision, text_config=text)

    assert config.vision_config is vision
    assert config.text_config is text
    assert config.vision_config.depth == 2
    assert config.text_config.num_hidden_layers == 2


def test_composite_config_still_accepts_dicts_and_none():
    """The pre-existing dict and default paths are unchanged."""
    from_dict = Qwen3VLMoeConfig(vision_config={"depth": 3}, text_config={"num_hidden_layers": 4})
    assert from_dict.vision_config.depth == 3
    assert from_dict.text_config.num_hidden_layers == 4

    defaulted = Qwen3VLMoeConfig()
    assert isinstance(defaulted.vision_config, Qwen3VLMoeVisionConfig)
    assert isinstance(defaulted.text_config, Qwen3VLMoeTextConfig)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_explicit_empty_deepstack_indexes_are_preserved():
    from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLVisionConfig

    assert Qwen3VLVisionConfig(deepstack_visual_indexes=[]).deepstack_visual_indexes == []
    assert Qwen3VLMoeVisionConfig(deepstack_visual_indexes=[]).deepstack_visual_indexes == []


def test_moe_sliding_window_applies_to_leading_layers_like_dense_qwen3_vl():
    config = Qwen3VLMoeTextConfig(
        num_hidden_layers=6,
        use_sliding_window=True,
        sliding_window=128,
        max_window_layers=4,
    )
    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ]
    assert set(config.get_mask_details()) == {0, 1, 2, 3}


def test_moe_text_and_vision_registry_metadata():
    from easydel.infra.factory import TaskType, registry
    from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel

    text = registry.get_module_registration(TaskType.BASE_MODULE, "qwen3_vl_moe")
    assert text.module is Qwen3VLMoeTextModel
    assert text.config is Qwen3VLMoeTextConfig

    vision = registry.get_module_registration(TaskType.BASE_VISION, "qwen3_vl_moe")
    assert vision.module is Qwen3VLMoeVisionTransformerPretrainedModel
    assert vision.config is Qwen3VLMoeConfig
    assert Qwen3VLMoeVisionTransformerPretrainedModel.config_class is Qwen3VLMoeVisionConfig


def test_dense_and_moe_sliding_metadata_share_one_schedule():
    from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLTextConfig

    for cls in (Qwen3VLTextConfig, Qwen3VLMoeTextConfig):
        cfg = cls(num_hidden_layers=4, use_sliding_window=True, sliding_window=128, max_window_layers=2)
        assert cfg.layer_types == ["sliding_attention", "sliding_attention", "full_attention", "full_attention"]
        assert set(cfg.get_mask_details()) == {0, 1}
        with pytest.raises(ValueError, match="layer_types"):
            cls(num_hidden_layers=4, layer_types=["full_attention"])


def test_moe_video_rope_schedule_matches_dense_family():
    from easydel.modules.qwen3_vl.modeling_qwen3_vl import get_rope_index as dense_rope
    from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import get_rope_index as moe_rope

    ids = np.array([[99, 98, 99, 98]], np.int32)
    kwargs = dict(
        input_ids=ids,
        video_grid_thw=np.array([[2, 2, 2]], np.int32),
        attention_mask=np.ones_like(ids),
        spatial_merge_size=2,
        video_token_id=98,
        vision_start_token_id=99,
        second_per_grid_ts=[0.5, 0.5],
    )
    dense_pos, dense_delta = dense_rope(**kwargs)
    moe_pos, moe_delta = moe_rope(**kwargs)
    np.testing.assert_array_equal(moe_pos, dense_pos)
    np.testing.assert_array_equal(moe_delta, dense_delta)


def test_moe_deepstack_casts_features_and_leaves_text_positions_unchanged():
    from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextModel

    hidden = jnp.arange(24, dtype=jnp.float16).reshape(1, 3, 8)
    mask = jnp.array([[False, True, False]])
    visual = jnp.full((1, 8), 2.0, jnp.float32)
    got = Qwen3VLMoeTextModel._deepstack_process(object(), hidden, mask, visual)
    assert got.dtype == hidden.dtype
    assert jnp.array_equal(got[:, 0], hidden[:, 0])
    assert jnp.array_equal(got[:, 2], hidden[:, 2])
    assert jnp.array_equal(got[:, 1], hidden[:, 1] + jnp.asarray(2, hidden.dtype))


def test_moe_vlm_forward_returns_router_aux_loss(monkeypatch):
    from types import SimpleNamespace

    import easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe as modeling

    monkeypatch.setattr(modeling, "apply_logical_sharding", lambda x, **kwargs: x)

    class Dummy:
        config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            runtime_sharding_resolver=None,
            get_text_config=lambda: SimpleNamespace(output_router_logits=True),
        )

        def model(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=jnp.ones((1, 2, 4)),
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                rope_deltas=None,
                router_logits=(jnp.ones((2, 3)),),
            )

        def compute_lm_logits(self, hidden):
            return hidden

        def apply_logit_cap(self, logits):
            return logits

        def compute_router_aux_loss(self, outputs):
            return jnp.asarray(1.25)

    out = modeling.Qwen3VLMoeForConditionalGeneration.forward(Dummy(), input_ids=jnp.ones((1, 2), jnp.int32))
    assert out.aux_loss == jnp.asarray(1.25)


def test_dense_explicit_layer_types_drive_mask_metadata():
    from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLTextConfig

    cfg = Qwen3VLTextConfig(
        num_hidden_layers=4,
        use_sliding_window=True,
        sliding_window=128,
        layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
    )
    assert set(cfg.get_mask_details()) == {1, 3}


def test_qwen3_vl_attention_disables_window_for_full_layers(monkeypatch):
    import easydel.modules.qwen3_vl.modeling_qwen3_vl as modeling
    from easydel.modules.qwen3_vl.qwen3_vl_configuration import Qwen3VLTextConfig

    calls = []
    monkeypatch.setattr(modeling.UnifiedAttention, "__init__", lambda self, **kwargs: calls.append(kwargs))
    cfg = Qwen3VLTextConfig(
        num_hidden_layers=2,
        use_sliding_window=True,
        sliding_window=128,
        layer_types=["sliding_attention", "full_attention"],
    )
    modeling.Qwen3VLTextAttention(cfg, rngs=object(), layer_idx=0)
    modeling.Qwen3VLTextAttention(cfg, rngs=object(), layer_idx=1)
    assert [call["sliding_window"] for call in calls] == [128, None]


def test_composite_configs_accept_generic_mapping_inputs():
    from collections import ChainMap

    from easydel.modules.qwen3_vl.qwen3_vl_configuration import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    dense = Qwen3VLConfig(
        vision_config=ChainMap({"depth": 2}),
        text_config=ChainMap({"num_hidden_layers": 3}),
    )
    moe = Qwen3VLMoeConfig(
        vision_config=ChainMap({"depth": 2}),
        text_config=ChainMap({"num_hidden_layers": 3}),
    )
    assert isinstance(dense.vision_config, Qwen3VLVisionConfig)
    assert isinstance(dense.text_config, Qwen3VLTextConfig)
    assert isinstance(moe.vision_config, Qwen3VLMoeVisionConfig)
    assert isinstance(moe.text_config, Qwen3VLMoeTextConfig)
