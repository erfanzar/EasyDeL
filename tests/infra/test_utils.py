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

"""Tests for pure helpers in ``easydel.infra.utils``.

The module is large (~2000 LoC) and many functions need real models; this
batch covers the genuinely pure subset:

* ``quick_gelu`` and ``ACT2FN`` registry
* ``canonicalize_dtype`` -- dtype inference / promotion / inexact enforcement
* ``get_gradient_checkpoint_policy`` -- policy lookup, error paths,
  custom save/exclude name policies
* ``ActivationType`` / ``flop_activation`` -- per-activation FLOP estimate
* ``AttnMaskType.from_hf`` -- alias resolution for HF attention type strings
* ``AttnMaskDetail`` -- pytree dataclass round-trip
* ``flop_*`` family -- LM head, classification head, layernorm, attention,
  cross-attention, MLP (with/without GLU and MoE), transformer body composition
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from easydel.infra.factory import TaskType
from easydel.infra.utils import (
    ACT2FN,
    ActivationType,
    AttnMaskDetail,
    AttnMaskType,
    FlopCalcConfig,
    canonicalize_dtype,
    flop_activation,
    flop_attention,
    flop_cls_head,
    flop_cross_attention,
    flop_layernorm,
    flop_lm_head,
    flop_loss,
    flop_mlp,
    flop_seq2seq,
    flop_transformer_body,
    flop_vision_tower,
    flops_per_token,
    get_gradient_checkpoint_policy,
    quick_gelu,
)


def test_quick_gelu_matches_x_times_sigmoid_1_702x():
    x = jnp.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=jnp.float32)
    expected = x * (1.0 / (1.0 + jnp.exp(-1.702 * x)))
    assert jnp.allclose(quick_gelu(x), expected, atol=1e-6)


def test_act2fn_contains_canonical_activations():
    """Every name documented in the module's docstring/registry is present."""
    for name in (
        "gelu",
        "relu",
        "silu",
        "swish",
        "gelu_new",
        "gelu_pytorch_tanh",
        "tanh",
        "sigmoid",
        "leaky_relu",
        "glu",
        "elu",
        "softmax",
        "quick_gelu",
    ):
        assert name in ACT2FN
        assert callable(ACT2FN[name])


def test_act2fn_swish_and_silu_are_same_callable():
    """SiLU and Swish are aliases per the registry."""
    assert ACT2FN["silu"] is ACT2FN["swish"]


def test_canonicalize_dtype_returns_explicit_dtype():
    assert canonicalize_dtype(dtype=jnp.float32) == jnp.float32


def test_canonicalize_dtype_infers_from_args():

    out = canonicalize_dtype(jnp.array([1, 2, 3], dtype=jnp.int32))
    assert jnp.issubdtype(out, jnp.inexact)


def test_canonicalize_dtype_inexact_false_allows_integer():
    out = canonicalize_dtype(jnp.array([1, 2], dtype=jnp.int32), inexact=False)
    assert out == jnp.int32


def test_canonicalize_dtype_inexact_true_rejects_explicit_int():
    """When the caller asks for an inexact dtype but provides an int, it raises."""
    with pytest.raises(ValueError, match="Dtype must be inexact"):
        canonicalize_dtype(dtype=jnp.int32, inexact=True)


def test_canonicalize_dtype_ignores_none_args():
    """``None`` args are filtered out before result_type."""
    out = canonicalize_dtype(None, jnp.array([1.0], dtype=jnp.float16), None)
    assert jnp.issubdtype(out, jnp.inexact)


def test_get_gradient_checkpoint_policy_known_policy_returns_callable():
    policy = get_gradient_checkpoint_policy("nothing_saveable")
    assert callable(policy)


def test_get_gradient_checkpoint_policy_save_only_these_requires_names():
    with pytest.raises(ValueError, match="save_names"):
        get_gradient_checkpoint_policy("save_only_these_names")


def test_get_gradient_checkpoint_policy_save_only_these_with_names():
    policy = get_gradient_checkpoint_policy(
        "save_only_these_names",
        save_names=["my_marker", "other_marker"],
    )
    assert callable(policy)


def test_get_gradient_checkpoint_policy_exclude_requires_names():
    with pytest.raises(ValueError, match="exclude_names"):
        get_gradient_checkpoint_policy("save_anything_except_these_names")


def test_get_gradient_checkpoint_policy_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        get_gradient_checkpoint_policy("definitely_not_a_policy")


def test_activation_type_str_enum_values():
    assert ActivationType.RELU.value == "relu"
    assert ActivationType.GELU.value == "gelu"
    assert ActivationType("silu") is ActivationType.SILU


@pytest.mark.parametrize(
    "act,expected_per_element",
    [
        (ActivationType.RELU, 1),
        (ActivationType.SILU, 4),
        (ActivationType.SWISH, 4),
        (ActivationType.GELU, 8),
        (ActivationType.QUICK_GELU, 2),
        (ActivationType.ELU, 2),
        (ActivationType.SOFTMAX, 5),
    ],
)
def test_flop_activation_scales_with_dim(act, expected_per_element):
    assert flop_activation(act, dim=1024) == expected_per_element * 1024
    assert flop_activation(act, dim=0) == 0


def test_flop_activation_unknown_type_falls_back_to_one_per_element():
    """A string not in the table defaults to 1 FLOP / element."""
    assert flop_activation("__not_an_activation__", dim=10) == 10


@pytest.mark.parametrize(
    "hf_type,expected",
    [
        ("sliding_attention", AttnMaskType.SLIDING),
        ("full_attention", AttnMaskType.FULL),
        ("linear_attention", AttnMaskType.FULL),
        ("kda_linear_attention", AttnMaskType.FULL),
        ("hybrid", AttnMaskType.FULL),
        ("parallel_hybrid", AttnMaskType.FULL),
        ("chunk_attention", AttnMaskType.CHUNK),
        ("chunked_attention", AttnMaskType.CHUNK),
    ],
)
def test_attn_mask_type_from_hf(hf_type, expected):
    assert AttnMaskType.from_hf(hf_type) is expected


def test_attn_mask_type_from_hf_unknown_raises():
    with pytest.raises(ValueError, match="not available"):
        AttnMaskType.from_hf("not-a-real-attention-type")


def test_attn_mask_detail_required_and_optional_fields():
    detail = AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=512)
    assert detail.mask_type is AttnMaskType.SLIDING
    assert detail.size == 512
    assert detail.offset is None
    assert detail.chunks is None
    assert detail.bricks is None


def test_attn_mask_detail_with_all_fields():
    detail = AttnMaskDetail(
        mask_type=AttnMaskType.CHUNK,
        size=256,
        offset=10,
        chunks=4,
        bricks=2,
    )
    assert detail.offset == 10
    assert detail.chunks == 4
    assert detail.bricks == 2


def test_flop_layernorm_proportional_to_hidden_dim():
    assert flop_layernorm(1) == 8
    assert flop_layernorm(128) == 8 * 128


def test_flop_lm_head_formula():
    assert flop_lm_head(hidden_dim=8, vocab_size=100) == 2 * 8 * 100 + 5 * 100


def test_flop_cls_head_formula():
    assert flop_cls_head(hidden_dim=8, num_labels=10) == 2 * 8 * 10 + 5 * 10


def test_flop_loss_linear_in_num_classes():
    assert flop_loss(10) == 3 * 10 + 2
    assert flop_loss(0) == 2


def test_flop_attention_uses_default_head_dim_when_none():
    """``head_dim=None`` -> defaults to ``hidden_dim / num_heads``."""
    flops_default = flop_attention(hidden_dim=64, num_heads=8, num_kv_heads=8, head_dim=None, seq_len=16)
    flops_explicit = flop_attention(hidden_dim=64, num_heads=8, num_kv_heads=8, head_dim=8, seq_len=16)
    assert flops_default == flops_explicit


def test_flop_cross_attention_scales_with_lengths():
    """Cross-attention FLOPs grow with both encoder and decoder seq lengths."""
    f_short = flop_cross_attention(hidden_dim=64, num_heads=8, enc_seq_len=10, dec_seq_len=10)
    f_long_enc = flop_cross_attention(hidden_dim=64, num_heads=8, enc_seq_len=20, dec_seq_len=10)
    f_long_dec = flop_cross_attention(hidden_dim=64, num_heads=8, enc_seq_len=10, dec_seq_len=20)
    assert f_long_enc > f_short
    assert f_long_dec > f_short


def _basic_cfg(**overrides) -> FlopCalcConfig:
    base = dict(
        hidden_dim=128,
        intermediate_dim=512,
        num_layers=2,
        num_heads=4,
        kv_heads=4,
        head_dim=32,
        seq_len=64,
        vocab_size=1000,
    )
    base.update(overrides)
    return FlopCalcConfig(**base)


def test_flop_mlp_glu_costs_more_than_non_glu():
    """GLU has 3 weight matrices vs 2 for vanilla MLP -> larger FLOP count."""
    cfg_no_glu = _basic_cfg(glu=False)
    cfg_glu = _basic_cfg(glu=True)
    assert flop_mlp(cfg_glu, hidden_dim=128, intermediate_dim=512) > flop_mlp(
        cfg_no_glu, hidden_dim=128, intermediate_dim=512
    )


def test_flop_mlp_more_experts_per_tok_increase_cost():
    """Each additional active expert contributes a full FFN's worth of FLOPs."""
    cfg_one = _basic_cfg(num_experts_per_tok=1, num_experts=8)
    cfg_two = _basic_cfg(num_experts_per_tok=2, num_experts=8)
    assert flop_mlp(cfg_two, hidden_dim=128, intermediate_dim=512) > flop_mlp(cfg_one, 128, 512)


def test_flop_mlp_router_added_when_num_experts_gt_one():
    """The router term ``2*hidden*num_experts`` only fires for true MoE."""
    cfg_dense = _basic_cfg(num_experts=1)
    cfg_moe = _basic_cfg(num_experts=8, num_experts_per_tok=2)
    f_dense = flop_mlp(cfg_dense, 128, 512)
    f_moe = flop_mlp(cfg_moe, 128, 512)

    assert f_moe > f_dense


def test_flop_transformer_body_scales_linearly_with_layers():
    cfg = _basic_cfg(num_layers=1)
    f_one_layer = flop_transformer_body(layers=1, seq_len=64, hidden_dim=128, intermediate_dim=512, cfg=cfg)
    f_two_layers = flop_transformer_body(layers=2, seq_len=64, hidden_dim=128, intermediate_dim=512, cfg=cfg)
    assert f_two_layers == pytest.approx(2 * f_one_layer)


def test_flop_seq2seq_more_encoder_layers_costs_more():
    """A seq2seq model with deeper encoder costs more than a shallow one."""
    cfg_short = _basic_cfg(enc_num_layers=2, enc_seq_len=64)
    cfg_deep = _basic_cfg(enc_num_layers=8, enc_seq_len=64)
    assert flop_seq2seq(cfg_deep) > flop_seq2seq(cfg_short)


def test_flop_vision_tower_scales_with_vision_layers():
    """Vision-tower FLOPs scale with vision_num_layers (when seq_len > 0)."""
    cfg_one = _basic_cfg(
        vision_num_layers=1,
        vision_hidden_dim=64,
        vision_intermediate_dim=128,
        vision_num_heads=4,
        vision_seq_len=16,
    )
    cfg_four = _basic_cfg(
        vision_num_layers=4,
        vision_hidden_dim=64,
        vision_intermediate_dim=128,
        vision_num_heads=4,
        vision_seq_len=16,
    )
    assert flop_vision_tower(cfg_four) == pytest.approx(4 * flop_vision_tower(cfg_one))


def test_flop_attention_zero_seq_len_raises_division_by_zero():
    """Production-code edge case: flop_attention(seq_len=0) divides by zero.

    Recorded as a regression guard so a future fix is visible -- if the
    function ever starts handling seq_len=0 gracefully, this test will fail
    and the maintainer can update the contract.
    """
    with pytest.raises(ZeroDivisionError):
        flop_attention(hidden_dim=64, num_heads=8, num_kv_heads=8, head_dim=8, seq_len=0)


def test_flops_per_token_causal_lm_includes_lm_head_cost():
    """Removing the LM head (vocab=0) drops FLOPs significantly for a CAUSAL_LM config."""
    cfg_full = _basic_cfg(task=TaskType.CAUSAL_LM, vocab_size=10000)
    cfg_no_head = _basic_cfg(task=TaskType.CAUSAL_LM, vocab_size=0)
    assert flops_per_token(cfg_full) > flops_per_token(cfg_no_head)


def test_flops_per_token_classification_uses_num_labels():
    """Classification heads scale with num_labels, not vocab_size."""
    cfg = _basic_cfg(task=TaskType.SEQUENCE_CLASSIFICATION, num_labels=10, vocab_size=0)
    flops = flops_per_token(cfg)
    assert flops > 0


def test_flops_per_token_include_loss_adds_cost():
    """When include_loss=True, vocab-size-proportional loss FLOPs are added."""
    base = _basic_cfg(task=TaskType.CAUSAL_LM, vocab_size=10000, include_loss=False)
    with_loss = _basic_cfg(task=TaskType.CAUSAL_LM, vocab_size=10000, include_loss=True)
    assert flops_per_token(with_loss) > flops_per_token(base)
