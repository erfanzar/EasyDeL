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

"""FLOP-per-token accounting for MoE model families.

``EasyDeLBaseModule.flops_per_token`` builds a :class:`FlopCalcConfig` from the
model config. Two schema-mapping bugs are covered here:

* The MoE-layer schedule for families like ``minimax_m3_vl`` / ``mistral4`` lives
  in ``mlp_layer_types`` / ``first_k_dense_replace`` -- ``layer_types`` holds the
  *attention* schedule. Reading ``layer_types`` counted zero MoE layers and
  undercounted the trunk cost.
* The shared expert of ``hy_v3`` / ``mistral4`` / ``deepseek_v3`` has width
  ``moe_intermediate_size`` (not the dense ``intermediate_size``). Sizing it at
  the dense width overcounted the shared-expert contribution.

Each expectation is an independent hand-built :class:`FlopCalcConfig` reference
computed via the public ``flops_per_token`` FLOP helper, and each test also
asserts the reference discriminates the specific bug it guards.
"""

import dataclasses

import easydel as ed
import jax.numpy as jnp
import spectrax as spx
from easydel.infra.factory import TaskType
from easydel.infra.utils import ActivationType, FlopCalcConfig
from easydel.infra.utils import flops_per_token as flops_per_token_from_cfg

_SEQ = 128


def _build(model_cls, config):
    return model_cls.lazy_init(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=None,
        rngs=spx.Rngs(0),
    )


def _actual(model):
    return model.flops_per_token(sequence_length=_SEQ, include_loss=True, include_backward=False)


def test_hy_v3_shared_expert_uses_moe_intermediate_size():
    """hy_v3 shared expert width must be moe_intermediate_size, not the dense width."""
    config = ed.HyV3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=256,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        max_position_embeddings=128,
    )
    # Default schedule: one dense layer, three sparse (MoE) layers.
    assert config.mlp_layer_types == ["dense", "sparse", "sparse", "sparse"]

    model = _build(ed.HyV3ForCausalLM, config)
    reference = FlopCalcConfig(
        hidden_dim=64,
        intermediate_dim=256,
        num_layers=4,
        num_heads=8,
        kv_heads=4,
        head_dim=8,
        seq_len=_SEQ,
        activation_type=ActivationType.SILU,
        task=TaskType.CAUSAL_LM,
        vocab_size=256,
        include_loss=True,
        glu=True,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_dim=32,
        shared_expert_intermediate_dim=32,  # == moe_intermediate_size (the fix)
        num_moe_layers=3,
        layer_types=None,
    )
    expected = flops_per_token_from_cfg(reference)
    buggy = flops_per_token_from_cfg(dataclasses.replace(reference, shared_expert_intermediate_dim=256))

    assert _actual(model) == expected
    # Sizing the shared expert at the dense width (the bug) is a different, larger number.
    assert expected != buggy
    assert _actual(model) != buggy


def test_mistral4_counts_moe_layers_and_shared_expert_width():
    """mistral4: MoE layers come from first_k_dense_replace, shared width from moe_intermediate_size."""
    config = ed.Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=256,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=16,
        q_lora_rank=32,
        first_k_dense_replace=1,
        max_position_embeddings=128,
    )
    # layer_types is the attention schedule -- it must not drive MoE counting.
    assert set(config.layer_types) == {"full_attention"}
    assert config.head_dim == 16

    model = _build(ed.Mistral4ForCausalLM, config)
    reference = FlopCalcConfig(
        hidden_dim=64,
        intermediate_dim=256,
        num_layers=4,
        num_heads=8,
        kv_heads=4,
        head_dim=16,
        seq_len=_SEQ,
        activation_type=ActivationType.SILU,
        task=TaskType.CAUSAL_LM,
        vocab_size=256,
        include_loss=True,
        glu=True,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_dim=32,
        shared_expert_intermediate_dim=32,  # == moe_intermediate_size (the fix)
        num_moe_layers=3,  # layers 1..3 (first layer is dense)
        layer_types=["full_attention"] * 4,
    )
    expected = flops_per_token_from_cfg(reference)
    buggy_moe = flops_per_token_from_cfg(dataclasses.replace(reference, num_moe_layers=0))
    buggy_shared = flops_per_token_from_cfg(dataclasses.replace(reference, shared_expert_intermediate_dim=256))

    assert _actual(model) == expected
    assert expected != buggy_moe  # reading attention layer_types would count 0 MoE layers
    assert expected != buggy_shared  # dense-width shared expert overcounts


def test_minimax_m3_vl_counts_moe_layers_from_mlp_layer_types():
    """minimax_m3_vl: MoE layers come from mlp_layer_types, not the attention layer_types."""
    config = ed.MiniMaxM3VLTextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        dense_intermediate_size=256,
        shared_intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=8,
        num_experts_per_tok=4,
        max_position_embeddings=128,
    )
    # Every MLP is a routed MoE block; every attention layer is full attention.
    assert config.mlp_layer_types == ["sparse"] * 4
    assert set(config.layer_types) == {"full_attention"}

    model = _build(ed.MiniMaxM3VLForCausalLM, config)
    reference = FlopCalcConfig(
        hidden_dim=64,
        intermediate_dim=32,
        num_layers=4,
        num_heads=8,
        kv_heads=4,
        head_dim=8,
        seq_len=_SEQ,
        activation_type=ActivationType.SILU,
        task=TaskType.CAUSAL_LM,
        vocab_size=256,
        include_loss=True,
        glu=True,
        num_experts=8,
        num_experts_per_tok=4,
        num_shared_experts=0,
        moe_intermediate_dim=None,  # no moe_intermediate_size field -> expert width is intermediate_size
        shared_expert_intermediate_dim=0,
        num_moe_layers=4,  # all four layers are MoE (mlp_layer_types all "sparse")
        layer_types=["full_attention"] * 4,
    )
    expected = flops_per_token_from_cfg(reference)
    buggy = flops_per_token_from_cfg(dataclasses.replace(reference, num_moe_layers=0))

    assert _actual(model) == expected
    # Counting MoE layers from the attention layer_types yields 0 and undercounts.
    assert expected > buggy
    assert _actual(model) != buggy
