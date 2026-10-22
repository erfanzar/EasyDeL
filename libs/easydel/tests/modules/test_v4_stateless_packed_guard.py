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

"""DeepSeek-V4 rejects stateless packed-document calls (computation test: run on TPU/GPU)."""

import types

import easydel as ed
import jax.numpy as jnp
import pytest
import spectrax as spx


def _model():
    config = ed.DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=128,
        layer_types=["sliding_attention", "compressed_sparse_attention"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        mlp_layer_types=["moe", "moe"],
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        partial_rotary_factor=0.25,
        tie_word_embeddings=False,
    )
    config.moe_force_xla_gmm = True
    return ed.DeepseekV4ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def test_stateless_packed_document_call_is_rejected():
    model = _model()
    input_ids = jnp.arange(8, dtype=jnp.int32)[None, :]
    packed_metadata = types.SimpleNamespace(
        query_start_loc=jnp.asarray([0, 8], dtype=jnp.int32),
        num_seqs=jnp.asarray([1], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="stateless packed-document"):
        model(input_ids=input_ids, position_ids=jnp.arange(8, dtype=jnp.int32)[None, :], cache_metadata=packed_metadata)
