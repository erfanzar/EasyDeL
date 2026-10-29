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

"""Sparse DSA training attention on a 4-device mesh (Pallas under shard_map) vs the dense path."""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.glm_moe_dsa import modeling_glm_moe_dsa
from ejkernel.types import MaskInfo
from jax.sharding import NamedSharding, PartitionSpec
from spectrax import common_types

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu" or jax.device_count() < 4, reason="needs 4 TPU devices")

HID, SEQ = 32, 300


def _build(sparse: bool):
    config = ed.GlmMoeDsaConfig(
        vocab_size=128,
        hidden_size=HID,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
        max_position_embeddings=512,
        index_topk=32,
        index_head_dim=16,
        index_n_heads=4,
        mlp_layer_types=["dense"],
        sharding_axis_dims=(1, 1, 2, 1, 2, 1),
    )
    config.attn_dtype = jnp.float32
    config.dsa_sparse_training = sparse
    with config.mesh:
        model = ed.GlmMoeDsaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    return config, model, model.model.layers[0].self_attn


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def test_sharded_sparse_matches_dense(monkeypatch):
    calls = {"n": 0}
    real = modeling_glm_moe_dsa.sparse_selected_training_attention

    def counting(**kw):
        calls["n"] += 1
        return real(**kw)

    monkeypatch.setattr(modeling_glm_moe_dsa, "sparse_selected_training_attention", counting)
    _, _ref_model, ref_layer = _build(sparse=False)
    config, model, layer = _build(sparse=True)
    rng = np.random.default_rng(0)
    valid = np.ones((2, SEQ), bool)
    valid[1, SEQ - 9 :] = False
    ct = jnp.asarray(rng.standard_normal((2, SEQ, HID)), jnp.float32) * jnp.asarray(valid, jnp.float32)[..., None]

    with jax.default_matmul_precision("highest"), config.mesh:
        h = jax.device_put(
            jnp.asarray(rng.standard_normal((2, SEQ, HID)), jnp.float32),
            NamedSharding(config.mesh, PartitionSpec(("dp", "fsdp"), None, None)),
        )
        freqs = jnp.asarray(model.model.frequencies)
        pos = jnp.broadcast_to(jnp.arange(SEQ)[None], (2, SEQ))

        def fwd(lyr, x):
            mi = MaskInfo.dynamic_init(
                input_ids=jnp.ones((2, SEQ), jnp.int32), attention_mask=jnp.asarray(valid, jnp.int32)
            )
            return lyr.forward_mla(x, mi, pos, common_types.MODE_TRAIN, frequencies=freqs).attention_output

        ref_out = jax.jit(lambda x: fwd(ref_layer, x))(h)
        out = jax.jit(lambda x: fwd(layer, x))(h)
        assert calls["n"] >= 1, "sparse training branch was not taken"
        ref_grad = jax.jit(jax.grad(lambda x: jnp.sum(fwd(ref_layer, x) * ct)))(h)
        grad = jax.jit(jax.grad(lambda x: jnp.sum(fwd(layer, x) * ct)))(h)

    v = valid[..., None]
    assert _rel(np.asarray(out) * v, np.asarray(ref_out) * v) < 1e-4
    assert _rel(np.asarray(grad) * v, np.asarray(ref_grad) * v) < 1e-4
