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

"""MLA query-temperature parity for both cache paths (accelerator only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import MLARaggedPagesCacheView
from easydel.modules.mistral4.modeling_mistral4 import Mistral4Attention
from easydel.operations import AttentionOutput
from jax import numpy as jnp


@jax.default_matmul_precision("highest")
@pytest.mark.parametrize("absorbed", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("temperature", [False, True])
def test_mistral4_mla_matches_dense_reference_for_both_cache_paths(absorbed, interleaved, temperature):
    # Isolate projections and cache storage, but execute the actual inherited MLA
    # forward, rotary transform, query hook and absorbed-weight reconstruction.
    rng = np.random.default_rng(37)
    batch, seq, heads, nope_dim, rope_dim, latent_dim, value_dim = 1, 3, 2, 2, 4, 3, 2
    query = rng.normal(size=(batch, seq, heads * (nope_dim + rope_dim))).astype(np.float32)
    compressed = rng.normal(size=(batch, seq, latent_dim + rope_dim)).astype(np.float32)
    kv_weight = rng.normal(size=(latent_dim, heads * (nope_dim + value_dim))).astype(np.float32)
    positions = np.asarray([[0, 8, 16]], dtype=np.int32)
    angles = rng.normal(size=(17, rope_dim // 2)).astype(np.float32)
    frequencies = np.concatenate((np.cos(angles), np.sin(angles)), axis=-1)
    scale = float((nope_dim + rope_dim) ** -0.5)
    returned_cache = object()

    class KVProjection:
        weight = SimpleNamespace(value=jnp.asarray(kv_weight))

        def __call__(self, x):
            return x @ self.weight.value

    def perform(**kwargs):
        if absorbed:
            scores = jnp.einsum("bsha,bta->bhst", kwargs["queries_nope"], kwargs["keys_values"])
            scores += jnp.einsum("bshd,btd->bhst", kwargs["queries_pe"], kwargs["keys_pe"])
            probabilities = jax.nn.softmax(scores * scale, axis=-1)
            result = jnp.einsum("bhst,bta->bsha", probabilities, kwargs["keys_values"])
            result = result.reshape(batch * seq, heads, latent_dim)
        else:
            scores = jnp.einsum("bshd,bthd->bhst", kwargs["query_states"], kwargs["key_states"])
            probabilities = jax.nn.softmax(scores * scale, axis=-1)
            result = jnp.einsum("bhst,bthv->bshv", probabilities, kwargs["value_states"])
        # Match the public operation result, including its optional cache field.
        # A missing cache update must retain the view returned by concatenate.
        return AttentionOutput(attention_outputs=result, attention_weights=probabilities)

    def concatenate(**kwargs):
        return kwargs["key"], kwargs["value"], kwargs["mask_info"], None, returned_cache, None

    model = object.__new__(Mistral4Attention)
    attributes = dict(
        config=SimpleNamespace(rope_interleave=interleaved),
        use_mla_lora=False,
        num_heads=heads,
        q_head_dim=nope_dim + rope_dim,
        qk_nope_head_dim=nope_dim,
        qk_rope_head_dim=rope_dim,
        kv_lora_rank=latent_dim,
        v_head_dim=value_dim,
        mla_q_proj=lambda _: jnp.asarray(query),
        mla_kv_a_proj_with_mqa=lambda _: jnp.asarray(compressed),
        mla_kv_a_layernorm=lambda x: x,
        mla_kv_b_proj=KVProjection(),
        causal=False,
        sliding_window=None,
        concatenate=concatenate,
        _softmax_aux=lambda: None,
        attention_performer=SimpleNamespace(forward=perform, softmax_scale=scale),
        _merge_heads=lambda x: x.reshape(batch, seq, heads * value_dim),
        shard_attention_prod=lambda x, **kwargs: x,
        output_projection=lambda x: x,
        llama_4_scaling_beta=0.75 if temperature else None,
        llama_4_original_max_position_embeddings=8.0 if temperature else None,
    )
    for name, value in attributes.items():
        # Logical projections are read-only properties; populate the concrete
        # checkpoint attributes that those properties resolve through instead.
        object.__setattr__(model, model.projection_mapping.get(name, name), value)
    cache = object.__new__(MLARaggedPagesCacheView) if absorbed else None
    output = Mistral4Attention.forward_mla(
        model,
        jnp.zeros((batch, seq, 1), dtype=jnp.float32),
        mask_info=None,
        position_ids=jnp.asarray(positions),
        mode="prefill",
        cache_view=cache,
        output_attentions=True,
        frequencies=jnp.asarray(frequencies),
    )

    # Independent dense NumPy reference, including both query components and
    # the position-dependent scale beyond the original context boundary.
    q = query.reshape(batch, seq, heads, nope_dim + rope_dim)
    k_latent = compressed[..., :latent_dim]
    kv = (k_latent @ kv_weight).reshape(batch, seq, heads, nope_dim + value_dim)
    cos, sin = np.cos(angles[positions])[:, :, None, :], np.sin(angles[positions])[:, :, None, :]

    def rotate(x):
        if interleaved:
            even, odd = x[..., ::2], x[..., 1::2]
            return np.stack((even * cos - odd * sin, odd * cos + even * sin), axis=-1).reshape(x.shape)
        first, second = np.split(x, 2, axis=-1)
        return np.concatenate((first * cos - second * sin, second * cos + first * sin), axis=-1)

    q = np.concatenate((q[..., :nope_dim], rotate(q[..., nope_dim:])), axis=-1)
    if temperature:
        q *= (1 + 0.75 * np.log1p(np.floor(positions / 8.0)))[:, :, None, None]
    k_pe = np.broadcast_to(rotate(compressed[..., latent_dim:][:, :, None, :]), (batch, seq, heads, rope_dim))
    k = np.concatenate((kv[..., :nope_dim], k_pe), axis=-1)
    scores = np.einsum("bshd,bthd->bhst", q, k) * scale
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhst,bthv->bshv", probabilities, kv[..., nope_dim:]).reshape(batch, seq, -1)
    np.testing.assert_allclose(np.asarray(output.attention_output), expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.asarray(output.attention_weight), probabilities, rtol=2e-4, atol=2e-4)
    assert output.cache_view is returned_cache
