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

"""DeepSeek-Sparse-Attention runtime settings are ``EasyDeLBaseConfig`` fields every DSA / indexer model honours.

``index_scoring_kernel``, ``dsa_sparse_training``, ``dsa_pool_training``, ``dsa_pool_query_chunk`` and
``dsa_pool_kernel`` are validated at both config entry points and survive serialization. GLM-5-Next applies its own
kpool constraints to them, and ``index_scoring_kernel="pallas"`` switches the DeepSeek-V4 / V4.1 indexers to their
fused scorer with the same scores.
"""

from __future__ import annotations

import collections

import easydel as ed
import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v4.deepseek_v4_configuration import DeepseekV4Config
from easydel.modules.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer
from easydel.modules.deepseek_v41 import DeepseekV41Config
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Indexer
from jax import numpy as jnp

HI = jax.lax.Precision.HIGHEST


def _rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def _primitive_counts(fn, *args):
    counts = collections.Counter()

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            counts.update([eqn.primitive.name])
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        walk(sub)

    walk(jax.make_jaxpr(fn)(*args).jaxpr)
    return counts


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "field,bad,good",
    [
        ("index_scoring_kernel", "fused", "pallas"),
        ("dsa_pool_kernel", "dense", "gather"),
        ("dsa_pool_query_chunk", 0, 512),
        ("dsa_pool_query_chunk", 1.5, 1024),
        ("dsa_sparse_training", "yes", True),
    ],
)
def test_dsa_settings_are_validated_by_both_config_entry_points(field, bad, good):
    with pytest.raises((TypeError, ValueError), match=field):
        ed.LlamaConfig(hidden_size=64, num_attention_heads=4, **{field: bad})
    config = ed.LlamaConfig(hidden_size=64, num_attention_heads=4)
    with pytest.raises((TypeError, ValueError), match=field):
        config.add_basic_configurations(**{field: bad})
    config.add_basic_configurations(**{field: good})
    assert getattr(config, field) == good


@pytest.mark.cpu_ok
def test_dsa_settings_have_backward_compatible_defaults_and_survive_serialization():
    default = ed.LlamaConfig(hidden_size=64, num_attention_heads=4)
    assert (
        default.index_scoring_kernel,
        default.dsa_sparse_training,
        default.dsa_pool_training,
        default.dsa_pool_query_chunk,
        default.dsa_pool_kernel,
    ) == ("xla", None, False, 2048, "bitmap")
    settings = dict(
        index_scoring_kernel="pallas",
        dsa_sparse_training=False,
        dsa_pool_training=True,
        dsa_pool_query_chunk=512,
        dsa_pool_kernel="gather",
    )
    config = ed.LlamaConfig(hidden_size=64, num_attention_heads=4, **settings)
    restored = ed.LlamaConfig.from_dict(config.to_dict())
    assert {key: getattr(restored, key) for key in settings} == settings


@pytest.mark.cpu_ok
def test_glm5_next_keeps_its_family_constraints_on_the_shared_fields():
    """GLM-5-Next accepts the shared fields and applies its own kpool constraints to them."""
    common = dict(
        hidden_size=32, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=16, qk_rope_head_dim=0,
        index_topk=8, index_kpool=4,
    )  # fmt: skip
    config = ed.Glm5NextTextConfig(dsa_pool_training=True, dsa_pool_kernel="gather", dsa_pool_query_chunk=64, **common)
    assert (config.dsa_pool_training, config.dsa_pool_kernel, config.dsa_pool_query_chunk) == (True, "gather", 64)
    with pytest.raises(ValueError, match="even index_kpool"):
        ed.Glm5NextTextConfig(dsa_pool_kernel="gather", **{**common, "index_kpool": 1, "index_topk": 7})


def _v4_indexer(kernel):
    config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=2,
        q_lora_rank=8,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=4,
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
        rms_norm_eps=1e-6,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        index_scoring_kernel=kernel,
    )
    with config.mesh:
        module = DeepseekV4Indexer(config, dtype=jnp.float32, param_dtype=jnp.float32, precision=HI, rngs=spx.Rngs(3))
    return config, module


def test_index_scoring_kernel_reaches_the_deepseek_v4_indexer():
    """Same weights; ``"pallas"`` lowers a different scoring program and gives the ``"xla"`` scores."""
    (config, off), (_, on) = _v4_indexer("xla"), _v4_indexer("pallas")
    rng = np.random.default_rng(0)
    hidden = jnp.asarray(rng.normal(size=(2, 32, 16)), jnp.float32)
    residual = jnp.asarray(rng.normal(size=(2, 32, 8)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(32, dtype=jnp.int32)[None], (2, 32))

    def scores(module):
        def fn(h, r):
            return module(h, r, positions, return_scores=True)[1]

        return fn

    with config.mesh:
        want = scores(off)(hidden, residual)
        got = scores(on)(hidden, residual)
        assert _primitive_counts(scores(on), hidden, residual) != _primitive_counts(scores(off), hidden, residual)
    finite = np.isfinite(np.asarray(want))
    np.testing.assert_array_equal(finite, np.isfinite(np.asarray(got)))
    assert _rel(np.asarray(got)[finite], np.asarray(want)[finite]) < 1e-5


def _v41_indexer(kernel):
    config = DeepseekV41Config(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=6,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        compress_ratios=[0, 2, 2, 1, 1, 1, 0],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        candidate_source_layer_id=3,
        candidate_topk_blocks=64,
        candidate_block_size=2,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=128,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        sliding_window=8,
        rms_norm_eps=1e-6,
        engram_layer_ids=[],
        dspark_target_layer_ids=[3, 4, 5],
        max_position_embeddings=512,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        index_scoring_kernel=kernel,
    )
    with config.mesh:
        module = DeepseekV41Indexer(
            config, 1, dtype=jnp.float32, param_dtype=jnp.float32, precision=HI, rngs=spx.Rngs(5)
        )
    return config, module


def test_index_scoring_kernel_reaches_the_deepseek_v41_indexer():
    """The shared-key indexer's differentiable scores switch to the fused path, with the same values and gradients."""
    (config, off), (_, on) = _v41_indexer("xla"), _v41_indexer("pallas")
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.normal(size=(2, 48, 4, 16)), jnp.float32)
    weights = jnp.asarray(rng.normal(size=(2, 48, 4)), jnp.float32)
    keys = jnp.asarray(rng.normal(size=(2, 24, 16)), jnp.float32)
    positions = jnp.broadcast_to(jnp.arange(48, dtype=jnp.int32)[None], (2, 48))

    def scores(module):
        return lambda a, b, c: module.reachable_scores(a, b, c, positions)

    with config.mesh:
        want = scores(off)(q, weights, keys)
        got = scores(on)(q, weights, keys)
        assert _primitive_counts(scores(on), q, weights, keys) != _primitive_counts(scores(off), q, weights, keys)
        finite = jnp.isfinite(want)

        def grads(module):
            def loss(a, b, c):
                s = scores(module)(a, b, c)
                return jnp.sum(jnp.where(finite, s, 0.0) * jnp.cos(jnp.arange(s.size).reshape(s.shape)))

            return jax.grad(loss, argnums=(0, 1, 2))(q, weights, keys)

        grads_off, grads_on = grads(off), grads(on)
    finite = np.asarray(finite)
    np.testing.assert_array_equal(finite, np.isfinite(np.asarray(got)))
    assert _rel(np.asarray(got)[finite], np.asarray(want)[finite]) < 1e-5
    for g_on, g_off in zip(grads_on, grads_off, strict=True):
        assert _rel(g_on, g_off) < 1e-5
