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

"""scan_layers must actually lower the qwen3_moe stack to lax.scan.

Regression: the stack used to hard-code ``trace=True`` into
``self.layers.scan`` (spectrax semantics: ``trace=True`` = python unroll),
so ``config.scan_layers`` was a silent no-op — flipping it produced
byte-identical compiled programs.
"""

import jax
import numpy as np
import spectrax as spx
from easydel.modules.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from jax import numpy as jnp
from jax._src import core as jcore


def _count_prims(jaxpr, counts):
    for eqn in jaxpr.eqns:
        counts[eqn.primitive.name] = counts.get(eqn.primitive.name, 0) + 1
        for v in eqn.params.values():
            for x in v if isinstance(v, (list, tuple)) else [v]:
                inner = getattr(x, "jaxpr", x if isinstance(x, jcore.Jaxpr) else None)
                if inner is not None:
                    _count_prims(inner, counts)


def _build(scan_layers: bool) -> Qwen3MoeForCausalLM:
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        scan_layers=scan_layers,
        attn_mechanism="vanilla",
    )
    return Qwen3MoeForCausalLM(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_scan_layers_lowers_stack_and_matches_unrolled():
    ids = jnp.asarray(np.random.default_rng(0).integers(0, 64, (1, 8)), np.int32)
    m_unrolled = _build(False)
    m_scanned = _build(True)

    out_unrolled = m_unrolled(input_ids=ids).last_hidden_state
    out_scanned = m_scanned(input_ids=ids).last_hidden_state
    np.testing.assert_allclose(np.asarray(out_unrolled), np.asarray(out_scanned), atol=1e-5)

    counts_unrolled: dict[str, int] = {}
    counts_scanned: dict[str, int] = {}
    _count_prims(jax.make_jaxpr(lambda i: m_unrolled(input_ids=i).last_hidden_state)(ids).jaxpr, counts_unrolled)
    _count_prims(jax.make_jaxpr(lambda i: m_scanned(input_ids=i).last_hidden_state)(ids).jaxpr, counts_scanned)

    # the scanned program carries more scan primitives and fewer unrolled dots
    assert counts_scanned.get("scan", 0) > counts_unrolled.get("scan", 0)
    assert counts_scanned.get("dot_general", 0) < counts_unrolled.get("dot_general", 0)


def test_router_logits_force_python_path():
    # growing per-layer router-logit tuples cannot ride a lax.scan carry;
    # requesting them must transparently fall back to the unrolled path
    ids = jnp.asarray(np.random.default_rng(1).integers(0, 64, (1, 8)), np.int32)
    m_scanned = _build(True)
    out = m_scanned(input_ids=ids, output_router_logits=True)
    assert out.router_logits is not None
    assert len(out.router_logits) == 4
