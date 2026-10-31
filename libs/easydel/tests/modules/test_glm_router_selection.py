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

"""Public grouped-router selection regressions (accelerator-only computation)."""

import numpy as np
import spectrax as spx
from easydel.modules.glm4_moe import Glm4MoeConfig
from easydel.modules.glm4_moe.modeling_glm4_moe import Glm4MoeTopKRouter
from jax import numpy as jnp


def test_public_router_selection_uses_scores_and_live_bias():
    config = Glm4MoeConfig(
        hidden_size=8,
        num_attention_heads=2,
        n_routed_experts=4,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=1,
    )
    router = Glm4MoeTopKRouter(config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    scores = jnp.array([[0.9, 0.8, 0.7, 0.6], [0.2, 0.1, 0.8, 0.7]])
    np.testing.assert_array_equal(router.get_selected_experts(scores), [[0], [2]])
    router.e_score_correction_bias.value = jnp.array([0.0, 0.0, 0.0, 2.0])
    np.testing.assert_array_equal(router.get_selected_experts(scores), [[3], [3]])


def test_public_router_selection_supports_single_expert_groups():
    config = Glm4MoeConfig(
        hidden_size=8,
        num_attention_heads=2,
        n_routed_experts=4,
        n_group=4,
        topk_group=2,
        num_experts_per_tok=2,
    )
    router = Glm4MoeTopKRouter(config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    np.testing.assert_array_equal(router.get_selected_experts(jnp.array([[0.1, 0.9, 0.2, 0.8]])), [[1, 3]])
