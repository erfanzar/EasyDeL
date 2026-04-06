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

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor


def _make_min_executor(*, bind_graphstate_for_aot: bool) -> ModelStepExecutor:
    from easydel.inference.esurge.runners.execution_types import BackboneOutputs

    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.use_aot_forward = True
    executor.bind_graphstate_for_aot = bind_graphstate_for_aot
    executor._cache_capacity = 8
    executor._cache = OrderedDict()
    executor._backbone_cache = OrderedDict()
    executor._lm_head_cache = OrderedDict()
    executor.graphdef = 0
    # Minimal backbone: output depends on graphstate + kv_pages.
    executor._backbone_fn = jax.jit(
        lambda graphdef, graphstate, graphother, kv_pages, metadata: BackboneOutputs(
            kv_pages=kv_pages + graphstate,
            hidden_states=kv_pages,
        ),
        static_argnums=(0,),
    )
    # Minimal lm_head: identity (just return pre-gathered hidden_states).
    executor._lm_head_fn = jax.jit(
        lambda graphdef, graphstate, graphother, hs: hs,
        static_argnums=(0,),
    )
    # Mock model for compile_lm_head (hidden_dim/dtype).
    executor.model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=1)),
        dtype=jnp.float32,
    )
    return executor


def test_model_executor_aot_bound_graphstate_uses_compile_time_constants():
    executor = _make_min_executor(bind_graphstate_for_aot=True)
    # Use a plain array as batch_metadata — the backbone fn ignores it.
    inputs = SimpleNamespace(
        kv_pages=jnp.array([1.0], dtype=jnp.float32),
        batch_metadata=jnp.array([0], dtype=jnp.int32),
    )

    executor.compile_backbone(
        num_tokens=1,
        graphdef=0,
        graphstate=jnp.array([2.0], dtype=jnp.float32),
        graphother=jnp.array([0.0], dtype=jnp.float32),
        inputs=inputs,
    )
    backbone_fn = executor.get_backbone(num_tokens=1)
    out = backbone_fn(
        jnp.array([10.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([3.0], dtype=jnp.float32),
        jnp.array([0], dtype=jnp.int32),
    )

    # backbone: kv_pages(3.0) + compile_time_graphstate(2.0) = 5.0
    np.testing.assert_allclose(np.asarray(out.kv_pages), np.asarray(jnp.array([5.0], dtype=jnp.float32)))


def test_model_executor_aot_unbound_graphstate_uses_runtime_weights():
    executor = _make_min_executor(bind_graphstate_for_aot=False)
    inputs = SimpleNamespace(
        kv_pages=jnp.array([1.0], dtype=jnp.float32),
        batch_metadata=jnp.array([0], dtype=jnp.int32),
    )

    executor.compile_backbone(
        num_tokens=1,
        graphdef=0,
        graphstate=jnp.array([2.0], dtype=jnp.float32),
        graphother=jnp.array([0.0], dtype=jnp.float32),
        inputs=inputs,
    )
    backbone_fn = executor.get_backbone(num_tokens=1)
    out = backbone_fn(
        jnp.array([10.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([3.0], dtype=jnp.float32),
        jnp.array([0], dtype=jnp.int32),
    )

    # backbone: kv_pages(3.0) + runtime_graphstate(10.0) = 13.0
    np.testing.assert_allclose(np.asarray(out.kv_pages), np.asarray(jnp.array([13.0], dtype=jnp.float32)))
