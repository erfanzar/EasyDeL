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
    executor = ModelStepExecutor.__new__(ModelStepExecutor)
    executor.use_aot_forward = True
    executor.bind_graphstate_for_aot = bind_graphstate_for_aot
    executor._cache_capacity = 8
    executor._cache = OrderedDict()
    executor.graphdef = 0
    # Minimal step body: output depends on graphstate + kv_pages.
    executor._model_step_fn = jax.jit(
        lambda graphdef, graphstate, graphother, kv_pages, metadata: kv_pages + graphstate,
        static_argnums=(0,),
    )
    return executor


def test_model_executor_aot_bound_graphstate_uses_compile_time_constants():
    executor = _make_min_executor(bind_graphstate_for_aot=True)
    inputs = SimpleNamespace(
        kv_pages=jnp.array([1.0], dtype=jnp.float32),
        batch_metadata=jnp.array([0], dtype=jnp.int32),
    )

    executor.compile(
        num_tokens=1,
        padded_num_reqs=1,
        graphdef=0,
        graphstate=jnp.array([2.0], dtype=jnp.float32),
        graphother=jnp.array([0.0], dtype=jnp.float32),
        inputs=inputs,
    )
    fn = executor.get_compiled(num_tokens=1, padded_num_reqs=1)
    out = fn(
        jnp.array([10.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([3.0], dtype=jnp.float32),
        jnp.array([0], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(jnp.array([5.0], dtype=jnp.float32)))


def test_model_executor_aot_unbound_graphstate_uses_runtime_weights():
    executor = _make_min_executor(bind_graphstate_for_aot=False)
    inputs = SimpleNamespace(
        kv_pages=jnp.array([1.0], dtype=jnp.float32),
        batch_metadata=jnp.array([0], dtype=jnp.int32),
    )

    executor.compile(
        num_tokens=1,
        padded_num_reqs=1,
        graphdef=0,
        graphstate=jnp.array([2.0], dtype=jnp.float32),
        graphother=jnp.array([0.0], dtype=jnp.float32),
        inputs=inputs,
    )
    fn = executor.get_compiled(num_tokens=1, padded_num_reqs=1)
    out = fn(
        jnp.array([10.0], dtype=jnp.float32),
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([3.0], dtype=jnp.float32),
        jnp.array([0], dtype=jnp.int32),
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(jnp.array([13.0], dtype=jnp.float32)))
