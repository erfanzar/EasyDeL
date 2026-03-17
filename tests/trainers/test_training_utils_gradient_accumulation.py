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

import dataclasses

import jax
import jax.numpy as jnp

from easydel.trainers.training_utils import make_assertions_and_get_sizes, minibatch_call


@dataclasses.dataclass
class _DummyState:
    graphstate: jax.Array


def test_make_assertions_skips_scalar_leaves_for_batch_size():
    batch = {"running_mean": jnp.asarray(0.0, dtype=jnp.float32), "x": jnp.zeros((4, 3), dtype=jnp.float32)}
    batch_size, minibatch_size, _ = make_assertions_and_get_sizes(batch=batch, gradient_accumulation_steps=2)
    assert batch_size == 4
    assert minibatch_size == 2


def test_minibatch_call_matches_full_batch_gradients():
    state = _DummyState(graphstate=jnp.asarray(2.0, dtype=jnp.float32))
    batch = {"x": jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=jnp.float32)}

    def loss_fn(param, minibatch):
        loss = jnp.mean(param * minibatch["x"])
        return loss, loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (full_loss, _), full_grads = grad_fn(state.graphstate, batch)

    grads, metrics = minibatch_call(state=state, batch=batch, minibatch_size=2, grad_fn=grad_fn)

    assert jnp.allclose(grads, full_grads)
    assert jnp.allclose(metrics, full_loss)
