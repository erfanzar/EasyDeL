"""Public shared-router losses must preserve exact token-sharded selection.

Exercises the production loss and its gradients, including compiled execution,
against a host reference.
"""

import jax
import numpy as np
import pytest
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .test_llama4_router_loss import _LOGITS, _reference


@pytest.mark.parametrize("shard_axis", [None, "tokens", "experts"])
@pytest.mark.parametrize("top_k", [1, 2])
def test_shared_router_loss_and_gradient_preserve_global_expert_assignment(shard_axis, top_k):
    logits = np.tile(_LOGITS, (4, 1))
    mask = np.tile(np.asarray([[1, 1, 1, 0]], dtype=np.int32), (4, 1))
    expected_loss, probabilities, selected = _reference(logits, mask, top_k)
    valid = mask.reshape(-1).astype(bool)
    frequencies = np.eye(4)[selected][valid].mean(axis=0).sum(axis=0)
    # d[ E * sum_e(f_e * mean_valid(p_e)) ] / dz_t,e.
    # Assignments are piecewise constant away from the deliberately separated
    # logits' top-k boundaries; padding contributes neither value nor gradient.
    expected_gradient = (
        4 / valid.sum() * probabilities * (frequencies - (probabilities * frequencies).sum(axis=-1, keepdims=True))
    )
    expected_gradient[~valid] = 0

    mesh = Mesh(np.asarray(jax.devices()[:4]), ("router",))
    spec = {
        None: PartitionSpec(),
        "tokens": PartitionSpec("router", None),
        "experts": PartitionSpec(None, "router"),
    }[shard_axis]
    operand = jax.device_put(logits, NamedSharding(mesh, spec))
    attention_mask = jnp.asarray(mask)

    def objective(values):
        return auxiliary_load_balancing_loss_func((values,), 4, top_k, attention_mask)

    # Check the eager value first, then the training and JIT forms. All values come
    # from the production loss; the reference is host-side NumPy.
    np.testing.assert_allclose(np.asarray(objective(operand)), expected_loss, rtol=1e-4, atol=1e-6)
    loss, gradient = jax.value_and_grad(objective)(operand)
    np.testing.assert_allclose(np.asarray(loss), expected_loss, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(gradient), expected_gradient, rtol=1e-4, atol=1e-6)
    compiled_loss, compiled_gradient = jax.jit(jax.value_and_grad(objective))(operand)
    np.testing.assert_allclose(np.asarray(compiled_loss), expected_loss, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(compiled_gradient), expected_gradient, rtol=1e-4, atol=1e-6)
