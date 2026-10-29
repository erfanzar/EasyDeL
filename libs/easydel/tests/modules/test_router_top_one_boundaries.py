"""Keep exact top-one selection fixes compatible with ties and one expert."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.infra import loss_utils
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .test_llama4_router_loss import _reference


def _inputs(single_expert):
    if single_expert:
        logits = np.asarray([[2], [-1], [0], [5]], dtype=np.float32)
    else:
        # Exact ties select lower indices. Different untied rows make a wrong
        # tied assignment visible in the public scalar, not only diagnostics.
        logits = np.asarray([[2, 2, -1, 0], [0, 4, 1, -2], [3, 1, 3, 0], [0, 1, 2, 3]], dtype=np.float32)
    return np.tile(logits, (4, 1)), np.tile(np.asarray([[1, 1, 1, 0]], dtype=np.int32), (4, 1))


@pytest.mark.cpu_ok
@pytest.mark.parametrize("single_expert", [False, True])
def test_top_one_boundary_contract_with_numpy_backend(monkeypatch, single_expert):
    """Exercise the public loss with a strict host backend, without JAX compute."""

    def top_k(values, k):
        # Match the primitive's shape contract: requesting k=2 for a one-expert
        # model is invalid, even if NumPy slicing would silently accept it.
        if not 0 <= k <= values.shape[-1]:
            raise ValueError("top_k exceeds the number of experts")
        indices = np.argsort(-values, axis=-1, kind="stable")[:, :k]
        return np.take_along_axis(values, indices, axis=-1), indices

    def softmax(values, axis):
        weights = np.exp(values - values.max(axis=axis, keepdims=True))
        return weights / weights.sum(axis=axis, keepdims=True)

    monkeypatch.setattr(loss_utils, "jnp", np)
    monkeypatch.setattr(
        loss_utils,
        "jax",
        SimpleNamespace(
            lax=SimpleNamespace(top_k=top_k),
            nn=SimpleNamespace(
                softmax=softmax,
                one_hot=lambda indices, num_classes, dtype: np.eye(num_classes, dtype=dtype)[indices],
            ),
        ),
    )
    logits, mask = _inputs(single_expert)
    expected, _, _ = _reference(logits, mask, 1)
    actual = loss_utils.auxiliary_load_balancing_loss_func((logits,), logits.shape[-1], 1, mask)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("single_expert", [False, True])
@pytest.mark.parametrize("shard_axis", [None, "tokens"])
def test_top_one_sharded_boundary_contract(single_expert, shard_axis):
    """Compare eager and compiled public losses against an independent oracle."""
    logits, mask = _inputs(single_expert)
    expected, _, _ = _reference(logits, mask, 1)
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("router",))
    spec = PartitionSpec() if shard_axis is None else PartitionSpec("router", None)
    operand = jax.device_put(logits, NamedSharding(mesh, spec))
    attention_mask = jnp.asarray(mask)

    def objective(values):
        return loss_utils.auxiliary_load_balancing_loss_func((values,), logits.shape[-1], 1, attention_mask)

    np.testing.assert_allclose(np.asarray(objective(operand)), expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.asarray(jax.jit(objective)(operand)), expected, rtol=1e-4, atol=1e-6)
