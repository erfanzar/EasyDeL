"""Shared router-loss ordering and fp32 accumulation regressions (accelerator only)."""

import jax
import ml_dtypes
import numpy as np
import pytest
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@pytest.mark.parametrize("shard_axis", [None, "tokens", "experts"])
@pytest.mark.parametrize("dtype", [np.float32, ml_dtypes.bfloat16])
@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("mask_kind", ["none", "padding", "empty"])
def test_multilayer_router_loss_preserves_logit_order_and_fp32_probabilities(dtype, top_k, mask_kind, shard_axis):
    # The near-tie row loses its ordering after softmax, even in fp32. The other
    # rows make the resulting incorrect expert counts observable in the loss.
    rows = np.asarray(
        [[0, 2e-8, 1e-8, -1e-8], [-2, 4, -1, 0], [0.5, -0.25, 1.75, 0], [1, -1, 0.25, -0.5]],
        dtype=np.float32,
    )
    layers = [np.tile(rows, (2, 1)).astype(dtype), np.tile(rows[:, ::-1], (2, 1)).astype(dtype)]
    mask = None
    if mask_kind == "padding":
        mask = np.tile(np.asarray([[1, 1, 1, 0]], dtype=np.int32), (2, 1))
    elif mask_kind == "empty":
        mask = np.zeros((2, 4), dtype=np.int32)

    # Quantize the inputs first, then evaluate the independent reference in
    # float64: expected routing reflects exactly the logits sent to the device.
    logits = np.concatenate([layer.astype(np.float64) for layer in layers])
    valid = np.ones(logits.shape[0], dtype=bool) if mask is None else np.tile(mask.reshape(-1).astype(bool), 2)
    expected = 0.0
    if valid.any():
        logits = logits[valid]
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        selected = np.argsort(-logits, axis=-1, kind="stable")[:, :top_k]
        expected = 4 * sum(
            (selected == expert).sum(axis=-1).mean() * probabilities[:, expert].mean() for expert in range(4)
        )

    mesh = Mesh(np.asarray(jax.devices()[:4]), ("router",))
    spec = {
        None: PartitionSpec(),
        "tokens": PartitionSpec("router", None),
        "experts": PartitionSpec(None, "router"),
    }[shard_axis]
    device_layers = tuple(jax.device_put(layer, NamedSharding(mesh, spec)) for layer in layers)
    device_mask = None if mask is None else jnp.asarray(mask)

    def objective(gates):
        return auxiliary_load_balancing_loss_func(gates, num_experts=4, top_k=top_k, attention_mask=device_mask)

    # Eager placement triggered the recorded token-sharded single-expert failure;
    # also retain the compiled check, which may fuse selection with reductions.
    for actual in (objective(device_layers), jax.jit(objective)(device_layers)):
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("num_experts", [1, 2, 4])
def test_single_selection_preserves_low_index_ties_and_single_expert_boundary(num_experts):
    rows = np.asarray([[1, 1, 1, 1], [0, 3, -2, -1], [2, 2, 1, 0], [4, -3, -2, -1]], dtype=np.float32)
    logits = np.tile(rows[:, :num_experts], (2, 1))
    host_logits = logits.astype(np.float64)
    probabilities = np.exp(host_logits - host_logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    # Stable low-index ties matter because the other rows have unequal routing
    # probabilities; assigning tied rows elsewhere changes the public scalar.
    selected = np.argmax(host_logits, axis=-1)
    expected = num_experts * sum(
        np.mean(selected == expert) * probabilities[:, expert].mean() for expert in range(num_experts)
    )
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("router",))
    device_logits = jax.device_put(logits, NamedSharding(mesh, PartitionSpec("router", None)))

    def objective(values):
        return auxiliary_load_balancing_loss_func(values, num_experts=num_experts, top_k=1)

    for actual in (objective(device_logits), jax.jit(objective)(device_logits)):
        assert actual.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-6)
