"""Deterministic shared-loss tests for the Llama4 router auxiliary loss.

The fixed logits remove model initialization and MoE dispatch from the tested
boundary. The sharded numerical tests require an accelerator.
"""

import json
import sys
from contextlib import contextmanager

import jax
import numpy as np
import pytest
from easydel.infra import loss_utils
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Representative fp32 logits from the public Llama4 seed-0 CPU diagnostic, not
# golden losses: all expected values below are calculated independently.
_LOGITS = np.asarray(
    [
        [-0.06350876, -0.00379330, -0.11980015, 0.04838553],
        [-0.09021328, -0.02136140, -0.10304523, -0.00088539],
        [0.05295847, 0.06950933, 0.01114256, -0.19430208],
        [-0.11355325, 0.17772572, -0.02251274, 0.08428074],
    ],
    dtype=np.float32,
)


@contextmanager
def _capture_router_loss():
    """Observe actual loss intermediates without replacing numerical operations.

    Capture the returning frame rather than recomputing the loss: a separate
    top-k call need not reproduce a failure in the original call's placement or
    compilation. Restore the thread-local profiler even when an assertion fails.
    """
    observations = []
    previous = sys.getprofile()
    code = loss_utils.auxiliary_load_balancing_loss_func.__code__

    def observe(frame, event, arg):
        if callable(previous):
            previous(frame, event, arg)
        if event == "return" and frame.f_code is code:
            observations.append(dict(frame.f_locals))

    sys.setprofile(observe)
    try:
        yield observations
    finally:
        sys.setprofile(previous)


def _router_loss_diagnostics(observations):
    """Serialize the real shared-loss operations only after a failed assertion."""
    names = (
        "concatenated_gate_logits",
        "routing_weights",
        "selected_experts",
        "expert_mask",
        "expert_attention_mask",
        "masked_expert_contributions",
        "tokens_per_expert_numerator",
        "tokens_per_expert_denominator",
        "tokens_per_expert",
        "router_per_expert_attention_mask",
        "router_prob_numerator",
        "router_prob_denominator",
        "router_prob_per_expert",
        "per_expert_loss_terms",
        "final_loss",
    )
    records = []
    for observation in observations:
        record = {}
        for name in names:
            if name in observation:
                value = observation[name]
                host = np.asarray(value)
                record[name] = {
                    "shape": list(host.shape),
                    "dtype": str(host.dtype),
                    "sharding": str(getattr(value, "sharding", None)),
                    "values": host.tolist(),
                }
        records.append(record)
    return json.dumps(records)


def _reference(logits, mask, top_k):
    logits = logits.astype(np.float64)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    selected = np.argsort(-logits, axis=-1, kind="stable")[:, :top_k]
    valid = mask.reshape(-1).astype(bool)
    frequencies = np.eye(logits.shape[-1])[selected][valid].mean(axis=0)
    probabilities = weights[valid].mean(axis=0)
    return logits.shape[-1] * np.sum(frequencies * probabilities[None, :]), weights, selected


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("shard_axis", [None, "tokens", "experts"])
def test_fixed_llama4_router_loss_localizes_selection_and_mask_reduction(top_k, padding, shard_axis):
    logits = np.tile(_LOGITS, (4, 1))
    mask = np.tile(np.asarray([[1, 1, 1, 0 if padding else 1]], dtype=np.int32), (4, 1))
    expected, probabilities, selected = _reference(logits, mask, top_k)
    # Explicit placements distinguish local selection from distributed selection
    # and reduction; all dimensions are divisible by the four-device mesh.
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("router",))
    spec = {
        None: PartitionSpec(),
        "tokens": PartitionSpec("router", None),
        "experts": PartitionSpec(None, "router"),
    }[shard_axis]
    device_logits = jax.device_put(logits, NamedSharding(mesh, spec))
    with _capture_router_loss() as observations:
        loss = loss_utils.auxiliary_load_balancing_loss_func((device_logits,), 4, top_k, jnp.asarray(mask))
    # Evaluate production's scalar before any diagnostic can abort the test.
    try:
        np.testing.assert_allclose(np.asarray(loss), expected, rtol=1e-4, atol=1e-6)
    except AssertionError as exc:
        raise AssertionError(f"{exc}\nactual_shared_loss_operations={_router_loss_diagnostics(observations)}") from exc
    assert len(observations) == 1
    actual = observations[0]
    # Inspect the intermediates of the production call, not a separate native top-k.
    np.testing.assert_allclose(np.asarray(actual["routing_weights"]), probabilities, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(actual["selected_experts"]), selected)
    np.testing.assert_array_equal(np.asarray(actual["expert_mask"]), np.eye(4)[selected])


@pytest.mark.parametrize("shard_axis", [None, "tokens", "experts"])
def test_shared_router_loss_selects_distinct_logits_before_softmax_rounding(shard_axis):
    # Softmax rounds the first token's four fp32 probabilities to exactly 0.25.
    # Selecting from those probabilities changes its true expert from 1 to 0;
    # the second token makes that spurious assignment observable in the loss.
    logits = np.tile(np.asarray([[0, 2e-8, 1e-8, -1e-8], [-2, 4, -1, 0]], dtype=np.float32), (4, 1))
    mask = np.ones((4, 2), dtype=np.int32)
    expected, probabilities, selected = _reference(logits, mask, 1)
    mesh = Mesh(np.asarray(jax.devices()[:4]), ("router",))
    spec = {
        None: PartitionSpec(),
        "tokens": PartitionSpec("router", None),
        "experts": PartitionSpec(None, "router"),
    }[shard_axis]
    device_logits = jax.device_put(logits, NamedSharding(mesh, spec))
    with _capture_router_loss() as observations:
        loss = loss_utils.auxiliary_load_balancing_loss_func((device_logits,), 4, 1, jnp.asarray(mask))
    assert len(observations) == 1
    actual = observations[0]
    # Observe production's actual assignments, not a separate top-k from
    # rounded probabilities.
    np.testing.assert_array_equal(np.asarray(actual["selected_experts"]), selected)
    np.testing.assert_allclose(np.asarray(actual["routing_weights"]), probabilities, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(np.asarray(actual["expert_mask"]), np.eye(4)[selected])
    np.testing.assert_allclose(np.asarray(loss), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("separated_logits", [False, True])
def test_independent_reference_detects_softmax_order_loss_without_jax_compute(monkeypatch, separated_logits):
    # Host-only regression against miscounting assignments after probability
    # rounding. The real loss body runs with NumPy substituted for its numerical backend.
    from types import SimpleNamespace

    def softmax(values, axis):
        shifted = values - values.max(axis=axis, keepdims=True)
        weights = np.exp(shifted)
        return weights / weights.sum(axis=axis, keepdims=True)

    monkeypatch.setattr(loss_utils, "jnp", np)
    monkeypatch.setattr(
        loss_utils,
        "jax",
        SimpleNamespace(
            nn=SimpleNamespace(
                softmax=softmax,
                one_hot=lambda indices, num_classes, dtype: np.eye(num_classes, dtype=dtype)[indices],
            ),
            lax=SimpleNamespace(
                top_k=lambda values, k: (
                    np.sort(values, axis=-1)[:, -k:][:, ::-1],
                    np.argsort(-values, axis=-1, kind="stable")[:, :k],
                )
            ),
        ),
    )
    if separated_logits:
        logits = np.tile(_LOGITS, (4, 1))
        mask = np.tile(np.asarray([[1, 1, 1, 0]], dtype=np.int32), (4, 1))
    else:
        logits = np.tile(np.asarray([[0, 2e-8, 1e-8, -1e-8], [-2, 4, -1, 0]], dtype=np.float32), (4, 1))
        mask = np.ones((4, 2), dtype=np.int32)
    expected, _, selected = _reference(logits, mask, 1)
    previous_profiler = sys.getprofile()
    with _capture_router_loss() as observations:
        loss = loss_utils.auxiliary_load_balancing_loss_func((logits,), 4, 1, mask)
    assert sys.getprofile() is previous_profiler
    np.testing.assert_allclose(loss, expected, rtol=1e-5, atol=1e-6)
    # Exercise the failure-payload path against actual returned intermediates,
    # without forcing the public loss to fail or altering its implementation.
    diagnostic = json.loads(_router_loss_diagnostics(observations))[0]
    np.testing.assert_array_equal(diagnostic["selected_experts"]["values"], selected)
    np.testing.assert_allclose(diagnostic["final_loss"]["values"], expected, rtol=1e-5, atol=1e-6)
