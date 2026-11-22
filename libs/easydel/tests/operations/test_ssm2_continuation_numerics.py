"""Registered segmented SSM2 continuation/padding parity (accelerator only)."""

import math

import jax
import numpy as np
import pytest
from easydel.operations import OperationRegistry
from jax import numpy as jnp

from .test_paged_reference_contract import _metadata


def _reference(x, segments, initial, gate=None, rms=False):
    state = initial.copy()
    output = np.zeros_like(x)
    for batch in range(x.shape[0]):
        previous, seen = -1, False
        for token in range(x.shape[1]):
            segment = segments[batch, token]
            if segment >= 0:
                if seen and segment != previous:
                    state[batch] = 0
                state[batch] = math.exp(-0.5) * state[batch] + 0.5 * x[batch, token, :, :, None]
                output[batch, token] = state[batch].sum(axis=-1) + 0.25 * x[batch, token]
                seen = True
            previous = segment
    if gate is not None:
        flat = output.reshape(*output.shape[:2], -1)
        if rms:
            flat /= np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-5)
        flat *= gate / (1 + np.exp(-gate))
        output = flat.reshape(output.shape)
    return output, state


@pytest.mark.parametrize("segments", [[[0, 0, 0, 0]], [[-1, 0, 0, -1]], [[0, -1, 0, 1]], [[-1, -1, -1, -1]]])
@pytest.mark.parametrize("gating", ["none", "silu", "rms"])
@pytest.mark.parametrize("compiled", [False, True])
def test_registered_ssm2_continues_initial_state_and_preserves_padding(segments, gating, compiled):
    segments = np.asarray(segments, dtype=np.int32)
    x = np.array([1.0, 9.0, 2.0, 3.0], dtype=np.float32).reshape(1, 4, 1, 1)
    initial = np.full((1, 1, 1, 1), 0.75, dtype=np.float32)
    gate = np.full((1, 4, 1), 0.4, dtype=np.float32) if gating != "none" else None
    operation = OperationRegistry.create("ssm2", _metadata())

    def forward(x, state, ids):
        result = operation(
            x=x,
            A=jnp.zeros((1,)),
            B=jnp.ones(x.shape),
            C=jnp.ones(x.shape),
            D=jnp.full((1,), 0.25),
            dt=jnp.full(x.shape[:3], 0.5),
            ssm_state=state,
            segment_ids=ids,
            gate=None if gate is None else jnp.asarray(gate),
            use_gated_rmsnorm=gating == "rms",
        )
        return result.attention_outputs, result.ssm_state

    if compiled:
        forward = jax.jit(forward)
    actual, state = forward(jnp.asarray(x), jnp.asarray(initial), jnp.asarray(segments))
    expected, final = _reference(x, segments, initial, gate, gating == "rms")
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(state, final, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("valid", [False, True])
def test_registered_ssm2_single_token_cached_validity(valid):
    operation = OperationRegistry.create("ssm2", _metadata())
    state = jnp.full((1, 1, 1, 1), 0.75)
    x = jnp.zeros((1, 1, 1, 1))
    result = operation(
        x,
        jnp.zeros((1,)),
        x,
        jnp.ones_like(x),
        jnp.zeros((1,)),
        jnp.ones((1, 1, 1)),
        ssm_state=state,
        segment_ids=jnp.array([[0 if valid else -1]]),
    )
    expected = 0.75 * math.exp(-1) if valid else 0.75
    np.testing.assert_allclose(result.ssm_state, expected, rtol=1e-6)
    np.testing.assert_allclose(result.attention_outputs, expected if valid else 0, rtol=1e-6)


def test_registered_ssm2_full_sequence_matches_cached_chunks():
    operation = OperationRegistry.create("ssm2", _metadata())
    x = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1, 1)

    def run(x, state):
        return operation(
            x,
            jnp.zeros((1,)),
            jnp.ones_like(x),
            jnp.ones_like(x),
            jnp.zeros((1,)),
            jnp.full(x.shape[:3], 0.5),
            ssm_state=state,
            segment_ids=jnp.zeros(x.shape[:2], dtype=jnp.int32),
        )

    initial = jnp.full((1, 1, 1, 1), 0.75)
    full = run(x, initial)
    prefix = run(x[:, :2], initial)
    suffix = run(x[:, 2:], prefix.ssm_state)
    np.testing.assert_allclose(
        jnp.concatenate((prefix.attention_outputs, suffix.attention_outputs), axis=1), full.attention_outputs, rtol=1e-6
    )
    np.testing.assert_allclose(suffix.ssm_state, full.ssm_state, rtol=1e-6)
