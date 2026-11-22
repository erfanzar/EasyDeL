"""Independent registered SSM1 scan/cache parity; accelerator execution only."""

import jax
import numpy as np
import pytest
from ejkernel.modules import state_space_v1
from ejkernel.modules.operations.configs import StateSpaceV1Config
from jax import numpy as jnp


def _reference(x, a, b, c, d, dt, initial, gate):
    state = initial.astype(np.float64).copy()
    outputs = np.zeros_like(x, dtype=np.float64)
    for token in range(x.shape[1]):
        state = np.exp(dt[:, token, :, None] * a[None]) * state + (
            dt[:, token, :, None] * b[:, token, None, :] * x[:, token, :, None]
        )
        y = (state * c[:, token, None, :]).sum(axis=-1) + d * x[:, token]
        outputs[:, token] = y * gate[:, token] / (1 + np.exp(-gate[:, token]))
    return outputs, state


@pytest.mark.parametrize("associative", [False, True])
@pytest.mark.parametrize("length", [0, 1, 5])
@pytest.mark.parametrize("compiled", [False, True])
def test_registered_scan_selector_preserves_recurrence_cache_gate_and_gradients(associative, length, compiled):
    rng = np.random.default_rng(525)
    shapes = [(1, length, 2), (2, 3), (1, length, 3), (1, length, 3), (2,), (1, length, 2), (1, 2, 3)]
    values = [rng.uniform(0.05, 0.5, size=shape).astype(np.float32) for shape in shapes]
    values[1] *= -1
    gate = rng.uniform(-0.5, 0.5, size=shapes[0]).astype(np.float32)
    expected_y, expected_state = _reference(*values, gate)
    conv = jnp.ones((1, 2, 4), dtype=jnp.float32)

    def forward(*args):
        x, a, b, c, d, dt, initial = args
        return state_space_v1(
            x,
            a,
            b,
            c,
            d,
            dt,
            initial_state=initial,
            gate=jnp.asarray(gate),
            conv_state=conv,
            use_associative_scan=associative,
            cfg=StateSpaceV1Config(platform="xla", backend="any"),
        )

    fn = jax.jit(forward) if compiled else forward
    y, state, new_conv = fn(*map(jnp.asarray, values))
    np.testing.assert_allclose(y, expected_y, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(state, expected_state, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(new_conv, conv)

    def loss(*args):
        output, final, _ = forward(*args)
        return output.sum() + final.sum()

    gradient_fn = jax.grad(loss, argnums=(0, 6))
    if compiled:
        gradient_fn = jax.jit(gradient_fn)
    gradients = gradient_fn(*map(jnp.asarray, values))
    # Independent scalar-loop finite differences include the final-state loss,
    # so gradients through the initial state are checked, including empty input.
    for argnum, actual in zip((0, 6), gradients, strict=True):
        expected = np.zeros_like(values[argnum])
        for index in np.ndindex(expected.shape):
            losses = []
            for direction in (1, -1):
                perturbed = [value.astype(np.float64).copy() for value in values]
                perturbed[argnum][index] += direction * 1e-4
                out, final = _reference(*perturbed, gate)
                losses.append(out.sum() + final.sum())
            expected[index] = (losses[0] - losses[1]) / 2e-4
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
