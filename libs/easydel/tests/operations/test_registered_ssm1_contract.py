"""Registered algorithm/config contracts by abstract tracing, without array execution."""

import jax
import pytest
from ejkernel.modules import state_space_v1
from ejkernel.modules.operations.configs import StateSpaceV1Config
from jax import numpy as jnp

from .test_ssm_scan_page_axis_contracts import _has_scan

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("associative", [False, True])
@pytest.mark.parametrize("length", [0, 1, 5])
def test_registered_selector_controls_algorithm_and_differentiable_cache(associative, length):
    shapes = [(2, length, 3), (3, 4), (2, length, 4), (2, length, 4), (3,), (2, length, 3), (2, 3, 4)]
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes]

    def forward(x, a, b, c, d, dt, state):
        output, final, _ = state_space_v1(
            x,
            a,
            b,
            c,
            d,
            dt,
            initial_state=state,
            use_associative_scan=associative,
            # A cached/tuned native GPU configuration must not override an
            # explicit parallel request; no GPU kernel is loaded/executed.
            cfg=StateSpaceV1Config(
                platform="tilelang" if associative else "xla", backend="gpu" if associative else "any"
            ),
        )
        return output, final

    program = jax.make_jaxpr(forward)(*descriptors)
    assert [aval.shape for aval in program.out_avals] == [(2, length, 3), (2, 3, 4)]
    assert _has_scan(program) == (not associative and length > 1)

    def loss(*args):
        output, state = forward(*args)
        return output.sum() + state.sum()

    gradients = jax.make_jaxpr(jax.grad(loss, argnums=tuple(range(7))))(*descriptors)
    assert [aval.shape for aval in gradients.out_avals] == shapes
    assert all(aval.dtype == jnp.float32 for aval in gradients.out_avals)
