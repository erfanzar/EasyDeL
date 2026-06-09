from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import state_space_v1, state_space_v2

from ._utils import assert_allclose


def test_state_space_v1_shapes_and_gate_and_conv_state_passthrough():
    b, t, intermediate, n = 2, 4, 8, 4
    hidden = jax.random.normal(jax.random.PRNGKey(0), (b, t, intermediate), dtype=jnp.float32).astype(jnp.bfloat16)
    A = (jax.random.normal(jax.random.PRNGKey(1), (intermediate, n), dtype=jnp.float32) * -0.1).astype(jnp.bfloat16)
    B = jax.random.normal(jax.random.PRNGKey(2), (b, t, n), dtype=jnp.float32).astype(jnp.bfloat16)
    C = jax.random.normal(jax.random.PRNGKey(3), (b, t, n), dtype=jnp.float32).astype(jnp.bfloat16)
    D = jax.random.normal(jax.random.PRNGKey(4), (intermediate,), dtype=jnp.float32).astype(jnp.bfloat16)
    dt = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(5), (b, t, intermediate), dtype=jnp.float32)).astype(
        jnp.bfloat16
    )

    out, state, conv = state_space_v1(hidden, A, B, C, D, dt, platform="xla")
    assert out.shape == (b, t, intermediate)
    assert state.shape == (b, intermediate, n)
    assert conv is None

    gate = jax.random.normal(jax.random.PRNGKey(6), (b, t, intermediate), dtype=jnp.float32).astype(jnp.bfloat16)
    init = jax.random.normal(jax.random.PRNGKey(7), (b, intermediate, n), dtype=jnp.float32).astype(jnp.bfloat16)
    conv_state = jax.random.normal(jax.random.PRNGKey(8), (b, intermediate, 3), dtype=jnp.float32).astype(jnp.bfloat16)
    out2, state2, conv2 = state_space_v1(
        hidden,
        A,
        B,
        C,
        D,
        dt,
        gate=gate,
        initial_state=init,
        conv_state=conv_state,
        act_fn=jax.nn.silu,
        platform="xla",
    )
    assert out2.shape == (b, t, intermediate)
    assert state2.shape == (b, intermediate, n)
    assert_allclose(conv2, conv_state, atol=0.0)


def test_state_space_v2_shapes_groups_gated_rmsnorm_and_single_step():
    b, t, heads, head_dim, n_groups, n = 1, 4, 4, 8, 2, 4
    x = jax.random.normal(jax.random.PRNGKey(10), (b, t, heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    A = (jax.random.normal(jax.random.PRNGKey(11), (heads,), dtype=jnp.float32) * -0.1).astype(jnp.bfloat16)
    B = jax.random.normal(jax.random.PRNGKey(12), (b, t, n_groups, n), dtype=jnp.float32).astype(jnp.bfloat16)
    C = jax.random.normal(jax.random.PRNGKey(13), (b, t, n_groups, n), dtype=jnp.float32).astype(jnp.bfloat16)
    D = jax.random.normal(jax.random.PRNGKey(14), (heads,), dtype=jnp.float32).astype(jnp.bfloat16)
    dt = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(15), (b, t, heads), dtype=jnp.float32)).astype(
        jnp.bfloat16
    )
    gate = jax.random.normal(jax.random.PRNGKey(16), (b, t, heads * head_dim), dtype=jnp.float32).astype(jnp.bfloat16)

    out, state, conv = state_space_v2(
        x,
        A,
        B,
        C,
        D,
        dt,
        gate=gate,
        n_groups=n_groups,
        use_gated_rmsnorm=True,
        act_fn=jax.nn.silu,
        platform="xla",
    )
    assert out.shape == (b, t, heads * head_dim)
    assert state.shape == (b, heads, head_dim, n)
    assert conv is None

    x1 = x[:, :1]
    B1 = B[:, :1]
    C1 = C[:, :1]
    dt1 = dt[:, :1]
    out1, state1, _ = state_space_v2(
        x1,
        A,
        B1,
        C1,
        D,
        dt1,
        initial_state=state,
        n_groups=n_groups,
        platform="xla",
    )
    assert out1.shape == (b, 1, heads * head_dim)
    assert state1.shape == (b, heads, head_dim, n)
