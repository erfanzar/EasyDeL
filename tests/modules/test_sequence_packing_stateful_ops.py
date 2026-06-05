import jax
import jax.numpy as jnp

from easydel.infra.sequence_packing import segmented_depthwise_causal_conv1d
from easydel.operations.kernels.kda import _recurrent_kda_fwd
from easydel.operations.kernels.ssm1 import _segmented_ssm1_fwd
from easydel.operations.kernels.ssm2 import _segmented_ssm2_fwd


def test_segmented_depthwise_conv_resets_window_at_boundaries():
    x = jnp.arange(1, 7, dtype=jnp.float32).reshape(1, 6, 1)
    kernel = jnp.array([[1.0, 10.0, 100.0]], dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1, 1, -1]], dtype=jnp.int32)

    output, final_state = segmented_depthwise_causal_conv1d(x, kernel, segment_ids)

    expected = jnp.array([[[100.0], [210.0], [300.0], [430.0], [543.0], [0.0]]], dtype=jnp.float32)
    assert output.shape == x.shape
    assert final_state.shape == (1, 1, 3)
    assert jnp.allclose(output, expected)


def test_segmented_ssm1_matches_independent_segments():
    key = jax.random.PRNGKey(0)
    hidden = jax.random.normal(key, (1, 5, 3), dtype=jnp.float32)
    B = jax.random.normal(jax.random.fold_in(key, 1), (1, 5, 2), dtype=jnp.float32) * 0.1
    C = jax.random.normal(jax.random.fold_in(key, 2), (1, 5, 2), dtype=jnp.float32) * 0.1
    dt = jax.nn.softplus(jax.random.normal(jax.random.fold_in(key, 3), (1, 5, 3), dtype=jnp.float32))
    A_real = -jnp.exp(jax.random.normal(jax.random.fold_in(key, 4), (3, 2), dtype=jnp.float32) * 0.1)
    D = jnp.linspace(0.1, 0.3, 3, dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1, -1]], dtype=jnp.int32)

    packed, _ = _segmented_ssm1_fwd(hidden, A_real, B, C, D, dt, segment_ids)
    first, _ = _segmented_ssm1_fwd(hidden[:, :2], A_real, B[:, :2], C[:, :2], D, dt[:, :2], segment_ids[:, :2])
    second, _ = _segmented_ssm1_fwd(hidden[:, 2:4], A_real, B[:, 2:4], C[:, 2:4], D, dt[:, 2:4], segment_ids[:, 2:4])

    expected = jnp.concatenate([first, second, jnp.zeros_like(packed[:, 4:])], axis=1)
    assert jnp.allclose(packed, expected, atol=1e-5)


def test_segmented_ssm2_matches_independent_segments():
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (1, 5, 2, 3), dtype=jnp.float32)
    B = jax.random.normal(jax.random.fold_in(key, 1), (1, 5, 1, 2), dtype=jnp.float32) * 0.1
    C = jax.random.normal(jax.random.fold_in(key, 2), (1, 5, 1, 2), dtype=jnp.float32) * 0.1
    dt = jax.nn.softplus(jax.random.normal(jax.random.fold_in(key, 3), (1, 5, 2), dtype=jnp.float32))
    A_real = -jnp.exp(jax.random.normal(jax.random.fold_in(key, 4), (2,), dtype=jnp.float32) * 0.1)
    D = jnp.array([0.2, 0.4], dtype=jnp.float32)
    segment_ids = jnp.array([[0, 0, 1, 1, -1]], dtype=jnp.int32)

    packed, _ = _segmented_ssm2_fwd(x, A_real, B, C, D, dt, segment_ids)
    first, _ = _segmented_ssm2_fwd(x[:, :2], A_real, B[:, :2], C[:, :2], D, dt[:, :2], segment_ids[:, :2])
    second, _ = _segmented_ssm2_fwd(x[:, 2:4], A_real, B[:, 2:4], C[:, 2:4], D, dt[:, 2:4], segment_ids[:, 2:4])

    expected = jnp.concatenate([first, second, jnp.zeros_like(packed[:, 4:])], axis=1)
    assert jnp.allclose(packed, expected, atol=1e-5)


def test_segmented_kda_matches_independent_segments():
    key = jax.random.PRNGKey(2)
    query = jax.random.normal(key, (1, 2, 5, 3), dtype=jnp.float32)
    key_states = jax.random.normal(jax.random.fold_in(key, 1), (1, 2, 5, 3), dtype=jnp.float32)
    value = jax.random.normal(jax.random.fold_in(key, 2), (1, 2, 5, 4), dtype=jnp.float32)
    beta = jax.nn.sigmoid(jax.random.normal(jax.random.fold_in(key, 3), (1, 2, 5), dtype=jnp.float32))
    decay = -jax.nn.softplus(jax.random.normal(jax.random.fold_in(key, 4), (1, 2, 5), dtype=jnp.float32)) * 0.1
    segment_ids = jnp.array([[0, 0, 1, 1, -1]], dtype=jnp.int32)

    packed, _ = _recurrent_kda_fwd(query, key_states, value, beta, decay, segment_ids=segment_ids)
    first, _ = _recurrent_kda_fwd(
        query[:, :, :2],
        key_states[:, :, :2],
        value[:, :, :2],
        beta[:, :, :2],
        decay[:, :, :2],
        segment_ids=segment_ids[:, :2],
    )
    second, _ = _recurrent_kda_fwd(
        query[:, :, 2:4],
        key_states[:, :, 2:4],
        value[:, :, 2:4],
        beta[:, :, 2:4],
        decay[:, :, 2:4],
        segment_ids=segment_ids[:, 2:4],
    )

    expected = jnp.concatenate([first, second, jnp.zeros_like(packed[:, :, 4:])], axis=2)
    assert jnp.allclose(packed, expected, atol=1e-5)
