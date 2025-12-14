import jax.numpy as jnp
import pytest

pytest.importorskip("eformer.loggings")


def _compute_sampling_valid_mask(*args, **kwargs):
    from easydel.inference.esurge.runners.execution_manager import _compute_sampling_valid_mask

    return _compute_sampling_valid_mask(*args, **kwargs)


def test_sampling_valid_mask_allows_in_progress_requests():
    i_reqs = jnp.arange(4, dtype=jnp.int32)
    num_requests = jnp.int32(2)
    active_mask = jnp.array([True, True, True, True])
    scheduled = jnp.array([1, 1, 1, 1], dtype=jnp.int32)

    seq_lens_now = jnp.array([3, 4, 5, 6], dtype=jnp.int32)
    req_num_tokens = jnp.array([10, 5, 10, 10], dtype=jnp.int32)

    valid = _compute_sampling_valid_mask(
        i_reqs=i_reqs,
        num_requests=num_requests,
        active_mask_slice=active_mask,
        scheduled_slice=scheduled,
        seq_lens_now=seq_lens_now,
        req_num_tokens_slice=req_num_tokens,
    )

    assert jnp.array_equal(valid, jnp.array([True, True, False, False]))


def test_sampling_valid_mask_blocks_completed_or_unscheduled_requests():
    i_reqs = jnp.arange(3, dtype=jnp.int32)
    num_requests = jnp.int32(3)
    active_mask = jnp.array([True, True, False])
    scheduled = jnp.array([1, 0, 1], dtype=jnp.int32)

    seq_lens_now = jnp.array([5, 5, 1], dtype=jnp.int32)
    req_num_tokens = jnp.array([5, 10, 10], dtype=jnp.int32)

    valid = _compute_sampling_valid_mask(
        i_reqs=i_reqs,
        num_requests=num_requests,
        active_mask_slice=active_mask,
        scheduled_slice=scheduled,
        seq_lens_now=seq_lens_now,
        req_num_tokens_slice=req_num_tokens,
    )

    assert jnp.array_equal(valid, jnp.array([False, False, False]))
