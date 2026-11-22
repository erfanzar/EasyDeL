import jax
import pytest
from easydel.operations.kernels._paged_common import request_distribution_bounds
from jax import numpy as jnp


@pytest.mark.parametrize(
    "scheduled, context_lens",
    [
        ([], []),
        ([0, 0], [0, 9]),
        ([1], [1]),
        ([1], [2]),
        ([1, 1, 2, 0], [3, 1, 5, 8]),
        ([0, 3, 1, -1, 1], [0, 7, 9, 8, 1]),
        ([2, 3], [2, 9]),
    ],
)
@pytest.mark.parametrize("compiled", [False, True])
def test_distribution_matches_independent_counts(scheduled, context_lens, compiled):
    if jax.default_backend() not in {"tpu", "gpu", "cuda", "rocm"}:
        pytest.skip("Request-distribution numerics require an accelerator.")
    decode = sum(count == 1 and context > 1 for count, context in zip(scheduled, context_lens, strict=True))
    total = sum(count > 0 for count in scheduled)
    operation = jax.jit(request_distribution_bounds) if compiled else request_distribution_bounds
    result = operation(jnp.asarray(scheduled, dtype=jnp.int32), jnp.asarray(context_lens, dtype=jnp.int32))
    assert result.shape == (3,)
    assert result.dtype == jnp.dtype(jnp.int32)
    assert result.tolist() == [decode, total, total]
