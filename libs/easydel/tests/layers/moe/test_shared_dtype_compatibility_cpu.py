"""Behavioral coverage for historical dtype imports (casts require an accelerator)."""

import numpy as np
import pytest
from easydel.layers.linears._linear import promote_dtype as linear_promote_dtype
from easydel.layers.linears._linear_moe import promote_dtype as moe_promote_dtype
from easydel.layers.norms import promote_dtype
from jax import numpy as jnp


@pytest.mark.cpu_ok
@pytest.mark.parametrize("promoter", [promote_dtype, linear_promote_dtype, moe_promote_dtype])
@pytest.mark.parametrize("container", [list, tuple, iter])
def test_legacy_dtype_helpers_preserve_unspecified_dtype_identity(promoter, container):
    values = container([object(), None])
    assert promoter(values, dtype=None) is values


@pytest.mark.cpu_ok
@pytest.mark.parametrize("promoter", [linear_promote_dtype, moe_promote_dtype])
def test_legacy_dtype_helpers_preserve_absent_values_with_explicit_dtype(promoter):
    # No arrays are created: missing bias/kernel placeholders must stay missing.
    assert promoter([None, None], dtype="float32") == (None, None)


@pytest.mark.parametrize("promoter", [linear_promote_dtype, moe_promote_dtype])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_linear_dtype_imports_cast_nonmissing_values(promoter, dtype):
    values = jnp.asarray([1.25, -2.5, 0.0], dtype=jnp.float32)
    actual, missing = promoter([values, None], dtype=dtype)
    assert actual.dtype == dtype
    assert missing is None
    np.testing.assert_array_equal(np.asarray(actual).astype(np.float32), [1.25, -2.5, 0.0])
