"""Public compatibility APIs remain useful without in-tree production callers."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.norms import promote_dtype
from easydel.layers.residual import MeanHyperHead
from easydel.layers.rotary import gather_mla_cos_sin
from jax import numpy as jnp


def test_gather_mla_cos_sin_gathers_wrapped_tables_and_casts():
    table = np.arange(48, dtype=np.float32).reshape(6, 8)
    positions = np.array([[4, 1], [0, 5]], dtype=np.int32)
    cos, sin = gather_mla_cos_sin(SimpleNamespace(value=jnp.asarray(table)), jnp.asarray(positions), jnp.bfloat16)
    assert cos.dtype == sin.dtype == jnp.bfloat16
    assert cos.shape == sin.shape == (2, 1, 2, 4)
    np.testing.assert_array_equal(np.asarray(cos, dtype=np.float32), table[positions, :4][:, None])
    np.testing.assert_array_equal(np.asarray(sin, dtype=np.float32), table[positions, 4:][:, None])


def test_mean_hyper_head_module_computes_unweighted_mean():
    streams = np.arange(48, dtype=np.float32).reshape(1, 2, 3, 8)
    out = MeanHyperHead()(jnp.asarray(streams))
    assert out.shape == (1, 2, 8)
    assert out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(out), streams.mean(axis=2))


def test_shared_promote_dtype_casts_and_preserves_missing_values():
    value, missing = promote_dtype((jnp.ones((2,), dtype=jnp.float32), None), dtype=jnp.bfloat16, inexact=True)
    assert value.dtype == jnp.bfloat16
    assert missing is None
    np.testing.assert_array_equal(np.asarray(value, dtype=np.float32), np.ones(2, dtype=np.float32))


@pytest.mark.cpu_ok
@pytest.mark.parametrize("inexact", [None, False, True])
def test_shared_promote_dtype_without_target_preserves_values(inexact):
    values = [object(), None]
    assert promote_dtype(values, inexact=inexact) is values
    assert promote_dtype((None,), dtype=jnp.float32, inexact=inexact) == (None,)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("scale", [0.5, 1.0, 4.0, 16.0])
def test_yarn_mscale_public_scalar_formula(scale):
    import math

    from easydel.layers.rotary import yarn_get_mscale

    expected = 1.0 if scale <= 1 else 1.0 + 0.1 * math.log(scale)
    assert yarn_get_mscale(scale) == pytest.approx(expected)
    assert yarn_get_mscale(scale, mscale=1.0) == pytest.approx(expected)


@pytest.mark.cpu_ok
def test_sparse_attention_compatibility_exports_are_canonical():
    from easydel.layers import indexer, sparse_attention

    assert sparse_attention.BlockTopKIndexer is indexer.BlockTopKIndexer
    assert sparse_attention.apply_partial_rope is indexer.apply_partial_rope
