"""NumPy-only local-attention parity through the public cache concatenation API."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.attention import AttentionModule, _flexible
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


@dataclass
class _Mask:
    attention_mask: np.ndarray
    sliding_window_baked_in: bool = False

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class _CachedAttention(AttentionModule):
    def _handle_cache_concat(self, *, key, value, mask_info, cache_view, **kwargs):
        # The fixture supplies the already-written cache (including spare capacity).
        # Window selection and all mask changes run in the production layer.
        return key, value, mask_info, cache_view, None


def _numpy_vmap(fn, *, in_axes, out_axes):
    def apply(*args):
        batch = args[0].shape[0]
        rows = [
            fn(*(arg[i] if axis == 0 else arg for arg, axis in zip(args, in_axes, strict=True))) for i in range(batch)
        ]
        return tuple(np.stack([row[j] for row in rows]) for j in range(len(out_axes)))

    return apply


@pytest.mark.parametrize("capacity", [8, 16])
@pytest.mark.parametrize("window", [2, (2, 1)])
@pytest.mark.parametrize("decode", [False, True])
def test_every_query_retains_its_own_local_keys_and_matches_independent_attention(monkeypatch, capacity, window, decode):
    monkeypatch.setattr(_flexible, "jnp", np)
    monkeypatch.setattr(_flexible.jax, "vmap", _numpy_vmap)
    monkeypatch.setattr(
        _flexible.jax.lax,
        "broadcasted_iota",
        lambda dtype, shape, axis: np.broadcast_to(
            np.arange(shape[axis], dtype=dtype).reshape(tuple(shape[i] if i == axis else 1 for i in range(len(shape)))),
            shape,
        ),
    )

    def dynamic_slice(array, start, size, axis):
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(int(start), int(start) + size)
        return array[tuple(slices)]

    monkeypatch.setattr(_flexible.jax.lax, "dynamic_slice_in_dim", dynamic_slice)
    attention = _CachedAttention.__new__(_CachedAttention)
    attention.dtype = np.float32
    length = 3 if decode else 8
    first = 8 - length if decode else 0
    rows = np.arange(first, first + length)
    keys = np.arange(capacity, dtype=np.float32)[None, :, None, None]
    values = (keys + 1) ** 2
    queries = np.ones((1, length, 1, 1), dtype=np.float32) * 0.1
    mask = _Mask((np.arange(capacity)[None, :] <= rows[:, None])[None, None])
    cache = SimpleNamespace(key=None, starts=np.array([0]), indexes=np.array([8]), masking_details=None)
    mode = common_types.MODE_DECODE if decode else common_types.MODE_PREFILL
    selected_keys, selected_values, selected_mask, _, _, _ = attention.concatenate(
        query=queries, key=keys, value=values, mask_info=mask, mode=mode, cache_view=cache, sliding_window=window
    )
    left, right = (window - 1, window - 1) if isinstance(window, int) else window
    for query_index, position in enumerate(rows):
        expected_positions = np.arange(max(0, position - left), min(8, position + right + 1, position + 1))
        retained = selected_keys[0, :, 0, 0][selected_mask.attention_mask[0, 0, query_index]]
        np.testing.assert_array_equal(retained, expected_positions)
        logits = queries[0, query_index, 0, 0] * selected_keys[0, :, 0, 0]
        logits = np.where(selected_mask.attention_mask[0, 0, query_index], logits, -np.inf)
        weights = np.exp(logits - logits.max())
        actual = weights @ selected_values[0, :, 0, 0] / weights.sum()
        reference_weights = np.exp(0.1 * expected_positions - 0.1 * expected_positions.max())
        expected = reference_weights @ (expected_positions + 1) ** 2 / reference_weights.sum()
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert selected_mask.sliding_window_baked_in
    if not decode:
        assert selected_keys.shape[1] == capacity
