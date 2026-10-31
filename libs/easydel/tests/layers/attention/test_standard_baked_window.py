"""Standard attention retains its configured window until mask preparation."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.layers.attention import UnifiedAttention, _flexible, _unified
from easydel.operations import AttentionOutput
from ejkernel.types import MaskInfo
from spectrax import common_types


class _HostStandardAttention(UnifiedAttention):
    """Use host projections with the real standard forward and concatenate paths."""

    def _project_qkv(self, hidden_states):
        return hidden_states, hidden_states, hidden_states

    def apply_qkv_shardings(self, query, key, value):
        return query, key, value

    def _apply_rotary(self, query, key, position_ids, frequencies):
        return query, key

    def _softmax_aux(self):
        return None

    def output_projection(self, states):
        return states

    def shard_attention_prod(self, states, **kwargs):
        return states


class _HostMaskPerformer:
    def forward(self, *, query_states, key_states, value_states, mask_info, sliding_window, causal, **kwargs):
        # Host-only reference backend: zero scores give uniform probability over
        # visible keys. Also implement the kernel's optional local-index window
        # so retaining an already-baked window can affect the public result.
        visible = np.array(mask_info.attention_mask, copy=True)
        relative = np.arange(key_states.shape[1])[None, :] - np.arange(query_states.shape[1])[:, None]
        if sliding_window is not None:
            left, right = (max(sliding_window - 1, 0),) * 2 if isinstance(sliding_window, int) else sliding_window
            visible &= ((relative >= -left) & (relative <= right))[None, None]
        if causal:
            visible &= (relative <= 0)[None, None]
        weights = visible.astype(np.float32)
        weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1)
        output = np.einsum("bhqk,bkhd->bqhd", weights, value_states)
        return AttentionOutput(attention_outputs=output, attention_weights=weights)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("baked", [False, True])
@pytest.mark.parametrize("window", [0, 1, 2, (0, 1), (2, 0), 5, None])
def test_standard_forward_intersects_configured_window_with_caller_mask(monkeypatch, length, baked, window):
    monkeypatch.setattr(_flexible, "jnp", np)
    monkeypatch.setattr(_unified, "checkpoint_name", lambda value, name: value)
    attention = _HostStandardAttention.__new__(_HostStandardAttention)
    attention.dtype = np.float32
    attention.config = SimpleNamespace(runtime_sharding_resolver=None, kv_cache_quantization_config=None)
    attention.num_heads = attention.num_key_value_heads = 1
    attention.head_dim = 2
    attention.causal = False
    attention.sliding_window = window
    attention.precision = None
    attention.attention_performer = _HostMaskPerformer()

    # Explicit nonlocal coordinates catch accidental reapplication by the kernel
    # in local-column coordinates as well as premature suppression of the window.
    query_positions = np.arange(10, 10 + length)[None]
    key_positions = np.arange(9, 9 + length)[None]
    relative = key_positions[:, None, :] - query_positions[:, :, None]
    caller_visible = np.ones((1, length, length), dtype=bool)
    if length > 1:
        caller_visible[:, 1, 1] = False
    mask = MaskInfo(
        _attention_mask=caller_visible[:, None],
        q_positions=query_positions,
        kv_positions=key_positions,
        sliding_window_baked_in=baked,
    )
    states = np.arange(length * 2, dtype=np.float32).reshape(1, length, 2) + 1
    output = attention.forward_standard(
        hidden_states=states,
        mask_info=mask,
        position_ids=query_positions,
        mode=common_types.MODE_PREFILL,
        output_attentions=True,
    )
    expected_visible = caller_visible.copy()
    if window is not None:
        left, right = (max(window - 1, 0),) * 2 if isinstance(window, int) else window
        expected_visible &= (relative >= -left) & (relative <= right)
    expected_weights = expected_visible[:, None].astype(np.float32)
    expected_weights /= np.maximum(expected_weights.sum(axis=-1, keepdims=True), 1)
    np.testing.assert_array_equal(output.attention_weight, expected_weights)
    expected_output = np.einsum("bhqk,bkd->bqd", expected_weights, states)
    np.testing.assert_allclose(output.attention_output, expected_output, rtol=1e-6, atol=1e-6)
