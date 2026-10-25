"""Host-only argument/layout regressions; NumPy fixtures never dispatch JAX."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules import _drafting
from easydel.modules._drafting import DraftAttention, extract_context_feature
from easydel.modules.dspark.modeling_dspark import DSparkModel

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("layout", ["layer_first", "sequence_first"])
def test_explicit_stacked_target_layout_selects_subset_of_complete_stack(monkeypatch, layout):
    monkeypatch.setattr(_drafting, "jnp", np)
    stack = np.broadcast_to(np.arange(9)[:, None, None, None], (9, 1, 16, 2))
    values = stack if layout == "layer_first" else stack.transpose(1, 2, 0, 3)
    expected = np.broadcast_to([1, 1, 3, 3], (1, 16, 4))
    np.testing.assert_array_equal(extract_context_feature(values, [0, 2], stacked_layout=layout), expected)
    np.testing.assert_array_equal(extract_context_feature(values, [0, 2], num_target_layers=8), expected)


def test_public_drafter_projection_rejects_ambiguous_layout_instead_of_corrupting_axes(monkeypatch):
    monkeypatch.setattr(_drafting, "jnp", np)

    class HostDrafter(DSparkModel):
        def __init__(self):
            self.config = SimpleNamespace(target_layer_ids=(0, 2), target_hidden_size=2)
            self.fc = lambda values: values
            self.hidden_norm = lambda values: values

    model = HostDrafter()
    stack = np.broadcast_to(np.arange(9)[:, None, None, None], (9, 1, 16, 2))
    with pytest.raises(ValueError, match=r"Ambiguous.*stacked_layout"):
        model.project_target_hidden_states(stack)
    # A short sequence unambiguously identifies the complete layer-first
    # stack, even though selectors omit all upper layers.
    np.testing.assert_array_equal(
        model.project_target_hidden_states(stack[:, :, :2]), np.broadcast_to([1, 1, 3, 3], (1, 2, 4))
    )


@pytest.mark.parametrize("position_mode", ["distinct", "query_only", "key_only"])
def test_public_draft_attention_honors_distinct_same_length_key_positions(monkeypatch, position_mode):
    monkeypatch.setattr(_drafting, "jnp", np)

    def host_softmax(scores, axis):
        scores = np.exp(scores - scores.max(axis=axis, keepdims=True))
        return scores / scores.sum(axis=axis, keepdims=True)

    monkeypatch.setattr(_drafting.jax.nn, "softmax", host_softmax)

    class HostAttention(DraftAttention):
        def __init__(self):
            self.num_heads = self.num_kv_heads = self.num_key_value_groups = self.head_dim = 1
            self.scaling = 1.0
            self.q_proj = self.k_proj = self.v_proj = self.o_proj = lambda value: value
            self.q_norm = self.k_norm = lambda value: value

        def rotary(self, *, positions, query, key, frequencies):
            # A position-labeled host rotary fixture makes routing errors
            # observable through the public attention output.
            offsets = positions[:, :, None, None]
            return query + offsets, key + offsets

    model = HostAttention()
    queries = np.array([[[0.5], [1.0]]])
    keys = np.array([[[1.0], [3.0]]])
    q_positions = None if position_mode == "key_only" else np.array([[0, 1]])
    kv_positions = None if position_mode == "query_only" else np.array([[8, 10]])
    q_offsets = 0 if q_positions is None else q_positions
    kv_offsets = 0 if kv_positions is None else kv_positions
    scores = (queries[..., 0] + q_offsets)[:, :, None] * (keys[..., 0] + kv_offsets)[:, None, :]
    expected = host_softmax(scores, axis=-1) @ keys
    np.testing.assert_allclose(
        model(queries, keys, q_position_ids=q_positions, kv_position_ids=kv_positions), expected, rtol=1e-6
    )
    wrong = model(queries, keys, q_position_ids=q_positions, kv_position_ids=q_positions)
    assert not np.allclose(wrong, expected)
