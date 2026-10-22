# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WindowRingDraftAttentionMixin ring semantics: slot ``p % window`` holds position ``p``; unwritten slots are hidden."""

import numpy as np
from easydel.layers.attention import _compressed_window as cw
from easydel.layers.compressed_window import MASK_MIN
from jax import numpy as jnp

WINDOW = 4


class _Stub(cw.WindowRingDraftAttentionMixin):
    sliding_window = WINDOW
    config = None
    rope_label = "main"

    def __init__(self):
        self.captured = None

    def main_kv(self, main_x, position_ids):
        return main_x

    def project_query(self, hidden_states, cos, sin):
        return hidden_states, hidden_states[:, None]

    def project_kv(self, hidden_states, cos, sin):
        return hidden_states

    def sink_attend(self, q, kv, attn_bias):
        self.captured = (kv, attn_bias)
        return q

    def project_output(self, attn_output, cos, sin):
        return attn_output


def _positions_value(seq):
    """``[1, seq, 1]`` main hidden whose value equals its position + 1 (so 0 means 'never written')."""
    return (jnp.arange(seq, dtype=jnp.float32) + 1.0)[None, :, None]


def test_seed_ring_short_prompt_fills_leading_slots():
    ring = np.asarray(_Stub().seed_ring(_positions_value(3)))
    np.testing.assert_array_equal(ring[0, :, 0], [1.0, 2.0, 3.0, 0.0])


def test_seed_ring_long_prompt_keeps_last_window_in_modulo_slots():
    ring = np.asarray(_Stub().seed_ring(_positions_value(6)))
    # positions 2..5 survive; position p lives in slot p % 4 -> slots [4, 5, 2, 3] (value = p + 1)
    np.testing.assert_array_equal(ring[0, :, 0], [5.0, 6.0, 3.0, 4.0])


def test_draft_attend_writes_ring_slot_and_hides_unwritten_slots(monkeypatch):
    monkeypatch.setattr(cw, "labeled_rope_cos_sin", lambda *args, **kwargs: (jnp.zeros(()), jnp.zeros(())))
    stub = _Stub()
    batch, block = 2, 2
    ring = jnp.zeros((batch, WINDOW, 1), jnp.float32)
    hidden = jnp.ones((batch, block, 1), jnp.float32)
    main_x = jnp.asarray([[[10.0]], [[60.0]]])
    position = jnp.asarray([1, 6], jnp.int32)

    _, new_ring = stub.draft_attend(hidden, main_x, ring, position)

    new_ring = np.asarray(new_ring)
    assert new_ring[0, 1, 0] == 10.0  # position 1 -> slot 1
    assert new_ring[1, 6 % WINDOW, 0] == 60.0  # position 6 -> slot 2
    _, bias = stub.captured
    bias = np.asarray(bias)
    assert bias.shape == (batch, 1, 1, WINDOW + block)
    # row 0: only slots 0..1 written; row 1: the ring is full
    np.testing.assert_array_equal(bias[0, 0, 0, :WINDOW] == 0.0, [True, True, False, False])
    assert (bias[0, 0, 0, :WINDOW][2:] == np.float32(MASK_MIN)).all()
    assert (bias[1, 0, 0, :WINDOW] == 0.0).all()
    assert (bias[:, 0, 0, WINDOW:] == 0.0).all()  # the draft block's own keys stay visible
