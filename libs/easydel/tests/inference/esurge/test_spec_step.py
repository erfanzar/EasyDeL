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

"""Packed on-device speculative-step parity."""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.runners.spec.verify import split_packed
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35 as make_tiny_model

_HIDDEN = 128


def test_spec_step_matches_fused_reference_and_packs_every_row():
    """Greedy full/partial/zero acceptance and an idle row match the fused reference exactly."""
    model = make_tiny_model()
    n, k = 4, 3
    gathered = jnp.asarray(np.random.default_rng(7).normal(size=(n * (k + 1), _HIDDEN)), dtype=jnp.float32)
    argm = np.asarray(jnp.argmax(model.apply_lm_head(gathered), axis=-1))
    row_offsets = np.asarray([0, 4, 8, 0], dtype=np.int32)
    draft_tokens = np.asarray(
        [
            [int(argm[0]), int(argm[1]), int(argm[2])],
            [int(argm[4]), int(argm[5]) + 1, 7],
            [int(argm[8]) + 1, 3, 9],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    draft_counts = np.asarray([k, k, k, 0], dtype=np.int32)
    seed_pos_base = np.asarray([10, 20, 5, 0], dtype=np.int32)
    committed_base = np.zeros((n,), dtype=np.int32)
    fresh_mask = np.ones((n,), dtype=bool)
    collected_mask = np.asarray([True, True, True, False])

    reference = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    arg_ref, acc_ref, pos_ref, drafts_ref = reference.verify_draft_batched(
        gathered_hidden=gathered,
        row_offsets=row_offsets,
        draft_tokens=draft_tokens,
        draft_counts=draft_counts,
        seed_pos_base=seed_pos_base,
        committed_base=committed_base,
        fresh_mask=fresh_mask,
        collected_mask=collected_mask,
        num_draft_tokens=k,
    )

    candidate = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    arg_new, acc_new, pos_new, packed = candidate.spec_step(
        gathered_hidden=gathered,
        row_offsets=row_offsets,
        draft_tokens=draft_tokens,
        draft_counts=draft_counts,
        seed_pos_base=seed_pos_base,
        committed_base=committed_base,
        fresh_mask=fresh_mask,
        collected_mask=collected_mask,
        num_draft_tokens=k,
    )
    emitted, accepted, drafts = split_packed(np.asarray(packed), k)

    np.testing.assert_array_equal(np.asarray(arg_new), np.asarray(arg_ref))
    np.testing.assert_array_equal(np.asarray(acc_new[:3]), np.asarray(acc_ref[:3]))
    assert int(acc_new[3]) == -1
    np.testing.assert_array_equal(np.asarray(pos_new), np.asarray(pos_ref))
    np.testing.assert_array_equal(drafts[:3], np.asarray(drafts_ref[:3]))
    assert np.all(drafts[3] == -1)
    np.testing.assert_array_equal(accepted[:3], np.asarray(acc_ref[:3]))
    assert accepted[3] == -1
    assert packed.shape == (n, 2 * k + 2)
    assert np.all(emitted[3] == -1) and accepted[3] == -1 and np.all(drafts[3] == -1)

    np.testing.assert_array_equal(emitted[0], [draft_tokens[0, 0], draft_tokens[0, 1], draft_tokens[0, 2], argm[3]])
    np.testing.assert_array_equal(emitted[1, :2], [draft_tokens[1, 0], argm[5]])
    np.testing.assert_array_equal(emitted[2, 0], argm[8])
    assert np.all(emitted[1, 2:] == -1) and np.all(emitted[2, 1:] == -1)

    for row in range(n):
        old_index = np.asarray(reference._mtp_cache.views[0].indexes).reshape(-1)[row]
        new_index = np.asarray(candidate._mtp_cache.views[0].indexes).reshape(-1)[row]
        assert old_index == new_index
