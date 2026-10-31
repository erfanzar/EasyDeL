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

"""``collate_packed_embeds``: flat vs window-blocked embed layouts, ``rows_per_window`` bounds and the
per-token attribution sideband arrays. NumPy-only (host-side collation, no array-library computation)."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from easydel.data import ASSISTANT_ONLY_FIELD, DATASET_ID_FIELD
from easydel.data.transforms.collators import collate_packed_embeds

pytestmark = pytest.mark.cpu_ok

PAD_ID, IMAGE_TOKEN_ID, EMBED_DIM = 0, 99, 4
SEQ_LEN = 16


def _position_id_fn(
    input_ids, mm_token_type_ids=None, image_grid_thw=None, attention_mask=None, return_jax_arrays=False
):
    length = np.asarray(input_ids).shape[-1]
    return np.arange(length, dtype=np.int32).reshape(1, 1, -1).repeat(3, axis=0), None


def _row(n_images: int, tag: float, n_text: int = 2, **extra):
    ids = [1] * n_text + [IMAGE_TOKEN_ID] * n_images
    embeds = (np.arange(n_images * EMBED_DIM, dtype=np.float32) + tag).reshape(n_images, EMBED_DIM)
    row = {
        "input_ids": np.asarray(ids, dtype=np.int32),
        "attention_mask": np.ones(len(ids), dtype=np.int32),
        "labels": np.asarray(ids, dtype=np.int32),
        "image_embeds": [embeds.astype(ml_dtypes.bfloat16).tobytes()] if n_images else [],
        "embed_n_tok": [n_images] if n_images else [],
        "image_grid_thw": np.ones((n_images, 3), dtype=np.int32),
    }
    row.update(extra)
    return row


def _collate(counts, *, rows_per_window=None, max_embed_rows=None, seq_len=SEQ_LEN):
    packs = [[_row(c, 100.0 * (w + 1))] for w, c in enumerate(counts)]
    n_windows = len(counts)
    if max_embed_rows is None:
        max_embed_rows = n_windows * rows_per_window if rows_per_window else max(sum(counts), 1)
    return collate_packed_embeds(
        packs,
        PAD_ID,
        seq_len,
        max_embed_rows,
        n_windows,
        image_token_id=IMAGE_TOKEN_ID,
        embed_dim=EMBED_DIM,
        position_id_fn=_position_id_fn,
        rows_per_window=rows_per_window,
    )


def _gather(out, rows_per_window=None):
    """Independent host reference of a merge: embed row used by each placeholder, in placeholder order."""
    ids = np.asarray(out["input_ids"])
    emb = np.asarray(out["image_embeds"], dtype=np.float32)
    result = np.zeros((*ids.shape, EMBED_DIM), np.float32)
    if rows_per_window is None:
        result[ids == IMAGE_TOKEN_ID] = emb[: int((ids == IMAGE_TOKEN_ID).sum())]
    else:
        blocks = emb.reshape(ids.shape[0], rows_per_window, EMBED_DIM)
        for w in range(ids.shape[0]):
            mask = ids[w] == IMAGE_TOKEN_ID
            result[w][mask] = blocks[w][: int(mask.sum())]
    return result


def _scatter_from_coordinates(out):
    """Place embed rows using the collator's own (window, position) coordinates and mask."""
    ids = np.asarray(out["input_ids"])
    emb = np.asarray(out["image_embeds"], dtype=np.float32)
    result = np.zeros((*ids.shape, EMBED_DIM), np.float32)
    for idx in np.flatnonzero(np.asarray(out["image_embed_mask"])):
        w, p = out["image_embed_positions"][idx]
        result[w, p] = emb[idx]
    return result


def test_flat_layout_is_the_default():
    out = _collate([1, 2])
    assert out["image_embeds"].shape == (3, EMBED_DIM)
    assert int(out["image_embed_mask"].sum()) == 3
    np.testing.assert_allclose(out["image_embeds"][0], np.arange(EMBED_DIM) + 100.0)
    np.testing.assert_allclose(out["image_embeds"][1], np.arange(EMBED_DIM) + 200.0)


def test_window_blocked_places_each_window_in_its_own_block():
    k = 4
    emb = np.asarray(_collate([1, 2], rows_per_window=k)["image_embeds"], np.float32)
    assert emb.shape == (2 * k, EMBED_DIM)
    np.testing.assert_allclose(emb[0], np.arange(EMBED_DIM) + 100.0)
    np.testing.assert_allclose(emb[k], np.arange(EMBED_DIM) + 200.0)
    np.testing.assert_allclose(emb[k + 1], np.arange(EMBED_DIM, 2 * EMBED_DIM) + 200.0)
    np.testing.assert_allclose(emb[1:k], 0.0)
    np.testing.assert_allclose(emb[k + 2 :], 0.0)


DISTRIBUTIONS = [[1, 2], [0, 3], [0, 0], [3, 3], [5, 1, 0, 2], [1], [0, 0, 0, 4], [4, 0, 0, 0]]


@pytest.mark.parametrize("counts", DISTRIBUTIONS)
def test_layouts_attach_identical_embeds_to_every_placeholder(counts):
    k = max(max(counts), 1)
    flat, blocked = _collate(counts), _collate(counts, rows_per_window=k)
    np.testing.assert_array_equal(flat["input_ids"], blocked["input_ids"])
    reference = _gather(flat)
    np.testing.assert_array_equal(_gather(blocked, rows_per_window=k), reference)
    # The coordinate/mask rows land at the same index as their embed rows in both layouts.
    np.testing.assert_array_equal(_scatter_from_coordinates(flat), reference)
    np.testing.assert_array_equal(_scatter_from_coordinates(blocked), reference)
    for w, c in enumerate(counts):
        if c:  # each window carries its OWN tag (100*(w+1)), so a cross-window leak shows as a wrong tag
            assert reference[w][flat["input_ids"][w] == IMAGE_TOKEN_ID][0, 0] == 100.0 * (w + 1)


@pytest.mark.parametrize("counts", DISTRIBUTIONS)
def test_blocked_marked_rows_never_leave_their_window_block(counts):
    k = max(max(counts), 1)
    out = _collate(counts, rows_per_window=k)
    for idx in np.flatnonzero(out["image_embed_mask"]):
        assert out["image_embed_positions"][idx, 0] == idx // k


def test_randomized_layout_equivalence_and_overprovisioned_k():
    rng = np.random.default_rng(0)
    for _ in range(40):
        counts = [int(rng.integers(0, 5)) for _ in range(int(rng.integers(1, 5)))]
        k = max(max(counts), 1) + int(rng.integers(0, 3))
        np.testing.assert_array_equal(
            _gather(_collate(counts, rows_per_window=k), rows_per_window=k),
            _gather(_collate(counts)),
            err_msg=f"counts={counts} K={k}",
        )


def test_blocked_requires_max_embed_rows_to_match():
    with pytest.raises(ValueError, match="rows_per_window"):
        _collate([1, 2], rows_per_window=4, max_embed_rows=7)


def test_blocked_refuses_a_window_overflowing_its_block():
    with pytest.raises(ValueError, match="NEXT window"):
        _collate([1, 3], rows_per_window=2)


@pytest.mark.parametrize("bad", [0, -1, True, 2.0])
def test_blocked_rejects_invalid_k(bad):
    with pytest.raises(ValueError, match="rows_per_window"):
        _collate([1, 2], rows_per_window=bad, max_embed_rows=2)


def test_k_beyond_seq_len_is_refused_with_actionable_message():
    k = SEQ_LEN + 1
    with pytest.raises(ValueError, match=r"exceeds seq_len.*max\(placeholders per window\)"):
        _collate([1, 2], rows_per_window=k)


def test_k_equal_to_seq_len_is_allowed_but_dense_k_is_smaller():
    worst = _collate([1, 2], rows_per_window=SEQ_LEN)
    dense = _collate([1, 2], rows_per_window=2)
    assert worst["image_embeds"].shape[0] == 2 * SEQ_LEN
    assert worst["image_embeds"].shape[0] / dense["image_embeds"].shape[0] == SEQ_LEN / 2


def test_blocked_refuses_per_row_placeholder_embed_drift():
    """Two rows drifting in opposite directions satisfy a batch-wide count check but would mis-slice blocks."""
    bad = _row(2, 100.0)
    bad["input_ids"] = np.asarray([1, 1, IMAGE_TOKEN_ID], np.int32)  # 1 placeholder, 2 embed rows
    bad["attention_mask"] = np.ones(3, np.int32)
    bad["labels"] = bad["input_ids"].copy()
    with pytest.raises(ValueError, match="placeholder"):
        collate_packed_embeds(
            [[bad], [_row(1, 200.0)]],
            PAD_ID,
            SEQ_LEN,
            2 * 3,
            2,
            image_token_id=IMAGE_TOKEN_ID,
            embed_dim=EMBED_DIM,
            position_id_fn=_position_id_fn,
            rows_per_window=3,
        )


def _text_collate(packs, n_windows=2, seq_len=8):
    return collate_packed_embeds(
        packs, PAD_ID, seq_len, 1, n_windows, image_token_id=IMAGE_TOKEN_ID, embed_dim=EMBED_DIM
    )


def test_dataset_ids_and_assistant_only_are_broadcast_per_document():
    packs = [
        [
            _row(0, 0.0, n_text=3, **{DATASET_ID_FIELD: 0, ASSISTANT_ONLY_FIELD: True}),
            _row(0, 0.0, n_text=4, **{DATASET_ID_FIELD: 2, ASSISTANT_ONLY_FIELD: -1}),
        ],
        [_row(0, 0.0, n_text=2, **{DATASET_ID_FIELD: np.int64(5), ASSISTANT_ONLY_FIELD: np.asarray(False)})],
    ]
    out = _text_collate(packs)
    assert out["dataset_ids"].shape == out["assistant_only_mask"].shape == (2, 8)
    assert out["dataset_ids"].dtype == np.int32
    np.testing.assert_array_equal(out["dataset_ids"][0], [0, 0, 0, 2, 2, 2, 2, -1])
    np.testing.assert_array_equal(out["dataset_ids"][1], [5, 5, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(out["assistant_only_mask"][0], [1, 1, 1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(out["assistant_only_mask"][1], [0, 0, -1, -1, -1, -1, -1, -1])


def test_rows_without_sideband_stay_unattributed():
    out = _text_collate([[_row(0, 0.0, n_text=3)]], n_windows=2)
    assert (out["dataset_ids"] == -1).all() and (out["assistant_only_mask"] == -1).all()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (DATASET_ID_FIELD, [1]),
        (DATASET_ID_FIELD, np.asarray([1, 2])),
        (DATASET_ID_FIELD, True),
        (DATASET_ID_FIELD, 1.5),
        (ASSISTANT_ONLY_FIELD, [False]),
        (ASSISTANT_ONLY_FIELD, 2),
        (ASSISTANT_ONLY_FIELD, "yes"),
    ],
)
def test_non_scalar_or_out_of_domain_sideband_is_refused(field, value):
    with pytest.raises(TypeError, match=field):
        _text_collate([[_row(0, 0.0, n_text=3, **{field: value})]])
