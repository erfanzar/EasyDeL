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
"""Per-row dataset attribution must survive tokenization/purify and reach the packed collator and census."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.data import ASSISTANT_ONLY_FIELD, DATASET_ID_FIELD
from easydel.data.transforms.collators import collate_packed_embeds
from easydel.trainers.assistant_only_mask import ZeroSupervisionStats, zero_supervision_stats
from easydel.trainers.data_lanes import StampSourceSideband
from easydel.trainers.prompt_transforms import SFTPreprocessTransform, purify_example

pytestmark = pytest.mark.cpu_ok

_WORDS = ["[PAD]", "[UNK]", "<e>", *[f"w{i}" for i in range(20)]]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(_WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="<e>")


def test_purify_keeps_both_sideband_carriers_and_drops_text():
    out = purify_example(
        {"input_ids": [1, 2], "attention_mask": [1, 1], DATASET_ID_FIELD: 3, ASSISTANT_ONLY_FIELD: 0, "messages": "x"}
    )
    assert out[DATASET_ID_FIELD] == 3
    assert out[ASSISTANT_ONLY_FIELD] == 0
    assert "messages" not in out


def test_sft_transform_carries_dataset_id_through_text_tokenization():
    transform = SFTPreprocessTransform(tokenizer=_tokenizer(), max_length=16, text_field="text", padding=False)
    out = transform({"text": "w1 w2 w3", DATASET_ID_FIELD: 4, "extra_text": "dropped"})
    assert out[DATASET_ID_FIELD] == 4
    assert out["input_ids"][:3] == [_WORDS.index(w) for w in ("w1", "w2", "w3")]
    assert "extra_text" not in out


def test_purified_rows_attribute_per_token_in_the_packed_collator():
    rows = []
    for ds_id in (0, 2):
        row = purify_example(
            {"input_ids": [5, 6, 7, 8], "attention_mask": [1] * 4, "labels": [5, 6, 7, 8], DATASET_ID_FIELD: ds_id}
        )
        row.update(image_grid_thw=[], image_embeds=[], embed_n_tok=[], embed_dim=8, n_images=0)
        rows.append(row)
    packed = collate_packed_embeds(
        [rows], pad_id=0, seq_len=8, max_embed_rows=1, n_windows=1, image_token_id=-1, embed_dim=8
    )
    ds = np.asarray(packed["dataset_ids"])[0]
    assert ds[:4].tolist() == [0] * 4
    assert ds[4:].tolist() == [2] * 4


def test_census_buckets_zero_supervision_documents_by_dataset():
    ids = np.arange(2, 10, dtype=np.int32)[None, :]
    seg = np.array([[0] * 4 + [1] * 4], dtype=np.int32)
    cm = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int32)
    ds_tok = np.array([[0] * 4 + [2] * 4], dtype=np.int32)
    stats = zero_supervision_stats(ids, cm, segment_ids=seg, dataset_ids_per_token=ds_tok)
    assert (stats.n_docs, stats.n_zero) == (2, 1)
    assert stats.per_dataset == {0: (1, 0), 2: (1, 1)}
    assert len(stats.samples) == 1 and stats.samples[0][0] == 2


def test_census_metrics_use_dataset_names_and_samples_are_capped():
    named = ZeroSupervisionStats(
        n_docs=2,
        n_zero=1,
        n_zero_with_header=0,
        n_zero_without_header=1,
        per_dataset={0: (1, 0), 2: (1, 1)},
        dataset_names={0: "chat", 2: "reasoning"},
    )
    metrics = named.as_metrics()
    assert metrics["data/zero_supervision_frac/reasoning"] == 1.0
    assert metrics["data/zero_supervision_frac/chat"] == 0.0

    n, length = 40, 8
    stats = zero_supervision_stats(
        np.arange(n * length).reshape(n, length) + 2,
        np.zeros((n, length), np.int32),
        attention_mask=np.ones((n, length), np.int32),
        sample_limit=3,
    )
    assert stats.n_zero == n and len(stats.samples) == 3


def test_census_separates_truncated_headers_from_rows_without_assistant_turn():
    header = (7, 8)
    ids = np.array([[1, 7, 8, 9], [1, 2, 3, 4]], dtype=np.int32)
    stats = zero_supervision_stats(
        ids, np.zeros_like(ids), attention_mask=np.ones_like(ids), header_ids=header, sample_limit=0
    )
    assert (stats.n_zero_with_header, stats.n_zero_without_header) == (1, 1)


def test_grain_lane_stamp_is_tristate_and_does_not_mutate_the_row():
    assert StampSourceSideband(None, 0)({"input_ids": [1]})[ASSISTANT_ONLY_FIELD] == -1
    opt_out = StampSourceSideband(False, 5)({"input_ids": [1]})
    assert (opt_out[ASSISTANT_ONLY_FIELD], opt_out[DATASET_ID_FIELD]) == (0, 5)
    row = {"input_ids": [1]}
    assert StampSourceSideband(True, 1)(row)[ASSISTANT_ONLY_FIELD] == 1
    assert row == {"input_ids": [1]}


def test_dataset_id_names_follow_the_stamp_order_including_duplicates():
    from easydel.infra.elarge.builders import resolve_dataset_id_names

    names = resolve_dataset_id_names({"informs": [{"name": "a"}, {"name": "b"}, {"name": "b"}]})
    assert names == ["a", "b", "b-2"]
