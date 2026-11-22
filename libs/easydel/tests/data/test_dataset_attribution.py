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

"""Dataset attribution: ``build_dataset`` stamps per-row dataset ids / assistant-only choices on every inform."""

from __future__ import annotations

import json
from collections import Counter
from itertools import islice

import pytest
from easydel.data import ASSISTANT_ONLY_FIELD, DATASET_ID_FIELD, DatasetMixture, TextDatasetInform
from easydel.data.execution.pipeline import build_dataset

pytestmark = pytest.mark.cpu_ok


def _informs(tmp_path, sizes, assistant_only=(None, None)):
    informs = []
    for (name, size), choice in zip(sizes.items(), assistant_only, strict=True):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"text": f"{name}:{i}", "junk": i}) + "\n" for i in range(size)))
        informs.append(TextDatasetInform(name=name, data_files=str(path), content_field="text", assistant_only=choice))
    return informs


def _mixture(tmp_path, *, sizes=None, assistant_only=(None, None), **kwargs):
    sizes = sizes or {"a": 40, "b": 40}
    kwargs.setdefault("streaming", True)
    return DatasetMixture(
        informs=_informs(tmp_path, sizes, assistant_only),
        use_fast_loader=True,
        batch_size=1,
        seed=42,
        cache_dir=str(tmp_path / "cache"),
        **kwargs,
    )


@pytest.mark.parametrize("streaming", [True, False])
def test_dataset_ids_follow_inform_order(tmp_path, streaming):
    mixture = _mixture(tmp_path, attach_dataset_ids=True, streaming=streaming, mixture_block_size=8)
    rows = list(islice(iter(build_dataset(mixture)), 80))
    assert rows
    for row in rows:
        assert row[DATASET_ID_FIELD] == {"a": 0, "b": 1}[row["text"].split(":")[0]]
        assert "junk" not in row  # the whitelist keeps only content + sideband
    assert set(Counter(r[DATASET_ID_FIELD] for r in rows)) == {0, 1}


def test_no_sideband_columns_by_default(tmp_path):
    row = next(iter(build_dataset(_mixture(tmp_path, mixture_block_size=8))))
    assert DATASET_ID_FIELD not in row and ASSISTANT_ONLY_FIELD not in row


def test_assistant_only_is_tristate_and_uniform_across_informs(tmp_path):
    """Any inform setting ``assistant_only`` stamps the column on all of them; ``None`` stays inherit (-1)."""
    mixture = _mixture(tmp_path, assistant_only=(False, None), mixture_block_size=8)
    rows = list(islice(iter(build_dataset(mixture)), 40))
    by_source = {r["text"].split(":")[0]: r[ASSISTANT_ONLY_FIELD] for r in rows}
    assert by_source == {"a": 0, "b": -1}, "False must stay opt-OUT, None must stay inherit"
    opt_in = _mixture(tmp_path, assistant_only=(True, None), mixture_block_size=8)
    assert {r["text"].split(":")[0]: r[ASSISTANT_ONLY_FIELD] for r in islice(iter(build_dataset(opt_in)), 40)} == {
        "a": 1,
        "b": -1,
    }


def test_error_strategy_streams_until_first_exhaustion(tmp_path):
    mixture = _mixture(tmp_path, sizes={"a": 3, "b": 40}, stop_strategy="error", mixture_block_size=8)
    with pytest.raises(RuntimeError, match="exhausted"):
        list(islice(iter(build_dataset(mixture)), 80))


@pytest.mark.parametrize("kwargs", [{"block_mixture": False}, {"streaming": False, "mixture_weights": {"a": 1, "b": 1}}])
def test_error_strategy_is_refused_where_it_cannot_be_implemented(tmp_path, kwargs):
    with pytest.raises(ValueError, match="stop_strategy='error'"):
        build_dataset(_mixture(tmp_path, stop_strategy="error", **kwargs))
