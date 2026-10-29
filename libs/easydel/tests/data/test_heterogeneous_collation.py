"""CPU-only regressions for heterogeneous row collation."""

import numpy as np
import pytest
from easydel.data.execution.loader import batch_iterator, collate_batch

pytestmark = pytest.mark.cpu_ok


def test_collate_batch_preserves_ordered_key_union_and_row_alignment():
    rows = [
        {"input_ids": np.array([1, 2], dtype=np.int32), "lang": "en"},
        {"input_ids": np.array([3, 4], dtype=np.int32), "score": 7},
        {"input_ids": np.array([5, 6], dtype=np.int32), "lang": "fr", "score": 9},
    ]
    batch = collate_batch(rows)
    assert list(batch) == ["input_ids", "lang", "score"]
    np.testing.assert_array_equal(batch["input_ids"], [[1, 2], [3, 4], [5, 6]])
    assert batch["input_ids"].dtype == np.int32
    assert batch["lang"].dtype == object
    assert batch["lang"].tolist() == ["en", None, "fr"]
    assert batch["score"].dtype == object
    assert batch["score"].tolist() == [None, 7, 9]
    assert list(rows[0]) == ["input_ids", "lang"]
    assert list(rows[1]) == ["input_ids", "score"]


@pytest.mark.parametrize("absent", [{}, {"embeds": None}])
@pytest.mark.parametrize("present_first", [False, True])
def test_optional_vector_fields_use_one_object_slot_per_row(absent, present_first):
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    present = {"embeds": vectors}
    rows = [present, absent] if present_first else [absent, present]
    batch = collate_batch(rows)
    assert batch["embeds"].dtype == object
    assert batch["embeds"].shape == (2,)
    assert batch["embeds"][1 if present_first else 0] is None
    np.testing.assert_array_equal(batch["embeds"][0 if present_first else 1], vectors)


def test_batch_iterator_keeps_optional_fields_in_partial_batches():
    rows = [{}, {"label": 7}, {"label": None, "metadata": {"source": "last"}}]
    batches = list(batch_iterator(iter(rows), batch_size=2, drop_last=False))
    assert len(batches) == 2
    assert batches[0]["label"].tolist() == [None, 7]
    assert batches[1]["label"].tolist() == [None]
    assert batches[1]["metadata"].tolist() == [{"source": "last"}]
    assert collate_batch([]) == {}


def test_fully_present_fields_still_stack_and_ragged_fields_survive():
    rows = [{"value": 1.5, "tokens": [1]}, {"value": 2.5, "tokens": [2, 3]}]
    batch = collate_batch(rows)
    np.testing.assert_array_equal(batch["value"], [1.5, 2.5])
    assert batch["value"].dtype == np.float64
    assert batch["tokens"].dtype == object
    assert batch["tokens"].tolist() == [[1], [2, 3]]
