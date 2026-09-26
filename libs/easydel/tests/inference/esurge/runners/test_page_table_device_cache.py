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

"""Characterization: the page-table device transfer happens only on version change.

``BatchMetadataPreparer`` caches the device-side ``pages_tables`` array keyed by
``page_table_version`` (plus the row count copied). Steady-state decode steps
must reuse the cached device array — re-transferring a
``[num_reqs, max_pages_per_req]`` table every step is a silent throughput
regression. These tests lock:

* same version twice -> the *identical* ``jax.Array`` object is returned
  (no new transfer), and
* a bumped version -> a fresh device array (stale reuse would corrupt KV
  addressing), and
* the async double-buffer contract: a second ``start_async_prep`` without an
  intervening ``get_async_prep_result`` raises.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer

MAX_NUM_TOKENS = 8
MAX_NUM_REQS = 4
MAX_MODEL_LEN = 16
MAX_PAGES_PER_REQ = 2


def _dummy_metadata():
    # dp_size=1: the device page-table cache path only applies off the SPMD-DP
    # rank-major path.
    return SimpleNamespace(
        version="v3",
        get_max_num_seqs=lambda: MAX_NUM_REQS,
        max_num_pages_per_req=MAX_PAGES_PER_REQ,
        data_parallel_size=1,
        page_size=16,
        num_pages=0,
    )


def _make_preparer() -> BatchMetadataPreparer:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    return BatchMetadataPreparer(
        metadata=_dummy_metadata(),
        empty_sharding=sharding,
        max_num_tokens=MAX_NUM_TOKENS,
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        min_input_pad=1,
        input_sharding=sharding,
    )


def _prep_kwargs(page_table_version: int | None):
    """One decode-shaped batch (1 token per request, all rows active)."""
    return dict(
        num_tokens_static=MAX_NUM_TOKENS,
        scheduled_full_cpu=np.ones((MAX_NUM_REQS,), dtype=np.int32),
        active_mask_full_cpu=np.ones((MAX_NUM_REQS,), dtype=bool),
        token_ids_cpu=np.arange(MAX_NUM_REQS * MAX_MODEL_LEN, dtype=np.int32).reshape(MAX_NUM_REQS, MAX_MODEL_LEN),
        num_computed_tokens_cpu=np.full((MAX_NUM_REQS,), 3, dtype=np.int32),
        temperature_cpu=np.zeros((MAX_NUM_REQS,), dtype=np.float32),
        top_p_cpu=np.ones((MAX_NUM_REQS,), dtype=np.float32),
        top_k_cpu=np.zeros((MAX_NUM_REQS,), dtype=np.int32),
        min_p_cpu=np.zeros((MAX_NUM_REQS,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((MAX_NUM_REQS,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((MAX_NUM_REQS,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((MAX_NUM_REQS,), dtype=np.float32),
        page_table_cpu=np.arange(MAX_NUM_REQS * MAX_PAGES_PER_REQ, dtype=np.int32).reshape(
            MAX_NUM_REQS, MAX_PAGES_PER_REQ
        ),
        padded_num_reqs_in=MAX_NUM_REQS,
        page_table_version=page_table_version,
        window_row_indices_cpu=np.arange(MAX_NUM_REQS, dtype=np.int32),
    )


def _device_buffers(preparer: BatchMetadataPreparer) -> tuple[jax.Array, jax.Array]:
    sharding = preparer._input_sharding
    input_ids_buf = jax.device_put(np.zeros((MAX_NUM_TOKENS,), dtype=np.int32), sharding)
    position_ids_buf = jax.device_put(np.zeros((MAX_NUM_TOKENS,), dtype=np.int32), sharding)
    return input_ids_buf, position_ids_buf


def _prepare(preparer: BatchMetadataPreparer, version: int | None):
    input_ids_buf, position_ids_buf = _device_buffers(preparer)
    batch_metadata, *_ = preparer.prepare_batch_metadata(
        input_ids_buf=input_ids_buf,
        position_ids_buf=position_ids_buf,
        **_prep_kwargs(version),
    )
    return batch_metadata


def test_same_page_table_version_reuses_device_array():
    preparer = _make_preparer()

    first = _prepare(preparer, version=7)
    second = _prepare(preparer, version=7)

    # Object identity, not value equality: a value-equal copy still means a
    # fresh host->device transfer happened on the decode hot path.
    assert second.pages_tables is first.pages_tables
    assert preparer._cached_page_table_version == 7
    assert second.pages_tables is preparer._cached_pages_tables_dev


def test_bumped_page_table_version_transfers_fresh_array():
    preparer = _make_preparer()

    first = _prepare(preparer, version=7)
    second = _prepare(preparer, version=8)

    assert second.pages_tables is not first.pages_tables
    assert preparer._cached_page_table_version == 8


def test_none_version_never_reuses():
    """Callers that do not thread a version degrade to transfer-every-step.

    This locks the None-handling contract: no crash, no stale reuse. (A
    refactor that silently drops the version plumbing is caught by
    ``test_same_page_table_version_reuses_device_array`` going red at the
    integration layer; here we pin that None disables reuse rather than
    aliasing the previous cache entry.)
    """
    preparer = _make_preparer()

    _prepare(preparer, version=7)
    third = _prepare(preparer, version=None)

    assert preparer._cached_page_table_version == 7  # cache untouched by None
    assert third.pages_tables is not preparer._cached_pages_tables_dev


def test_start_async_prep_rejects_double_buffer_overrun():
    preparer = _make_preparer()
    input_ids_buf, position_ids_buf = _device_buffers(preparer)

    preparer.start_async_prep(
        input_ids_buf=input_ids_buf,
        position_ids_buf=position_ids_buf,
        **_prep_kwargs(page_table_version=1),
    )
    with pytest.raises(RuntimeError, match="in-flight"):
        preparer.start_async_prep(
            input_ids_buf=input_ids_buf,
            position_ids_buf=position_ids_buf,
            **_prep_kwargs(page_table_version=2),
        )

    # Draining the pending transfer re-arms the double buffer.
    assert preparer.get_async_prep_result() is not None
    preparer.start_async_prep(
        input_ids_buf=input_ids_buf,
        position_ids_buf=position_ids_buf,
        **_prep_kwargs(page_table_version=3),
    )
