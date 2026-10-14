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

"""Steady-decode incremental batch-prep: equality with the full path + invalidation."""

import dataclasses

import jax
import numpy as np
import pytest
from easydel.caching import RaggedPagesCacheConfig
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer
from jax.sharding import Mesh, NamedSharding, PartitionSpec

MAX_REQS = 8
MAX_TOKENS = 64
MAX_MODEL_LEN = 256
PAGE_SIZE = 64


@pytest.fixture()
def env():
    mesh = Mesh(np.array(jax.devices()).reshape(-1), axis_names=("d",))
    empty_sharding = NamedSharding(mesh, PartitionSpec())
    cache_cfg = RaggedPagesCacheConfig(
        num_hidden_layers=2,
        max_model_length=MAX_MODEL_LEN,
        num_kv_heads=2,
        k_headdim=64,
        v_headdim=64,
        page_size=PAGE_SIZE,
        num_pages=128,
        max_num_tokens=MAX_TOKENS,
    )

    def make_preparer():
        return BatchMetadataPreparer(
            metadata=cache_cfg,
            empty_sharding=empty_sharding,
            max_num_tokens=MAX_TOKENS,
            max_num_reqs=MAX_REQS,
            max_model_len=MAX_MODEL_LEN,
            min_input_pad=4,
        )

    rng = np.random.default_rng(0)
    inputs = dict(
        page_table=rng.integers(1, 100, size=(MAX_REQS, MAX_MODEL_LEN // PAGE_SIZE)).astype(np.int32),
        token_ids=rng.integers(0, 1000, size=(MAX_REQS, MAX_MODEL_LEN)).astype(np.int32),
        scheduled=np.ones((MAX_REQS,), np.int32),
        active=np.ones((MAX_REQS,), bool),
        temperature=np.zeros((MAX_REQS,), np.float32),
        ones_f=np.ones((MAX_REQS,), np.float32),
        zeros_f=np.zeros((MAX_REQS,), np.float32),
        topk=np.zeros((MAX_REQS,), np.int32),
        empty_sharding=empty_sharding,
    )
    return make_preparer, inputs


def _step(prep, inputs, computed, *, page_table_version=3, temperature=None, scheduled=None):
    ids_buf = jax.device_put(np.zeros((MAX_TOKENS,), np.int32), inputs["empty_sharding"])
    pos_buf = jax.device_put(np.zeros((MAX_TOKENS,), np.int32), inputs["empty_sharding"])
    return prep.prepare_batch_metadata(
        num_tokens_static=MAX_REQS,
        scheduled_full_cpu=inputs["scheduled"] if scheduled is None else scheduled,
        active_mask_full_cpu=inputs["active"],
        input_ids_buf=ids_buf,
        position_ids_buf=pos_buf,
        token_ids_cpu=inputs["token_ids"],
        num_computed_tokens_cpu=computed,
        temperature_cpu=inputs["temperature"] if temperature is None else temperature,
        top_p_cpu=inputs["ones_f"],
        top_k_cpu=inputs["topk"],
        min_p_cpu=inputs["zeros_f"],
        frequency_penalties_cpu=inputs["zeros_f"],
        presence_penalties_cpu=inputs["zeros_f"],
        repetition_penalties_cpu=inputs["ones_f"],
        page_table_cpu=inputs["page_table"],
        padded_num_reqs_in=MAX_REQS,
        page_table_version=page_table_version,
    )


def test_incremental_equals_full_path(env):
    make_preparer, inputs = env
    warm = make_preparer()
    base = np.full((MAX_REQS,), 10, np.int32)
    _step(warm, inputs, base)

    for s in (1, 2, 3):
        computed = base + s
        md_inc, *_ = _step(warm, inputs, computed)
        assert warm.last_prep_stats.get("prep_incremental") == 1.0

        fresh = make_preparer()
        md_full, *_ = _step(fresh, inputs, computed)

        for f in dataclasses.fields(md_inc):
            v_inc, v_full = getattr(md_inc, f.name), getattr(md_full, f.name)
            if v_inc is None and v_full is None:
                continue
            assert (v_inc is None) == (v_full is None), f.name
            if isinstance(v_inc, tuple):
                continue
            assert np.array_equal(np.asarray(v_inc), np.asarray(v_full)), f.name


def test_invalidation_on_page_version_sampling_and_schedule(env):
    make_preparer, inputs = env
    prep = make_preparer()
    base = np.full((MAX_REQS,), 10, np.int32)
    _step(prep, inputs, base)

    _step(prep, inputs, base + 1, page_table_version=4)
    assert prep.last_prep_stats.get("prep_incremental") != 1.0

    temp = inputs["temperature"].copy()
    temp[2] = 0.8
    _step(prep, inputs, base + 2, page_table_version=4, temperature=temp)
    assert prep.last_prep_stats.get("prep_incremental") != 1.0

    sched = inputs["scheduled"].copy()
    sched[0] = 0  # budget-skipped row
    sched[1] = 2  # a prefill chunk appears -> not a pure-decode window (total stays in bucket)
    _step(prep, inputs, base + 3, page_table_version=4, scheduled=sched)
    assert prep.last_prep_stats.get("prep_incremental") != 1.0
    # a non-decode window must also refuse to PRIME (next pure-decode step is a miss, then hits)
    _step(prep, inputs, base + 4, page_table_version=4)
    assert prep.last_prep_stats.get("prep_incremental") != 1.0
    _step(prep, inputs, base + 5, page_table_version=4)
    assert prep.last_prep_stats.get("prep_incremental") == 1.0


def test_hits_resume_after_reprime(env):
    make_preparer, inputs = env
    prep = make_preparer()
    base = np.full((MAX_REQS,), 10, np.int32)
    _step(prep, inputs, base)
    _step(prep, inputs, base + 1)
    assert prep.incremental_prep_hits == 1
    _step(prep, inputs, base + 2, page_table_version=9)  # miss + reprime
    _step(prep, inputs, base + 3, page_table_version=9)  # hit again
    assert prep.incremental_prep_hits == 2
    assert prep.incremental_prep_misses >= 1
