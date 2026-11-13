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

"""The engine core's load-aware verification budget, seen through the driver's ``SchedulerOutput``."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.engine.core_driver import EngineCoreDriver
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import AdmissionRecord
from easydel.inference.sampling_params import SamplingParams

# Host bookkeeping only: the core is numpy-in/numpy-out.
pytestmark = pytest.mark.cpu_ok

K = 3


def _driver(num_draft_tokens: int = K, async_scheduling: bool = False) -> EngineCoreDriver:
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    return EngineCoreDriver(
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec, layer_names=None)],
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=64,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        num_draft_tokens=num_draft_tokens,
    )


def _admit(driver: EngineCoreDriver, n: int = 4) -> list[str]:
    ids = [f"r{i}" for i in range(n)]
    driver.add_requests(
        [
            AdmissionRecord(
                request_id=rid,
                prompt_token_ids=[1, 2, 3, 4],
                sampling_params=SamplingParams(max_tokens=32),
                eos_token_id=0,
            )
            for rid in ids
        ]
    )
    return ids


def _commit(driver: EngineCoreDriver, so, accepted: dict[str, int], new_drafts: int = K) -> None:
    ids = list(so.num_scheduled_tokens)
    driver.commit(
        so,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[7] * (accepted.get(rid, 0) + 1) for rid in ids],
            spec_token_ids=[[9] * new_drafts for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            num_accepted_spec_tokens=accepted or None,
        ),
    )


def _draft_counts(so) -> dict[str, int]:
    return {rid: len(so.scheduled_spec_decode_tokens.get(rid, [])) for rid in so.num_scheduled_tokens}


def test_without_a_budget_the_runner_drafts_the_configured_count():
    driver = _driver()
    ids = _admit(driver)
    so = driver.schedule()
    assert so.num_draft_tokens_next == K
    _commit(driver, so, {})
    so = driver.schedule()
    assert _draft_counts(so) == dict.fromkeys(ids, K)
    assert so.num_draft_tokens_next == K
    _commit(driver, so, dict.fromkeys(ids, K))
    off = _driver(num_draft_tokens=0)
    _admit(off)
    assert off.schedule().num_draft_tokens_next == 0


def test_budget_cuts_drafts_learns_acceptance_and_can_verify_nothing():
    driver = _driver()
    ids = _admit(driver)
    _commit(driver, driver.schedule(), {})
    # Up to 8 tokens cost the same as 4; 16 cost four times as much. With prior 0.8 each request verifies one draft.
    driver.set_verify_budget({4: 10.0, 8: 10.0, 16: 40.0}, prior=0.8, alpha=0.5, correction=0.0)
    so = driver.schedule()
    assert _draft_counts(so) == dict.fromkeys(ids, 1)
    assert so.num_scheduled_tokens == dict.fromkeys(ids, 2)
    assert so.num_draft_tokens_next == 1
    # r0 accepts, the others reject: r0 gets three of the four free rows, r1 the fourth.
    _commit(driver, so, {"r0": 1, "r1": 0, "r2": 0, "r3": 0})
    so = driver.schedule()
    assert _draft_counts(so) == {"r0": 3, "r1": 1, "r2": 0, "r3": 0}
    assert so.total_num_scheduled_tokens == 8
    _commit(driver, so, {"r0": 3, "r1": 0})
    driver.check()

    # Verification that costs more than it returns: no drafts this step, and the drafter is skipped.
    driver.set_verify_budget({4: 10.0, 8: 100.0}, prior=0.8, alpha=0.5, correction=0.0)
    so = driver.schedule()
    assert so.scheduled_spec_decode_tokens == {}
    assert so.num_scheduled_tokens == dict.fromkeys(ids, 1)
    assert so.num_draft_tokens_next == 0
    _commit(driver, so, {}, new_drafts=0)
    driver.check()


def test_from_runner_applies_the_runner_step_time_table():
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec, layer_names=None)],
        metadata=SimpleNamespace(num_pages=64, page_size=4, data_parallel_size=1),
        max_num_seqs=4,
        max_model_len=64,
        max_num_seq_buckets=[4],
        _has_recurrent_layers=False,
        step_time_table={4: 10.0, 8: 100.0},
    )
    driver = EngineCoreDriver.from_runner(
        runner, max_num_batched_tokens=32, enable_prefix_caching=False, async_scheduling=False, num_draft_tokens=K
    )
    ids = _admit(driver)
    _commit(driver, driver.schedule(), {})
    so = driver.schedule()
    assert so.scheduled_spec_decode_tokens == {} and so.num_draft_tokens_next == 0
    assert so.num_scheduled_tokens == dict.fromkeys(ids, 1)


def test_budget_refuses_async_scheduling_and_bad_tables():
    with pytest.raises(ValueError, match="synchronous"):
        _driver(async_scheduling=True).set_verify_budget({4: 1.0})
    with pytest.raises(ValueError, match="num_spec_tokens"):
        _driver(num_draft_tokens=0).set_verify_budget({4: 1.0})
    with pytest.raises(ValueError, match="positive and finite"):
        _driver().set_verify_budget({4: 0.0})
