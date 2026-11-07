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

"""Async speculative steps through the driver: a verify window scheduled while the previous one is in flight."""

from __future__ import annotations

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.engine.core_driver import EngineCoreDriver
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import AdmissionRecord
from easydel.inference.sampling_params import SamplingParams

# Host bookkeeping only: the core is numpy-in/numpy-out.
pytestmark = pytest.mark.cpu_ok

K = 2


def _driver() -> EngineCoreDriver:
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    driver = EngineCoreDriver(
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec, layer_names=None)],
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=64,
        num_pages=64,
        enable_prefix_caching=False,
        async_scheduling=True,
        num_draft_tokens=K,
    )
    driver.add_requests(
        [
            AdmissionRecord(
                request_id=rid,
                prompt_token_ids=[1, 2, 3, 4],
                sampling_params=SamplingParams(max_tokens=32),
                eos_token_id=0,
            )
            for rid in ("r0", "r1")
        ]
    )
    return driver


def _commit(driver, so, sampled: dict[str, list[int]], accepted: dict[str, int] | None = None) -> dict[str, list[int]]:
    ids = list(so.num_scheduled_tokens)
    out = driver.commit(
        so,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[sampled[rid] for rid in ids],
            spec_token_ids=[[9] * K for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            num_accepted_spec_tokens=accepted,
        ),
    )
    return {o.request_id: o.new_token_ids for o in out[0].outputs} if out else {}


def test_a_verify_window_overlaps_the_previous_one_and_commits_correct():
    driver = _driver()
    prefill = driver.schedule()
    assert not prefill.async_safe and prefill.spec_slots == {"r0": 1, "r1": 1}
    assert _commit(driver, prefill, {"r0": [7], "r1": [8]}) == {"r0": [7], "r1": [8]}

    # Step N: real drafts, a full window of 1 + k, safe to overlap.
    step_n = driver.schedule()
    assert step_n.async_safe
    assert step_n.num_scheduled_tokens == {"r0": 1 + K, "r1": 1 + K}
    assert step_n.scheduled_spec_decode_tokens == {"r0": [9, 9], "r1": [9, 9]}
    assert step_n.spec_slots == {"r0": 1 + K, "r1": 1 + K}

    # Step N+1, scheduled before N's commit: optimistic starts, placeholder drafts.
    step_n1 = driver.schedule()
    assert step_n1.async_safe
    assert step_n1.num_scheduled_tokens == {"r0": 1 + K, "r1": 1 + K}
    assert step_n1.scheduled_spec_decode_tokens == {"r0": [-1, -1], "r1": [-1, -1]}
    assert step_n1.scheduled_cached_reqs.num_computed_tokens == [7, 7]

    # Commit N: r0 accepts both drafts, r1 rejects both; the in-flight N+1 entries are left alone.
    assert _commit(driver, step_n, {"r0": [9, 9, 11], "r1": [12]}, {"r0": 2, "r1": 0}) == {
        "r0": [9, 9, 11],
        "r1": [12],
    }
    # Commit N+1 rolls back its own rejected drafts against its 1 + k.
    assert _commit(driver, step_n1, {"r0": [21, 22], "r1": [31]}, {"r0": 1, "r1": 0}) == {
        "r0": [21, 22],
        "r1": [31],
    }
    driver.check()
    # (status, computed, num_tokens, drafts): every request is one token short of computed, with real drafts again.
    assert driver.request_state("r0") == (1, 9, 10, [9, 9])
    assert driver.request_state("r1") == (1, 6, 7, [9, 9])
