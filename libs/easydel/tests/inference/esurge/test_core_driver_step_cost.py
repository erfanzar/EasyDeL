# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bucket-aligned prefill chunks from the runner's step-time table, seen through the driver's ``SchedulerOutput``."""

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

STEP_MS = {4: 10.0, 8: 10.0, 16: 10.0, 32: 40.0, 64: 80.0}


def _driver(step_time_table: dict[int, float] | None) -> EngineCoreDriver:
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec, layer_names=None)],
        metadata=SimpleNamespace(num_pages=128, page_size=4, data_parallel_size=1),
        max_num_seqs=4,
        max_model_len=256,
        max_num_seq_buckets=[4],
        _has_recurrent_layers=False,
        step_time_table=step_time_table,
    )
    return EngineCoreDriver.from_runner(
        runner,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        async_scheduling=False,
        chunked_prefill_enabled=True,
    )


def _admit(driver: EngineCoreDriver, rid: str, prompt: list[int]) -> None:
    driver.add_requests(
        [
            AdmissionRecord(
                request_id=rid,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=32),
                eos_token_id=0,
            )
        ]
    )


def _commit_all(driver: EngineCoreDriver, so) -> None:
    ids = list(so.num_scheduled_tokens)
    driver.commit(
        so,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=[[7] for _ in ids],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            num_accepted_spec_tokens=None,
        ),
    )


@pytest.mark.parametrize("table", [STEP_MS, None])
def test_from_runner_trims_a_prefill_chunk_with_the_runner_step_time_table(table):
    driver = _driver(table)
    for i in range(3):
        _admit(driver, f"d{i}", [10 + i, 11, 12, 13])
    so = driver.schedule()
    assert so.num_scheduled_tokens == {"d0": 4, "d1": 4, "d2": 4}
    _commit_all(driver, so)
    # 3 decode rows + a 20-token prompt = 23 tokens (bucket 32, 40 ms). With the table the prompt runs 13 tokens so
    # the step fits bucket 16 (10 ms), and gives 7 tokens to the next step.
    _admit(driver, "p", list(range(100, 120)))
    so = driver.schedule()
    expected = 13 if table else 20
    assert so.num_scheduled_tokens == {"d0": 1, "d1": 1, "d2": 1, "p": expected}
    driver.check()


def test_set_step_cost_refuses_a_bad_table():
    driver = _driver(None)
    with pytest.raises(ValueError, match="step"):
        driver.set_step_cost({4: 0.0})
