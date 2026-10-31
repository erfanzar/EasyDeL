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

"""Scheduler regression: speculative drafts of the request that exhausts the token budget."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

# Only dtype metadata is borrowed from JAX; scheduling, token IDs, and page allocation are Python bookkeeping.
pytestmark = pytest.mark.cpu_ok


def _kv_config() -> CacheGroupsConfig:
    return CacheGroupsConfig(
        num_pages=64,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=4, num_kv_heads=1, head_size=4, dtype=jnp.float32, use_mla=False
                ),
                layer_names=None,
            )
        ],
    )


def test_spec_tokens_recorded_when_request_uses_up_token_budget() -> None:
    scheduler = Scheduler(
        kv_cache_config=_kv_config(),
        max_num_seqs=4,
        max_num_batched_tokens=4,
        max_model_len=64,
        num_pages=64,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_draft_tokens=3,
    )
    request = EngineRequest(
        request_id="r0",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=16),
        eos_token_id=0,
    )
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"r0": 3}

    scheduler.update_from_output(
        first,
        ModelRunnerOutput(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            sampled_token_ids=[[100]],
            spec_token_ids=[[7, 8, 9]],
            logprobs=None,
            prompt_logprobs_dict={},
            req_id_to_row_index={"r0": 0},
        ),
    )

    # 1 sampled token + 3 drafts == exactly the per-step budget of 4.
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"r0": 4}
    assert second.scheduled_spec_decode_tokens == {"r0": [7, 8, 9]}
    assert second.scheduled_cached_reqs.new_token_ids == [[100]]

    scheduler.update_from_output(
        second,
        ModelRunnerOutput(
            req_ids=["r0"],
            req_id_to_index={"r0": 0},
            sampled_token_ids=[[101]],
            spec_token_ids=[[]],
            num_accepted_spec_tokens={"r0": 0},
            logprobs=None,
            prompt_logprobs_dict={},
            req_id_to_row_index={"r0": 0},
        ),
    )
    # All three drafts were rejected, so only one new token remains computed.
    assert request.num_computed_tokens == 4


@pytest.mark.parametrize("data_parallel_size", [1, 2])
def test_from_runner_preserves_optional_scheduler_capabilities(data_parallel_size) -> None:
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        kv_cache_groups=_kv_config().kv_cache_groups,
        metadata=SimpleNamespace(num_pages=64, page_size=4, data_parallel_size=data_parallel_size),
        max_num_seqs=4,
        max_model_len=64,
        max_num_seq_buckets=[4],
    )
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=6,
        enable_prefix_caching=False,
        async_scheduling=False,
        chunked_prefill_enabled=True,
        token_safety_margin=999,
        include_finished_set=True,
    )
    request_ids = [f"r{i}" for i in range(2 * data_parallel_size)]
    try:
        for req_id in request_ids:
            scheduler.add_request(
                EngineRequest(
                    request_id=req_id,
                    prompt_token_ids=[1, 2, 3, 4],
                    sampling_params=SamplingParams(max_tokens=4),
                    eos_token_id=0,
                )
            )
        first = scheduler.schedule()
        assert first.num_scheduled_tokens == {
            req_id: 4 if i < data_parallel_size else 2 for i, req_id in enumerate(request_ids)
        }
        # The reserve suppresses new admission, not progress of already-admitted requests.
        # Complete the first runner step before asking for the next one.
        scheduler.update_from_output(
            first,
            ModelRunnerOutput(
                req_ids=request_ids,
                req_id_to_index={req_id: i for i, req_id in enumerate(request_ids)},
                sampled_token_ids=[[10] if i < data_parallel_size else [] for i in range(len(request_ids))],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        waiting_id = "reserve-waiting"
        scheduler.add_request(
            EngineRequest(
                request_id=waiting_id,
                prompt_token_ids=[1],
                sampling_params=SamplingParams(max_tokens=4),
                eos_token_id=0,
            )
        )
        second = scheduler.schedule()
        assert second.num_scheduled_tokens == {
            req_id: 1 if i < data_parallel_size else 2 for i, req_id in enumerate(request_ids)
        }
        assert waiting_id not in second.num_scheduled_tokens
        scheduler.update_from_output(
            second,
            ModelRunnerOutput(
                req_ids=request_ids,
                req_id_to_index={req_id: i for i, req_id in enumerate(request_ids)},
                sampled_token_ids=[[11] for _ in request_ids],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        request_ids.append(waiting_id)
        scheduler.finish_requests(request_ids, EngineRequestStatus.FINISHED_ABORTED)
        output = scheduler.update_from_output(
            scheduler.schedule(),
            ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        assert output[0].finished_requests == set(request_ids)
    finally:
        scheduler.shutdown()


def test_explicit_legacy_dp_scheduler_retains_shard_affinity() -> None:
    scheduler = Scheduler(
        kv_cache_config=_kv_config(),
        max_num_seqs=4,
        max_num_batched_tokens=8,
        max_model_len=64,
        num_pages=64,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
    )
    scheduler.data_parallel_size = 2
    for req_id in ("legacy0", "legacy1"):
        scheduler.add_request(
            EngineRequest(
                request_id=req_id,
                prompt_token_ids=[1, 2, 3, 4],
                sampling_params=SamplingParams(max_tokens=4),
                eos_token_id=0,
            )
        )
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"legacy0": 4, "legacy1": 4}
    assert output.req_id_to_dp_rank == {"legacy0": 0, "legacy1": 1}
    assert [req.dp_rank for req in output.scheduled_new_reqs] == [0, 1]
