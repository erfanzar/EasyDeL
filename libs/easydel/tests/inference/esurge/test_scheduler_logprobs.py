# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only scheduler probability transport, stop alignment and DP routing."""

import pytest
from easydel.inference.esurge.engine_types import FinishReason
from easydel.inference.esurge.outputs import LogprobsLists, ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.async_scheduler import AsyncScheduler
from easydel.inference.esurge.scheduler.dp_scheduler import DPScheduler
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

from .test_scheduler_spec_budget import _kv_config

pytestmark = pytest.mark.cpu_ok


def _scheduler(dp_size, async_scheduling, *, num_draft_tokens=0):
    kwargs = dict(
        kv_cache_config=_kv_config(),
        max_num_seqs=4,
        max_num_batched_tokens=16,
        max_model_len=64,
        num_pages=64,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        num_draft_tokens=num_draft_tokens,
    )
    if dp_size > 1:
        return DPScheduler(dp_size=dp_size, use_worker_processes=False, **kwargs)
    return (AsyncScheduler if async_scheduling else Scheduler)(**kwargs)


def _runner_output(rows, *, with_logprobs=True):
    # Reverse scheduler admission order and use unrelated sequence-buffer rows.
    req_ids = list(reversed(rows))
    flat_tokens = [token for req_id in req_ids for token in rows[req_id]]
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        req_id_to_row_index={req_id: 9 - i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[list(rows[req_id]) for req_id in req_ids],
        spec_token_ids=None,
        logprobs=(
            LogprobsLists(
                [[token, token + 100] for token in flat_tokens],
                [[-token / 10, -token / 10 - 1] for token in flat_tokens],
                [token % 3 for token in flat_tokens],
            )
            if with_logprobs
            else None
        ),
        prompt_logprobs_dict={},
    )


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("stop_kind", ["length", "eos", "stop_token"])
def test_scheduler_logprobs_follow_emitted_tokens_and_preserve_runner_payload(dp_size, async_scheduling, stop_kind):
    scheduler = _scheduler(dp_size, async_scheduling)
    retained_count = 1 if async_scheduling else 2
    stop_token = {"length": 22, "eos": 0, "stop_token": 55}[stop_kind]
    try:
        for req_id in ("stop", "keep", "unrequested", "prefill"):
            scheduler.add_request(
                EngineRequest(
                    request_id=req_id,
                    prompt_token_ids=[1, 2],
                    sampling_params=SamplingParams(
                        max_tokens=retained_count if req_id == "stop" and stop_kind == "length" else 8,
                        logprobs=None if req_id == "unrequested" else 0,
                        stop_token_ids=[55] if stop_kind == "stop_token" else [],
                    ),
                    eos_token_id=0,
                )
            )
        scheduled = scheduler.schedule()
        rows = {
            # Ordinary async scheduling reserves one sample placeholder per row;
            # speculative multi-token output is covered by the next regression.
            "stop": [stop_token] if async_scheduling else [21, stop_token, 23],
            "keep": [31] if async_scheduling else [31, 32],
            "unrequested": [41],
            "prefill": [],
        }
        runner_output = _runner_output(rows)
        outputs = scheduler.update_from_output(scheduled, runner_output)[0].outputs
        by_id = {output.request_id: output for output in outputs}
        assert set(by_id) == {"stop", "keep", "unrequested"}
        assert by_id["stop"].new_token_ids == rows["stop"][:retained_count]
        assert by_id["stop"].finish_reason == (FinishReason.LENGTH if stop_kind == "length" else FinishReason.STOP)
        # EOS intentionally has no custom stop_reason; custom stop tokens do.
        assert by_id["stop"].stop_reason == (55 if stop_kind == "stop_token" else None)
        assert by_id["keep"].new_token_ids == rows["keep"]
        assert by_id["keep"].finish_reason is None
        for req_id in ("stop", "keep"):
            tokens = by_id[req_id].new_token_ids
            probabilities = by_id[req_id].new_logprobs
            assert probabilities.logprob_token_ids == [[token, token + 100] for token in tokens]
            assert probabilities.logprobs == [[-token / 10, -token / 10 - 1] for token in tokens]
            assert probabilities.sampled_token_ranks == [token % 3 for token in tokens]
        assert by_id["unrequested"].new_logprobs is None
        assert runner_output.sampled_token_ids == [rows[req_id] for req_id in runner_output.req_ids]
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_speculative_logprobs_trim_after_acceptance_and_stop(async_scheduling):
    scheduler = _scheduler(1, async_scheduling, num_draft_tokens=3)
    try:
        request = EngineRequest(
            request_id="spec",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=3, logprobs=0),
            eos_token_id=0,
        )
        scheduler.add_request(request)
        first_output = _runner_output({"spec": [10]})
        first_output.spec_token_ids = [[21, 22, 23]]
        scheduler.update_from_output(scheduler.schedule(), first_output)
        scheduled = scheduler.schedule()
        assert scheduled.scheduled_spec_decode_tokens == {"spec": [21, 22, 23]}
        output = _runner_output({"spec": [21, 22, 24]})
        output.num_accepted_spec_tokens = {"spec": 2}
        output.spec_token_ids = [[]]
        result = scheduler.update_from_output(scheduled, output)[0].outputs[0]
        assert result.new_token_ids == [21, 22]
        assert result.new_logprobs.logprob_token_ids == [[21, 121], [22, 122]]
        assert result.new_logprobs.logprobs == [[-2.1, -3.1], [-2.2, -3.2]]
        assert result.finish_reason == FinishReason.LENGTH
        assert request.num_computed_tokens == 5
        assert output.sampled_token_ids == [[21, 22, 24]]
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("dp_size", [1, 2])
def test_scheduler_keeps_optional_logprobs_optional(dp_size):
    scheduler = _scheduler(dp_size, False)
    try:
        scheduler.add_request(
            EngineRequest(
                request_id="no-probabilities",
                prompt_token_ids=[1, 2],
                sampling_params=SamplingParams(max_tokens=8, logprobs=0),
                eos_token_id=0,
            )
        )
        output = scheduler.update_from_output(
            scheduler.schedule(), _runner_output({"no-probabilities": [11]}, with_logprobs=False)
        )[0].outputs[0]
        assert output.new_token_ids == [11]
        assert output.new_logprobs is None
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("dp_size", [1, 2])
def test_scheduler_rejects_misaligned_probabilities_before_mutating_requests(dp_size):
    scheduler = _scheduler(dp_size, False)
    try:
        request = EngineRequest(
            request_id="bad-probabilities",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=8, logprobs=0),
            eos_token_id=0,
        )
        scheduler.add_request(request)
        scheduled = scheduler.schedule()
        runner_output = _runner_output({request.request_id: [11, 12]})
        runner_output.logprobs = runner_output.logprobs.slice(0, 1)
        with pytest.raises(ValueError, match="one position per sampled token"):
            scheduler.update_from_output(scheduled, runner_output)
        assert request.num_output_tokens == 0
    finally:
        scheduler.shutdown()
