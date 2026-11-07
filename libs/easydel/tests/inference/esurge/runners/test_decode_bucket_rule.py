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

"""Every decode step runs the decode-layout program of its request bucket, for any ``max_num_seqs``."""

import jax
import pytest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.window_planner import WindowPlanner
from easydel.inference.sampling_params import SamplingParams

try:
    from .._common import make_core_for_runner, make_tiny_qwen35, served_request
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, make_tiny_qwen35, served_request

PROMPT_LEN = 12


@pytest.fixture(scope="module")
def model():
    return make_tiny_qwen35(mtp_layers=0, precision=jax.lax.Precision.HIGHEST)


@pytest.mark.parametrize(
    ("token_paddings", "request_buckets"),
    [
        pytest.param([32, 64, 128, 256], [32], id="seq-buckets-32"),
        pytest.param([16, 32, 64, 128], [16, 32], id="default-ladder-32"),
        pytest.param([16, 32, 64], [1, 2, 4, 8, 16], id="request-buckets-below-the-floor"),
    ],
)
def test_power_of_two_ladders_are_unchanged(token_paddings, request_buckets):
    assert WindowPlanner.add_request_token_buckets(token_paddings, request_buckets) == token_paddings


@pytest.mark.parametrize("max_num_seqs", [6, 14, 16])
def test_decode_steps_run_the_request_bucket_program(model, max_num_seqs):
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=16,
        max_cache_tokens=4096,
        max_model_len=128,
        max_num_batched_tokens=64,
        min_input_pad=16,
        max_num_seqs=max_num_seqs,
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
    )
    runner.compile(max_num_batched_tokens=64)
    (bucket,) = runner.max_num_seq_buckets
    assert bucket == max_num_seqs

    calls: list[tuple[int, int, bool]] = []
    execute = runner.executor_manager.execute

    def recording_execute(*args, **kwargs):
        reqs = int(kwargs["padded_num_reqs"])
        scheduled = kwargs["scheduled_full_cpu"][:reqs]
        active = scheduled[kwargs["active_mask_full_cpu"][:reqs] & (scheduled > 0)]
        calls.append((int(kwargs["num_tokens"]), reqs, bool(active.size) and bool((active == 1).all())))
        return execute(*args, **kwargs)

    runner.executor_manager.execute = recording_execute
    core = make_core_for_runner(runner, max_num_batched_tokens=64, enable_prefix_caching=False, async_scheduling=False)
    # Different output lengths shrink the decode batch step by step, through every count down to one.
    requests = [
        served_request(
            request_id=f"r{i}",
            prompt_token_ids=[((5 * i + 7 * j) % 200) + 2 for j in range(PROMPT_LEN)],
            sampling_params=SamplingParams(max_tokens=2 + i, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        for i in range(max_num_seqs)
    ]
    core.run(runner, requests, max_steps=200)
    assert [len(r.output_token_ids) for r in requests] == [2 + i for i in range(max_num_seqs)]

    decode = [(tokens, reqs) for tokens, reqs, decode_only in calls if decode_only]
    assert len(decode) >= max_num_seqs // 2
    # (tokens, requests) equal is the bucket the decode-layout program is traced for.
    assert set(decode) == {(bucket, bucket)}
