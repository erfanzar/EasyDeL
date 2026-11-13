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

"""Preempted requests resume to the same greedy tokens (computation test).

A tiny GLM-5-Next hybrid (3 KDA + 1 DSA layer) serves the same greedy requests through the full eSurge engine on a
page pool large enough that nothing is preempted, and on one so small that running requests are preempted. A resumed
request recomputes its prompt and earlier outputs in prefill chunks that start past the prompt; none of those chunks
may sample a token.
"""

from __future__ import annotations

import numpy as np
import pytest
from easydel.inference.sampling_params import SamplingParams

try:
    from .runtime_pass._tiny_runtime_common import load_test_tokenizer
    from .test_glm5_next_mtp_spec import VOCAB, _make_model
except ImportError:  # standalone `python test_x.py`
    from runtime_pass._tiny_runtime_common import load_test_tokenizer  # pyright: ignore[reportImplicitRelativeImport]
    from test_glm5_next_mtp_spec import VOCAB, _make_model  # pyright: ignore[reportImplicitRelativeImport]

NUM_REQUESTS = 32
PROMPT_LEN = 64
OUTPUT_LEN = 320
SMALL_POOL_TOKENS = 4096  # about a third of what 32 full requests need: running requests are preempted
LARGE_POOL_TOKENS = 65536


def _serve(model, tokenizer, prompts, *, async_scheduling: bool, pool_tokens: int):
    """Greedy-decode ``prompts`` through a fresh engine; return the output tokens and the number of preemptions."""
    from easydel.inference.esurge.esurge_engine import eSurge

    engine = eSurge(
        model=model,
        processor=tokenizer,
        runtime={
            "max_model_len": 512,
            "min_input_pad": 1,
            "max_num_seqs": NUM_REQUESTS,
            "max_num_seq_buckets": [NUM_REQUESTS],
            "max_num_batched_tokens": 128,
            "async_scheduling": async_scheduling,
            "overlap_execution": async_scheduling,
        },
        cache={
            "page_size": 16,
            "hbm_utilization": 0.02,
            "max_cache_tokens": pool_tokens,
            "enable_prefix_caching": True,
        },
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )
    engine.initiate()
    preempted = 0
    schedule = engine.core.schedule

    def counting_schedule():
        nonlocal preempted
        scheduler_output = schedule()
        preempted += len(scheduler_output.preempted_req_ids)
        return scheduler_output

    engine.core.schedule = counting_schedule
    try:
        outputs = engine.generate(
            [f"p{i}" for i in range(len(prompts))],
            SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0, ignore_eos=True),
            use_tqdm=False,
            prompt_token_ids=prompts,
        )
    finally:
        engine.terminate()
    return [list(output.outputs[0].token_ids) for output in outputs], preempted


@pytest.fixture(scope="module")
def served():
    tokenizer = load_test_tokenizer()
    model = _make_model()
    rng = np.random.default_rng(0)
    prompts = [[int(t) for t in rng.integers(1, VOCAB - 1, size=PROMPT_LEN)] for _ in range(NUM_REQUESTS)]
    reference, preempted = _serve(model, tokenizer, prompts, async_scheduling=False, pool_tokens=LARGE_POOL_TOKENS)
    assert preempted == 0, "the reference pool preempted requests"
    assert all(len(tokens) == OUTPUT_LEN for tokens in reference)
    return model, tokenizer, prompts, reference


@pytest.mark.parametrize("async_scheduling", [False, True], ids=["sync", "async"])
def test_preempted_requests_resume_to_the_reference_tokens(served, async_scheduling):
    model, tokenizer, prompts, reference = served
    outputs, preempted = _serve(
        model, tokenizer, prompts, async_scheduling=async_scheduling, pool_tokens=SMALL_POOL_TOKENS
    )
    assert preempted > 0, "the small pool preempted nothing"
    assert outputs == reference
