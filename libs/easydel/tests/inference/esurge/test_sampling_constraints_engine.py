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

"""Per-request logit bias through the full eSurge engine (computation test).

The OpenAI chat path adds a logit bias to every request without tools, so a biased request must run on the default
engine (AOT-compiled model step, async scheduling) and the bias must reach both the sampler and greedy decoding.
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

NUM_REQUESTS = 8
PROMPT_LEN = 16
OUTPUT_LEN = 12
FORCED_TOKEN = 7


@pytest.fixture(scope="module")
def served():
    from easydel.inference.esurge.esurge_engine import eSurge

    engine = eSurge(
        model=_make_model(),
        processor=load_test_tokenizer(),
        runtime={
            "max_model_len": 256,
            "min_input_pad": 1,
            "max_num_seqs": NUM_REQUESTS,
            "max_num_seq_buckets": [NUM_REQUESTS],
            "max_num_batched_tokens": 128,
        },
        cache={"page_size": 16, "hbm_utilization": 0.02, "max_cache_tokens": 4096, "enable_prefix_caching": False},
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )
    engine.initiate()
    rng = np.random.default_rng(0)
    prompts = [[int(t) for t in rng.integers(1, VOCAB - 1, size=PROMPT_LEN)] for _ in range(NUM_REQUESTS)]

    def generate(**params):
        outputs = engine.generate(
            [f"p{i}" for i in range(len(prompts))],
            SamplingParams(max_tokens=OUTPUT_LEN, ignore_eos=True, **params),
            use_tqdm=False,
            prompt_token_ids=prompts,
        )
        return [list(output.outputs[0].token_ids) for output in outputs]

    try:
        yield generate
    finally:
        engine.terminate()


@pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["greedy", "sampled"])
def test_positive_bias_forces_its_token(served, temperature):
    outputs = served(temperature=temperature, logit_bias={FORCED_TOKEN: 1.0e4})
    assert outputs == [[FORCED_TOKEN] * OUTPUT_LEN] * NUM_REQUESTS


def test_negative_bias_removes_the_greedy_tokens_and_leaves_later_requests_unchanged(served):
    reference = served(temperature=0.0)
    banned = sorted({t for tokens in reference for t in tokens})
    assert len(banned) < VOCAB
    biased = served(temperature=0.0, logit_bias={t: -1.0e4 for t in banned})
    assert all(len(tokens) == OUTPUT_LEN for tokens in biased)
    assert not {t for tokens in biased for t in tokens} & set(banned)
    assert served(temperature=0.0) == reference
