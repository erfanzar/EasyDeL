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

"""Rank-major SPMD DP: warmup fits every rank's token budget and pure-decode steps run the decode layout."""

from __future__ import annotations

import jax
import pytest
import spectrax as spx
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

try:
    from .._common import make_core_for_runner, served_request
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, served_request

try:
    from tests.inference.esurge.runtime_pass._tiny_runtime_common import load_test_tokenizer
except ImportError:  # pragma: no cover - direct invocation
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime_pass"))
    from _tiny_runtime_common import load_test_tokenizer  # pyright: ignore[reportImplicitRelativeImport]

DP = 4
MAX_MODEL_LEN = 256
MAX_NUM_SEQS = 128
PROMPT_LEN = 4
OUTPUT_LEN = 4


def _build_model(tokenizer):
    import easydel as ed

    config = ed.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=MAX_MODEL_LEN,
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        attn_mechanism="ragged_page_attention_v3",
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sharding_axis_dims=(1, DP, 1, 1, 1, 1),
    )
    with config.mesh:
        return ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


def _serve(min_token_pad: int):
    """Build a DP=4 engine (warmup runs in the constructor), then greedy-decode 128 requests."""
    from easydel.inference.esurge.esurge_engine import eSurge

    tokenizer = load_test_tokenizer()
    engine = eSurge(
        model=_build_model(tokenizer),
        processor=tokenizer,
        runtime={
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "min_input_pad": 32,
            "min_token_pad": min_token_pad,
            "max_num_batched_tokens": 128,
            "async_scheduling": False,
            "overlap_execution": False,
        },
        cache={
            "page_size": 16,
            "hbm_utilization": 0.05,
            "max_cache_tokens": 16384,
            "enable_prefix_caching": False,
            "data_parallelism_axis": "dp",
        },
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )
    try:
        runner = engine.runner
        if runner.executor_manager.kv_pages is None:
            runner.initialize_kv_cache()
        manager = runner.executor_manager
        compiled_keys = sorted(k for k in manager._model_executor._cache if str(k[2]).startswith("model_step"))
        steps: list[tuple[bool, tuple]] = []
        execute_model = manager.execute_model

        def recording_execute_model(*args, **kwargs):
            out = execute_model(*args, **kwargs)
            steps.append((bool(kwargs.get("decode_only", False)), manager.last_model_step_key))
            return out

        manager.execute_model = recording_execute_model
        runner.reset_state()
        core = make_core_for_runner(runner, enable_prefix_caching=False, async_scheduling=False, num_draft_tokens=0)
        sampling = SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0, ignore_eos=True)
        requests = [
            served_request(
                request_id=f"r{i}",
                prompt_token_ids=[(7 * i + j) % 200 + 1 for j in range(PROMPT_LEN)],
                sampling_params=sampling,
                eos_token_id=None,
            )
            for i in range(MAX_NUM_SEQS)
        ]
        core.run(runner, requests, max_steps=64)
        outputs = [list(r.output_token_ids) for r in requests]
        return compiled_keys, steps, outputs
    finally:
        engine.terminate()


@pytest.fixture(scope="module")
def served():
    return {pad: _serve(pad) for pad in (32, 128)}


@pytest.mark.parametrize("min_token_pad", [32, 128])
def test_warmup_fits_rank_budget_and_decode_runs_decode_layout(served, min_token_pad):
    compiled_keys, steps, outputs = served[min_token_pad]
    assert compiled_keys, "warmup compiled no model-step program"
    assert all(len(o) == OUTPUT_LEN for o in outputs)
    decode_steps = [key for decode_only, key in steps if decode_only]
    assert decode_steps, "no pure-decode step ran"
    for key in decode_steps:
        # Pure decode always runs a decode-layout program: an equal-bucket pair or the separate decode variant.
        assert key[0] == key[1] or key[2] == "model_step_decode", key
    for decode_only, key in steps:
        if not decode_only:
            assert key[2] == "model_step", key
    # 32 decode rows per DP worker: the full 128-request decode step ran.
    assert any(key[1] == MAX_NUM_SEQS for key in decode_steps)


def test_decode_layout_variants_bounded(served):
    keys_32, _, _ = served[32]
    keys_128, _, _ = served[128]
    general_128 = {(k[0], k[1]) for k in keys_128 if k[2] == "model_step"}
    decode_128 = {(k[0], k[1]) for k in keys_128 if k[2] == "model_step_decode"}
    # One decode twin at most per request bucket, only where the token floor exceeds the bucket.
    assert decode_128 <= general_128
    assert len(decode_128) <= len({r for _, r in general_128})
    assert all(t > r for t, r in decode_128)
    assert not [k for k in keys_32 if k[2] == "model_step_decode"]


def test_greedy_outputs_match_across_programs(served):
    # min_token_pad=32 decodes on equal-bucket programs, 128 on the separate decode variants: same tokens.
    assert served[32][2] == served[128][2]
