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

"""NullSpeculation keeps the non-speculative runner path inert (phase 10.2).

A runner constructed without a drafter must carry the inert strategy
(``runner.spec`` is :class:`NullSpeculation`), report zero speculative
tokens/counters, and produce deterministic greedy generations across two
back-to-back runs on the same runner — the same determinism property
``test_no_recompile_after_warmup.py::test_second_generation_is_greedy_deterministic``
locks for the plain decode path.

The model is a tiny pure paged-attention Llama (as in the referenced
determinism test): hybrid recurrent models carry per-slot recurrent state
across ``reset_state()`` (a pre-existing, speculation-unrelated property),
so same-runner cross-run determinism only holds for pure paged attention.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax.numpy as jnp
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.spec import NullSpeculation
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.modules.llama import LlamaConfig, LlamaForCausalLM

PROMPT_TOKEN_IDS = [3, 1, 4, 1, 5, 9]
OUTPUT_LEN = 8


def _make_tiny_model() -> LlamaForCausalLM:
    """Tiny pure paged-attention Llama, as in test_no_recompile_after_warmup."""
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
    )
    return LlamaForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _generate_tokens(runner: eSurgeRunner) -> list[int]:
    """Run one full greedy generation through scheduler + runner; return tokens."""
    runner.reset_state()
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_speculative_tokens=runner.num_speculative_tokens,
    )
    request = EngineRequest(
        request_id="null-spec-0",
        prompt_token_ids=list(PROMPT_TOKEN_IDS),
        sampling_params=SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0, ignore_eos=True),
        eos_token_id=None,
    )
    scheduler.add_request(request)

    for _ in range(64):
        scheduler_output = scheduler.schedule()
        output = runner.execute_model(scheduler_output)
        scheduler.update_from_output(scheduler_output, output)
        if request.is_finished():
            break
    else:
        raise AssertionError("generation did not finish")

    return [int(t) for t in request.output_token_ids]


def test_null_speculation_inert():
    model = _make_tiny_model()
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=16,
        max_cache_tokens=1024,
        max_model_len=64,
        max_num_batched_tokens=16,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=1,
        max_num_seq_buckets=[1],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        drafter=None,
    )
    runner.compile(max_num_batched_tokens=16)

    assert isinstance(runner.spec, NullSpeculation), f"expected NullSpeculation, got {type(runner.spec)!r}"
    assert runner.num_speculative_tokens == 0
    assert runner.spec.num_speculative_tokens == 0

    first = _generate_tokens(runner)
    second = _generate_tokens(runner)

    assert first, "greedy generation emitted no tokens"
    assert len(first) == OUTPUT_LEN
    assert first == second, f"greedy output changed across runs\n  first ={first}\n  second={second}"

    # The inert strategy must never have engaged the speculative machinery.
    assert runner.spec_decode_num_drafts_generated == 0
    assert runner.spec_decode_num_drafts_accepted == 0
    assert runner.spec_decode_num_verify_steps == 0

    runner.shutdown()


if __name__ == "__main__":
    test_null_speculation_inert()
