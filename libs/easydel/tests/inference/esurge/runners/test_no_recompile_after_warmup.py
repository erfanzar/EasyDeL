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

"""Characterization: no XLA lowering happens after warmup for a fixed workload.

The eSurge execution stack precompiles (num_tokens, padded_num_reqs) bucket
variants and disables runtime compilation for them, but several auxiliary jits
(greedy argmax fast path, sampler-output scatter, penalty-count rebuild) are
plain ``jax.jit`` calls that would recompile *silently* — no error, only
latency — if a refactor perturbed their static arguments or input shapes.

This test drives a full greedy generation twice through the real runner +
scheduler on the calling thread (JAX's lowering counters are thread-local, so
the engine's background scheduler thread cannot be used) and asserts the
second, shape-identical generation triggers zero lowerings and leaves the
executor bucket caches untouched.
"""

from __future__ import annotations

import jax
import pytest
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp
from jax._src import test_util as jtu

try:
    from tests.inference.esurge.runtime_pass._tiny_runtime_common import load_test_tokenizer
except ImportError:  # pragma: no cover - direct invocation
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "runtime_pass"))
    from _tiny_runtime_common import load_test_tokenizer  # pyright: ignore[reportImplicitRelativeImport]

MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 2
PAGE_SIZE = 16
PROMPT_LEN = 12
OUTPUT_LEN = 8


def _build_tiny_llama(tokenizer):
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sharding_axis_dims=(1, 1, 1, 1, jax.device_count(), 1),
    )
    with config.mesh:
        return ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


@pytest.fixture(scope="module")
def warm_runner():
    """A compiled sync-mode runner, warmed by one full greedy generation."""
    from easydel.inference.esurge.esurge_engine import eSurge

    tokenizer = load_test_tokenizer()
    model = _build_tiny_llama(tokenizer)
    engine = eSurge(
        model=model,
        processor=tokenizer,
        runtime={
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "async_scheduling": False,
            "overlap_execution": False,
        },
        cache={
            "page_size": PAGE_SIZE,
            "hbm_utilization": 0.05,
            "max_cache_tokens": 2048,
            "enable_prefix_caching": False,
        },
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )
    runner = engine.runner
    if runner.executor_manager.kv_pages is None:
        runner.initialize_kv_cache()
    _drive_generation(runner)  # warm every lazily-compiled auxiliary jit
    yield runner
    engine.terminate()


def _drive_generation(runner) -> int:
    """One synchronous greedy generation on the calling thread; returns tokens."""
    runner.reset_state()
    scheduler = Scheduler.from_runner(
        runner,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_draft_tokens=0,
    )
    sampling = SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0, ignore_eos=True)
    request = EngineRequest(
        request_id="warmup-0",
        prompt_token_ids=list(range(1, PROMPT_LEN + 1)),
        sampling_params=sampling,
        eos_token_id=None,
    )
    scheduler.add_request(request)

    for _ in range(PROMPT_LEN + OUTPUT_LEN + 8):
        scheduler_output = scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            if request.is_finished():
                break
            continue
        model_output = runner.execute_model(scheduler_output)
        scheduler.update_from_output(scheduler_output, model_output)
        if request.is_finished():
            break
    assert request.is_finished(), "generation did not finish within the step budget"
    assert len(request.output_token_ids) == OUTPUT_LEN
    return len(request.output_token_ids)


def test_second_generation_triggers_zero_lowerings(warm_runner):
    model_keys = sorted(map(str, warm_runner.executor_manager._model_executor.cache_keys()))
    sampler_keys = sorted(map(str, warm_runner.executor_manager._sampler_executor.cache_keys()))
    assert model_keys, "expected precompiled model-step variants after engine construction"

    with jtu.count_jit_and_pmap_lowerings() as lowering_count:
        _drive_generation(warm_runner)

    assert lowering_count() == 0, (
        f"steady-state generation lowered {lowering_count()} new computation(s); "
        "an auxiliary jit or bucket key is no longer warm after warmup"
    )
    assert sorted(map(str, warm_runner.executor_manager._model_executor.cache_keys())) == model_keys
    assert sorted(map(str, warm_runner.executor_manager._sampler_executor.cache_keys())) == sampler_keys


def test_second_generation_is_greedy_deterministic(warm_runner):
    """Same prompt, same sync runner -> identical tokens across generations.

    Cheap rider on the same fixture: locks device-side determinism of the
    fused step + sampler for a fixed batch shape (the property the multi-host
    replay design and the TPU bench bucket comparison both lean on).
    """
    runner = warm_runner

    def _generate_tokens() -> list[int]:
        runner.reset_state()
        scheduler = Scheduler.from_runner(
            runner,
            enable_prefix_caching=False,
            async_scheduling=False,
            num_draft_tokens=0,
        )
        request = EngineRequest(
            request_id="det-0",
            prompt_token_ids=list(range(1, PROMPT_LEN + 1)),
            sampling_params=SamplingParams(max_tokens=OUTPUT_LEN, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        scheduler.add_request(request)
        for _ in range(PROMPT_LEN + OUTPUT_LEN + 8):
            scheduler_output = scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens == 0:
                if request.is_finished():
                    break
                continue
            model_output = runner.execute_model(scheduler_output)
            scheduler.update_from_output(scheduler_output, model_output)
            if request.is_finished():
                break
        assert request.is_finished()
        return [int(t) for t in request.output_token_ids]

    assert _generate_tokens() == _generate_tokens()
