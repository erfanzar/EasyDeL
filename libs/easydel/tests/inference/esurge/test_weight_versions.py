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

"""eSurge weight-version provenance: versions, per-token segments, the stale-version fence and prefix-cache safety.

Tests marked ``cpu_ok`` drive the host-side output pipeline and rollout array helpers. The engine tests build a tiny
engine and are computation tests (the suite's conftest refuses them on a CPU backend).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.engine.loop import EngineLoop
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs, FinishReason
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.sampling_params import SamplingParams
from easydel.trainers._shared import esurge_generation
from easydel.workers.esurge.pipeline import Detokenizer
from jax import numpy as jnp

try:
    from tests.inference.esurge.runtime_pass._tiny_runtime_common import load_test_tokenizer
except ImportError:  # pragma: no cover - direct invocation
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "runtime_pass"))
    from _tiny_runtime_common import load_test_tokenizer  # pyright: ignore[reportImplicitRelativeImport]

MAX_MODEL_LEN = 512


# ----------------------------------------------------------------------------------------------- host-only


class _CharTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)

    def get_vocab(self):
        return {chr(v): v for v in range(128)}


def _pipeline_with_request(n_samples: int = 1):
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="r",
        prompt="p",
        prompt_token_ids=[1],
        outputs=[CompletionOutput(index=i, text="", token_ids=[]) for i in range(n_samples)],
    )
    registry.register("r", RequestRecord(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=64)), output)
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=Detokenizer(_CharTokenizer(), context_window=0),
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=1000.0,
        on_stop_strings=lambda matches: None,
        on_activity=lambda: None,
        on_fatal=lambda *args: pytest.fail("Output pipeline unexpectedly failed"),
    )
    return pipeline, output


def _bundle(tokens, *, finished=False, t=1.0):
    kwargs = {"finish_reason": FinishReason.LENGTH} if finished else {}
    return {0: EngineCoreOutputs(outputs=[EngineCoreOutput("r", tokens, **kwargs)], timestamp=t)}


@pytest.mark.cpu_ok
def test_pipeline_records_run_length_version_segments_per_emitted_token():
    pipeline, output = _pipeline_with_request()
    pipeline.submit(_bundle([65, 66]), weight_version=0)
    pipeline.submit(_bundle([67], t=2.0), weight_version=0)
    snapshot = output.outputs[0].weight_version_segments
    pipeline.submit(_bundle([68, 69], t=3.0), weight_version=3)
    pipeline.submit(_bundle([70], finished=True, t=4.0), weight_version=4)

    comp = output.outputs[0]
    assert list(comp.token_ids) == [65, 66, 67, 68, 69, 70]
    assert comp.weight_version_segments == [(0, 0), (3, 3), (5, 4)]
    assert (comp.oldest_weight_version, comp.newest_weight_version) == (0, 4)
    assert (output.oldest_weight_version, output.newest_weight_version) == (0, 4)
    # A snapshot handed out earlier keeps its value.
    assert snapshot == [(0, 0)]
    # Expanded per token, aligned with token_ids.
    assert esurge_generation.expand_weight_version_segments(comp.weight_version_segments, 6) == [0, 0, 0, 3, 3, 4]


@pytest.mark.cpu_ok
def test_pipeline_without_versions_keeps_outputs_unversioned():
    pipeline, output = _pipeline_with_request()
    pipeline.submit(_bundle([65, 66], finished=True))
    comp = output.outputs[0]
    assert comp.weight_version_segments is None
    assert comp.oldest_weight_version is None and output.newest_weight_version is None


@pytest.mark.cpu_ok
def test_outputs_to_arrays_exposes_per_token_versions_only_when_present():
    def completion(tokens, segments):
        return SimpleNamespace(text="x", token_ids=tokens, logprobs=None, weight_version_segments=segments)

    versioned = SimpleNamespace(
        prompt="p",
        prompt_token_ids=[[5, 6]],
        outputs=[completion([1, 2, 3], [(0, 2), (2, 3)]), completion([4], None)],
    )
    kwargs = dict(input_ids=None, attention_mask=None, return_prompts=None, pad_token_id=0, max_seq_len=2)
    arrays = esurge_generation.outputs_to_arrays([versioned], max_new_tokens=4, **kwargs)
    np.testing.assert_array_equal(np.asarray(arrays.completion_weight_versions), [[2, 2, 3, -1], [-1, -1, -1, -1]])
    assert arrays.completion_weight_versions.dtype == jnp.int32

    plain = SimpleNamespace(prompt="p", prompt_token_ids=[[5]], outputs=[SimpleNamespace(text="x", token_ids=[1])])
    assert esurge_generation.outputs_to_arrays([plain], max_new_tokens=2, **kwargs).completion_weight_versions is None


@pytest.mark.cpu_ok
def test_between_steps_call_runs_inline_and_propagates_errors_when_loop_is_not_running():
    loop = EngineLoop(
        core=SimpleNamespace(),
        runner=SimpleNamespace(),
        coordinator=None,
        scheduler_lock=threading.RLock(),
        emit_outputs=lambda outputs: None,
        drain_parser_stops=lambda: None,
        on_fatal=lambda exc: None,
        heartbeat=lambda: None,
        handle_profiling_step=lambda: None,
        is_nonrecoverable=lambda exc: False,
        overlap_execution=False,
        async_scheduling=False,
        runner_verbose=False,
        info=lambda *args: None,
    )
    assert loop.run_between_steps(lambda: 7) == 7

    def boom():
        raise KeyError("x")

    with pytest.raises(KeyError):
        loop.run_between_steps(boom)


# ------------------------------------------------------------------------------------------- engine (TPU)


def _llama(seed: int):
    import easydel as ed

    tokenizer = load_test_tokenizer()
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
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(seed),
        )
    return model, tokenizer


@pytest.fixture(scope="module")
def models():
    model_a, tokenizer = _llama(0)
    model_b, _ = _llama(1)
    return model_a, model_b, tokenizer


def _engine(model, tokenizer, *, prefix_caching: bool, max_num_seqs: int = 2):
    return esurge_engine.eSurge(
        model=model,
        processor=tokenizer,
        runtime={"max_model_len": MAX_MODEL_LEN, "max_num_seqs": max_num_seqs},
        cache={
            "page_size": 16,
            "hbm_utilization": 0.05,
            "max_cache_tokens": 4096,
            "enable_prefix_caching": prefix_caching,
        },
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )


def _greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)


def test_weight_versions_increase_and_tag_generated_tokens(models):
    model_a, model_b, tokenizer = models
    engine = _engine(model_a, tokenizer, prefix_caching=False)
    try:
        assert engine.weight_version == 0
        out = engine.generate("hello world", _greedy(8), use_tqdm=False)[0]
        assert out.outputs[0].weight_version_segments == [(0, 0)]
        assert engine.update_model_weights(model_b) == 1
        assert engine.update_model_weights(model_a, version=5) == 5
        with pytest.raises(ValueError, match="must increase"):
            engine.update_model_weights(model_b, version=5)
        assert engine.weight_version == 5
        out = engine.generate("hello world", _greedy(8), use_tqdm=False)[0]
        assert out.outputs[0].weight_version_segments == [(0, 5)]
        assert (out.oldest_weight_version, out.newest_weight_version) == (5, 5)
    finally:
        engine.terminate()


def test_prefix_cached_prompt_is_recomputed_after_install(models):
    model_a, model_b, tokenizer = models
    prompt = "The quick brown fox jumps over the lazy dog. " * 4
    engine = _engine(model_a, tokenizer, prefix_caching=True)
    try:
        engine.generate(prompt, _greedy(4), use_tqdm=False)
        warm = engine.generate(prompt, _greedy(4), use_tqdm=False)[0]
        assert warm.metrics["cache_read"] > 0, "the repeated prompt did not hit the prefix cache"
        engine.update_model_weights(model_b)
        after = engine.generate(prompt, _greedy(16), use_tqdm=False)[0]
        assert after.metrics["cache_read"] == 0
    finally:
        engine.terminate()
    fresh = _engine(model_b, tokenizer, prefix_caching=True)
    try:
        reference = fresh.generate(prompt, _greedy(16), use_tqdm=False)[0]
    finally:
        fresh.terminate()
    assert list(after.outputs[0].token_ids) == list(reference.outputs[0].token_ids)


def _stream_in_thread(engine, prompt, params, request_id):
    started = threading.Event()
    result: dict = {}

    def run():
        last = None
        try:
            for last in engine.stream(prompt, params, request_id=request_id):
                if last.outputs[0].token_ids and len(last.outputs[0].token_ids) >= 2:
                    started.set()
        except Exception as exc:  # surfaced through result
            result["error"] = exc
        finally:
            started.set()
            result["final"] = last

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, started, result


def test_keep_running_install_splits_segments_mid_request(models):
    model_a, model_b, tokenizer = models
    engine = _engine(model_a, tokenizer, prefix_caching=False)
    try:
        baseline = engine.generate("once upon a time", _greedy(400), use_tqdm=False)[0].outputs[0]
        thread, started, result = _stream_in_thread(engine, "once upon a time", _greedy(400), "kept")
        assert started.wait(timeout=120)
        assert engine.update_model_weights(model_b, keep_requests=True) == 1
        thread.join(timeout=300)
        assert "error" not in result
        comp = result["final"].outputs[0]
        assert comp.finish_reason == "length"
        assert len(comp.token_ids) == 400
        segments = comp.weight_version_segments
        assert [version for _, version in segments] == [0, 1]
        switch = segments[1][0]
        assert 2 <= switch < 400
        # Tokens produced before the install came from the old weights, exactly as without the swap.
        assert list(comp.token_ids)[:switch] == list(baseline.token_ids)[:switch]
    finally:
        engine.terminate()


def test_fence_aborts_only_requests_with_older_tokens(models):
    model_a, model_b, tokenizer = models
    engine = _engine(model_a, tokenizer, prefix_caching=False, max_num_seqs=1)
    try:
        thread, started, result = _stream_in_thread(engine, "once upon a time", _greedy(400), "old")
        assert started.wait(timeout=120)
        engine.update_model_weights(model_b, keep_requests=True)
        queued: dict = {}
        queued_thread = threading.Thread(
            target=lambda: queued.update(
                out=engine.generate("a queued prompt", _greedy(8), request_id="queued", use_tqdm=False)[0]
            ),
            daemon=True,
        )
        queued_thread.start()
        # max_num_seqs=1: the second request waits behind the first and has no tokens yet.
        assert engine.abort_requests_before_version(1) == ["old"]
        thread.join(timeout=120)
        queued_thread.join(timeout=300)
        assert result["final"].outputs[0].finish_reason == "abort"
        kept = queued["out"].outputs[0]
        assert kept.finish_reason == "length"
        assert kept.weight_version_segments == [(0, 1)]
        assert engine.abort_requests_before_version(2) == []
    finally:
        engine.terminate()
