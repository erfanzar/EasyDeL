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

"""Shared helpers for tiny in-memory eSurge CPU runtime tests.

These tests drive the *full* eSurge engine (scheduler thread, paged ragged
cache, bucket runners, greedy sampler) with a tiny randomly initialized model
on CPU, then validate the cached decode path against the plain uncached model
forward. They run under the standard CPU trio::

    ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=8

The tiny models are built in float32 (params, activations, and KV cache) so
the decode-vs-forward comparison is an exact argmax parity check rather than a
bf16-noise-tolerant one; any paged-cache slot, position-id, or RoPE-offset bug
shows up as a hard argmax divergence within a couple of decode steps.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import pytest
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

TOKENIZER_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 2
PAGE_SIZE = 16  # several pages per sequence so decode crosses page boundaries
MAX_TOKENS = 24


def load_test_tokenizer():
    """Load the small cached tokenizer used by the tiny runtime tests.

    Returns:
        PreTrainedTokenizerBase: Tokenizer with a pad token set.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    except OSError as exc:  # pragma: no cover - offline/cache-miss environments
        pytest.skip(f"tokenizer {TOKENIZER_REPO!r} unavailable: {exc}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_engine(model, tokenizer, *, sync: bool = False):
    """Construct an eSurge engine around a preloaded tiny model.

    Small buckets keep CPU compilation cheap; ``page_size=16`` forces every
    generation in these tests to allocate multiple KV pages so page-crossing
    decode is exercised. Prefix caching is disabled so every run goes through
    real prefill + decode. Tool/reasoning parsers are disabled explicitly:
    auto-detection keys off ``model_type`` and would otherwise pair (e.g.) the
    Mistral tool parser with a tokenizer that lacks its control tokens.

    Args:
        model: A preloaded EasyDeL causal LM module.
        tokenizer: Tokenizer/processor for the engine.
        sync: Disable async scheduling and overlap execution. In async mode a
            decode step may run either the handoff-patched or host-repaired
            compiled variant depending on drain timing; both are correct but
            are distinct executables, so bit-exact reproducibility is only
            contractual in sync mode.

    Returns:
        eSurge: An initialized (not yet initiated) engine.
    """
    from easydel.inference.esurge.esurge_engine import eSurge

    runtime: dict = {
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
    }
    if sync:
        runtime["async_scheduling"] = False
        runtime["overlap_execution"] = False
    return eSurge(
        model=model,
        processor=tokenizer,
        runtime=runtime,
        cache={
            "page_size": PAGE_SIZE,
            "hbm_utilization": 0.05,
            "max_cache_tokens": 2048,
            "enable_prefix_caching": False,
        },
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    )


def greedy_params() -> SamplingParams:
    """Greedy sampling params used by every tiny runtime run."""
    return SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True)


def run_greedy_batch(engine, prompts: list[str]) -> dict:
    """Run one greedy batch (>= 2 concurrent requests) through the engine.

    Args:
        engine: An eSurge engine (``initiate()`` is called here).
        prompts: Batch of prompt strings.

    Returns:
        dict: Prompt -> ``RequestOutput``.
    """
    engine.initiate()
    outputs = engine.generate(prompts, sampling_params=greedy_params(), use_tqdm=False)
    return {out.prompt: out for out in outputs}


def run_greedy_single_twice(engine, prompt: str):
    """Run the same single-request greedy generation twice.

    With a single in-flight request on a sync-configured engine every step has
    a fixed batch composition and a single compiled variant, so the two runs
    must match token-for-token.

    Args:
        engine: An eSurge engine (``initiate()`` is called here).
        prompt: Prompt string.

    Returns:
        tuple[RequestOutput, RequestOutput]: Outputs of the two runs.
    """
    engine.initiate()
    first = engine.generate([prompt], sampling_params=greedy_params(), use_tqdm=False)
    second = engine.generate([prompt], sampling_params=greedy_params(), use_tqdm=False)
    return first[0], second[0]


def request_token_ids(request_output) -> tuple[list[int], list[int]]:
    """Extract flat prompt token ids and generated token ids from an output.

    Args:
        request_output: An eSurge ``RequestOutput``.

    Returns:
        tuple[list[int], list[int]]: ``(prompt_ids, generated_ids)``.
    """
    prompt_ids = list(request_output.prompt_token_ids)
    if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
        prompt_ids = list(prompt_ids[0])
    return [int(t) for t in prompt_ids], [int(t) for t in request_output.outputs[0].token_ids]


def assert_cached_decode_matches_uncached_forward(
    model,
    prompt_ids: tp.Sequence[int],
    generated_ids: tp.Sequence[int],
    *,
    tie_tolerance: float = 5e-3,
) -> None:
    """Assert the engine's greedy continuation matches the uncached forward.

    Teacher-forces ``prompt + generated`` through the plain (cache-free) model
    forward and asserts that at every continuation position the engine's token
    is the argmax of the reference logits. This catches paged-cache slot
    mapping, absolute-position / RoPE-offset, and stale-token-feedback bugs
    that generation-only smokes cannot see.

    Args:
        model: The original (unconverted) EasyDeL module used to build the
            engine; its plain forward is the reference.
        prompt_ids: Prompt token ids as returned by the engine.
        generated_ids: Greedy-generated token ids from the engine.
        tie_tolerance: Maximum reference logit gap under which a non-argmax
            token is still accepted. Randomly initialized tiny models can have
            near-exact top-logit ties that legally flip with batching-induced
            float reduction-order changes (~1e-5); real cache/position bugs
            produce gaps orders of magnitude above this (~0.2+ observed).
    """
    full = np.asarray([*prompt_ids, *generated_ids], dtype=np.int32)[None, :]
    with model.mesh:
        outputs = model(
            input_ids=jnp.asarray(full),
            attention_mask=jnp.ones_like(jnp.asarray(full)),
        )
    logits = np.asarray(outputs.logits.astype(jnp.float32))[0]
    assert np.isfinite(logits).all(), "reference forward produced non-finite logits"

    n_prompt = len(prompt_ids)
    mismatches = []
    for step, token in enumerate(generated_ids):
        row = logits[n_prompt + step - 1]
        ref_argmax = int(row.argmax())
        gap = float(row.max() - row[int(token)])
        if ref_argmax != int(token) and gap > tie_tolerance:
            mismatches.append((step, int(token), ref_argmax, gap))
    assert not mismatches, (
        "cached decode diverged from the uncached teacher-forced forward "
        f"(step, engine_token, reference_argmax, logit_gap): {mismatches}"
    )
