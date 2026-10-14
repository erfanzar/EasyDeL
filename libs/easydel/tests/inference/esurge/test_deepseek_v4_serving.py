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
# See the License for the specific language governing permissions and
# limitations under the License.

"""eSurge serving parity tests for DeepSeek-V4's compressed-window cache (CPU, tiny model).

Drives the real ``eSurgeRunner`` + ``Scheduler`` (same harness pattern as
``test_vlm_families_serving.py``) against a tiny random-init DeepSeek-V4 that
covers all three attention regimes (sliding / CSA rate 2 / HCA rate 4,
``sliding_window=8``) plus the hash-MoE/MoE layer mix, in float32 end to end.

What this validates (the compressed-window serving contract):
    * Packed multi-request serving against per-slot cache rows: batch of 2
      requests at different prompt lengths, prefill + >= 16 greedy decode
      steps, each engine token matching the argmax of a teacher-forced
      *stateless* dense forward of that request's full sequence.
    * Chunked prefill: prompts split across steps via
      ``long_prefill_token_threshold`` resume from per-slot positions.
    * Requests finishing at different times (short and long ``max_tokens``)
      without corrupting each other's slots.
    * A second wave of requests reusing freed slots cleanly (slot reset).
    * Deterministic repeat on the same sync engine.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import easydel as ed
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

VOCAB = 256
MAX_NEW_TOKENS = 16
_REFERENCE_PAD_LEN = 64

# Two prompts at DIFFERENT lengths so per-slot stream positions diverge
# immediately (8 tokens crosses a full sliding window + several CSA/HCA
# windows during prefill; 5 stays mid-window).
TEXT_PROMPTS = (
    [3, 1, 4, 1, 5, 9, 2, 6],
    [10, 11, 12, 13, 14],
)


def _make_tiny_v4():
    """Build the tiny fp32 DeepSeek-V4 covering all three layer types."""
    cfg = ed.DeepseekV4Config(
        vocab_size=VOCAB,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=512,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
        hc_mult=2,
        hc_sinkhorn_iters=5,
        mlp_layer_types=["hash_moe", "moe", "moe", "moe"],
        swiglu_limit=10.0,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=16,
        # Deterministic indexer regime (top-k >= possible entries): a
        # truncating top-k over relu-dead exact-zero ties orders
        # backend-dependently, which is not a portable parity target.
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        # Wider weights keep the greedy stream varied; a tiny default-init
        # model degenerates to echoing one token, which would let
        # position/slot bugs hide behind a constant sequence.
        initializer_range=0.2,
        tie_word_embeddings=False,
    )
    # CPU MoE path: XLA grouped-matmul, no expert-parallel collectives.
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False
    # Exact fp32 numerics end to end for the argmax parity bar.
    cfg.attn_dtype = jnp.float32
    cfg.kvdtype = jnp.float32
    _, module_class = ed.get_modules_by_type("deepseek_v4", ed.TaskType.CAUSAL_LM)
    model = module_class(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    model.eval()
    return model


class _ServingHarness:
    """One compiled eSurgeRunner + Scheduler serving several greedy batches."""

    def __init__(self, model, *, long_prefill_token_threshold: int | None = None):
        self.model = model
        self.runner = eSurgeRunner(
            model=model,
            hbm_utilization=0.02,
            page_size=16,
            max_cache_tokens=2048,
            max_model_len=128,
            max_num_batched_tokens=64,
            min_input_pad=1,
            min_token_pad=16,
            max_num_seqs=2,
            max_num_seq_buckets=[2],
            async_scheduling=False,
            use_aot_forward=False,
            verbose=False,
            enable_overlap_execution=False,
            enable_sampler_metrics=False,
        )
        self.runner.compile(max_num_batched_tokens=64)
        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=64,
            # The compressed-window guard must force this off; passing True
            # exercises the guard on every harness build.
            enable_prefix_caching=True,
            async_scheduling=False,
            long_prefill_token_threshold=long_prefill_token_threshold,
        )
        self._req_counter = 0

    def generate_greedy(self, request_specs: list[dict]) -> list[list[int]]:
        """Serve one batch of requests to completion with greedy sampling.

        Args:
            request_specs: Per-request dicts with ``prompt_token_ids`` and an
                optional ``max_tokens`` (defaults to ``MAX_NEW_TOKENS``).

        Returns:
            Per-request generated token-id lists (prompt excluded).
        """
        requests = []
        for spec in request_specs:
            max_tokens = int(spec.get("max_tokens", MAX_NEW_TOKENS))
            req = EngineRequest(
                request_id=f"req-{self._req_counter}",
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            self._req_counter += 1
            requests.append(req)
            self.scheduler.add_request(req)

        for _ in range(64 * len(requests)):
            scheduler_output = self.scheduler.schedule()
            output = self.runner.execute_model(scheduler_output)
            self.scheduler.update_from_output(scheduler_output, output)
            if all(r.is_finished() for r in requests):
                break
        else:
            raise AssertionError("generation did not finish within the step budget")

        outputs = [list(r.output_token_ids) for r in requests]
        for spec, toks in zip(request_specs, outputs, strict=True):
            expected_len = int(spec.get("max_tokens", MAX_NEW_TOKENS))
            assert len(toks) == expected_len, f"expected {expected_len} tokens, got {len(toks)}"
            assert all(0 <= t < VOCAB for t in toks), f"token ids out of range: {toks}"
        return outputs

    def close(self) -> None:
        """Release the compiled runner and its cache state."""
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        self.scheduler = None
        gc.collect()


def _assert_matches_stateless_reference(
    model,
    prompt_ids: list[int],
    generated_ids: list[int],
    *,
    tie_tolerance: float = 5e-4,
) -> None:
    """Teacher-force ``prompt + generated`` through the stateless dense forward.

    Asserts that at every continuation position the engine's token is the
    argmax of the reference logits (up to near-exact logit ties, which can
    legally flip under fp32 reduction-order changes between the packed scan
    and the dense forward). Real slot/position/reset bugs produce logit gaps
    orders of magnitude above the tolerance.
    """
    from easydel.utils.inference_mode import set_inference_mode

    full = [*prompt_ids, *generated_ids]
    assert len(full) <= _REFERENCE_PAD_LEN, "reference pad length too small"
    padded = full + [0] * (_REFERENCE_PAD_LEN - len(full))
    ids = jnp.asarray([padded], dtype=jnp.int32)
    # This reference IS an inference forward, so declare it as one. The model
    # takes the default `(1, 1, -1, 1, 1, 1)` mesh, i.e. fsdp == device_count,
    # and the batch here is 1: under the CPU trio's 8 fake devices the fused-MoE
    # `('dp', 'fsdp')` batch spec cannot divide it. `BaseMoeModule` drops the
    # batch axis in that case only for inference traces -- a TRAINING batch that
    # cannot divide keeps the loud shard_map error on purpose -- so without this
    # scope the reference dies with "8 does not evenly divide 1" before eSurge
    # is ever compared against anything. The engine's own compiles already run
    # under this scope.
    with model.mesh, set_inference_mode():
        logits = np.asarray(
            model(input_ids=ids, position_ids=jnp.arange(_REFERENCE_PAD_LEN, dtype=jnp.int32)[None, :]).logits.astype(
                jnp.float32
            )
        )[0]
    assert np.isfinite(logits[: len(full)]).all(), "non-finite reference logits"

    n_prompt = len(prompt_ids)
    mismatches = []
    for step, token in enumerate(generated_ids):
        row = logits[n_prompt + step - 1]
        ref_argmax = int(row.argmax())
        gap = float(row.max() - row[int(token)])
        if ref_argmax != int(token) and gap > tie_tolerance:
            mismatches.append((step, int(token), ref_argmax, gap))
    assert not mismatches, (
        "eSurge greedy decode diverged from the teacher-forced stateless forward "
        f"(step, engine_token, reference_argmax, logit_gap): {mismatches}"
    )


@pytest.fixture(scope="module")
def tiny_v4_model():
    return _make_tiny_v4()


def test_deepseek_v4_esurge_greedy_parity_and_slot_lifecycle(tiny_v4_model):
    """Full serving protocol on one engine: parity, waves, finish skew, repeat."""
    model = tiny_v4_model
    harness = _ServingHarness(model)
    try:
        # Prefix caching must have been forced off by the family guard.
        assert harness.scheduler.enable_prefix_caching is False

        # Phase 1: batch of 2 at different lengths, >= 16 greedy decode steps,
        # per-request parity against the stateless dense forward.
        specs = [{"prompt_token_ids": p} for p in TEXT_PROMPTS]
        served = harness.generate_greedy(specs)
        for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)

        # Phase 2: requests finishing at different times. The short request
        # frees its slot mid-flight; the long request must keep decoding from
        # uncorrupted state (verified by parity on the full continuation).
        skew_specs = [
            {"prompt_token_ids": TEXT_PROMPTS[0], "max_tokens": 6},
            {"prompt_token_ids": TEXT_PROMPTS[1], "max_tokens": 18},
        ]
        skew_served = harness.generate_greedy(skew_specs)
        for spec, generated in zip(skew_specs, skew_served, strict=True):
            _assert_matches_stateless_reference(model, list(spec["prompt_token_ids"]), generated)
        # The 18-token continuation extends the 16-token phase-1 stream of the
        # same prompt; their common greedy prefix must agree.
        assert skew_served[1][:MAX_NEW_TOKENS] == served[1]

        # Phase 3: a second wave reusing the freed slots cleanly. New prompts
        # (different content and lengths) must not inherit stale ring/
        # compressor/indexer state from phase 1/2 occupants.
        wave_prompts = ([42, 7, 99, 5], [23, 55, 8, 13, 21, 34, 2])
        wave_specs = [{"prompt_token_ids": list(p)} for p in wave_prompts]
        wave_served = harness.generate_greedy(wave_specs)
        for prompt, generated in zip(wave_prompts, wave_served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)

        # Phase 4: deterministic repeat of phase 1 on the same sync engine.
        repeated = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        assert repeated == served, (
            f"repeated generation on the same engine is not deterministic\n  first ={served}\n  second={repeated}"
        )
    finally:
        harness.close()


def test_deepseek_v4_esurge_chunked_prefill_parity(tiny_v4_model):
    """Chunked prefill resumes compressed-window state from per-slot positions.

    ``long_prefill_token_threshold=3`` splits both prompts across several
    scheduler steps (chunks smaller than the sliding window AND misaligned
    with the CSA/HCA compression rates), so buffers carry open-window state
    between chunks. Parity against the stateless forward proves the resume
    path is exact.
    """
    model = tiny_v4_model
    harness = _ServingHarness(model, long_prefill_token_threshold=3)
    try:
        specs = [{"prompt_token_ids": p} for p in TEXT_PROMPTS]
        served = harness.generate_greedy(specs)
        for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)
    finally:
        harness.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
