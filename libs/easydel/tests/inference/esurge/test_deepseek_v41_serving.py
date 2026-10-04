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

"""eSurge serving parity for DeepSeek-V4.1 on the shared compressed-window cache (computation test).

Drives the real ``eSurgeRunner`` + ``Scheduler`` (same harness as ``test_deepseek_v4_serving.py``) against a tiny
fp32 DeepSeek-V4.1 with ratio-2 and ratio-1 layers, cross-layer KV sharing (2 KV sources), index sharing and a
candidate source. Every engine token must equal the argmax of a teacher-forced stateless forward of that request's
sequence (the stateless forward is itself pinned to DeepSeek's reference by ``modules/spmd/test_deepseek_v41.py``).

Covered: packed multi-request serving at different prompt lengths, chunked prefill resuming open compression
groups, requests finishing at different times, freed-slot reuse, deterministic repeat. ``index_topk`` and the
candidate budget cover every reachable entry here: a truncating top-k over exact relu-zero ties orders
backend-dependently between the dense and gathered paths, which is not a portable parity target. Real truncating
selection on the cached path is covered by ``test_cached_decode_matches_stateless``.
"""

from __future__ import annotations

import gc
import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.modules.deepseek_v41 import DeepseekV41Config

VOCAB = 256
MAX_NEW_TOKENS = 16
_REFERENCE_PAD_LEN = 64
TEXT_PROMPTS = (
    [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
    [10, 11, 12, 13, 14],
)


def _make_tiny_v41():
    cfg = DeepseekV41Config(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=6,
        num_nextn_predict_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        compress_ratios=[0, 2, 2, 1, 1, 1, 0],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        candidate_source_layer_id=3,
        candidate_topk_blocks=64,
        candidate_block_size=2,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=128,
        hc_mult=2,
        hc_sinkhorn_iters=5,
        sliding_window=8,
        rms_norm_eps=1e-6,
        engram_layer_ids=[],
        dspark_target_layer_ids=[3, 4, 5],
        dspark_noise_token_id=VOCAB - 1,
        image_token_id=VOCAB - 2,
        max_position_embeddings=512,
        initializer_range=0.2,
    )
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False
    cfg.attn_dtype = jnp.float32
    cfg.kvdtype = jnp.float32
    _, module_class = ed.get_modules_by_type("deepseek_v41", ed.TaskType.CAUSAL_LM)
    # HIGHEST: on TPU an fp32 matmul at default precision is a single bf16 MXU pass, so the packed decode and the
    # dense reference would compare two bf16-floor computations of different shapes (gaps ~1e-2..1, not a bug).
    model = module_class(
        config=cfg,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    model.eval()
    return model


class _ServingHarness:
    def __init__(self, model, *, long_prefill_token_threshold: int | None = None):
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
            enable_prefix_caching=True,  # the compressed-window guard must force this off
            async_scheduling=False,
            long_prefill_token_threshold=long_prefill_token_threshold,
        )
        self._req_counter = 0

    def generate_greedy(self, request_specs: list[dict]) -> list[list[int]]:
        requests = []
        for spec in request_specs:
            req = EngineRequest(
                request_id=f"req-{self._req_counter}",
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(
                    max_tokens=int(spec.get("max_tokens", MAX_NEW_TOKENS)), temperature=0.0, ignore_eos=True
                ),
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
            assert len(toks) == int(spec.get("max_tokens", MAX_NEW_TOKENS))
        return outputs

    def close(self) -> None:
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        self.scheduler = None
        gc.collect()


def _assert_matches_stateless_reference(model, prompt_ids, generated_ids, *, tie_tolerance: float = 5e-4) -> None:
    from easydel.utils.inference_mode import set_inference_mode

    full = [*prompt_ids, *generated_ids]
    padded = full + [0] * (_REFERENCE_PAD_LEN - len(full))
    ids = jnp.asarray([padded], dtype=jnp.int32)
    with model.mesh, set_inference_mode():
        logits = np.asarray(
            model(input_ids=ids, position_ids=jnp.arange(_REFERENCE_PAD_LEN, dtype=jnp.int32)[None, :]).logits.astype(
                jnp.float32
            )
        )[0]
    assert np.isfinite(logits[: len(full)]).all()
    n_prompt = len(prompt_ids)
    mismatches = []
    for step, token in enumerate(generated_ids):
        row = logits[n_prompt + step - 1]
        gap = float(row.max() - row[int(token)])
        if int(row.argmax()) != int(token) and gap > tie_tolerance:
            mismatches.append((step, int(token), int(row.argmax()), gap))
    assert not mismatches, f"eSurge diverged from the stateless forward (step, engine, ref, gap): {mismatches}"


@pytest.fixture(scope="module")
def tiny_v41_model():
    return _make_tiny_v41()


def test_deepseek_v41_esurge_greedy_parity_and_slot_lifecycle(tiny_v41_model):
    model = tiny_v41_model
    harness = _ServingHarness(model)
    try:
        assert harness.scheduler.enable_prefix_caching is False
        from easydel.caching.compressed_window import SharedCompressedWindowCache

        assert isinstance(harness.runner.executor_manager.kv_pages, SharedCompressedWindowCache)
        served = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        assert len({tuple(s) for s in served}) == 2 and all(len(set(s)) > 1 for s in served), served
        for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)

        skew = [
            {"prompt_token_ids": TEXT_PROMPTS[0], "max_tokens": 6},
            {"prompt_token_ids": TEXT_PROMPTS[1], "max_tokens": 18},
        ]
        skew_served = harness.generate_greedy(skew)
        for spec, generated in zip(skew, skew_served, strict=True):
            _assert_matches_stateless_reference(model, list(spec["prompt_token_ids"]), generated)
        assert skew_served[1][:MAX_NEW_TOKENS] == served[1]

        wave_prompts = ([42, 7, 99, 5], [23, 55, 8, 13, 21, 34, 2])
        wave_served = harness.generate_greedy([{"prompt_token_ids": list(p)} for p in wave_prompts])
        for prompt, generated in zip(wave_prompts, wave_served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)

        repeated = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        assert repeated == served
    finally:
        harness.close()


def test_deepseek_v41_esurge_chunked_prefill_parity(tiny_v41_model):
    """Chunks of 3 tokens split ratio-2 groups across scheduler steps; open-group state must carry over."""
    model = tiny_v41_model
    harness = _ServingHarness(model, long_prefill_token_threshold=3)
    try:
        served = harness.generate_greedy([{"prompt_token_ids": p} for p in TEXT_PROMPTS])
        for prompt, generated in zip(TEXT_PROMPTS, served, strict=True):
            _assert_matches_stateless_reference(model, list(prompt), generated)
    finally:
        harness.close()
