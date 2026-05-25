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

"""Standalone eSurge runner-native speculative-decoding smoke test."""

from __future__ import annotations

import gc
import os
import traceback

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx

from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import DraftStep
from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig


def make_tiny_model(mtp_layers: int = 1):
    cfg = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=mtp_layers,
        mtp_loss_coef=0.0,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )
    return Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


class BaselineSequenceDrafter:
    """One-token drafter that proposes the next token from a known greedy stream."""

    num_draft_tokens = 1

    def __init__(self, greedy_tokens: list[int]):
        self.greedy_tokens = list(greedy_tokens)
        self.cursor = 0

    def reset(self, batch_size: int) -> None:
        del batch_size

    def draft(
        self,
        input_ids,
        target_hidden_states=None,
        target_kv_cache=None,
        position_ids=None,
        sample: bool = False,
        rng_key=None,
    ) -> DraftStep:
        del target_hidden_states, target_kv_cache, position_ids, rng_key
        if sample:
            raise AssertionError("test drafter is greedy-only")
        seed = int(np.asarray(input_ids).reshape(-1)[-1])
        next_token = 0
        for idx in range(self.cursor, len(self.greedy_tokens) - 1):
            if int(self.greedy_tokens[idx]) == seed:
                next_token = int(self.greedy_tokens[idx + 1])
                self.cursor = idx + 1
                break
        token_ids = jnp.asarray([next_token], dtype=jnp.int32)
        log_probs = jax.nn.log_softmax(
            jax.nn.one_hot(token_ids, 256, dtype=jnp.float32) * 30.0,
            axis=-1,
        )
        token_log_probs = jnp.zeros((1,), dtype=jnp.float32)
        return DraftStep(token_ids=token_ids, log_probs=token_log_probs, full_log_probs=log_probs)


def run_generation(model, *, drafter, max_new_tokens: int = 8) -> tuple[list[int], eSurgeRunner]:
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
        drafter=drafter,
    )
    runner.compile(max_num_batched_tokens=16)
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_speculative_tokens=runner.num_speculative_tokens,
    )
    request = EngineRequest(
        request_id="req-0",
        prompt_token_ids=[3, 1, 4, 1, 5, 9],
        sampling_params=SamplingParams(max_tokens=max_new_tokens, temperature=0.0, ignore_eos=True),
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

    return list(request.output_token_ids), runner


def test_esurge_runner_spec_decode():
    print("\n[test] eSurge runner-native greedy speculative decoding")
    model = make_tiny_model(mtp_layers=1)
    assert model.has_mtp(), "tiny Qwen3.5 model must expose MTP for this integration test"

    baseline_tokens, baseline_runner = run_generation(model, drafter=None)
    baseline_drafts_generated = baseline_runner.spec_decode_num_drafts_generated
    baseline_runner.shutdown()
    baseline_runner.executor_manager.kv_pages = None
    del baseline_runner
    gc.collect()
    spec_tokens, spec_runner = run_generation(model, drafter=BaselineSequenceDrafter(baseline_tokens))

    assert spec_tokens, "speculative run emitted no tokens"
    assert min(spec_tokens) >= 0 and max(spec_tokens) < 256, f"invalid token ids: {spec_tokens}"
    assert spec_runner.spec_decode_num_drafts_accepted > 0, "expected at least one accepted draft"
    assert spec_tokens == baseline_tokens, (
        f"greedy output changed under speculative decoding\n  baseline={baseline_tokens}\n  spec    ={spec_tokens}"
    )
    print(f"     baseline={baseline_tokens}")
    print(f"     spec    ={spec_tokens}")
    print(
        "     accepted="
        f"{spec_runner.spec_decode_num_drafts_accepted} "
        f"generated={spec_runner.spec_decode_num_drafts_generated} "
        f"verify_steps={spec_runner.spec_decode_num_verify_steps}"
    )
    assert baseline_drafts_generated == 0
    print("  PASS")


if __name__ == "__main__":
    try:
        test_esurge_runner_spec_decode()
    except Exception:
        traceback.print_exc()
        raise
