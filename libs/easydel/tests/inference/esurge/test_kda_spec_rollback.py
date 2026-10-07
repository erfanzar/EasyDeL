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

"""Speculative rollback on KDA (Kimi-style linear attention) caches: greedy parity with plain serving.

A tiny fp32 GLM-5-Next hybrid (3 KDA layers + 1 DSA/MLA layer, HIGHEST precision) is served by the real
``eSurgeRunner`` + ``Scheduler`` with and without an oracle drafter whose proposals are the plain greedy continuation
with every 3rd window corrupted. Rejected drafts must leave the KDA short-conv windows and recurrent rows exactly as
plain decoding would. KDA pools keep speculative candidate rows (decode-exact state after every verify prefix,
written by ``kda_packed_rows_forward``); the accepted prefix's row is committed into the live row at the next step
(regressions: ``KDACacheView`` was once invisible to the rollback, so a rejected draft left the KDA state advanced
through the rejected tokens; a commit that never reaches the kernel would do the same).

The default (fast) recurrent path is tested on purpose: ``EASYDEL_SPEC_RECURRENT_REPLAY`` is left unset.
"""

from __future__ import annotations

import gc

import easydel as ed
import jax
import jax.numpy as jnp
import pytest
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import DraftStep

VOCAB = 128
K = 3
STEP_BUDGET = 256

WAVE1 = [
    {"id": "w1-a", "prompt_token_ids": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], "max_tokens": 14},
    {"id": "w1-b", "prompt_token_ids": [10, 11, 12, 13, 14], "max_tokens": 11},
]
WAVE2 = [
    {"id": "w2-a", "prompt_token_ids": [42, 7, 99, 5, 23, 55, 8, 13], "max_tokens": 12},
    {"id": "w2-b", "prompt_token_ids": [21, 34, 2], "max_tokens": 13},
]
WAVES = (WAVE1, WAVE2)


@pytest.fixture(autouse=True)
def _default_fast_recurrent_path(monkeypatch):
    monkeypatch.delenv("EASYDEL_SPEC_RECURRENT_REPLAY", raising=False)


def _make_tiny_glm5_next():
    config = ed.Glm5NextTextConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=8,
        v_head_dim=8,
        index_topk=64,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 4,
        linear_head_dim=8,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        hc_mult=2,
        max_position_embeddings=512,
        # Wider weights keep the greedy stream varied (a near-constant stream would hide state bugs).
        initializer_range=0.2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    config.kvdtype = jnp.float32
    with config.mesh:
        model = ed.Glm5NextForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    model.eval()
    return model


class _OracleDrafter:
    """Generic chained drafter proposing the plain greedy continuation (every N-th window corrupted).

    The runner calls ``reset_request`` once per verify window, then ``draft`` ``K`` times with the previous token and
    a position advancing by one from the seed position (the seed token sits at ``position + 1`` of the full
    prompt + output sequence).
    """

    num_draft_tokens = K
    supports_return_full_log_probs = False
    supports_prefix_draft = False
    requires_target_kv_cache = False

    def __init__(self, sequences: dict[str, list[int]], corrupt_every: int):
        self.sequences = sequences
        self.corrupt_every = corrupt_every
        self.windows = 0
        self.errors: list[tuple] = []
        self._req: str | None = None
        self._step = 0

    def reset(self, batch_size: int) -> None:
        del batch_size

    def reset_request(self, request_id: str | None = None) -> None:
        self._req = None if request_id is None else str(request_id)
        self._step = 0
        self.windows += 1

    def draft(self, input_ids, target_hidden_states=None, position_ids=None, sample=False, rng_key=None, **kw):
        # The runner swallows drafter exceptions (it just skips the window), so record them for the test to assert.
        try:
            return self._draft(input_ids, position_ids, sample)
        except Exception as exc:
            pos = None if position_ids is None else int(jnp.asarray(position_ids).reshape(-1)[-1])
            seed = int(jnp.asarray(input_ids).reshape(-1)[-1])
            self.errors.append((self._req, self._step, pos, seed, repr(exc)))
            raise

    def _draft(self, input_ids, position_ids, sample):
        assert not sample, "oracle drafter is greedy-only"
        sequence = self.sequences[str(self._req)]
        pos = int(jnp.asarray(position_ids).reshape(-1)[-1])
        if pos + 1 >= len(sequence):
            # The runner drafts once more after the request's last token; the seed then lies past the baseline.
            self._step += 1
            return DraftStep(
                token_ids=jnp.zeros((1,), jnp.int32), log_probs=jnp.zeros((1,), jnp.float32), full_log_probs=None
            )
        if self._step == 0:
            seed = int(jnp.asarray(input_ids).reshape(-1)[-1])
            assert seed == sequence[pos + 1], "seed is not the baseline token"
        idx = pos + 2
        token = sequence[idx] if idx < len(sequence) else 0
        if self._step == 1 and self.windows % self.corrupt_every == 0:
            token = (token + 1) % VOCAB
        self._step += 1
        token_ids = jnp.asarray([token], dtype=jnp.int32)
        return DraftStep(token_ids=token_ids, log_probs=jnp.zeros((1,), jnp.float32), full_log_probs=None)


class _Harness:
    def __init__(self, model, *, drafter=None):
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
            # AOT like the real engine (the executed program is the one traced under the bucket's scopes).
            use_aot_forward=True,
            verbose=False,
            enable_overlap_execution=False,
            enable_sampler_metrics=False,
            drafter=drafter,
        )
        self.runner.compile(max_num_batched_tokens=64)
        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
            async_scheduling=False,
            num_draft_tokens=self.runner.num_draft_tokens,
        )

    def generate_greedy(self, specs: list[dict]) -> list[list[int]]:
        requests = []
        for spec in specs:
            req = EngineRequest(
                request_id=spec["id"],
                prompt_token_ids=list(spec["prompt_token_ids"]),
                sampling_params=SamplingParams(max_tokens=int(spec["max_tokens"]), temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            requests.append(req)
            self.scheduler.add_request(req)
        for _ in range(STEP_BUDGET):
            scheduler_output = self.scheduler.schedule()
            output = self.runner.execute_model(scheduler_output)
            self.scheduler.update_from_output(scheduler_output, output)
            if all(r.is_finished() for r in requests):
                break
        else:
            raise AssertionError("generation did not finish within the step budget")
        outputs = [list(r.output_token_ids) for r in requests]
        for spec, toks in zip(specs, outputs, strict=True):
            assert len(toks) == int(spec["max_tokens"]), (spec["id"], toks)
        return outputs

    def run_waves(self) -> dict[str, list[int]]:
        served: dict[str, list[int]] = {}
        for wave in WAVES:
            for spec, toks in zip(wave, self.generate_greedy(wave), strict=True):
                served[spec["id"]] = toks
        return served

    def close(self) -> None:
        self.runner.shutdown()
        self.runner.executor_manager.kv_pages = None
        self.runner = None
        self.scheduler = None
        gc.collect()


@pytest.fixture(scope="module")
def tiny_model():
    return _make_tiny_glm5_next()


@pytest.fixture(scope="module")
def baseline(tiny_model) -> dict[str, list[int]]:
    harness = _Harness(tiny_model)
    try:
        served = harness.run_waves()
    finally:
        harness.close()
    assert all(len(set(toks)) > 2 for toks in served.values()), served
    return served


def test_kda_rejected_drafts_roll_back_to_plain_greedy(tiny_model, baseline):
    prompts = {s["id"]: s["prompt_token_ids"] for wave in WAVES for s in wave}
    sequences = {rid: list(prompts[rid]) + baseline[rid] for rid in baseline}
    drafter = _OracleDrafter(sequences, corrupt_every=3)
    harness = _Harness(tiny_model, drafter=drafter)
    try:
        served = harness.run_waves()
        runner = harness.runner
        generated = runner.spec_decode_num_drafts_generated
        accepted = runner.spec_decode_num_drafts_accepted
        print(f"[kda spec rollback] generated={generated} accepted={accepted} windows={drafter.windows}")
        print(f"[kda spec rollback] drafter errors={drafter.errors}")
        assert not drafter.errors, drafter.errors
        assert runner.spec.recurrent_candidate_count() == K + 1, "KDA candidate rows are not allocated"
        assert accepted > 0, "no draft was ever accepted"
        assert accepted < generated, "no draft was ever rejected"
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}
