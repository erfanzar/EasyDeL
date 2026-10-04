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

"""DSpark speculative decoding for DeepSeek-V4.1 in eSurge: greedy parity with plain decoding (computation test).

A tiny fp32 / ``HIGHEST`` DeepSeek-V4.1 (shared compressed-window cache, ratio-2 and ratio-1 layers) is served through the
real ``eSurgeRunner`` + ``Scheduler`` with and without a :class:`DeepseekV41DSparkDrafter`. Every run schedules two
requests of different prompt lengths and different ``max_tokens`` (so they finish at different steps), then a second wave
that reuses the freed cache slots and drafter ring rows. Greedy outputs must be identical to the run without speculation.

Three drafters are exercised:

- the real DSpark network with random weights (acceptance is ~0, so almost every verify window is rejected and the
  snapshot-restore-replay rollback of the non-paged compressed-window cache runs on nearly every step);
- the same network wrapped so that its proposals are replaced by the target's own greedy continuation (the bookkeeping
  of the DSpark rings still runs), with every third window corrupted after its first token: full acceptance, partial
  acceptance and rejection all occur;
- the same oracle drafter, whose recorded real draft proposals are compared with a stateless
  ``DeepseekV41DSpark.prefill`` / ``draft`` over ``output_dspark_hidden`` of the target (ring contents and the lazy
  catch-up of accepted draft positions are therefore pinned to the draft model itself).
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
from easydel.modules.deepseek_v41 import DeepseekV41Config, DeepseekV41DSpark
from easydel.modules.deepseek_v41.modeling_deepseek_v41_dspark import DeepseekV41DSparkDrafter
from easydel.utils import set_inference_mode

VOCAB = 256
BLOCK = 3
_REFERENCE_PAD_LEN = 64
STEP_BUDGET = 256

WAVE1 = [
    {"id": "w1-a", "prompt_token_ids": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], "max_tokens": 14},
    {"id": "w1-b", "prompt_token_ids": [10, 11, 12, 13, 14], "max_tokens": 9},
]
WAVE2 = [
    {"id": "w2-a", "prompt_token_ids": [42, 7, 99, 5, 23, 55, 8, 13], "max_tokens": 12},
    {"id": "w2-b", "prompt_token_ids": [21, 34, 2], "max_tokens": 11},
]
WAVES = (WAVE1, WAVE2)


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
        dspark_block_size=BLOCK,
        dspark_target_layer_ids=[3, 4, 5],
        dspark_noise_token_id=VOCAB - 1,
        dspark_markov_rank=8,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        image_token_id=VOCAB - 2,
        max_position_embeddings=512,
        initializer_range=0.2,
    )
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False
    cfg.attn_dtype = jnp.float32
    cfg.kvdtype = jnp.float32
    _, module_class = ed.get_modules_by_type("deepseek_v41", ed.TaskType.CAUSAL_LM)
    # HIGHEST: an fp32 matmul at default precision is one bf16 MXU pass on TPU, so differently shaped windows (packed
    # verify vs single-token decode) would compare two bf16-floor computations (gaps ~1e-2, not a bug).
    model = module_class(
        config=cfg,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    model.eval()
    return model


def _build_dspark(model) -> DeepseekV41DSpark:
    with model.config.mesh:
        return DeepseekV41DSpark(
            model.config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
        )


class _OracleDrafter(DeepseekV41DSparkDrafter):
    """DSpark drafter whose proposals are the target's own greedy continuation (every N-th window corrupted)."""

    def __init__(self, *args, sequences: dict[str, list[int]], corrupt_every: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequences = sequences
        self.corrupt_every = corrupt_every
        self.windows = 0
        self.records: list[tuple[str, int, int, list[int]]] = []

    def draft_window(self, **kwargs) -> list[int]:
        real = super().draft_window(**kwargs)  # the DSpark network runs and the rings advance as in production
        if not real:
            return []
        req_id, position, k = str(kwargs["req_id"]), int(kwargs["seed_position"]), int(kwargs["k"])
        sequence = self.sequences[req_id]
        if position + 1 >= len(sequence):
            # The scheduler can ask for a window seeded by the request's final token; the baseline has no
            # continuation there, so the oracle proposes nothing (the real DSpark call above already ran).
            return []
        assert int(kwargs["seed_token"]) == sequence[position + 1], "seed is not the baseline token"
        self.records.append((req_id, position, int(kwargs["seed_token"]), list(real)))
        oracle = list(sequence[position + 2 : position + 2 + k])
        self.windows += 1
        if len(oracle) >= 2 and self.windows % self.corrupt_every == 0:
            oracle[1] = (oracle[1] + 1) % VOCAB  # accepted == 1, then a rejection
        return oracle


class _Harness:
    """Real runner + scheduler, optionally with a drafter."""

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
            use_aot_forward=False,
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
        for wave in WAVES:  # wave 2 reuses the slots (and drafter rows) freed by wave 1
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
def tiny_v41_model():
    return _make_tiny_v41()


@pytest.fixture(scope="module")
def baseline(tiny_v41_model) -> dict[str, list[int]]:
    """Greedy outputs of every request without speculation."""
    harness = _Harness(tiny_v41_model)
    try:
        served = harness.run_waves()
    finally:
        harness.close()
    assert all(len(set(toks)) > 1 for toks in served.values()), served  # not a degenerate constant stream
    return served


def _stateless_main_hidden(model, tokens: list[int]):
    padded = tokens + [0] * (_REFERENCE_PAD_LEN - len(tokens))
    ids = jnp.asarray([padded], dtype=jnp.int32)
    with model.mesh, set_inference_mode():
        out = model(
            input_ids=ids,
            position_ids=jnp.arange(_REFERENCE_PAD_LEN, dtype=jnp.int32)[None, :],
            output_dspark_hidden=True,
        )
    return out.main_hidden


def test_registry_builds_dspark_drafter(tiny_v41_model):
    dspark = _build_dspark(tiny_v41_model)
    drafter = tiny_v41_model.drafter(method="deepseek_v41_dspark", drafter_model=dspark, num_draft_tokens=2)
    assert isinstance(drafter, DeepseekV41DSparkDrafter)
    assert drafter.num_draft_tokens == 2 and drafter.block_size == BLOCK
    clamped = tiny_v41_model.drafter(method="v41_dspark", assistant_model=dspark, num_draft_tokens=99)
    assert clamped.num_draft_tokens == BLOCK  # never more than dspark_block_size
    with pytest.raises(ValueError, match="drafter_model"):
        tiny_v41_model.drafter(method="deepseek_v41_dspark")


def test_random_dspark_drafter_matches_baseline(tiny_v41_model, baseline):
    """Real DSpark proposals (almost all rejected): rollback by snapshot-restore-replay keeps the stream exact."""
    drafter = DeepseekV41DSparkDrafter(
        target_model=tiny_v41_model, dspark=_build_dspark(tiny_v41_model), num_draft_tokens=BLOCK
    )
    harness = _Harness(tiny_v41_model, drafter=drafter)
    try:
        assert harness.runner.num_draft_tokens == BLOCK
        served = harness.run_waves()
        runner = harness.runner
        generated = runner.spec_decode_num_drafts_generated
        accepted = runner.spec_decode_num_drafts_accepted
        assert generated > 0 and runner.spec_decode_num_verify_steps > 0, "speculation never ran"
        assert 0 <= accepted <= generated
        rate = runner.spec.acceptance_rate
        assert rate == pytest.approx(accepted / generated)
        print(f"[dspark random] generated={generated} accepted={accepted} acceptance_rate={rate:.3f}")
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}


def test_oracle_dspark_drafter_accepts_rejects_and_matches_draft_model(tiny_v41_model, baseline):
    """Accepted, partially accepted and rejected windows reproduce the baseline; drafts equal the draft model's."""
    model = tiny_v41_model
    dspark = _build_dspark(model)
    prompts = {s["id"]: s["prompt_token_ids"] for wave in WAVES for s in wave}
    sequences = {rid: list(prompts[rid]) + baseline[rid] for rid in baseline}
    drafter = _OracleDrafter(
        target_model=model, dspark=dspark, num_draft_tokens=BLOCK, sequences=sequences, corrupt_every=3
    )
    harness = _Harness(model, drafter=drafter)
    try:
        served = harness.run_waves()
        runner = harness.runner
        generated = runner.spec_decode_num_drafts_generated
        accepted = runner.spec_decode_num_drafts_accepted
        print(
            f"[dspark oracle] generated={generated} accepted={accepted} "
            f"acceptance_rate={runner.spec.acceptance_rate:.3f} windows={drafter.windows}"
        )
        assert accepted > 0, "no draft was ever accepted: the acceptance path was not exercised"
        assert accepted < generated, "no draft was ever rejected: the rollback path was not exercised"
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}

    checked = 0
    for req_id, position, token, real in drafter.records:
        if position < 1 or checked >= 8:
            continue
        main_hidden = _stateless_main_hidden(model, sequences[req_id])
        with model.mesh, set_inference_mode():
            state = dspark.prefill(main_hidden[:, :position])
            ids, *_ = dspark.draft(
                jnp.asarray([token], dtype=jnp.int32),
                main_hidden[:, position : position + 1],
                state,
                model.compute_embedding,
                model.compute_lm_logits,
            )
        expected = [int(t) for t in np.asarray(ids)[0, 1 : 1 + len(real)]]
        assert real == expected, (req_id, position, real, expected)
        checked += 1
    assert checked > 0, "no draft window could be compared with the draft model"
