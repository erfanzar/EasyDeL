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

"""DSpark speculative decoding for DeepSeek-V4 in eSurge: greedy parity with plain decoding (computation test).

A tiny fp32 DeepSeek-V4 (sliding / CSA / HCA layers, hash + learned MoE) is served by the real ``eSurgeRunner`` +
``Scheduler`` with and without a :class:`DeepseekV4DSparkDrafter` at ``precision=HIGHEST``. Two drafters:

- the real DSpark network (random weights, nearly every draft rejected): exercises the compressed-window snapshot /
  restore / replay rollback on every window;
- the same network wrapped so its proposals are the target's own greedy continuation with every 3rd window corrupted:
  exercises full acceptance, partial acceptance and rejection.

Both must reproduce plain greedy serving exactly. The backbone's ``output_dspark_hidden`` hook (stream mean AFTER each
target layer) runs inside the packed serving step.
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
from easydel.modules.deepseek_v4 import DeepseekV4DSpark, DeepseekV4DSparkDrafter

VOCAB = 256
BLOCK = 3
STAGES = 2
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


def _make_tiny_v4():
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
        index_topk=128,
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        initializer_range=0.2,
        tie_word_embeddings=False,
        dspark_block_size=BLOCK,
        dspark_target_layer_ids=[2, 3],
        dspark_noise_token_id=VOCAB - 1,
        dspark_markov_rank=8,
    )
    cfg.moe_force_xla_gmm = True
    cfg.fsdp_is_ep_bound = False
    cfg.attn_dtype = jnp.float32
    cfg.kvdtype = jnp.float32
    _, module_class = ed.get_modules_by_type("deepseek_v4", ed.TaskType.CAUSAL_LM)
    # HIGHEST: verify (multi-token) and decode (one token) windows must agree exactly for the greedy parity bar.
    model = module_class(
        config=cfg,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )
    model.eval()
    return model


def _build_dspark(model) -> DeepseekV4DSpark:
    with model.config.mesh:
        return DeepseekV4DSpark(
            model.config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
            num_stages=STAGES,
        )


class _OracleDrafter(DeepseekV4DSparkDrafter):
    """V4 DSpark drafter whose proposals are the target's own greedy continuation (every N-th window corrupted)."""

    def __init__(self, *args, sequences: dict[str, list[int]], corrupt_every: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequences = sequences
        self.corrupt_every = corrupt_every
        self.windows = 0

    def _oracle(self, req_id: str, position: int, seed_token: int, k: int, real: list[int]) -> list[int]:
        if not real:
            return []
        sequence = self.sequences[str(req_id)]
        if position + 1 >= len(sequence):
            return []
        assert int(seed_token) == sequence[position + 1], "seed is not the baseline token"
        oracle = list(sequence[position + 2 : position + 2 + int(k)])
        self.windows += 1
        if len(oracle) >= 2 and self.windows % self.corrupt_every == 0:
            oracle[1] = (oracle[1] + 1) % VOCAB
        return oracle

    def draft_window(self, **kwargs) -> list[int]:
        real = super().draft_window(**kwargs)
        return self._oracle(
            str(kwargs["req_id"]), int(kwargs["seed_position"]), int(kwargs["seed_token"]), int(kwargs["k"]), real
        )

    def draft_windows(self, hidden, entries) -> list[list[int]]:
        self.batched_calls = getattr(self, "batched_calls", 0) + 1
        reals = super().draft_windows(hidden, entries)
        return [
            self._oracle(str(rid), int(pos), int(tok), int(k), real)
            for (rid, _row, tok, pos, _off, _start, _n, k), real in zip(entries, reals, strict=True)
        ]


class _Harness:
    """Real runner + scheduler, optionally with a drafter."""

    def __init__(
        self,
        model,
        *,
        drafter=None,
        max_num_seqs: int = 2,
        min_token_pad: int = 16,
        min_input_pad: int = 1,
        use_aot_forward: bool = False,
    ):
        self.runner = eSurgeRunner(
            model=model,
            hbm_utilization=0.02,
            page_size=16,
            max_cache_tokens=2048,
            max_model_len=128,
            max_num_batched_tokens=64,
            min_input_pad=min_input_pad,
            min_token_pad=min_token_pad,
            max_num_seqs=max_num_seqs,
            max_num_seq_buckets=[max_num_seqs],
            async_scheduling=False,
            use_aot_forward=use_aot_forward,
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
def tiny_v4_model():
    return _make_tiny_v4()


@pytest.fixture(scope="module")
def baseline(tiny_v4_model) -> dict[str, list[int]]:
    harness = _Harness(tiny_v4_model)
    try:
        served = harness.run_waves()
    finally:
        harness.close()
    assert all(len(set(toks)) > 1 for toks in served.values()), served
    return served


def test_registry_builds_v4_dspark_drafter(tiny_v4_model):
    dspark = _build_dspark(tiny_v4_model)
    drafter = tiny_v4_model.drafter(method="deepseek_v4_dspark", drafter_model=dspark, num_draft_tokens=2)
    assert isinstance(drafter, DeepseekV4DSparkDrafter)
    assert drafter.num_draft_tokens == 2 and drafter.block_size == BLOCK
    assert drafter.dspark.num_stages == STAGES


def test_random_v4_dspark_drafter_matches_baseline(tiny_v4_model, baseline):
    """Real V4 DSpark proposals (almost all rejected): snapshot-restore-replay keeps the stream exact."""
    drafter = DeepseekV4DSparkDrafter(
        target_model=tiny_v4_model, dspark=_build_dspark(tiny_v4_model), num_draft_tokens=BLOCK
    )
    harness = _Harness(tiny_v4_model, drafter=drafter)
    try:
        served = harness.run_waves()
        runner = harness.runner
        generated = runner.spec_decode_num_drafts_generated
        accepted = runner.spec_decode_num_drafts_accepted
        assert generated > 0 and runner.spec_decode_num_verify_steps > 0, "speculation never ran"
        print(f"[v4 dspark random] generated={generated} accepted={accepted}")
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}


def test_oracle_v4_dspark_drafter_accepts_rejects_and_matches_baseline(tiny_v4_model, baseline):
    """Accepted, partially accepted and rejected windows all reproduce plain V4 serving."""
    prompts = {s["id"]: s["prompt_token_ids"] for wave in WAVES for s in wave}
    sequences = {rid: list(prompts[rid]) + baseline[rid] for rid in baseline}
    drafter = _OracleDrafter(
        target_model=tiny_v4_model,
        dspark=_build_dspark(tiny_v4_model),
        num_draft_tokens=BLOCK,
        sequences=sequences,
        corrupt_every=3,
    )
    harness = _Harness(tiny_v4_model, drafter=drafter)
    try:
        served = harness.run_waves()
        runner = harness.runner
        generated = runner.spec_decode_num_drafts_generated
        accepted = runner.spec_decode_num_drafts_accepted
        print(f"[v4 dspark oracle] generated={generated} accepted={accepted} windows={drafter.windows}")
        assert accepted > 0, "no draft was ever accepted"
        assert accepted < generated, "no draft was ever rejected"
        assert getattr(drafter, "batched_calls", 0) > 0, "the batched whole-block draft path never ran"
        assert runner.spec.num_replay_free_commits > 0, "rejected windows never took the replay-free rollback"
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}


def test_single_request_verify_in_slot_sized_bucket_matches_baseline(tiny_v4_model, baseline):
    """One request at a time on 8 slots with 8-token and 8-request bucket floors (eSurge's default
    ``min_input_pad=16`` pads a single request to every slot of an 8-slot engine the same way).

    Each verify window (1 + k tokens on ONE slot) then lands in an ``8 tokens / 8 reqs`` bucket, which the executor
    traces under the decode-layout scope. The packed path must still process every token of the window
    (regression: the one-token-per-slot fast path used to keep only one token per slot and drop the drafts).
    """
    prompts = {s["id"]: s["prompt_token_ids"] for wave in WAVES for s in wave}
    sequences = {rid: list(prompts[rid]) + baseline[rid] for rid in baseline}
    drafter = _OracleDrafter(
        target_model=tiny_v4_model,
        dspark=_build_dspark(tiny_v4_model),
        num_draft_tokens=BLOCK,
        sequences=sequences,
        corrupt_every=3,
    )
    # AOT like the real engine: the non-AOT jit retraces at run time outside the decode-layout scope, so only the
    # AOT executable is the one traced under it (a non-AOT run of this test passes even with the bug present).
    harness = _Harness(
        tiny_v4_model, drafter=drafter, max_num_seqs=8, min_token_pad=8, min_input_pad=8, use_aot_forward=True
    )
    try:
        assert min(harness.runner.num_tokens_paddings) <= 8, harness.runner.num_tokens_paddings
        served = {s["id"]: harness.generate_greedy([s])[0] for wave in WAVES for s in wave}
        runner = harness.runner
        generated = runner.spec_decode_num_drafts_generated
        accepted = runner.spec_decode_num_drafts_accepted
        print(
            f"[v4 dspark single-slot verify] generated={generated} accepted={accepted} "
            f"replay_free={runner.spec.num_replay_free_commits}"
        )
        assert 0 < accepted < generated, "both acceptance and rollback must be exercised"
        assert runner.spec.num_replay_free_commits > 0, "rejected windows never took the replay-free rollback"
    finally:
        harness.close()
    assert served == baseline, {k: (served[k], baseline[k]) for k in baseline if served[k] != baseline[k]}
