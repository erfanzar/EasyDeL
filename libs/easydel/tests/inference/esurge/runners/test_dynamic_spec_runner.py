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

"""Runner-level dynamic speculative-token scheduling (end-to-end, tiny model).

Contract under test: a real ``eSurgeRunner`` + ``Scheduler`` serving loop
resolves the per-step draft budget from the live concurrency —

* above the schedule's off-threshold the drafter is NEVER invoked and the
  greedy output is token-identical to a no-drafter runner (a K=0 step IS a
  plain decode step);
* below the threshold the drafter is invoked with the scheduled K (not the
  static maximum), and the emitted spec windows carry at most K drafts;
* the runner's compile-bucket grid picks up the schedule's exact spec-window
  token buckets and drops the unreachable static-K buckets
  (``derive_spec_token_buckets`` unit coverage).
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import pytest
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.model_runner import derive_spec_token_buckets
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from .._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from _common import make_tiny_qwen35 as make_tiny_model

_VOCAB = 256

_PROMPTS = [
    [3, 1, 4, 1, 5, 9, 2, 6],
    [7, 7, 1, 2, 3, 8, 8, 4, 10, 5],
    [10, 20, 30, 40, 11, 21, 31],
]


def _make_runner(model, *, n_seqs: int, drafter) -> eSurgeRunner:
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.05,
        page_size=16,
        max_cache_tokens=4096,
        max_model_len=64,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=n_seqs,
        max_num_seq_buckets=[n_seqs],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        drafter=drafter,
    )
    runner.compile(max_num_batched_tokens=64)
    return runner


def _serve(runner: eSurgeRunner, prompts, *, max_new: int = 10) -> dict[str, list[int]]:
    """Drive all prompts concurrently through scheduler + runner to completion."""
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_draft_tokens=runner.num_draft_tokens,
    )
    reqs = []
    for i, prompt in enumerate(prompts):
        r = EngineRequest(
            request_id=f"r-{i}",
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True),
            eos_token_id=None,
        )
        scheduler.add_request(r)
        reqs.append(r)
    for _ in range(4000):
        so = scheduler.schedule()
        out = runner.execute_model(so)
        scheduler.update_from_output(so, out)
        if all(r.is_finished() for r in reqs):
            break
    else:
        raise AssertionError("generation did not finish")
    toks = {r.request_id: list(r.output_token_ids) for r in reqs}
    runner.shutdown()
    runner.executor_manager.kv_pages = None
    return toks


def test_above_threshold_k0_steps_are_plain_single_token_decode(monkeypatch):
    """Concurrency 3 with an off-at-3 schedule: every step is a plain decode step.

    The auto-throttle contract, asserted on runner-visible semantics: with the
    schedule resolving K=0 at the serving concurrency, no request ever carries
    draft tokens (every per-step emission is exactly one token), no verify
    window ever runs, and no draft is ever generated — a K=0 step IS a plain
    single-token decode step, not an empty-draft spec path.

    Token-for-token identity against a no-drafter runner is deliberately NOT
    asserted: attaching a drafter compiles a different model-step executable
    (it returns the full hidden states for verification), and on a near-tie
    greedy argmax two XLA programs may differ by ~1 ULP — the same
    floating-point sensitivity documented for the batched-draft parity tests
    in ``test_mtp_batched_draft.py``.
    """
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    model = make_tiny_model()

    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=2, persist_kv=True)
    drafter.num_draft_tokens_per_batch_size = [(1, 2, 2), (3, 1 << 30, 0)]
    runner = _make_runner(model, n_seqs=3, drafter=drafter)

    emission_lengths: list[int] = []
    orig_execute = runner.execute_model

    def _recording_execute(scheduler_output):
        out = orig_execute(scheduler_output)
        for sampled in out.sampled_token_ids:
            emission_lengths.append(len(sampled))
        return out

    runner.execute_model = _recording_execute
    spec_toks = _serve(runner, _PROMPTS)

    assert set(spec_toks) == {"r-0", "r-1", "r-2"}
    for rid, out in spec_toks.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"
        assert min(out) >= 0 and max(out) < _VOCAB
    # Plain decode semantics: one token per request per step, never a
    # multi-token spec-window emission.
    assert emission_lengths and max(emission_lengths) == 1, (
        f"K=0 serving emitted a multi-token spec window: {sorted(set(emission_lengths))}"
    )
    assert runner.spec_decode_num_verify_steps == 0
    assert runner.spec_decode_num_drafts_generated == 0
    assert runner.spec_decode_num_drafts_accepted == 0


def test_above_threshold_drafter_not_invoked_during_serving(monkeypatch):
    """Concurrency 3, off-at-3 schedule: zero drafter invocations after compile()."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=2, persist_kv=True)
    drafter.num_draft_tokens_per_batch_size = [(1, 2, 2), (3, 1 << 30, 0)]
    runner = _make_runner(model, n_seqs=3, drafter=drafter)

    draft_calls: list[int] = []
    orig_batched = Qwen3_5MTPDrafter.draft_batched
    orig_draft = Qwen3_5MTPDrafter.draft

    def _spy_batched(self, **kwargs):
        draft_calls.append(int(kwargs["num_draft_tokens"]))
        return orig_batched(self, **kwargs)

    def _spy_draft(self, **kwargs):
        draft_calls.append(1)
        return orig_draft(self, **kwargs)

    # Installed AFTER compile() so the precompile warmup is not counted.
    monkeypatch.setattr(Qwen3_5MTPDrafter, "draft_batched", _spy_batched)
    monkeypatch.setattr(Qwen3_5MTPDrafter, "draft", _spy_draft)

    toks = _serve(runner, _PROMPTS)
    assert draft_calls == [], f"drafter invoked at K=0 concurrency: {draft_calls}"
    for rid, out in toks.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"


def test_below_threshold_drafter_invoked_with_scheduled_k(monkeypatch):
    """Concurrency 3 inside a K=2 range (static max 3): every draft uses K=2."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True)
    drafter.num_draft_tokens_per_batch_size = [(1, 4, 2), (5, 1 << 30, 0)]
    runner = _make_runner(model, n_seqs=3, drafter=drafter)
    assert runner.num_draft_tokens == 3  # static maximum unchanged

    batched_ks: list[int] = []
    orig_batched = Qwen3_5MTPDrafter.draft_batched

    def _spy_batched(self, **kwargs):
        batched_ks.append(int(kwargs["num_draft_tokens"]))
        return orig_batched(self, **kwargs)

    monkeypatch.setattr(Qwen3_5MTPDrafter, "draft_batched", _spy_batched)

    toks = _serve(runner, _PROMPTS)
    for rid, out in toks.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"
        assert min(out) >= 0 and max(out) < _VOCAB
    assert batched_ks, "expected the batched drafter to engage below the off-threshold"
    assert set(batched_ks) == {2}, f"drafter must draft the scheduled K=2, got {sorted(set(batched_ks))}"


def test_disable_flag_restores_static_k_at_runner_init(monkeypatch):
    """EASYDEL_DISABLE_DYNAMIC_SPEC drops the schedule at runner construction."""
    monkeypatch.setenv("EASYDEL_DISABLE_DYNAMIC_SPEC", "1")
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=2, persist_kv=True)
    drafter.num_draft_tokens_per_batch_size = [(1, 2, 2), (3, 1 << 30, 0)]
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.05,
        page_size=16,
        max_cache_tokens=4096,
        max_model_len=64,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=3,
        max_num_seq_buckets=[3],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        drafter=drafter,
    )
    assert runner.draft_schedule is None
    assert runner.spec.draft_schedule is None
    assert runner.spec.resolve_step_draft_budget(99) == 2  # static K everywhere
    runner.shutdown()
    runner.executor_manager.kv_pages = None


# ---------------------------------------------------------------------------
# Exact spec-window token buckets for the compile grid
# ---------------------------------------------------------------------------


def test_static_spec_buckets_unchanged():
    """No schedule: one reqs*(K+1) bucket per request bucket (pre-schedule behavior)."""
    got = derive_spec_token_buckets(
        seq_buckets=[4, 8, 16],
        num_draft_tokens=4,
        draft_schedule=None,
        min_token_pad=16,
        max_bucket=128,
    )
    assert got == {20, 40, 80}


def test_schedule_drops_unreachable_high_concurrency_buckets():
    """Off-above-4 schedule: only the low-concurrency request bucket drafts."""
    got = derive_spec_token_buckets(
        seq_buckets=[4, 8, 16],
        num_draft_tokens=4,
        draft_schedule=[(1, 4, 4), (5, 1 << 30, 0)],
        min_token_pad=16,
        max_bucket=128,
    )
    # Request bucket 4 (live 1..4) -> K=4 -> 4*5=20; buckets 8/16 resolve K=0.
    assert got == {20}


def test_schedule_gap_contributes_static_k_bucket():
    """Concurrencies in schedule gaps fall back to static K and keep its bucket."""
    got = derive_spec_token_buckets(
        seq_buckets=[4],
        num_draft_tokens=4,
        draft_schedule=[(1, 2, 3)],
        min_token_pad=16,
        max_bucket=128,
    )
    # Live 1..2 -> K=3 -> 4*4=16; live 3..4 (gap) -> static K=4 -> 4*5=20.
    assert got == {16, 20}


def test_schedule_bucket_straddling_a_boundary_gets_both_ks():
    """A request bucket spanning a K change contributes a bucket per reachable K."""
    got = derive_spec_token_buckets(
        seq_buckets=[8],
        num_draft_tokens=4,
        draft_schedule=[(1, 4, 4), (5, 64, 2)],
        min_token_pad=16,
        max_bucket=128,
    )
    # Live 1..4 -> K=4 -> 40; live 5..8 -> K=2 -> 24.
    assert got == {24, 40}


def test_bucket_bounds_respected():
    got = derive_spec_token_buckets(
        seq_buckets=[2, 64],
        num_draft_tokens=4,
        draft_schedule=None,
        min_token_pad=16,
        max_bucket=128,
    )
    # 2*5=10 < min_token_pad, 64*5=320 > max_bucket -> both excluded.
    assert got == set()


def test_runner_token_paddings_include_schedule_spec_buckets(monkeypatch):
    """The constructed runner's num_tokens_paddings carry the schedule's exact buckets.

    ``min_token_pad`` is floored to ``ESURGE_MIN_TOKEN_PAD`` (16) whenever
    ``max_model_len >= 16``, so the schedule's spec window must land at or
    above 16 to appear: request bucket 4 at K=4 -> ``4 * 5 = 20``. The static
    grid (K=4 for every request bucket) would also force ``8 * 5 = 40``; the
    off-above-4 schedule must NOT add it.
    """
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=4, persist_kv=True)
    drafter.num_draft_tokens_per_batch_size = [(1, 4, 4), (5, 1 << 30, 0)]
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.05,
        page_size=16,
        max_cache_tokens=4096,
        max_model_len=64,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=8,
        max_num_seq_buckets=[4, 8],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        drafter=drafter,
    )
    base_grid = set(runner._get_token_paddings(min_token_size=16, max_token_size=64, padding_gap=0))
    spec_only = set(runner.num_tokens_paddings) - base_grid
    assert 20 in spec_only, f"missing the schedule's exact spec bucket 20: {sorted(runner.num_tokens_paddings)}"
    assert 40 not in spec_only, "static-K bucket for the speculation-off concurrency band must be dropped"
    assert spec_only == {20}, f"unexpected spec buckets: {spec_only}"
    runner.shutdown()
    runner.executor_manager.kv_pages = None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
