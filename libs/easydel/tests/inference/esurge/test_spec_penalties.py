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

"""Greedy speculative decoding under presence / frequency / repetition penalties.

Contracts:

* the verify-path kernel (``runners/spec/penalties.py``) picks exactly the argmax the sampler's own
  ``apply_history_penalties_from_counts`` picks for the same count state, in float32 and bfloat16, including exact
  ties;
* the eligibility gate lets greedy penalised requests speculate and keeps sampled penalised / ``n > 1`` requests on
  the plain path;
* a runner serving with speculation emits the identical token stream to the non-speculative penalised greedy run,
  for a lone request, a pooled multi-request window, plain-sampler steps interleaved with verify steps (stop-loss
  backoff), and the exact recurrent replay path.
"""

from __future__ import annotations

import gc
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.inference.esurge.core.sampler import apply_history_penalties_from_counts
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.spec.penalties import (
    HistoryCountCache,
    HistoryCounts,
    VerifyPenaltyRows,
    history_penalty_params,
    penalized_verify_argmax,
)
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import DraftStep
from easydel.modules.llama import LlamaConfig, LlamaForCausalLM

try:
    from ._common import make_tiny_qwen35
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35

_VOCAB = 256
_PRESETS = [
    (1.5, 0.0, 1.0),
    (2.0, 0.0, 1.0),
    (0.0, 0.5, 1.0),
    (0.0, 0.0, 1.2),
    (0.0, 0.0, 0.8),
    (1.5, 0.3, 1.1),
    (-1.0, 0.0, 1.0),
]


def _dense_reference(logits: jax.Array, specs: list[VerifyPenaltyRows]) -> np.ndarray:
    """Penalise with dense per-row counts through the sampler's own function, then argmax."""
    rows, vocab = int(logits.shape[0]), int(logits.shape[1])
    counts = np.zeros((rows, vocab), dtype=np.uint32)
    active = np.zeros((rows,), dtype=np.bool_)
    presence = np.zeros((rows,), dtype=np.float32)
    frequency = np.zeros((rows,), dtype=np.float32)
    repetition = np.ones((rows,), dtype=np.float32)
    for spec in specs:
        for i in range(spec.num_rows):
            r = spec.offset + i
            for tok, cnt in zip(np.asarray(spec.history_ids), np.asarray(spec.history_counts), strict=True):
                counts[r, int(tok)] += int(cnt)
            for tok in spec.draft_tokens[: spec.draft_prefix + i]:
                counts[r, int(tok)] += 1
            active[r] = True
            presence[r] = spec.presence_penalty
            frequency[r] = spec.frequency_penalty
            repetition[r] = spec.repetition_penalty

    @jax.jit
    def _reference(lg, cn, act, pres, freq, rep):
        penalized = apply_history_penalties_from_counts(
            lg,
            token_counts=cn,
            active_mask=act,
            presence_penalties=pres,
            frequency_penalties=freq,
            repetition_penalties=rep,
        )
        return jnp.argmax(penalized, axis=-1)

    return np.asarray(_reference(logits, counts, active, presence, frequency, repetition)).astype(np.int64)


def _history(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist = HistoryCounts(capacity=4)
    hist.extend(int(t) for t in tokens)
    return hist.arrays()


def _quantized_logits(rng: np.random.Generator, rows: int, vocab: int, dtype) -> jax.Array:
    # A coarse grid makes exact ties common, both before and after a penalty shift: the tie-breaking (lowest index)
    # must match the dense argmax.
    values = np.round(rng.normal(0.0, 2.0, size=(rows, vocab)) * 4.0) / 4.0
    return jnp.asarray(values.astype(np.float32)).astype(dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize(("presence", "frequency", "repetition"), _PRESETS)
def test_kernel_matches_sampler_penalty(dtype, presence, frequency, repetition):
    rng = np.random.default_rng(0xC0FFEE)
    vocab, k = 512, 3
    logits = _quantized_logits(rng, 3 * (k + 1) + 2, vocab, dtype)
    specs = []
    for group in range(3):
        # Small id range so histories repeat tokens and drafts hit history ids.
        hist_ids, hist_counts = _history(rng.integers(0, 48, size=(20 + 9 * group,)))
        specs.append(
            VerifyPenaltyRows(
                offset=group * (k + 1),
                num_rows=k + 1,
                history_ids=hist_ids,
                history_counts=hist_counts,
                draft_tokens=tuple(int(t) for t in rng.integers(0, 48, size=(k,))),
                presence_penalty=presence,
                frequency_penalty=frequency,
                repetition_penalty=repetition,
            )
        )
    got = penalized_verify_argmax(logits, specs)
    np.testing.assert_array_equal(got, _dense_reference(logits, specs))
    # The two trailing rows are not covered and keep their plain argmax.
    np.testing.assert_array_equal(got[-2:], np.asarray(jnp.argmax(logits[-2:], axis=-1)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["f32", "bf16"])
def test_kernel_mixed_penalties_and_draft_prefix(dtype):
    rng = np.random.default_rng(7)
    vocab = 300
    logits = _quantized_logits(rng, 9, vocab, dtype)
    specs = []
    for group, (offset, num_rows, prefix, params) in enumerate(
        [(0, 2, 0, (2.0, 0.0, 1.0)), (3, 1, 2, (0.0, 0.4, 1.3)), (5, 4, 0, (0.5, 0.1, 0.9))]
    ):
        hist_ids, hist_counts = _history(rng.integers(0, 32, size=(10 + group,)))
        specs.append(
            VerifyPenaltyRows(
                offset=offset,
                num_rows=num_rows,
                history_ids=hist_ids,
                history_counts=hist_counts,
                draft_tokens=tuple(int(t) for t in rng.integers(0, 32, size=(4,))),
                presence_penalty=params[0],
                frequency_penalty=params[1],
                repetition_penalty=params[2],
                draft_prefix=prefix,
            )
        )
    np.testing.assert_array_equal(penalized_verify_argmax(logits, specs), _dense_reference(logits, specs))


def test_penalty_moves_the_pick_and_drafts_count_per_row():
    logits = np.full((2, 64), -5.0, dtype=np.float32)
    logits[:, 10] = 3.0
    logits[:, 20] = 2.5
    logits = jnp.asarray(logits)

    def rows(history: list[int], drafts: tuple[int, ...], presence: float) -> VerifyPenaltyRows:
        ids, counts = _history(np.asarray(history))
        return VerifyPenaltyRows(0, 2, ids, counts, drafts, presence, 0.0, 1.0)

    # Token 10 is repeated: 3.0 - 2.0 < 2.5 on both rows (the draft 30 sits at -5 and never matters).
    assert penalized_verify_argmax(logits, [rows([10], (30,), 2.0)]).tolist() == [20, 20]
    # A penalty too small to close the gap leaves the pick alone.
    assert penalized_verify_argmax(logits, [rows([10], (30,), 0.25)]).tolist() == [10, 10]
    # Token 10 only appears as the first draft: row 0 has not seen it, row 1 has.
    assert penalized_verify_argmax(logits, [rows([1], (10,), 2.0)]).tolist() == [10, 20]


@pytest.mark.cpu_ok
def test_kernel_rejects_bad_layouts():
    logits = np.zeros((4, 8), dtype=np.float32)
    ids, counts = np.asarray([1]), np.asarray([1])
    with pytest.raises(ValueError, match="rows, vocab"):
        penalized_verify_argmax(np.zeros((4,), dtype=np.float32), [])
    with pytest.raises(ValueError, match="fall outside"):
        penalized_verify_argmax(logits, [VerifyPenaltyRows(3, 2, ids, counts, (1,), 1.0, 0.0, 1.0)])
    with pytest.raises(ValueError, match="overlap"):
        penalized_verify_argmax(
            logits,
            [
                VerifyPenaltyRows(0, 2, ids, counts, (1,), 1.0, 0.0, 1.0),
                VerifyPenaltyRows(1, 2, ids, counts, (1,), 1.0, 0.0, 1.0),
            ],
        )
    with pytest.raises(ValueError, match="more draft tokens"):
        penalized_verify_argmax(logits, [VerifyPenaltyRows(0, 3, ids, counts, (1,), 1.0, 0.0, 1.0)])


@pytest.mark.cpu_ok
def test_history_counts_match_unique_and_skip_padding():
    rng = np.random.default_rng(3)
    tokens = rng.integers(-1, 40, size=(300,))
    ids, counts = _history(tokens)
    valid = tokens[tokens >= 0]
    want_ids, want_counts = np.unique(valid, return_counts=True)
    order = np.argsort(ids)
    np.testing.assert_array_equal(ids[order], want_ids)
    np.testing.assert_array_equal(counts[order], want_counts)


@pytest.mark.cpu_ok
def test_history_cache_is_incremental_and_restarts_on_shorter_history():
    seq = [5, 5, 7, 9, 9, 9, 3, 7]
    reads: list[int] = []

    def token_at(pos: int) -> int:
        reads.append(pos)
        return seq[pos]

    cache = HistoryCountCache()
    ids, counts = cache.counts("r", token_at, 6)
    assert dict(zip(ids.tolist(), counts.tolist(), strict=True)) == {5: 2, 7: 1, 9: 3}
    reads.clear()
    ids, counts = cache.counts("r", token_at, 8)
    assert reads == [6, 7], "only the new positions are read"
    assert dict(zip(ids.tolist(), counts.tolist(), strict=True)) == {5: 2, 7: 2, 9: 3, 3: 1}
    reads.clear()
    ids, counts = cache.counts("r", token_at, 3)
    assert reads == [0, 1, 2]
    assert dict(zip(ids.tolist(), counts.tolist(), strict=True)) == {5: 2, 7: 1}
    cache.drop("r")
    assert "r" not in cache


class _Strategy(DrafterSpeculation):
    """The real strategy methods over a minimal runner, without building a drafter-backed runner."""

    def __init__(self, runner=None, num_draft_tokens: int = 2):
        self._runner = runner
        self.drafter = object()
        self.num_draft_tokens = int(num_draft_tokens)
        self._history_counts = HistoryCountCache()
        self._penalty_counts_stale: set[str] = set()
        self.num_penalized_verify_rows = 0


class _Req:
    """Request state exposing the committed-token surface the strategy reads."""

    def __init__(self, prompt, output, **sampling):
        self.prompt_token_ids = list(prompt)
        self.output_token_ids = list(output)
        sampling.setdefault("temperature", 0.0)
        self.sampling_params = types.SimpleNamespace(n=sampling.pop("n", 1), **sampling)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_token_id(self, idx: int) -> int:
        full = self.prompt_token_ids + self.output_token_ids
        return full[idx]


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("params", "eligible"),
    [
        ({"presence_penalty": 1.5}, True),
        ({"presence_penalty": 2.0}, True),
        ({"presence_penalty": -0.5}, True),
        ({"frequency_penalty": 0.4}, True),
        ({"repetition_penalty": 1.15}, True),
        ({"presence_penalty": 1.5, "temperature": 0.8}, False),
        ({"frequency_penalty": 0.4, "temperature": 0.8}, False),
        ({"repetition_penalty": 1.15, "temperature": 0.8}, False),
        ({"temperature": 0.9}, True),
        ({}, True),
        ({"n": 2}, False),
    ],
)
def test_spec_gate(params, eligible):
    assert _Strategy().is_spec_request(_Req([1], [], **params)) is eligible


@pytest.mark.cpu_ok
def test_history_penalty_predicate():
    sp = types.SimpleNamespace
    assert history_penalty_params(sp(presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0)) is None
    assert history_penalty_params(sp(presence_penalty=1.5, frequency_penalty=0.0, repetition_penalty=1.0)) == (
        1.5,
        0.0,
        1.0,
    )
    assert history_penalty_params(sp(repetition_penalty=0.9)) == (0.0, 0.0, 0.9)
    assert DrafterSpeculation.uses_history_penalties(None) is False


@pytest.mark.cpu_ok
def test_verify_penalty_rows_use_committed_history_and_drafts():
    req = _Req([9, 9, 4], [7, 7, 8], presence_penalty=1.5)
    meta = types.SimpleNamespace(request_id="r", start_pos=4, real_count=1, buffer_draft_tokens=[8, 3])
    spec = _Strategy().verify_penalty_rows(meta=meta, req_state=req, offset=6, num_rows=3)
    assert spec is not None
    assert dict(zip(spec.history_ids.tolist(), spec.history_counts.tolist(), strict=True)) == {9: 2, 4: 1, 7: 2}
    assert spec.draft_tokens == (8, 3) and spec.offset == 6 and spec.num_rows == 3
    assert (spec.presence_penalty, spec.frequency_penalty, spec.repetition_penalty) == (1.5, 0.0, 1.0)
    plain = _Req([9], [7])
    assert _Strategy().verify_penalty_rows(meta=meta, req_state=plain, offset=0, num_rows=1) is None


@pytest.mark.cpu_ok
def test_sync_penalty_counts_rewrites_only_drifted_rows():
    writes: list[list[tuple[int, dict[int, int]]]] = []

    class _Manager:
        def overwrite_sampler_penalty_rows(self, rows):
            writes.append([(row, dict(zip(ids.tolist(), cnt.tolist(), strict=True))) for row, ids, cnt in rows])

    reqs = {
        "pen": _Req([1, 2], [2], presence_penalty=2.0),
        "plain": _Req([1, 2], [3]),
        "sampled": _Req([1, 2], [3], presence_penalty=2.0, temperature=0.7),
        "fresh": _Req([5, 5], [], presence_penalty=1.0),
    }
    runner = types.SimpleNamespace(
        requests=reqs,
        sequence_buffer=types.SimpleNamespace(req_id_to_index={"pen": 0, "plain": 1, "sampled": 2, "fresh": 3}),
        executor_manager=_Manager(),
    )
    strategy = _Strategy(runner)

    # Step 1: "pen" and "plain" verify drafts; "sampled" is never spec-eligible; "fresh" runs its first plain step.
    strategy.sync_penalty_counts(
        types.SimpleNamespace(
            num_scheduled_tokens={"pen": 3, "plain": 3, "sampled": 1, "fresh": 2},
            scheduled_spec_decode_tokens={"pen": [4, 4], "plain": [4, 4]},
        )
    )
    assert writes == [], "no row has drifted before a verify step"

    # The verify step emitted two more tokens for "pen"; it now takes the plain sampler path.
    reqs["pen"].output_token_ids.extend([4, 6])
    strategy.sync_penalty_counts(
        types.SimpleNamespace(
            num_scheduled_tokens={"pen": 1, "plain": 1, "sampled": 1},
            scheduled_spec_decode_tokens={},
        )
    )
    assert writes == [[(0, {1: 1, 2: 2, 4: 1, 6: 1})]]


_PROMPT_A = [3, 1, 4, 1, 5, 9]
_PROMPT_B = [7, 7, 1, 2, 3, 8, 8, 4]


class _PositionOracleDrafter:
    """Proposes a known reference stream by position, deliberately wrong at every ``wrong_every``-th position.

    A draft call at position ``p`` (the seed hidden position) proposes the token at ``p + 2``.
    """

    def __init__(self, reference: list[int], *, num_draft_tokens: int, wrong_every: int):
        self.reference = list(reference)
        self.num_draft_tokens = int(num_draft_tokens)
        self.wrong_every = int(wrong_every)

    def reset(self, batch_size: int) -> None:
        del batch_size

    def draft(self, input_ids, target_hidden_states=None, position_ids=None, sample=False, rng_key=None, **kwargs):
        del input_ids, target_hidden_states, rng_key, kwargs
        if sample:
            raise AssertionError("greedy-only test drafter")
        target = int(np.asarray(position_ids).reshape(-1)[-1]) + 2
        token = self.reference[target] if 0 <= target < len(self.reference) else 0
        if target % self.wrong_every == 0:
            token = (token + 1) % _VOCAB
        token_ids = jnp.asarray([token], dtype=jnp.int32)
        log_probs = jax.nn.log_softmax(jax.nn.one_hot(token_ids, _VOCAB, dtype=jnp.float32) * 30.0, axis=-1)
        return DraftStep(token_ids=token_ids, log_probs=jnp.zeros((1,), dtype=jnp.float32), full_log_probs=log_probs)


def _tiny_llama() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
    )
    # Exact-token parity compares differently shaped LM-head projections; HIGHEST keeps TPU matmuls from flipping
    # near-tied logits independently of the penalty logic under test.
    return LlamaForCausalLM(
        config=cfg,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


def _serve(model, requests: dict[str, tuple[list[int], dict]], *, drafter=None, backoff: int = 0, max_new: int = 16):
    n = len(requests)
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=0.05,
        page_size=16,
        max_cache_tokens=4096,
        max_model_len=64,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=n,
        max_num_seq_buckets=[n],
        async_scheduling=False,
        use_aot_forward=False,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
        drafter=drafter,
    )
    if backoff:
        runner.spec_decode_reject_backoff_steps = backoff
    runner.compile(max_num_batched_tokens=64)
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        async_scheduling=False,
        num_draft_tokens=runner.num_draft_tokens,
    )
    engine_requests = []
    for rid, (prompt, penalties) in requests.items():
        request = EngineRequest(
            request_id=rid,
            prompt_token_ids=list(prompt),
            sampling_params=SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True, **penalties),
            eos_token_id=None,
        )
        scheduler.add_request(request)
        engine_requests.append(request)
    for _ in range(400):
        scheduler_output = scheduler.schedule()
        output = runner.execute_model(scheduler_output)
        scheduler.update_from_output(scheduler_output, output)
        if all(r.is_finished() for r in engine_requests):
            break
    else:
        raise AssertionError("generation did not finish")
    tokens = {r.request_id: [int(t) for t in r.output_token_ids] for r in engine_requests}
    stats = {
        "accepted": int(runner.spec_decode_num_drafts_accepted),
        "generated": int(runner.spec_decode_num_drafts_generated),
        "penalized_rows": int(getattr(runner.spec, "num_penalized_verify_rows", 0)),
    }
    runner.shutdown()
    runner.executor_manager.kv_pages = None
    del runner
    gc.collect()
    return tokens, stats


def _assert_spec_matches(model, requests, *, reference_rid, num_draft_tokens, wrong_every, backoff=0):
    baseline, _ = _serve(model, requests)
    reference = list(requests[reference_rid][0]) + baseline[reference_rid]
    drafter = _PositionOracleDrafter(reference, num_draft_tokens=num_draft_tokens, wrong_every=wrong_every)
    spec, stats = _serve(model, requests, drafter=drafter, backoff=backoff)
    for rid in requests:
        assert spec[rid] == baseline[rid], (
            f"{rid}: speculative stream diverged from penalised greedy\n  plain={baseline[rid]}\n  spec ={spec[rid]}"
        )
    assert stats["accepted"] > 0, f"no draft was accepted: {stats}"
    assert stats["generated"] > stats["accepted"], f"no draft was rejected: {stats}"
    assert stats["penalized_rows"] > 0, f"penalised verify path never engaged: {stats}"
    return baseline


def test_lone_penalised_request_speculates_losslessly():
    """Batch size 1: the per-request verify path must penalise too."""
    _assert_spec_matches(
        _tiny_llama(),
        {"pen": (_PROMPT_A, {"presence_penalty": 2.0})},
        reference_rid="pen",
        num_draft_tokens=2,
        wrong_every=3,
    )


def test_pooled_window_mixed_penalties_speculates_losslessly():
    """Three requests verified in one pooled projection: penalised, unpenalised, frequency + repetition."""
    baseline = _assert_spec_matches(
        _tiny_llama(),
        {
            "pen": (_PROMPT_A, {"presence_penalty": 1.5}),
            "plain": (_PROMPT_A, {}),
            "freq_rep": (_PROMPT_B, {"frequency_penalty": 0.5, "repetition_penalty": 1.3}),
        },
        reference_rid="pen",
        num_draft_tokens=2,
        wrong_every=4,
    )
    assert baseline["pen"] != baseline["plain"], "the penalty must change the greedy stream for this test to bite"


def test_plain_steps_between_verify_steps_keep_exact_counts():
    """Stop-loss backoff interleaves plain sampler steps with verify steps; the sampler's counts must stay exact."""
    _assert_spec_matches(
        _tiny_llama(),
        {"pen": (_PROMPT_A, {"presence_penalty": 2.0, "frequency_penalty": 0.3})},
        reference_rid="pen",
        num_draft_tokens=1,
        wrong_every=2,
        backoff=2,
    )


def test_exact_recurrent_replay_path_speculates_losslessly(monkeypatch):
    """Hybrid recurrent model with exact replay: the corrected token comes from the replay forward."""
    monkeypatch.setenv("EASYDEL_SPEC_RECURRENT_REPLAY", "1")
    _assert_spec_matches(
        make_tiny_qwen35(mtp_layers=1, precision=jax.lax.Precision.HIGHEST),
        {"pen": (_PROMPT_A, {"presence_penalty": 2.0})},
        reference_rid="pen",
        num_draft_tokens=2,
        wrong_every=3,
    )
