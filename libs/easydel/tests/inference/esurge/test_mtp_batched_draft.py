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

"""Batched inline-MTP drafting parity (fixes the per-request batch-1 collapse).

The eSurge speculative strategy used to draft ONE request at a time
(``DrafterSpeculation.draft_next`` -> ``Qwen3_5MTPDrafter.draft`` with a batch-1,
row-targeted forward), so a running batch of ``N`` requests cost ``N * k`` tiny
batch-1 MTP forwards per decode step. ``draft_next_batched`` /
``Qwen3_5MTPDrafter.draft_batched`` draft the whole batch in ONE pooled pass
(``k`` batched forwards).

These CPU/TPU tests assert the batched pass is drafting-EQUIVALENT to the
per-request path it replaces. Exact-token reference cases request HIGHEST
matmul precision locally: float32 storage alone still permits TPU bf16
multipliers, whose batch-dependent rounding can flip a near-tie argmax.
The full-runner integration test retains the default precision:

* the drafted tokens for a batch of requests EQUAL drafting each request
  individually (both first window and a second window that exercises the
  per-row EAGLE rollback), and
* idle pool rows (requests not drafted this step) are left completely untouched
  by the batched forward — their MTP cache write index does not move.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_core_for_runner, served_request
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, served_request
    from _common import make_tiny_qwen35 as make_tiny_model

_VOCAB = 256
_HIDDEN = 128


def _seed_hidden(i: int) -> jnp.ndarray:
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(11), i), (_HIDDEN,), dtype=jnp.float32)


def _make_strategy(drafter: Qwen3_5MTPDrafter, n_rows: int, k: int) -> DrafterSpeculation:
    runner = types.SimpleNamespace(max_num_reqs=n_rows, max_num_seqs=n_rows)
    return DrafterSpeculation(runner=runner, drafter=drafter, num_draft_tokens=k)


# Greedy (temperature 0) so drafting is deterministic argmax and spec-eligible.
_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=32))


def _row_index(drafter: Qwen3_5MTPDrafter, row: int) -> int:
    return int(np.asarray(drafter._mtp_cache.views[0].indexes).reshape(-1)[row])


def _reference_drafts(model, reqs, n, k):
    """Per-request ``draft_next`` on a shared pool drafter (the old path)."""
    dref = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sref = _make_strategy(dref, n, k)
    out = {}
    for rid, row, tok, pos, hidden in reqs:
        out[rid] = sref.draft_next(
            req_id=rid,
            row_pos=row,
            seed_token=tok,
            seed_position=pos,
            seed_hidden=hidden,
            req_state=_REQ_STATE,
        )
    return out, dref


def test_batched_draft_equals_per_request_first_window():
    """Batched drafts for B=4 requests EQUAL drafting each request individually."""
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    n, k = 4, 3
    reqs = [
        ("A", 0, 5, 4, _seed_hidden(0)),
        ("B", 1, 9, 10, _seed_hidden(1)),
        ("C", 2, 13, 7, _seed_hidden(2)),
        ("D", 3, 21, 2, _seed_hidden(3)),
    ]
    ref, _ = _reference_drafts(model, reqs, n, k)

    dbat = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sbat = _make_strategy(dbat, n, k)
    got = sbat.draft_next_batched([(rid, row, tok, pos, h) for rid, row, tok, pos, h in reqs])

    for rid, _row, _tok, _pos, _h in reqs:
        assert len(got[rid]) == k
        assert got[rid] == ref[rid], f"{rid}: batched {got[rid]} != per-request {ref[rid]}"


def test_batched_draft_equals_per_request_two_windows():
    """Parity holds across a second window that exercises per-row EAGLE rollback."""
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    n, k = 4, 2
    # Window 1 seeds (base positions) and window 2 seeds (advanced by accepted+1).
    w1 = [
        ("A", 0, 5, 4, _seed_hidden(0)),
        ("B", 1, 9, 10, _seed_hidden(1)),
        ("C", 2, 13, 7, _seed_hidden(2)),
        ("D", 3, 21, 2, _seed_hidden(3)),
    ]
    w2 = [
        ("A", 0, 31, 6, _seed_hidden(10)),
        ("B", 1, 33, 12, _seed_hidden(11)),
        ("C", 2, 37, 9, _seed_hidden(12)),
        ("D", 3, 41, 4, _seed_hidden(13)),
    ]

    # Reference: per-request across both windows on one shared drafter.
    dref = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sref = _make_strategy(dref, n, k)
    for rid, row, tok, pos, h in w1:
        sref.draft_next(req_id=rid, row_pos=row, seed_token=tok, seed_position=pos, seed_hidden=h, req_state=_REQ_STATE)
    ref_w2 = {}
    for rid, row, tok, pos, h in w2:
        ref_w2[rid] = sref.draft_next(
            req_id=rid, row_pos=row, seed_token=tok, seed_position=pos, seed_hidden=h, req_state=_REQ_STATE
        )

    # Batched: same two windows on a fresh drafter.
    dbat = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sbat = _make_strategy(dbat, n, k)
    sbat.draft_next_batched([(rid, row, tok, pos, h) for rid, row, tok, pos, h in w1])
    got_w2 = sbat.draft_next_batched([(rid, row, tok, pos, h) for rid, row, tok, pos, h in w2])

    for rid, _row, _tok, _pos, _h in w2:
        assert got_w2[rid] == ref_w2[rid], f"{rid} w2: batched {got_w2[rid]} != per-request {ref_w2[rid]}"

    # One compiled batched program (N fixed); rollback targets are traced, not keyed.
    assert len(dbat._jit_batched) == 1, "batched draft must not recompile across windows"


def test_batched_draft_leaves_idle_rows_untouched():
    """A row not drafted this step keeps its MTP cache write index (no corruption)."""
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    n, k = 4, 3
    # Only rows 0 and 2 are collected; rows 1 and 3 are idle this step.
    collected = [
        ("A", 0, 5, 4, _seed_hidden(0)),
        ("C", 2, 13, 7, _seed_hidden(2)),
    ]

    # Reference (per-request) for the collected rows only.
    ref, _ = _reference_drafts(model, collected, n, k)

    dbat = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sbat = _make_strategy(dbat, n, k)
    got = sbat.draft_next_batched([(rid, row, tok, pos, h) for rid, row, tok, pos, h in collected])

    for rid, _row, _tok, _pos, _h in collected:
        assert got[rid] == ref[rid], f"{rid}: batched {got[rid]} != per-request {ref[rid]}"

    # Collected requests' (lifetime-stable, strategy-allocated) rows advanced by
    # k; every other row must remain at write index 0.
    row_a = sbat._mtp_row_by_req["A"]
    row_c = sbat._mtp_row_by_req["C"]
    assert _row_index(dbat, row_a) == k
    assert _row_index(dbat, row_c) == k
    for idle in set(range(n)) - {row_a, row_c}:
        assert _row_index(dbat, idle) == 0, f"idle row {idle} must be untouched by the batched forward"


def test_batched_draft_different_committed_lens_per_row():
    """Batched == per-request when each row rolls back to a DIFFERENT committed length.

    The concurrent-serving case the earlier two-window test did not cover: requests
    accept different numbers of drafts per window, so their per-row EAGLE rollback
    targets differ within a single batched call (row A -> 2, B -> 3, C -> 1). This
    exercises the batched forward's per-row ``indexes`` handling — a batched decode
    that used one uniform cache length would corrupt the rows whose committed length
    differs from the first row's.
    """
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    n, k = 3, 3
    w1 = [("A", 0, 5, 4, _seed_hidden(0)), ("B", 1, 9, 10, _seed_hidden(1)), ("C", 2, 13, 7, _seed_hidden(2))]
    # Advance each row by a different amount -> committed A=2, B=3, C=1.
    w2 = [("A", 0, 31, 6, _seed_hidden(10)), ("B", 1, 33, 13, _seed_hidden(11)), ("C", 2, 37, 8, _seed_hidden(12))]

    dref = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sref = _make_strategy(dref, n, k)
    for rid, row, tok, pos, h in w1:
        sref.draft_next(req_id=rid, row_pos=row, seed_token=tok, seed_position=pos, seed_hidden=h, req_state=_REQ_STATE)
    ref_w2 = {
        rid: sref.draft_next(
            req_id=rid, row_pos=row, seed_token=tok, seed_position=pos, seed_hidden=h, req_state=_REQ_STATE
        )
        for rid, row, tok, pos, h in w2
    }

    dbat = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    sbat = _make_strategy(dbat, n, k)
    sbat.draft_next_batched([(rid, row, tok, pos, h) for rid, row, tok, pos, h in w1])
    got_w2 = sbat.draft_next_batched([(rid, row, tok, pos, h) for rid, row, tok, pos, h in w2])
    for rid in ref_w2:
        assert got_w2[rid] == ref_w2[rid], f"{rid}: batched {got_w2[rid]} != per-request {ref_w2[rid]}"


def test_batched_draft_window_from_host_bounds_only():
    """The drafter-cache window comes from host arrays only, and never changes the drafts.

    Host ``committed_lens`` run the narrowest window that holds every collected row; a device ``committed_lens``
    runs the whole capacity, because reading it would block the host. Both draft the same tokens.
    """
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    n, k, capacity = 4, 3, 1024
    seed_tokens = jnp.asarray([5, 9, 13, 21], dtype=jnp.int32)
    seed_hidden = jnp.stack([_seed_hidden(i) for i in range(n)])[:, None, :]
    seed_positions = jnp.asarray([4, 10, 7, 2], dtype=jnp.int32)
    mask = np.ones((n,), dtype=bool)

    def run(committed):
        drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
        drafter.set_max_length(capacity)
        drafts = drafter.draft_batched(
            seed_tokens=seed_tokens,
            seed_hidden=seed_hidden,
            seed_positions=seed_positions,
            committed_lens=committed,
            collected_mask=mask,
            num_draft_tokens=k,
        )
        assert drafter.mtp_cache_capacity() == capacity
        assert [_row_index(drafter, r) for r in range(n)] == [k] * n
        (key,) = drafter._jit_batched
        return np.asarray(drafts), key[-1]

    host, host_width = run(np.zeros((n,), dtype=np.int32))
    full, full_width = run(jnp.zeros((n,), dtype=jnp.int32))
    assert (host_width, full_width) == (256, None)
    np.testing.assert_array_equal(host, full)


def test_drafter_programs_return_the_cache_placed_as_allocated():
    """The batched and fused drafter programs hand the drafter cache back placed exactly as it was allocated.

    Each drafter program is compiled for the placement of the cache it receives: a program that returned the cache
    placed differently made the next drafter program compile again in serving.
    """
    model = make_tiny_model()
    n, k = 4, 3
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True, cache_length=64)
    drafter.ensure_batch_size(n)

    def placement():
        return [leaf.sharding for leaf in jax.tree.leaves(drafter._mtp_cache)]

    allocated = placement()
    assert allocated and all(isinstance(s, jax.sharding.NamedSharding) for s in allocated), allocated
    drafter.draft_batched(
        seed_tokens=np.asarray([5, 9, 13, 21], dtype=np.int32),
        seed_hidden=jnp.stack([_seed_hidden(i) for i in range(n)])[:, None, :],
        seed_positions=np.asarray([4, 10, 7, 2], dtype=np.int32),
        committed_lens=np.zeros((n,), dtype=np.int32),
        collected_mask=np.ones((n,), dtype=bool),
        num_draft_tokens=k,
    )
    assert placement() == allocated, "the batched draft returned the drafter cache placed differently"
    drafter.verify_draft_batched(
        gathered_hidden=jnp.stack([_seed_hidden(i) for i in range(n * (k + 1))]),
        row_offsets=np.arange(n, dtype=np.int32) * (k + 1),
        draft_tokens=np.zeros((n, k), dtype=np.int32),
        draft_counts=np.full((n,), k, dtype=np.int32),
        seed_pos_base=np.asarray([5, 11, 8, 3], dtype=np.int32),
        committed_base=np.full((n,), k, dtype=np.int32),
        fresh_mask=np.zeros((n,), dtype=bool),
        collected_mask=np.ones((n,), dtype=bool),
        num_draft_tokens=k,
    )
    assert placement() == allocated, "the fused verify+draft returned the drafter cache placed differently"


def test_full_drafter_row_restarts_instead_of_starving():
    """A persisted drafter row that the next window would overflow restarts from that window.

    Such a row used to stop drafting for good (a 1,024-token prompt in a 1,024-slot drafter cache never drafted).
    The restarted row holds only this window's slots and drafts what a fresh row drafts from the same seed.
    """
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    n, k, capacity = 2, 3, 8

    def strategy():
        drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True, cache_length=capacity)
        return _make_strategy(drafter, n, k), drafter

    spec, drafter = strategy()
    # Two windows that accept every draft: the row then holds 2 * k of its 8 slots.
    for window, pos in enumerate((4, 4 + k + 1)):
        spec.draft_next_batched([("A", 0, 5 + window, pos, _seed_hidden(window))])
    row = spec._mtp_row_by_req["A"]
    assert _row_index(drafter, row) == 2 * k
    seed = ("A", 0, 9, 4 + 2 * (k + 1), _seed_hidden(7))
    got = spec.draft_next_batched([seed])["A"]
    assert _row_index(drafter, row) == k, "the full row must restart from this window"
    fresh, _ = strategy()
    assert len(got) == k
    assert got == fresh.draft_next_batched([seed])["A"]


def test_per_request_draft_restarts_a_full_row():
    """The per-request draft restarts a full row too: a request whose prompt does not fit the drafter cache drafts
    from its next window on, as a fresh row does."""
    model = make_tiny_model(precision=jax.lax.Precision.HIGHEST)
    k, capacity, prompt = 2, 8, 10

    def strategy():
        drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True, cache_length=capacity)
        return _make_strategy(drafter, 1, k)

    spec = strategy()
    first = spec.draft_next(
        req_id="A",
        row_pos=0,
        seed_token=7,
        seed_position=prompt - 1,
        seed_hidden=_seed_hidden(20),
        req_state=_REQ_STATE,
        prefix_input_ids=jnp.arange(1, prompt + 1, dtype=jnp.int32)[None, :],
        prefix_hidden_states=jnp.stack([_seed_hidden(i) for i in range(prompt)])[None],
        prefix_position_ids=jnp.arange(prompt, dtype=jnp.int32)[None, :],
    )
    assert first == [], "a prefix longer than the drafter cache cannot be drafted"
    window = dict(
        req_id="A", row_pos=0, seed_token=8, seed_position=prompt, seed_hidden=_seed_hidden(21), req_state=_REQ_STATE
    )
    got = spec.draft_next(**window)
    assert len(got) == k
    assert got == strategy().draft_next(**window)


# --- Full-runner multi-request integration (collect + >=2 gate + scatter) ---
#
# A full-runner token-for-token batched-vs-per-request comparison is NOT a valid
# bit-exact test for >= 2 requests: XLA's batched matmul over [B, ...] differs from
# B separate [1, ...] matmuls by ~1 ULP (measured 1.19e-7 on this model), so on a
# near-tie greedy argmax the two land on opposite sides — the same floating-point
# sensitivity any batched-decode engine (vLLM / SGLang) has. Bit-identity IS
# guaranteed for a SINGLE request under exact recurrent replay: the runner then only
# engages the pooled forward when >= 2 spec-active requests are verified together
# (``eSurgeRunner`` uses the ``step_allows_batch`` gate), so one request drafts
# in-loop through the exact per-request path — that bit-identity is asserted by
# ``test_mtp_persist_kv.test_runner_greedy_output_unchanged_with_persist``. Here we
# assert the multi-request runner INTEGRATION instead of near-tie-sensitive tokens:
# the pooled batched draft actually engages for B >= 2 and every request's drafts
# are scattered back to it. The per-row draft-value equivalence is covered
# deterministically by ``test_batched_draft_different_committed_lens_per_row``.


def test_batched_runner_engages_and_scatters_for_multi_request():
    """B=3 through a real eSurgeRunner: the >=2 gate engages the pooled batched draft
    and every request's drafts are scattered back to it (collect/scatter integration).
    """
    from easydel.inference.esurge.runners.spec import strategy as _strategy

    seed_counts: list[int] = []
    orig = _strategy.DrafterSpeculation.draft_next_batched
    orig_begin = _strategy.DrafterSpeculation.begin_batched_draft

    def _spy(self, seeds):
        seed_counts.append(len(seeds))
        return orig(self, seeds)

    def _spy_begin(self, seeds, **kwargs):
        # The early-dispatch pre-pass is the batched pool path too.
        seed_counts.append(len(seeds))
        return orig_begin(self, seeds, **kwargs)

    _strategy.DrafterSpeculation.draft_next_batched = _spy
    _strategy.DrafterSpeculation.begin_batched_draft = _spy_begin
    try:
        model = make_tiny_model()
        assert model.has_mtp()
        prompts = [
            [3, 1, 4, 1, 5, 9, 2, 6],
            [7, 7, 1, 2, 3, 8, 8, 4, 10, 5],
            [10, 20, 30, 40, 11, 21, 31],
        ]
        toks = _run_runner_generation(model, n_seqs=3, k=3, disable_batched=False, prompts=prompts)
    finally:
        _strategy.DrafterSpeculation.draft_next_batched = orig
        _strategy.DrafterSpeculation.begin_batched_draft = orig_begin

    # Every request completed with a full, valid decode.
    assert set(toks) == {"r-0", "r-1", "r-2"}
    for rid, out in toks.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"
        assert min(out) >= 0 and max(out) < _VOCAB
    # The step that prefilled the three requests drafted their first windows in one pooled call, and their verify
    # windows drafted together after it; a request left verifying alone drafts in a one-row call.
    assert 3 in seed_counts, f"the three first windows were not drafted together: {seed_counts}"
    assert sum(count >= 2 for count in seed_counts) >= 2, f"verify windows did not draft together: {seed_counts}"


def _run_runner_generation(model, *, n_seqs, k, disable_batched, prompts, max_new=10, drafter=None):
    """Drive ``n_seqs`` concurrent requests through a real eSurgeRunner to completion.

    ``disable_batched`` toggles the batched pool draft path off (per-request
    fallback) via ``EASYDEL_DISABLE_BATCHED_DRAFT`` so the same runner code can be
    exercised in both modes. ``drafter`` (default: a new persisting
    ``Qwen3_5MTPDrafter``) is the runner's drafter. Returns
    ``{req_id: [emitted token ids]}``.
    """
    prev_batch = os.environ.get("EASYDEL_DISABLE_BATCHED_DRAFT")
    prev_persist = os.environ.get("EASYDEL_MTP_PERSIST_KV")
    prev_fused = os.environ.get("EASYDEL_DISABLE_DRAFT_IN_VERIFY")
    os.environ["EASYDEL_DISABLE_BATCHED_DRAFT"] = "1" if disable_batched else "0"
    os.environ["EASYDEL_MTP_PERSIST_KV"] = "1"
    # This file's runner integration asserts the TWO-DISPATCH pooled batched
    # draft engages (begin/finish_batched_draft); pin the fused draft-in-verify
    # program off so that path is actually exercised. The fused path has its
    # own engagement + parity coverage in ``test_mtp_draft_in_verify.py``.
    os.environ["EASYDEL_DISABLE_DRAFT_IN_VERIFY"] = "1"
    try:
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
            drafter=drafter if drafter is not None else Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True),
        )
        runner.compile(max_num_batched_tokens=64)
        scheduler = make_core_for_runner(
            runner,
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
            async_scheduling=False,
            num_draft_tokens=runner.num_draft_tokens,
        )
        reqs = []
        for i in range(n_seqs):
            r = served_request(
                request_id=f"r-{i}",
                prompt_token_ids=list(prompts[i]),
                sampling_params=SamplingParams(max_tokens=max_new, temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            scheduler.add_request(r)
            reqs.append(r)
        for _ in range(4000):
            so = scheduler.schedule()
            out = runner.execute_model(so)
            scheduler.commit(so, out)
            if all(r.is_finished() for r in reqs):
                break
        else:
            raise AssertionError("generation did not finish")
        toks = {r.request_id: list(r.output_token_ids) for r in reqs}
        runner.shutdown()
        runner.executor_manager.kv_pages = None
        return toks
    finally:
        for key, prev in (
            ("EASYDEL_DISABLE_BATCHED_DRAFT", prev_batch),
            ("EASYDEL_MTP_PERSIST_KV", prev_persist),
            ("EASYDEL_DISABLE_DRAFT_IN_VERIFY", prev_fused),
        ):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
