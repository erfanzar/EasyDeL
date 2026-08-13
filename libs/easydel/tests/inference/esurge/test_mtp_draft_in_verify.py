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

"""DRAFT-IN-VERIFY parity: the fused verify+draft program vs the two-dispatch flow.

The eSurge greedy spec-decode step used to be TWO device dispatches separated by
a host round-trip: (1) project + argmax the verify rows, block on the argmax
readback, compute acceptance on host, then (2) dispatch the batched MTP drafter
with the host-derived seeds and read the drafts back after the emission loop.
The fused draft-in-verify program (``Qwen3_5MTPDrafter.verify_draft_batched``)
folds the argmax, the greedy acceptance compare, the seed selection, the per-row
EAGLE rollback, and the K recursive MTP forwards into ONE compiled program, so
the drafter dispatch and its preceding host round-trip disappear.

Parity strategy (mirrors ``test_mtp_batched_draft.py`` /
``test_mtp_batched_emit.py``):

* **Strategy-level parity** drives ``begin/finish_draft_in_verify`` against the
  exact two-dispatch reference composition (host argmax -> host acceptance ->
  ``begin/finish_batched_draft``) on the REAL tiny Qwen3.5 + MTP with engineered
  hidden rows, so full/partial/zero acceptance, multi-window persistence,
  per-row rollback, request relocation, and mixed fresh+persisted rows are each
  asserted token-exactly (drafts AND argmaxes AND accepted counts).
* **Runner integration** asserts the fused program engages by default for >= 2
  greedy spec requests, honors the dynamic-K live budget, falls back to the
  two-dispatch flow on ``EASYDEL_DISABLE_DRAFT_IN_VERIFY=1``, and keeps the
  executable count at one per (request bucket, K variant).
* **Donation** is asserted at the op level: the compiled fused program aliases
  the MTP pool buffers input->output (``input_output_alias``) and contains ZERO
  pool-shaped copy ops — a defensive per-step copy of the pool would eat the
  fusion win.
"""

from __future__ import annotations

import os
import re
import types

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation, _SpecVerifyMetadata
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35 as make_tiny_model

_VOCAB = 256
_HIDDEN = 128

_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=32))


# --------------------------------------------------------------------------- #
# Strategy-level parity: fused program vs the two-dispatch reference.
# --------------------------------------------------------------------------- #


def _make_strategy(model, n: int, k: int) -> DrafterSpeculation:
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    runner = types.SimpleNamespace(max_num_reqs=n, max_num_seqs=n, requests={})
    return DrafterSpeculation(runner=runner, drafter=drafter, num_draft_tokens=k)


def _meta(rid: str, req_idx: int, start_pos: int, real_count: int, drafts: list[int], offset: int):
    nd = len(drafts)
    return _SpecVerifyMetadata(
        request_id=rid,
        row_pos=req_idx,
        req_idx=req_idx,
        start_pos=start_pos,
        real_count=real_count,
        target_local_indices=list(range(offset, offset + nd)),
        bonus_local_index=offset + nd,
        draft_token_positions=[start_pos + real_count + i for i in range(nd)],
        scheduled_draft_tokens=list(drafts),
        buffer_draft_tokens=list(drafts),
    )


def _gathered(seed: int, rows: int) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(seed), (rows, _HIDDEN), dtype=jnp.float32)


def _reference_two_dispatch(spec: DrafterSpeculation, entries, gathered) -> dict:
    """The exact flow the fused program replaces: eager LM-head projection +
    host argmax + the emission loop's greedy acceptance + begin/finish_batched_draft."""
    logits = spec.project_hidden_rows(gathered)
    argm = np.asarray(jnp.argmax(logits, axis=-1)).astype(np.int64)
    seeds, per = [], {}
    for meta, off, cnt in entries:
        acc = 0
        for i, tok in enumerate(meta.buffer_draft_tokens):
            if int(argm[off + i]) != int(tok):
                break
            acc += 1
        corrected = int(argm[off + acc])
        known = meta.start_pos + meta.real_count + acc + 1
        seeds.append((meta.request_id, meta.req_idx, corrected, max(0, known - 2), off + acc))
        per[meta.request_id] = {
            "argmaxes": [int(x) for x in argm[off : off + cnt]],
            "accepted": acc,
            "known_len": known,
        }
    drafts = spec.finish_batched_draft(spec.begin_batched_draft(seeds, seed_hidden_matrix=gathered))
    for rid in per:
        per[rid]["drafts"] = drafts[rid]
    return per


def _fused(spec: DrafterSpeculation, entries, gathered, k: int) -> dict:
    handle = spec.begin_draft_in_verify(entries, gathered_hidden=gathered, k_pad=k)
    assert handle is not None, "fused draft-in-verify did not engage"
    return spec.finish_draft_in_verify(handle)


def _apply_lm_head_projection(model):
    """`project_hidden_rows` needs a compiled runner LM head; strategy-level tests
    replace it with the model's own eager apply_lm_head (the same projection the
    fused program's default in-program head runs)."""

    def _proj(self, hidden_rows):
        return model.apply_lm_head(hidden_rows)

    return _proj


def _assert_row_parity(ref: dict, fus: dict, tag: str) -> None:
    assert set(ref) == set(fus)
    for rid in ref:
        for key in ("argmaxes", "accepted", "known_len", "drafts"):
            assert ref[rid][key] == fus[rid][key], f"{tag} {rid} {key}: {ref[rid][key]} != {fus[rid][key]}"


def _row_index(drafter: Qwen3_5MTPDrafter, row: int) -> int:
    return int(np.asarray(drafter._mtp_cache.views[0].indexes).reshape(-1)[row])


def test_fused_equals_two_dispatch_acceptance_matrix(monkeypatch):
    """Full acceptance, partial acceptance, and full rejection rows in one window
    match the two-dispatch reference token-exactly (argmaxes/accepted/drafts)."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    model = make_tiny_model()
    n, k = 4, 3
    monkeypatch.setattr(DrafterSpeculation, "project_hidden_rows", _apply_lm_head_projection(model))
    spec_ref = _make_strategy(model, n, k)
    spec_fus = _make_strategy(model, n, k)

    rows = n * (k + 1)
    g = _gathered(0, rows)
    argm = np.asarray(jnp.argmax(model.apply_lm_head(g), axis=-1))

    entries, off = [], 0
    # A: full acceptance (drafts == the argmax prefix -> bonus token emitted).
    entries.append((_meta("A", 0, 10, 1, [int(argm[off + i]) for i in range(k)], off), off, k + 1))
    off += k + 1
    # B: partial acceptance (mismatch at the second draft).
    entries.append((_meta("B", 1, 20, 1, [int(argm[off]), int(argm[off + 1]) + 1, 7], off), off, k + 1))
    off += k + 1
    # C: full rejection (mismatch at the first draft).
    entries.append((_meta("C", 2, 5, 1, [int(argm[off]) + 1, 3, 9], off), off, k + 1))
    off += k + 1

    ref = _reference_two_dispatch(spec_ref, entries, g)
    fus = _fused(spec_fus, entries, g, k)
    _assert_row_parity(ref, fus, "w1")
    # The engineered acceptance pattern actually happened.
    assert ref["A"]["accepted"] == k and ref["C"]["accepted"] == 0 and 0 < ref["B"]["accepted"] < k


def test_fused_multi_window_persistence_and_rollback(monkeypatch):
    """Windows 2..3 with per-row different accepted counts: parity holds and the
    persistent MTP cache write indexes stay identical between the two flows."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    model = make_tiny_model()
    n, k = 4, 3
    monkeypatch.setattr(DrafterSpeculation, "project_hidden_rows", _apply_lm_head_projection(model))
    spec_ref = _make_strategy(model, n, k)
    spec_fus = _make_strategy(model, n, k)

    rows = n * (k + 1)
    prev = None
    for widx in range(3):
        g = _gathered(widx, rows)
        argm = np.asarray(jnp.argmax(model.apply_lm_head(g), axis=-1))
        entries, off = [], 0
        for i, rid in enumerate(("A", "B", "C")):
            start = 10 * (i + 1) if prev is None else prev[rid]["known_len"]
            # Engineer a different accepted count per row and per window.
            want = (widx + i) % (k + 1)
            drafts = [int(argm[off + j]) if j < want else int(argm[off + j]) + 1 + j for j in range(k)]
            entries.append((_meta(rid, i, start, 1, drafts, off), off, k + 1))
            off += k + 1
        ref = _reference_two_dispatch(spec_ref, entries, g)
        fus = _fused(spec_fus, entries, g, k)
        _assert_row_parity(ref, fus, f"w{widx + 1}")
        prev = ref

    for row in range(n):
        assert _row_index(spec_ref.drafter, row) == _row_index(spec_fus.drafter, row), (
            f"row {row}: cache write index diverged between fused and two-dispatch"
        )
    # One executable across all windows (rollback targets are traced inputs).
    assert len(spec_fus.drafter._jit_verify_draft) == 1


def test_fused_mixed_fresh_and_persisted_rows(monkeypatch):
    """A request joining mid-serving (fresh row) alongside persisted rows keeps
    parity — the fresh row rolls to 0, the persisted rows to committed+accepted."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    model = make_tiny_model()
    n, k = 4, 2
    monkeypatch.setattr(DrafterSpeculation, "project_hidden_rows", _apply_lm_head_projection(model))
    spec_ref = _make_strategy(model, n, k)
    spec_fus = _make_strategy(model, n, k)

    rows = n * (k + 1)
    g1 = _gathered(7, rows)
    argm1 = np.asarray(jnp.argmax(model.apply_lm_head(g1), axis=-1))
    e1 = [
        (_meta("A", 0, 10, 1, [int(argm1[0]), int(argm1[1])], 0), 0, k + 1),
        (_meta("B", 1, 20, 1, [int(argm1[3]) + 1, 5], k + 1), k + 1, k + 1),
    ]
    ref1 = _reference_two_dispatch(spec_ref, e1, g1)
    fus1 = _fused(spec_fus, e1, g1, k)
    _assert_row_parity(ref1, fus1, "w1")

    g2 = _gathered(8, rows)
    argm2 = np.asarray(jnp.argmax(model.apply_lm_head(g2), axis=-1))
    e2 = [
        (_meta("A", 0, ref1["A"]["known_len"], 1, [int(argm2[0]), 9], 0), 0, k + 1),
        (_meta("B", 1, ref1["B"]["known_len"], 1, [int(argm2[3]), int(argm2[4])], k + 1), k + 1, k + 1),
        # C is FRESH in window 2 (mixed fresh+persisted batch).
        (_meta("C", 2, 30, 1, [int(argm2[6]), int(argm2[7]) + 1], 2 * (k + 1)), 2 * (k + 1), k + 1),
    ]
    ref2 = _reference_two_dispatch(spec_ref, e2, g2)
    fus2 = _fused(spec_fus, e2, g2, k)
    _assert_row_parity(ref2, fus2, "w2")


def test_fused_request_relocation_keeps_stable_rows(monkeypatch):
    """A request relocated to a different sequence-buffer row between windows
    (condense/reorder) keeps its lifetime-stable drafter row: parity holds and
    no extra pool row is claimed."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    model = make_tiny_model()
    n, k = 4, 2
    monkeypatch.setattr(DrafterSpeculation, "project_hidden_rows", _apply_lm_head_projection(model))
    spec_ref = _make_strategy(model, n, k)
    spec_fus = _make_strategy(model, n, k)

    rows = n * (k + 1)
    g1 = _gathered(11, rows)
    argm1 = np.asarray(jnp.argmax(model.apply_lm_head(g1), axis=-1))
    e1 = [
        (_meta("A", 0, 10, 1, [int(argm1[0]), int(argm1[1])], 0), 0, k + 1),
        (_meta("B", 1, 20, 1, [int(argm1[3]), 5], k + 1), k + 1, k + 1),
    ]
    ref1 = _reference_two_dispatch(spec_ref, e1, g1)
    fus1 = _fused(spec_fus, e1, g1, k)
    _assert_row_parity(ref1, fus1, "w1")
    row_a = spec_fus._mtp_row_by_req["A"]

    # Window 2: A now sits at req_idx 3 (relocated); B unchanged.
    g2 = _gathered(12, rows)
    argm2 = np.asarray(jnp.argmax(model.apply_lm_head(g2), axis=-1))
    e2 = [
        (_meta("A", 3, ref1["A"]["known_len"], 1, [int(argm2[0]), int(argm2[1])], 0), 0, k + 1),
        (_meta("B", 1, ref1["B"]["known_len"], 1, [int(argm2[3]) + 2, 6], k + 1), k + 1, k + 1),
    ]
    ref2 = _reference_two_dispatch(spec_ref, e2, g2)
    fus2 = _fused(spec_fus, e2, g2, k)
    _assert_row_parity(ref2, fus2, "w2")
    assert spec_fus._mtp_row_by_req["A"] == row_a, "relocation must not change the stable drafter row"


def test_fused_gates_k_transitions_and_k0_to_fallback(monkeypatch):
    """begin_draft_in_verify declines a window whose width differs from the live
    budget (dynamic-K transition) and a K=0 budget — those fall back untouched."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    model = make_tiny_model()
    n, k = 4, 3
    spec = _make_strategy(model, n, k)
    g = _gathered(0, n * (k + 1))
    entries = [(_meta("A", 0, 10, 1, [1, 2, 3], 0), 0, k + 1)]

    spec.live_num_draft_tokens = k - 1  # transition step: window is k wide
    assert spec.begin_draft_in_verify(entries, gathered_hidden=g, k_pad=k) is None
    spec.live_num_draft_tokens = 0  # throttled off
    assert spec.begin_draft_in_verify(entries, gathered_hidden=g, k_pad=0) is None
    monkeypatch.setenv("EASYDEL_DISABLE_DRAFT_IN_VERIFY", "1")
    spec.live_num_draft_tokens = k
    assert spec.can_draft_in_verify() is False


def test_fused_pool_is_donated_not_copied():
    """Op-level check: the compiled fused program aliases the MTP pool buffers
    input->output and contains ZERO pool-shaped copy ops (a defensive copy of
    the pool per step would eat the fusion win)."""
    model = make_tiny_model()
    n, k = 4, 3
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True, cache_length=512)
    drafter.ensure_batch_size(n)
    rows = n * (k + 1)
    gathered = _gathered(0, rows)
    mtp_cache = drafter._mtp_cache
    fn = drafter._get_verify_draft_fn(k_pad=k, k_live=k, gathered_hidden=gathered, mtp_cache=mtp_cache, project_fn=None)
    _gds, (mtp_st, embed_st, head_st), freqs, _tie = drafter._batched_bundle_graph_and_states()
    args = (
        mtp_st,
        embed_st,
        head_st,
        freqs,
        gathered,
        jnp.zeros((n,), jnp.int32),
        jnp.zeros((n, k), jnp.int32),
        jnp.zeros((n,), jnp.int32),
        jnp.zeros((n,), jnp.int32),
        jnp.zeros((n,), jnp.int32),
        jnp.zeros((n,), bool),
        jnp.zeros((n,), bool),
        mtp_cache,
    )
    hlo = fn.lower(*args).compile().as_text()

    assert "input_output_alias" in hlo, "fused program lost the MTP-pool donation aliasing"
    pool_view = mtp_cache.views[0]
    dim_str = ",".join(str(d) for d in pool_view.key.shape)
    pool_copies = [ln for ln in hlo.splitlines() if re.search(r"\bcopy\(", ln) and dim_str in ln]
    assert pool_copies == [], f"fused program defensively copies the MTP pool: {pool_copies[:3]}"

    # Donation is real at runtime too: the input pool buffer is consumed.
    out = fn(*args)
    jax.block_until_ready(out[0])
    deleted = False
    try:
        _ = np.asarray(pool_view.key)
    except RuntimeError:
        deleted = True
    assert deleted, "input MTP pool buffer survived the call (donation not applied)"
    # The drafter-level wrapper must therefore re-point at the returned pool.
    drafter._mtp_cache = out[4]


# --------------------------------------------------------------------------- #
# Runner integration: engagement, dynamic-K, fallback flag, compile counts.
# --------------------------------------------------------------------------- #


def _traceable_project(hidden_rows):
    """Batch-invariant deterministic verify projection (traceable, so it can run
    both host-side via ``project_hidden_rows`` and in-program via the fused
    projection hook). ``argmax`` over the feature axis is a per-row reduction, so
    a row projected alone is bit-identical to the same row inside any batch."""
    hr = jnp.asarray(hidden_rows)
    feat = jnp.argmax(hr.astype(jnp.float32), axis=-1).astype(jnp.int32)
    tok = jnp.mod(feat * 7 + 3, _VOCAB).astype(jnp.int32)
    return jax.nn.one_hot(tok, _VOCAB, dtype=jnp.float32) * 30.0


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


_PROMPTS = [
    [3, 1, 4, 1, 5, 9, 2, 6],
    [7, 7, 1, 2, 3, 8, 8, 4, 10, 5],
    [10, 20, 30, 40, 11, 21, 31],
]


def _serve(runner, prompts, *, max_new=10):
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


def test_runner_fused_engages_by_default_and_flag_falls_back(monkeypatch):
    """B=3 greedy serving: the fused program is the default steady-state path
    (>= 2 spec rows per call, no two-dispatch drafter dispatch), and the escape
    hatch restores the two-dispatch flow."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DRAFT_IN_VERIFY", raising=False)
    model = make_tiny_model()

    fused_rows: list[int] = []
    two_dispatch_calls: list[int] = []
    orig_fused = Qwen3_5MTPDrafter.verify_draft_batched
    orig_batched = Qwen3_5MTPDrafter.draft_batched

    def _spy_fused(self, **kwargs):
        fused_rows.append(int(np.asarray(kwargs["collected_mask"]).sum()))
        return orig_fused(self, **kwargs)

    def _spy_batched(self, **kwargs):
        two_dispatch_calls.append(int(np.asarray(kwargs["collected_mask"]).sum()))
        return orig_batched(self, **kwargs)

    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True)
    runner = _make_runner(model, n_seqs=3, drafter=drafter)
    monkeypatch.setattr(Qwen3_5MTPDrafter, "verify_draft_batched", _spy_fused)
    monkeypatch.setattr(Qwen3_5MTPDrafter, "draft_batched", _spy_batched)
    toks = _serve(runner, _PROMPTS)
    for rid, out in toks.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"
        assert min(out) >= 0 and max(out) < _VOCAB
    engaged = [c for c in fused_rows if c >= 2]
    assert engaged, "fused draft-in-verify never engaged for a 3-request greedy batch"
    assert two_dispatch_calls == [], (
        f"steady-state serving still dispatched the two-dispatch batched drafter: {two_dispatch_calls}"
    )

    # Escape hatch: the two-dispatch flow comes back, fused stays silent.
    monkeypatch.setenv("EASYDEL_DISABLE_DRAFT_IN_VERIFY", "1")
    fused_rows.clear()
    two_dispatch_calls.clear()
    drafter2 = Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True)
    runner2 = _make_runner(model, n_seqs=3, drafter=drafter2)
    toks2 = _serve(runner2, _PROMPTS)
    for rid, out in toks2.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"
    assert fused_rows == [], "EASYDEL_DISABLE_DRAFT_IN_VERIFY=1 must silence the fused program"
    assert any(c >= 2 for c in two_dispatch_calls), "fallback did not engage the two-dispatch batched drafter"


def test_runner_fused_executable_count_one_per_bucket_k(monkeypatch):
    """Compile-bucket discipline: after compile() + serving, the fused program
    holds exactly one executable per (request bucket, schedule-K) pair —
    serving must not mint new variants."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DRAFT_IN_VERIFY", raising=False)
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True)
    runner = _make_runner(model, n_seqs=3, drafter=drafter)

    buckets = {int(b) for b in runner.active_num_seq_buckets}
    k_values = {int(runner.num_draft_tokens)}
    expected = len(buckets) * len(k_values)
    warmed = set(drafter._jit_verify_draft.keys())
    assert len(warmed) == expected, (
        f"warmup minted {len(warmed)} fused executables, expected {expected} "
        f"(buckets={sorted(buckets)}, ks={sorted(k_values)})"
    )

    _ = _serve(runner, _PROMPTS)
    assert set(drafter._jit_verify_draft.keys()) == warmed, (
        "serving compiled new fused draft-in-verify variants beyond the warmed (bucket, K) grid"
    )


def _det_hidden(shape, dtype, step):
    """Deterministic forward hidden (see ``test_mtp_batched_emit.py``): a fixed
    function of (flat token index, feature, step), per-row distinct so the
    argmax-of-features projection yields a non-trivial per-row token."""
    t = jnp.arange(shape[0], dtype=jnp.float32)[:, None]
    d = jnp.arange(shape[1], dtype=jnp.float32)[None, :]
    return jnp.sin(t * 0.7 + d * 0.13 + float(step) * 1.1).astype(dtype)


def _run_deterministic(model, *, n, k, fused: bool, prompts):
    """Fully-deterministic generation through the REAL runner: the model forward
    output is replaced by a fixed function of (flat index, step) and the verify
    projection by the batch-invariant ``_traceable_project`` — injected into
    BOTH the host pre-pass (``project_hidden_rows``) and the fused program
    (``draft_in_verify_project_fn``), so the only difference between the two
    arms is fused vs two-dispatch. Drafts come from the REAL MTP recurrence in
    both arms (in-program vs ``draft_batched``). Returns {rid: emitted tokens}.
    """
    import types as _types

    from easydel.inference.esurge.runners.spec import strategy as _strategy

    saved_proj = _strategy.DrafterSpeculation.project_hidden_rows
    saved_hook = _strategy.DrafterSpeculation.draft_in_verify_project_fn
    saved_env = os.environ.get("EASYDEL_DISABLE_DRAFT_IN_VERIFY")
    saved_persist = os.environ.get("EASYDEL_MTP_PERSIST_KV")
    _strategy.DrafterSpeculation.project_hidden_rows = lambda self, hr: _traceable_project(hr)
    _strategy.DrafterSpeculation.draft_in_verify_project_fn = lambda self: _traceable_project
    os.environ["EASYDEL_DISABLE_DRAFT_IN_VERIFY"] = "0" if fused else "1"
    os.environ["EASYDEL_MTP_PERSIST_KV"] = "1"
    try:
        drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
        runner = _make_runner(model, n_seqs=n, drafter=drafter)

        real_execute = runner.executor_manager.execute
        step_counter = _types.SimpleNamespace(v=0)

        def _det_execute(**kw):
            out_tokens, valid, iib, pib, hidden, logits, metrics = real_execute(**kw)
            s = step_counter.v
            step_counter.v += 1
            if hidden is not None:
                hidden = jax.device_put(_det_hidden(hidden.shape, hidden.dtype, s), hidden.sharding)
            if out_tokens is not None:
                r = out_tokens.shape[0]
                det_t = jnp.mod(jnp.arange(r, dtype=jnp.int32) * 7 + s * 13 + 5, _VOCAB).astype(out_tokens.dtype)
                out_tokens = jax.device_put(det_t, out_tokens.sharding)
            return out_tokens, valid, iib, pib, hidden, logits, metrics

        runner.executor_manager.execute = _det_execute
        return _serve(runner, prompts, max_new=14)
    finally:
        _strategy.DrafterSpeculation.project_hidden_rows = saved_proj
        _strategy.DrafterSpeculation.draft_in_verify_project_fn = saved_hook
        for key, val in (
            ("EASYDEL_DISABLE_DRAFT_IN_VERIFY", saved_env),
            ("EASYDEL_MTP_PERSIST_KV", saved_persist),
        ):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def test_runner_fused_token_parity_vs_two_dispatch_b3_k3():
    """Full-runner deterministic token parity: fused draft-in-verify serving ==
    two-dispatch serving, token-for-token, on identical deterministic inputs
    (B=3, k=3: multi-window persistence + mixed accept patterns end to end)."""
    model = make_tiny_model()
    fused = _run_deterministic(model, n=3, k=3, fused=True, prompts=_PROMPTS)
    twod = _run_deterministic(model, n=3, k=3, fused=False, prompts=_PROMPTS)
    assert set(fused) == {"r-0", "r-1", "r-2"}
    for rid in fused:
        assert len(fused[rid]) == 14, f"{rid} incomplete: {fused[rid]}"
        assert fused[rid] == twod[rid], f"{rid}: fused {fused[rid]} != two-dispatch {twod[rid]}"


def test_runner_fused_dynamic_k_uses_live_budget(monkeypatch):
    """With a dynamic schedule, every fused call drafts the scheduled live K;
    transition windows (verify width != live K) fall back without erroring."""
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DRAFT_IN_VERIFY", raising=False)
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=3, persist_kv=True)
    # Concurrency 1-2 -> K=3, concurrency 3+ -> K=2: K transitions as requests finish.
    drafter.num_draft_tokens_per_batch_size = [(1, 2, 3), (3, 1 << 30, 2)]

    fused_ks: list[int] = []
    orig_fused = Qwen3_5MTPDrafter.verify_draft_batched

    def _spy_fused(self, **kwargs):
        fused_ks.append(int(kwargs["num_draft_tokens"]))
        return orig_fused(self, **kwargs)

    runner = _make_runner(model, n_seqs=3, drafter=drafter)
    monkeypatch.setattr(Qwen3_5MTPDrafter, "verify_draft_batched", _spy_fused)
    toks = _serve(runner, _PROMPTS)
    for rid, out in toks.items():
        assert len(out) == 10, f"{rid} did not complete: {out}"
        assert min(out) >= 0 and max(out) < _VOCAB
    assert fused_ks, "fused draft-in-verify never engaged under the dynamic schedule"
    assert set(fused_ks) <= {2, 3}, f"fused drafts must use schedule Ks, got {sorted(set(fused_ks))}"
    assert 2 in fused_ks, "the concurrency-3 band (K=2) never used the fused program"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
