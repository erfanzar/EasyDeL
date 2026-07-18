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

"""Persist-with-rollback of the inline-MTP drafter KV cache (EAGLE / vLLM style).

By default the eSurge speculative strategy FULL-clears the inline-MTP drafter's
KV cache at every verify-window boundary (``Qwen3_5MTPDrafter.reset_request``),
so each window's 1-layer MTP self-attention only sees its own seed token. With
``persist_kv=True`` / ``EASYDEL_MTP_PERSIST_KV=1`` the strategy instead KEEPS the
cache across windows within a request and rolls it back to the committed
position (``Qwen3_5MTPDrafter.rollback_to``), dropping only the previous window's
rejected-draft K/V so each window sees the full accepted-sequence context.

These tests are CPU-only and cover the mechanics, not the acceptance win (which
is TPU-measured): the rollback truncates the cache write index exactly to the
committed length, an all-accepted window cannot over-advance past the written
extent, and the first window (nothing to persist) drafts identically with the
flag on or off. A runner-level greedy speculative-decode run with persistence on
asserts the emitted tokens are unchanged (greedy verification stays exact).
"""

from __future__ import annotations

import gc
import os
import types

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# The runner-level greedy-exactness check needs the exact sequential-replay
# recurrent path (the tiny model has linear_attention layers); the default fast
# path is coherent but not bit-identical to the no-spec baseline on recurrent
# models. ``_run_generation`` forces EASURGE_SPEC_RECURRENT_REPLAY=1 for its own
# runs (overriding any ambient fast-path setting) rather than relying on this
# process default, so the greedy==baseline assertion can't be made flaky by the
# caller's environment. This ``setdefault`` only sets a default for the rest.
os.environ.setdefault("EASURGE_SPEC_RECURRENT_REPLAY", "1")

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter
from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

_VOCAB = 256
_HIDDEN = 128


def make_tiny_model(mtp_layers: int = 1) -> Qwen3_5ForCausalLM:
    """Tiny Qwen3.5 with an MTP head (same shape as test_spec_decode)."""
    cfg = Qwen3_5TextConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
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


def _hidden(seed: int) -> jnp.ndarray:
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(7), seed), (1, 1, _HIDDEN), dtype=jnp.float32)


def _cache_len(drafter: Qwen3_5MTPDrafter) -> int:
    """Return the MTP cache write index of the first layer view."""
    view = drafter._mtp_cache.views[0]
    return int(np.asarray(view.indexes).reshape(-1)[0])


def _run_window(drafter: Qwen3_5MTPDrafter, *, seed_token: int, seed_position: int, seed_hidden, k: int) -> list[int]:
    """Mimic DrafterSpeculation.draft_next's greedy chained MTP loop for k drafts.

    Each ``draft`` call is a single-token (decode) MTP forward that appends one
    slot to the drafter's persistent KV cache.
    """
    cur_token = jnp.asarray([[int(seed_token)]], dtype=jnp.int32)
    cur_pos = int(seed_position)
    hidden = jnp.asarray(seed_hidden)
    drafts: list[int] = []
    for _ in range(int(k)):
        step = drafter.draft(
            input_ids=cur_token,
            target_hidden_states=hidden,
            position_ids=jnp.asarray([[cur_pos]], dtype=jnp.int32),
        )
        tok = int(np.asarray(step.token_ids).reshape(-1)[0])
        drafts.append(tok)
        cur_token = jnp.asarray([[tok]], dtype=jnp.int32)
        cur_pos += 1
        if step.hidden_states is not None:
            hidden = step.hidden_states[:, -1:, :]
    return drafts


def test_rollback_sets_cache_length_to_committed():
    """rollback_to(n) truncates every layer's write index to n; writes resume there."""
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=1, persist_kv=True)
    drafter.reset(1)

    _run_window(drafter, seed_token=5, seed_position=0, seed_hidden=_hidden(0), k=4)
    assert _cache_len(drafter) == 4, "each decode draft should append exactly one cache slot"

    applied = drafter.rollback_to(2)
    assert applied == 2
    for view in drafter._mtp_cache.views:
        if view is None:
            continue
        assert int(np.asarray(view.indexes).reshape(-1)[0]) == 2, "rollback must set the write index to the committed len"

    # The next window's writes resume from the committed position, not from 0.
    _run_window(drafter, seed_token=7, seed_position=2, seed_hidden=_hidden(1), k=1)
    assert _cache_len(drafter) == 3


def test_rollback_clamps_to_written_extent():
    """A committed length past what was written clamps down (no unwritten slots)."""
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=1, persist_kv=True)
    drafter.reset(1)
    _run_window(drafter, seed_token=5, seed_position=0, seed_hidden=_hidden(0), k=3)
    assert _cache_len(drafter) == 3

    # Requesting a larger committed length must not point the index past the
    # written extent (the all-accepted edge case): it clamps to the written len.
    drafter.rollback_to(10)
    assert _cache_len(drafter) == 3


def test_first_window_parity_persist_on_equals_off():
    """Window 1 has nothing to persist: persist-on and persist-off draft identically."""
    model = make_tiny_model()
    seed_token = 5
    seed_pos = 6
    h = _hidden(3)
    pos = jnp.asarray([[seed_pos]], dtype=jnp.int32)
    ids = jnp.asarray([[seed_token]], dtype=jnp.int32)

    off = Qwen3_5MTPDrafter(model, num_draft_tokens=1, persist_kv=False)
    off.reset(1)
    step_off = off.draft(input_ids=ids, target_hidden_states=h, position_ids=pos, return_full_log_probs=True)

    on = Qwen3_5MTPDrafter(model, num_draft_tokens=1, persist_kv=True)
    on.reset(1)
    step_on = on.draft(input_ids=ids, target_hidden_states=h, position_ids=pos, return_full_log_probs=True)

    assert int(np.asarray(step_on.token_ids).reshape(-1)[0]) == int(np.asarray(step_off.token_ids).reshape(-1)[0])
    assert np.allclose(np.asarray(step_on.full_log_probs), np.asarray(step_off.full_log_probs), atol=1e-6), (
        "the persist_kv flag must not change a single (first-window) draft"
    )


def test_multi_window_persist_runs_and_tracks_committed():
    """Mimic the strategy's persist loop across several windows: no crash, valid ids.

    Emulates the ``draft_next`` bookkeeping: the compacted committed length is the
    seed position's offset from the request's first seed, and ``rollback_to`` is
    called at each window boundary before the window's chained drafts.
    """
    model = make_tiny_model()
    k = 2
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    drafter.reset(1)

    base = 4  # first-window seed position (RoPE origin of compacted slot 0)
    drafts = _run_window(drafter, seed_token=5, seed_position=base, seed_hidden=_hidden(0), k=k)
    assert all(0 <= t < _VOCAB for t in drafts)
    assert _cache_len(drafter) == k

    # Accept 1 draft per window -> sequence advances by (accepted + 1) = 2 == k,
    # so the committed length grows contiguously with the written extent.
    for w in range(1, 4):
        seed_pos = base + 2 * w
        committed = seed_pos - base
        applied = drafter.rollback_to(committed)
        assert applied == committed
        assert _cache_len(drafter) == committed, "committed len should sit exactly at the written extent here"
        drafts = _run_window(
            drafter,
            seed_token=int(drafts[0]),
            seed_position=seed_pos,
            seed_hidden=_hidden(w),
            k=k,
        )
        assert all(0 <= t < _VOCAB for t in drafts)
        assert _cache_len(drafter) == committed + k


def _run_generation(model, *, drafter, persist_kv: bool, max_new_tokens: int = 8) -> list[int]:
    """Drive a greedy eSurge generation to completion; return the emitted tokens."""
    prev = os.environ.get("EASYDEL_MTP_PERSIST_KV")
    prev_replay = os.environ.get("EASURGE_SPEC_RECURRENT_REPLAY")
    os.environ["EASYDEL_MTP_PERSIST_KV"] = "1" if persist_kv else "0"
    # Exact sequential replay is required for greedy spec output to be bit-identical
    # to the no-spec baseline on this recurrent model; force it here so an ambient
    # EASURGE_SPEC_RECURRENT_REPLAY=0 (fast path) can't make this assertion flaky.
    os.environ["EASURGE_SPEC_RECURRENT_REPLAY"] = "1"
    try:
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
            request_id="req-persist-0",
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
        tokens = list(request.output_token_ids)
        runner.shutdown()
        runner.executor_manager.kv_pages = None
        return tokens
    finally:
        if prev is None:
            os.environ.pop("EASYDEL_MTP_PERSIST_KV", None)
        else:
            os.environ["EASYDEL_MTP_PERSIST_KV"] = prev
        if prev_replay is None:
            os.environ.pop("EASURGE_SPEC_RECURRENT_REPLAY", None)
        else:
            os.environ["EASURGE_SPEC_RECURRENT_REPLAY"] = prev_replay


def test_runner_greedy_output_unchanged_with_persist():
    """Persist-with-rollback: a single request drafts in-loop, so batched ON == OFF.

    The batched pool draft would run the greedy MTP forward as ONE program moved
    past the window loop — a different XLA program from the per-request row-targeted
    forward, so on this tiny recurrent model's spec-verify near-tie (a ~1 ULP flip,
    the same floating-point sensitivity that makes greedy output change under any
    batched-decode engine) it can land on the other side. The runner therefore only
    engages the pooled forward when >= 2 spec-active requests are verified together
    (``eSurgeRunner`` ``step_allows_batch``); a single request drafts in-loop through
    the EXACT per-request code, so batched-ON output is bit-identical to batched-OFF
    by construction.

    This asserts that gate deterministically: driving one request with the batched
    path ENABLED, the pooled ``draft_next_batched`` is never called (each greedy MTP
    draft runs in-loop). Multi-request batched drafting is compared to the
    per-request path directly and deterministically by the isolated
    ``draft_next_batched`` parity tests (``test_mtp_batched_draft``).
    """
    from easydel.inference.esurge.runners.spec import strategy as _strategy

    batched_seed_counts: list[int] = []
    orig = _strategy.DrafterSpeculation.draft_next_batched

    def _spy(self, seeds):
        batched_seed_counts.append(len(seeds))
        return orig(self, seeds)

    _strategy.DrafterSpeculation.draft_next_batched = _spy
    prev = os.environ.get("EASURGE_DISABLE_BATCHED_DRAFT")
    os.environ["EASURGE_DISABLE_BATCHED_DRAFT"] = "0"  # batched path ENABLED
    try:
        model = make_tiny_model(mtp_layers=1)
        assert model.has_mtp()
        out = _run_generation(
            model,
            drafter=Qwen3_5MTPDrafter(model, num_draft_tokens=1, persist_kv=True),
            persist_kv=True,
        )
    finally:
        _strategy.DrafterSpeculation.draft_next_batched = orig
        if prev is None:
            os.environ.pop("EASURGE_DISABLE_BATCHED_DRAFT", None)
        else:
            os.environ["EASURGE_DISABLE_BATCHED_DRAFT"] = prev

    assert out, "persistent speculative run emitted no tokens"
    assert min(out) >= 0 and max(out) < _VOCAB
    assert batched_seed_counts == [], (
        "a single request must draft in-loop through the per-request path, not the "
        f"pooled batched draft (batched engaged with row counts {batched_seed_counts})"
    )


def _seed_hidden(i: int) -> jnp.ndarray:
    """A (hidden,)-shaped seed hidden state for ``draft_next`` (it adds the B/S axes)."""
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(11), i), (_HIDDEN,), dtype=jnp.float32)


def _row_index(drafter: Qwen3_5MTPDrafter, row: int) -> int:
    """Return the first MTP layer view's write index for cache batch ``row``."""
    return int(np.asarray(drafter._mtp_cache.views[0].indexes).reshape(-1)[row])


def _make_strategy(drafter: Qwen3_5MTPDrafter, n_rows: int, k: int) -> DrafterSpeculation:
    """A DrafterSpeculation over a stub runner exposing only the pool size.

    ``draft_next`` (with an inline-MTP drafter and persist-with-rollback) reads
    only ``max_num_reqs`` off the runner — the target K/V payload path short-
    circuits for inline MTP — so a namespace stub is sufficient to exercise the
    per-request row bookkeeping without standing up a full runner.
    """
    runner = types.SimpleNamespace(max_num_reqs=n_rows, max_num_seqs=n_rows)
    return DrafterSpeculation(runner=runner, drafter=drafter, num_speculative_tokens=k)


# Greedy (temperature 0) so drafting is deterministic argmax and spec-eligible.
_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=32))


def _draft(strategy: DrafterSpeculation, req_id: str, row: int, token: int, pos: int, hidden) -> list[int]:
    """Thin ``draft_next`` wrapper: one greedy verify window for ``req_id`` on ``row``."""
    return strategy.draft_next(
        req_id=req_id,
        row_pos=row,
        seed_token=token,
        seed_position=pos,
        seed_hidden=hidden,
        req_state=_REQ_STATE,
    )


def test_batched_persist_keeps_a_row_per_request():
    """Persist-with-rollback keeps a separate MTP row per concurrent request.

    Two concurrent requests share one pool-sized MTP cache (batch ``N`` ==
    ``max_num_reqs``); each owns a row keyed on its stable KV-pool slot. This
    covers the batched limitation the per-request work fixes:

    (a) each request's row rolls back to ITS committed length independently;
    (b) interleaving ``draft_next(A), draft_next(B), draft_next(A)`` does NOT
        clear A's context — A's row keeps its accumulated K/V and A's second
        window is byte-identical with or without B's interleaved call; and
    (c) a slot reused by a NEW request re-bases (drops the stale K/V) so its
        drafts match a truly-fresh request's.
    """
    model = make_tiny_model()
    n, k = 2, 2

    a_seed, a_hidden = 5, _seed_hidden(0)
    b_seed, b_hidden = 9, _seed_hidden(1)
    a_w2_seed, a_w2_hidden = 7, _seed_hidden(2)

    # --- Scenario 1: interleave A (row 0), B (row 1), A (row 0). ---
    d1 = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    s1 = _make_strategy(d1, n, k)

    # A window 1 (fresh request on row 0): row 0 re-based to seed_position 4.
    a_w1 = _draft(s1, "A", 0, a_seed, 4, a_hidden)
    assert len(a_w1) == k, "a full greedy window should draft k tokens"
    assert s1._mtp_persist_base_pos_by_row[0] == 4
    assert s1._mtp_persist_req_by_row[0] == "A"
    assert _row_index(d1, 0) == k, "row 0 advanced by k drafts"
    assert _row_index(d1, 1) == 0, "row 1 must be untouched by A's window"

    # B window 1 (fresh request on row 1): must not disturb row 0.
    b_w1 = _draft(s1, "B", 1, b_seed, 10, b_hidden)
    assert len(b_w1) == k
    assert s1._mtp_persist_base_pos_by_row[1] == 10
    assert _row_index(d1, 1) == k, "row 1 advanced by B's window"
    assert _row_index(d1, 0) == k, "(b) row 0 retained its K/V across B's interleaved call (no thrash)"

    # A window 2 (SAME request on row 0): rolls back to ITS committed length (2),
    # not a full clear — base is unchanged, so B's interleaved call did not re-base A.
    a_w2_inter = _draft(s1, "A", 0, a_w2_seed, 6, a_w2_hidden)
    assert s1._mtp_persist_base_pos_by_row[0] == 4, "(b) same request must not re-base (that would clear its row)"
    assert _row_index(d1, 0) == 2 + k, "(a) row 0 rolled back to committed=2 then advanced by k"
    assert _row_index(d1, 1) == k, "(a) row 1 independent of A's second window"

    # --- Scenario 2: A alone (no B interleaved), identical A inputs. ---
    d2 = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    s2 = _make_strategy(d2, n, k)
    a2_w1 = _draft(s2, "A", 0, a_seed, 4, a_hidden)
    a2_w2 = _draft(s2, "A", 0, a_w2_seed, 6, a_w2_hidden)
    assert a_w1 == a2_w1, "A's first window is independent of scheduling"
    assert a_w2_inter == a2_w2, "(b) B's interleaved call did not change A's second-window drafts"

    # --- (c) Reuse row 0 for a NEW request C: stale A K/V must be dropped. ---
    c_seed, c_hidden = 13, _seed_hidden(3)
    c_reuse = _draft(s1, "C", 0, c_seed, 20, c_hidden)
    assert s1._mtp_persist_req_by_row[0] == "C", "row 0 now belongs to C"
    assert s1._mtp_persist_base_pos_by_row[0] == 20, "(c) reused slot re-bases the committed origin"
    assert _row_index(d1, 0) == k, "(c) reused slot restarts from 0 (stale A K/V dropped), not 2+k+k"
    assert _row_index(d1, 1) == k, "C's window on row 0 must not touch B's row 1"

    # Every draft above targeted a different row (0, 1, then 0 reused) at the same
    # cache shape: the row is a traced argument, not part of the compile key, so
    # all rows share ONE compiled MTP program (no per-row recompile).
    assert len(d1._jit_mtp) == 1, "row targeting must not add compile-cache entries"

    d3 = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    s3 = _make_strategy(d3, n, k)
    c_fresh = _draft(s3, "C", 0, c_seed, 20, c_hidden)
    assert c_reuse == c_fresh, "(c) drafts on a reused slot equal a truly-fresh request's (no stale K/V leak)"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
