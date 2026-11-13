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

"""Optimistic verify windows rewritten on the device equal the windows the host builds from exact results."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.inference.esurge.runners.async_types import (
    DeviceSpecWindowHandoff,
    SpecWindowCarry,
    spec_window_sources,
)
from easydel.inference.esurge.runners.execution_types import CacheGroupInputs
from easydel.inference.esurge.runners.executors.batch_preparer import BatchMetadataPreparer

N_TOK, N_REQ, MODEL_LEN, N_PAGES, PAGE, K = 16, 4, 64, 4, 16, 3


def _config(version: str = "v3"):
    return SimpleNamespace(
        version=version,
        get_max_num_seqs=lambda: N_REQ,
        max_num_pages_per_req=N_PAGES,
        data_parallel_size=1,
        page_size=PAGE,
        num_pages=0,
        num_slices_per_kv_cache_update_page=1,
        get_padded_num_slices=lambda tokens, reqs: tokens + reqs,
    )


def _preparer(groups: bool) -> BatchMetadataPreparer:
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    config = _config()
    return BatchMetadataPreparer(
        metadata=config,
        empty_sharding=sharding,
        max_num_tokens=N_TOK,
        max_num_reqs=N_REQ,
        max_model_len=MODEL_LEN,
        min_input_pad=1,
        enable_spec_recurrent_commit=True,
        input_sharding=sharding,
        cache_group_configs=(config, _config()) if groups else (),
        layer_to_cache_group=(0, 1) if groups else (),
    )


def _build(preparer, *, token_ids, computed, commit, groups, live=3):
    pages = np.tile(np.arange(N_PAGES, dtype=np.int32), (N_REQ, 1))
    kwargs = dict(
        num_tokens_static=N_TOK,
        scheduled_full_cpu=np.array([1 + K] * live + [0] * (N_REQ - live), dtype=np.int32),
        active_mask_full_cpu=np.array([True] * live + [False] * (N_REQ - live)),
        input_ids_buf=None,
        position_ids_buf=None,
        token_ids_cpu=token_ids,
        num_computed_tokens_cpu=computed,
        temperature_cpu=np.zeros((N_REQ,), dtype=np.float32),
        top_p_cpu=np.ones((N_REQ,), dtype=np.float32),
        top_k_cpu=np.zeros((N_REQ,), dtype=np.int32),
        min_p_cpu=np.zeros((N_REQ,), dtype=np.float32),
        frequency_penalties_cpu=np.zeros((N_REQ,), dtype=np.float32),
        presence_penalties_cpu=np.zeros((N_REQ,), dtype=np.float32),
        repetition_penalties_cpu=np.ones((N_REQ,), dtype=np.float32),
        page_table_cpu=pages,
        padded_num_reqs_in=N_REQ,
        page_table_version=1,
        window_row_indices_cpu=np.arange(N_REQ, dtype=np.int32),
        spec_recurrent_commit_cpu=commit,
    )
    if groups:
        identity = tuple(range(N_REQ))
        kwargs["cache_group_inputs"] = (
            CacheGroupInputs(pages, 1, identity),
            CacheGroupInputs(pages + 2, 1, identity, np.maximum(computed - 4, 0).astype(np.int32)),
        )
    metadata, *_ = preparer.prepare_batch_metadata(**kwargs)
    return metadata


def _pack(rows: dict[int, tuple[list[int], int, list[int]]]) -> np.ndarray:
    """The packed per-step output: emit in [0, k+1), accepted at k+1, next drafts in [k+2, 2k+2); -1 elsewhere."""
    out = np.full((N_REQ, 2 * K + 2), -1, dtype=np.int32)
    for row, (emit, accepted, drafts) in rows.items():
        out[row, : len(emit)] = emit
        out[row, K + 1] = accepted
        out[row, K + 2 : K + 2 + len(drafts)] = drafts
    return out


def _assert_windows_equal(fixed, exact, live, groups=False):
    n = live * (1 + K)
    np.testing.assert_array_equal(np.asarray(fixed.input_ids_buf)[:n], np.asarray(exact.input_ids_buf)[:n])
    np.testing.assert_array_equal(np.asarray(fixed.position_ids_buf)[:n], np.asarray(exact.position_ids_buf)[:n])
    np.testing.assert_array_equal(np.asarray(fixed.seq_lens)[:live], np.asarray(exact.seq_lens)[:live])
    np.testing.assert_array_equal(np.asarray(fixed.query_start_loc), np.asarray(exact.query_start_loc))
    np.testing.assert_array_equal(
        np.asarray(fixed.spec_recurrent_commit)[0, :live], np.asarray(exact.spec_recurrent_commit)[0, :live]
    )
    assert len(fixed.cache_groups) == (2 if groups else 0)
    for got, want in zip(fixed.cache_groups, exact.cache_groups, strict=True):
        np.testing.assert_array_equal(np.asarray(got.context_lens)[:live], np.asarray(want.context_lens)[:live])
        np.testing.assert_array_equal(np.asarray(got.position_ids)[:n], np.asarray(want.position_ids)[:n])
        np.testing.assert_array_equal(
            np.asarray(got.spec_recurrent_commit)[0, :live], np.asarray(want.spec_recurrent_commit)[0, :live]
        )


def test_unpack_packed_layout_rows():
    # Row 0 accepted every draft (bonus last), row 1 none, row 2 verified nothing (a = -1: its sampled token is
    # emit[0]) and has no next drafts (-1 -> placeholder 0), row 3 idle.
    spec_out = _pack({0: ([5, 6, 7, 8], 3, [9, 10, 11]), 1: ([20], 0, [21, 22, 23]), 2: ([30], -1, [])})
    handoff = DeviceSpecWindowHandoff.from_spec_out(
        spec_out=jnp.asarray(spec_out), num_draft_tokens=K, sources=np.zeros((2, N_REQ), np.int32)
    )
    np.testing.assert_array_equal(np.asarray(handoff.last_token)[:3], [8, 20, 30])
    np.testing.assert_array_equal(np.asarray(handoff.accepted)[:3], [3, 0, 0])
    np.testing.assert_array_equal(np.asarray(handoff.drafts)[:3], [[9, 10, 11], [21, 22, 23], [0, 0, 0]])


def test_carry_maps_requests_to_previous_rows():
    carry = SpecWindowCarry(
        spec_out=jnp.zeros((N_REQ, 2 * K + 2), jnp.int32),
        num_draft_tokens=K,
        rows={"a": 2, "b": 0, "gone": 1, "plain": 3},
    )
    # The plan's in-flight drafts: b verified no draft at step N (one emitted token, still a placeholder input);
    # "gone" left after step N; "new" was admitted from committed state.
    handoff = carry.handoff_for(["b", "new", "a", None], {"a": K, "b": 0}, N_REQ)
    np.testing.assert_array_equal(handoff.sources, [[0, -1, 2, -1], [0, 0, K, 0]])
    # A plan built from committed state (nothing in flight) needs no rewrite.
    assert carry.handoff_for(["new", "plain", None, None], {}, N_REQ) is None
    with pytest.raises(ValueError):
        carry.handoff_for(["x", None, None, None], {"x": K}, N_REQ)


@pytest.mark.parametrize("groups", [False, True])
def test_device_rewrite_matches_exact_host_windows(groups):
    # Step N's packed output, rows = its sequence-buffer rows: row 2 verified 3 drafts and accepted 1, row 0
    # accepted all 3.
    spec_out = _pack({0: ([50, 51, 52, 53], 3, [501, 502, 503]), 2: ([70, 71], 1, [701, 702, 703])})
    # Window row 0 is fed by output row 2, row 1 by output row 0, row 2 has exact host inputs, row 3 is idle.
    sources = spec_window_sources(["b", "a", "c", None], {"a": (0, K), "b": (2, K), "gone": (1, K)}, N_REQ)
    np.testing.assert_array_equal(sources, np.array([[2, 0, -1, -1], [K, K, 0, 0]], dtype=np.int32))
    true_computed = np.array([10, 7, 5, 0], dtype=np.int32)
    delta = np.array([K - 1, K - 3, 0, 0], dtype=np.int32)
    optimistic = true_computed + delta

    history = np.arange(N_REQ * MODEL_LEN, dtype=np.int32).reshape(N_REQ, MODEL_LEN) + 1000
    exact_ids = history.copy()
    optimistic_ids = history.copy()
    for row, window in ((0, [71, 701, 702, 703]), (1, [53, 501, 502, 503])):
        exact_ids[row, true_computed[row] : true_computed[row] + 1 + K] = window
        optimistic_ids[row, optimistic[row] : optimistic[row] + 1 + K] = 0  # placeholders
    # Slot bases: the host's are stale by the drafts step N rejected; the exact base is context before - accepted.
    exact_commit = np.zeros((2, N_REQ), dtype=np.int32)
    exact_commit[0] = [10 - 1, 7 - 3, 4, 0]
    optimistic_commit = exact_commit.copy()
    optimistic_commit[0, :2] = [99, 98]

    preparer = _preparer(groups)
    exact = _build(preparer, token_ids=exact_ids, computed=true_computed, commit=exact_commit, groups=groups)
    built = _build(preparer, token_ids=optimistic_ids, computed=optimistic, commit=optimistic_commit, groups=groups)
    handoff = DeviceSpecWindowHandoff.from_spec_out(
        spec_out=jnp.asarray(spec_out), num_draft_tokens=K, sources=sources, rewrite_recurrent_base=True
    )
    with jax.transfer_guard_device_to_host("disallow"):
        fixed, window_delta = preparer.apply_spec_window_handoff(built, handoff)

    np.testing.assert_array_equal(np.asarray(window_delta), delta)
    _assert_windows_equal(fixed, exact, live=3, groups=groups)
    # The host-built input is left untouched (the rewrite returns new arrays).
    assert int(np.asarray(built.input_ids_buf)[0]) == 0


def test_multi_step_stream_with_finish_preemption_and_admission():
    """Synthetic oracle stream over seven async steps: every acceptance count 0..k, a request finishing, one
    preempted and resumed, one admitted, and row orders that change every step. Each step's device-rewritten
    windows equal the exact host windows."""
    rng = np.random.default_rng(0)
    preparer = _preparer(False)
    # Truth per request: start of its next window, its first input token, its next drafts.
    truth = {
        rid: {"c": c, "last": int(rng.integers(1, 900)), "drafts": rng.integers(1, 900, K).tolist()}
        for rid, c in (("A", 5), ("B", 9), ("C", 3), ("D", 6))
    }
    orders = [
        ["A", "B", "C"],
        ["C", "A", "B"],
        ["B", "A"],
        ["D", "A", "B"],
        ["A", "D"],
        ["B", "D", "A"],
        ["A", "B", "D"],
    ]
    accept = {
        "A": [3, 0, 1, 2, 3, 1, 0],
        "B": [1, 2, 0, 3, 0, 2, 3],
        "C": [0, 3, 0, 0, 0, 0, 0],
        "D": [0, 0, 0, 2, 0, 3, 1],
    }
    # B is preempted at the commit of step 3 while step 4 is in flight: its step-3 results are dropped, it sits
    # out step 4 and comes back at step 5 with exact host inputs (no source).
    preempted_at = {"B": 3}
    carry = None
    for step, order in enumerate(orders):
        live = len(order)
        window = [*order] + [None] * (N_REQ - live)
        exact_computed = np.zeros((N_REQ,), np.int32)
        opt_computed = np.zeros((N_REQ,), np.int32)
        exact_ids = np.zeros((N_REQ, MODEL_LEN), np.int32)
        opt_ids = np.zeros((N_REQ, MODEL_LEN), np.int32)
        exact_commit = np.zeros((2, N_REQ), np.int32)
        opt_commit = np.zeros((2, N_REQ), np.int32)
        # The plan's in-flight drafts: requests scheduled while the previous step (which ran them) is in flight.
        inflight = {rid: K for rid in order if "prev_a" in truth[rid]}
        handoff = None
        if carry is not None:
            handoff = carry.handoff_for(window, inflight, N_REQ, rewrite_recurrent_base=True)
        fed = set() if handoff is None else {rid for row, rid in enumerate(order) if handoff.sources[0, row] >= 0}
        assert fed == set(inflight)
        for row, rid in enumerate(order):
            t = truth[rid]
            c = t["c"]
            exact_computed[row] = c
            exact_ids[row, c : c + 1 + K] = [t["last"], *t["drafts"]]
            if rid in fed:
                a = t["prev_a"]
                opt_computed[row] = c + (K - a)  # the core advanced as if every draft was accepted
                exact_commit[0, row] = c - a
                opt_commit[0, row] = 999  # stale: the host never saw the accepted count
            else:
                opt_computed[row] = c
                opt_ids[row, c : c + 1 + K] = exact_ids[row, c : c + 1 + K]
                exact_commit[0, row] = opt_commit[0, row] = max(c - 1, 0)
        exact = _build(
            preparer, token_ids=exact_ids, computed=exact_computed, commit=exact_commit, groups=False, live=live
        )
        built = _build(preparer, token_ids=opt_ids, computed=opt_computed, commit=opt_commit, groups=False, live=live)
        if handoff is None:
            assert not fed
            fixed = built
        else:
            with jax.transfer_guard_device_to_host("disallow"):
                fixed, _delta = preparer.apply_spec_window_handoff(built, handoff)
        _assert_windows_equal(fixed, exact, live=live)

        # Run the step: sequence-buffer rows differ from window rows and move every step.
        seq_rows = {rid: (i * 3 + step) % N_REQ for i, rid in enumerate(order)}
        packed = {}
        for rid in order:
            t, a = truth[rid], accept[rid][step]
            emit = [*t["drafts"][:a], int(rng.integers(1, 900))]
            nxt = rng.integers(1, 900, K).tolist()
            packed[seq_rows[rid]] = (emit, a, nxt)
            if preempted_at.get(rid) == step:
                t.pop("prev_a", None)
                continue  # results dropped; the resumed request starts from exact host inputs
            t.update(c=t["c"] + 1 + a, last=emit[-1], drafts=nxt, prev_a=a)
        carry = SpecWindowCarry(spec_out=jnp.asarray(_pack(packed)), num_draft_tokens=K, rows=seq_rows)
        for rid in [r for r in truth if r not in (orders[step + 1] if step + 1 < len(orders) else [])]:
            truth[rid].pop("prev_a", None)  # a request absent from the next step has no source when it returns
