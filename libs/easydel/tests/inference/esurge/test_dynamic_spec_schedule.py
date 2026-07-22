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

"""Dynamic speculative-token scheduling (vLLM's num_speculative_tokens_per_batch_size).

Contract under test: the number of draft tokens K adapts to the live
concurrency through an inclusive ``[(start, end, k), ...]`` schedule —

* schema validation (non-overlapping ranges, ``k >= 0``,
  ``k <= num_draft_tokens``) in the ``eSurgeDrafterConfig`` post-init and the
  eLarge ``esurge.drafter`` chain-through;
* resolution semantics: inclusive boundaries, gaps fall back to the static
  ``num_draft_tokens``, ``EASYDEL_DISABLE_DYNAMIC_SPEC`` forces static;
* the engine's plug-and-play ``drafter=True`` path installs the measured
  default schedule ``[(1, 4, K), (5, 2**30, 0)]`` while explicit static
  configs stay static;
* strategy-level behavior: the same request drafted across windows with the
  budget forced 4 -> 2 -> 0 -> 4 produces exactly-K drafts per window, a K=0
  window runs no drafter call, and the persistent MTP KV row stays coherent
  (rollback-to-committed) across every transition.
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
import pytest
from easydel.inference.esurge.config import (
    default_auto_draft_schedule,
    eSurgeDrafterConfig,
    normalize_draft_schedule,
    resolve_num_draft_tokens,
)
from easydel.inference.esurge.engine.bootstrap import _build_drafter
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35 as make_tiny_model

_HIDDEN = 128


# ---------------------------------------------------------------------------
# Config schema validation
# ---------------------------------------------------------------------------


def test_config_accepts_and_normalizes_schedule():
    """Lists-of-lists (YAML shape) normalize to sorted int 3-tuples."""
    cfg = eSurgeDrafterConfig.coerce_config(
        {"method": "mtp", "num_draft_tokens": 4, "num_draft_tokens_per_batch_size": [[5, 1 << 30, 0], [1, 4, 4]]}
    )
    assert cfg.num_draft_tokens_per_batch_size == [(1, 4, 4), (5, 1 << 30, 0)]


def test_config_default_is_static():
    """No schedule field -> None (static num_draft_tokens, today's behavior)."""
    cfg = eSurgeDrafterConfig.coerce_config({"method": "mtp"})
    assert cfg.num_draft_tokens_per_batch_size is None


def test_config_rejects_overlapping_ranges():
    with pytest.raises(ValueError, match="overlap"):
        eSurgeDrafterConfig.coerce_config(
            {"method": "mtp", "num_draft_tokens_per_batch_size": [(1, 8, 2), (8, 16, 0)]}
        )


def test_config_rejects_k_above_num_draft_tokens():
    with pytest.raises(ValueError, match="exceeds num_draft_tokens"):
        eSurgeDrafterConfig.coerce_config(
            {"method": "mtp", "num_draft_tokens": 2, "num_draft_tokens_per_batch_size": [(1, 4, 3)]}
        )


def test_config_rejects_malformed_entries():
    with pytest.raises(ValueError, match="triples"):
        eSurgeDrafterConfig.coerce_config({"method": "mtp", "num_draft_tokens_per_batch_size": [(1, 4)]})
    with pytest.raises(ValueError, match="start must be >= 1"):
        eSurgeDrafterConfig.coerce_config({"method": "mtp", "num_draft_tokens_per_batch_size": [(0, 4, 2)]})
    with pytest.raises(ValueError, match="end must be >= start"):
        eSurgeDrafterConfig.coerce_config({"method": "mtp", "num_draft_tokens_per_batch_size": [(4, 1, 2)]})
    with pytest.raises(ValueError, match=">= 0"):
        eSurgeDrafterConfig.coerce_config({"method": "mtp", "num_draft_tokens_per_batch_size": [(1, 4, -1)]})


def test_elarge_drafter_section_chains_schedule_through():
    """The eLarge ``esurge.drafter`` mapping passes the schedule to the engine config."""
    from easydel.infra.elarge.builders import to_esurge_kwargs

    cfg = {
        "model": {"name_or_path": "stub/tiny"},
        "esurge": {
            "drafter": {
                "enabled": True,
                "num_draft_tokens": 4,
                "num_draft_tokens_per_batch_size": [[1, 4, 4], [5, 32, 0]],
            }
        },
    }
    dc = to_esurge_kwargs(cfg)["drafter_config"]
    assert dc["enabled"] is True
    assert dc.num_draft_tokens_per_batch_size == [(1, 4, 4), (5, 32, 0)]


# ---------------------------------------------------------------------------
# Resolution semantics
# ---------------------------------------------------------------------------


def test_resolution_boundaries_are_inclusive_and_gaps_fall_back_to_static():
    sched = normalize_draft_schedule([(1, 4, 4), (9, 16, 2)], max_num_draft_tokens=4)
    # Inclusive on both ends.
    assert resolve_num_draft_tokens(sched, batch_size=1, num_draft_tokens=4) == 4
    assert resolve_num_draft_tokens(sched, batch_size=4, num_draft_tokens=4) == 4
    assert resolve_num_draft_tokens(sched, batch_size=9, num_draft_tokens=4) == 2
    assert resolve_num_draft_tokens(sched, batch_size=16, num_draft_tokens=4) == 2
    # Gap (5..8) and beyond-last (17+): fall back to the static count.
    assert resolve_num_draft_tokens(sched, batch_size=5, num_draft_tokens=4) == 4
    assert resolve_num_draft_tokens(sched, batch_size=17, num_draft_tokens=4) == 4
    # No schedule: always static.
    assert resolve_num_draft_tokens(None, batch_size=99, num_draft_tokens=3) == 3


def test_disable_flag_forces_static(monkeypatch):
    """EASYDEL_DISABLE_DYNAMIC_SPEC=1 makes the strategy resolve the static K."""
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=4)
    runner = types.SimpleNamespace(max_num_reqs=2, max_num_seqs=2)
    strategy = DrafterSpeculation(
        runner=runner, drafter=drafter, num_draft_tokens=4, draft_schedule=[(1, 4, 1), (5, 1 << 30, 0)]
    )
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    assert strategy.resolve_step_draft_budget(2) == 1
    assert strategy.resolve_step_draft_budget(8) == 0
    monkeypatch.setenv("EASYDEL_DISABLE_DYNAMIC_SPEC", "1")
    assert strategy.resolve_step_draft_budget(2) == 4
    assert strategy.resolve_step_draft_budget(8) == 4


# ---------------------------------------------------------------------------
# Engine bootstrap: drafter=True default schedule vs explicit static configs
# ---------------------------------------------------------------------------


def test_drafter_true_auto_path_installs_default_schedule():
    """Plug-and-play drafter=True gets the measured never-a-regression schedule."""
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config(None)
    drafter = _build_drafter(model=model, drafter=True, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert isinstance(drafter, Qwen3_5MTPDrafter)
    assert drafter.num_draft_tokens_per_batch_size == default_auto_draft_schedule(4)
    assert drafter.num_draft_tokens_per_batch_size == [(1, 4, 4), (5, 1 << 30, 0)]


def test_drafter_true_respects_user_schedule():
    """drafter=True + an explicit config schedule keeps the user's schedule."""
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config({"num_draft_tokens_per_batch_size": [(1, 8, 2)]})
    drafter = _build_drafter(model=model, drafter=True, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert drafter.num_draft_tokens_per_batch_size == [(1, 8, 2)]


def test_explicit_static_config_stays_static():
    """An explicit enabled config without a schedule must NOT silently go dynamic."""
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config({"method": "mtp", "num_draft_tokens": 2})
    drafter = _build_drafter(model=model, drafter=None, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert isinstance(drafter, Qwen3_5MTPDrafter)
    assert drafter.num_draft_tokens_per_batch_size is None


def test_explicit_config_with_schedule_carries_it():
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config(
        {"method": "mtp", "num_draft_tokens": 4, "num_draft_tokens_per_batch_size": [(1, 2, 4), (3, 64, 0)]}
    )
    drafter = _build_drafter(model=model, drafter=None, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert drafter.num_draft_tokens_per_batch_size == [(1, 2, 4), (3, 64, 0)]


# ---------------------------------------------------------------------------
# Strategy behavior: K transitions across verify windows for one request
# ---------------------------------------------------------------------------

_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=64))


def _seed_hidden(seed: int) -> jnp.ndarray:
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(7), seed), (_HIDDEN,), dtype=jnp.float32)


def _row_index(drafter: Qwen3_5MTPDrafter, row: int) -> int:
    return int(np.asarray(drafter._mtp_cache.views[0].indexes).reshape(-1)[row])


def test_k_transitions_keep_persistent_mtp_rows_coherent(monkeypatch):
    """Same request across windows with K forced 4 -> 2 -> 0 -> 4.

    Draft counts follow the schedule; the K=0 window performs NO drafter call
    (its cache write index does not move — plain decode semantics: the
    sequence advances by the verified token only); every window rolls the
    persistent MTP KV row back to the request's committed length, so the
    write index equals ``committed + k_live`` after each drafted window
    regardless of the K used by earlier windows.
    """
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    monkeypatch.delenv("EASYDEL_DISABLE_DYNAMIC_SPEC", raising=False)
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=4, persist_kv=True)
    runner = types.SimpleNamespace(max_num_reqs=2, max_num_seqs=2)
    # Concurrency n=1 -> K=4, n=2 -> K=2, n=3 -> K=0 (schedule-driven forcing).
    strategy = DrafterSpeculation(
        runner=runner,
        drafter=drafter,
        num_draft_tokens=4,
        draft_schedule=[(1, 1, 4), (2, 2, 2), (3, 3, 0)],
    )

    # (concurrency, seed_position). seed_position advances by the committed
    # tokens (accepted + corrected) of the previous window; the committed MTP
    # length each window is seed_position - 10 (the first window's base).
    windows = [(1, 10), (2, 12), (3, 14), (1, 14)]
    expected_k = [4, 2, 0, 4]
    row = None
    indices: list[int] = []
    for (concurrency, pos), k_expected in zip(windows, expected_k, strict=True):
        assert strategy.resolve_step_draft_budget(concurrency) == k_expected
        drafts = strategy.draft_next(
            req_id="A",
            row_pos=0,
            seed_token=5 + pos,
            seed_position=pos,
            seed_hidden=_seed_hidden(pos),
            req_state=_REQ_STATE,
        )
        assert len(drafts) == k_expected, f"window@{pos} n={concurrency}: {len(drafts)} drafts != {k_expected}"
        if row is None and strategy._mtp_row_by_req:
            row = strategy._mtp_row_by_req["A"]
        indices.append(_row_index(drafter, row) if row is not None else 0)

    # Window 1: rollback 0 + 4 drafts. Window 2: rollback (12-10)=2 + 2 drafts.
    # Window 3 (K=0): untouched. Window 4: rollback (14-10)=4 + 4 drafts.
    assert indices == [4, 4, 4, 8], f"persistent MTP row index trajectory {indices}"


def test_lower_k_is_a_prefix_of_higher_k_drafts():
    """Greedy chained drafting: K=2 drafts equal the first 2 of K=4 drafts.

    The dynamic budget only truncates the draft stream — it cannot change the
    tokens drafted at shallower depths, so K transitions keep greedy outputs
    coherent (a shorter window verifies a prefix of what the longer window
    would have proposed).
    """
    model = make_tiny_model()
    runner = types.SimpleNamespace(max_num_reqs=2, max_num_seqs=2)

    def _draft_with_budget(k_live: int) -> list[int]:
        drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=4, persist_kv=True)
        strategy = DrafterSpeculation(
            runner=runner, drafter=drafter, num_draft_tokens=4, draft_schedule=[(1, 1, 4), (2, 2, 2)]
        )
        concurrency = 1 if k_live == 4 else 2
        assert strategy.resolve_step_draft_budget(concurrency) == k_live
        return strategy.draft_next(
            req_id="A",
            row_pos=0,
            seed_token=17,
            seed_position=9,
            seed_hidden=_seed_hidden(3),
            req_state=_REQ_STATE,
        )

    k4 = _draft_with_budget(4)
    k2 = _draft_with_budget(2)
    assert len(k4) == 4 and len(k2) == 2
    assert k2 == k4[:2], f"K=2 drafts {k2} are not a prefix of K=4 drafts {k4}"


def test_batched_draft_uses_live_budget_and_k0_skips_device_work():
    """begin_batched_draft drafts live-K tokens; a 0 budget returns the empty handle."""
    model = make_tiny_model()
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=4, persist_kv=True)
    runner = types.SimpleNamespace(max_num_reqs=4, max_num_seqs=4)
    strategy = DrafterSpeculation(
        runner=runner,
        drafter=drafter,
        num_draft_tokens=4,
        draft_schedule=[(1, 2, 2), (3, 1 << 30, 0)],
    )
    seeds = [("A", 0, 5, 4, _seed_hidden(0)), ("B", 1, 9, 10, _seed_hidden(1))]

    strategy.resolve_step_draft_budget(2)  # -> K=2
    out = strategy.finish_batched_draft(strategy.begin_batched_draft(seeds))
    assert {len(v) for v in out.values()} == {2}

    calls: list[int] = []
    orig = drafter.draft_batched

    def _spy(**kwargs):
        calls.append(int(kwargs["num_draft_tokens"]))
        return orig(**kwargs)

    drafter.draft_batched = _spy
    strategy.resolve_step_draft_budget(3)  # -> K=0
    out0 = strategy.finish_batched_draft(strategy.begin_batched_draft(seeds))
    assert all(v == [] for v in out0.values())
    assert calls == [], "K=0 budget must not invoke the batched drafter at all"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
