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

"""Adaptive (early-exit) MTP draft length for eSurge speculative decoding.

The fixed-K MTP drafter runs ``num_draft_tokens`` sequential MTP forwards
per verify window regardless of how confident the drafter is; deep recursive
drafts are weak and mostly rejected, so a large K overpays on the drafter cost.
:meth:`Qwen3_5MTPDrafter.draft_adaptive` instead runs the whole ``k_max``-step
recursion as a single ``jax.lax.while_loop`` whose ``cond`` early-exits once the
MTP's top-1 confidence drops below a threshold, so it executes FEWER MTP forwards
on device when the drafter is unsure — with no per-step host sync.

These tests are CPU-only and cover the mechanics (correctness of HOW MANY tokens
are proposed; the speedup is TPU-measured):

* ``conf_threshold <= 0`` disables early exit -> the loop drafts exactly ``k_max``
  and is byte-identical to ``k_max`` chained :meth:`draft` calls (OFF == today);
* a high threshold early-exits (``valid_count < k_max``) and the returned prefix
  still equals the fixed-K prefix (same argmax path, only truncated);
* the data-dependent-trip-count while_loop compiles ONCE across thresholds and
  early-exit points (no per-count recompile);
* per-request persist-with-rollback (row-targeted, batch > 1) keeps working
  inside the loop;
* the strategy's ``draft_next`` routes greedy drafts through the adaptive path
  only when ``EASYDEL_SPEC_ADAPTIVE_CONF > 0`` and otherwise keeps the fixed-K
  Python loop (default OFF).
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
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35 as make_tiny_model

_VOCAB = 256
_HIDDEN = 128




def _seed_hidden(i: int) -> jnp.ndarray:
    """A (hidden,)-shaped seed hidden state (draft/draft_adaptive add the B/S axes)."""
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(11), i), (_HIDDEN,), dtype=jnp.float32)


def _row_index(drafter: Qwen3_5MTPDrafter, row: int) -> int:
    """Return the first MTP layer view's write index for cache batch ``row``."""
    return int(np.asarray(drafter._mtp_cache.views[0].indexes).reshape(-1)[row])


def _fixed_k_draft(
    drafter: Qwen3_5MTPDrafter,
    *,
    seed_token: int,
    seed_position: int,
    seed_hidden,
    k: int,
    row_pos: int | None = None,
) -> list[int]:
    """Chain ``k`` greedy :meth:`draft` calls (the fixed-K reference the loop replaces)."""
    cur_token = jnp.asarray([[int(seed_token)]], dtype=jnp.int32)
    cur_pos = int(seed_position)
    hidden = jnp.asarray(seed_hidden)
    if hidden.ndim == 1:
        hidden = hidden[None, None, :]
    drafts: list[int] = []
    for _ in range(int(k)):
        kwargs = {
            "input_ids": cur_token,
            "target_hidden_states": hidden,
            "position_ids": jnp.asarray([[cur_pos]], dtype=jnp.int32),
        }
        if row_pos is not None:
            kwargs["row_pos"] = int(row_pos)
        step = drafter.draft(**kwargs)
        tok = int(np.asarray(step.token_ids).reshape(-1)[0])
        drafts.append(tok)
        cur_token = jnp.asarray([[tok]], dtype=jnp.int32)
        cur_pos += 1
        if step.hidden_states is not None:
            hidden = step.hidden_states[:, -1:, :]
    return drafts


def _adaptive(drafter: Qwen3_5MTPDrafter, **kwargs) -> tuple[int, list[int]]:
    """Call draft_adaptive and read back (valid_count, valid draft tokens)."""
    drafts_dev, valid_dev = drafter.draft_adaptive(**kwargs)
    drafts_host, valid_host = jax.device_get((drafts_dev, valid_dev))
    valid = int(np.asarray(valid_host).reshape(-1)[0])
    return valid, [int(t) for t in np.asarray(drafts_host).reshape(-1)[:valid]]


def test_adaptive_fixed_k_matches_chained_draft():
    """conf_threshold<=0 -> exactly k_max drafts, byte-identical to chained draft()."""
    model = make_tiny_model()
    k = 4
    seed_token, seed_pos, h = 5, 3, _seed_hidden(0)

    ref = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    ref.reset(1)
    fixed = _fixed_k_draft(ref, seed_token=seed_token, seed_position=seed_pos, seed_hidden=h, k=k)

    for thr in (0.0, -1.0):
        ad = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
        ad.reset(1)
        valid, drafts = _adaptive(
            ad,
            seed_token=seed_token,
            seed_hidden=h,
            seed_position=seed_pos,
            k_max=k,
            conf_threshold=thr,
        )
        assert valid == k, f"conf_threshold={thr} (<=0) must draft exactly k_max ({valid} != {k})"
        assert drafts == fixed, f"adaptive fixed-K must match chained draft(): {drafts} vs {fixed}"


def test_adaptive_early_exit_returns_fewer():
    """A high threshold early-exits: valid_count < k_max, prefix matches fixed-K."""
    model = make_tiny_model()
    k = 6
    seed_token, seed_pos, h = 7, 2, _seed_hidden(1)

    ref = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    ref.reset(1)
    fixed = _fixed_k_draft(ref, seed_token=seed_token, seed_position=seed_pos, seed_hidden=h, k=k)

    ad = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    ad.reset(1)
    valid, drafts = _adaptive(
        ad,
        seed_token=seed_token,
        seed_hidden=h,
        seed_position=seed_pos,
        k_max=k,
        conf_threshold=0.999,
    )
    # A random-init tiny model over 256 tokens is nowhere near 0.999 confident, so
    # the loop stops after the first (low-confidence) token is proposed.
    assert 1 <= valid < k, f"high threshold must early-exit (1 <= {valid} < {k})"
    assert drafts == fixed[:valid], "the early-exit prefix must equal the fixed-K prefix (same argmax path)"


def test_adaptive_compiles_once():
    """The data-dependent while_loop compiles once across thresholds/exit points."""
    model = make_tiny_model()
    k = 4
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    for i, thr in enumerate([0.0, 0.3, 0.9, 0.999, -1.0]):
        drafter.reset(1)  # same-shape fresh cache -> compile key is stable
        _adaptive(
            drafter,
            seed_token=3 + i,
            seed_hidden=_seed_hidden(i),
            seed_position=i,
            k_max=k,
            conf_threshold=thr,
        )
    assert len(drafter._jit_adaptive) == 1, (
        "the early-exit while_loop must compile once across thresholds and early-exit "
        f"points (conf_threshold + row are traced args); got {len(drafter._jit_adaptive)} entries"
    )


def test_adaptive_row_targeted_matches_chained():
    """Row-targeted adaptive fixed-K path equals chained draft(row_pos=...)."""
    model = make_tiny_model()
    n, k = 2, 3
    ref = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    ref.reset(n)
    fixed = _fixed_k_draft(ref, seed_token=5, seed_position=4, seed_hidden=_seed_hidden(0), k=k, row_pos=0)

    ad = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    ad.reset(n)
    valid, drafts = _adaptive(
        ad,
        seed_token=5,
        seed_hidden=_seed_hidden(0),
        seed_position=4,
        k_max=k,
        conf_threshold=0.0,
        row_pos=0,
    )
    assert valid == k
    assert drafts == fixed, f"row-targeted adaptive must match chained draft(row_pos=0): {drafts} vs {fixed}"


def test_adaptive_persist_row_targeted_batch2():
    """Per-request persist-with-rollback works inside the adaptive loop (batch 2)."""
    model = make_tiny_model()
    n, k = 2, 3
    drafter = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
    drafter.reset(n)  # pool-sized cache (batch n)

    # Row 0 window 1.
    v0, d0 = _adaptive(
        drafter, seed_token=5, seed_hidden=_seed_hidden(0), seed_position=4, k_max=k, conf_threshold=0.0, row_pos=0
    )
    assert v0 == k
    assert _row_index(drafter, 0) == k, "row 0 advanced by k drafts"
    assert _row_index(drafter, 1) == 0, "row 1 untouched by the row-0 window"

    # Row 1 window 1 must not disturb row 0.
    v1, _d1 = _adaptive(
        drafter, seed_token=9, seed_hidden=_seed_hidden(1), seed_position=10, k_max=k, conf_threshold=0.0, row_pos=1
    )
    assert v1 == k
    assert _row_index(drafter, 1) == k, "row 1 advanced by k drafts"
    assert _row_index(drafter, 0) == k, "row 0 retained its K/V across the interleaved row-1 window"

    # Roll row 0 back to committed=1, then a second window: only row 0 changes.
    drafter.rollback_to(1, row_pos=0)
    assert _row_index(drafter, 0) == 1
    _adaptive(
        drafter,
        seed_token=int(d0[0]),
        seed_hidden=_seed_hidden(2),
        seed_position=5,
        k_max=k,
        conf_threshold=0.0,
        row_pos=0,
    )
    assert _row_index(drafter, 0) == 1 + k, "row 0 rolled back to committed=1 then advanced by k"
    assert _row_index(drafter, 1) == k, "row 1 independent of the row-0 second window"

    assert len(drafter._jit_adaptive) == 1, "row is a traced arg; row targeting must not add compile entries"


_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=32))


def _make_strategy(drafter: Qwen3_5MTPDrafter, n_rows: int, k: int) -> DrafterSpeculation:
    """A DrafterSpeculation over a stub runner exposing only the pool size.

    Inline-MTP ``draft_next`` (persist-with-rollback) reads only ``max_num_reqs``
    off the runner — the target K/V payload path short-circuits for inline MTP —
    so a namespace stub is enough to exercise the adaptive routing + row
    bookkeeping without standing up a full runner.
    """
    runner = types.SimpleNamespace(max_num_reqs=n_rows, max_num_seqs=n_rows)
    return DrafterSpeculation(runner=runner, drafter=drafter, num_draft_tokens=k)


def test_draft_next_adaptive_engages_via_env():
    """draft_next routes greedy drafts to the adaptive loop only when the env > 0."""
    model = make_tiny_model()
    n, k = 2, 4
    prev_conf = os.environ.get("EASYDEL_SPEC_ADAPTIVE_CONF")
    prev_persist = os.environ.get("EASYDEL_MTP_PERSIST_KV")
    os.environ["EASYDEL_MTP_PERSIST_KV"] = "1"  # exercise the row-targeted persist path
    try:
        # env OFF -> fixed-K Python loop; adaptive while_loop never built.
        os.environ["EASYDEL_SPEC_ADAPTIVE_CONF"] = "0"
        d_off = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
        s_off = _make_strategy(d_off, n, k)
        off = s_off.draft_next(
            req_id="A", row_pos=0, seed_token=5, seed_position=4, seed_hidden=_seed_hidden(0), req_state=_REQ_STATE
        )
        assert len(off) == k, f"env off must use the fixed-K loop (k drafts); got {len(off)}"
        assert len(d_off._jit_adaptive) == 0, "adaptive loop must not be built when the env is off"

        # env ON, tiny threshold -> adaptive full length, byte-identical to fixed-K.
        os.environ["EASYDEL_SPEC_ADAPTIVE_CONF"] = "1e-9"
        d_on = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
        s_on = _make_strategy(d_on, n, k)
        on = s_on.draft_next(
            req_id="A", row_pos=0, seed_token=5, seed_position=4, seed_hidden=_seed_hidden(0), req_state=_REQ_STATE
        )
        assert len(d_on._jit_adaptive) == 1, "adaptive loop must run when the env > 0"
        assert on == off, f"adaptive full-length (tiny threshold) must match fixed-K draft_next: {on} vs {off}"

        # env ON, high threshold -> early exit, variable-length draft list.
        os.environ["EASYDEL_SPEC_ADAPTIVE_CONF"] = "0.999"
        d_hi = Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True)
        s_hi = _make_strategy(d_hi, n, k)
        hi = s_hi.draft_next(
            req_id="A", row_pos=0, seed_token=5, seed_position=4, seed_hidden=_seed_hidden(0), req_state=_REQ_STATE
        )
        assert 1 <= len(hi) < k, f"high threshold must early-exit through draft_next; got {hi}"
        assert hi == off[: len(hi)], "early-exit drafts must be the fixed-K prefix"
    finally:
        for name, prev in (("EASYDEL_SPEC_ADAPTIVE_CONF", prev_conf), ("EASYDEL_MTP_PERSIST_KV", prev_persist)):
            if prev is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = prev


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
