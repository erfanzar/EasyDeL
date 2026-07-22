"""Plug-and-play drafter auto-discovery for eSurge.

Contract under test: a user should get working MTP speculative decoding from
eSurge without constructing a drafter by hand —

* ``model.drafter("auto")`` resolves the model's own inline MTP head with the
  tuned defaults (4 draft tokens, persistent MTP KV);
* the engine's ``drafter=True`` path (bootstrap ``_build_drafter``) builds that
  same drafter from the loaded model, while the default (no drafter argument)
  keeps speculation OFF (no accidental MTP);
* ``EASYDEL_MTP_PERSIST_KV`` is a tri-state override on the strategy: unset →
  follow the drafter's ``persist_kv`` flag (default on), ``"0"`` → force the
  per-window clear, ``"1"`` → force persistence.
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
from easydel.inference.esurge.config import eSurgeDrafterConfig
from easydel.inference.esurge.engine.bootstrap import _build_drafter
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35 as make_tiny_model

_VOCAB = 256
_HIDDEN = 128




def test_model_drafter_auto_resolves_mtp_with_tuned_defaults():
    """model.drafter('auto') on an MTP model returns the inline MTP drafter (K=4, persist on)."""
    model = make_tiny_model()
    drafter = model.drafter(method="auto")
    assert isinstance(drafter, Qwen3_5MTPDrafter)
    assert drafter.num_draft_tokens == 4
    assert drafter.persist_kv is True


def test_engine_drafter_true_auto_discovers_mtp():
    """The engine's drafter=True path builds the model-owned MTP drafter with the tuned defaults."""
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config(None)
    drafter = _build_drafter(model=model, drafter=True, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert isinstance(drafter, Qwen3_5MTPDrafter)
    assert drafter.num_draft_tokens == 4
    assert drafter.persist_kv is True


def test_engine_default_keeps_speculation_off():
    """No drafter argument (and no config) must NOT accidentally enable MTP."""
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config(None)
    drafter = _build_drafter(model=model, drafter=None, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert drafter is None


def test_engine_drafter_config_mapping_overrides_k():
    """A declarative drafter config picks the family + K without a drafter class."""
    model = make_tiny_model()
    cfg = eSurgeDrafterConfig.coerce_config({"method": "mtp", "num_draft_tokens": 2})
    drafter = _build_drafter(model=model, drafter=None, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)
    assert isinstance(drafter, Qwen3_5MTPDrafter)
    assert drafter.num_draft_tokens == 2


def test_elarge_drafter_section_chains_to_auto_discovery():
    """The eLarge `esurge: {drafter: {enabled: true}}` section reaches the same auto path.

    eLarge coerces its ``esurge.drafter`` section through ``to_esurge_kwargs``
    into the engine's ``drafter_config``; ``enabled`` with no method must
    normalize to ``method="auto"`` and build the model-owned MTP drafter with
    the tuned defaults.
    """
    from easydel.infra.elarge.builders import to_esurge_kwargs

    cfg = {"model": {"name_or_path": "stub/tiny"}, "esurge": {"drafter": {"enabled": True}}}
    dc = to_esurge_kwargs(cfg)["drafter_config"]
    assert dc["enabled"] is True
    assert dc["method"] == "auto"
    drafter = _build_drafter(model=make_tiny_model(), drafter=None, drafter_config=dc, loading_kwargs=None, dtype=jnp.float32)
    assert isinstance(drafter, Qwen3_5MTPDrafter)
    assert drafter.num_draft_tokens == 4
    assert drafter.persist_kv is True


def test_engine_rejects_explicit_object_plus_enabled_config():
    """Explicit drafter object + enabled drafter_config is ambiguous → ValueError."""
    model = make_tiny_model()
    explicit = Qwen3_5MTPDrafter(model, num_draft_tokens=1)
    cfg = eSurgeDrafterConfig.coerce_config({"method": "mtp"})
    with pytest.raises(ValueError):
        _build_drafter(model=model, drafter=explicit, drafter_config=cfg, loading_kwargs=None, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# EASYDEL_MTP_PERSIST_KV tri-state override (behavioral: cache write index)
# ---------------------------------------------------------------------------

_REQ_STATE = types.SimpleNamespace(sampling_params=SamplingParams(temperature=0.0, max_tokens=32))


def _make_strategy(drafter: Qwen3_5MTPDrafter, n_rows: int, k: int) -> DrafterSpeculation:
    runner = types.SimpleNamespace(max_num_reqs=n_rows, max_num_seqs=n_rows)
    return DrafterSpeculation(runner=runner, drafter=drafter, num_speculative_tokens=k)


def _row_index(drafter: Qwen3_5MTPDrafter, row: int) -> int:
    return int(np.asarray(drafter._mtp_cache.views[0].indexes).reshape(-1)[row])


def _seed_hidden(seed: int) -> jnp.ndarray:
    """A (hidden,)-shaped seed hidden state for ``draft_next`` (it adds the B/S axes)."""
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(7), seed), (_HIDDEN,), dtype=jnp.float32)


def _two_windows(drafter: Qwen3_5MTPDrafter, k: int) -> int:
    """Run two verify windows for one request and return the row-0 cache index.

    Window 1 seeds at position 10 (fresh request → committed 0), window 2 at
    position 12 (committed 2). With persistence the second window rolls back to
    2 and appends k drafts (index 2 + k); with the per-window clear it restarts
    (index k).
    """
    strategy = _make_strategy(drafter, n_rows=2, k=k)
    for pos, seed in ((10, 5), (12, 7)):
        drafts = strategy.draft_next(
            req_id="A",
            row_pos=0,
            seed_token=seed,
            seed_position=pos,
            seed_hidden=_seed_hidden(pos),
            req_state=_REQ_STATE,
        )
        assert len(drafts) == k
    return _row_index(drafter, 0)


def test_persist_env_unset_follows_drafter_default_on(monkeypatch):
    monkeypatch.delenv("EASYDEL_MTP_PERSIST_KV", raising=False)
    model = make_tiny_model()
    k = 2
    assert _two_windows(Qwen3_5MTPDrafter(model, num_draft_tokens=k), k) == 2 + k


def test_persist_env_zero_forces_per_window_clear(monkeypatch):
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "0")
    model = make_tiny_model()
    k = 2
    assert _two_windows(Qwen3_5MTPDrafter(model, num_draft_tokens=k), k) == k


def test_persist_env_one_forces_persist_for_optout_drafter(monkeypatch):
    monkeypatch.setenv("EASYDEL_MTP_PERSIST_KV", "1")
    model = make_tiny_model()
    k = 2
    assert _two_windows(Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=False), k) == 2 + k


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
