# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the checkpoint commit marker, retention, data state and on_step policy.

These use a fake array manager, so they exercise only the Checkpointer's
bookkeeping and filesystem layout. Real TensorStore round trips are covered in
``test_checkpointer.py``.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from spectrax.serialization import (
    Checkpointer,
    CheckpointInterval,
    _fs,
    commit_checkpoint,
    find_latest_checkpoint,
    is_checkpoint_committed,
    list_checkpoints,
    read_data_state,
    write_data_state,
)
from spectrax.serialization import checkpointer as ckpt_mod

pytestmark = pytest.mark.cpu_ok


class _FakeGlobalManager:
    def wait_until_finished(self) -> None:
        return None


class _FakeManager:
    """Writes one JSON file per prefix instead of TensorStore arrays."""

    def __init__(self, fail_on_prefix: str | None = None) -> None:
        self.global_manager = _FakeGlobalManager()
        self.fail_on_prefix = fail_on_prefix
        self.saved_steps: list[int] = []

    def save_pytree(self, *, pytree, path, prefix, mesh, dtype, extras, write_index):
        if prefix == self.fail_on_prefix:
            raise RuntimeError(f"simulated crash while writing {prefix}")
        self.saved_steps.append(int(extras["step"]))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / f"{prefix}.json").write_text(json.dumps(extras))


def _checkpointer(tmp_path, manager=None, **kwargs) -> Checkpointer:
    return Checkpointer(
        str(tmp_path),
        save_interval=kwargs.pop("save_interval", None),
        step_policies=kwargs.pop("step_policies", ()),
        manager=manager or _FakeManager(),
        **kwargs,
    )


def _write_meta(path: Path, **meta) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta.setdefault("timestamp", dt.datetime.now().isoformat())
    (path / "metadata.json").write_text(json.dumps(meta))


def test_time_based_save_rearms_after_saving(tmp_path):
    manager = _FakeManager()
    ckpt = _checkpointer(tmp_path, manager, save_interval=timedelta(hours=1))
    ckpt._last_save_time = dt.datetime.now() - timedelta(hours=2)
    for step in (1, 2, 3):
        ckpt.on_step(mesh=None, pytree={"w": 1}, step=step)
    assert manager.saved_steps == [1]
    assert is_checkpoint_committed(str(tmp_path / "run-1"))


def test_on_step_commits_once_with_callbacks_and_keeps_extras(tmp_path, monkeypatch):
    commits: list[str] = []
    real_commit = ckpt_mod.commit_checkpoint
    monkeypatch.setattr(
        ckpt_mod, "commit_checkpoint", lambda path, **kw: (commits.append(path), real_commit(path, **kw))
    )
    ckpt = _checkpointer(tmp_path, step_policies=(CheckpointInterval(every=5),))
    extras = {"run": "abc"}
    seen: list[str] = []
    ckpt.on_step(mesh=None, pytree={"w": 1}, step=5, extras=extras, true_callbacks=[lambda p, m, meta: seen.append(p)])
    assert commits == seen == [str(tmp_path / "run-5")]
    assert extras == {"run": "abc"}


def test_save_pytree_does_not_mutate_extras(tmp_path):
    extras = {"a": 1}
    _checkpointer(tmp_path).save_pytree({"w": 1}, "model", step=3, extras=extras)
    assert extras == {"a": 1}
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "run-3")


def test_uncommitted_directory_is_invisible_and_legacy_is_visible(tmp_path):
    _write_meta(tmp_path / "run-1", step=1, is_temporary=False)
    _write_meta(tmp_path / "run-9", step=9, is_temporary=False, commit_protocol=1)
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "run-1")
    assert find_latest_checkpoint(str(tmp_path), include_uncommitted=True) == str(tmp_path / "run-9")
    infos = {i.step: i for i in list_checkpoints(str(tmp_path), include_uncommitted=True)}
    assert (infos[1].legacy, infos[1].committed) == (True, True)
    assert (infos[9].legacy, infos[9].committed) == (False, False)


def test_crash_mid_save_keeps_previous_checkpoint_latest(tmp_path):
    _checkpointer(tmp_path).save(1, {"model": {"w": 1}, "opt": {"m": 1}})
    crashing = _checkpointer(tmp_path, _FakeManager(fail_on_prefix="opt"))
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing.save(2, {"model": {"w": 2}, "opt": {"m": 2}})
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "run-1")
    assert not is_checkpoint_committed(str(tmp_path / "run-2"))


def test_overwrite_hides_old_commit_until_recommitted(tmp_path):
    ckpt = _checkpointer(tmp_path)
    ckpt.save(4, {"model": {"w": 1}})
    with pytest.raises(RuntimeError):
        _checkpointer(tmp_path, _FakeManager(fail_on_prefix="model")).save(4, {"model": {"w": 2}})
    assert not is_checkpoint_committed(str(tmp_path / "run-4"))
    ckpt.save(4, {"model": {"w": 3}})
    assert is_checkpoint_committed(str(tmp_path / "run-4"))


def test_retention_keeps_newest_and_removes_crashed_saves(tmp_path):
    _write_meta(tmp_path / "run-0", step=0, is_temporary=False, commit_protocol=1)
    _write_meta(tmp_path / "legacy", step=-5, is_temporary=False)
    ckpt = _checkpointer(tmp_path, max_to_keep=2)
    for step in (1, 2, 3, 4):
        ckpt.save(step, {"model": {"w": step}})
    ckpt.wait_until_finished()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["legacy", "run-3", "run-4"]


def test_temporary_checkpoint_is_replaced_by_next_save(tmp_path):
    ckpt = _checkpointer(tmp_path)
    ckpt.save(1, {"model": {"w": 1}}, temporary=True)
    ckpt.save(2, {"model": {"w": 2}})
    ckpt.wait_until_finished()
    assert not (tmp_path / "run-1").exists()
    assert ckpt.latest_checkpoint().step == 2


def test_max_to_keep_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        _checkpointer(tmp_path, max_to_keep=0)


def test_data_state_round_trip(tmp_path):
    ckpt = _checkpointer(tmp_path)
    state = {"position": 1234, "rng": b"\x00\x01\xff", "shard": [np.int64(3), 4], "nested": {"epoch": 2}}
    path = ckpt.save(7, {"model": {"w": 1}}, data_state=state)
    expected = {"position": 1234, "rng": b"\x00\x01\xff", "shard": [3, 4], "nested": {"epoch": 2}}
    assert read_data_state(path) == expected
    assert ckpt.load_data_state() == expected


def test_unserializable_data_state_leaves_no_commit(tmp_path):
    with pytest.raises(TypeError):
        _checkpointer(tmp_path).save(1, {"model": {"w": 1}}, data_state={"bad": object()})
    assert find_latest_checkpoint(str(tmp_path)) is None


def test_remote_discovery_skips_uncommitted():
    base = f"memory://spx-ckpt-{uuid.uuid4().hex}"
    for step in (1, 2):
        path = _fs.joinpath(base, f"run-{step}")
        ckpt_mod._write_checkpoint_metadata(path, step=step, is_temporary=False)
        write_data_state(path, {"pos": step})
        commit_checkpoint(path, step=step)
    ckpt_mod._write_checkpoint_metadata(_fs.joinpath(base, "run-3"), step=3, is_temporary=False)

    latest = find_latest_checkpoint(base)
    assert latest is not None and latest.endswith("run-2")
    assert read_data_state(latest) == {"pos": 2}
    assert [i.step for i in list_checkpoints(base)] == [1, 2]
