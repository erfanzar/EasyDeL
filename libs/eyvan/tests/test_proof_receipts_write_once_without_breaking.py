# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Proof receipts are write-once without breaking mutable soak checkpoints."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from eyvan.sdk.commands.proof import _ProofRun, evaluate_secret_delivery, proof
from eyvan.sdk.commands.receipts import write_receipt_atomic


def test_alias_refusal_happens_before_any_plane_rpc(tmp_path: Path) -> None:
    calls = []

    class Client:
        def request(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("a refused proof must not poll the plane")

    value_file = tmp_path / "synthetic-value.txt"
    value_file.write_text("synthetic-value", encoding="utf-8")
    target = tmp_path / "output.json"
    session = SimpleNamespace(
        client=Client(), workload_scope=lambda **kwargs: SimpleNamespace(org_id="o", project_id="p")
    )
    result = CliRunner().invoke(
        proof,
        ["secret", "wl-proof", "--value-file", str(value_file), "--receipt", str(target), "--evidence", str(target)],
        obj=session,
    )
    assert result.exit_code == 3, result.output
    assert not calls
    assert not target.exists()


def test_record_preserves_intervening_receipt_and_retained_evidence(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    evidence = tmp_path / "evidence.json"
    observations = {"workload": {"state": "succeeded"}}
    run = _ProofRun(
        "P4", evaluate_secret_delivery, receipt=str(target), evidence=str(evidence), observations=observations
    )
    run.preflight()
    sentinel = b"another run's immutable receipt\n"
    target.write_bytes(sentinel)
    result = evaluate_secret_delivery([], observations["workload"], "", secret_value="synthetic-value")
    with pytest.raises(FileExistsError):
        run.record(result)
    assert target.read_bytes() == sentinel
    assert json.loads(evidence.read_text(encoding="utf-8")) == observations
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_checkpoint_writer_keeps_existing_replacement_behavior(tmp_path: Path) -> None:
    target = tmp_path / "soak.json"
    write_receipt_atomic(target, {"status": "running"})
    write_receipt_atomic(target, {"status": "completed"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "completed"}
    assert not list(tmp_path.glob(".soak.json.*.tmp"))


@pytest.mark.parametrize("replace_existing", [True, False])
def test_invalid_json_is_not_published_and_own_temporary_is_cleaned(tmp_path: Path, replace_existing: bool) -> None:
    target = tmp_path / "receipt.json"
    with pytest.raises(ValueError):
        write_receipt_atomic(target, {"value": float("nan")}, replace_existing=replace_existing)
    assert not target.exists()
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_concurrent_immutable_publication_has_exactly_one_complete_winner(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    barrier = Barrier(2)
    documents = [{"run": index, "observations": "x" * 65536} for index in range(2)]

    def publish(index: int) -> tuple[int, bool]:
        barrier.wait(timeout=5)
        try:
            write_receipt_atomic(target, documents[index], replace_existing=False)
        except FileExistsError:
            return index, False
        return index, True

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, range(2)))
    winners = [index for index, succeeded in results if succeeded]
    assert len(winners) == 1
    assert json.loads(target.read_text(encoding="utf-8")) == documents[winners[0]]
    assert target.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))
