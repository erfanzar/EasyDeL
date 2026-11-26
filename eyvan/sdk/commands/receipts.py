# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Immutable receipts for a hardware proof.

``docs/proofs.md`` records each proof's outcome in prose, and prose has no way
to state *which* build a run judged. Every component of this system reports one
version string, so "P5 passed" is a claim about a candidate nobody can identify
afterwards -- which is why that page says no historical run qualifies the current
one. A spend that cannot be attributed to a build has to be repeated.

A receipt is what makes the attribution executable. It records the verdict, the
exact criterion that produced it (by digest of the function's own source), the
build of the client that ran it and of the plane it polled, when it ran, and --
the field this package cares about most -- what it does **not** establish.

**Unverified dimensions are computed, not authored.** A receipt whose plane
never reported a build says so; one written from a dirty checkout says the tree
rather than the commit is the identity. Nobody remembers to write those down at
the end of a spend, which is exactly when they matter, so they are derived from
the same observations the verdict is.

**A receipt is not a signature.** ``receipt_digest`` detects a receipt that was
edited or truncated after the fact. It does not detect one that was rewritten
wholesale, because there is no key here to sign with, and this module says so in
``unverified`` rather than implying a guarantee it cannot keep.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import socket
import subprocess
import tempfile
import typing as tp
from datetime import UTC, datetime
from pathlib import Path

from eyvan._internal.build_identity import build_identity

#: Bumped when a field's meaning changes, so a reader never interprets an old
#: receipt under new rules. Additive fields do not bump it.
RECEIPT_SCHEMA = "eyvan.proof-receipt/1"

#: Command-line options whose *value* is a credential or a path to one. The
#: receipt is written to be readable, and an argv is the most reliable way for a
#: token to end up in a durable file nobody thinks of as secret.
_REDACTED_OPTIONS = frozenset(
    {
        "--token",
        "--password",
        "--secret",
        "--value",
        "--value-file",
        "--token-file",
        "--bootstrap-token",
        "--api-key",
    }
)

_REDACTED = "<redacted>"

#: Ceiling on a subprocess used only to read local git state. A hung git costs
#: the receipt its commit field, never the receipt itself.
_GIT_TIMEOUT_S = 10.0


def _now() -> str:
    return datetime.now(UTC).isoformat()


def write_receipt_atomic(path: Path | str, receipt: tp.Mapping[str, object], *, replace_existing: bool = True) -> None:
    """Fsync and atomically publish one finite JSON receipt.

    Mutable soak checkpoints retain replacement by default. Immutable proof
    receipts set ``replace_existing=False``: a same-directory hard link publishes
    the complete file only if the target is absent, including at publication time.

    ``allow_nan=False`` is load-bearing: a NaN would serialize as a bare token
    that is not JSON, and every reader of the evidence would fail on the one run
    that produced an unrepresentable number.
    """

    target = Path(path)
    parent = target.parent
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace_existing:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
        try:
            directory = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def canonical_digest(document: tp.Mapping[str, object]) -> str:
    """Return the digest of ``document`` under one canonical serialization.

    Sorted keys and a fixed separator, so two receipts holding the same facts
    agree regardless of how either was assembled.
    """

    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def criterion_identity(evaluator: tp.Callable[..., object]) -> dict[str, object]:
    """Identify the code that decided a verdict.

    The criterion is the thing under change: a proof rerun after its evaluator
    was loosened is not the same proof, and a receipt that recorded only the
    proof's name would present the two as interchangeable. Hashing the
    function's own source is what lets a later reader refuse that.

    A source that cannot be read -- a frozen or generated callable -- gives a
    ``None`` digest and a stated reason rather than a digest of something else.
    """

    module = getattr(evaluator, "__module__", "") or ""
    qualname = getattr(evaluator, "__qualname__", None) or getattr(evaluator, "__name__", "<anonymous>")
    try:
        source = inspect.getsource(evaluator)
    except (OSError, TypeError) as error:
        return {
            "function": f"{module}.{qualname}" if module else qualname,
            "source_digest": None,
            "source_detail": f"the criterion's source could not be read: {error}",
        }
    normalized = inspect.cleandoc(source).encode("utf-8")
    return {
        "function": f"{module}.{qualname}" if module else qualname,
        "source_digest": f"sha256:{hashlib.sha256(normalized).hexdigest()}",
        "source_detail": "",
    }


def redact_command(argv: tp.Sequence[str]) -> list[str]:
    """Return ``argv`` with credential values replaced.

    Both spellings are handled, because a receipt that redacted ``--token x``
    and kept ``--token=x`` would be worse than one that redacted neither: it
    reads as safe.
    """

    redacted: list[str] = []
    expect_value = False
    for argument in argv:
        if expect_value:
            redacted.append(_REDACTED)
            expect_value = False
            continue
        if argument in _REDACTED_OPTIONS:
            redacted.append(argument)
            expect_value = True
            continue
        name, separator, _value = argument.partition("=")
        if separator and name in _REDACTED_OPTIONS:
            redacted.append(f"{name}={_REDACTED}")
            continue
        redacted.append(argument)
    if expect_value:
        # A trailing option whose value was never supplied. Nothing to redact,
        # and nothing to invent.
        pass
    return redacted


def _git(arguments: tp.Sequence[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_identity(root: Path | None = None) -> dict[str, object]:
    """Describe the checkout a client was run from, if it was one.

    A wheel installed on a head has no commit, and that is not a defect: the
    build digest is the identity there. Reporting ``None`` keeps the two cases
    distinguishable, where an empty string would make a wheel look like a
    checkout at an unnamed revision.
    """

    directory = Path(root) if root is not None else Path(__file__).resolve().parent
    inside = _git(["rev-parse", "--is-inside-work-tree"], cwd=directory)
    if inside != "true":
        return {"commit": None, "clean": None, "detail": "not run from a git checkout"}
    commit = _git(["rev-parse", "HEAD"], cwd=directory)
    if commit is None:
        return {"commit": None, "clean": None, "detail": "the checkout has no resolvable HEAD"}
    status = _git(["status", "--porcelain"], cwd=directory)
    if status is None:
        return {"commit": commit, "clean": None, "detail": "the working tree state could not be read"}
    return {"commit": commit, "clean": status == "", "detail": ""}


def file_evidence(path: Path | str) -> dict[str, object]:
    """Describe a written evidence file by size and digest.

    The receipt names evidence rather than embedding it: raw plane documents
    carry fields this module has no way to exhaustively redact, and a receipt is
    meant to be readable by people who are not entitled to all of them.
    """

    target = Path(path)
    digest = hashlib.sha256()
    size = 0
    with open(target, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return {
        "path": str(target),
        "media_type": "application/json",
        "size_bytes": size,
        "sha256": f"sha256:{digest.hexdigest()}",
    }


def unverified_dimensions(
    *,
    plane_build: tp.Mapping[str, object] | None,
    git: tp.Mapping[str, object],
    evidence: tp.Mapping[str, object] | None,
    fleet_builds: tp.Mapping[str, object] | None = None,
) -> list[str]:
    """State what this receipt does not establish.

    Derived rather than authored. The end of a spend is when nobody writes
    caveats down, and a receipt missing them reads as a stronger claim than the
    run supports -- which is the failure mode that made ``docs/proofs.md`` say
    no historical run qualifies the current candidate.
    """

    gaps: list[str] = []
    if plane_build is None:
        gaps.append(
            "the build of the control plane: it did not report one, so this receipt "
            "cannot say the plane polled is the build named here"
        )
    elif not plane_build.get("source_digest"):
        gaps.append(
            "the exact build of the control plane: it reported a version but no source digest "
            f"({plane_build.get('digest_detail') or 'no reason given'})"
        )
    gaps.extend(_agent_build_gaps(fleet_builds))
    if git.get("commit") is None:
        gaps.append(f"the candidate commit: {git.get('detail') or 'unknown'}")
    elif git.get("clean") is False:
        gaps.append("a released candidate: the working tree was dirty, so the build digest is the only identity")
    elif git.get("clean") is None:
        gaps.append("whether the checkout was clean: its working-tree state could not be read")
    if evidence is None:
        gaps.append("the raw observations: they were not retained, so only the composed checks can be re-read")
    gaps.append("cryptographic provenance: this receipt is unsigned and its digest detects editing, not forgery")
    return gaps


def _agent_build_gaps(fleet_builds: tp.Mapping[str, object] | None) -> list[str]:
    """State what the fleet's own builds do or do not establish.

    This is the dimension a receipt could not close at all until the agents
    learned to describe themselves, and the reason it mattered is on the record:
    a proof was run against a stale agent and read as evidence. A sweep that
    reaches every agent and finds one build closes it. Anything less says which
    part is missing, because an agent that did not answer is not an agent that
    agreed.

    The native worker inside each cell is still not identified by any of this.
    It is installed from the wheel the head staged, so it is *expected* to match
    the plane, but expected is not observed and this keeps saying so.
    """

    if fleet_builds is None:
        return [
            "the build of the cell agent and the worker that ran the observed work; "
            "only the client and the control plane are identified here"
        ]
    agents = fleet_builds.get("agents")
    unreachable = fleet_builds.get("unreachable")
    agreement = fleet_builds.get("in_agreement")
    counted = len(agents) if isinstance(agents, list) else 0
    missed = len(unreachable) if isinstance(unreachable, list) else 0
    gaps: list[str] = []
    if agreement is True:
        # Nothing to add. Agreement here means every agent *and* every native
        # worker of every scanned cell reported this plane's build, which is the
        # whole chain from the client that submitted to the process that ran it.
        return gaps
    if agreement is False:
        gaps.append(
            "nothing about agent builds can be assumed from this run: at least one cell agent "
            "was running different code than the control plane while it ran"
        )
        return gaps
    gaps.append(
        f"whether the fleet's agents ran this build: {counted} answered and {missed} could not be reached, "
        "so nothing disagreed rather than everything agreeing"
    )
    return gaps


def build_proof_receipt(
    *,
    proof: str,
    verdict: str,
    exit_status: int,
    checks: tp.Sequence[tp.Mapping[str, object]],
    criterion: tp.Mapping[str, object],
    plane_url: str | None,
    plane_build: tp.Mapping[str, object] | None,
    plane_build_detail: str,
    started_at: str,
    completed_at: str,
    command: tp.Sequence[str],
    evidence: tp.Mapping[str, object] | None = None,
    git: tp.Mapping[str, object] | None = None,
    fleet_builds: tp.Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble one receipt, digest included.

    The digest covers every other field, so a receipt whose verdict was edited
    afterwards no longer verifies.
    """

    resolved_git = dict(git) if git is not None else git_identity()
    document: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "proof": proof,
        "verdict": verdict,
        "exit_status": exit_status,
        "checks": [dict(check) for check in checks],
        "criterion": dict(criterion),
        "candidate": {"build": build_identity().as_dict(), "git": resolved_git},
        "plane": {
            "url": plane_url,
            "build": dict(plane_build) if plane_build is not None else None,
            "build_detail": plane_build_detail,
        },
        "fleet": dict(fleet_builds) if fleet_builds is not None else None,
        "run": {
            "started_at": started_at,
            "completed_at": completed_at,
            "command": list(redact_command(command)),
            "host": socket.gethostname(),
        },
        "evidence": dict(evidence) if evidence is not None else None,
        "unverified": unverified_dimensions(
            plane_build=plane_build,
            git=resolved_git,
            evidence=evidence,
            fleet_builds=fleet_builds,
        ),
        "signature": None,
    }
    document["receipt_digest"] = canonical_digest(document)
    return document


def receipt_digest_matches(document: tp.Mapping[str, object]) -> bool | None:
    """Return whether a receipt still digests to what it records.

    ``None`` when it records no digest at all: an unsigned receipt from a writer
    that predates this field is not evidence of tampering, and calling it one
    would make every old receipt look forged.
    """

    recorded = document.get("receipt_digest")
    if not isinstance(recorded, str) or not recorded:
        return None
    without = {key: value for key, value in document.items() if key != "receipt_digest"}
    return canonical_digest(without) == recorded


def read_receipt(path: Path | str) -> dict[str, object]:
    """Load a receipt, refusing anything that is not one.

    Raises:
        ValueError: If the file is not a JSON object carrying a known schema.
    """

    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("a receipt must be a JSON object")
    schema = document.get("schema")
    if schema != RECEIPT_SCHEMA:
        raise ValueError(f"unknown receipt schema {schema!r}; this build reads {RECEIPT_SCHEMA}")
    return document


__all__ = [
    "RECEIPT_SCHEMA",
    "build_proof_receipt",
    "canonical_digest",
    "criterion_identity",
    "file_evidence",
    "git_identity",
    "read_receipt",
    "receipt_digest_matches",
    "redact_command",
    "unverified_dimensions",
    "write_receipt_atomic",
]
