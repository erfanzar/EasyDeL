# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What one workload is, rendered for a person.

``eyvan workload status`` printed the durable record: sixty-odd keys, most of
them null, with the three facts somebody actually wants -- what state it is in,
what it runs, and why it is not running -- scattered between
``autoscale_block_reason`` and ``worker_resource_scope``. The record is the
right thing to *store* and the wrong thing to *show*, and ``--json`` still
returns it verbatim for anything parsing.

Every line here answers a question somebody asked out loud:

* **state** and, when it is not running, why not
* **what it runs** -- the command, or a callable's content hash, which is the
  only durable identity the code has once it is pickled
* **what it asked for** -- the shape, and the per-replica resources that shape
  resolved to, since a pool's resource vector is what admission actually checks
* **where** it is allowed to run, and which pool it was pinned to
* **how many attempts** it has cost, separated into failures and preemptions
  because on spot those are different stories

Nothing is invented. A field the record leaves null is either omitted or said to
be unknown; this renders what is there rather than filling gaps.
"""

from __future__ import annotations

import typing as tp

from eyvan._internal.json_types import JsonObject, is_json_object

#: How much of a content hash is enough to recognise a build by. Long enough to
#: be unambiguous in a project's history, short enough to read aloud.
_HASH_PREFIX = 12


def _text(value: object) -> str | None:
    """Return a non-empty trimmed string, or None for anything else."""

    return value.strip() if isinstance(value, str) and value.strip() else None


def _resources(gang: object) -> str | None:
    """Describe what one replica asked for, as admission sees it."""

    if not is_json_object(gang):
        return None
    per_replica = gang.get("per_replica")
    if not is_json_object(per_replica):
        return None
    parts: list[str] = []
    for key, label in (("tpu", "TPU"), ("gpu", "GPU")):
        count = per_replica.get(key)
        if isinstance(count, (int, float)) and count > 0:
            parts.append(f"{int(count)} {label}")
    cpu = per_replica.get("cpu")
    if isinstance(cpu, (int, float)) and cpu > 0:
        parts.append(f"{cpu:g} CPU")
    memory = per_replica.get("memory_gib")
    if isinstance(memory, (int, float)) and memory > 0:
        parts.append(f"{memory:g} GiB")
    replicas = gang.get("replicas")
    shape = ", ".join(parts) or None
    if shape and isinstance(replicas, int) and replicas > 1:
        return f"{replicas} x ({shape})"
    return shape


def _runs(execution: object) -> tuple[str, str] | None:
    """Return (label, value) naming the code this workload runs.

    A command is its own identity. A callable has none once it is pickled except
    its content hash, which is exactly what makes a run reproducible -- so that
    is what is shown rather than the bucket path it happens to sit at today.
    """

    if not is_json_object(execution):
        return None
    kind = execution.get("kind")
    if kind == "command":
        command = _text(execution.get("command"))
        return ("command", command) if command else None
    if kind == "callable":
        digest = _text(execution.get("sha256"))
        python = _text(execution.get("python"))
        if digest is None:
            return ("callable", "python " + python) if python else None
        suffix = f" · python {python}" if python else ""
        return ("callable", f"sha256:{digest[:_HASH_PREFIX]}{suffix}")
    return None


def _blocked(record: JsonObject) -> str | None:
    """Say why this is not progressing, in the most specific words available.

    Four fields answer it depending on where the workload stalled, and a person
    reading a status does not know which to look at. ``failure_reason`` is
    consulted first because it carries the *provider's* own sentence -- "User
    does not have permission to submit requests into this queue for accelerator
    type v6e-16 in location asia-east1-c" -- while the others carry Eyvan's
    category for it. Leading with the category buried the only line that says
    what to go and fix.
    """

    for field in ("failure_reason", "autoscale_block_reason", "scheduling_reason"):
        reason = _text(record.get(field))
        if reason:
            return reason
    denial = record.get("queue_admission_denial")
    if is_json_object(denial):
        shape = _text(denial.get("accelerator_type"))
        zone = _text(denial.get("zone"))
        located = " ".join(part for part in (shape, f"in {zone}" if zone else "") if part)
        return f"the provider refused a queued request for {located}" if located else "the provider refused the request"
    if record.get("schedulable") is False:
        return "not schedulable; no enabled pool matches this request"
    return None


def format_user_progress(reports: object) -> tuple[str, ...]:
    """Render self-reported ``UserProgressReport`` entries as compact human lines."""

    if not isinstance(reports, list):
        return ()
    valid = [entry for entry in reports if is_json_object(entry)]
    if not valid:
        return ()
    multi_rank = len(valid) > 1
    lines: list[str] = []
    for entry in valid:
        parts: list[str] = []
        step = entry.get("step")
        total = entry.get("total")
        if isinstance(step, int) and not isinstance(step, bool):
            if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                pct = min(100, round(step / total * 100))
                parts.append(f"step {step}/{total} ({pct}%)")
            else:
                parts.append(f"step {step}")
        metrics = entry.get("metrics")
        if is_json_object(metrics) and metrics:
            formatted = ", ".join(
                f"{key}={val:g}" if isinstance(val, (int, float)) and not isinstance(val, bool) else f"{key}={val}"
                for key, val in sorted(metrics.items())
            )
            parts.append(formatted)
        message = _text(entry.get("message"))
        if message:
            parts.append(message)
        if not parts:
            continue
        rank = entry.get("rank")
        prefix = f"rank {rank}: " if multi_rank and isinstance(rank, int) and not isinstance(rank, bool) else ""
        lines.append(f"{prefix}{' · '.join(parts)}")
    return tuple(lines)


def summarize(record: JsonObject) -> tuple[str, ...]:
    """Render one workload record as aligned lines, or nothing if unreadable."""

    spec = record.get("spec")
    spec = spec if is_json_object(spec) else {}
    lines: list[str] = []

    def row(label: str, value: str | None) -> None:
        if value:
            lines.append(f"  {label:<11} {value}")

    name = _text(spec.get("name"))
    identifier = _text(record.get("id")) or "unknown"
    row("workload", f"{identifier}" + (f" ({name})" if name else ""))
    state = _text(record.get("state")) or "unknown"
    attempt = record.get("attempt")
    attempt_note = f" · attempt {attempt}" if isinstance(attempt, int) and attempt > 1 else ""
    row("state", f"{state}{attempt_note}")
    row("blocked by", _blocked(record))

    runs = _runs(spec.get("execution"))
    if runs is not None:
        row(*runs)
    workspace = _text(spec.get("workspace"))
    execution = spec.get("execution")
    if workspace is None and is_json_object(execution):
        workspace = _text(execution.get("workspace_uri"))
    row("workspace", workspace)

    shape = _text(spec.get("accelerator_type"))
    capacity_class = _text(spec.get("capacity_class"))
    if shape:
        row("shape", shape + (f" · {capacity_class}" if capacity_class else ""))
    row("resources", _resources(spec.get("gang")))
    pods = spec.get("cell_group_count")
    if isinstance(pods, int) and pods > 1:
        row("slices", f"{pods}, allocated together")

    metadata = spec.get("metadata")
    pool = _text(metadata.get("eyvan_pool")) if is_json_object(metadata) else None
    raw_alternates = _text(metadata.get("eyvan_pool_alternates")) if is_json_object(metadata) else None
    alternates = [name for name in (raw_alternates or "").split(",") if name.strip()]
    region = _text(spec.get("region"))
    zones = spec.get("allowed_zones")
    where = ", ".join(value for value in zones if isinstance(value, str)) if isinstance(zones, list) else ""
    if alternates:
        # A cross-zone job may run in any of its candidates, so the first choice
        # is a request, not a placement, and its region binds nothing. Do not
        # present the requested pool as the workload's actual location.
        row("requested", f"{pool or 'unknown'} or {len(alternates)} alternate pool(s)")
        row("region", where or None)
    else:
        row("placed on", pool)
        row("region", where or region)

    lease = _text(record.get("lease_id"))
    row("lease", lease)
    failures = record.get("failure_count")
    preemptions = record.get("preemption_count")
    counts = [
        f"{failures} failure(s)" if isinstance(failures, int) and failures else "",
        f"{preemptions} preemption(s)" if isinstance(preemptions, int) and preemptions else "",
    ]
    row("retries", ", ".join(part for part in counts if part))

    for line in format_user_progress(record.get("user_progress")):
        row("progress", line)
    row("submitted", _text(record.get("created_by")))
    row("at", _text(record.get("created_at")))
    return tuple(lines)


def render(record: object) -> tuple[str, ...]:
    """Render a workload record, falling back to nothing when it is not one."""

    return summarize(record) if is_json_object(record) else ()


__all__ = ("format_user_progress", "render", "summarize")


if tp.TYPE_CHECKING:  # pragma: no cover - import-time typing only
    _: tp.Callable[[JsonObject], tuple[str, ...]] = summarize
