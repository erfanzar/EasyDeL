# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan research``: propose, approve, watch and stop budgeted research plans.

A plan is a set of jobs plus a budget (jobs, chip-hours, wall-clock time). You
propose it from a YAML or JSON file, approve it once, and the plane submits,
monitors and -- when ``iterate`` is set and a provider is configured -- plans
further rounds inside that budget, always as you. ``stop`` ends it at any time.
"""

from __future__ import annotations

import json
import time
import typing as tp
from pathlib import Path

import click
import yaml

from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession

#: Plan states the plane never leaves.
TERMINAL_STATES = frozenset({"finished", "stopped", "failed", "over_budget"})


def _object(payload: object) -> dict[str, tp.Any]:
    """Narrow one decoded API answer to a JSON object, or fail with a readable error."""

    if not isinstance(payload, dict):
        raise click.ClickException("the plane returned an unexpected answer")
    return payload


def _objects(value: object) -> list[dict[str, tp.Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _load_plan(path: Path) -> dict[str, tp.Any]:
    """Read a plan document (YAML or JSON); JSON is valid YAML, so one parser reads both."""

    try:
        text = path.read_text(encoding="utf-8")
        document = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(f"could not read research plan {path}: {exc}") from None
    if not isinstance(document, dict):
        raise click.ClickException(f"research plan {path} must be a mapping with title, jobs and budget")
    return document


def _spent_line(plan: dict[str, tp.Any]) -> str:
    raw_budget, raw_spent = plan.get("budget"), plan.get("spent")
    budget: dict[str, tp.Any] = raw_budget if isinstance(raw_budget, dict) else {}
    spent: dict[str, tp.Any] = raw_spent if isinstance(raw_spent, dict) else {}
    return (
        f"jobs {spent.get('jobs_submitted', 0)}/{budget.get('max_jobs', '?')}, "
        f"chip-hours {float(spent.get('chip_hours', 0) or 0):.2f}/{budget.get('max_chip_hours', '?')}, "
        f"round {int(plan.get('current_round', 0) or 0) + 1}"
    )


def _show(plan: dict[str, tp.Any]) -> None:
    click.echo(f"{plan.get('id')}  {plan.get('state')}  {plan.get('title')}")
    click.echo(f"  owner {plan.get('subject')} in {plan.get('org_id')}/{plan.get('project_id')}")
    click.echo(f"  {_spent_line(plan)}")
    if plan.get("stop_reason"):
        click.echo(f"  stop reason: {plan['stop_reason']}")
    jobs = _objects(plan.get("jobs"))
    if jobs:
        click.echo("  jobs:")
    for job in jobs:
        detail = job.get("workload_id") or job.get("error") or ""
        reason = f" ({job['failure_reason']})" if job.get("failure_reason") else ""
        click.echo(
            f"    [{job.get('index')}] r{job.get('round')} {job.get('state'):<13} {job.get('name')} {detail}{reason}"
        )
    notes = _objects(plan.get("notes"))
    if notes:
        click.echo("  notes:")
    for note in notes[-10:]:
        click.echo(f"    {note.get('t')} {note.get('kind')}: {note.get('text')}")


def _get(session: CliSession, plan_id: str) -> dict[str, tp.Any]:
    return _object(session.client.request("GET", f"/api/v1/research/plans/{plan_id}"))


@click.group("research")
def research() -> None:
    """Budgeted research plans: approve jobs and a budget once; the plane runs and iterates them as you."""


@research.command("list")
@click.option("--state", help="Only plans in this state (proposed, running, finished, ...).")
@click.option("--json", "as_json", is_flag=True, help="Emit the result as JSON.")
@click.pass_obj
def research_list(session: CliSession, state: str | None, as_json: bool) -> None:
    """Your research plans (fleet admins see everyone's)."""

    payload = _object(
        session.client.request("GET", "/api/v1/research/plans", params={"state": state} if state else None)
    )
    if as_json:
        JsonRenderer().emit(payload)
        return
    rows = _objects(payload.get("plans"))
    lines = [
        f"{row['id']}  {row['state']:<11} {row['subject']:<14} {row['title']}  ({_spent_line(row)})" for row in rows
    ]
    click.echo("\n".join(lines) or "no research plans")


@research.command("show")
@click.argument("plan_id")
@click.option("--json", "as_json", is_flag=True, help="Emit the plan as JSON.")
@click.pass_obj
def research_show(session: CliSession, plan_id: str, as_json: bool) -> None:
    """One plan: jobs and their workloads' states, spend against budget, notes."""

    plan = _get(session, plan_id)
    if as_json:
        JsonRenderer().emit(plan)
        return
    _show(plan)


@research.command("propose")
@click.option(
    "--file",
    "plan_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Plan document (YAML or JSON): title, goal, jobs [{name, spec}], budget, iterate.",
)
@click.option("--org", "org_id", help="Organization (default: the plan's, else the active scope).")
@click.option("--project", "project_id", help="Project (default: the plan's, else the active scope).")
@click.option("--json", "as_json", is_flag=True, help="Emit the proposed plan as JSON.")
@click.pass_obj
def research_propose(
    session: CliSession, plan_file: Path, org_id: str | None, project_id: str | None, as_json: bool
) -> None:
    """Propose a plan. Nothing runs until you approve it with ``eyvan research approve``."""

    body = _load_plan(plan_file)
    scope = session.workload_scope(
        org_id=org_id or tp.cast(str | None, body.get("org_id")),
        project_id=project_id or tp.cast(str | None, body.get("project_id")),
    )
    body.update(org_id=scope.org_id, project_id=scope.project_id)
    plan = _object(session.client.request("POST", "/api/v1/research/plans", json=body))
    if as_json:
        JsonRenderer().emit(plan)
        return
    _show(plan)
    click.echo(f"\nproposed; approve with: eyvan research approve {plan['id']}")


@research.command("approve")
@click.argument("plan_id")
@click.option("--json", "as_json", is_flag=True, help="Emit the approved plan as JSON.")
@click.pass_obj
def research_approve(session: CliSession, plan_id: str, as_json: bool) -> None:
    """Approve your proposed plan; the plane then submits and iterates it as you, within its budget."""

    plan = _object(session.client.request("POST", f"/api/v1/research/plans/{plan_id}/approve"))
    if as_json:
        JsonRenderer().emit(plan)
        return
    click.echo(f"approved {plan['id']} ({_spent_line(plan)}); follow it with: eyvan research watch {plan['id']}")


@research.command("stop")
@click.argument("plan_id")
@click.option("--reason", default="", help="Why you are stopping it (kept in the plan's notes).")
@click.option("--json", "as_json", is_flag=True, help="Emit the stopped plan as JSON.")
@click.pass_obj
def research_stop(session: CliSession, plan_id: str, reason: str, as_json: bool) -> None:
    """Stop a plan now; its running workloads are cancelled."""

    plan = _object(session.client.request("POST", f"/api/v1/research/plans/{plan_id}/stop", json={"reason": reason}))
    if as_json:
        JsonRenderer().emit(plan)
        return
    click.echo(f"{plan['id']} is {plan['state']}")


@research.command("watch")
@click.argument("plan_id")
@click.option("--interval", type=click.FloatRange(min=0.1, max=3600), default=10.0, show_default=True)
@click.option(
    "--timeout",
    type=click.FloatRange(min=0, max=30 * 86_400),
    default=24 * 3600.0,
    show_default=True,
    help="Stop watching (the plan keeps running) after this many seconds.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the final plan as JSON.")
@click.pass_obj
def research_watch(session: CliSession, plan_id: str, interval: float, timeout: float, as_json: bool) -> None:
    """Poll a plan until it ends (bounded by --timeout); prints each state change and new note."""

    deadline = time.monotonic() + timeout
    seen_notes = 0
    last_line = ""
    while True:
        plan = _get(session, plan_id)
        if not as_json:
            notes = _objects(plan.get("notes"))
            for note in notes[seen_notes:] if len(notes) >= seen_notes else notes:
                click.echo(f"{note.get('t')} {note.get('kind')}: {note.get('text')}")
            seen_notes = len(notes)
            line = f"[{plan.get('state')}] {_spent_line(plan)}"
            if line != last_line:
                click.echo(line)
                last_line = line
        if plan.get("state") in TERMINAL_STATES:
            if as_json:
                JsonRenderer().emit(plan)
            return
        if time.monotonic() + interval > deadline:
            raise click.ClickException(
                f"stopped watching after {timeout:g}s; {plan_id} is still {plan.get('state')} on the plane"
            )
        time.sleep(interval)


def register_research_commands(group: click.Group) -> None:
    """Attach ``eyvan research`` to the CLI root."""

    group.add_command(research)


__all__ = ("register_research_commands",)
