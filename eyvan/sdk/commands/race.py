# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan race``: find which zones will actually grant a TPU shape, by asking all of them at once."""

from __future__ import annotations

import json
import sys
import threading
import time
import typing as tp
from pathlib import Path

import click

from eyvan.capacity.providers.gcp.tpu_nodes import gcloud_active_account
from eyvan.capacity.providers.gcp.tpu_race import (
    RaceOutcome,
    RaceReport,
    RaceTiming,
    TpuQueueApi,
    TpuQueueRest,
    TpuRace,
    default_receipt,
    gcloud_token,
    sweep_race,
)

#: Built by the command; replaced by tests with an in-memory fake.
ApiFactory = tp.Callable[[str | None], TpuQueueApi]


def _rest_api(account: str | None) -> TpuQueueApi:
    return TpuQueueRest(token_source=lambda: gcloud_token(account))


_api_factory: ApiFactory = _rest_api
_echo_lock = threading.Lock()


def _event(zone: str, message: str) -> None:
    with _echo_lock:
        click.echo(f"{time.strftime('%H:%M:%S')}  {zone:26} {message}", err=True)


def _summary(report: RaceReport) -> None:
    order = {outcome: index for index, outcome in enumerate(RaceOutcome)}
    click.echo(
        f"\nrace {report.run_id}: {report.accelerator_type} ({'spot' if report.spot else 'on-demand'}) "
        f"in {report.project}, {len(report.zones)} zone(s)"
    )
    for zone in sorted(report.zones, key=lambda z: (order[z.outcome], z.granted_after_s or 0.0, z.zone)):
        detail = ""
        if zone.outcome is RaceOutcome.GRANTED:
            detail = f"after {zone.granted_after_s}s"
        elif zone.error:
            detail = zone.error[:110]
        elif zone.states:
            detail = f"last state {zone.states[-1][1]}"
        cleanup = f"  [{zone.cleanup}]" if zone.cleanup else ""
        click.echo(f"  {zone.zone:26} {zone.outcome.value:10} {detail}{cleanup}")
    if report.not_offered:
        click.echo(f"  not offered in: {', '.join(report.not_offered)}")
    if report.unknown_offer:
        click.echo(f"  could not check (not raced): {', '.join(report.unknown_offer)}")
    sweep = report.sweep
    if sweep is None:
        click.echo(f"cleanup: NOT confirmed; run `eyvan race sweep {report.run_id} --project {report.project}`")
    elif sweep.clean:
        click.echo("cleanup: final sweep found nothing left" + (" (except the kept grant)" if report.kept else ""))
    else:
        for zone, names in sweep.leftovers.items():
            if names:
                click.echo(f"cleanup: LEFT in {zone}: {', '.join(names)}")
        for zone, why in sweep.unreadable.items():
            click.echo(f"cleanup: could not list {zone} ({why}); state there is unknown")
        click.echo(f"retry with `eyvan race sweep {report.run_id} --project {report.project}`")
    kept = report.kept
    if kept is not None:
        click.echo(
            f"\nkept {kept.name} in {kept.zone}. It bills until deleted. Attach it with:\n"
            f"  eyvan gcp connect --tpu {kept.name} --project {report.project} --zone {kept.zone}\n"
            f"delete it with:\n"
            f"  gcloud compute tpus queued-resources delete {kept.name} --zone {kept.zone} "
            f"--project {report.project} --force"
        )


@click.group("race")
def race() -> None:
    """Request capacity in many zones at once to learn (or grab) what a project can really get."""


@race.command("tpu")
@click.argument("accelerator_type", metavar="TYPE")
@click.option("--project", required=True, metavar="P", help="GCP project to request in.")
@click.option("--zone", "zones", multiple=True, metavar="Z", help="Race only these zones (repeatable).")
@click.option("--on-demand", is_flag=True, help="Request on-demand instead of Spot capacity.")
@click.option("--keep-first", is_flag=True, help="Keep the first grant (it bills) and delete everything else.")
@click.option("--account", metavar="EMAIL", help="gcloud account to request as; gcloud config is not changed.")
@click.option(
    "--timeout",
    "timeout_min",
    type=click.FloatRange(min=1.0),
    default=30.0,
    show_default=True,
    help="Minutes to wait for a grant before giving up on a zone.",
)
@click.option("--runtime-version", metavar="V", help="TPU software version; defaults by generation.")
@click.option("--receipt", type=click.Path(dir_okay=False, path_type=Path), help="Where to write the JSON receipt.")
@click.option("--json", "as_json", is_flag=True, help="Print the receipt as JSON instead of a summary.")
@click.option("--yes", is_flag=True, help="Do not ask before creating billable requests.")
def race_tpu(
    accelerator_type: str,
    project: str,
    zones: tuple[str, ...],
    on_demand: bool,
    keep_first: bool,
    account: str | None,
    timeout_min: float,
    runtime_version: str | None,
    receipt: Path | None,
    as_json: bool,
    yes: bool,
) -> None:
    """Request TYPE in every zone of P that offers it, report who grants, then delete everything.

    Measure-only by default: a grant is deleted as soon as it is seen. With
    --keep-first the first grant is kept and every other request is deleted.
    """

    api = _api_factory(account)
    who = account or gcloud_active_account() or "the active gcloud account"
    try:
        tpu_race = TpuRace(
            api,
            project=project,
            accelerator_type=accelerator_type,
            zones=zones,
            spot=not on_demand,
            keep_first=keep_first,
            runtime_version=runtime_version,
            account=account,
            timing=RaceTiming(timeout_s=timeout_min * 60.0),
            on_event=_event,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    tpu_race.receipt = receipt or default_receipt(tpu_race.run_id)
    try:
        candidates = tpu_race.candidate_zones()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    if not candidates:
        raise click.ClickException(f"no zone of {project} offers {accelerator_type} (or none could be checked)")
    tpu_race.requested_zones = tuple(candidates)
    capacity = "on-demand" if on_demand else "Spot"
    plan = (
        f"race {tpu_race.run_id}: request {capacity} {accelerator_type} in {len(candidates)} zone(s) of {project} "
        f"as {who}: {', '.join(candidates)}\n"
        + ("the first grant is KEPT and bills until you delete it" if keep_first else "every grant is deleted at once")
    )
    click.echo(plan, err=True)
    if not yes:
        if not sys.stdin.isatty():
            raise click.UsageError("this creates billable cloud requests; pass --yes to run without a prompt")
        click.confirm("Go?", abort=True, err=True)
    click.echo(f"receipt: {tpu_race.receipt}", err=True)
    report = tpu_race.run()
    if as_json:
        click.echo(json.dumps(report.document(), indent=2))
    else:
        _summary(report)
    if not report.clean:
        raise click.ClickException(f"race {report.run_id} did not confirm its cleanup; see the summary above")
    if keep_first and report.kept is None:
        raise click.ClickException("no zone granted the request; nothing was kept")


@race.command("sweep")
@click.argument("run_id")
@click.option("--project", metavar="P", help="Project; defaults to the one in the race's receipt.")
@click.option("--zone", "zones", multiple=True, metavar="Z", help="Zones to sweep; defaults to the receipt's.")
@click.option("--account", metavar="EMAIL", help="gcloud account to delete as; gcloud config is not changed.")
@click.option("--include-kept", is_flag=True, help="Also delete a grant the race kept.")
def race_sweep(
    run_id: str, project: str | None, zones: tuple[str, ...], account: str | None, include_kept: bool
) -> None:
    """Delete everything race RUN_ID left behind (except a kept grant) and confirm the zones are clean."""

    receipt: dict[str, tp.Any] = {}
    path = default_receipt(run_id)
    if path.is_file():
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            receipt = {}
    project = project or receipt.get("project")
    zone_list = list(zones) or list(receipt.get("raced_zones") or [])
    if not project or not zone_list:
        raise click.UsageError(f"no receipt at {path}; pass --project and --zone")
    kept = receipt.get("kept") if isinstance(receipt.get("kept"), dict) else None
    keep = () if include_kept or kept is None else (str(kept.get("name")),)
    try:
        result = sweep_race(_api_factory(account), project=project, zones=zone_list, run_id=run_id, keep=keep)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(json.dumps(result.document(), indent=2))
    if not result.clean:
        raise click.ClickException(f"race {run_id} is not confirmed clean")


def register_race_commands(main: click.Group) -> None:
    """Register ``eyvan race`` at the composition root."""

    main.add_command(race)


__all__ = ("race", "register_race_commands")
