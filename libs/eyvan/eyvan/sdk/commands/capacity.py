# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read-only capacity inspection for the Eyvan CLI.

Every route these commands call already existed, along with its service method and
its client method -- what was missing was any way to reach them without writing
the HTTP call by hand. ``eyvan capacity`` in particular was reachable only as a
side effect of ``eyvan up --region``, which meant "what can I run on?" required
provisioning something first.

Deliberately no ``pool create``. Pools are created declaratively by ``eyvan up``
from a fleet config, so an imperative creator would be a second source of truth
for the same objects and the two would drift the first time someone used both.
The help text says so, which turns an inconsistency into a stated policy.
"""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import click

from eyvan._internal.json_types import JsonValue, is_json_object
from eyvan.capacity.autoscale.cache import AutoscaleConfigCache
from eyvan.domain.accelerators import sort_accelerator_types
from eyvan.records import CapacityClass
from eyvan.sdk.commands.common import JsonRenderer
from eyvan.sdk.commands.state import CliSession
from eyvan.sdk.contexts import DEFAULT_ORG_ID, DEFAULT_PROJECT_ID
from eyvan.sdk.shapes import summarize_shapes


def _emit(payload: object, *, as_json: bool, renderer: JsonRenderer) -> None:
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2, default=str))
        return
    renderer.emit(payload)


@click.group("capacity")
def capacity() -> None:
    """Inspect available capacity, pools, cells, and leases."""


@capacity.command("builds")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def capacity_builds(session: CliSession, as_json: bool) -> None:
    """Show which build each cell agent runs, beside this control plane's.

    Every component of Eyvan reports the same version string, so it cannot
    answer the question an upgrade raises: is the agent holding my capacity the
    build I deployed? The digest of the installed package can, and this puts the
    plane's beside every agent's.

    A hardware proof has already been run against a stale agent and read as
    evidence. This is the check that would have caught it.

    Exits non-zero on proven skew, and also when some agent could not say --
    "nothing disagreed" is not agreement, and an upgrade is exactly when that
    distinction matters.
    """

    payload = session.client.request("GET", "/api/v1/capacity/builds")
    document = payload if isinstance(payload, dict) else {}
    agreement = document.get("in_agreement")
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2, default=str))
        # Formatting must not turn skew or an unknown verdict into a passing gate.
        if agreement is False or agreement is None:
            raise click.exceptions.Exit(1)
        return
    plane = document.get("plane") or {}
    agents = document.get("agents") or []
    unreachable = document.get("unreachable") or []
    unscanned = document.get("unscanned_pools") or []

    plane_digest = plane.get("source_digest")
    click.echo(
        f"control plane  {plane.get('version')}  {plane_digest or 'no digest: ' + str(plane.get('digest_detail'))}"
    )

    # This client's own build, beside the plane's. The comparison used to start
    # at the control plane, so the one skew an operator meets first -- a CLI
    # newer than the head it is talking to -- was the one this command could not
    # show. That skew surfaces instead as a raw pydantic 422 from `eyvan up`.
    from eyvan._internal.build_identity import build_identity

    client = build_identity()
    client_digest = client.source_digest
    click.echo(f"this client   {client.version}  {client_digest or 'no digest: ' + str(client.digest_detail)}")
    if client_digest and plane_digest and client_digest != plane_digest:
        click.echo(
            "  this client and the control plane are different builds; "
            "run 'eyvan runtime deploy' to bring the plane up to this source"
        )
    silent = 0
    for entry in agents if isinstance(agents, list) else []:
        if not isinstance(entry, dict):
            continue
        build = entry.get("build") or {}
        digest = build.get("source_digest") if isinstance(build, dict) else None
        if not digest:
            silent += 1
            verdict = "does not report a build"
        elif digest == plane_digest:
            verdict = "same build"
        else:
            verdict = f"DIFFERENT BUILD  {digest}"
        click.echo(f"agent {entry.get('endpoint')}  pool={entry.get('pool_id')}  {verdict}")
        workers = entry.get("workers")
        if workers is None:
            silent += 1
            click.echo(f"  cell {entry.get('cell_id')}: its workers could not be asked")
            continue
        for worker in workers if isinstance(workers, list) else []:
            if not isinstance(worker, dict):
                continue
            worker_build = worker.get("build") or {}
            worker_digest = worker_build.get("source_digest") if isinstance(worker_build, dict) else None
            if not worker_digest:
                silent += 1
                click.echo(f"  rank {worker.get('rank')} {worker.get('endpoint')}  does not report a build")
            elif worker_digest == plane_digest:
                click.echo(f"  rank {worker.get('rank')} {worker.get('endpoint')}  same build")
            else:
                click.echo(f"  rank {worker.get('rank')} {worker.get('endpoint')}  DIFFERENT BUILD  {worker_digest}")

    for entry in unreachable if isinstance(unreachable, list) else []:
        click.echo(f"could not reach cell agent {entry.get('endpoint')}: {entry.get('detail')}", err=True)
    for pool_id in unscanned if isinstance(unscanned, list) else []:
        click.echo(f"pool {pool_id} holds no cell, so its agent was not scanned", err=True)

    if agreement is False:
        raise click.ClickException(
            "at least one cell agent is running different code than this control plane; "
            "list cells with `eyvan cells`, recycle each stale one with `eyvan cell-drain <id>` then "
            "`eyvan cell-delete <id>`, or redeploy every component from one source with `eyvan runtime deploy`"
        )
    if agreement is None:
        raise click.ClickException(
            f"the fleet could not be shown to agree: {len(agents)} agent(s) answered, {silent} without a digest, "
            f"{len(unreachable)} unreachable"
        )
    click.echo(f"every reachable agent and worker of {len(agents)} cell(s) runs this control plane's build")


@capacity.command("orphans")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def capacity_orphans(session: CliSession, as_json: bool) -> None:
    """List cells an agent still holds that this control plane does not claim.

    A fleet-state document outlives the control plane that wrote it, so a
    replaced database leaves its cells behind in the agent -- and any left
    desired-up are re-ensured forever, provisioning capacity nobody asked for.
    Before this, the only way to see them was reading the agent's state document
    out of object storage by hand.

    Reporting only. One agent can legitimately serve more than one control
    plane, so an unrecognised record is not proven unowned; reclaiming it is an
    operator decision, and this is the information that decision needs.
    """

    payload = session.client.request("GET", "/api/v1/capacity/orphans")
    document = (
        payload if isinstance(payload, dict) else {"orphans": payload or [], "scanned_endpoints": [], "unreachable": []}
    )
    orphans = document.get("orphans") or []
    scanned = document.get("scanned_endpoints") or []
    unreachable = document.get("unreachable") or []
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2, default=str))
        # A partial or absent scan is unknown in either output format.
        if unreachable or not scanned:
            raise click.exceptions.Exit(1)
        return

    for entry in orphans if isinstance(orphans, list) else []:
        if not isinstance(entry, dict):
            continue
        click.echo(
            f"{entry.get('name')}  state={entry.get('state')}  desired={entry.get('desired_state')}  "
            f"qr={entry.get('qr_id')}  agent={entry.get('endpoint')}"
        )

    # "None found" and "nothing was looked at" are different answers, and this
    # command is most often run to confirm a teardown left nothing behind --
    # exactly when reporting the second as the first is worst.
    if unreachable:
        for entry in unreachable if isinstance(unreachable, list) else []:
            click.echo(f"could not reach cell agent {entry.get('endpoint')}: {entry.get('detail')}", err=True)
        raise click.ClickException(
            f"{len(unreachable)} of {len(scanned)} cell agent(s) could not be reached; "
            "this is not a report that nothing is orphaned"
        )
    if not scanned:
        raise click.ClickException(
            "this control plane knows no cell agents, so nothing was scanned. That is expected when its "
            "database has been replaced -- which is also when orphans are most likely -- so point a client at "
            "the agent directly rather than reading this as a clean result."
        )
    if not orphans:
        click.echo(f"no orphaned cell records ({len(scanned)} agent(s) scanned)")


@capacity.command("list")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--accelerator-type", help="Only pools offering this accelerator.")
@click.option("--capacity-class", help="Only pools of this capacity class.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def list_capacity(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    accelerator_type: str | None,
    capacity_class: str | None,
    as_json: bool,
) -> None:
    """Show the capacity pools this project may submit to."""

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    payload = session.client.request(
        "GET",
        "/api/v1/capacity",
        params={"org_id": scope.org_id, "project_id": scope.project_id},
        headers=session.workload_headers(scope),
    )
    pools = payload.get("pools") if isinstance(payload, dict) else None
    if not isinstance(pools, list):
        raise click.ClickException(
            f"control plane at {session.base_url} returned an unexpected capacity response; "
            "a version mismatch between this CLI and the plane is the usual cause -- align them "
            "(redeploy the plane, or upgrade this CLI) rather than retrying"
        )
    rows = [
        pool
        for pool in pools
        if isinstance(pool, dict)
        and (accelerator_type is None or pool.get("accelerator_type") == accelerator_type)
        and (capacity_class is None or pool.get("capacity_class") == capacity_class)
    ]
    if as_json:
        click.echo(json.dumps({"pools": rows}, sort_keys=True, indent=2, default=str))
        return
    if not rows:
        # Distinguish "your filters matched nothing" from "the fleet is empty":
        # the two have completely different fixes and the same empty table.
        if pools:
            click.echo("No capacity pools match those filters.")
        else:
            click.echo("No enabled capacity pools are available to this project.")
        return
    click.echo("Available capacity pools:")
    for pool in rows:
        click.echo(
            f"  {pool.get('fleet', '-'):30} {pool.get('accelerator_type', '-') or '-':12} "
            f"{pool.get('capacity_class', '-'):10} {pool.get('region', '-'):12} "
            f"{pool.get('zone') or '-':14} max={pool.get('max_cells', '?')}"
        )


#: Rendered wherever a quota row has no usable usage observation. Never "0":
#: a missing Cloud Monitoring series says nothing about how full a zone is, and
#: a zero there is the one wrong answer that costs money to act on.
UNKNOWN_QUOTA = "unknown"


def _quota_number(value: object) -> str:
    """Render a quota figure, or the explicit unknown marker for a missing one."""

    return str(value) if isinstance(value, int) and not isinstance(value, bool) else UNKNOWN_QUOTA


def _quota_age(seconds: object) -> str:
    """Render an observation age in whole units without pretending precision."""

    if not isinstance(seconds, int | float) or isinstance(seconds, bool) or seconds < 0:
        return UNKNOWN_QUOTA
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    return f"{int(seconds // 3600)}h ago"


def _quota_fits(row: dict[str, object], accelerator_type: str | None, count: int) -> bool | None:
    """Return whether one row fits the request, or ``None`` when unknowable.

    ``None`` is the answer for a row with no measured balance, and it is
    deliberately not ``False``: "nobody could measure this zone" is a different
    answer from "this zone does not fit", and only one of them should stop an
    operator trying it.
    """

    if accelerator_type is not None and row.get("accelerator_type") != accelerator_type:
        return False
    slices = row.get("free_slices")
    if not isinstance(slices, int) or isinstance(slices, bool):
        return None
    return slices >= count


def _refused_spot_queues(cache_dir: Path | None, project: object) -> frozenset[tuple[str, str]]:
    """Return ``(zone, accelerator_type)`` Spot queues GCP refused for ``project``.

    Quota is not access: a zone can have Spot quota and still refuse permission
    to submit requests into its queue. Provisioning records those
    refusals in the local autoscale cache, which already keeps them out of
    automatic fleet selection; this is the same record read for the quota view.
    An unreadable cache yields no refusals rather than breaking the view.
    """

    if not isinstance(project, str) or not project:
        return frozenset()
    try:
        denials = AutoscaleConfigCache(cache_dir).list_queue_admission_denials()
    except (OSError, ValueError):
        return frozenset()
    return frozenset(
        (denial.key.zone, denial.key.accelerator_type)
        for denial in denials
        if denial.key.project == project and denial.key.capacity_class is CapacityClass.SPOT
    )


def _parse_fits(value: str) -> tuple[str, int]:
    """Parse ``TYPE`` or ``TYPE:COUNT`` into an explicit shape and multiplicity."""

    accelerator_type, separator, raw = value.strip().partition(":")
    if not accelerator_type:
        raise click.ClickException("--fits needs an accelerator type, for example v6e-8 or v6e-8:2")
    if not separator:
        return accelerator_type, 1
    try:
        count = int(raw)
    except ValueError as exc:
        raise click.ClickException("--fits count must be a positive integer, for example v6e-8:2") from exc
    if count < 1:
        raise click.ClickException("--fits count must be a positive integer, for example v6e-8:2")
    return accelerator_type, count


@capacity.command("quota")
@click.option("--project", help="GCP project to read. Defaults to the one this fleet's pools declare.")
@click.option("--zone", "zones", multiple=True, help="Restrict the provider read to these zones.")
@click.option("--generation", "generations", multiple=True, help="Only rows for these TPU generations.")
@click.option(
    "--capacity-class",
    "capacity_classes",
    multiple=True,
    help="Only spot or on_demand rows. Both are read; this filters what is printed.",
)
@click.option("--fits", help="Only zones that fit TYPE or TYPE:COUNT, for example v6e-8:2.")
@click.option("--refresh", is_flag=True, help="Re-read the provider instead of reusing a cached reading.")
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="Autoscale cache holding recorded Spot queue refusals. Defaults to the active fleet's cache.",
)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def capacity_quota(
    session: CliSession,
    project: str | None,
    zones: tuple[str, ...],
    generations: tuple[str, ...],
    capacity_classes: tuple[str, ...],
    fits: str | None,
    refresh: bool,
    cache_dir: Path | None,
    as_json: bool,
) -> None:
    """Show provider TPU quota, live usage, and free capacity per zone and type.

    The numbers ``eyvan autoscale generate`` already reads, answered for the
    person choosing where to run. Spot and on-demand are separate provider quota
    metrics and are reported separately.

    A zone whose usage cannot be read prints ``unknown`` rather than a free
    balance, and ``--fits`` keeps those rows under a separate heading instead of
    dropping them: "we could not tell" is a different answer from "it does not
    fit", and only one of them should stop you trying.
    """

    params: dict[str, str | list[str]] = {"refresh": "true" if refresh else "false"}
    if project:
        params["project"] = project
    if zones:
        params["zone"] = list(zones)
    payload = session.client.request("GET", "/api/v1/capacity/quota", params=params)
    if not isinstance(payload, dict):
        raise click.ClickException(
            f"control plane at {session.base_url} returned an unexpected quota response; "
            "a version mismatch between this CLI and the plane is the usual cause -- align them "
            "(redeploy the plane, or upgrade this CLI) rather than retrying"
        )
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2, default=str))
        return

    status = payload.get("status")
    if status == "unconfigured":
        click.echo("No GCP capacity pool declares a provider project, and --project was not given.")
        return
    wanted_generations = {value.strip().lower() for value in generations if value.strip()}
    wanted_classes = {value.strip().lower().replace("-", "_") for value in capacity_classes if value.strip()}
    # A zero floor with no type keeps every *measured* row, including one whose
    # balance is a measured zero -- a full zone is a real answer, unlike an
    # unmeasured one.
    fits_type, fits_count = _parse_fits(fits) if fits else (None, 0)
    raw_rows = payload.get("entries")
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    rows = [
        row
        for row in rows
        if (not wanted_generations or str(row.get("generation", "")).lower() in wanted_generations)
        and (not wanted_classes or str(row.get("capacity_class", "")).lower() in wanted_classes)
    ]
    project_name = payload.get("project")
    refused_queues = _refused_spot_queues(
        cache_dir if cache_dir is not None else session.autoscale_cache_dir, project_name
    )
    # A queue GCP refused is never capacity, whatever its quota says, so it is
    # listed on its own instead of among free or unmeasured zones.
    refused = [
        row
        for row in rows
        if str(row.get("capacity_class", "")).lower() == "spot"
        and (row.get("zone"), row.get("accelerator_type")) in refused_queues
        and (fits_type is None or row.get("accelerator_type") == fits_type)
    ]
    rows = [
        row
        for row in rows
        if not (
            str(row.get("capacity_class", "")).lower() == "spot"
            and (row.get("zone"), row.get("accelerator_type")) in refused_queues
        )
    ]
    fitting = [row for row in rows if _quota_fits(row, fits_type, fits_count) is True]
    # Unknown usage is reported separately rather than filtered away. A zone
    # nobody could measure is not a zone that does not fit.
    unknown = [row for row in rows if _quota_fits(row, fits_type, fits_count) is None]

    age = _quota_age(payload.get("age_s"))
    source = "cached" if payload.get("cached") else "read now"
    click.echo(f"TPU quota for {project_name or 'the configured project'} -- provider read {age} ({source})")
    for failure in payload.get("failures") or []:
        if isinstance(failure, dict):
            click.echo(f"  {failure.get('capacity_class', 'capacity')} quota could not be read: {failure.get('reason')}")
    if status == "unavailable":
        raise click.ClickException("no TPU quota could be read for this project")

    _echo_quota_rows("Free capacity" if not fits else f"Fits {fits_count} x {fits_type}", fitting)
    _echo_quota_rows("Usage unknown -- not proven empty and not proven full", unknown)
    _echo_quota_rows(
        "Spot queue refused this project -- quota is not access (recorded by an earlier request; expires after 24h)",
        refused,
    )
    if not fitting and not unknown and not refused:
        click.echo("  no quota rows match those filters")


def _echo_quota_rows(heading: str, rows: list[dict[str, object]]) -> None:
    """Print one quota section, leaving an unknown balance explicitly unknown."""

    if not rows:
        return
    click.echo(f"\n{heading}:")
    click.echo(f"  {'ZONE':18} {'TYPE':14} {'CLASS':10} {'LIMIT':>8} {'IN USE':>8} {'FREE':>8} {'SLICES':>7}  USAGE")
    for row in sorted(
        rows,
        key=lambda item: (
            -(item.get("free_chips") if isinstance(item.get("free_chips"), int) else -1),
            str(item.get("zone")),
            str(item.get("accelerator_type")),
        ),
    ):
        click.echo(
            f"  {row.get('zone', '-')!s:18} {row.get('accelerator_type', '-')!s:14} "
            f"{row.get('capacity_class', '-')!s:10} "
            f"{_quota_number(row.get('limit_chips')):>8} {_quota_number(row.get('in_use_chips')):>8} "
            f"{_quota_number(row.get('free_chips')):>8} {_quota_number(row.get('free_slices')):>7}  "
            f"{_quota_age(row.get('usage_age_s'))}"
        )


@capacity.command("warmth")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def warmth(session: CliSession, as_json: bool) -> None:
    """Report each pool's warm floor and whether it is currently held.

    Read alongside ``capacity billing``: warm capacity is deliberately idle, so
    its cost must be subtracted from the fleet's idle cost before that number
    means anything.
    """

    payload = session.client.request("GET", "/api/v1/capacity/warmth")
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@capacity.command("launch-latency")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def launch_latency(session: CliSession, as_json: bool) -> None:
    """Report how long each pool takes to hand over usable capacity.

    The same samples the scheduler ranks pools on, which is why every figure
    carries its sample count. A p95 over three provisions is not a p95, and a
    pool reporting no samples at all is not fast -- it is unmeasured, and the two
    are the same shape in a table sorted by latency.
    """

    _emit(session.client.request("GET", "/api/v1/capacity/launch-latency"), as_json=as_json, renderer=JsonRenderer())


@capacity.group("lease")
def lease() -> None:
    """Inspect one capacity lease."""


@lease.command("get")
@click.argument("lease_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def get_lease(session: CliSession, lease_id: str, as_json: bool) -> None:
    """Show one lease record.

    Listing leases is ``eyvan leases``; the single-lease read is what an
    operator needs when a workload names a lease and the listing is thousands of
    rows long.
    """

    _emit(session.client.request("GET", f"/api/v1/leases/{lease_id}"), as_json=as_json, renderer=JsonRenderer())


@capacity.command("queue-trend")
@click.option("--org", "org_id")
@click.option("--project", "project_id")
@click.option("--window-s", type=float, default=86_400.0, show_default=True)
@click.option("--buckets", type=int, default=24, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def queue_trend(
    session: CliSession,
    org_id: str | None,
    project_id: str | None,
    window_s: float,
    buckets: int,
    as_json: bool,
) -> None:
    """Report how deep the queue has been over a window.

    The number behind "should this fleet be bigger". Instantaneous queue depth
    answers it badly: a fleet that is empty now and was saturated for eight
    hours looks identical to one that has never been busy.
    """

    scope = session.workload_scope(org_id=org_id, project_id=project_id)
    payload = session.client.request(
        "GET",
        "/api/v1/capacity/queue-trend",
        params={
            "org_id": scope.org_id,
            "project_id": scope.project_id,
            "window_s": str(window_s),
            "buckets": str(buckets),
        },
    )
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@capacity.command("invariants")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def invariants(session: CliSession, as_json: bool) -> None:
    """Report which durability invariants the fleet is currently violating.

    The point-in-time ones. Budget monotonicity and storage growth are
    statements about a sequence of observations, so a single call cannot settle
    them; a soak run holds the previous snapshot and compares.

    An empty list is a real claim rather than a default: every check has a test
    that makes it fire, so silence means the checks ran and found nothing.
    """

    payload = session.client.request("GET", "/api/v1/system/invariants")
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@capacity.command("carbon")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def carbon(session: CliSession, as_json: bool) -> None:
    """Report grid intensity for the regions this fleet occupies.

    Expect ``no_provider`` everywhere: none ships, and Eyvan will not invent one.
    A number this control plane cannot vouch for is worse than none, because the
    placement decisions it produced would look deliberate -- so carbon is a
    report here and never an input to scheduling.
    """

    payload = session.client.request("GET", "/api/v1/capacity/carbon")
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@capacity.command("billing")
@click.option("--pool", "pool_id", help="Restrict the report to one capacity pool.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def billing(session: CliSession, pool_id: str | None, as_json: bool) -> None:
    """Report accrued and idle cell cost.

    The idle figure is not derivable from budgets: an idle cell holds no lease, so
    nothing settles and nothing is charged to a project. This is the only place a
    fleet's unused-but-paid-for capacity is visible.
    """

    payload = session.client.request(
        "GET",
        "/api/v1/capacity/billing",
        params={} if pool_id is None else {"pool_id": pool_id},
    )
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


def _entry(row: JsonValue) -> dict[str, tp.Any]:
    """Read one built fleet row, which this module constructed itself."""

    return tp.cast("dict[str, tp.Any]", row)


@capacity.command("fleets")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def fleets(session: CliSession, as_json: bool) -> None:
    """List the fleets this project can submit to, and what each provides.

    `capacity list` answers per pool, which is the right grain for choosing a
    shape and the wrong one for choosing a fleet: twenty rows of one fleet's
    pools do not tell you the fleet's name is the thing `eyvan use` wants. The
    commonest placement failure is a zone typed where a fleet name belongs.
    """

    pools = session.client.request(
        "GET",
        "/api/v1/capacity-pools",
        params={
            "org_id": session.org_id or DEFAULT_ORG_ID,
            "project_id": session.project_id or DEFAULT_PROJECT_ID,
        },
    )
    grouped: dict[str, dict[str, object]] = {}
    for pool in pools if isinstance(pools, list) else []:
        if not is_json_object(pool):
            continue
        name = str(pool.get("fleet_name") or "")
        row = grouped.setdefault(name, {"fleet": name, "pools": 0, "shapes": set(), "zones": set(), "disabled": 0})
        row["pools"] = int(row["pools"]) + 1  # type: ignore[call-overload]
        shape = pool.get("accelerator_type")
        if isinstance(shape, str) and shape:
            tp.cast("set[str]", row["shapes"]).add(shape)
        zone = pool.get("zone")
        if isinstance(zone, str) and zone:
            tp.cast("set[str]", row["zones"]).add(zone)
        if pool.get("enabled") is False:
            row["disabled"] = int(row["disabled"]) + 1  # type: ignore[call-overload]

    rows: JsonValue = [
        {
            "fleet": name,
            "pools": row["pools"],
            "disabled_pools": row["disabled"],
            "zones": sorted(tp.cast("set[str]", row["zones"])),
            "shapes": sort_accelerator_types(tp.cast("set[str]", row["shapes"])),
        }
        for name, row in sorted(grouped.items())
    ]
    if as_json:
        JsonRenderer().emit(rows)
        return
    if not rows:
        click.echo("No fleets are applied. Generate one with `eyvan autoscale generate --zone ZONE`.")
        return
    # A fleet whose pools are all disabled is one `eyvan down` drained. It is
    # still applied, and nothing can be submitted to it -- so it is listed under
    # its own heading rather than under a promise the plane will not keep.
    live = [row for row in rows if _entry(row)["pools"] != _entry(row)["disabled_pools"]]
    drained = [row for row in rows if _entry(row)["pools"] == _entry(row)["disabled_pools"]]
    if not live:
        click.echo("No fleet can accept work; every applied fleet is drained.")
    else:
        click.echo("Fleets you can submit to:")
    for row in live + drained:
        if drained and row is drained[0]:
            click.echo("\nDrained (apply a config to bring one back):")
        entry = _entry(row)
        shapes = tp.cast("list[str]", entry["shapes"])
        zones = tp.cast("list[str]", entry["zones"])
        # One entry per generation rather than the first three shapes and a
        # count: the count used to swallow whole families, because shapes sort
        # by generation and the v6e half of a v5p-heavy fleet fell inside it.
        listed = summarize_shapes(shapes)
        disabled = f"  ({entry['disabled_pools']} disabled)" if entry["disabled_pools"] else ""
        where = ", ".join(zones) or "local"
        click.echo(f"  {entry['fleet']:26} {where:16} {entry['pools']:>3} pools  {listed}{disabled}")


@capacity.command("usage")
@click.option("--since", help="Only count runs submitted after this ISO-8601 instant, e.g. 2026-08-01.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def usage(session: CliSession, since: str | None, as_json: bool) -> None:
    """Report what each person in this project used, and what it cost.

    Cost settles per lease and was only ever reported per pool, so a shared
    project could read its bill and never learn whose runs produced it.

    A zero cost beside non-zero hours means the pool declares no unit price --
    the count of such leases is reported rather than folded in, because a heavy
    user of unpriced capacity otherwise reads as an idle one.
    """

    params: dict[str, str] = {
        "org_id": session.org_id or DEFAULT_ORG_ID,
        "project_id": session.project_id or DEFAULT_PROJECT_ID,
    }
    if since:
        params["since"] = since
    payload = session.client.request("GET", "/api/v1/usage", params=params)
    if as_json:
        JsonRenderer().emit(payload)
        return
    subjects = payload.get("subjects") if is_json_object(payload) else None
    if not isinstance(subjects, list) or not subjects:
        click.echo("Nobody has submitted work in this project yet.")
        return
    click.echo(f"{'subject':16} {'runs':>5} {'hours':>8} {'cost':>10}")
    for entry in subjects:
        if not is_json_object(entry):
            continue
        hours = entry.get("accelerator_hours")
        cost = entry.get("settled_cost")
        unpriced = entry.get("unpriced_leases")
        # A zero cost beside real hours means the pool declares no price. Said
        # here rather than folded into the number, because the heaviest user of
        # unpriced capacity otherwise reads exactly like an idle one.
        note = "  (no unit price on the pools used)" if unpriced else ""
        click.echo(
            f"{entry.get('subject', '?')!s:16} {int(entry.get('workloads') or 0):>5} "
            f"{float(hours or 0):>8.2f} {float(cost or 0):>10.2f}{note}"
        )


@capacity.group("pool")
def pool() -> None:
    """Inspect capacity pools.

    There is no ``pool create``: pools are declared in a fleet config and applied
    by ``eyvan up``, so creating one here would fork the source of truth.
    """


@pool.command("list")
@click.option("--fleet", "fleet_name", help="Only pools in this fleet.")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def list_pools(session: CliSession, fleet_name: str | None, as_json: bool) -> None:
    """List capacity pool records."""

    payload = session.client.request(
        "GET",
        "/api/v1/capacity-pools",
        params={} if fleet_name is None else {"fleet_name": fleet_name},
    )
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@pool.command("get")
@click.argument("pool_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def get_pool(session: CliSession, pool_id: str, as_json: bool) -> None:
    """Show one capacity pool record."""

    payload = session.client.request("GET", f"/api/v1/capacity-pools/{pool_id}")
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@capacity.group("cell")
def cell() -> None:
    """Inspect one execution cell.

    Listing cells is ``eyvan cells``, which already exists; only the single-cell
    read was missing, so only that is added here.
    """


@cell.command("get")
@click.argument("cell_id")
@click.option("--json", "as_json", is_flag=True)
@click.pass_obj
def get_cell(session: CliSession, cell_id: str, as_json: bool) -> None:
    """Show one execution cell record."""

    payload = session.client.request("GET", f"/api/v1/cells/{cell_id}")
    _emit(payload, as_json=as_json, renderer=JsonRenderer())


@capacity.group("ingress")
def ingress() -> None:
    """Inspect and remove firewall rules opened for native GCP workers.

    These rules are created locally by the agent, via gcloud, and never through
    the control plane -- so they are cleaned up the same way, without a server.
    Provisioning deliberately never deletes them: a rule must survive a
    scale-to-zero for the next scale-up to reach its workers. That leaves
    removal as an operator decision, which is what these two commands are.
    """


@ingress.command("list")
@click.option("--project", required=True, help="GCP project to search for managed rules.")
@click.option("--json", "as_json", is_flag=True)
def list_ingress(project: str, as_json: bool) -> None:
    """List firewall rules this runtime created in a project."""

    from eyvan.capacity.providers.gcp.native_worker_ingress import list_managed_ingress_rules

    rules = list_managed_ingress_rules(project=project)
    payload = [
        {
            "name": rule.name,
            "network": rule.network.canonical_name,
            "network_tag": rule.network_tag,
            "port": rule.port,
            "source_range": rule.source_range,
        }
        for rule in rules
    ]
    if as_json:
        click.echo(json.dumps(payload, sort_keys=True, indent=2))
        return
    if not payload:
        click.echo(f"no Eyvan-managed native worker ingress rules in {project}")
        return
    for entry in payload:
        click.echo(f"{entry['name']}  tcp:{entry['port']}  from {entry['source_range']}  tag {entry['network_tag']}")


@ingress.command("release")
@click.option("--project", required=True, help="GCP project owning the rule.")
@click.option("--rule", "rule_names", multiple=True, help="Rule name; repeatable. Omit with --all to remove every rule.")
@click.option("--all", "release_all", is_flag=True, help="Release every managed rule found in the project.")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def release_ingress(project: str, rule_names: tuple[str, ...], release_all: bool, yes: bool) -> None:
    """Delete managed ingress rules, refusing anything this runtime does not own."""

    from eyvan.capacity.providers.gcp.native_worker_ingress import (
        list_managed_ingress_rules,
        release_managed_ingress_rule,
    )

    if release_all == bool(rule_names):
        raise click.UsageError("pass either --all or one or more --rule values, not both and not neither")

    targets = [rule.name for rule in list_managed_ingress_rules(project=project)] if release_all else list(rule_names)
    if not targets:
        click.echo(f"no Eyvan-managed native worker ingress rules in {project}")
        return

    # Named rules are shown before the prompt too. A firewall rule is removed
    # from a project that may be shared, so "what exactly is about to go" is
    # worth one extra line of output.
    click.echo(f"about to delete {len(targets)} firewall rule(s) in {project}:")
    for name in targets:
        click.echo(f"  {name}")
    if not yes:
        click.confirm("proceed?", abort=True)

    for name in targets:
        release_managed_ingress_rule(project=project, rule_name=name)
        click.echo(f"released {name}")


def register_capacity_commands(main: click.Group) -> None:
    """Attach the capacity command group to the CLI root."""

    main.add_command(capacity)


__all__ = ("capacity", "register_capacity_commands")
