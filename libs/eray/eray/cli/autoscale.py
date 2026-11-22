# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""`eray autoscale` — Ray cluster-launcher configs for elastic TPU clusters.

Generates per-zone launcher YAMLs (all-spot by default; every TPU node type
carries the eray-canonical resource labels so the pool scheduler works) and
wraps `ray up/down/get-head-ip`. For reliable large slices, prefer the
QR-based `eray fleet` path; this one is for elastic many-small-slices use.

`up`/`down`/`status` never require a YAML path: with no CONFIG they discover
generated profiles under `--dir` and either auto-select the only one or offer
a numbered picker. Every `up`/`down` also upserts a ``kind="launcher"``
record into the same fleet registry `eray fleet` uses, which is what lets
`eray autoscale status` (no CONFIG) and `eray dashboard` show every launcher
cluster's live state without the operator tracking YAML paths by hand.
"""

from __future__ import annotations

import json as json_lib
import time

import click

from ..provision.registry import ClusterRecord, ClusterRegistry
from .utils import detect_project, error, info, print_table, success, warning


def _pick_profile(profiles: list, *, verb: str):
    """Resolve which profile to act on when no CONFIG was given.

    Args:
        profiles: Discovered profiles (see `provision.launcher.list_profiles`).
        verb: The action being taken, for the prompt copy (e.g. "bring up").

    Returns:
        The chosen `AutoscaleProfile`.

    Raises:
        click.ClickException: No profiles exist, or no selection was made
            (e.g. stdin has nothing left to read — a non-interactive call).
    """
    if not profiles:
        raise click.ClickException("no generated profiles found — run `eray autoscale generate` first.")
    if len(profiles) == 1:
        profile = profiles[0]
        info(f"one profile found: {profile.name} — using it.")
        return profile
    click.echo(f"Multiple profiles found — choose one to {verb}:")
    for idx, profile in enumerate(profiles, start=1):
        click.echo(f"  {idx}) {profile.name}  ({profile.zone or '?'}, {profile.project or '?'})")
    try:
        choice = click.prompt("select", type=click.IntRange(1, len(profiles)))
    except click.Abort:
        raise click.ClickException(
            "no profile selected — pass a profile name explicitly (non-interactive session?)."
        ) from None
    return profiles[choice - 1]


def _upsert_launcher_record(profile, *, state: str, stamp: str, head_ip: str | None = None) -> None:
    """Persist a launcher-kind cluster's state after `up`/`down`.

    Args:
        profile: The `AutoscaleProfile` that was acted on.
        state: New `ClusterRecord.state` (``"HEALTHY"`` or ``"DOWN"``).
        stamp: Which timestamp field to stamp with now — ``"last_up_ts"`` or
            ``"last_down_ts"``.
        head_ip: Best-effort head IP (`up` only; `down` leaves it as-is).
    """
    registry = ClusterRegistry.from_config()
    if registry.get(profile.name) is None:
        registry.upsert(
            ClusterRecord(
                name=profile.name,
                kind="launcher",
                project=profile.project,
                zone=profile.zone,
                config_path=str(profile.path),
                state=state,
                head_ip=head_ip,
                **{stamp: time.time()},
            )
        )
        return

    def mutate(r: ClusterRecord) -> None:
        r.kind = "launcher"
        r.project = profile.project or r.project
        r.zone = profile.zone or r.zone
        r.config_path = str(profile.path)
        r.state = state
        if head_ip is not None:
            r.head_ip = head_ip
        if stamp == "last_up_ts":
            # A fresh bring-up invalidates any earlier "died at T" history —
            # otherwise a live cluster can flash DOWN/"died ... ago" if the
            # GCE probe runs before the new head instance is discoverable.
            r.last_down_ts = None
        setattr(r, stamp, time.time())

    registry.mutate_record(profile.name, mutate)


def launcher_row_status(
    cluster_name: str, *, project: str | None, zone: str | None, state: str, last_down_ts: float | None
) -> tuple[str, str]:
    """Live (status, since) for one launcher-kind cluster.

    Shared by `eray autoscale status` (no CONFIG) and `eray dashboard` so
    the two views can never disagree about what "running" means.

    Args:
        cluster_name: The launcher config's ``cluster_name``.
        project: GCP project id, or None (skips the live probe).
        zone: GCP zone, or None (skips the live probe).
        state: Fallback status when no live head instance is found and
            there's no `last_down_ts` to report against (e.g. registered
            but never brought up).
        last_down_ts: Unix timestamp of the last `eray autoscale down`, or
            None.

    Returns:
        ``(status, since)`` display strings — status is either a live GCE
        instance status (``RUNNING``, ``STAGING``, ...), ``"DOWN"``, or the
        fallback `state`; since is "up <duration>", "died <duration> ago",
        "?" (status known, age unknown), or "-".
    """
    from ..provision.launcher import gce_instances_for_cluster
    from .utils import format_ago, format_duration, parse_gcp_timestamp

    instances = gce_instances_for_cluster(cluster_name, project=project, zone=zone) if project and zone else []
    head = next((i for i in instances if (i.get("labels") or {}).get("ray-node-type") == "head"), None)
    if head is not None:
        status = str(head.get("status", "UNKNOWN"))
        created = parse_gcp_timestamp(head.get("creationTimestamp"))
        since = f"up {format_duration(time.time() - created)}" if created else "?"
    elif last_down_ts is not None:
        status = "DOWN"
        since = f"died {format_ago(last_down_ts)}"
    else:
        status = state
        since = "-"
    return status, since


def _launcher_status_rows(profiles_dir) -> list[dict]:
    """Merge on-disk profiles and registered launcher clusters into rows.

    Each row's live GCE probe is an independent gcloud subprocess; a thread
    pool runs them concurrently so `eray autoscale status` doesn't scale
    linearly with the number of profiles (mirrors `run_on_all_hosts` in
    `cli/utils.py`).

    Args:
        profiles_dir: Directory profiles are discovered in.

    Returns:
        One row per known profile (on disk, registered, or both): name,
        zone, status (a live GCE probe when project/zone are known), head,
        since (uptime, or "died N ago", or "-").
    """
    import concurrent.futures

    from ..provision.launcher import list_profiles

    registry = ClusterRegistry.from_config()
    records = {name: r for name, r in registry.load().items() if r.kind == "launcher"}
    profiles = {p.name: p for p in list_profiles(profiles_dir)}
    names = sorted(set(profiles) | set(records))
    if not names:
        return []

    def build_row(name: str) -> dict:
        profile = profiles.get(name)
        record = records.get(name)
        project = (profile.project if profile else None) or (record.project if record else None)
        zone = (profile.zone if profile else None) or (record.zone if record else None)
        cluster_name = profile.cluster_name if profile else name

        status, since = launcher_row_status(
            cluster_name,
            project=project,
            zone=zone,
            state=record.state if record else "UNREGISTERED",
            last_down_ts=record.last_down_ts if record else None,
        )
        return {
            "name": name,
            "zone": zone or "?",
            "status": status,
            "head": (record.head_ip if record and status == "RUNNING" else None) or "-",
            "since": since,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(names), 8)) as pool:
        return list(pool.map(build_row, names))


def register(cli: click.Group) -> None:
    """Register the `autoscale` command group on the root CLI.

    Args:
        cli: The root click group.
    """

    @cli.group()
    def autoscale() -> None:
        """Ray cluster-launcher configs for elastic (autoscaled) TPU clusters."""

    @autoscale.command()
    @click.option("--project", "-p", default=None, help="GCP project (auto-detected).")
    @click.option("--zones", "-z", default=None, help="Comma-separated zones (default: every TPU location).")
    @click.option("--families", default="v4,v5e,v5p,v6e", show_default=True, help="TPU families to include.")
    @click.option("--image", default=None, help="Docker image for head/workers (default: easydel latest-tpu).")
    @click.option(
        "--spot/--on-demand", "spot", default=True, show_default=True, help="Capacity tier for TPU node types."
    )
    @click.option(
        "--output-dir",
        "-o",
        "output_dir",
        default=None,
        help="Where to write the YAMLs (default: ~/.eray/autoscale).",
    )
    def generate(project, zones, families, image, spot, output_dir):
        """Generate per-zone cluster-launcher YAMLs.

        \b
        Examples:
            eray autoscale generate --zones us-central1-a
            eray autoscale generate --families v5p --on-demand
        """
        from ..provision.launcher import DEFAULT_IMAGE, DEFAULT_OUTPUT_DIR, generate_configs

        project = project or detect_project()
        if not project:
            raise click.ClickException("Could not resolve a GCP project; pass --project.")
        zone_list = [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        family_tuple = tuple(f.strip() for f in families.split(",") if f.strip())
        try:
            written = generate_configs(
                project,
                zones=zone_list,
                families=family_tuple,
                image=image or DEFAULT_IMAGE,
                spot=spot,
                output_dir=output_dir or DEFAULT_OUTPUT_DIR,
            )
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        if not written:
            info("no zones offered the requested TPU families; nothing written.")
            return
        for path in written:
            success(f"wrote {path}")

    _dir_opt = click.option(
        "--dir",
        "profiles_dir",
        default=None,
        help="Where generated profiles are discovered from (default: ~/.eray/autoscale).",
    )

    @autoscale.command()
    @click.argument("config", required=False)
    @_dir_opt
    @click.option("--yes", "-y", is_flag=True, default=False, help="Skip the ray up confirmation.")
    def up(config, profiles_dir, yes):
        """Bring a launcher cluster up (`ray up`).

        \b
        With no CONFIG: picks from generated profiles (auto-selects the
        only one, prompts when there's more than one). CONFIG may also be a
        YAML path or a bare profile name (its file stem) to skip the picker.

        \b
        Examples:
            eray autoscale up                       # picker
            eray autoscale up easydel-us-east5-a     # by profile name
            eray autoscale up ./my-cluster.yaml      # by path
        """
        from ..provision.launcher import (
            DEFAULT_OUTPUT_DIR,
            launcher_head_ip,
            list_profiles,
            load_profile,
            ray_up,
            resolve_profile,
        )

        profiles_dir = profiles_dir or DEFAULT_OUTPUT_DIR
        if config:
            try:
                path = resolve_profile(config, profiles_dir)
            except FileNotFoundError as exc:
                raise click.ClickException(str(exc)) from exc
        else:
            path = _pick_profile(list_profiles(profiles_dir), verb="bring up").path

        if not yes:
            click.confirm(f"ray up {path} — create/update this cluster?", abort=True)
        try:
            ray_up(path, yes=True)
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        success(f"cluster up: {path}")

        profile = load_profile(path)
        if profile is None:
            warning("could not re-read cluster_name from CONFIG; skipping fleet registry update.")
            return
        head_ip = None
        try:
            head_ip = launcher_head_ip(path)
        except Exception as exc:
            warning(f"could not resolve head IP ({exc}); registering without it — `eray dashboard ls` will show '-'.")
        _upsert_launcher_record(profile, state="HEALTHY", stamp="last_up_ts", head_ip=head_ip)

    @autoscale.command()
    @click.argument("config", required=False)
    @_dir_opt
    @click.option("--yes", "-y", is_flag=True, default=False, help="Skip the ray down confirmation.")
    def down(config, profiles_dir, yes):
        """Tear a launcher cluster down (`ray down`).

        \b
        With no CONFIG: picks from generated/registered profiles the same
        way `up` does. CONFIG may also be a YAML path or a bare profile name.
        """
        from ..provision.launcher import (
            DEFAULT_OUTPUT_DIR,
            list_profiles,
            load_profile,
            ray_down,
            resolve_profile,
        )

        profiles_dir = profiles_dir or DEFAULT_OUTPUT_DIR
        if config:
            try:
                path = resolve_profile(config, profiles_dir)
            except FileNotFoundError as exc:
                raise click.ClickException(str(exc)) from exc
        else:
            path = _pick_profile(list_profiles(profiles_dir), verb="tear down").path

        if not yes:
            click.confirm(f"ray down {path} — terminate this cluster's nodes?", abort=True)
        try:
            ray_down(path, yes=True)
        except Exception as exc:
            error(str(exc))
            raise click.ClickException(str(exc)) from exc
        success(f"cluster down: {path}")

        profile = load_profile(path)
        if profile is None:
            warning("could not re-read cluster_name from CONFIG; skipping fleet registry update.")
            return
        _upsert_launcher_record(profile, state="DOWN", stamp="last_down_ts")

    @autoscale.command()
    @click.argument("config", required=False)
    @_dir_opt
    @click.option("--json", "as_json", is_flag=True, default=False, help="Output the no-CONFIG table as JSON.")
    def status(config, profiles_dir, as_json):
        """Show launcher clusters: every known profile, or one cluster's detail.

        \b
        With no CONFIG: a table of every generated/registered profile —
        zone, live status (a GCE instance-label probe), head, and how long
        it's been up (or when it went down). With CONFIG: one cluster's
        head + resource-usage detail (today's behavior, unchanged).
        """
        from ..provision.launcher import DEFAULT_OUTPUT_DIR

        profiles_dir = profiles_dir or DEFAULT_OUTPUT_DIR
        if config:
            from ..provision.fleet import head_reachable
            from ..provision.launcher import launcher_head_ip, resolve_profile

            try:
                path = resolve_profile(config, profiles_dir)
            except FileNotFoundError as exc:
                raise click.ClickException(str(exc)) from exc
            try:
                head_ip = launcher_head_ip(path)
            except Exception as exc:
                error(f"cluster not reachable via ray get-head-ip: {exc}")
                raise click.ClickException(str(exc)) from exc
            info(f"head: {head_ip}")
            info(f"ray head port: {'up' if head_reachable(head_ip) else 'down'}")
            info(f"resources: eray resources -a {head_ip}:6379")
            return

        rows = _launcher_status_rows(profiles_dir)
        if as_json:
            click.echo(json_lib.dumps(rows, indent=2))
            return
        if not rows:
            info("no profiles found — run `eray autoscale generate` first.")
            return
        print_table(rows)
