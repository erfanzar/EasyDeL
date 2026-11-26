# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Terminal setup, generation, and support functions for Eyvan autoscale."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.capacity.autoscale import (
    AutoscaleConfigCache,
    AutoscaleDefaults,
    AutoscaleGenerationResult,
    AutoscaleGenerationService,
    CachedAutoscaleConfig,
    GenerationQuotaSnapshot,
    validate_gcp_project_id,
)
from eyvan.capacity.fleet_config import EyvanFleetConfig
from eyvan.capacity.providers.gcp.quota_discovery import GcloudQuotaError, GcpTpuQuotaDiscovery
from eyvan.records import CapacityClass
from eyvan.sdk.commands.fleet_status import is_interactive_output
from eyvan.sdk.terminal import print_block, terminal_console

_EGRESS_CHOICES: tuple[tuple[str, str, str], ...] = (
    ("1", "private-google-access", "Private Google APIs only"),
    ("2", "cloud-nat", "Private VM through Cloud NAT"),
    ("3", "public-internet", "VM with an external address"),
)
_NO_ELIGIBLE_POOLS_MESSAGE = "No eligible preemptible TPU worker pools were generated; inspect the emitted diagnostics."


def _emit(value: JsonValue) -> None:
    """Emit one JSON-safe command result without importing the root CLI module."""

    click.echo(json.dumps(value, indent=2, sort_keys=True))


def gcloud_default_project() -> str | None:
    """Read the active gcloud project without making a cloud API mutation."""

    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=provider_cli_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value and value != "(unset)" else None


def is_interactive() -> bool:
    """Return whether the autoscale command can safely prompt in a terminal."""

    return sys.stdin.isatty()


def autoscale_console(
    *,
    err: bool = False,
    force_terminal: bool | None = None,
    environment: Mapping[str, str] | None = None,
) -> Console:
    """Create a Rich console bound to Click's current output stream."""

    stream = click.get_text_stream("stderr" if err else "stdout")
    return terminal_console(stream, force_terminal=force_terminal, environment=environment)


def _quota_balance_label(quota: GenerationQuotaSnapshot) -> str:
    """Render an observed quota balance without presenting its ceiling as free."""

    if quota.quota_tpus is None:
        return "unbounded"
    if quota.available_tpus is None:
        return f"usage unavailable of {quota.quota_tpus}"
    return f"{quota.available_tpus} available of {quota.quota_tpus}"


@dataclass(slots=True)
class _SkippedReasonSummary:
    """Aggregate grouped omission diagnostics without expanding TPU catalog entries."""

    type_count: int = 0
    zones: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class AutoscaleGenerationRenderer:
    """Render one compact terminal summary for cached autoscale generation."""

    console: Console
    generation: AutoscaleGenerationResult

    def emit(self) -> None:
        """Display cached zone configs and bounded omitted-candidate diagnostics."""

        configs = self._configs() if self.generation.has_entries else None
        tables = [table for table in (configs, self._skipped()) if table is not None]
        # One width for both. Each table sized to its own content otherwise, and
        # the two disagreed by ten columns -- which looks careless side by side,
        # and is worse than that on a narrow terminal: the wider table folds
        # while the narrower one does not, so the pair reflows into each other
        # when somebody selects and copies them.
        widest = max((self.console.measure(table).maximum for table in tables), default=0)
        for table in tables:
            table.min_width = min(widest, self.console.width)
            print_block(self.console, table)

    def _configs(self) -> Table:
        """Render one row per generated zone instead of every TPU pool."""

        table = Table(
            title="[bold cyan]Eyvan TPU Autoscale[/]",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            pad_edge=False,
            # Sized to its content rather than stretched to the console. An
            # expanded table is exactly as wide as Rich *believes* the terminal
            # is, so every row overflows when that belief is wrong -- and the
            # overflow arrives as wrapped fragments interleaved with the rules.
        )
        table.add_column("Zone", style="bold white", no_wrap=True)
        table.add_column("TPU families", overflow="fold")
        table.add_column("Config", style="dim", overflow="fold")
        for entry in self.generation.entries:
            table.add_row(
                entry.config.provider.zone or "-",
                self._family_summary(entry),
                entry.path.name,
            )
        return table

    @staticmethod
    def _family_summary(entry: CachedAutoscaleConfig) -> str:
        """Return family counts and quota facts for one generated zone fleet."""

        family_counts: dict[str, int] = {}
        for pool in entry.config.pools.values():
            family = pool.accelerator_type.partition("-")[0]
            family_counts[family] = family_counts.get(family, 0) + 1
        quotas = {quota.generation: quota for quota in entry.generation_quotas}
        summaries: list[str] = []
        for family, count in sorted(family_counts.items()):
            summary = f"{family}: {count} {'type' if count == 1 else 'types'}"
            quota = quotas.get(family)
            if quota is not None:
                summary = f"{summary} | {_quota_balance_label(quota)}"
            summaries.append(summary)
        return "\n".join(summaries) or "-"

    def _skipped(self) -> Table | None:
        """Render grouped public skip diagnostics without relying on model internals."""

        summaries = self._skipped_reason_summaries()
        if not summaries:
            return None
        table = Table(
            title="[bold yellow]Skipped candidates[/]",
            box=box.SIMPLE_HEAD,
            header_style="bold yellow",
            pad_edge=False,
            # Sized to its content rather than stretched to the console. An
            # expanded table is exactly as wide as Rich *believes* the terminal
            # is, so every row overflows when that belief is wrong -- and the
            # overflow arrives as wrapped fragments interleaved with the rules.
        )
        table.add_column("Reason", overflow="fold")
        table.add_column("Summary", no_wrap=True)
        for reason, summary in sorted(summaries.items()):
            zone_count = len(summary.zones)
            type_label = "type" if summary.type_count == 1 else "types"
            zone_label = "zone" if zone_count == 1 else "zones"
            table.add_row(reason, f"{summary.type_count} {type_label} across {zone_count} {zone_label}")
        return table

    def _skipped_reason_summaries(self) -> dict[str, _SkippedReasonSummary]:
        """Read only stable JSON diagnostics so catalog grouping can evolve independently."""

        skipped = self.generation.as_dict()["skipped"]
        if not isinstance(skipped, list):
            return {}
        summaries: dict[str, _SkippedReasonSummary] = {}
        for candidate in skipped:
            if not is_json_object(candidate):
                continue
            reason = candidate.get("reason")
            zone = candidate.get("zone")
            count = candidate.get("type_count")
            if not isinstance(reason, str) or not isinstance(zone, str):
                continue
            type_count = count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 1
            summary = summaries.setdefault(reason, _SkippedReasonSummary())
            summary.type_count += type_count
            summary.zones.add(zone)
        return summaries


def _valid_default(value: str | None, validator: Callable[[str], str]) -> str | None:
    """Return a saved default only when it remains valid for reuse."""

    if value is None:
        return None
    try:
        return validator(value)
    except ValueError:
        return None


def _prompt_validated_text(
    console: Console,
    *,
    label: str,
    default: str | None,
    validator: Callable[[str], str],
    hint: str,
) -> str:
    """Prompt until a typed autoscale field passes its domain validator."""

    while True:
        value = Prompt.ask(label, default=default, console=console).strip()
        try:
            return validator(value)
        except ValueError as exc:
            console.print(f"[bold red]Invalid {hint}[/]: {exc}")


def _prompt_fleet_prefix(console: Console, default: str) -> str:
    """Prompt for a non-empty fleet-name prefix without imposing a size limit."""

    while True:
        value = Prompt.ask("Fleet prefix", default=default, console=console).strip()
        if value:
            return value
        console.print("[bold red]Fleet prefix is required.[/]")


def _prompt_bootstrap_egress(console: Console, default: str) -> str:
    """Present explicit network choices instead of a free-form egress string."""

    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("#", style="bold")
    table.add_column("Route", style="bold white")
    table.add_column("Use")
    default_choice = "3"
    for choice, value, description in _EGRESS_CHOICES:
        if value == default:
            default_choice = choice
        table.add_row(choice, value, description)
    print_block(console, table)
    selected = Prompt.ask(
        "Bootstrap network",
        choices=[choice for choice, _value, _description in _EGRESS_CHOICES],
        default=default_choice,
        console=console,
    )
    for choice, value, _description in _EGRESS_CHOICES:
        if choice == selected:
            return value
    raise AssertionError("prompt choice was not in the declared egress choices")


def _render_autoscale_defaults_review(console: Console, defaults: AutoscaleDefaults) -> None:
    """Render a compact review table before the profile is persisted."""

    project = defaults.project
    if project is None:
        raise AssertionError("prompted autoscale defaults are missing a GCP project")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Setting", style="bold cyan", min_width=24)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("GCP project", project)
    table.add_row("Fleet prefix", defaults.fleet_prefix)
    table.add_row("Bootstrap network", defaults.bootstrap_egress)
    table.add_row("Warm workers", str(defaults.min_workers))
    maximum = "unbounded" if defaults.max_workers == 0 else str(defaults.max_workers)
    table.add_row("Worker ceiling", maximum)
    print_block(console, Panel(table, title="[bold cyan]Review autoscale defaults[/]", border_style="cyan"))


def _render_autoscale_defaults_saved(console: Console, path: Path) -> None:
    """Show a concise successful completion state for the setup wizard."""

    print_block(
        console,
        Panel(
            "[bold green]Autoscale defaults saved[/]",
            title="[bold cyan]Eyvan[/]",
            border_style="green",
        ),
    )
    console.print(f"[dim]Profile:[/] {path}")


def _prompt_autoscale_defaults(
    current: AutoscaleDefaults,
    *,
    review: bool = True,
    err: bool = False,
) -> AutoscaleDefaults:
    """Run the terminal setup wizard for user-owned autoscale YAML fields."""

    console = autoscale_console(err=err)
    print_block(
        console,
        Panel(
            "Set reusable TPU autoscale defaults. Platform routing stays hidden.",
            title="[bold cyan]Eyvan TPU Autoscale[/]",
            border_style="cyan",
        ),
    )
    valid_project = _valid_default(current.project, validate_gcp_project_id)

    while True:
        project = _prompt_validated_text(
            console,
            label="GCP project ID",
            default=valid_project or gcloud_default_project(),
            validator=validate_gcp_project_id,
            hint="GCP project",
        )
        fleet_prefix = _prompt_fleet_prefix(console, current.fleet_prefix)
        bootstrap_egress = _prompt_bootstrap_egress(console, current.bootstrap_egress)
        min_workers = IntPrompt.ask("Warm workers", default=current.min_workers, console=console)
        while min_workers < 0 or min_workers > 10_000:
            console.print("[bold red]Warm workers must be between 0 and 10000.[/]")
            min_workers = IntPrompt.ask("Warm workers", default=current.min_workers, console=console)
        while True:
            max_workers = IntPrompt.ask("Worker ceiling (0 is unbounded)", default=current.max_workers, console=console)
            if 0 <= max_workers <= 10_000 and (max_workers == 0 or max_workers >= min_workers):
                break
            console.print("[bold red]Worker ceiling must be 0 or at least the warm-worker count.[/]")
        defaults = AutoscaleDefaults(
            project=project,
            fleet_prefix=fleet_prefix,
            bootstrap_egress=bootstrap_egress,
            min_workers=min_workers,
            max_workers=max_workers,
        )
        if not review:
            return defaults
        _render_autoscale_defaults_review(console, defaults)
        if Confirm.ask("Save these defaults", default=True, console=console):
            return defaults
        console.print("[yellow]Nothing saved. Update the fields below.[/]")


def load_autoscale_defaults(
    cache: AutoscaleConfigCache,
    defaults_path: Path | None,
    *,
    persist_if_prompted: bool = True,
) -> AutoscaleDefaults:
    """Load a complete YAML profile or create the default one through the TUI."""

    try:
        defaults = cache.load_defaults(defaults_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    validation_error: ValueError | None = None
    if defaults.complete:
        try:
            defaults.validate_for_generation()
        except ValueError as exc:
            validation_error = exc
        else:
            return defaults
    missing = ", ".join(defaults.missing_required_fields)
    if defaults_path is not None:
        problem = str(validation_error) if validation_error is not None else f"missing {missing}"
        raise click.ClickException(
            f"autoscale defaults {defaults_path} is invalid: {problem}; fix that YAML or run `eyvan autoscale init`"
        )
    if not is_interactive():
        problem = str(validation_error) if validation_error is not None else f"missing {missing}"
        raise click.ClickException(
            f"{cache.defaults_path} is invalid: {problem}; run `eyvan autoscale init` to repair the YAML profile"
        )
    if validation_error is not None:
        autoscale_console(err=True).print(f"[yellow]Saved defaults need repair:[/] {validation_error}")
    resolved = _prompt_autoscale_defaults(defaults, review=persist_if_prompted)
    if not persist_if_prompted:
        click.echo("Using prompted autoscale defaults for this dry run; nothing was saved.", err=True)
        return resolved
    try:
        cache.save_defaults(resolved)
    except OSError as exc:
        raise click.ClickException(f"could not save autoscale defaults {cache.defaults_path}: {exc}") from exc
    _render_autoscale_defaults_saved(autoscale_console(err=True), cache.defaults_path)
    return resolved


def choose_cached_config(
    entries: tuple[CachedAutoscaleConfig, ...],
    *,
    selected_fleet: str | None,
) -> CachedAutoscaleConfig:
    """Pick a cached fleet interactively or deterministically for automation."""

    if not entries:
        raise click.ClickException(
            "no cached autoscale configs have a live usable quota balance; rerun `eyvan up --quota-based`"
        )
    if selected_fleet is not None:
        selected = next((entry for entry in entries if entry.config.name == selected_fleet), None)
        if selected is None:
            available = ", ".join(entry.config.name for entry in entries)
            raise click.ClickException(
                f"cached autoscale fleet {selected_fleet!r} was not found; available: {available}"
            )
        return selected
    if len(entries) == 1:
        return entries[0]
    if not is_interactive():
        available = ", ".join(entry.config.name for entry in entries)
        raise click.ClickException(f"multiple cached autoscale configs: {available}; rerun with --select FLEET")
    console = autoscale_console()
    table = Table(title="[bold cyan]Cached Eyvan TPU zones[/]", header_style="bold cyan", pad_edge=False)
    table.add_column("#", style="bold", justify="right")
    table.add_column("Zone", style="bold white")
    table.add_column("TPU pools", overflow="fold")
    table.add_column("Fleet", style="dim", overflow="fold")
    for index, entry in enumerate(entries, start=1):
        generations: dict[str, list[str]] = {}
        for pool in entry.config.pools.values():
            generation = pool.accelerator_type.partition("-")[0]
            generations.setdefault(generation, []).append(pool.accelerator_type)
        quotas = {quota.generation: quota for quota in entry.generation_quotas}
        summaries: list[str] = []
        for generation, accelerator_types in sorted(generations.items()):
            count = len(accelerator_types)
            detail = f"{generation}: {count} {'type' if count == 1 else 'types'}"
            quota = quotas.get(generation)
            if quota is not None:
                detail = f"{detail} | {_quota_balance_label(quota)}"
            summaries.append(detail)
        table.add_row(
            str(index),
            entry.config.provider.zone or "-",
            "\n".join(summaries),
            entry.config.name,
        )
    print_block(console, table)
    selected_index = click.prompt("Select config", type=click.IntRange(1, len(entries)))
    return entries[selected_index - 1]


def render_fleet_applied(
    console: Console,
    config: EyvanFleetConfig,
    *,
    selected: CachedAutoscaleConfig | None,
) -> None:
    """Render the selected fleet without exposing the internal API record."""

    pool_groups: dict[str, int] = {}
    capacity_classes: set[str] = set()
    for pool in config.pools.values():
        group = pool.accelerator_type.partition("-")[0] if pool.kind == "tpu" else pool.kind
        pool_groups[group] = pool_groups.get(group, 0) + 1
        capacity_classes.add((pool.capacity_class or config.capacity_class).value)

    pool_summary = "\n".join(
        f"{group}: {count} {'type' if count == 1 else 'types'}" for group, count in sorted(pool_groups.items())
    )
    if capacity_classes == {"spot"}:
        capacity_summary = "Spot / preemptible"
    else:
        capacity_summary = ", ".join(value.replace("_", " ").title() for value in sorted(capacity_classes))

    public_limits = (
        selected.worker_limits
        if selected is not None
        else tuple(0 if pool.max_cells == 10_000 else pool.max_cells for pool in config.pools.values())
    )
    worker_ceiling = "unbounded" if any(limit == 0 for limit in public_limits) else str(sum(public_limits))

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", min_width=20)
    table.add_column("Value", style="white", overflow="fold")
    table.add_row("Fleet", config.name)
    table.add_row("Zone", config.provider.zone or "-")
    table.add_row("TPU pools" if all(pool.kind == "tpu" for pool in config.pools.values()) else "Pools", pool_summary)
    table.add_row("Capacity", capacity_summary)
    table.add_row("Warm workers", str(sum(pool.min_cells for pool in config.pools.values())))
    table.add_row("Worker ceiling", worker_ceiling)
    if selected is not None and selected.generation_quotas:
        quota_summary = "\n".join(
            f"{quota.generation}: {_quota_balance_label(quota)} TPU units" for quota in selected.generation_quotas
        )
        table.add_row("Live quota", quota_summary)
    table.add_row("Config", selected.path.name if selected is not None else "provided YAML")
    print_block(console, Panel(table, title="[bold green]Fleet applied[/]", border_style="green"))


def require_preemptible_tpus(config: EyvanFleetConfig) -> None:
    """Reject an explicit preemptible apply that contains non-Spot TPU pools."""

    tpu_pools = [pool for pool in config.pools.values() if pool.kind == "tpu"]
    if not tpu_pools:
        raise click.ClickException(
            "--preemptible only applies to fleets containing TPU pools; apply it to a TPU fleet or drop the flag"
        )
    non_spot = [
        pool.accelerator_type
        for pool in tpu_pools
        if (pool.capacity_class or config.capacity_class) is not CapacityClass.SPOT
    ]
    if non_spot:
        raise click.ClickException(
            f"--preemptible requires Spot TPU pools; {non_spot[0]!r} is not Spot; set that pool's "
            "capacity_class to spot in the fleet configuration, or drop --preemptible"
        )


def generate_autoscale_entries(
    *,
    cache: AutoscaleConfigCache,
    defaults: AutoscaleDefaults,
    zones: tuple[str, ...],
    generations: tuple[str, ...],
    persist: bool = True,
) -> AutoscaleGenerationResult:
    """Generate one zone fleet for every accessible Spot TPU quota zone."""

    try:
        return AutoscaleGenerationService(GcpTpuQuotaDiscovery()).generate(
            cache=cache,
            defaults=defaults,
            zones=zones,
            generations=generations,
            persist=persist,
        )
    except (GcloudQuotaError, ValueError) as exc:
        raise click.ClickException(f"could not discover preemptible TPU quotas: {exc}") from exc


@click.group("autoscale")
def autoscale_group() -> None:
    """Generate, cache, and select Spot TPU autoscale fleet configurations."""


@autoscale_group.command("init")
@click.option(
    "--defaults",
    "defaults_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Write this YAML profile instead of ~/.eyvan/autoscale/defaults.yaml.",
)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), help="Override ~/.eyvan/autoscale.")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the saved profile as JSON instead of the setup completion view.",
)
def initialize_autoscale(defaults_path: Path | None, cache_dir: Path | None, as_json: bool) -> None:
    """Create or edit the YAML profile used by autoscale generation."""

    if not is_interactive():
        raise click.ClickException(
            "autoscale init requires a terminal; create the defaults YAML directly for automation"
        )
    cache = AutoscaleConfigCache(cache_dir)
    try:
        current = cache.load_defaults(defaults_path)
        defaults = _prompt_autoscale_defaults(current, err=as_json)
        target = cache.save_defaults(defaults, defaults_path)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"could not save autoscale defaults: {exc}") from exc
    if as_json:
        payload: JsonObject = {"defaults": defaults.as_document(), "path": str(target)}
        _emit(payload)
    else:
        _render_autoscale_defaults_saved(autoscale_console(), target)


@autoscale_group.command("generate")
@click.option("--zone", "zones", multiple=True, help="Generate for a zone; repeatable. Omit to discover matching zones.")
@click.option(
    "--tpu-generation", "generations", multiple=True, help="Generate for a TPU generation, such as v5p; repeatable."
)
@click.option(
    "--defaults",
    "defaults_path",
    type=click.Path(path_type=Path, dir_okay=False),
    help="Read autoscale identity/defaults from YAML instead of ~/.eyvan/autoscale/defaults.yaml.",
)
@click.option(
    "--quota-based",
    is_flag=True,
    expose_value=False,
    help="Compatibility flag; generated TPU fleets are always quota-backed and preemptible.",
)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), help="Override ~/.eyvan/autoscale.")
@click.option("--json", "as_json", is_flag=True, help="Emit generated config and quota details as JSON.")
def generate_autoscale(
    zones: tuple[str, ...],
    generations: tuple[str, ...],
    defaults_path: Path | None,
    cache_dir: Path | None,
    as_json: bool,
) -> None:
    """Cache one valid Spot TPU fleet YAML file per discovered GCP zone."""

    cache = AutoscaleConfigCache(cache_dir)
    defaults = load_autoscale_defaults(cache, defaults_path)
    generation = generate_autoscale_entries(
        cache=cache,
        defaults=defaults,
        zones=zones,
        generations=generations,
    )
    payload = generation.as_dict()
    del payload["cached"]
    if as_json or not is_interactive_output():
        _emit(payload)
    else:
        AutoscaleGenerationRenderer(autoscale_console(), generation).emit()
    if not generation.has_entries:
        click.echo(_NO_ELIGIBLE_POOLS_MESSAGE, err=True)
        raise click.exceptions.Exit(1)


def register_autoscale_commands(group: click.Group) -> None:
    """Attach autoscale terminal commands to the root command group."""

    group.add_command(autoscale_group)


__all__ = (
    "autoscale_console",
    "choose_cached_config",
    "gcloud_default_project",
    "generate_autoscale_entries",
    "is_interactive",
    "is_interactive_output",
    "load_autoscale_defaults",
    "register_autoscale_commands",
    "render_fleet_applied",
    "require_preemptible_tpus",
)
