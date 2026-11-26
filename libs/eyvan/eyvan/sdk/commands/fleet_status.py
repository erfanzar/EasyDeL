# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Human-oriented rendering for fleet status."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from eyvan.records import CapacityClass, CapacityPoolRecord, CellRecord, CellState, ResourceLimits
from eyvan.sdk.terminal import print_block, stream_is_terminal, terminal_console
from eyvan.server.api.schemas import FleetStatusResponse

_ATTENTION_CELL_LIMIT = 12


def is_interactive_output() -> bool:
    """Return whether stdout is a physical terminal suitable for human output."""

    return stream_is_terminal(click.get_text_stream("stdout"))


def fleet_status_console(
    *,
    force_terminal: bool | None = None,
    environment: Mapping[str, str] | None = None,
) -> Console:
    """Create a Rich console bound to Click's stdout stream."""

    return terminal_console(
        click.get_text_stream("stdout"),
        force_terminal=force_terminal,
        environment=environment,
    )


def _quantity(value: float | int) -> str:
    """Format one non-negative resource quantity without redundant decimals."""

    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _resources(resources: ResourceLimits) -> Text:
    """Render the non-zero resources in one worker shape."""

    values: list[str] = []
    if resources.tpu:
        values.append(f"{resources.tpu} TPU")
    if resources.gpu:
        values.append(f"{resources.gpu} GPU")
    if resources.cpu:
        values.append(f"{_quantity(resources.cpu)} CPU")
    if resources.memory_gib:
        values.append(f"{_quantity(resources.memory_gib)} GiB")
    values.extend(f"{_quantity(amount)} {name}" for name, amount in sorted(resources.custom.items()) if amount)
    return Text(", ".join(values) or "-")


def _capacity_class(capacity_class: CapacityClass) -> str:
    """Return the operator-facing capacity-class label."""

    if capacity_class is CapacityClass.SPOT:
        return "Spot / preemptible"
    return capacity_class.value.replace("_", " ").title()


def _pool_sort_key(pool: CapacityPoolRecord) -> tuple[str, int, str]:
    """Sort accelerator families by their numeric worker size when present."""

    family, separator, size = pool.node_type.rpartition("-")
    if separator and size.isdigit():
        return family, int(size), pool.node_type
    return pool.node_type, -1, pool.node_type


def _cell_state_style(state: CellState) -> str:
    """Return a Rich style for one durable cell state."""

    if state is CellState.READY:
        return "green"
    if state is CellState.UNAVAILABLE:
        return "bold red"
    if state in {CellState.REGISTERING, CellState.DRAINING, CellState.DELETING}:
        return "yellow"
    return "dim"


@dataclass(frozen=True, slots=True)
class PoolCellSummary:
    """Worker counts for cells belonging to one capacity pool."""

    ready: int
    active: int
    registering: int
    draining: int
    unavailable: int

    @classmethod
    def from_cells(cls, cells: tuple[CellRecord, ...]) -> PoolCellSummary:
        """Aggregate physical workers rather than logical cell records."""

        return cls(
            ready=sum(cell.worker_count for cell in cells if cell.state is CellState.READY),
            active=sum(cell.worker_count for cell in cells if cell.state is not CellState.DELETED),
            registering=sum(cell.worker_count for cell in cells if cell.state is CellState.REGISTERING),
            draining=sum(cell.worker_count for cell in cells if cell.state in {CellState.DRAINING, CellState.DELETING}),
            unavailable=sum(cell.worker_count for cell in cells if cell.state is CellState.UNAVAILABLE),
        )

    def label(self, target: int) -> Text:
        """Return a compact worker-count label."""

        suffix = f" / {target} target" if self.active < target else ""
        return Text(f"{self.ready} ready / {self.active} active{suffix}")

    def state(self, target: int) -> tuple[str, str]:
        """Return the pool's operator-facing state and Rich style."""

        if self.unavailable:
            return "degraded", "bold red"
        if self.draining:
            return "draining", "yellow"
        if self.registering:
            return "scaling", "yellow"
        if self.active < target:
            return "under-provisioned", "yellow"
        if self.ready:
            return "ready", "green"
        return "idle", "dim"


@dataclass(frozen=True, slots=True)
class SharedQuotaSummary:
    """Usage of one provider quota shared by several worker shapes."""

    generation: str
    location: str
    capacity_class: CapacityClass
    used: int
    limit: int

    def label(self) -> Text:
        """Return a compact public quota description."""

        return Text(
            f"{self.generation} | {_capacity_class(self.capacity_class)} | {self.location}: "
            f"{self.used} / {self.limit} TPU"
        )


@dataclass(frozen=True, slots=True)
class FleetStatusView:
    """Typed, presentation-ready aggregate of one fleet response."""

    status: FleetStatusResponse
    cells_by_pool: dict[str, tuple[CellRecord, ...]]
    now: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def build(cls, status: FleetStatusResponse) -> FleetStatusView:
        """Index cells and capture one timestamp for consistent warm floors."""

        indexed: dict[str, list[CellRecord]] = {pool.id: [] for pool in status.pools}
        for cell in status.cells:
            if cell.pool_id is not None and cell.pool_id in indexed:
                indexed[cell.pool_id].append(cell)
        return cls(
            status=status,
            cells_by_pool={
                pool_id: tuple(sorted(cells, key=lambda cell: cell.id)) for pool_id, cells in indexed.items()
            },
        )

    def pool_cells(self, pool: CapacityPoolRecord) -> tuple[CellRecord, ...]:
        """Return cells owned by ``pool``."""

        return self.cells_by_pool[pool.id]

    def pool_target(self, pool: CapacityPoolRecord) -> int:
        """Return the autoscaler's warm or first-apply floor at the view timestamp."""

        if not pool.enabled:
            return 0
        return pool.initial_target if pool.initialized_at is None else pool.warm_floor(self.now)

    @property
    def active_cells(self) -> tuple[CellRecord, ...]:
        """Return non-deleted fleet cells."""

        return tuple(cell for cell in self.status.cells if cell.state is not CellState.DELETED)

    @property
    def attention_cells(self) -> tuple[CellRecord, ...]:
        """Return active cells that are not ready for work."""

        return tuple(cell for cell in self.active_cells if cell.state is not CellState.READY)

    @property
    def fleet_state(self) -> tuple[str, str]:
        """Return the fleet's aggregate state and Rich style."""

        active = self.active_cells
        if any(cell.state is CellState.UNAVAILABLE for cell in active):
            return "degraded", "bold red"
        if any(
            PoolCellSummary.from_cells(self.pool_cells(pool)).active < self.pool_target(pool)
            for pool in self.status.pools
        ):
            return "under-provisioned", "yellow"
        if any(cell.state in {CellState.REGISTERING, CellState.DRAINING, CellState.DELETING} for cell in active):
            return "changing", "yellow"
        if active:
            return "ready", "green"
        return "idle", "dim"

    @property
    def shared_quotas(self) -> tuple[SharedQuotaSummary, ...]:
        """Aggregate quota usage without repeating a shared limit per pool."""

        grouped: dict[str, list[CapacityPoolRecord]] = {}
        for pool in self.status.pools:
            if pool.quota_group is not None:
                grouped.setdefault(pool.quota_group, []).append(pool)
        summaries: list[SharedQuotaSummary] = []
        for pools in grouped.values():
            first = pools[0]
            if first.quota_limit is None:
                continue
            used = sum(
                cell.worker_count * pool.quota_units_per_cell
                for pool in pools
                if pool.quota_units_per_cell is not None
                for cell in self.pool_cells(pool)
                if cell.state is not CellState.DELETED
            )
            generations = {
                pool.accelerator_type.partition("-")[0] for pool in pools if pool.accelerator_type is not None
            }
            generation = ", ".join(sorted(generations)) or "TPU"
            locations = {pool.zone or pool.region for pool in pools}
            summaries.append(
                SharedQuotaSummary(
                    generation=generation,
                    location=", ".join(sorted(locations)),
                    capacity_class=first.capacity_class,
                    used=used,
                    limit=first.quota_limit,
                )
            )
        return tuple(sorted(summaries, key=lambda quota: (quota.location, quota.generation)))


@dataclass(frozen=True, slots=True)
class FleetStatusRenderer:
    """Render a compact fleet summary without exposing platform internals."""

    console: Console

    def emit(self, status: FleetStatusResponse) -> None:
        """Write one human-oriented fleet status report."""

        view = FleetStatusView.build(status)
        self._summary(view)
        self._pools(view)
        self._quotas(view)
        self._attention_cells(view)

    def _summary(self, view: FleetStatusView) -> None:
        pools = view.status.pools
        active = view.active_cells
        ready_workers = sum(cell.worker_count for cell in active if cell.state is CellState.READY)
        active_workers = sum(cell.worker_count for cell in active)
        enabled_pools = sum(pool.enabled for pool in pools)
        locations = ", ".join(sorted({pool.zone or pool.region for pool in pools}))
        capacity_classes = ", ".join(sorted({_capacity_class(pool.capacity_class) for pool in pools}))
        state, state_style = view.fleet_state

        table = Table(show_header=False, box=None, pad_edge=False, expand=True)
        table.add_column("Field", style="bold cyan", width=13)
        table.add_column("Value", style="white", overflow="fold")
        table.add_row("Fleet", Text(view.status.name))
        table.add_row("Location", Text(locations or "-"))
        table.add_row("Capacity", Text(capacity_classes or "-"))
        table.add_row("Pools", Text(f"{enabled_pools} enabled / {len(pools)} total"))
        table.add_row("Workers", Text(f"{ready_workers} ready / {active_workers} active"))
        table.add_row("State", Text(state, style=state_style))
        print_block(self.console, Panel(table, title="[bold cyan]Eyvan fleet[/]", border_style="cyan"))

    def _pools(self, view: FleetStatusView) -> None:
        table = Table(
            title="[bold cyan]Worker pools[/]",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            pad_edge=False,
            expand=True,
        )
        table.add_column("Type", style="bold white", no_wrap=True)
        table.add_column("Resources / worker", overflow="fold")
        table.add_column("Autoscale", no_wrap=True)
        table.add_column("Workers", no_wrap=True)
        table.add_column("State", no_wrap=True)
        for pool in sorted(view.status.pools, key=_pool_sort_key):
            cells = PoolCellSummary.from_cells(view.pool_cells(pool))
            target = view.pool_target(pool)
            maximum = str(pool.max_cells)
            state, state_style = cells.state(target) if pool.enabled else ("disabled", "dim")
            table.add_row(
                Text(pool.node_type),
                _resources(pool.capacity),
                Text(f"{pool.min_cells} .. {maximum}"),
                cells.label(target),
                Text(state, style=state_style),
            )
        print_block(self.console, table)

    def _quotas(self, view: FleetStatusView) -> None:
        quotas = view.shared_quotas
        if not quotas:
            return
        table = Table(show_header=False, box=None, pad_edge=False, expand=True)
        table.add_column("Label", style="bold cyan", width=13)
        table.add_column("Value", overflow="fold")
        for index, quota in enumerate(quotas):
            table.add_row("Shared quota" if index == 0 else "", quota.label())
        print_block(self.console, table)

    def _attention_cells(self, view: FleetStatusView) -> None:
        attention = view.attention_cells
        if not attention:
            return
        pool_types = {pool.id: pool.node_type for pool in view.status.pools}
        table = Table(
            title="[bold yellow]Cells needing attention[/]",
            box=box.SIMPLE_HEAD,
            header_style="bold yellow",
            pad_edge=False,
            expand=True,
        )
        table.add_column("Cell", style="white", overflow="fold")
        table.add_column("Pool", no_wrap=True)
        table.add_column("Workers", justify="right")
        table.add_column("State", no_wrap=True)
        table.add_column("Reason", overflow="fold")
        for cell in attention[:_ATTENTION_CELL_LIMIT]:
            # The reason code is a closed vocabulary other paths compare; the
            # detail is the provider's own word for what happened. Both: the
            # code says what class of problem it is, and without the detail
            # the operator is back to reading agent storage by hand — which
            # is the log-diving this table exists to prevent.
            reason = cell.unavailable_reason or "-"
            if cell.unavailable_detail:
                detail = cell.unavailable_detail
                if len(detail) > 160:
                    detail = detail[:157] + "..."
                reason = f"{reason}: {detail}"
            table.add_row(
                Text(cell.id),
                Text(pool_types.get(cell.pool_id, "-") if cell.pool_id is not None else "-"),
                Text(str(cell.worker_count)),
                Text(cell.state.value, style=_cell_state_style(cell.state)),
                Text(reason),
            )
        print_block(self.console, table)
        remaining = len(attention) - _ATTENTION_CELL_LIMIT
        if remaining > 0:
            self.console.print(
                Text(f"{remaining} more cells omitted; rerun with --json for the full record.", style="dim")
            )


__all__ = (
    "FleetStatusRenderer",
    "FleetStatusView",
    "fleet_status_console",
    "is_interactive_output",
)
