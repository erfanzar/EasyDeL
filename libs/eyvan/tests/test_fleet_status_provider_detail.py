# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Operator-facing fleet output carries the provider's diagnostic safely."""

from io import StringIO

from rich.console import Console

from eyvan.records import CellRecord, CellState, ResourceLimits
from eyvan.sdk.commands.fleet_status import FleetStatusRenderer
from eyvan.server.api.schemas import FleetStatusResponse


def test_fleet_status_shows_provider_detail_as_literal_text() -> None:
    output = StringIO()
    cell = CellRecord(
        id="test-cell",
        provider="manual",
        region="test",
        state=CellState.UNAVAILABLE,
        capacity=ResourceLimits(cpu=1),
        allocatable=ResourceLimits(cpu=1),
        unavailable_reason="provider_error",
        unavailable_detail="[red]quota exceeded[/red]",
    )
    FleetStatusRenderer(Console(file=output, width=180, color_system=None)).emit(
        FleetStatusResponse(name="test", pools=[], cells=[cell])
    )
    text = output.getvalue()
    assert "provider_error" in text
    assert "[red]quota exceeded[/red]" in text
