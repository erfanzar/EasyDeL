# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Terminal presentation for durable Eyvan workload submissions and lifecycle updates."""

from __future__ import annotations

import sys
import typing as tp
from collections.abc import Mapping
from dataclasses import dataclass

from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from eyvan._internal.compat import StrEnum
from eyvan.sdk.terminal import print_block, terminal_console

#: States reached before the job produces a line of its own. Everything here is
#: waiting -- for admission, for a provider, for a host to answer -- so each is
#: a place a run can sit for many minutes with nothing to show, which is exactly
#: where a scrolling block per change reads as breakage rather than progress.
_WAITING_STATES = frozenset({"requesting", "queued", "provisioning", "allocated", "connecting", "ready", "dispatching"})


class JobLifecycleState(StrEnum):
    """User-facing lifecycle categories for one durable workload."""

    REQUESTING = "requesting"
    QUEUED = "queued"
    PROVISIONING = "provisioning"
    ALLOCATED = "allocated"
    CONNECTING = "connecting"
    READY = "ready"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    CANCELLING = "cancelling"
    PREEMPTED = "preempted"
    BLOCKED = "blocked"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    CANCELLED = "cancelled"


_STATE_LABELS: dict[JobLifecycleState, str] = {
    JobLifecycleState.REQUESTING: "REQUESTING",
    JobLifecycleState.QUEUED: "QUEUED",
    JobLifecycleState.PROVISIONING: "PROVISIONING",
    JobLifecycleState.ALLOCATED: "ALLOCATED",
    JobLifecycleState.CONNECTING: "CONNECTING",
    JobLifecycleState.READY: "READY",
    JobLifecycleState.DISPATCHING: "DISPATCHING",
    JobLifecycleState.RUNNING: "RUNNING",
    JobLifecycleState.CHECKPOINTING: "CHECKPOINTING",
    JobLifecycleState.CANCELLING: "CANCELLING",
    JobLifecycleState.PREEMPTED: "PREEMPTED",
    JobLifecycleState.BLOCKED: "BLOCKED",
    JobLifecycleState.FAILED: "FAILED",
    JobLifecycleState.SUCCEEDED: "SUCCEEDED",
    JobLifecycleState.CANCELLED: "CANCELLED",
}

_STATE_STYLES: dict[JobLifecycleState, str] = {
    JobLifecycleState.REQUESTING: "bold cyan",
    JobLifecycleState.QUEUED: "bold yellow",
    JobLifecycleState.PROVISIONING: "bold yellow",
    JobLifecycleState.ALLOCATED: "bold cyan",
    JobLifecycleState.CONNECTING: "bold cyan",
    JobLifecycleState.READY: "bold cyan",
    JobLifecycleState.DISPATCHING: "bold cyan",
    JobLifecycleState.RUNNING: "bold green",
    JobLifecycleState.CHECKPOINTING: "bold yellow",
    JobLifecycleState.CANCELLING: "bold yellow",
    JobLifecycleState.PREEMPTED: "bold yellow",
    JobLifecycleState.BLOCKED: "bold red",
    JobLifecycleState.FAILED: "bold red",
    JobLifecycleState.SUCCEEDED: "bold green",
    JobLifecycleState.CANCELLED: "bold yellow",
}


@dataclass(frozen=True, slots=True)
class JobSubmissionReceipt:
    """Durable job identity and copy-pasteable controls shown after submission."""

    workload_id: str
    fleet: str | None = None
    resource_shape: str | None = None
    atomic: bool = False

    @property
    def plain_lines(self) -> tuple[str, ...]:
        """Return an ANSI-free receipt suitable for redirected stderr."""

        destination = f" to fleet {self.fleet}" if self.fleet else ""
        resource = (f"  request  {self.resource_shape}",) if self.resource_shape else ()
        allocation = ("  allocation  atomic",) if self.atomic else ("  allocation  elastic",)
        return (
            f"Eyvan: job {self.workload_id} submitted{destination}.",
            *resource,
            *allocation,
            "  this job keeps running if this client exits.",
            f"  status  eyvan workload status {self.workload_id}",
            f"  logs    eyvan workload logs {self.workload_id}",
            f"  cancel  eyvan workload cancel {self.workload_id}",
        )


@dataclass(frozen=True, slots=True)
class JobLifecycleUpdate:
    """One deduplicated state transition rendered while a caller waits for a job."""

    state: JobLifecycleState
    scope: str | None
    summary: str | None = None
    diagnostic: str | None = None

    @property
    def detail_lines(self) -> tuple[str, ...]:
        """Return just the indented progress lines under a state heading."""

        lines = []
        if self.summary:
            lines.append(f"  {self.summary}")
        if self.diagnostic:
            lines.append(f"  {self.diagnostic}")
        return tuple(lines)

    @property
    def plain_lines(self) -> tuple[str, ...]:
        """Return a compact ANSI-free lifecycle update."""

        headline = f"Eyvan: {_STATE_LABELS[self.state].lower()}"
        if self.scope:
            headline = f"{headline} | {self.scope}"
        lines = [headline]
        if self.summary:
            lines.append(f"  {self.summary}")
        if self.diagnostic:
            lines.append(f"  {self.diagnostic}")
        return tuple(lines)


class JobTerminalOutput:
    """Render compact Rich job output on a terminal and plain text when redirected."""

    def __init__(
        self,
        *,
        writer: tp.Callable[[str], None] | None = None,
        stream: tp.TextIO | None = None,
        force_terminal: bool | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if writer is not None and stream is not None:
            raise ValueError("JobTerminalOutput accepts writer or stream, not both")
        self._writer = writer
        #: The in-place waiting line, live only while the job has yet to say
        #: anything of its own. ``None`` whenever nothing is being repainted.
        self._live: Live | None = None
        self._console = (
            None
            if writer is not None
            else terminal_console(
                sys.stderr if stream is None else stream,
                force_terminal=force_terminal,
                environment=environment,
            )
        )

    def submission(self, receipt: JobSubmissionReceipt) -> None:
        """Render the one-time submission receipt."""

        writer = self._writer
        if writer is not None:
            self._write_plain(writer, receipt.plain_lines)
            return
        console = self._console
        assert console is not None
        if not console.is_terminal:
            self._write_plain(console.print, receipt.plain_lines)
            return

        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        table.add_row("State", Text("Submitted", style="bold green"))
        table.add_row("Job", Text(receipt.workload_id, style="bold cyan"))
        if receipt.fleet:
            table.add_row("Fleet", Text(receipt.fleet, style="bold white"))
        if receipt.resource_shape:
            table.add_row("Request", Text(receipt.resource_shape, style="bold white"))
        if receipt.atomic:
            table.add_row("Allocation", Text("atomic", style="bold white"))
        else:
            table.add_row("Allocation", Text("elastic (packed, one world per cell)", style="bold white"))
        table.add_row("", Text(""))
        table.add_row("Check", Text(f"eyvan workload status {receipt.workload_id}", style="cyan"))
        table.add_row("Logs", Text(f"eyvan workload logs {receipt.workload_id}", style="cyan"))
        table.add_row("Cancel", Text(f"eyvan workload cancel {receipt.workload_id}", style="cyan"))
        print_block(
            console,
            Panel(
                table,
                title="[bold cyan]Eyvan job submitted[/]",
                title_align="left",
                border_style="cyan",
                padding=(0, 1),
                expand=False,
            ),
        )

    def _render_waiting(self, update: JobLifecycleUpdate) -> None:
        """Draw or redraw the single line that stands in for a waiting job."""

        console = self._console
        assert console is not None
        line = Text.assemble(
            (_STATE_LABELS[update.state], _STATE_STYLES[update.state]),
            ("  ", ""),
            (update.scope or "", "dim"),
        )
        if update.summary:
            line.append("  ")
            line.append(update.summary, style="white")
        spinner = Spinner("dots", text=line, style="cyan")
        if self._live is None:
            # transient=False so the last thing it drew survives the stop: what
            # the job was waiting on is part of the story of the run.
            self._live = Live(spinner, console=console, refresh_per_second=8, transient=False)
            self._live.start()
        else:
            self._live.update(spinner)

    def settle(self) -> None:
        """Stop the live waiting line, leaving its last frame on screen.

        Called before anything else prints, so a log line or a state change is
        never interleaved into the region the live display is repainting.
        """

        live, self._live = self._live, None
        if live is not None:
            live.stop()

    def lifecycle(self, update: JobLifecycleUpdate) -> None:
        """Render one changed lifecycle observation without repeating the job receipt.

        While the job is still waiting -- queued, provisioning, connecting -- the
        line is drawn *in place* rather than appended. Waiting for a spot TPU is
        minutes of state changes with no output of the job's own, and printing a
        fresh block for each one buries the run in its own status. Once the job
        reaches a state that produces output, or a terminal one, the live line is
        settled into permanent scrollback and everything after it prints
        normally, so the transcript still records how the run got there.
        """

        writer = self._writer
        if writer is not None:
            self._write_plain(writer, update.plain_lines)
            return
        console = self._console
        assert console is not None
        if not console.is_terminal:
            self._write_plain(console.print, update.plain_lines)
            return
        if str(update.state) in _WAITING_STATES:
            self._render_waiting(update)
            return
        # Anything else is the job itself starting, stopping or going wrong;
        # settle the live line so this lands above it and stays.
        self.settle()

        heading = Text("Eyvan", style="bold cyan")
        heading.append("  ")
        heading.append(_STATE_LABELS[update.state], style=_STATE_STYLES[update.state])
        if update.scope:
            heading.append("  ")
            heading.append(update.scope, style="dim")
        console.print(heading)
        if update.summary:
            console.print(Text(f"  {update.summary}", style="white"))
        if update.diagnostic:
            style = "bold red" if update.state in {JobLifecycleState.BLOCKED, JobLifecycleState.FAILED} else "yellow"
            console.print(Text(f"  {update.diagnostic}", style=style))

    def lifecycle_detail(self, update: JobLifecycleUpdate) -> None:
        """Render only the progress line when the state heading is unchanged.

        A worker count that moves while the state does not -- a cell draining
        in the moments before SUCCEEDED -- is new information, but reprinting
        the whole status block for it reads as the state changing twice.
        """

        if not update.detail_lines:
            return
        writer = self._writer
        if writer is not None:
            self._write_plain(writer, update.detail_lines)
            return
        console = self._console
        assert console is not None
        if not console.is_terminal:
            self._write_plain(console.print, update.detail_lines)
            return
        if str(update.state) in _WAITING_STATES:
            self._render_waiting(update)
            return
        self.settle()
        if update.summary:
            console.print(Text(f"  {update.summary}", style="white"))
        if update.diagnostic:
            style = "bold red" if update.state in {JobLifecycleState.BLOCKED, JobLifecycleState.FAILED} else "yellow"
            console.print(Text(f"  {update.diagnostic}", style=style))

    def note(self, message: str) -> None:
        """Render one aside about the run that is not a state change."""

        writer = self._writer
        if writer is not None:
            self._write_plain(writer, (f"Eyvan: {message}",))
            return
        console = self._console
        assert console is not None
        if not console.is_terminal:
            self._write_plain(console.print, (f"Eyvan: {message}",))
            return
        heading = Text("Eyvan", style="bold cyan")
        heading.append("  ")
        heading.append(message, style="yellow")
        console.print(heading)

    def output_line(self, text: str | Text) -> None:
        """Print one line of the job's own output into the transcript.

        The streamed counterpart of ``lifecycle``: while the client waits on a
        running job, its stdout/stderr lands here as it is produced, so the
        run reads like a local process rather than a status page.
        """

        writer = self._writer
        if writer is not None:
            self._write_plain(writer, (text if isinstance(text, str) else text.plain,))
            return
        console = self._console
        assert console is not None
        self.settle()
        if not console.is_terminal:
            self._write_plain(console.print, (text if isinstance(text, str) else text.plain,))
            return
        console.print(text if isinstance(text, Text) else Text(text))

    def detached(self, workload_id: str) -> None:
        """Say that this client stopped watching, not that the job stopped.

        The distinction is the whole message. A durable job survives its
        submitter, so a user who pressed Ctrl-C has to be told the run is still
        going and how to get back to it -- otherwise the safe assumption is that
        they just killed it, and the usual next action is to submit it again.
        """

        lines = (
            f"Eyvan: detached from {workload_id}. The job is still running.",
            f"  reattach  eyvan workload logs {workload_id} --follow",
            f"  cancel    eyvan workload cancel {workload_id}",
        )
        writer = self._writer
        if writer is not None:
            self._write_plain(writer, lines)
            return
        console = self._console
        assert console is not None
        if not console.is_terminal:
            self._write_plain(console.print, lines)
            return
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column()
        table.add_row("Job", Text(workload_id, style="bold cyan"))
        table.add_row("State", Text("still running", style="bold green"))
        table.add_row("Reattach", Text(f"eyvan workload logs {workload_id} --follow", style="cyan"))
        table.add_row("Cancel", Text(f"eyvan workload cancel {workload_id}", style="cyan"))
        print_block(
            console,
            Panel(
                table,
                title="[bold yellow]Detached[/]",
                title_align="left",
                border_style="yellow",
                padding=(0, 1),
                expand=False,
            ),
        )

    @staticmethod
    def _write_plain(writer: tp.Callable[[str], None], lines: tuple[str, ...]) -> None:
        """Write the exact plain fallback lines to a terminal-independent sink."""

        for line in lines:
            writer(line)


__all__ = (
    "JobLifecycleState",
    "JobLifecycleUpdate",
    "JobSubmissionReceipt",
    "JobTerminalOutput",
)
