# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The first screen: what already exists, before anybody creates anything.

Every other head command needs you to know which head you mean. That is fine on
the tenth day and useless on the first -- and equally useless the day you open a
project somebody else configured, where the only question is what is already
running.

The failure this has to avoid is specific and expensive: reporting "no heads
here" when the truth is "I could not look". They render identically, and only
one of them makes it safe to create another head. So most of what is asserted
here is that a discovery failure stays visible, and that a head which exists but
is stopped is still listed rather than hidden.
"""

from __future__ import annotations

import io
import subprocess
import typing as tp

import pytest

from eyvan.sdk.commands.head import HeadCommandError
from eyvan.sdk.commands.head_discovery import (
    DiscoveredHead,
    configured_project,
    discover_heads,
    discovery_action,
    parse_heads,
)
from eyvan.sdk.commands.start import (
    LOCAL_URL,
    discover,
    plan_targets,
    python_snippet,
    render_outcome,
    start_flow,
)
from eyvan.sdk.terminal import terminal_console

RUNNING = """[
  {"name": "eyvan-head-research", "status": "RUNNING",
   "zone": "https://x/zones/us-east5-a",
   "machineType": "https://x/machineTypes/e2-standard-4",
   "labels": {"eyvan-managed": "control-plane-head", "eyvan-head-fleet": "research"},
   "networkInterfaces": [{"networkIP": "10.0.0.5"}]}
]"""


def _runner(stdout: str = "[]", *, returncode: int = 0, stderr: str = ""):
    def run(args: tp.Sequence[str], *, stdin_text: str | None, timeout: float):
        del stdin_text, timeout
        return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr=stderr)

    return run


# --------------------------------------------------------------------------- what discovery asks for


def test_discovery_filters_on_the_label_that_grants_ownership() -> None:
    """The same label `head up` refuses to adopt an instance without.

    A name-prefix search would find instances Eyvan would then refuse to manage,
    which is a worse answer than finding nothing: it offers you a head you
    cannot use.
    """

    args = discovery_action("my-project").args

    assert "labels.eyvan-managed=control-plane-head" in args
    assert "--project" in args and "my-project" in args
    # Server-side. A project can hold thousands of instances, and sieving them
    # in the client is what times out on exactly the fleets big enough to care.
    assert args[args.index("--filter") - 1] != "--format"


def test_an_instance_without_a_fleet_label_is_skipped_rather_than_guessed() -> None:
    """The fleet name is what every following command is given.

    Inventing one from the instance name would point `head down` at a resource
    under a name nobody wrote.
    """

    document = """[
      {"name": "half-labelled", "status": "RUNNING", "zone": "z",
       "labels": {"eyvan-managed": "control-plane-head"}}
    ]"""

    assert parse_heads(document) == []


def test_a_head_reports_the_url_it_serves() -> None:
    heads = parse_heads(RUNNING)

    assert heads[0].fleet == "research"
    assert heads[0].zone == "us-east5-a"
    assert heads[0].machine_type == "e2-standard-4"
    assert heads[0].url == "http://10.0.0.5:8080"


def test_a_head_with_no_address_has_no_url_rather_than_a_broken_one() -> None:
    """A caller given a string would try it and report a network problem.

    The cause is a stopped instance, and that is what it should read as.
    """

    document = """[
      {"name": "stopped", "status": "TERMINATED", "zone": "https://x/zones/us-east5-a",
       "labels": {"eyvan-managed": "control-plane-head", "eyvan-head-fleet": "research"}}
    ]"""

    assert parse_heads(document)[0].url is None


def test_a_failed_listing_is_an_error_rather_than_an_empty_project() -> None:
    """The expensive confusion this whole module has to avoid.

    "There are no heads here" is the answer that makes somebody create a second
    one beside the head they already have.
    """

    with pytest.raises(HeadCommandError, match="could not list instances"):
        discover_heads("my-project", runner=_runner("", returncode=1, stderr="PERMISSION_DENIED"))


def test_a_machine_without_gcloud_is_told_so() -> None:
    def missing(args, *, stdin_text, timeout):
        del args, stdin_text, timeout
        raise FileNotFoundError

    with pytest.raises(HeadCommandError, match="gcloud is not installed"):
        discover_heads("my-project", runner=missing)


def test_an_unconfigured_project_is_absent_rather_than_a_guess() -> None:
    assert configured_project(_runner("(unset)\n")) is None
    assert configured_project(_runner("", returncode=1)) is None
    assert configured_project(_runner("my-project\n")) == "my-project"


# --------------------------------------------------------------------------- what the screen offers


def _head(fleet: str, status: str = "RUNNING", address: str | None = "10.0.0.5") -> DiscoveredHead:
    return DiscoveredHead(
        fleet=fleet,
        instance=f"eyvan-head-{fleet}",
        zone="us-east5-a",
        status=status,
        address=address,
    )


def test_what_already_runs_is_offered_before_what_would_be_created() -> None:
    """Nobody should reach "create one" before seeing the one they have."""

    targets = plan_targets([_head("research")], local_reachable=True)

    assert [target.kind for target in targets] == ["head", "local", "new-head"]


def test_a_stopped_head_is_listed_rather_than_hidden() -> None:
    """The useful move is starting the head that exists, not making another."""

    targets = plan_targets([_head("old", status="TERMINATED", address=None)], local_reachable=False)

    stopped = next(target for target in targets if target.kind == "head")
    assert not stopped.reachable
    assert "start it before connecting" in stopped.detail


def test_making_this_machine_a_head_is_offered_only_when_one_is_not_running() -> None:
    """Offering it beside a plane already answering invites two on one port."""

    assert any(target.kind == "new-local" for target in plan_targets([], local_reachable=False))
    assert not any(target.kind == "new-local" for target in plan_targets([], local_reachable=True))


def test_a_discovery_failure_is_printed_rather_than_folded_into_an_empty_list() -> None:
    """On screen the two look identical, and only one is safe to act on."""

    outcome = discover("my-project", runner=_runner("", returncode=1, stderr="denied"), probe=lambda _url: False)
    stream = io.StringIO()
    render_outcome(terminal_console(stream), outcome)
    rendered = stream.getvalue()

    assert outcome.discovery_error is not None
    assert "Could not list heads" in rendered
    assert "denied" in rendered


def test_a_project_that_is_not_configured_says_how_to_configure_it() -> None:
    outcome = discover(None, runner=_runner("(unset)\n"), probe=lambda _url: False)
    stream = io.StringIO()
    render_outcome(terminal_console(stream), outcome)

    assert "gcloud config set project" in stream.getvalue()


def test_nothing_is_created_by_looking(tmp_path) -> None:
    """The whole flow is reads. A first-run tool that provisions while somebody
    is still working out the vocabulary is the one thing this must not be."""

    issued: list[tuple[str, ...]] = []

    def record(args, *, stdin_text, timeout):
        del stdin_text, timeout
        issued.append(tuple(args))
        return subprocess.CompletedProcess(list(args), 0, stdout=RUNNING, stderr="")

    start_flow(
        terminal_console(io.StringIO()),
        project="my-project",
        runner=record,
        probe=lambda _url: False,
        choose=lambda _count: 1,
    )

    assert issued, "discovery should have run"
    for args in issued:
        assert "create" not in args
        assert "delete" not in args
        assert args[:3] == ("compute", "instances", "list")


def test_connecting_to_a_head_ends_with_runnable_python() -> None:
    """The gap between "you can submit from Python" and a file you can run is
    where a first session usually ends, and it is four lines wide."""

    stream = io.StringIO()
    start_flow(
        terminal_console(stream),
        project="my-project",
        runner=_runner(RUNNING),
        probe=lambda _url: False,
        choose=lambda _count: 1,
    )
    rendered = stream.getvalue()

    assert "http://10.0.0.5:8080" in rendered
    assert "@execute" in rendered


def test_the_snippet_points_at_the_plane_that_was_chosen() -> None:
    snippet = python_snippet("http://10.0.0.5:8080", "research")

    assert 'eyvan.init("http://10.0.0.5:8080", fleet="research")' in snippet
    assert "os.environ" not in snippet, "setting the environment from the process that reads it is a detour"


def test_the_local_snippet_pins_the_selected_plane_over_saved_defaults() -> None:
    """A saved context or environment can name a different plane, even locally."""

    snippet = python_snippet(LOCAL_URL, None)

    assert f'eyvan.init("{LOCAL_URL}")' in snippet
    assert "eyvan.init()" not in snippet
