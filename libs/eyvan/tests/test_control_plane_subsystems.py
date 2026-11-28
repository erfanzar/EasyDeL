# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The public service stays reachable while implementation modules stay bounded."""

from configparser import ConfigParser
from pathlib import Path

from eyvan.server.service import ControlPlaneService

ROOT = Path(__file__).resolve().parents[1] / "eyvan" / "server" / "service"


def test_control_plane_implementation_modules_are_small_enough_to_review() -> None:
    modules = [ROOT / "control_plane.py", *sorted((ROOT / "_control_plane").glob("*.py"))]
    assert len(modules) > 1, "subsystem implementations must not remain in the facade"
    assert all(path.stat().st_size < 50000 for path in modules), {
        path.name: path.stat().st_size for path in modules if path.stat().st_size >= 50000
    }


def test_subsystems_do_not_depend_on_method_resolution_order_to_override_each_other() -> None:
    owners = {}
    for cls in ControlPlaneService.__mro__:
        if cls is object:
            continue
        for name, value in vars(cls).items():
            if callable(value) or isinstance(value, (staticmethod, classmethod)):
                assert name not in owners, f"{name} exists on both {owners.get(name)} and {cls.__name__}"
                owners[name] = cls.__name__
    assert {"submit_workload", "create_lease", "apply_fleet", "cancel_workload", "_audit"} <= owners.keys()


def test_every_subsystem_is_covered_by_the_dependency_contract() -> None:
    config = ConfigParser()
    config.read(ROOT.parents[2] / ".importlinter")
    configured = set(config["importlinter:contract:control-plane-subsystems"]["modules"].split())
    actual = {
        f"eyvan.server.service._control_plane.{path.stem}"
        for path in (ROOT / "_control_plane").glob("*.py")
        if path.stem not in {"__init__", "shared"}
    }
    assert configured == actual


def test_the_architecture_tour_names_modules_that_exist() -> None:
    import re

    root = ROOT.parents[2]
    document = root / "docs/reference/control-plane-architecture.md"
    assert document.is_file()
    paths = re.findall(r"`(eyvan/[A-Za-z0-9_/]+\.py)`", document.read_text())
    assert len(paths) >= 6, "the tour must name concrete implementation entry points"
    for path in paths:
        assert (root / path).is_file(), f"architecture tour names a missing module: {path}"
