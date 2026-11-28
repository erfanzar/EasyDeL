# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Automatic development image publication must not retarget release locators."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _publication(role: str) -> tuple[dict, dict, dict, str]:
    workflow = yaml.safe_load((_ROOT / ".github/workflows/image.yaml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["build"]
    matrix = next(entry for entry in job["strategy"]["matrix"]["include"] if entry["role"] == role)
    push = next(step["with"] for step in job["steps"] if step.get("uses", "").startswith("docker/build-push-action@"))
    lock = json.loads((_ROOT / "oci/release-lock.json").read_text(encoding="utf-8"))
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return matrix, push, lock, project["project"]["version"]


def _destinations(matrix: dict, push: dict, lock: dict, version: str) -> set[str]:
    owner = lock["sourceRepository"].removeprefix("https://github.com/").split("/")[0]
    substitutions = {
        "github.repository_owner": owner,
        "matrix.image": matrix["image"],
        "steps.locked.outputs.version": version,
        "github.sha": _SHA,
    }
    tags = push["tags"]
    for expression, value in substitutions.items():
        tags = tags.replace("${{ " + expression + " }}", value)
    assert "${{" not in tags, "unexpanded publication destination"
    return set(tags.split())


@pytest.mark.parametrize("role", ["control-plane", "agent", "worker"])
def test_automatic_destinations_do_not_overlap_gated_release_version_tags(role: str) -> None:
    matrix, push, lock, version = _publication(role)
    assert push["push"] is True
    release_tags = {f"{image['repository']}:{version}" for image in lock["images"].values()}
    # Check the actual promotion lane as well as the release lock's repositories.
    release = (_ROOT / ".github/workflows/eyvan-oci-release.yaml").read_text(encoding="utf-8")
    promoted = set(re.findall(r'crane copy "[^"\n]+" "([^"\n]+)"', release))
    promoted_tags = {tag.replace("${{ inputs.version }}", version) for tag in promoted}
    assert release_tags <= promoted_tags
    assert _destinations(matrix, push, lock, version).isdisjoint(release_tags)


@pytest.mark.parametrize("role", ["control-plane", "agent", "worker"])
def test_automatic_images_keep_only_source_sha_tags_and_version_metadata(role: str) -> None:
    matrix, push, lock, version = _publication(role)
    owner = lock["sourceRepository"].removeprefix("https://github.com/").split("/")[0]
    assert _destinations(matrix, push, lock, version) == {f"ghcr.io/{owner}/{matrix['image']}:sha-{_SHA}"}
    assert "EYVAN_VERSION=${{ steps.locked.outputs.version }}" in push["build-args"]
    assert "EYVAN_REVISION=${{ github.sha }}" in push["build-args"]
    assert push["provenance"] is True
    assert push["sbom"] is True
