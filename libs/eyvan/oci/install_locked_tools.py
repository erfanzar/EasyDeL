#!/usr/bin/env python3
# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download selected release-lock tools after verifying their exact SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def _download(url: str, destination: Path, *, expected_size: int) -> str:
    """Stream one locked asset without ever accepting more or fewer bytes."""
    digest = hashlib.sha256()
    received = 0
    request = urllib.request.Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "eyvan-oci-release/1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        if response.status != 200:
            raise RuntimeError(f"download returned HTTP {response.status}: {url}")
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            if not content_length.isdecimal() or int(content_length) != expected_size:
                raise RuntimeError(f"download Content-Length mismatch for {url}: {content_length!r} != {expected_size}")
        while chunk := response.read(min(1024 * 1024, expected_size - received + 1)):
            received += len(chunk)
            if received > expected_size:
                raise RuntimeError(f"download exceeded locked size for {url}: more than {expected_size} bytes")
            output.write(chunk)
            digest.update(chunk)
    if received != expected_size:
        raise RuntimeError(f"download size mismatch for {url}: {received} != {expected_size}")
    return digest.hexdigest()


def _extract_locked_file(archive: Path, archive_path: str, destination: Path) -> None:
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, mode="r:gz") as bundle:
            try:
                member = bundle.getmember(archive_path)
            except KeyError as error:
                raise RuntimeError(f"archive does not contain locked path {archive_path!r}") from error
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(f"locked archive member is not a regular file: {archive_path!r}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read locked archive member: {archive_path!r}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    else:
        if archive_path != destination.name:
            raise RuntimeError("a direct binary's archivePath must equal its tool name")
        shutil.copyfile(archive, destination)
    executable_mode = (
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )
    destination.chmod(executable_mode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("tools", nargs="+")
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    known = lock.get("tools")
    if not isinstance(known, dict):
        raise SystemExit("release lock has no tools object")
    unknown = sorted(set(args.tools) - set(known))
    if unknown:
        raise SystemExit(f"unknown locked tool(s): {unknown}")
    if len(set(args.tools)) != len(args.tools):
        raise SystemExit("tool names must not be repeated")

    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eyvan-locked-tools-") as scratch_text:
        scratch = Path(scratch_text)
        for name in args.tools:
            spec = known[name]
            if set(spec) != {"archivePath", "sha256", "size", "url", "version"}:
                raise SystemExit(f"locked tool {name!r} has an unexpected schema")
            expected_size = spec["size"]
            if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
                raise SystemExit(f"locked tool {name!r} has an invalid exact byte size")
            download_name = Path(spec["url"].split("?", 1)[0]).name
            archive = scratch / download_name
            actual = _download(spec["url"], archive, expected_size=expected_size)
            if actual != spec["sha256"]:
                raise SystemExit(f"{name} download hash mismatch: {actual} != {spec['sha256']}")
            _extract_locked_file(archive, spec["archivePath"], args.destination / name)
            print(f"installed {name} {spec['version']} ({actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
