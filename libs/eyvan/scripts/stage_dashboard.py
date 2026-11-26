# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Put the built dashboard inside the package, and check that a release carries it.

``uv build`` packs every file under ``eyvan/``, so copying ``dashboard/dist`` to
``eyvan/_dashboard`` before a build is all it takes for a wheel or sdist to serve
the UI after ``pip install eyvan``. The copy is a build output and is gitignored.

Usage::

    npm --prefix dashboard run build
    python3 scripts/stage_dashboard.py                  # dashboard/dist -> eyvan/_dashboard
    python3 scripts/stage_dashboard.py --check dist/*.whl dist/*.tar.gz

``--check`` exits non-zero when an artifact lacks ``index.html`` or any
``/assets/...`` file that ``index.html`` loads. Standard library only.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys
import tarfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dashboard" / "dist"
TARGET = ROOT / "eyvan" / "_dashboard"
PACKAGED = "eyvan/_dashboard"
_ASSET = re.compile(r"""(?:src|href)\s*=\s*["']/(assets/[^"'?#]+)["']""")


def stage(source: pathlib.Path = SOURCE, target: pathlib.Path = TARGET) -> int:
    """Replace ``target`` with a copy of ``source`` and return the number of files copied."""

    if not (source / "index.html").is_file():
        raise SystemExit(f"{source} has no index.html; run `npm --prefix dashboard run build` first")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return sum(1 for path in target.rglob("*") if path.is_file())


def _names(artifact: pathlib.Path) -> tuple[set[str], str | None]:
    """Return the package-relative member names and the text of the packaged index."""

    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            names = set(archive.namelist())
            index = f"{PACKAGED}/index.html"
            return names, archive.read(index).decode("utf-8") if index in names else None
    with tarfile.open(artifact, "r:gz") as archive:
        members = {member.name.split("/", 1)[1]: member for member in archive.getmembers() if "/" in member.name}
        index_member = members.get(f"{PACKAGED}/index.html")
        text = None
        if index_member is not None:
            handle = archive.extractfile(index_member)
            text = handle.read().decode("utf-8") if handle is not None else None
        return set(members), text


def missing(artifact: pathlib.Path) -> list[str]:
    """List the dashboard files ``artifact`` should carry and does not."""

    names, index = _names(artifact)
    if index is None:
        return [f"{PACKAGED}/index.html"]
    wanted = sorted({f"{PACKAGED}/{asset}" for asset in _ASSET.findall(index)})
    return [name for name in wanted if name not in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", nargs="+", type=pathlib.Path, metavar="ARTIFACT")
    args = parser.parse_args(argv)
    if not args.check:
        print(f"staged {stage()} dashboard files into {TARGET.relative_to(ROOT)}")
        return 0
    failed = False
    for artifact in args.check:
        absent = missing(artifact)
        if absent:
            failed = True
            print(f"{artifact}: missing {', '.join(absent)}", file=sys.stderr)
        else:
            print(f"{artifact}: carries the dashboard")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
