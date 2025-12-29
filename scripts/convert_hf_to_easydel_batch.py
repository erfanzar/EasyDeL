# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelJob:
    source: str
    repo_id: str
    out_dir: Path


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_models_file(path: str | os.PathLike, *, default_owner: str, out_root: Path) -> list[ModelJob]:
    jobs: list[ModelJob] = []
    text = Path(path).read_text(encoding="utf-8")

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line:
            continue

        source, repo_id = _parse_model_line(line, default_owner=default_owner)
        name = repo_id.split("/", 1)[-1]
        jobs.append(ModelJob(source=source, repo_id=repo_id, out_dir=out_root / name))

    return jobs


def _parse_model_line(line: str, *, default_owner: str) -> tuple[str, str]:
    # Supported formats:
    # - source
    # - source -> owner/name
    # - source,owner/name
    # - source owner/name
    if "->" in line:
        left, right = (part.strip() for part in line.split("->", 1))
        source = left
        repo_id = right
        if not source or not repo_id:
            raise ValueError(f"Invalid mapping line: {line!r}")
        return source, repo_id

    if "," in line:
        left, right = (part.strip() for part in line.split(",", 1))
        source = left
        repo_id = right
        if not source or not repo_id:
            raise ValueError(f"Invalid CSV mapping line: {line!r}")
        return source, repo_id

    parts = line.split()
    if len(parts) == 2:
        return parts[0], parts[1]

    if len(parts) == 1:
        source = parts[0]
        name = source.split("/", 1)[-1]
        return source, f"{default_owner}/{name}"

    raise ValueError(f"Could not parse line: {line!r}")


def _mask_secrets(argv: list[str]) -> list[str]:
    masked = argv[:]
    for i, arg in enumerate(masked):
        if arg in {"--token", "--hf-token", "--huggingface-token"} and i + 1 < len(masked):
            masked[i + 1] = "****"
    return masked


def _format_cmd(argv: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in argv)


def _default_convert_script() -> Path:
    return Path(__file__).resolve().parent / "convert_hf_to_easydel.py"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch wrapper around scripts/convert_hf_to_easydel.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="HF source model id/path (repeatable).",
    )
    parser.add_argument(
        "--models-file",
        help="File with one model per line (supports: 'source', 'source owner/name', 'source -> owner/name').",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output root directory; each model writes to <out-root>/<repo-name>.",
    )
    parser.add_argument(
        "--repo-owner",
        default="EasyDeL",
        help="Default output HF owner/org when a repo id isn't specified in models-file.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter to run convert script (default: current interpreter).",
    )
    parser.add_argument(
        "--convert-script",
        default=None,
        help="Path to scripts/convert_hf_to_easydel.py (default: repo-local).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue converting remaining models even if one fails.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose output directory already exists and is non-empty.",
    )

    args, pass_through = parser.parse_known_args(argv)

    out_root = Path(args.out_root).expanduser().resolve()
    convert_script = (
        Path(args.convert_script).expanduser().resolve() if args.convert_script else _default_convert_script()
    )
    python_exe = args.python or sys.executable

    jobs: list[ModelJob] = []

    if args.models_file:
        jobs.extend(_parse_models_file(args.models_file, default_owner=args.repo_owner, out_root=out_root))

    for source in args.source:
        source = source.strip()
        if not source:
            continue
        name = source.split("/", 1)[-1]
        repo_id = f"{args.repo_owner}/{name}"
        jobs.append(ModelJob(source=source, repo_id=repo_id, out_dir=out_root / name))

    # De-dup while keeping order
    seen: set[tuple[str, str, Path]] = set()
    unique: list[ModelJob] = []
    for job in jobs:
        key = (job.source, job.repo_id, job.out_dir)
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    jobs = unique

    if not jobs:
        raise SystemExit("No models selected. Use --source and/or --models-file.")

    out_root.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    failed = 0

    for idx, job in enumerate(jobs, start=1):
        if args.skip_existing and job.out_dir.exists() and any(job.out_dir.iterdir()):
            print(f"[{idx}/{len(jobs)}] [skip] {job.source} -> {job.out_dir} (exists)")
            skipped += 1
            continue

        job.out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_exe,
            str(convert_script),
            "--source",
            job.source,
            "--out",
            str(job.out_dir),
            "--repo-id",
            job.repo_id,
            *pass_through,
        ]

        print(f"[{idx}/{len(jobs)}] {job.source} -> {job.repo_id}")
        print(_format_cmd(_mask_secrets(cmd)))

        if args.dry_run:
            ok += 1
            continue

        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            ok += 1
            continue

        failed += 1
        if not args.continue_on_error:
            return proc.returncode

    print(f"done: ok={ok} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
