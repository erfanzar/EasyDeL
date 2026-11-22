# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Batch wrapper around ``scripts/convert_hf_to_easydel.py``.

Reads a list of HF model sources (from ``--source`` flags and/or a
``--models-file``), de-duplicates them, then spawns one ``convert_hf_to_easydel``
subprocess per model with the per-model ``--source``, ``--out``, and
``--repo-id`` filled in. Any unrecognized arguments are forwarded verbatim to
the per-model command, so the same conversion flags accepted by the single
converter (``--convert-mode``, ``--torch-streaming-cache``, ``--token``, ...)
apply to every model in the batch.

Side effects:
    - Spawns child ``python convert_hf_to_easydel.py`` processes.
    - Creates ``<--out-root>/<model-name>`` directories.
    - The child processes may download from HF and push to HF (see the
      single-converter docstring).

How to use

Batch wrapper around `scripts/convert_hf_to_easydel.py`.

Create a models file (one per line). Supported formats:
- `source`
- `source owner/name`
- `source -> owner/name`

Example `models.txt`:
  meta-llama/Llama-3.1-8B
  meta-llama/Llama-3.1-8B-Instruct -> EasyDeL/Llama-3.1-8B-Instruct

Run (all unknown flags are forwarded to `convert_hf_to_easydel.py`):

  python scripts/convert_hf_to_easydel_batch.py \\
    --models-file models.txt \\
    --out-root /mnt/gcs/easydel \\
    --convert-mode sequential \\
    --no-push-to-hub \\
    --torch-streaming-cache temp \\
    --torch-streaming-tmp-dir /tmp/hf-shards \\
    --token $HF_TOKEN
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from eformer.aparser import DataClassArgumentParser


@dataclass(frozen=True)
class ModelJob:
    """One model conversion job to dispatch as a subprocess.

    Attributes:
        source: HF source repo id or local path.
        repo_id: Output HF repo id (e.g. ``"EasyDeL/Llama-3.1-8B"``).
        out_dir: Local output directory under ``--out-root``.
    """

    source: str
    repo_id: str
    out_dir: Path


def _strip_comment(line: str) -> str:
    """Drop trailing ``#`` comments and surrounding whitespace from a line.

    Args:
        line: Raw text line, possibly containing a comment.

    Returns:
        str: Line with the first ``#`` and everything after it removed,
            trimmed of leading/trailing whitespace.
    """
    return line.split("#", 1)[0].strip()


def _parse_models_file(path: str | os.PathLike, *, default_owner: str, out_root: Path) -> list[ModelJob]:
    """Read a model-list file and turn each non-empty line into a :class:`ModelJob`.

    Args:
        path: Path to a UTF-8 text file with one model per line. See the
            module docstring for the accepted line formats.
        default_owner: HF owner/org to prefix when a line has no explicit
            ``owner/name``.
        out_root: Root directory under which each job's output folder lives.

    Returns:
        list[ModelJob]: One job per non-empty, non-comment line.
    """
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
    """Parse a single model-list line into ``(source, repo_id)``.

    Supported formats:

    * ``source`` — repo id defaults to ``"<default_owner>/<basename>"``.
    * ``source -> owner/name``
    * ``source,owner/name``
    * ``source owner/name``

    Args:
        line: A pre-stripped line from the models file.
        default_owner: HF owner used when no explicit repo id is provided.

    Returns:
        tuple[str, str]: ``(source, repo_id)``.

    Raises:
        ValueError: If the line cannot be parsed into a source / repo pair.
    """
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
    """Return a copy of ``argv`` with values after token-like flags hidden.

    Useful for safely printing the per-model command before exec'ing it.

    Args:
        argv: Argument list to sanitize.

    Returns:
        list[str]: New list where values following ``--token``, ``--hf-token``,
            or ``--huggingface-token`` are replaced with ``"****"``.
    """
    masked = argv[:]
    for i, arg in enumerate(masked):
        if arg in {"--token", "--hf-token", "--huggingface-token"} and i + 1 < len(masked):
            masked[i + 1] = "****"
    return masked


def _format_cmd(argv: list[str]) -> str:
    """Format an ``argv`` list as a copy-pasteable shell command string.

    Args:
        argv: Argument list, typically already masked by :func:`_mask_secrets`.

    Returns:
        str: Space-joined shell-quoted command.
    """
    return " ".join(shlex.quote(x) for x in argv)


def _default_convert_script() -> Path:
    """Resolve the default path to the per-model converter script.

    Returns:
        Path: ``<scripts/>/convert_hf_to_easydel.py`` next to this file.
    """
    return Path(__file__).resolve().parent / "convert_hf_to_easydel.py"


@dataclass
class BatchArgs:
    """CLI arguments for :func:`main`.

    Attributes:
        out_root: Output root directory; each model writes to
            ``<out_root>/<repo-name>``.
        source: List of HF source ids (repeatable ``--source`` flag).
        models_file: Optional path to a file with one model per line.
        repo_owner: Default output HF owner when a line in ``models_file`` does
            not specify one.
        python: Override the Python interpreter used for child processes.
        convert_script: Override the converter script path.
        dry_run: Print commands without executing.
        continue_on_error: Continue processing remaining models when one
            subprocess fails.
        skip_existing: Skip models whose output dir already exists and is
            non-empty.
    """

    out_root: str = field(metadata={"help": "Output root directory; each model writes to <out-root>/<repo-name>."})
    source: str = field(
        default_factory=list,
        metadata={"action": "append", "help": "HF source model id/path (repeatable)."},
    )
    models_file: str | None = field(
        default=None,
        metadata={
            "help": "File with one model per line (supports: 'source', 'source owner/name', 'source -> owner/name')."
        },
    )
    repo_owner: str = field(
        default="EasyDeL",
        metadata={"help": "Default output HF owner/org when a repo id isn't specified in models-file."},
    )
    python: str | None = field(
        default=None,
        metadata={"help": "Python interpreter to run convert script (default: current interpreter)."},
    )
    convert_script: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional path to a convert script. If omitted, runs the repo-local `easydel/scripts/convert_hf_to_easydel.py`."
            )
        },
    )
    dry_run: bool = field(default=False, metadata={"help": "Print commands without executing."})
    continue_on_error: bool = field(
        default=False,
        metadata={"help": "Continue converting remaining models even if one fails."},
    )
    skip_existing: bool = field(
        default=False,
        metadata={"help": "Skip models whose output directory already exists and is non-empty."},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the batch converter wrapper.

    Builds the deduplicated list of :class:`ModelJob` entries from
    ``--source`` flags and ``--models-file``, then either prints (when
    ``--dry-run``) or runs the per-model ``convert_hf_to_easydel.py``
    subprocess for each one. Forwarded ``pass_through`` arguments are
    appended verbatim to every child command.

    Args:
        argv: Optional list of command-line tokens. When ``None`` the parser
            reads from ``sys.argv``.

    Returns:
        int: ``0`` if every conversion succeeded (or all jobs were skipped),
            the failing subprocess's return code when one fails and
            ``--continue-on-error`` is not set, or ``2`` if some failed but
            execution continued.

    Raises:
        SystemExit: If no models were selected.
    """
    parser = DataClassArgumentParser(
        BatchArgs,
        description="Batch wrapper around scripts/convert_hf_to_easydel.py",
    )
    args, pass_through = parser.parse_args_into_dataclasses(
        args=argv,
        return_remaining_strings=True,
        look_for_args_file=False,
    )

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
