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
"""Batch wrapper around ``convert_hf_to_easydel.py``.

Reads a list of source models (from CLI ``--source`` flags and/or a
``--models-file``) and invokes the single-model conversion script as a
subprocess for each entry. Unknown flags are forwarded verbatim to the
per-model script, so any ``ConvertArgs`` flag is supported.

Models file formats (one per line, ``#`` comments allowed):
    - ``source``
    - ``source owner/name``
    - ``source,owner/name``
    - ``source -> owner/name``

Example ``models.txt``:
    meta-llama/Llama-3.1-8B
    meta-llama/Llama-3.1-8B-Instruct -> EasyDeL/Llama-3.1-8B-Instruct

Example invocation:

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
    """Immutable record describing one (source, target, output) triple to convert.

    Built by :func:`_parse_models_file` (one per non-comment line)
    and via the repeated ``--source`` CLI flag in :func:`main`.
    Frozen so the per-job dedupe set in :func:`main` can hash
    instances directly. Each job becomes one subprocess invocation
    of ``convert_hf_to_easydel.py``.

    Attributes:
        source (str): HuggingFace source repo id (e.g.
            ``"meta-llama/Llama-3.1-8B"``) or local path passed
            verbatim to the per-model script's ``--source``.
        repo_id (str): Target Hub repo id (``owner/name``) used as
            ``--repo-id``. Auto-derived from ``source`` when the
            models file omits it (using
            :attr:`BatchArgs.repo_owner`).
        out_dir (Path): Filesystem directory under ``out_root``
            where the converted checkpoint is written; passed as
            ``--out`` to the per-model script.
    """

    source: str
    repo_id: str
    out_dir: Path


def _strip_comment(line: str) -> str:
    """Strip ``# ...`` trailing comments and surrounding whitespace from a single line.

    Used by :func:`_parse_models_file` to skip blank lines and
    full-line comments while still parsing inline comments
    correctly. The comment marker is the unescaped ``#``; there
    is no support for backslash-escaping a literal ``#`` inside
    a source name (no real-world repo id contains one anyway).

    Args:
        line: Raw line from the models file.

    Returns:
        str: Comment-free, whitespace-stripped line. Empty when
        the input is fully blank or comment-only.
    """
    return line.split("#", 1)[0].strip()


def _parse_models_file(path: str | os.PathLike, *, default_owner: str, out_root: Path) -> list[ModelJob]:
    """Read a models file and turn each non-empty line into a :class:`ModelJob`.

    Strips comments via :func:`_strip_comment`, skips blank
    lines, and delegates per-line parsing to
    :func:`_parse_model_line`. The ``out_dir`` for each job is
    computed by concatenating ``out_root`` with the destination
    repo's name (last path component of ``repo_id``), so jobs
    that share a target name share an output directory — this is
    intentional, allowing the same model to be re-converted in
    place across runs.

    Args:
        path: Filesystem path to the models file (UTF-8).
        default_owner: HF owner/org used by
            :func:`_parse_model_line` when a line gives only a
            source.
        out_root: Output root directory; combined with each
            job's repo name.

    Returns:
        list[ModelJob]: Ordered jobs, in the same order as the
        non-empty lines in the file.
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
    """Parse a single models-file line into ``(source, repo_id)``.

    Supported formats:
        - ``source``
        - ``source -> owner/name``
        - ``source,owner/name``
        - ``source owner/name``

    Args:
        line: Pre-stripped line (no trailing comment, non-empty).
        default_owner: HF owner used when only a source is provided.

    Returns:
        Tuple ``(source, repo_id)``.

    Raises:
        ValueError: If the line cannot be parsed into the expected fields.
    """
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
    """Hide HF tokens in argv before logging or printing the planned command line.

    Looks for the recognised token flag forms (``--token``,
    ``--hf-token``, ``--huggingface-token``) and replaces the
    *next* argument (the actual token value) with ``"****"``.
    Returns a fresh list so the caller's argv stays unmodified.

    Args:
        argv: Subprocess command-line arguments to mask.

    Returns:
        list[str]: Copy of ``argv`` with any token values
        replaced by ``"****"``.
    """
    masked = argv[:]
    for i, arg in enumerate(masked):
        if arg in {"--token", "--hf-token", "--huggingface-token"} and i + 1 < len(masked):
            masked[i + 1] = "****"
    return masked


def _format_cmd(argv: list[str]) -> str:
    """Render an argv list as a single shell-quoted command string for printing.

    Used purely for human-readable logging — every argument is
    quoted by :func:`shlex.quote` so the printed line is
    copy-paste safe even when arguments contain spaces or
    special characters.

    Args:
        argv: Subprocess command-line arguments.

    Returns:
        str: Single shell-quoted command line.
    """
    return " ".join(shlex.quote(x) for x in argv)


def _default_convert_script() -> Path:
    """Locate the bundled per-model converter that lives next to this script.

    Resolves ``convert_hf_to_easydel.py`` relative to the current
    file so the batch runner works regardless of where the user
    invokes it from. The :func:`main` function uses this when the
    user does not pass ``--convert-script``.

    Returns:
        Path: Absolute path to ``convert_hf_to_easydel.py`` in the
        same directory as this module.
    """
    return Path(__file__).resolve().parent / "convert_hf_to_easydel.py"


@dataclass
class BatchArgs:
    """CLI arguments specific to the batch converter (delegated flags pass through).

    Captures only the batch-runner concerns — list of source models,
    output root directory, owner default, subprocess interpreter,
    and the runner-level safety flags (dry-run, continue-on-error,
    skip-existing). Anything else on the command line is forwarded
    verbatim to ``convert_hf_to_easydel.py`` via the
    ``return_remaining_strings`` mechanism in :func:`main`.

    Attributes:
        out_root (str): Directory under which each model gets a
            ``<out-root>/<repo-name>/`` subdirectory.
        source (str): Repeatable ``--source`` flag; collected into
            a list by ``argparse``. Each entry becomes one
            :class:`ModelJob`.
        models_file (str | None): Optional path to a text file
            with one source per line; format described in the
            module docstring.
        repo_owner (str): HF owner/org used to build target
            ``repo_id``s when the models file/CLI does not
            specify one. Defaults to ``"EasyDeL"``.
        python (str | None): Path to the Python interpreter used
            to run the per-model script. ``None`` uses the
            current ``sys.executable``.
        convert_script (str | None): Override for the per-model
            script path. Defaults to the bundled
            ``convert_hf_to_easydel.py``.
        dry_run (bool): When ``True``, prints commands instead of
            executing them — useful for verifying the planned
            invocations.
        continue_on_error (bool): When ``True``, a failed
            subprocess does not abort the remaining jobs; the
            runner returns ``2`` at the end if any job failed.
        skip_existing (bool): When ``True``, skip jobs whose
            output directory already exists and is non-empty.
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
            "help": "Optional path to a convert script. If omitted, runs the repo-local `easydel/scripts/convert_hf_to_easydel.py`."
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
    """Run batch model conversion from HuggingFace to EasyDeL format.

    Reads model specifications from ``--source`` flags and/or a
    ``--models-file``, then invokes ``convert_hf_to_easydel.py`` for
    each model as a subprocess. Unknown flags are forwarded to the
    per-model conversion script.

    Args:
        argv: Command-line arguments. Uses ``sys.argv`` when ``None``.

    Returns:
        Exit code: 0 on success, 2 if any conversions failed, or the
        failing subprocess exit code if ``--continue-on-error`` is not set.
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
