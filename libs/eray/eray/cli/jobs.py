# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Job-lifecycle commands for the eray CLI: run, status, logs, stop.

Design notes live in the workspace repo under ``.agents/projects/eray-cli.md``.
The guiding rule is truthfulness: Ray reports a job SUCCEEDED even when the
remote function raised and the driver printed the exception via
``print_remote_raise`` — ``eray status`` re-derives a verdict from the driver
log so that class of silent failure is visible in the table.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import click

from .utils import NC, RED, YELLOW, error, info, warning

DEFAULT_DASHBOARD = "http://127.0.0.1:8265"

# Host-machine state that must not leak into a job's runtime env.
ENV_DENY_EXACT = frozenset(
    {
        "PATH",
        "HOME",
        "PWD",
        "OLDPWD",
        "SHELL",
        "SHLVL",
        "TERM",
        "TERMINFO",
        "USER",
        "LOGNAME",
        "HOSTNAME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "DISPLAY",
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "LS_COLORS",
        "_",
    }
)
ENV_DENY_PREFIXES = (
    "SSH_",
    "XDG_",
    "LC_",
    "LD_",
    "CONDA_",
    "RAY_",
    "BASH_FUNC_",
    "CLAUDE_",
    "GPG_",
    "DBUS_",
)

_SECRET_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)

# Directories that Ray's packaging excludes by default (approximation used by
# the pre-upload size guard).
_PACKAGE_SKIP_DIRS = frozenset(
    {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", "node_modules"}
)
PACKAGE_WARN_BYTES = 500 * 1024**2
PACKAGE_ABORT_BYTES = 2 * 1024**3

# Ordered error signatures: first match wins the verdict suffix.
_ERROR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("Failed to merge the Job's runtime env", "env-conflict"),
    ("CompileTimeHbmOom", "oom-compile"),
    ("RESOURCE_EXHAUSTED", "oom"),
    ("abstract trainable parameter", "load-incomplete"),
    ("Traceback (most recent call last)", "remote-raise"),
)

# Ordered phase markers: the *last* one present in the log tail wins.
_PHASE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Uploading package", "packaging"),
    ("Loading:", "loading"),
    ("loaded state step", "loaded"),
    ("Compiling", "compiling"),
    ("time took for configure shard", "compiling"),
    ("Converting shard", "converting"),
)

_STEP_RE = re.compile(r"'train_step': (\d+)")
_KL_RE = re.compile(r"'kl_loss': ([0-9.]+)")
_LOSS_RE = re.compile(r"'loss': ([0-9.]+)")

_PROGRESS_SPAM_RE = re.compile(r"tensor/s|\.\.\.\s*\d+%|\d+/\d+ \[\d+:\d+<")


def resolve_address(explicit: str | None) -> str:
    """Resolve the Ray dashboard address for job-submission clients.

    Order: explicit flag, ``RAY_ADDRESS`` env var, then the local default
    dashboard. Bare ``host:port`` values are normalized to ``http://``.

    Args:
        explicit: Address passed on the command line, if any.

    Returns:
        An ``http(s)://host:port`` URL.
    """
    addr = explicit or os.environ.get("RAY_ADDRESS") or DEFAULT_DASHBOARD
    if addr.startswith(("http://", "https://")):
        return addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        if port == "6379":  # GCS port — jobs API lives on the dashboard
            return f"http://{host}:8265"
        return f"http://{addr}"
    return f"http://{addr}:8265"


def make_client(address: str | None):
    """Build a ``JobSubmissionClient`` for *address* (module-level for tests)."""
    from ray.job_submission import JobSubmissionClient

    return JobSubmissionClient(resolve_address(address))


def inherited_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Filter the current process env down to what a job should inherit.

    Args:
        environ: Environment mapping (defaults to ``os.environ``).

    Returns:
        Env vars safe to inject into the job's runtime env: host-machine
        state (paths, shell, session, ``RAY_*``) removed, everything else —
        including secrets like ``HF_TOKEN`` and ``PYTHONPATH`` — kept.
    """
    environ = dict(os.environ) if environ is None else environ
    out: dict[str, str] = {}
    for key, value in environ.items():
        if key in ENV_DENY_EXACT or key.startswith(ENV_DENY_PREFIXES):
            continue
        out[key] = value
    return out


def mask_value(key: str, value: str) -> str:
    """Mask *value* for terminal echo when *key* looks like a secret."""
    if _SECRET_KEY_RE.search(key) and len(value) > 8:
        return f"{value[:4]}…({len(value)} chars)"
    return value


def package_size_bytes(root: str | Path) -> int:
    """Approximate the bytes Ray would package for *root* as a working dir.

    Skips the directories Ray excludes by default (``.git``, ``.venv``,
    caches) and never follows symlinks.
    """
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in _PACKAGE_SKIP_DIRS]
        for name in filenames:
            with_suppress = os.path.join(dirpath, name)
            try:
                st = os.lstat(with_suppress)
            except OSError:
                continue
            total += st.st_size
    return total


def generate_submission_id(entrypoint: tuple[str, ...]) -> str:
    """Derive a human-meaningful submission id from the entrypoint.

    ``('python', 'launch.py')`` → ``launch-<user>-<YYYYmmdd-HHMMSS>``.
    """
    stem = "job"
    for token in reversed(entrypoint):
        base = os.path.basename(token)
        if base.endswith((".py", ".sh")):
            stem = base.rsplit(".", 1)[0]
            break
    stem = re.sub(r"[^A-Za-z0-9_-]", "-", stem) or "job"
    user = re.sub(r"[^A-Za-z0-9_-]", "-", getpass.getuser())
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stem}-{user}-{stamp}"


def git_metadata(cwd: str | Path = ".") -> dict[str, str]:
    """Best-effort git provenance (sha + dirty flag) for job metadata."""
    meta: dict[str, str] = {}
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            meta["git_sha"] = sha.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5)
            meta["git_dirty"] = "1" if dirty.stdout.strip() else "0"
    except Exception:
        pass
    return meta


def scan_log_tail(text: str) -> tuple[str | None, str | None]:
    """Extract an error signature and a phase description from a log tail.

    Args:
        text: The last chunk of a job's driver log.

    Returns:
        ``(error, phase)`` where ``error`` is a short signature name (or
        ``None`` when no failure marker is present) and ``phase`` is the most
        advanced progress marker seen (``step N (kl X)`` once metrics flow).
    """
    err = None
    for needle, name in _ERROR_SIGNATURES:
        if needle in text:
            err = name
            break

    phase = None
    best_pos = -1
    for needle, name in _PHASE_MARKERS:
        pos = text.rfind(needle)
        if pos > best_pos:
            best_pos = pos
            phase = name
    steps = _STEP_RE.findall(text)
    if steps:
        phase = f"step {steps[-1]}"
        kls = _KL_RE.findall(text)
        losses = _LOSS_RE.findall(text)
        if kls:
            phase += f" (kl {kls[-1]})"
        elif losses:
            phase += f" (loss {losses[-1]})"
    return err, phase


def verdict_for(status: str, error_sig: str | None) -> str:
    """Combine Ray's job status with the log-derived error signature.

    A SUCCEEDED job whose driver log carries a failure marker is reported as
    ``failed(<sig>)`` — Ray's status is not trustworthy for launchers that
    print remote exceptions and exit 0.
    """
    status = (status or "").upper()
    if status == "SUCCEEDED":
        return f"failed({error_sig})" if error_sig else "ok"
    if status == "RUNNING":
        return f"erroring({error_sig})" if error_sig else "-"
    if status == "STOPPED":
        return f"user-stop({error_sig})" if error_sig else "user-stop"
    if status == "FAILED":
        return f"failed({error_sig})" if error_sig else "failed"
    return status.lower() or "?"


def _driver_log_tail_from_fs(submission_id: str, max_bytes: int = 65536) -> str | None:
    """Fast path: read the driver log tail from the local Ray session dir."""
    try:
        from ..core.monitoring import _ray_session_log_dirs

        for logs_dir in _ray_session_log_dirs():
            path = os.path.join(logs_dir, f"job-driver-{submission_id}.log")
            if os.path.isfile(path):
                size = os.path.getsize(path)
                with open(path, "rb") as f:
                    if size > max_bytes:
                        f.seek(-max_bytes, os.SEEK_END)
                    return f.read().decode("utf-8", errors="replace")
    except Exception:
        pass
    return None


def get_log_tail(client, submission_id: str, max_bytes: int = 65536) -> str:
    """Driver-log tail for *submission_id*: local file fast path, HTTP fallback."""
    tail = _driver_log_tail_from_fs(submission_id, max_bytes)
    if tail is not None:
        return tail
    try:
        return (client.get_job_logs(submission_id) or "")[-max_bytes:]
    except Exception:
        return ""


def _age(ms: float | None) -> str:
    """Render a start-timestamp (ms) as a compact age string."""
    if not ms:
        return "-"
    delta = max(time.time() - ms / 1000.0, 0)
    if delta < 90:
        return f"{delta:.0f}s"
    if delta < 5400:
        return f"{delta / 60:.0f}m"
    if delta < 48 * 3600:
        return f"{delta / 3600:.1f}h"
    return f"{delta / 86400:.1f}d"


def _fmt_seconds(s: float) -> str:
    """Format a duration in seconds compactly."""
    s = max(s, 0)
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"


def _history_append(record: dict) -> None:
    """Append a submission record to ``~/.eray/history.jsonl`` (best effort)."""
    try:
        hist_dir = Path.home() / ".eray"
        hist_dir.mkdir(parents=True, exist_ok=True)
        with open(hist_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _resolve_last(client) -> str | None:
    """Most recently started submission id on the cluster."""
    jobs = [j for j in client.list_jobs() if j.submission_id]
    if not jobs:
        return None
    jobs.sort(key=lambda j: j.start_time or 0, reverse=True)
    return jobs[0].submission_id


def _parse_env_file(path: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` per line env file (comments/blank lines ok)."""
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"')
    return out


# ── commands ──────────────────────────────────────────────────────


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("entrypoint", nargs=-1, type=click.UNPROCESSED, required=True)
@click.option("--address", "-a", default=None, help="Ray dashboard address (default: RAY_ADDRESS or local).")
@click.option(
    "--working-dir",
    default=".",
    show_default=True,
    help="Directory packaged as the job's working dir.",
)
@click.option("--no-working-dir", is_flag=True, default=False, help="Do not package a working dir.")
@click.option(
    "--env-inherit/--no-env-inherit",
    default=True,
    show_default=True,
    help="Inject the current shell env (minus host-machine vars) into the job.",
)
@click.option("--env", "-e", "env_pairs", multiple=True, help="Extra KEY=VALUE env var (repeatable).")
@click.option("--env-file", default=None, type=click.Path(exists=True), help="KEY=VALUE file to load.")
@click.option("--id", "submission_id", default=None, help="Submission id (default: derived from script+user+time).")
@click.option("--force-package", is_flag=True, default=False, help="Skip the working-dir size guard.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Tail driver logs after submitting.")
def run(
    entrypoint,
    address,
    working_dir,
    no_working_dir,
    env_inherit,
    env_pairs,
    env_file,
    submission_id,
    force_package,
    follow,
):
    """Submit a job: eray run [opts] -- <command...>.

    The current directory is packaged as the working dir by default and the
    current shell environment (minus host-machine variables) is injected into
    the job, so launch scripts stop hardcoding credentials.
    """
    entrypoint = tuple(t for t in entrypoint if t != "--")
    if not entrypoint:
        raise click.UsageError("no entrypoint given; usage: eray run -- python launch.py")

    env_vars: dict[str, str] = {}
    if env_inherit:
        env_vars.update(inherited_env())
    if env_file:
        env_vars.update(_parse_env_file(env_file))
    for pair in env_pairs:
        if "=" not in pair:
            raise click.UsageError(f"--env expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        env_vars[key] = value

    runtime_env: dict = {}
    if env_vars:
        runtime_env["env_vars"] = env_vars
    if not no_working_dir:
        root = os.path.abspath(working_dir)
        size = package_size_bytes(root)
        if size > PACKAGE_ABORT_BYTES and not force_package:
            error(
                f"working dir {root} is {size / 1024**3:.1f} GB (> {PACKAGE_ABORT_BYTES / 1024**3:.0f} GB); "
                "refusing to package it. Narrow --working-dir, add a .rayignore, or pass --force-package."
            )
            raise SystemExit(2)
        if size > PACKAGE_WARN_BYTES:
            warning(f"working dir {root} is {size / 1024**2:.0f} MB — packaging will be slow.")
        runtime_env["working_dir"] = root

    sid = submission_id or generate_submission_id(entrypoint)
    metadata = {"cwd": os.getcwd(), "user": getpass.getuser(), **git_metadata()}

    client = make_client(address)
    submitted = client.submit_job(
        entrypoint=" ".join(shlex.quote(t) for t in entrypoint),
        submission_id=sid,
        runtime_env=runtime_env or None,
        metadata=metadata,
    )
    _history_append(
        {
            "submission_id": submitted,
            "entrypoint": list(entrypoint),
            "time": datetime.now(UTC).isoformat(),
            **metadata,
        }
    )

    info(f"submitted: {submitted}")
    if "working_dir" in runtime_env:
        info(f"working dir: {runtime_env['working_dir']}")
    if env_vars:
        shown = ", ".join(f"{k}={mask_value(k, v)}" for k, v in sorted(env_vars.items())[:6])
        info(f"env vars injected: {len(env_vars)} ({shown}{', …' if len(env_vars) > 6 else ''})")
    info(f"logs: eray logs {submitted} -f")

    if follow:
        _follow_logs(client, submitted)


def _follow_logs(client, submission_id: str) -> None:
    """Stream driver logs for *submission_id* until the job ends."""
    import asyncio

    async def _tail():
        async for chunk in client.tail_job_logs(submission_id):
            print(chunk, end="", flush=True)

    try:
        asyncio.run(_tail())
    except KeyboardInterrupt:
        warning("stopped following (job keeps running).")


@click.command()
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("-n", "limit", default=10, show_default=True, help="How many recent jobs to show.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def status(address, limit, as_json):
    """Recent jobs with truthful verdicts (log-derived, not just Ray's word)."""
    client = make_client(address)
    jobs = [j for j in client.list_jobs() if j.submission_id]
    jobs.sort(key=lambda j: j.start_time or 0, reverse=True)
    jobs = jobs[:limit]

    rows = []
    any_failed = False
    for job in jobs:
        state = str(getattr(job.status, "value", job.status))
        err_sig, phase = scan_log_tail(get_log_tail(client, job.submission_id))
        verdict = verdict_for(state, err_sig)
        if verdict.startswith(("failed", "erroring")):
            any_failed = True
        rows.append(
            {
                "id": job.submission_id,
                "state": state,
                "verdict": verdict,
                "phase": phase or "-",
                "age": _age(job.start_time),
                "runtime": _fmt_seconds(((job.end_time or time.time() * 1000) - job.start_time) / 1000.0)
                if job.start_time
                else "-",
                "entrypoint": (job.entrypoint or "")[:48],
            }
        )

    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        if not rows:
            info("no jobs found.")
            return
        widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in rows[0]}
        header = "  ".join(k.upper().ljust(widths[k]) for k in rows[0])
        print(header)
        for r in rows:
            line = "  ".join(str(r[k]).ljust(widths[k]) for k in r)
            if r["verdict"].startswith(("failed", "erroring")):
                line = f"{RED}{line}{NC}"
            elif r["verdict"].startswith("user-stop"):
                line = f"{YELLOW}{line}{NC}"
            print(line)
    if any_failed:
        raise SystemExit(1)


@click.command()
@click.argument("job_id", default="last")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--errors", is_flag=True, default=False, help="Show only tracebacks and error-signature lines.")
@click.option("--grep", "pattern", default=None, help="Only lines matching this regex.")
@click.option("--raw", is_flag=True, default=False, help="Do not filter progress-bar spam.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Stream logs until the job ends.")
def logs(job_id, address, errors, pattern, raw, follow):
    """Driver logs for a job (default: the most recent one)."""
    client = make_client(address)
    if job_id == "last":
        job_id = _resolve_last(client)
        if job_id is None:
            error("no jobs found.")
            raise SystemExit(1)
        info(f"showing logs for {job_id}")

    if follow:
        _follow_logs(client, job_id)
        return

    text = client.get_job_logs(job_id) or ""
    matcher = re.compile(pattern) if pattern else None
    in_traceback = False
    for line in text.splitlines():
        if errors:
            if "Traceback (most recent call last)" in line:
                in_traceback = True
            sig_hit = any(needle in line for needle, _ in _ERROR_SIGNATURES)
            if in_traceback or sig_hit:
                print(line)
                if in_traceback and line and not line.startswith((" ", "\t")) and "Traceback" not in line:
                    in_traceback = False
            continue
        if matcher and not matcher.search(line):
            continue
        if not raw and _PROGRESS_SPAM_RE.search(line):
            continue
        print(line)


@click.command()
@click.argument("job_id", required=False)
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--last", "use_last", is_flag=True, default=False, help="Stop the most recent job.")
def stop(job_id, address, use_last):
    """Stop a job by id (or --last)."""
    client = make_client(address)
    if use_last and not job_id:
        job_id = _resolve_last(client)
    if not job_id:
        raise click.UsageError("give a job id or --last")
    client.stop_job(job_id)
    info(f"stop requested: {job_id}")


def register(cli_group: click.Group) -> None:
    """Attach the job commands to the top-level CLI group."""
    cli_group.add_command(run)
    cli_group.add_command(status)
    cli_group.add_command(status, name="ps")
    cli_group.add_command(logs)
    cli_group.add_command(stop)
