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

# Log-scanning patterns. NOTHING here is baked in: these are the shipped
# defaults, and every list is overridable (replaced wholesale, not appended)
# via JSON at ``~/.eray/patterns.json``, ``$ERAY_PATTERNS``, or a
# project-local ``./.eray-patterns.json`` — later files win. Schema mirrors
# this dict: errors/phases are ordered [needle, name] pairs (first error hit
# wins; the *last* phase marker present wins), step_metric/progress_metrics
# name the ``'metric': value`` fields to surface, spam is a list of regexes
# filtered out of ``eray logs`` output.
DEFAULT_PATTERNS: dict = {
    "errors": [
        ["Failed to merge the Job's runtime env", "env-conflict"],
        ["CompileTimeHbmOom", "oom-compile"],
        ["RESOURCE_EXHAUSTED", "oom"],
        ["abstract trainable parameter", "load-incomplete"],
        ["Traceback (most recent call last)", "remote-raise"],
    ],
    "phases": [
        ["Uploading package", "packaging"],
        ["Loading:", "loading"],
        ["loaded state step", "loaded"],
        ["Compiling", "compiling"],
        ["time took for configure shard", "compiling"],
        ["Converting shard", "converting"],
    ],
    "step_metric": "train_step",
    "progress_metrics": ["kl_loss", "loss"],
    "spam": ["tensor/s", "\\.\\.\\.\\s*\\d+%", "\\d+/\\d+ \\[\\d+:\\d+<"],
}

PATTERNS_ENV = "ERAY_PATTERNS"
_PATTERN_FILES = (str(Path.home() / ".eray" / "patterns.json"), "./.eray-patterns.json")


def load_patterns() -> dict:
    """Merge the default log-scanning patterns with user/project overrides.

    Sources, later wins per top-level key: built-in defaults, then
    ``~/.eray/patterns.json``, then ``$ERAY_PATTERNS`` (a path), then
    ``./.eray-patterns.json``.
    """
    merged = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_PATTERNS.items()}
    paths = [_PATTERN_FILES[0], os.environ.get(PATTERNS_ENV) or "", _PATTERN_FILES[1]]
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            overrides = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            warning(f"ignoring unreadable patterns file {path}: {exc}")
            continue
        for key, value in overrides.items():
            merged[key] = value
    return merged


def _spam_re(patterns: dict) -> re.Pattern:
    """Compiled alternation of the spam-filter regexes."""
    return re.compile("|".join(patterns.get("spam") or ["(?!x)x"]))


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


def scan_log_tail(text: str, patterns: dict | None = None) -> tuple[str | None, str | None]:
    """Extract an error signature and a phase description from a log tail.

    Args:
        text: The last chunk of a job's driver log.

    Returns:
        ``(error, phase)`` where ``error`` is a short signature name (or
        ``None`` when no failure marker is present) and ``phase`` is the most
        advanced progress marker seen (``step N (kl X)`` once metrics flow).
    """
    patterns = load_patterns() if patterns is None else patterns

    err = None
    for needle, name in patterns.get("errors", ()):
        if needle in text:
            err = name
            break

    phase = None
    best_pos = -1
    for needle, name in patterns.get("phases", ()):
        pos = text.rfind(needle)
        if pos > best_pos:
            best_pos = pos
            phase = name
    step_metric = patterns.get("step_metric")
    step = latest_metric(text, step_metric) if step_metric else None
    if step is not None:
        phase = f"step {int(step)}"
        for metric in patterns.get("progress_metrics", ()):
            value = latest_metric(text, metric)
            if value is not None:
                short = metric.removesuffix("_loss") or metric
                phase += f" ({short} {value:g})"
                break
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
@click.option("--queue", is_flag=True, default=False, help="Wait until no job is running before submitting.")
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
    queue,
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
    if queue:
        while True:
            busy = [
                j.submission_id
                for j in client.list_jobs()
                if str(getattr(j.status, "value", j.status)).upper() in ("RUNNING", "PENDING")
            ]
            if not busy:
                break
            info(f"queued behind {busy[0]}" + (f" (+{len(busy) - 1} more)" if len(busy) > 1 else "") + " …")
            time.sleep(30)
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
    patterns = load_patterns()
    spam_re = _spam_re(patterns)
    matcher = re.compile(pattern) if pattern else None
    in_traceback = False
    for line in text.splitlines():
        if errors:
            if "Traceback (most recent call last)" in line:
                in_traceback = True
            sig_hit = any(needle in line for needle, _ in patterns.get("errors", ()))
            if in_traceback or sig_hit:
                print(line)
                if in_traceback and line and not line.startswith((" ", "\t")) and "Traceback" not in line:
                    in_traceback = False
            continue
        if matcher and not matcher.search(line):
            continue
        if not raw and spam_re.search(line):
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
    cli_group.add_command(watch)
    cli_group.add_command(doctor)
    cli_group.add_command(clean)
    cli_group.add_command(nodes)
    cli_group.add_command(rerun)
    cli_group.add_command(diff)


# ── P2/P3: watch, doctor, clean, nodes, rerun, diff ──────────────

_ALERT_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|>|<)\s*([0-9.]+)\s*$")


def latest_metric(text: str, name: str) -> float | None:
    """Latest value of ``'name': X`` style metric in a log tail."""
    values = re.findall(rf"'{re.escape(name)}': ([0-9.]+)", text)
    return float(values[-1]) if values else None


def evaluate_alerts(text: str, exprs: tuple[str, ...]) -> list[str]:
    """Evaluate ``metric>threshold`` style alert expressions against a log tail.

    Args:
        text: Driver-log tail.
        exprs: Expressions like ``kl_loss>5`` or ``train_step_time>=120``.

    Returns:
        Human-readable violation strings for every alert that fires.

    Raises:
        click.UsageError: On a malformed expression.
    """
    ops = {">": float.__gt__, "<": float.__lt__, ">=": float.__ge__, "<=": float.__le__}
    fired = []
    for expr in exprs:
        m = _ALERT_RE.match(expr)
        if not m:
            raise click.UsageError(f"bad --alert {expr!r}; expected e.g. kl_loss>5")
        name, op, threshold = m.group(1), m.group(2), float(m.group(3))
        value = latest_metric(text, name)
        if value is not None and ops[op](value, threshold):
            fired.append(f"{name}={value} {op} {threshold}")
    return fired


@click.command()
@click.argument("job_id", default="last")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--interval", default=30, show_default=True, help="Poll interval seconds.")
@click.option("--until-step", default=None, type=int, help="Exit 0 once this train_step is reached.")
@click.option("--alert", "alerts", multiple=True, help="Fire on metric threshold, e.g. 'kl_loss>5' (repeatable).")
@click.option("--timeout-min", default=120, show_default=True, help="Give up after this many minutes.")
def watch(job_id, address, interval, until_step, alerts, timeout_min):
    """Watch a job: phase transitions, metric alerts, error signatures.

    Exit codes: 0 until-step reached or job succeeded; 1 error signature or
    job failed; 2 an --alert fired; 3 watch timeout.
    """
    client = make_client(address)
    if job_id == "last":
        job_id = _resolve_last(client)
        if job_id is None:
            error("no jobs found.")
            raise SystemExit(1)
        info(f"watching {job_id}")

    deadline = time.time() + timeout_min * 60
    last_phase = None
    while time.time() < deadline:
        tail = get_log_tail(client, job_id, max_bytes=131072)
        err_sig, phase = scan_log_tail(tail)
        if phase and phase != last_phase:
            info(f"phase: {phase}")
            last_phase = phase
        if err_sig:
            error(f"error signature: {err_sig}")
            raise SystemExit(1)
        fired = evaluate_alerts(tail, alerts)
        if fired:
            for f in fired:
                error(f"alert: {f}")
            raise SystemExit(2)
        if until_step is not None:
            step = latest_metric(tail, "train_step")
            if step is not None and step >= until_step:
                info(f"reached step {int(step)}")
                raise SystemExit(0)
        try:
            state = str(getattr(client.get_job_status(job_id), "value", ""))
            if state in ("SUCCEEDED", "FAILED", "STOPPED"):
                info(f"job ended: {state} — verdict {verdict_for(state, err_sig)}")
                raise SystemExit(0 if state == "SUCCEEDED" and not err_sig else 1)
        except SystemExit:
            raise
        except Exception:
            pass
        time.sleep(interval)
    error("watch timeout.")
    raise SystemExit(3)


def component_log_report(max_bytes: int) -> list[dict]:
    """Sizes of Ray component logs on this host, flagged when oversized."""
    from ..core.monitoring import _RAYLET_LOG_GUARD_FILES, _ray_session_log_dirs

    rows = []
    for logs_dir in _ray_session_log_dirs():
        for name in _RAYLET_LOG_GUARD_FILES:
            path = os.path.join(logs_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            rows.append({"path": path, "bytes": size, "flagged": size > max_bytes})
    return rows


def referenced_packages(jobs_list) -> set[str]:
    """Package basenames referenced by live jobs' runtime envs."""
    keep: set[str] = set()
    for job in jobs_list:
        state = str(getattr(job.status, "value", job.status)).upper()
        if state not in ("RUNNING", "PENDING"):
            continue
        env = getattr(job, "runtime_env", None) or {}
        wd = str(env.get("working_dir", ""))
        if "_ray_pkg_" in wd:
            keep.add(os.path.basename(wd).replace(".zip", ""))
    return keep


def find_stale_packages(session_dir: str, referenced: set[str]) -> list[tuple[str, int]]:
    """Unreferenced ``_ray_pkg_*`` working-dir snapshots under a session dir."""
    stale: list[tuple[str, int]] = []
    root = os.path.join(session_dir, "runtime_resources", "working_dir_files")
    if not os.path.isdir(root):
        return stale
    for entry in os.listdir(root):
        if not entry.startswith("_ray_pkg_") or entry.replace(".zip", "") in referenced:
            continue
        path = os.path.join(root, entry)
        size = package_size_bytes(path) if os.path.isdir(path) else os.path.getsize(path)
        stale.append((path, size))
    return stale


@click.command()
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--log-max-gb", default=5.0, show_default=True, help="Component-log size flag threshold.")
def doctor(address, log_max_gb):
    """Host + cluster health: disk, raylet-log spam, packages, nodes, jobs API."""
    import shutil

    exit_code = 0

    for mount in dict.fromkeys(["/", os.environ.get("RAY_TMPDIR") or "/tmp"]):
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        pct = usage.used / usage.total * 100
        line = f"disk {mount}: {usage.free / 1024**3:.1f} GB free ({pct:.0f}% used)"
        if pct > 90:
            error(line + "  ← CRITICAL")
            exit_code = 1
        elif pct > 80:
            warning(line)
        else:
            info(line)

    max_bytes = int(log_max_gb * 1024**3)
    for row in component_log_report(max_bytes):
        line = f"log {row['path']}: {row['bytes'] / 1024**3:.2f} GB"
        if row["flagged"]:
            error(line + "  ← oversized, run: eray clean raylet")
            exit_code = 1
        elif row["bytes"] > max_bytes // 4:
            warning(line)

    client = None
    try:
        client = make_client(address)
        jobs_list = client.list_jobs()
        running = sum(1 for j in jobs_list if str(getattr(j.status, "value", j.status)).upper() == "RUNNING")
        info(f"jobs API reachable: {len(jobs_list)} jobs known, {running} running")
    except Exception as exc:
        error(f"jobs API unreachable: {exc}")
        exit_code = 1

    if client is not None:
        try:
            from ..core.monitoring import _ray_session_log_dirs

            for logs_dir in _ray_session_log_dirs():
                session_dir = os.path.dirname(logs_dir)
                stale = find_stale_packages(session_dir, referenced_packages(client.list_jobs()))
                if stale:
                    total = sum(s for _, s in stale)
                    warning(
                        f"{len(stale)} stale working-dir package(s), {total / 1024**3:.1f} GB "
                        f"under {session_dir} — run: eray clean packages"
                    )
        except Exception:
            pass

    try:
        from ray.util.state import list_nodes

        nodes_list = list_nodes(address=resolve_address(address), limit=1000)
        alive = sum(1 for n in nodes_list if str(getattr(n, "state", "")).upper() == "ALIVE")
        line = f"nodes: {alive}/{len(nodes_list)} alive"
        (info if alive == len(nodes_list) else warning)(line)
    except Exception as exc:
        warning(f"node state API unavailable: {exc}")

    raise SystemExit(exit_code)


@click.group()
def clean() -> None:
    """Reclaim disk: oversized Ray logs, stale working-dir packages."""


@clean.command("raylet")
@click.option("--max-gb", default=5.0, show_default=True, help="Truncate component logs above this size.")
def clean_raylet(max_gb):
    """Truncate oversized raylet/GCS logs in place (safe on the live daemon)."""
    from ..core.monitoring import sweep_raylet_logs

    truncated = sweep_raylet_logs(max_bytes=int(max_gb * 1024**3))
    if not truncated:
        info("nothing to truncate.")
        return
    for path, size in truncated:
        info(f"truncated {path} (was {size / 1024**3:.1f} GB)")


@clean.command("packages")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--yes", is_flag=True, default=False, help="Actually delete (default: dry run).")
def clean_packages(address, yes):
    """Delete _ray_pkg_* snapshots not referenced by any live job."""
    import shutil

    from ..core.monitoring import _ray_session_log_dirs

    try:
        referenced = referenced_packages(make_client(address).list_jobs())
    except Exception:
        warning("jobs API unreachable — refusing to guess which packages are live.")
        raise SystemExit(1) from None

    found = False
    for logs_dir in _ray_session_log_dirs():
        for path, size in find_stale_packages(os.path.dirname(logs_dir), referenced):
            found = True
            if yes:
                shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.unlink(path)
                info(f"deleted {path} ({size / 1024**2:.0f} MB)")
            else:
                info(f"stale: {path} ({size / 1024**2:.0f} MB)  [dry run — pass --yes to delete]")
    if not found:
        info("no stale packages.")


@click.command()
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def nodes(address, as_json):
    """Cluster nodes: alive/dead and TPU resource totals."""
    from ray.util.state import list_nodes

    nodes_list = list_nodes(address=resolve_address(address), limit=1000)
    rows = []
    for n in nodes_list:
        resources = getattr(n, "resources_total", None) or {}
        rows.append(
            {
                "ip": getattr(n, "node_ip", "?"),
                "state": str(getattr(n, "state", "?")),
                "tpu": int(resources.get("TPU", 0)),
                "head": bool(resources.get("head-node", 0)),
            }
        )
    alive = sum(1 for r in rows if r["state"].upper() == "ALIVE")
    tpus = sum(r["tpu"] for r in rows if r["state"].upper() == "ALIVE")
    if as_json:
        print(json.dumps({"alive": alive, "total": len(rows), "tpu_total": tpus, "nodes": rows}, indent=2))
        return
    info(f"{alive}/{len(rows)} nodes alive, {tpus} TPU chips")
    for r in sorted(rows, key=lambda r: (not r["head"], r["ip"])):
        marker = " (head)" if r["head"] else ""
        line = f"  {r['ip']:<16} {r['state']:<6} TPU={r['tpu']}{marker}"
        print(line if r["state"].upper() == "ALIVE" else f"{RED}{line}{NC}")


@click.command()
@click.argument("job_id", default="last")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--id", "new_id", default=None, help="Submission id for the rerun.")
def rerun(job_id, address, new_id):
    """Resubmit a previous job with its recorded entrypoint and runtime env."""
    client = make_client(address)
    if job_id == "last":
        job_id = _resolve_last(client)
        if job_id is None:
            error("no jobs found.")
            raise SystemExit(1)
    job = client.get_job_info(job_id)
    sid = new_id or f"{job_id}-r{datetime.now(UTC).strftime('%H%M%S')}"
    submitted = client.submit_job(
        entrypoint=job.entrypoint,
        submission_id=sid,
        runtime_env=getattr(job, "runtime_env", None),
        metadata={**(getattr(job, "metadata", None) or {}), "rerun_of": job_id},
    )
    info(f"resubmitted {job_id} as {submitted}")


@click.command()
@click.argument("job_a")
@click.argument("job_b")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
def diff(job_a, job_b, address):
    """Compare two jobs: entrypoint, git provenance, env-var deltas."""
    client = make_client(address)
    infos = {jid: client.get_job_info(jid) for jid in (job_a, job_b)}

    def _field(jid, name):
        return (getattr(infos[jid], "metadata", None) or {}).get(name, "-")

    for name in ("git_sha", "git_dirty", "cwd", "user"):
        va, vb = _field(job_a, name), _field(job_b, name)
        print(f"{name:<10} {va:<24} {'==' if va == vb else '!='} {vb}")
    ea, eb = infos[job_a].entrypoint or "", infos[job_b].entrypoint or ""
    print(f"{'entrypoint':<10} {'==' if ea == eb else '!='}  {ea!r}  vs  {eb!r}")

    env_a = ((getattr(infos[job_a], "runtime_env", None) or {}).get("env_vars")) or {}
    env_b = ((getattr(infos[job_b], "runtime_env", None) or {}).get("env_vars")) or {}
    for key in sorted(set(env_a) | set(env_b)):
        va, vb = env_a.get(key), env_b.get(key)
        if va == vb:
            continue
        fa = "<unset>" if va is None else mask_value(key, va)
        fb = "<unset>" if vb is None else mask_value(key, vb)
        print(f"env {key}: {fa} -> {fb}")
