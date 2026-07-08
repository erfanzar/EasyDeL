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

import asyncio
import getpass
import glob
import json
import os
import re
import shlex
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import click

from ..provision.fleet import RAY_DASHBOARD_PORT
from .utils import NC, RED, YELLOW, error, info, warning

DEFAULT_DASHBOARD = f"http://127.0.0.1:{RAY_DASHBOARD_PORT}"

# Bound on the `watch` pre-flight job-existence probe. Module-level so tests
# can shrink it; see the probe in `watch` for why a daemon thread (not an HTTP
# timeout or a thread pool) is what actually enforces it.
_PREFLIGHT_TIMEOUT_S = 15.0

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
    "progress_metrics": [["kl_loss", "kl"], ["loss", "loss"], ["train_step_time", "s/step"]],
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


def _auto_dashboard_address() -> str | None:
    """``http://127.0.0.1:<port>`` for a lone open eray dashboard tunnel.

    Lets `eray status`/`logs`/`watch` work from a laptop with no ``-a`` and
    no ``RAY_ADDRESS``, and correctly picks up the tunnel's actual local
    port even when it wasn't 8265 (8265 was busy). Returns None when there
    is no tunnel, or more than one (ambiguous — fall back to the default).
    """
    from ..provision.tunnel import tunnels_for_remote_port

    dash = tunnels_for_remote_port(RAY_DASHBOARD_PORT)
    if len(dash) == 1:
        return f"http://127.0.0.1:{dash[0].local_port}"
    return None


def resolve_address(explicit: str | None) -> str:
    """Resolve the Ray dashboard address for job-submission clients.

    Order: explicit flag, ``RAY_ADDRESS`` env var, a lone open eray
    dashboard tunnel, then the local default dashboard. Bare ``host:port``
    values are normalized to ``http://``.

    Args:
        explicit: Address passed on the command line, if any.

    Returns:
        An ``http(s)://host:port`` URL.
    """
    addr = explicit or os.environ.get("RAY_ADDRESS") or _auto_dashboard_address() or DEFAULT_DASHBOARD
    if addr.startswith(("http://", "https://")):
        return addr
    if ":" in addr:
        host, port = addr.rsplit(":", 1)
        if port == "6379":  # GCS port — jobs API lives on the dashboard
            return f"http://{host}:{RAY_DASHBOARD_PORT}"
        return f"http://{addr}"
    return f"http://{addr}:{RAY_DASHBOARD_PORT}"


def make_client(address: str | None):
    """Build a ``JobSubmissionClient`` for *address* (module-level for tests)."""
    from ray.job_submission import JobSubmissionClient

    return JobSubmissionClient(resolve_address(address))


def resolve_cluster_address(cluster: str) -> str:
    """Dashboard address for a fleet-registered cluster.

    Args:
        cluster: Registered cluster name (see ``eray fleet add``).

    Returns:
        ``http://<head_ip>:8265``.

    Raises:
        click.ClickException: If the cluster is unknown or has no recorded
            head yet (run ``eray fleet ensure <cluster>`` first).
    """
    from ..provision.registry import ClusterRegistry

    record = ClusterRegistry.from_config().get(cluster)
    if record is None:
        raise click.ClickException(f"cluster {cluster!r} is not registered (eray fleet add ...)")
    if not record.head_ip:
        raise click.ClickException(f"cluster {cluster!r} has no known head yet (eray fleet ensure {cluster})")
    return f"http://{record.head_ip}:{RAY_DASHBOARD_PORT}"


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
        parts = []
        seen_values = set()
        for entry in patterns.get("progress_metrics", ()):
            metric, label = entry if isinstance(entry, (list, tuple)) else (entry, entry)
            value = latest_metric(text, metric)
            # skip aliases that resolve to the same number (distill logs
            # report loss == kl_loss; showing both is noise)
            if value is None or value in seen_values:
                continue
            seen_values.add(value)
            parts.append(f"{label} {value:g}")
        if parts:
            phase += f" ({', '.join(parts)})"
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


def extract_metric_rows(text: str, patterns: dict | None = None) -> list[dict]:
    """One row per step from metric log lines, columns from the patterns config."""
    patterns = load_patterns() if patterns is None else patterns
    step_metric = patterns.get("step_metric") or "train_step"
    columns = [e if isinstance(e, (list, tuple)) else (e, e) for e in patterns.get("progress_metrics", ())]
    rows: list[dict] = []
    seen_steps: set[int] = set()
    for line in text.splitlines():
        step_vals = re.findall(rf"'{re.escape(step_metric)}': ([0-9]+)", line)
        if not step_vals:
            continue
        step = int(step_vals[-1])
        if step in seen_steps:
            continue
        seen_steps.add(step)
        row = {"step": step}
        for metric, label in columns:
            value = latest_metric(line, metric)
            if value is not None:
                row[label] = f"{value:g}"
        rows.append(row)
    return rows


# ── commands ──────────────────────────────────────────────────────


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("entrypoint", nargs=-1, type=click.UNPROCESSED, required=True)
@click.option("--address", "-a", default=None, help="Ray dashboard address (default: RAY_ADDRESS or local).")
@click.option(
    "--cluster", "-c", default=None, help="Fleet-registered cluster name (resolves the address from the registry)."
)
@click.option(
    "--restartable",
    is_flag=True,
    default=False,
    help="Mark the job safe for automatic resubmission after a spot preemption (used by eray fleet watch).",
)
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
@click.option("--json", "as_json", is_flag=True, default=False, help="Print the submission record as JSON.")
def run(
    entrypoint,
    address,
    cluster,
    restartable,
    working_dir,
    no_working_dir,
    env_inherit,
    env_pairs,
    env_file,
    submission_id,
    force_package,
    follow,
    queue,
    as_json,
):
    """Submit a job: eray run [opts] -- <command...>.

    The current directory is packaged as the working dir by default and the
    current shell environment (minus host-machine variables) is injected into
    the job, so launch scripts stop hardcoding credentials. Address
    precedence: --address, then --cluster (fleet registry), then RAY_ADDRESS,
    then the local dashboard.
    """
    entrypoint = tuple(t for t in entrypoint if t != "--")
    if not entrypoint:
        raise click.UsageError("no entrypoint given; usage: eray run -- python launch.py")
    if address is None and cluster:
        address = resolve_cluster_address(cluster)

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
    if cluster:
        metadata["cluster"] = cluster
    if restartable:
        metadata["restartable"] = "1"

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

    if as_json:
        print(
            json.dumps(
                {
                    "submission_id": submitted,
                    "entrypoint": list(entrypoint),
                    "working_dir": runtime_env.get("working_dir"),
                    "env_vars": {k: mask_value(k, v) for k, v in env_vars.items()},
                    **metadata,
                }
            )
        )
        if follow:
            _follow_logs(client, submitted)
        return
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
@click.option("--watch", "refresh_s", default=0, show_default=True, help="Redraw every N seconds (0 = once).")
def status(address, limit, as_json, refresh_s):
    """Recent jobs with truthful verdicts (log-derived, not just Ray's word)."""
    client = make_client(address)
    while refresh_s > 0:
        click.clear()
        try:
            _status_once(client, limit, as_json)
        except SystemExit:
            pass
        time.sleep(refresh_s)
    _status_once(client, limit, as_json)


def _status_once(client, limit, as_json):
    """Render the status table once (nonzero SystemExit on failing verdicts)."""
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
@click.option("--metrics", "metrics_only", is_flag=True, default=False, help="Compact per-step metrics table.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Stream logs until the job ends.")
def logs(job_id, address, errors, pattern, raw, metrics_only, follow):
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
    if metrics_only:
        rows = extract_metric_rows(text, patterns)
        if not rows:
            info("no metric lines found.")
            return
        headers = list(rows[0])
        widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
        print("  ".join(h.upper().ljust(widths[h]) for h in headers))
        for r in rows:
            print("  ".join(str(r.get(h, "-")).ljust(widths[h]) for h in headers))
        return
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


async def _stream_and_scan(client, job_id: str, alerts: tuple[str, ...], until_step: int | None, deadline: float) -> int:
    """Stream *job_id*'s full driver log to stdout, scanning it live.

    Reuses the same websocket tail Ray exposes for ``eray logs -f`` (see
    ``_follow_logs``), so the operator sees real output instead of a
    phase-only summary. ``tail_job_logs`` always replays the driver log from
    byte 0 on (re)connect, so a running ``seen`` cursor de-dupes replayed
    content across reconnects instead of re-printing the whole log.

    Returns:
        The process exit code ``watch`` should raise: 0 (until-step reached
        or job succeeded), 1 (error signature or job failed), 2 (alert
        fired), or 3 (the overall deadline elapsed).
    """
    seen = 0
    scan_buf = ""
    last_phase = None

    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return 3
        replayed = 0
        try:
            stream = client.tail_job_logs(job_id).__aiter__()
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return 3
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=remaining)

                new_text = chunk[seen - replayed :] if replayed < seen else chunk
                replayed += len(chunk)
                if not new_text:
                    continue

                print(new_text, end="", flush=True)
                seen += len(new_text)
                scan_buf = (scan_buf + new_text)[-131072:]

                err_sig, phase = scan_log_tail(scan_buf)
                if phase and phase != last_phase:
                    info(f"phase: {phase}")
                    last_phase = phase
                if err_sig:
                    error(f"error signature: {err_sig}")
                    return 1
                fired = evaluate_alerts(scan_buf, alerts)
                if fired:
                    for f in fired:
                        error(f"alert: {f}")
                    return 2
                if until_step is not None:
                    step = latest_metric(scan_buf, "train_step")
                    if step is not None and step >= until_step:
                        info(f"reached step {int(step)}")
                        return 0
        except StopAsyncIteration:
            pass  # server closed the stream — check below whether the job actually ended
        except TimeoutError:
            return 3
        except Exception as exc:
            warning(f"log stream interrupted ({exc}); reconnecting...")
            await asyncio.sleep(2)
            continue

        try:
            state = str(getattr(client.get_job_status(job_id), "value", ""))
        except Exception:
            state = ""
        if state in ("SUCCEEDED", "FAILED", "STOPPED"):
            err_sig, _ = scan_log_tail(scan_buf)
            info(f"job ended: {state} — verdict {verdict_for(state, err_sig)}")
            return 0 if state == "SUCCEEDED" and not err_sig else 1
        await asyncio.sleep(2)  # stream closed but job still running — reconnect


@click.command()
@click.argument("job_id", default="last")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--until-step", default=None, type=int, help="Exit 0 once this train_step is reached.")
@click.option("--alert", "alerts", multiple=True, help="Fire on metric threshold, e.g. 'kl_loss>5' (repeatable).")
@click.option("--timeout-min", default=120, show_default=True, help="Give up after this many minutes.")
@click.option(
    "--interval",
    default=None,
    type=int,
    hidden=True,
    help="Deprecated, ignored: watch now streams live instead of polling.",
)
def watch(job_id, address, until_step, alerts, timeout_min, interval):
    """Stream a job's full driver log live, watching for phases/alerts/errors.

    Exit codes: 0 until-step reached or job succeeded; 1 error signature or
    job failed; 2 an --alert fired; 3 watch timeout.
    """
    if interval is not None:
        warning("--interval is deprecated and ignored: watch now streams live instead of polling.")
    client = make_client(address)
    if job_id == "last":
        job_id = _resolve_last(client)
        if job_id is None:
            error("no jobs found.")
            raise SystemExit(1)
    # Check existence via list_jobs() rather than get_job_status(): Ray's SDK
    # raises the same generic error for "job doesn't exist" and "dashboard
    # unreachable", so a bare except-and-report-not-found here would tell the
    # operator monitoring a live run that their job vanished when the real
    # problem is a transient dashboard blip. The probe runs on a DAEMON thread
    # with a bounded join: list_jobs() carries no HTTP timeout of its own
    # (Ray's _do_request passes none to requests), so a black-holed connection
    # would otherwise hang this pre-flight forever — and a thread pool can't
    # enforce the bound either, because both ThreadPoolExecutor's context exit
    # and concurrent.futures' atexit hook join their (non-daemon) workers,
    # re-hanging on the stuck call right after the "timeout" fired.
    probe: dict = {}

    def _probe() -> None:
        try:
            probe["ids"] = {j.submission_id for j in client.list_jobs()}
        except Exception as exc:
            probe["err"] = exc

    prober = threading.Thread(target=_probe, daemon=True, name="eray-watch-preflight")
    prober.start()
    prober.join(_PREFLIGHT_TIMEOUT_S)
    if prober.is_alive():
        error(f"could not reach the Ray dashboard within {_PREFLIGHT_TIMEOUT_S:.0f}s to check job {job_id}.")
        raise SystemExit(1)
    if "err" in probe:
        error(f"could not reach the Ray dashboard to check job {job_id}: {probe['err']}")
        raise SystemExit(1)
    if job_id not in probe["ids"]:
        error(f"job {job_id} not found.")
        raise SystemExit(1)
    info(f"watching {job_id}")

    deadline = time.time() + timeout_min * 60
    try:
        code = asyncio.run(_stream_and_scan(client, job_id, alerts, until_step, deadline))
    except KeyboardInterrupt:
        warning("stopped watching (job keeps running).")
        raise SystemExit(130) from None

    if code == 3:
        error("watch timeout.")
    raise SystemExit(code)


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


def tpu_device_holders() -> list[dict]:
    """Processes holding TPU device files open (/dev/accel*, /dev/vfio/*)."""
    holders: list[dict] = []
    device_prefixes = ("/dev/accel", "/dev/vfio")
    for pid_dir in glob.glob("/proc/[0-9]*"):
        try:
            fds = os.listdir(os.path.join(pid_dir, "fd"))
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(pid_dir, "fd", fd))
            except OSError:
                continue
            if target.startswith(device_prefixes):
                try:
                    cmd = Path(pid_dir, "cmdline").read_bytes().replace(b"\0", b" ").decode()[:120].strip()
                except OSError:
                    cmd = "?"
                holders.append({"pid": int(os.path.basename(pid_dir)), "device": target, "cmd": cmd})
                break
    return holders


@click.command()
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--log-max-gb", default=5.0, show_default=True, help="Component-log size flag threshold.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Emit findings as JSON.")
def doctor(address, log_max_gb, as_json):
    """Host + cluster health: disk, raylet-log spam, TPU locks, packages, nodes."""
    import shutil

    exit_code = 0
    findings: list[dict] = []

    def _emit(level, kind, line, **extra):
        findings.append({"level": level, "kind": kind, "detail": line, **extra})
        if not as_json:
            {"info": info, "warn": warning, "error": error}[level](line)

    for mount in dict.fromkeys(["/", os.environ.get("RAY_TMPDIR") or "/tmp"]):
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        pct = usage.used / usage.total * 100
        line = f"disk {mount}: {usage.free / 1024**3:.1f} GB free ({pct:.0f}% used)"
        if pct > 90:
            _emit("error", "disk", line + "  ← CRITICAL")
            exit_code = 1
        elif pct > 80:
            _emit("warn", "disk", line)
        else:
            _emit("info", "disk", line)

    max_bytes = int(log_max_gb * 1024**3)
    for row in component_log_report(max_bytes):
        line = f"log {row['path']}: {row['bytes'] / 1024**3:.2f} GB"
        if row["flagged"]:
            _emit("error", "component-log", line + "  ← oversized, run: eray clean raylet")
            exit_code = 1
        elif row["bytes"] > max_bytes // 4:
            _emit("warn", "component-log", line)

    holders = tpu_device_holders()
    if holders:
        for h in holders[:8]:
            _emit("info", "tpu-lock", f"TPU device {h['device']} held by pid {h['pid']}: {h['cmd']}", **h)
    else:
        _emit("info", "tpu-lock", "no process holds a TPU device on this host")

    client = None
    try:
        client = make_client(address)
        jobs_list = client.list_jobs()
        running = sum(1 for j in jobs_list if str(getattr(j.status, "value", j.status)).upper() == "RUNNING")
        _emit("info", "jobs-api", f"jobs API reachable: {len(jobs_list)} jobs known, {running} running")
    except Exception as exc:
        _emit("error", "jobs-api", f"jobs API unreachable: {exc}")
        exit_code = 1

    if client is not None:
        try:
            from ..core.monitoring import _ray_session_log_dirs

            for logs_dir in _ray_session_log_dirs():
                session_dir = os.path.dirname(logs_dir)
                stale = find_stale_packages(session_dir, referenced_packages(client.list_jobs()))
                if stale:
                    total = sum(s for _, s in stale)
                    _emit(
                        "warn",
                        "stale-packages",
                        f"{len(stale)} stale working-dir package(s), {total / 1024**3:.1f} GB "
                        f"under {session_dir} — run: eray clean packages",
                    )
        except Exception:
            pass

    try:
        from ray.util.state import list_nodes

        nodes_list = list_nodes(address=resolve_address(address), limit=1000)
        alive = sum(1 for n in nodes_list if str(getattr(n, "state", "")).upper() == "ALIVE")
        line = f"nodes: {alive}/{len(nodes_list)} alive"
        _emit("info" if alive == len(nodes_list) else "warn", "nodes", line)
    except Exception as exc:
        _emit("warn", "nodes", f"node state API unavailable: {exc}")

    # Fleet registry (best-effort, registry-only — no gcloud probes so doctor stays fast).
    try:
        from ..provision.registry import ClusterRegistry

        registry = ClusterRegistry.from_config()
        records = registry.load()
        if records:
            holder = registry.lease_holder()
            _emit(
                "info",
                "fleet",
                f"fleet registry: {len(records)} cluster(s), watcher lease "
                f"{'held by ' + holder if holder else 'free (no live watcher)'}",
            )
            for name, rec in sorted(records.items()):
                line = f"fleet {name}: {rec.state or 'UNKNOWN'} (desired {rec.desired_state}, gen {rec.generation})"
                if rec.state.startswith("HALTED"):
                    _emit("error", "fleet", line + "  ← parked, run: eray fleet resume " + name)
                    exit_code = 1
                elif rec.state == "NEEDS_BOOTSTRAP":
                    _emit("warn", "fleet", line + "  ← bootstrap failed; fix, then: eray fleet resume " + name)
                elif rec.desired_state == "up" and rec.state not in ("HEALTHY", "CONNECTED"):
                    _emit("warn", "fleet", line)
                else:
                    _emit("info", "fleet", line)
    except Exception as exc:
        _emit("warn", "fleet", f"fleet registry unreadable: {exc}")

    if as_json:
        print(json.dumps({"ok": exit_code == 0, "findings": findings}, indent=2))
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


def find_dead_sessions() -> list[tuple[str, int]]:
    """Ray session dirs on this host other than the live one (by session_latest)."""
    from ..core.monitoring import _ray_session_log_dirs

    dead: list[tuple[str, int]] = []
    seen_roots: set[str] = set()
    for logs_dir in _ray_session_log_dirs():
        root = os.path.dirname(os.path.dirname(logs_dir))  # …/ray containing session_*
        if root in seen_roots:
            continue
        seen_roots.add(root)
        latest = os.path.realpath(os.path.join(root, "session_latest"))
        for entry in glob.glob(os.path.join(root, "session_*")):
            real = os.path.realpath(entry)
            if real == latest or os.path.islink(entry):
                continue
            dead.append((real, package_size_bytes(real)))
    return dead


@clean.command("sessions")
@click.option("--yes", is_flag=True, default=False, help="Actually delete (default: dry run).")
def clean_sessions(yes):
    """Delete dead Ray session directories (everything but session_latest)."""
    import shutil

    dead = find_dead_sessions()
    if not dead:
        info("no dead sessions.")
        return
    for path, size in dead:
        if yes:
            shutil.rmtree(path, ignore_errors=True)
            info(f"deleted {path} ({size / 1024**3:.1f} GB)")
        else:
            info(f"dead session: {path} ({size / 1024**3:.1f} GB)  [dry run — pass --yes to delete]")


@clean.command("all")
@click.option("--address", "-a", default=None, help="Ray dashboard address.")
@click.option("--max-gb", default=5.0, show_default=True, help="Component-log truncation threshold.")
@click.option("--yes", is_flag=True, default=False, help="Actually delete packages/sessions (default: dry run).")
@click.pass_context
def clean_all(ctx, address, max_gb, yes):
    """Everything: truncate oversized logs, then packages and dead sessions."""
    ctx.invoke(clean_raylet, max_gb=max_gb)
    ctx.invoke(clean_packages, address=address, yes=yes)
    ctx.invoke(clean_sessions, yes=yes)


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
