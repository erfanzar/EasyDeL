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

"""`eray tpu job` and `eray tpu sync` — one detached process per host of a TPU slice, without Ray.

Multi-host JAX programs (a serving world, a training run) start the same command on every host at once. Each host
keeps its job under ``<root>/<name>/``: ``cmd`` (the command line), ``log`` (stdout+stderr, previous run in
``log.1``), ``pid`` (the process-group leader; the job runs in its own session) and ``rc`` (exit code, written when
the command returns). ``stop`` signals the whole process group, so children go with it.

With the TPU check on (the default), ``start`` refuses unless no process on any host holds the TPU devices, checked on
every host before anything starts; ``stop`` reports TPU holders left behind. Because nothing holds the chips once that
check passes, a ``/tmp/libtpu_lockfile`` that this user cannot write (left by another user's finished job) is bypassed
with ``ALLOW_MULTIPLE_LIBTPU_LOAD=1`` instead of failing libtpu startup.
"""

from __future__ import annotations

import concurrent.futures
import getpass
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import click

from .utils import TpuInfo, _local_ips, error, info, print_table, run_on_all_hosts, run_on_host, success

DEFAULT_ROOT = "~/.eray/hostjobs"
DEFAULT_DEVICE_GLOB = "/dev/vfio/*"
_MARK = "ERAY_JOB"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GLOB_RE = re.compile(r"^[A-Za-z0-9_./*-]+$")
_START_SETTLE_S = 0.5


def shell_path(path: str) -> str:
    """Quote ``path`` for bash, expanding a leading ``~/`` (or a bare ``~``) to ``$HOME``.

    Raises:
        ValueError: For a relative path.
    """
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    if path.startswith("/"):
        return shlex.quote(path)
    raise ValueError(f"path must be absolute or start with ~/, got {path!r}")


def job_dir(root: str, name: str) -> str:
    """Bash expression for the directory of job ``name`` under ``root``.

    Raises:
        ValueError: For an invalid job name or a relative root.
    """
    if not _NAME_RE.match(name):
        raise ValueError(f"job name must match {_NAME_RE.pattern}, got {name!r}")
    return shell_path(f"{root.rstrip('/')}/{name}")


def _holders(device_glob: str) -> str:
    """Bash that sets ``holders`` to the comma-separated PIDs holding ``device_glob`` (empty when free).

    ``sudo -n`` sees every user's processes; without passwordless sudo only this user's are visible.
    """
    if not _GLOB_RE.match(device_glob):
        raise ValueError(f"device glob may contain only [A-Za-z0-9_./*-], got {device_glob!r}")
    probe = f"sudo -n fuser {device_glob} 2>/dev/null || fuser {device_glob} 2>/dev/null"
    return f"holders=$( {{ {probe}; }} | tr -s ' \\t' '\\n' | grep -E '^[0-9]+$' | sort -un | paste -sd, -)\n"


def _running_guard(directory: str) -> str:
    return (
        f"d={directory}\n"
        'pid=$(cat "$d/pid" 2>/dev/null)\n'
        'if [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null; then\n'
        f'  echo "{_MARK} state=running pid=$pid error=already-running"; exit 3\n'
        "fi\n"
    )


def _tpu_guard(device_glob: str) -> str:
    return (
        _holders(device_glob)
        + f'if [ -n "$holders" ]; then echo "{_MARK} state=blocked error=tpu-busy holders=$holders"; exit 4; fi\n'
    )


def preflight_script(
    name: str, *, root: str = DEFAULT_ROOT, tpu_check: bool = True, device_glob: str = DEFAULT_DEVICE_GLOB
) -> str:
    """Bash that exits 0 (``state=ready``) when ``name`` can start on this host."""
    script = _running_guard(job_dir(root, name))
    if tpu_check:
        script += _tpu_guard(device_glob)
    return script + f'echo "{_MARK} state=ready"\n'


def start_script(
    name: str,
    command: Sequence[str],
    *,
    root: str = DEFAULT_ROOT,
    cwd: str = "~",
    env: Mapping[str, str] | None = None,
    tpu_check: bool = True,
    device_glob: str = DEFAULT_DEVICE_GLOB,
) -> str:
    """Bash that starts ``command`` detached in its own session as job ``name``.

    Raises:
        ValueError: For an empty command, an invalid environment key, name, root or cwd.
    """
    if not command:
        raise ValueError("a job needs a command")
    env = dict(env or {})
    bad = [key for key in env if not _ENV_KEY_RE.match(key)]
    if bad:
        raise ValueError(f"invalid environment variable names {bad}")
    directory = job_dir(root, name)
    env_words = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
    cmd_words = " ".join(shlex.quote(str(part)) for part in command)
    inner = 'd=$1; shift; echo $$ > "$d/pid"; "$@"; rc=$?; echo "$rc" > "$d/rc"; exit "$rc"'
    script = _running_guard(directory)
    script += "extra=\n"
    if tpu_check:
        script += _tpu_guard(device_glob)
        # Nothing holds the chips (checked above), so another user's stale lockfile must not block libtpu.
        script += (
            "if [ -e /tmp/libtpu_lockfile ] && [ ! -w /tmp/libtpu_lockfile ]; then "
            "extra=ALLOW_MULTIPLE_LIBTPU_LOAD=1; fi\n"
        )
    script += (
        f'mkdir -p "$d" || {{ echo "{_MARK} state=failed error=mkdir"; exit 2; }}\n'
        'if [ -e "$d/log" ]; then mv -f "$d/log" "$d/log.1"; fi\n'
        'rm -f "$d/rc" "$d/pid"\n'
        f"printf '%s\\n' {shlex.quote(cmd_words)} > \"$d/cmd\"\n"
        f'cd {shell_path(cwd)} || {{ echo "{_MARK} state=failed error=cwd"; exit 5; }}\n'
        f'setsid nohup bash -c {shlex.quote(inner)} eray-job "$d" env $extra {env_words} {cmd_words} '
        '> "$d/log" 2>&1 < /dev/null &\n'
        'i=0; while [ ! -s "$d/pid" ] && [ "$i" -lt 100 ]; do sleep 0.05; i=$((i + 1)); done\n'
        f'if [ ! -s "$d/pid" ]; then echo "{_MARK} state=failed error=no-pid"; exit 6; fi\n'
        f"sleep {_START_SETTLE_S}\n"
        f'if [ -s "$d/rc" ]; then echo "{_MARK} state=exited pid=$(cat "$d/pid") rc=$(cat "$d/rc")"; exit 6; fi\n'
        f'echo "{_MARK} state=started pid=$(cat "$d/pid")"\n'
    )
    return script


def status_script(name: str, *, root: str = DEFAULT_ROOT) -> str:
    """Bash that reports ``running``, ``exited`` (with ``rc``), ``stopped`` or ``absent``."""
    return (
        f"d={job_dir(root, name)}\n"
        f'if [ ! -d "$d" ]; then echo "{_MARK} state=absent"; exit 0; fi\n'
        'pid=$(cat "$d/pid" 2>/dev/null); rc=$(cat "$d/rc" 2>/dev/null)\n'
        'if [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null; then state=running\n'
        'elif [ -n "$rc" ]; then state=exited\n'
        "else state=stopped; fi\n"
        f'echo "{_MARK} state=$state pid=$pid rc=$rc"\n'
    )


def stop_script(
    name: str,
    *,
    root: str = DEFAULT_ROOT,
    grace_s: float = 30.0,
    kill_wait_s: float = 30.0,
    tpu_check: bool = True,
    device_glob: str = DEFAULT_DEVICE_GLOB,
) -> str:
    """Bash that sends SIGTERM to the job's process group, then SIGKILL after ``grace_s`` seconds.

    After SIGKILL it waits up to ``kill_wait_s`` seconds for the group to disappear: a process tearing down a TPU
    runtime can take many seconds to exit even after SIGKILL. Exits 7 when the group is still there after that, 8
    when (with the TPU check) the TPU devices are still held.
    """
    ticks = max(1, int(float(grace_s) * 10))
    kill_ticks = max(1, int(float(kill_wait_s) * 10))
    script = (
        f"d={job_dir(root, name)}\n"
        'pid=$(cat "$d/pid" 2>/dev/null); forced=0\n'
        'if [ -n "$pid" ] && kill -0 -- "-$pid" 2>/dev/null; then\n'
        '  kill -TERM -- "-$pid" 2>/dev/null\n'
        f'  i=0; while kill -0 -- "-$pid" 2>/dev/null && [ "$i" -lt {ticks} ]; do sleep 0.1; i=$((i + 1)); done\n'
        '  if kill -0 -- "-$pid" 2>/dev/null; then\n'
        '    forced=1; kill -KILL -- "-$pid" 2>/dev/null\n'
        f'    i=0; while kill -0 -- "-$pid" 2>/dev/null && [ "$i" -lt {kill_ticks} ]; do sleep 0.1; i=$((i + 1)); done\n'
        "  fi\n"
        '  if kill -0 -- "-$pid" 2>/dev/null; then\n'
        f'    echo "{_MARK} state=running pid=$pid error=survived-kill"; exit 7\n'
        "  fi\n"
        "fi\n"
        "holders=\n"
    )
    if tpu_check:
        script += _holders(device_glob)
        script += (
            f'if [ -n "$holders" ]; then echo "{_MARK} state=stopped pid=$pid forced=$forced '
            'error=tpu-still-held holders=$holders"; exit 8; fi\n'
        )
    return script + f'echo "{_MARK} state=stopped pid=$pid forced=$forced"\n'


def logs_script(name: str, *, root: str = DEFAULT_ROOT, lines: int = 50) -> str:
    """Bash that prints the last ``lines`` lines of the job's log."""
    return f"tail -n {max(1, int(lines))} {job_dir(root, name)}/log 2>&1\n"


def parse_report(stdout: str) -> dict[str, str]:
    """``key=value`` fields of the last ``ERAY_JOB`` line in ``stdout`` (empty when there is none)."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(_MARK + " "):
            return dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
    return {}


@dataclass(frozen=True)
class HostReport:
    """One host's outcome."""

    host: int
    ip: str
    returncode: int
    fields: dict[str, str] = field(default_factory=dict)
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def row(self) -> dict[str, str]:
        detail = self.fields.get("error", "")
        if self.fields.get("holders"):
            detail = f"{detail} holders={self.fields['holders']}".strip()
        if not self.fields and self.stderr:
            detail = self.stderr.strip().splitlines()[-1][:80]
        return {
            "host": str(self.host),
            "ip": self.ip,
            "state": self.fields.get("state", "unreachable" if not self.fields else "?"),
            "pid": self.fields.get("pid", ""),
            "rc": self.fields.get("rc", ""),
            "detail": detail,
        }


def run_everywhere(
    tpu: TpuInfo, script: str, *, timeout: int = 120, user: str | None = None, ssh_key: str | None = None
) -> list[HostReport]:
    """Run a bash ``script`` on every host in parallel and parse each ``ERAY_JOB`` report."""
    results = run_on_all_hosts(tpu, "bash -c " + shlex.quote(script), timeout=timeout, user=user, ssh_key=ssh_key)
    return [
        HostReport(
            host, tpu.internal_ips[host], result.returncode, parse_report(result.stdout or ""), result.stderr or ""
        )
        for host, result in results
    ]


def sync_tree(
    tpu: TpuInfo,
    src: str,
    dest: str,
    *,
    user: str,
    ssh_key: str,
    excludes: Sequence[str] = (),
    delete: bool = False,
    timeout: int = 900,
) -> list[tuple[int, subprocess.CompletedProcess]]:
    """rsync directory ``src`` to ``dest`` on every host of the slice.

    On this host the copy is local, and it is skipped only when ``dest`` is ``src`` itself (the tree is already
    there); a different ``dest`` is copied here too, so every host ends up with the same files.
    """
    ssh = f"ssh -i {shlex.quote(ssh_key)} -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15"
    local = _local_ips()
    same_path = os.path.realpath(os.path.expanduser(src)) == os.path.realpath(os.path.expanduser(dest))
    targets = [(index, ip) for index, ip in enumerate(tpu.internal_ips) if ip not in local or not same_path]

    def push(index: int, ip: str) -> tuple[int, subprocess.CompletedProcess]:
        options = [f"--exclude={pattern}" for pattern in excludes] + (["--delete"] if delete else [])
        if ip in local:
            target = os.path.expanduser(dest)
            os.makedirs(target, exist_ok=True)
            cmd = ["rsync", "-a", *options, src.rstrip("/") + "/", target.rstrip("/") + "/"]
        else:
            cmd = ["rsync", "-a", "-e", ssh, "--rsync-path", f"mkdir -p {shlex.quote(dest)} && rsync", *options]
            cmd += [src.rstrip("/") + "/", f"{user}@{ip}:{dest.rstrip('/')}/"]
        return index, subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(targets))) as pool:
        done = [future.result() for future in [pool.submit(push, index, ip) for index, ip in targets]]
    return sorted(done, key=lambda item: item[0])


def _parse_env(pairs: Sequence[str]) -> dict[str, str]:
    env = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not _ENV_KEY_RE.match(key):
            raise click.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--env")
        env[key] = value
    return env


def register(
    tpu_group: click.Group,
    *,
    resolve_tpu: Callable[..., tuple[TpuInfo, str | None, str | None]],
    target_options: Sequence[Callable],
) -> None:
    """Attach ``job`` and ``sync`` to the ``eray tpu`` group.

    Args:
        tpu_group: The ``tpu`` click group.
        resolve_tpu: ``(tpu_name, project, zone, ips, tpu_type, user, ssh_key) -> (TpuInfo, user, ssh_key)``.
        target_options: Click options providing those seven parameters.
    """

    def with_target(fn):
        for option in reversed(target_options):
            fn = option(fn)
        return fn

    def target(tpu_name, project, zone, ips, tpu_type, user, ssh_key):
        tpu, user, ssh_key = resolve_tpu(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
        if ssh_key and tpu.is_gcloud_managed:
            # An SSH key means plain SSH to the internal IPs instead of `gcloud compute tpus tpu-vm ssh`.
            tpu = TpuInfo.from_ips(tpu.internal_ips, tpu.accelerator_type, state=tpu.state)
        if ssh_key and not user:
            user = getpass.getuser()
        return tpu, user, ssh_key

    def report(rows: list[HostReport], what: str) -> None:
        print_table([row.row() for row in rows])
        failed = [row.host for row in rows if not row.ok]
        if failed:
            raise click.ClickException(f"{what} failed on hosts {failed}")

    name_opt = click.option("--name", required=True, help="Job name; one job of a name per host.")
    root_opt = click.option("--root", default=DEFAULT_ROOT, show_default=True, help="Per-host job directory root.")
    tpu_opt = click.option(
        "--tpu-check/--no-tpu-check", default=True, show_default=True, help="Require (and report) free TPU devices."
    )
    glob_opt = click.option("--device-glob", default=DEFAULT_DEVICE_GLOB, hidden=True)

    @tpu_group.group(name="job")
    def job() -> None:
        """Detached processes on every host of a TPU slice (start, stop, status, logs)."""

    @job.command(context_settings={"ignore_unknown_options": True, "allow_interspersed_args": False})
    @with_target
    @name_opt
    @root_opt
    @tpu_opt
    @glob_opt
    @click.option("--cwd", default="~", show_default=True, help="Working directory on each host.")
    @click.option("--env", "env_pairs", multiple=True, help="KEY=VALUE set for the command; repeatable.")
    @click.option("--timeout", default=120, show_default=True, help="Per-host SSH timeout (seconds).")
    @click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
    def start(
        tpu_name,
        project,
        zone,
        ips,
        tpu_type,
        user,
        ssh_key,
        name,
        root,
        tpu_check,
        device_glob,
        cwd,
        env_pairs,
        timeout,
        command,
    ):
        """Start COMMAND on every host (after `--`), only when every host is ready.

        \b
        Example:
            eray tpu job start --name esurge --ssh-key ~/.ssh/google_compute_engine \\
              -- ~/venv/bin/python -m easydel.inference.esurge.serving serve --config ~/serve.yaml
        """
        tpu, user, ssh_key = target(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
        env = _parse_env(env_pairs)
        info(f"checking {tpu.num_hosts} host(s) before starting '{name}'")
        ready = run_everywhere(
            tpu,
            preflight_script(name, root=root, tpu_check=tpu_check, device_glob=device_glob),
            timeout=timeout,
            user=user,
            ssh_key=ssh_key,
        )
        if not all(row.ok for row in ready):
            report(ready, f"start '{name}' (nothing started; preflight)")
        script = start_script(name, command, root=root, cwd=cwd, env=env, tpu_check=tpu_check, device_glob=device_glob)
        started = run_everywhere(tpu, script, timeout=timeout, user=user, ssh_key=ssh_key)
        if not all(row.ok for row in started):
            error(f"'{name}' failed to start on some hosts; stopping it everywhere")
            run_everywhere(
                tpu,
                stop_script(name, root=root, grace_s=5, tpu_check=False),
                timeout=timeout,
                user=user,
                ssh_key=ssh_key,
            )
        report(started, f"start '{name}'")
        success(f"'{name}' started on {tpu.num_hosts} host(s)")

    @job.command()
    @with_target
    @name_opt
    @root_opt
    @click.option("--timeout", default=60, show_default=True)
    def status(tpu_name, project, zone, ips, tpu_type, user, ssh_key, name, root, timeout):
        """State of job NAME on every host: running, exited (rc), stopped or absent."""
        tpu, user, ssh_key = target(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
        report(
            run_everywhere(tpu, status_script(name, root=root), timeout=timeout, user=user, ssh_key=ssh_key), "status"
        )

    @job.command()
    @with_target
    @name_opt
    @root_opt
    @tpu_opt
    @glob_opt
    @click.option("--grace", default=30.0, show_default=True, help="Seconds between SIGTERM and SIGKILL.")
    @click.option(
        "--kill-wait", default=30.0, show_default=True, help="Seconds to wait for the group to exit after SIGKILL."
    )
    @click.option("--timeout", default=180, show_default=True)
    def stop(
        tpu_name, project, zone, ips, tpu_type, user, ssh_key, name, root, tpu_check, device_glob, grace, kill_wait,
        timeout,
    ):  # fmt: skip
        """Stop job NAME's process group on every host and confirm the TPU devices are free."""
        tpu, user, ssh_key = target(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
        script = stop_script(
            name, root=root, grace_s=grace, kill_wait_s=kill_wait, tpu_check=tpu_check, device_glob=device_glob
        )
        report(run_everywhere(tpu, script, timeout=timeout, user=user, ssh_key=ssh_key), f"stop '{name}'")
        success(f"'{name}' stopped on {tpu.num_hosts} host(s)")

    @job.command()
    @with_target
    @name_opt
    @root_opt
    @click.option("--host", "host_index", default=0, show_default=True, help="Host index (worker order).")
    @click.option("--lines", "-n", default=50, show_default=True)
    def logs(tpu_name, project, zone, ips, tpu_type, user, ssh_key, name, root, host_index, lines):
        """Print the tail of job NAME's log on one host."""
        tpu, user, ssh_key = target(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
        if not 0 <= host_index < tpu.num_hosts:
            raise click.BadParameter(f"host must be in 0..{tpu.num_hosts - 1}", param_hint="--host")
        result = run_on_host(
            tpu,
            host_index,
            "bash -c " + shlex.quote(logs_script(name, root=root, lines=lines)),
            timeout=60,
            user=user,
            ssh_key=ssh_key,
        )
        click.echo(result.stdout, nl=False)
        if result.returncode != 0:
            raise click.ClickException(f"cannot read the log on host {host_index}: {result.stderr.strip()}")

    @tpu_group.command()
    @with_target
    @click.argument("src")
    @click.argument("dest")
    @click.option("--exclude", "excludes", multiple=True, help="rsync exclude pattern; repeatable.")
    @click.option("--delete", is_flag=True, help="Delete files on the hosts that are absent from SRC.")
    def sync(tpu_name, project, zone, ips, tpu_type, user, ssh_key, src, dest, excludes, delete):
        """rsync directory SRC to DEST on every other host of the slice (needs --ssh-key)."""
        tpu, user, ssh_key = target(tpu_name, project, zone, ips, tpu_type, user, ssh_key)
        if not ssh_key:
            raise click.ClickException("sync uses plain SSH; pass --ssh-key")
        results = sync_tree(tpu, src, dest, user=user, ssh_key=ssh_key, excludes=excludes, delete=delete)
        failed = [index for index, result in results if result.returncode != 0]
        for index, result in results:
            if result.returncode != 0:
                error(f"host {index}: {result.stderr.strip()[-300:]}")
        if failed:
            raise click.ClickException(f"sync failed on hosts {failed}")
        success(f"synced {src} to {dest} on {len(results)} host(s)")
