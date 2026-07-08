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


"""Tracked background port-forwards, the primitive under `eray dashboard`.

Local-machine-only bookkeeping — this never syncs across machines (that's
what the fleet registry's GCS backend is for). A single small JSON file
records what eray itself has forwarded, so asking to open a dashboard twice
detects and reuses the existing forward instead of colliding on the port —
`ray dashboard` itself has no memory of what it already has listening, which
is the failure mode this module exists to fix.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .registry import LocalBackend

STORE_PATH = Path("~/.eray/tunnels.json").expanduser()
LOG_DIR = Path("~/.eray/tunnel-logs").expanduser()

#: A live PID whose real start time differs from the one we recorded at spawn
#: by more than this (seconds) is a *different* process that inherited the id
#: (PID reuse — most likely across a reboot, since the store persists). The
#: window only has to absorb the gap between Popen and our `started_ts` stamp.
_PID_START_SLACK_S = 30.0

#: How long to wait after spawning a forwarder before deciding it survived
#: startup. A fast failure (bad config/auth/host) exits well within this; a
#: healthy forwarder is still negotiating the SSH connection (~15-20s).
_STARTUP_PROBE_S = 1.0


@dataclass(frozen=True)
class TunnelSession:
    """One tracked background port-forward.

    Attributes:
        name: The cluster/profile name this tunnel forwards to (also the
            fleet registry key it corresponds to).
        kind: ``"qr"`` (fleet, gcloud TPU SSH) or ``"launcher"`` (autoscale,
            ``ray dashboard``).
        pid: Process id of the background forwarder.
        local_port: Local port serving the forward.
        remote_port: Remote port being forwarded (Ray dashboard: 8265).
        started_ts: Unix timestamp the tunnel was opened.
    """

    name: str
    kind: str
    pid: int
    local_port: int
    remote_port: int
    started_ts: float

    def to_dict(self) -> dict:
        """Serialize for the tunnels store."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TunnelSession:
        """Deserialize one entry from the tunnels store.

        Args:
            data: One session dict as persisted by `to_dict`.

        Returns:
            The session.
        """
        return cls(
            name=data["name"],
            kind=data["kind"],
            pid=int(data["pid"]),
            local_port=int(data["local_port"]),
            remote_port=int(data["remote_port"]),
            started_ts=float(data["started_ts"]),
        )


def _proc_start_ts(pid: int) -> float | None:
    """The process's real start time (unix seconds), or None if unknown.

    Used to tell a forwarder we launched apart from an unrelated process
    that later inherited the same id. Best-effort: `psutil` ships with
    ``ray[default]`` (eray always has Ray), but if it is somehow absent the
    identity check is simply skipped rather than failing liveness probes.

    Args:
        pid: Process id to query.

    Returns:
        The process creation time, or None if psutil is unavailable or the
        process could not be inspected.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _pid_alive(pid: int, *, started_ts: float | None = None) -> bool:
    """Whether the tracked forwarder for a PID is still running.

    Args:
        pid: Process id to probe.
        started_ts: When we recorded spawning this forwarder. When given, a
            live PID whose real start time is more than `_PID_START_SLACK_S`
            away is rejected as a *different* process that inherited the id
            (PID reuse, e.g. across a reboot — the store persists), so we
            never mistake it for our tunnel and, crucially, never signal it.

    Returns:
        True if the process is still running and (when `started_ts` is
        given) is plausibly the forwarder we launched; False if it's gone,
        unsignalable, or a recycled id.
    """
    # A terminated-but-unreaped direct child is a zombie: it still answers
    # the signal-0 probe below even though it's dead (matters when `pid` was
    # opened by *this* process, e.g. tests). Reap it first. When `pid` isn't
    # our child at all (the common case — a separate `eray` invocation
    # checking a tunnel opened by an earlier one), waitpid raises
    # ChildProcessError and we fall through to the signal probe.
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if started_ts is not None:
        start = _proc_start_ts(pid)
        if start is not None and abs(start - started_ts) > _PID_START_SLACK_S:
            return False
    return True


def _store() -> LocalBackend:
    """A `LocalBackend` bound to the current `STORE_PATH`.

    Built fresh per call (not cached at import time) so tests that
    monkeypatch the `STORE_PATH` module attribute take effect immediately.
    The flat ``{name: session-dict}`` shape (`empty_doc=dict`) is distinct
    from the fleet registry's own document shape — this reuses only the
    flock + atomic-rename read/write primitive, not registry semantics.
    """
    return LocalBackend(STORE_PATH, empty_doc=dict)


def find_free_port(preferred: int = 8265, *, exclude: tuple[int, ...] = ()) -> int:
    """Pick a local port to bind a new tunnel to.

    Args:
        preferred: Tried first (the Ray dashboard default, 8265).
        exclude: Ports to refuse even if free — pass the port(s) already
            chosen for the same forwarder so the dashboard and GCS forwards
            (bind-then-close probes moments apart) can't be handed the same
            just-released ephemeral port and then collide on one ``-L``.

    Returns:
        `preferred` if it's free and not excluded, else an OS-assigned free
        port not in `exclude`.

    Raises:
        RuntimeError: If no local port could be bound at all.
    """
    excluded = set(exclude)
    # preferred first (unless already claimed), then several ephemeral tries:
    # the kernel can re-hand a just-released ephemeral port, so re-rolling on
    # a collision with `excluded` is what keeps two back-to-back calls apart.
    candidates = ([] if preferred in excluded else [preferred]) + [0] * 8
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            chosen = sock.getsockname()[1]
            if chosen in excluded:
                continue
            return chosen
    raise RuntimeError("could not bind any local port")


def open_tunnel(name: str, argv: list[str], *, kind: str, local_port: int, remote_port: int = 8265) -> TunnelSession:
    """Spawn a background port-forward and track it.

    Args:
        name: Cluster/profile name (the tunnels-store key).
        argv: Full command to run (a gcloud SSH forward, or `ray dashboard`).
        kind: ``"qr"`` or ``"launcher"`` — informational, shown by `eray
            dashboard ls`.
        local_port: Local port the forward binds.
        remote_port: Remote port being forwarded.

    Returns:
        The tracked session. This does not check for an existing live
        tunnel under `name` first — callers (`eray dashboard open`) decide
        whether to reuse one via `get_tunnel` before calling this.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    with open(log_path, "a") as log_file:
        process = subprocess.Popen(
            argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # A forwarder that fails fast (bad config path, auth/host error) exits
    # within about a second; catch that here instead of recording a dead pid
    # and reporting "opening" for a tunnel that will never answer.
    time.sleep(_STARTUP_PROBE_S)
    if process.poll() is not None:
        raise RuntimeError(
            f"port-forward for {name!r} exited immediately (code {process.returncode}); see {log_path}"
        )
    session = TunnelSession(
        name=name,
        kind=kind,
        pid=process.pid,
        local_port=local_port,
        remote_port=remote_port,
        started_ts=time.time(),
    )
    try:
        _store().update(lambda doc: doc.__setitem__(name, session.to_dict()))
    except Exception:
        # Don't leak the forwarder we just spawned if we can't record it.
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        raise
    return session


def register_alias(name: str, *, pid: int, kind: str, local_port: int, remote_port: int) -> TunnelSession:
    """Track an extra port forwarded by an already-running tunnel process.

    A single SSH process can forward several ports (e.g. the Ray dashboard
    *and* GCS ports over one connection — one SSH avoids the ControlMaster
    collision two concurrent ``ray dashboard`` processes hit on a
    cluster-launcher cluster). Each forwarded port gets its own store entry
    for lookup (`tunnels_for_remote_port`) and display, all sharing the
    forwarder's pid — so liveness and `stop_tunnel` track them together
    (killing the pid drops every forward, and the stale sibling entries
    self-heal on the next `list_tunnels`).

    Args:
        name: Store key for this alias (e.g. ``"<cluster>-gcs"``).
        pid: The already-running forwarder's process id.
        kind: ``"qr"`` or ``"launcher"``.
        local_port: Local port this entry represents.
        remote_port: Remote port this entry represents.

    Returns:
        The tracked alias session.
    """
    session = TunnelSession(
        name=name, kind=kind, pid=pid, local_port=local_port, remote_port=remote_port, started_ts=time.time()
    )
    _store().update(lambda doc: doc.__setitem__(name, session.to_dict()))
    return session


def stop_tunnel(name: str) -> bool:
    """Close a tracked tunnel.

    Args:
        name: Cluster/profile name.

    Returns:
        True if a live tunnel was found and signaled; False if none was
        tracked (or it had already died). Either way the store entry is
        removed.
    """
    result = {"stopped": False}

    def mutate(doc: dict) -> None:
        entry = doc.pop(name, None)
        if entry is None:
            return
        try:
            pid = int(entry["pid"])
            started_ts = float(entry["started_ts"])
        except (KeyError, TypeError, ValueError):
            return
        if _pid_alive(pid, started_ts=started_ts):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except OSError:
                pass
            result["stopped"] = True

    _store().update(mutate)
    return bool(result["stopped"])


def list_tunnels() -> dict[str, TunnelSession]:
    """Every tracked tunnel whose process is still alive.

    Self-heals: entries whose PID is no longer running are dropped from the
    store as a side effect of listing, so a tunnel killed from outside eray
    doesn't linger as a phantom "already open". The common case (every
    tracked tunnel still alive) is a plain unlocked read — the atomic
    tmp+rename write in `LocalBackend` guarantees a reader never sees a
    torn file, so no lock is needed just to check liveness; the flock +
    write path only runs when there's actually something to prune, so a
    quick `eray dashboard ls` doesn't take a write lock and touch disk on
    every invocation.

    Returns:
        Mapping of name -> live session.
    """
    store = _store()
    doc, _ = store.read()
    live: dict[str, TunnelSession] = {}
    dead: list[str] = []
    for name, raw in doc.items():
        try:
            session = TunnelSession.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            dead.append(name)
            continue
        if _pid_alive(session.pid, started_ts=session.started_ts):
            live[name] = session
        else:
            dead.append(name)

    if dead:
        # Re-verify each candidate *under the lock* against the entry as it
        # stands now, not the (unlocked) snapshot `dead` was computed from:
        # a concurrent `open_tunnel` may have re-created this name as a live
        # forwarder in the meantime, and popping it by stale name would
        # orphan that tunnel. Only drop entries that are still dead/garbage.
        dead_set = set(dead)

        def prune(d: dict) -> None:
            for name in list(d):
                if name not in dead_set:
                    continue
                try:
                    session = TunnelSession.from_dict(d[name])
                except (KeyError, TypeError, ValueError):
                    d.pop(name, None)
                    continue
                if not _pid_alive(session.pid, started_ts=session.started_ts):
                    d.pop(name, None)

        store.update(prune)
    return live


def get_tunnel(name: str) -> TunnelSession | None:
    """One tracked tunnel, if it's still alive.

    Args:
        name: Cluster/profile name.

    Returns:
        The session, or None if untracked or its process has died.
    """
    return list_tunnels().get(name)


def tunnels_for_remote_port(remote_port: int) -> list[TunnelSession]:
    """Live tracked tunnels forwarding a given remote port.

    The `remote_port` identifies what a tunnel *is*, independent of its
    (possibly OS-assigned) local port or its store key: 8265 → a Ray
    dashboard / Jobs API tunnel, 6379 (`RAY_HEAD_PORT`) → a GCS tunnel.
    Lets `eray resources`/`eray status` auto-resolve their address from an
    already-open tunnel instead of the operator hand-typing
    ``-a 127.0.0.1:<port>``.

    Args:
        remote_port: The forwarded remote port to match.

    Returns:
        Matching live sessions (possibly empty), sorted by name.
    """
    return sorted(
        (s for s in list_tunnels().values() if s.remote_port == remote_port),
        key=lambda s: s.name,
    )


__all__ = [
    "LOG_DIR",
    "STORE_PATH",
    "TunnelSession",
    "find_free_port",
    "get_tunnel",
    "list_tunnels",
    "open_tunnel",
    "register_alias",
    "stop_tunnel",
    "tunnels_for_remote_port",
]
