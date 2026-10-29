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

"""Tests for the tracked background port-forwards under `eray dashboard`."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

import eray.provision.tunnel as tunnel_module

SLEEP_ARGV = [sys.executable, "-c", "import time; time.sleep(30)"]
EXIT_IMMEDIATELY_ARGV = [sys.executable, "-c", "pass"]

# The tunnel store is isolated per-test by the autouse fixture in conftest.py.


def _wait_until(predicate, *, timeout=3.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class TestFindFreePort:
    def test_returns_preferred_when_free(self):
        # A high, unlikely-to-collide port for a deterministic assertion.
        assert tunnel_module.find_free_port(59123) == 59123

    def test_falls_back_when_preferred_is_taken(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        taken_port = holder.getsockname()[1]
        try:
            port = tunnel_module.find_free_port(taken_port)
            assert port != taken_port
            assert port > 0
        finally:
            holder.close()

    def test_never_returns_an_excluded_port(self):
        # The dashboard + GCS forwards are picked back-to-back for one SSH;
        # excluding the first pick keeps the second off the same local port.
        first = tunnel_module.find_free_port(59321)
        second = tunnel_module.find_free_port(59321, exclude=(first,))
        assert second != first


class TestPidIdentity:
    def test_live_pid_with_matching_start_is_alive(self):
        proc = subprocess.Popen(SLEEP_ARGV, start_new_session=True)
        try:
            start = tunnel_module._proc_start_ts(proc.pid)
            assert start is not None  # psutil ships with ray[default]
            assert tunnel_module._pid_alive(proc.pid, started_ts=start) is True
        finally:
            proc.kill()
            proc.wait()

    def test_live_pid_with_mismatched_start_is_rejected(self):
        # Simulates PID reuse: the id is live, but the recorded spawn time is
        # far from the process's real start, so it is not our forwarder and
        # must not be treated as alive (and never signalled by stop_tunnel).
        proc = subprocess.Popen(SLEEP_ARGV, start_new_session=True)
        try:
            real_start = tunnel_module._proc_start_ts(proc.pid)
            stale_ts = real_start + tunnel_module._PID_START_SLACK_S + 3600
            assert tunnel_module._pid_alive(proc.pid, started_ts=stale_ts) is False
        finally:
            proc.kill()
            proc.wait()


class TestOpenGetStopTunnel:
    def test_open_tracks_pid_and_ports(self):
        port = tunnel_module.find_free_port(59201)
        session = tunnel_module.open_tunnel("cluster-a", SLEEP_ARGV, kind="launcher", local_port=port)
        try:
            assert session.name == "cluster-a"
            assert session.kind == "launcher"
            assert session.local_port == port
            assert session.remote_port == 8265
            assert session.pid > 0

            got = tunnel_module.get_tunnel("cluster-a")
            assert got == session
            assert tunnel_module.list_tunnels() == {"cluster-a": session}
        finally:
            tunnel_module.stop_tunnel("cluster-a")

    def test_open_writes_a_log_file(self):
        port = tunnel_module.find_free_port(59202)
        tunnel_module.open_tunnel("cluster-log", SLEEP_ARGV, kind="launcher", local_port=port)
        try:
            assert (tunnel_module.LOG_DIR / "cluster-log.log").exists()
        finally:
            tunnel_module.stop_tunnel("cluster-log")

    def test_stop_terminates_the_process_and_untracks_it(self):
        port = tunnel_module.find_free_port(59203)
        session = tunnel_module.open_tunnel("cluster-b", SLEEP_ARGV, kind="qr", local_port=port)

        assert tunnel_module.stop_tunnel("cluster-b") is True
        assert _wait_until(lambda: tunnel_module.get_tunnel("cluster-b") is None)
        assert tunnel_module.list_tunnels() == {}

        # stop_tunnel only reaps a pid while it's still in the store (see
        # `_pid_alive`); this test process is the tunnel's real OS parent
        # (unlike a live CLI, where `open` and `stop` are separate
        # processes and the OS reaps on its own), so reap here to confirm
        # SIGTERM actually landed rather than the process being ignored.
        reaped_pid, status = os.waitpid(session.pid, 0)
        assert reaped_pid == session.pid
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGTERM

    def test_stop_unknown_name_returns_false(self):
        assert tunnel_module.stop_tunnel("never-opened") is False

    def test_get_tunnel_unknown_name_returns_none(self):
        assert tunnel_module.get_tunnel("never-opened") is None

    def test_stop_malformed_entry_is_dropped_not_raised(self):
        # A corrupt/foreign-schema entry (missing "pid") must be handled
        # the same defensive way list_tunnels/get_tunnel already do, not
        # crash the `eray dashboard stop` CLI command with a KeyError.
        tunnel_module._store().update(lambda doc: doc.__setitem__("broken", {"name": "broken"}))
        assert tunnel_module.stop_tunnel("broken") is False
        assert tunnel_module.get_tunnel("broken") is None


class TestSelfHealing:
    def test_open_raises_when_forwarder_exits_immediately(self):
        # A forwarder that dies at startup (bad config/auth) must not be
        # recorded and reported as "opening" a tunnel that never answers.
        port = tunnel_module.find_free_port(59204)
        try:
            tunnel_module.open_tunnel("dead-on-arrival", EXIT_IMMEDIATELY_ARGV, kind="launcher", local_port=port)
            raise AssertionError("expected open_tunnel to raise for an immediately-exiting forwarder")
        except RuntimeError:
            pass
        # Nothing tracked, so a later `ls`/`open` doesn't see a phantom.
        assert tunnel_module.get_tunnel("dead-on-arrival") is None
        assert tunnel_module.list_tunnels() == {}

    def test_list_tunnels_prunes_entries_that_die_after_being_recorded(self):
        # A forwarder recorded live but later killed (reboot, dropped SSH)
        # self-heals out of the store on the next list_tunnels.
        port = tunnel_module.find_free_port(59208)
        session = tunnel_module.open_tunnel("later-death", SLEEP_ARGV, kind="launcher", local_port=port)
        assert tunnel_module.get_tunnel("later-death") is not None
        os.killpg(os.getpgid(session.pid), signal.SIGKILL)

        assert _wait_until(lambda: tunnel_module.list_tunnels() == {})
        assert tunnel_module.get_tunnel("later-death") is None

    def test_list_tunnels_does_not_write_when_nothing_needs_pruning(self):
        # eray dashboard/ls calls list_tunnels on every invocation just to
        # check status; it must not take the flock+write path (disk I/O +
        # lock contention) when every tracked tunnel is still alive.
        port = tunnel_module.find_free_port(59207)
        tunnel_module.open_tunnel("stable", SLEEP_ARGV, kind="launcher", local_port=port)
        try:
            mtime_before = tunnel_module.STORE_PATH.stat().st_mtime_ns
            assert tunnel_module.list_tunnels() != {}
            assert tunnel_module.STORE_PATH.stat().st_mtime_ns == mtime_before
        finally:
            tunnel_module.stop_tunnel("stable")

    def test_reusing_a_name_overwrites_the_old_entry(self):
        port_a = tunnel_module.find_free_port(59205)
        tunnel_module.open_tunnel("reused", SLEEP_ARGV, kind="launcher", local_port=port_a)
        first_pid = tunnel_module.get_tunnel("reused").pid
        tunnel_module.stop_tunnel("reused")
        _wait_until(lambda: tunnel_module.get_tunnel("reused") is None)

        port_b = tunnel_module.find_free_port(59206)
        tunnel_module.open_tunnel("reused", SLEEP_ARGV, kind="launcher", local_port=port_b)
        try:
            second = tunnel_module.get_tunnel("reused")
            assert second.pid != first_pid
            assert second.local_port == port_b
        finally:
            tunnel_module.stop_tunnel("reused")


class TestTunnelsForRemotePort:
    def test_filters_by_remote_port(self):
        dash_port = tunnel_module.find_free_port(59301)
        tunnel_module.open_tunnel("c1", SLEEP_ARGV, kind="qr", local_port=dash_port, remote_port=8265)
        gcs_port = tunnel_module.find_free_port(59302)
        tunnel_module.open_tunnel("c1-gcs", SLEEP_ARGV, kind="qr", local_port=gcs_port, remote_port=6379)
        try:
            dash = tunnel_module.tunnels_for_remote_port(8265)
            gcs = tunnel_module.tunnels_for_remote_port(6379)
            assert [s.name for s in dash] == ["c1"]
            assert [s.name for s in gcs] == ["c1-gcs"]
            assert gcs[0].local_port == gcs_port
        finally:
            tunnel_module.stop_tunnel("c1")
            tunnel_module.stop_tunnel("c1-gcs")

    def test_empty_when_no_match(self):
        assert tunnel_module.tunnels_for_remote_port(6379) == []

    def test_sorted_by_name(self):
        pa = tunnel_module.find_free_port(59303)
        tunnel_module.open_tunnel("b-gcs", SLEEP_ARGV, kind="qr", local_port=pa, remote_port=6379)
        pb = tunnel_module.find_free_port(59304)
        tunnel_module.open_tunnel("a-gcs", SLEEP_ARGV, kind="qr", local_port=pb, remote_port=6379)
        try:
            assert [s.name for s in tunnel_module.tunnels_for_remote_port(6379)] == ["a-gcs", "b-gcs"]
        finally:
            tunnel_module.stop_tunnel("a-gcs")
            tunnel_module.stop_tunnel("b-gcs")
