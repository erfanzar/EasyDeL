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

"""Tests for the fleet registry (local + GCS CAS backends) and ensure_tpu."""

import json
import subprocess
import threading
from types import SimpleNamespace
from unittest import mock

import eray.provision.fleet as fleet_module
import eray.provision.registry as registry_module
import pytest
from click.testing import CliRunner
from eray.cli.main import cli
from eray.provision.registry import (
    ClusterRecord,
    ClusterRegistry,
    ConflictError,
    GcsBackend,
    LocalBackend,
)


def make_registry(tmp_path) -> ClusterRegistry:
    return ClusterRegistry(LocalBackend(tmp_path / "clusters.json"))


def make_record(**overrides) -> ClusterRecord:
    defaults = dict(name="trainer1", project="proj", zone="us-east5-a", accelerator_type="v5p-64")
    defaults.update(overrides)
    return ClusterRecord(**defaults)


class TestLocalRegistry:
    def test_roundtrip(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.upsert(make_record(bootstrap_cmd="setup.sh"))
        loaded = reg.get("trainer1")
        assert loaded.accelerator_type == "v5p-64"
        assert loaded.bootstrap_cmd == "setup.sh"
        assert loaded.desired_state == "up"
        assert reg.remove("trainer1") is True
        assert reg.get("trainer1") is None

    def test_mutate_record_and_next_qr_id(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.upsert(make_record())

        def bump(r: ClusterRecord) -> None:
            r.generation += 1
            r.qr_id = r.name + f"-r{r.generation}"

        reg.mutate_record("trainer1", bump)
        loaded = reg.get("trainer1")
        assert loaded.generation == 1
        assert loaded.qr_id == "trainer1-r1"
        assert loaded.next_qr_id() == "trainer1-r2"

    def test_mutate_unknown_raises(self, tmp_path):
        with pytest.raises(KeyError, match="not registered"):
            make_registry(tmp_path).mutate_record("ghost", lambda r: None)

    def test_forward_compat_unknown_fields_preserved(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.upsert(make_record())
        # Simulate a newer eray writing an unknown field.
        doc, token = reg.backend.read()
        doc["clusters"]["trainer1"]["future_field"] = {"x": 1}
        reg.backend.write(doc, token)
        loaded = reg.get("trainer1")
        assert loaded.extra["future_field"] == {"x": 1}

    def test_concurrent_mutations_all_applied(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.upsert(make_record(generation=0))

        def bump():
            reg.mutate_record("trainer1", lambda r: setattr(r, "generation", r.generation + 1))

        threads = [threading.Thread(target=bump) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert reg.get("trainer1").generation == 16


class FakeGcs:
    """In-memory GCS object with generation semantics for CAS tests."""

    def __init__(self):
        self.body: str | None = None
        self.generation = 0
        self.cp_calls: list[list[str]] = []
        self.fail_next_cp = 0

    def gcloud_json(self, args, **kwargs):
        assert args[:3] == ["storage", "objects", "describe"]
        if self.body is None:
            raise subprocess.CalledProcessError(1, "gcloud", stderr="ERROR: 404 not found")
        return {"generation": str(self.generation)}

    def gcloud(self, args, **kwargs):
        if args[:2] == ["storage", "cat"]:
            if self.body is None:
                raise subprocess.CalledProcessError(1, "gcloud", stderr="No URLs matched")
            return self.body
        if args[:2] == ["storage", "cp"]:
            self.cp_calls.append(list(args))
            match = next(a for a in args if a.startswith("--if-generation-match="))
            expected = int(match.split("=", 1)[1])
            if self.fail_next_cp > 0:
                self.fail_next_cp -= 1
                raise subprocess.CalledProcessError(1, "gcloud", stderr="ERROR: 412 Precondition Failed")
            if expected != self.generation:
                raise subprocess.CalledProcessError(1, "gcloud", stderr="ERROR: 412 Precondition Failed")
            self.body = open(args[2]).read()
            self.generation += 1
            return ""
        raise AssertionError(f"unexpected gcloud call: {args}")


@pytest.fixture
def fake_gcs(monkeypatch):
    fake = FakeGcs()
    monkeypatch.setattr(registry_module, "gcloud", fake.gcloud)
    monkeypatch.setattr(registry_module, "gcloud_json", fake.gcloud_json)
    return fake


class TestGcsBackend:
    def test_create_uses_generation_zero(self, fake_gcs):
        backend = GcsBackend("gs://bucket/eray/clusters.json")
        backend.update(lambda doc: doc["clusters"].update({"a": {"name": "a"}}))
        assert fake_gcs.generation == 1
        assert "--if-generation-match=0" in fake_gcs.cp_calls[0]
        assert json.loads(fake_gcs.body)["clusters"]["a"]["name"] == "a"

    def test_cas_retry_applies_mutation_to_fresh_doc(self, fake_gcs, monkeypatch):
        monkeypatch.setattr(registry_module.time, "sleep", lambda s: None)
        backend = GcsBackend("gs://bucket/eray/clusters.json")
        backend.update(lambda doc: doc["clusters"].update({"a": {"name": "a"}}))
        # Losing one race must re-read (picking up 'a') before re-applying.
        fake_gcs.fail_next_cp = 1
        backend.update(lambda doc: doc["clusters"].update({"b": {"name": "b"}}))
        final = json.loads(fake_gcs.body)
        assert set(final["clusters"]) == {"a", "b"}
        matches = [next(x for x in call if "--if-generation-match" in x) for call in fake_gcs.cp_calls]
        assert matches == ["--if-generation-match=0", "--if-generation-match=1", "--if-generation-match=1"]

    def test_exhausted_retries_raise_conflict(self, fake_gcs, monkeypatch):
        monkeypatch.setattr(registry_module.time, "sleep", lambda s: None)
        backend = GcsBackend("gs://bucket/eray/clusters.json")
        fake_gcs.fail_next_cp = 99
        with pytest.raises(ConflictError):
            backend.update(lambda doc: None, retries=3)

    def test_rejects_non_gs_uri(self):
        with pytest.raises(ValueError, match="gs://"):
            GcsBackend("/tmp/clusters.json")


class TestLease:
    def test_acquire_renew_respect_steal(self, tmp_path):
        reg = make_registry(tmp_path)
        assert reg.acquire_lease(now=1000.0) is True
        assert reg.lease_holder(now=1001.0) == registry_module.ClusterRegistry._holder()
        # Same holder renews.
        assert reg.acquire_lease(now=1050.0) is True
        # A different live holder is respected.
        with mock.patch.object(ClusterRegistry, "_holder", return_value="other:1"):
            other = ClusterRegistry(reg.backend)
            assert other.acquire_lease(now=1060.0) is False
            # Expired lease is stealable.
            assert other.acquire_lease(now=5000.0) is True
            assert reg.lease_holder(now=5001.0) == "other:1"

    def test_release(self, tmp_path):
        reg = make_registry(tmp_path)
        reg.acquire_lease(now=0.0)
        reg.release_lease()
        assert reg.lease_holder(now=1.0) is None


def fake_node(state="READY", ips=("10.0.0.5",)):
    return SimpleNamespace(state=state, internal_ips=list(ips), num_hosts=len(ips))


class TestEnsureTpu:
    def _setup(self, tmp_path, monkeypatch, record=None, node=None, qr=None, reachable=False):
        reg = make_registry(tmp_path)
        reg.upsert(record or make_record())
        monkeypatch.setattr(fleet_module, "describe_node", lambda *a, **k: node)
        monkeypatch.setattr(fleet_module, "describe_queued_resource", lambda *a, **k: qr)
        monkeypatch.setattr(fleet_module, "head_reachable", lambda *a, **k: reachable)
        return reg

    def test_desired_down_is_noop(self, tmp_path, monkeypatch):
        reg = self._setup(tmp_path, monkeypatch, record=make_record(desired_state="down"))
        result = fleet_module.ensure_tpu("trainer1", registry=reg)
        assert result["state"] == "DOWN"

    def test_ready_and_reachable_is_healthy(self, tmp_path, monkeypatch):
        reg = self._setup(tmp_path, monkeypatch, node=fake_node(), reachable=True)
        result = fleet_module.ensure_tpu("trainer1", registry=reg)
        assert result["state"] == "HEALTHY"
        assert reg.get("trainer1").head_ip == "10.0.0.5"

    def test_ready_unreachable_connects_and_bootstraps_once(self, tmp_path, monkeypatch):
        record = make_record(bootstrap_cmd="curl setup | bash", generation=2, bootstrapped_generation=1)
        reg = self._setup(tmp_path, monkeypatch, record=record, node=fake_node(), reachable=False)
        bootstrap_calls = []
        monkeypatch.setattr(
            "eray.cli.utils.run_on_all_hosts",
            lambda node, cmd, **k: bootstrap_calls.append(cmd) or [(0, SimpleNamespace(returncode=0))],
        )
        connect_result = SimpleNamespace(head_ip="10.0.0.5", num_hosts=1)
        monkeypatch.setattr("eray.cli.tpu.connect_tpus", lambda node, **k: connect_result)
        result = fleet_module.ensure_tpu("trainer1", registry=reg)
        assert result["state"] == "CONNECTED"
        assert bootstrap_calls == ["curl setup | bash"]
        loaded = reg.get("trainer1")
        assert loaded.bootstrapped_generation == 2
        assert loaded.head_ip == "10.0.0.5"
        assert loaded.state == "HEALTHY"

    def test_bootstrap_skipped_when_generation_matches(self, tmp_path, monkeypatch):
        record = make_record(bootstrap_cmd="setup", generation=1, bootstrapped_generation=1)
        reg = self._setup(tmp_path, monkeypatch, record=record, node=fake_node(), reachable=False)
        monkeypatch.setattr(
            "eray.cli.utils.run_on_all_hosts", lambda *a, **k: pytest.fail("bootstrap must not run")
        )
        monkeypatch.setattr(
            "eray.cli.tpu.connect_tpus", lambda node, **k: SimpleNamespace(head_ip="10.0.0.5", num_hosts=1)
        )
        assert fleet_module.ensure_tpu("trainer1", registry=reg)["state"] == "CONNECTED"

    def test_no_node_no_qr_creates_and_clears_intent(self, tmp_path, monkeypatch):
        reg = self._setup(tmp_path, monkeypatch, node=None, qr=None)
        created = {}

        def fake_create(spec, *, qr_id=None):
            created["spec"] = spec
            created["qr_id"] = qr_id
            return SimpleNamespace(qr_id=qr_id, state="WAITING_FOR_RESOURCES")

        monkeypatch.setattr(fleet_module, "create_queued_resource", fake_create)
        result = fleet_module.ensure_tpu("trainer1", registry=reg)
        assert result["state"] == "WAITING"
        assert created["qr_id"] == "trainer1"
        assert created["spec"].capacity == "spot"
        assert created["spec"].labels == {"eray-cluster": "trainer1"}
        loaded = reg.get("trainer1")
        assert loaded.intent is None
        assert loaded.qr_id == "trainer1"

    def test_pending_qr_reports_waiting(self, tmp_path, monkeypatch):
        qr = SimpleNamespace(qr_id="trainer1", state="WAITING_FOR_RESOURCES", is_active=False, is_terminal=False)
        reg = self._setup(tmp_path, monkeypatch, node=None, qr=qr)
        assert fleet_module.ensure_tpu("trainer1", registry=reg)["state"] == "WAITING"

    def test_terminal_qr_reports_needs_recreate(self, tmp_path, monkeypatch):
        qr = SimpleNamespace(qr_id="trainer1", state="SUSPENDED", is_active=False, is_terminal=True)
        reg = self._setup(tmp_path, monkeypatch, node=None, qr=qr)
        result = fleet_module.ensure_tpu("trainer1", registry=reg)
        assert result["state"] == "NEEDS_RECREATE"

    def test_unregistered_raises(self, tmp_path):
        with pytest.raises(KeyError, match="not registered"):
            fleet_module.ensure_tpu("ghost", registry=make_registry(tmp_path))


class TestFleetCli:
    @pytest.fixture
    def local_registry(self, tmp_path, monkeypatch):
        reg = make_registry(tmp_path)
        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
        return reg

    def test_add_with_adoption(self, local_registry):
        adopted = SimpleNamespace(qr_id="n_server_spot_m", state="ACTIVE", accelerator_type="v5p-8")
        runner = CliRunner()
        with mock.patch("eray.cli.fleet.describe_queued_resource", return_value=adopted):
            result = runner.invoke(
                cli, ["fleet", "add", "n_server_spot_m", "--zone", "us-central1-a", "--project", "proj"]
            )
        assert result.exit_code == 0, result.output
        record = local_registry.get("n_server_spot_m")
        assert record.accelerator_type == "v5p-8"
        assert record.qr_id == "n_server_spot_m"
        assert record.state == "ADOPTED"

    def test_add_requires_type_when_nothing_to_adopt(self, local_registry):
        runner = CliRunner()
        with mock.patch("eray.cli.fleet.describe_queued_resource", return_value=None):
            result = runner.invoke(cli, ["fleet", "add", "newone", "--zone", "z", "--project", "p"])
        assert result.exit_code != 0
        assert "--type is required" in result.output

    def test_up_down_toggle_desired_state(self, local_registry):
        local_registry.upsert(make_record(desired_state="down"))
        runner = CliRunner()
        fake_result = {"name": "trainer1", "state": "WAITING", "detail": ""}
        with mock.patch("eray.cli.fleet.ensure_tpu", return_value=fake_result):
            up = runner.invoke(cli, ["fleet", "up", "trainer1"])
        assert up.exit_code == 0, up.output
        assert local_registry.get("trainer1").desired_state == "up"
        down = CliRunner().invoke(cli, ["fleet", "down", "trainer1"])
        assert down.exit_code == 0
        assert local_registry.get("trainer1").desired_state == "down"

    def test_status_table(self, local_registry):
        local_registry.upsert(make_record(state="HEALTHY", head_ip="10.0.0.5"))
        runner = CliRunner()
        result = runner.invoke(cli, ["fleet", "status", "--no-probe"])
        assert result.exit_code == 0, result.output
        assert "trainer1" in result.output
        assert "HEALTHY" in result.output

    def test_remove(self, local_registry):
        local_registry.upsert(make_record())
        result = CliRunner().invoke(cli, ["fleet", "remove", "trainer1"])
        assert result.exit_code == 0
        assert local_registry.get("trainer1") is None


class TestRunClusterFlag:
    def test_resolve_cluster_address(self, tmp_path, monkeypatch):
        from eray.cli.jobs import resolve_cluster_address

        reg = make_registry(tmp_path)
        reg.upsert(make_record(head_ip="10.0.0.9"))
        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
        assert resolve_cluster_address("trainer1") == "http://10.0.0.9:8265"

    def test_unknown_cluster_fails(self, tmp_path, monkeypatch):
        import click
        from eray.cli.jobs import resolve_cluster_address

        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: make_registry(tmp_path)))
        with pytest.raises(click.ClickException, match="not registered"):
            resolve_cluster_address("ghost")

    def test_run_uses_cluster_address_and_marks_restartable(self, tmp_path, monkeypatch):
        reg = make_registry(tmp_path)
        reg.upsert(make_record(head_ip="10.0.0.9"))
        monkeypatch.setattr(ClusterRegistry, "from_config", classmethod(lambda cls: reg))
        captured = {}

        class FakeClient:
            def submit_job(self, **kwargs):
                captured["submit"] = kwargs
                return kwargs.get("submission_id") or "sub-1"

            def list_jobs(self):
                return []

        def fake_make_client(address):
            captured["address"] = address
            return FakeClient()

        monkeypatch.setattr("eray.cli.jobs.make_client", fake_make_client)
        monkeypatch.setattr("eray.cli.jobs._history_append", lambda record: captured.setdefault("history", record))
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "-c", "trainer1", "--restartable", "--no-working-dir", "--no-env-inherit", "--", "python", "x.py"],
        )
        assert result.exit_code == 0, result.output
        assert captured["address"] == "http://10.0.0.9:8265"
        assert captured["submit"]["metadata"]["cluster"] == "trainer1"
        assert captured["submit"]["metadata"]["restartable"] == "1"
        assert captured["history"]["cluster"] == "trainer1"
