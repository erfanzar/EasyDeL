from __future__ import annotations

import socket

import pytest

from easydel.inference.esurge.distributed import resolve_distributed_role, resolve_service_hosts
from easydel.inference.esurge.distributed.controller import DistributedController
from easydel.inference.esurge.distributed.protocol import STATUS_ERROR, STATUS_OK, compute_sampled_digest


class _ModelOutput:
    def __init__(self, req_ids, sampled_token_ids):
        self.req_ids = req_ids
        self.sampled_token_ids = sampled_token_ids


class _FakeWorkerClient:
    def __init__(self, response, *, fail_on_begin: bool = False):
        self.response = dict(response)
        self.fail_on_begin = bool(fail_on_begin)
        self.last_step_id = None
        self.begin_calls = 0
        self.finish_calls = 0
        self.reset_calls = 0

    def begin_step(self, *, step_id, scheduler_output):
        self.begin_calls += 1
        if self.fail_on_begin:
            raise RuntimeError("begin step failed")
        del scheduler_output
        self.last_step_id = int(step_id)

    def finish_step(self):
        self.finish_calls += 1
        return dict(self.response)

    def reset_connection(self):
        self.reset_calls += 1


def test_resolve_service_hosts_sorted_and_deduped(monkeypatch):
    def _fake_getaddrinfo(*args, **kwargs):
        del args, kwargs
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.10", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.2", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.10", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    result = resolve_service_hosts("esurge.svc", world_size=2)

    assert result.hosts == ["10.0.0.2", "10.0.0.10"]
    assert result.rank_to_host == {0: "10.0.0.2", 1: "10.0.0.10"}


def test_resolve_service_hosts_world_size_mismatch(monkeypatch):
    def _fake_getaddrinfo(*args, **kwargs):
        del args, kwargs
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)

    with pytest.raises(ValueError, match="world size mismatch"):
        resolve_service_hosts("esurge.svc", world_size=2)


def test_resolve_distributed_role_auto_and_constraints():
    assert resolve_distributed_role("auto", 0) == "leader"
    assert resolve_distributed_role("auto", 1) == "worker"

    with pytest.raises(ValueError, match="requires rank 0"):
        resolve_distributed_role("leader", 2)

    with pytest.raises(ValueError, match="cannot be used with rank 0"):
        resolve_distributed_role("worker", 0)


def test_controller_verify_step_success_and_digest_mismatch():
    controller = DistributedController(
        enabled=True,
        role="leader",
        rank=0,
        world_size=2,
        service_name="esurge.svc",
        control_port=19666,
        control_bind_host="0.0.0.0",
        advertise_addr=None,
        auth_token="secret",
        step_timeout_s=1.0,
        connect_timeout_s=1.0,
        verify_sampling_digest=True,
        config_fingerprint="fp",
        execute_step=None,
    )

    model_output = _ModelOutput(["req-1"], [[42]])
    good_digest = compute_sampled_digest(model_output.req_ids, model_output.sampled_token_ids)

    worker = _FakeWorkerClient(
        {
            "status": STATUS_OK,
            "step_id": 1,
            "sampled_digest": good_digest,
            "num_reqs": 1,
            "timing_ms": 1.5,
        }
    )

    controller._worker_clients = {1: worker}

    dispatch = controller.dispatch_step({"dummy": True})
    assert dispatch is not None

    controller.verify_step(dispatch, model_output)

    worker.response["sampled_digest"] = "bad-digest"
    with pytest.raises(ValueError, match="digest mismatch"):
        controller.verify_step(dispatch, model_output)


def test_controller_verify_step_worker_error():
    controller = DistributedController(
        enabled=True,
        role="leader",
        rank=0,
        world_size=2,
        service_name="esurge.svc",
        control_port=19666,
        control_bind_host="0.0.0.0",
        advertise_addr=None,
        auth_token="secret",
        step_timeout_s=1.0,
        connect_timeout_s=1.0,
        verify_sampling_digest=True,
        config_fingerprint="fp",
        execute_step=None,
    )

    worker = _FakeWorkerClient(
        {
            "status": STATUS_ERROR,
            "step_id": 1,
            "error": "boom",
            "timing_ms": 1.0,
        }
    )
    controller._worker_clients = {1: worker}

    dispatch = controller.dispatch_step({"dummy": True})
    assert dispatch is not None

    model_output = _ModelOutput(["req-1"], [[7]])

    with pytest.raises(ValueError, match="error=boom"):
        controller.verify_step(dispatch, model_output)


def test_controller_dispatch_step_partial_failure_resets_dispatched_clients():
    controller = DistributedController(
        enabled=True,
        role="leader",
        rank=0,
        world_size=3,
        service_name="esurge.svc",
        control_port=19666,
        control_bind_host="0.0.0.0",
        advertise_addr=None,
        auth_token="secret",
        step_timeout_s=1.0,
        connect_timeout_s=1.0,
        verify_sampling_digest=True,
        config_fingerprint="fp",
        execute_step=None,
    )

    worker_ok = _FakeWorkerClient(
        {
            "status": STATUS_OK,
            "step_id": 1,
            "sampled_digest": "digest",
            "num_reqs": 1,
            "timing_ms": 1.0,
        }
    )
    worker_fail = _FakeWorkerClient({"status": STATUS_OK}, fail_on_begin=True)

    controller._worker_clients = {1: worker_ok, 2: worker_fail}

    with pytest.raises(ValueError, match="failed to dispatch"):
        controller.dispatch_step({"dummy": True})

    assert worker_ok.begin_calls == 1
    assert worker_ok.finish_calls == 1
    assert worker_ok.reset_calls == 1

    assert worker_fail.begin_calls == 1
    assert worker_fail.finish_calls == 0
    assert worker_fail.reset_calls == 1


def test_controller_start_handshake_config_mismatch(monkeypatch):
    import easydel.inference.esurge.distributed.controller as controller_module

    class _DummyDiscovery:
        def __init__(self):
            self.hosts = ["10.0.0.1", "10.0.0.2"]

    class _DummyClient:
        def __init__(self, **kwargs):
            self.endpoint = kwargs["endpoint"]

        def hello(self):
            return {
                "status": STATUS_OK,
                "rank": 1,
                "world_size": 2,
                "config_fingerprint": "wrong-fingerprint",
            }

        def close(self):
            return None

    monkeypatch.setattr(controller_module, "resolve_service_hosts", lambda *args, **kwargs: _DummyDiscovery())
    monkeypatch.setattr(controller_module, "WorkerRpcClient", _DummyClient)

    controller = DistributedController(
        enabled=True,
        role="leader",
        rank=0,
        world_size=2,
        service_name="esurge.svc",
        control_port=19666,
        control_bind_host="0.0.0.0",
        advertise_addr=None,
        auth_token="secret",
        step_timeout_s=1.0,
        connect_timeout_s=1.0,
        verify_sampling_digest=True,
        config_fingerprint="expected-fingerprint",
        execute_step=None,
    )

    with pytest.raises(ValueError, match="config mismatch"):
        controller.start()
