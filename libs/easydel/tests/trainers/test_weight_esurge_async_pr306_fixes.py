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
"""Regression tests for PR #306 fixes in WeightTransport, eSurge weight-versioning,
BoundedAsyncRolloutCoordinator, and AsyncRolloutMixin.
"""

from __future__ import annotations

import datetime
import socket
import struct
import threading
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.inference.esurge import esurge_engine
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
from easydel.inference.esurge.engine.core_driver import EngineCoreDriver
from easydel.inference.esurge.engine.loop import EngineLoop
from easydel.inference.esurge.engine.output_pipeline import OutputPipeline
from easydel.inference.esurge.engine.registry import RequestRecord, RequestRegistry
from easydel.inference.esurge.engine_types import EngineCoreOutput, EngineCoreOutputs
from easydel.inference.esurge.esurge_engine import CompletionOutput, RequestOutput
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import AdmissionRecord
from easydel.inference.sampling_params import SamplingParams
from easydel.infra.errors import EasyDeLBreakRequest, EasyDeLPreemptionSignal
from easydel.infra.loss_utils import LossMetrics
from easydel.trainers._online_rl import weight_transport
from easydel.trainers._online_rl.async_rollout import BoundedAsyncRolloutCoordinator, PolicySnapshot
from easydel.trainers._online_rl.weight_transport import (
    JaxTransferClient,
    JaxTransferServerBackend,
    OfferOutcome,
    RemoteSubscriberBridge,
    RemoteWeightSubscriber,
    SocketMessageChannel,
    WeightOffer,
    WeightPublisher,
    WeightTransferStalled,
    WeightTransportClosed,
    WeightTransportError,
    _describe_sharding,
    _peer_is_local,
)
from easydel.trainers.async_rollout_mixin import AsyncRolloutMixin, AsyncRolloutResult
from easydel.workers.esurge.pipeline import Detokenizer
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

pytestmark = pytest.mark.cpu_ok


class _EchoFetcher:
    def fetch(self, offer: WeightOffer, *, timeout: float | None):
        del timeout
        return dict(offer.metadata)


class _CharTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens)

    def get_vocab(self):
        return {chr(v): v for v in range(128)}


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# --------------------------------------------------------------------------- #4: deadlock & peer EOF / bridge.close()


def test_remote_subscriber_close_unblocks_concurrent_receive_without_deadlock():
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "sub")
        remote = RemoteWeightSubscriber(SocketMessageChannel(right), _EchoFetcher())
        assert bridge.authenticated.wait(5.0)
        remote.request()

        errors: list[BaseException] = []
        entered = threading.Event()

        def receiver():
            entered.set()
            try:
                remote.receive(timeout=None)
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=receiver, daemon=True)
        t.start()
        assert entered.wait(2.0)
        # Wait until receiver holds remote._lock inside _call -> recv().
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not remote._lock.acquire(blocking=False):
                break
            remote._lock.release()
            time.sleep(0.01)

        t0 = time.monotonic()
        remote.close()
        t.join(timeout=3.0)
        bridge.join(timeout=3.0)
        assert time.monotonic() - t0 < 2.5
        assert not t.is_alive(), "concurrent receive(timeout=None) deadlocked on close()"
        assert not bridge._thread.is_alive(), "bridge failed to detect peer close during receive(timeout=None)"
        assert errors and isinstance(errors[0], WeightTransportClosed)


def test_bridge_detects_peer_eof_while_blocked_in_infinite_receive():
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "eof-sub")
        client = SocketMessageChannel(right)
        client.send({"op": "request"})
        assert client.recv()["ok"] is True
        # Start an unbounded receive RPC on the bridge, then abruptly close the peer socket.
        client.send({"op": "receive", "timeout": None})
        time.sleep(0.05)
        client.close()
        bridge.join(timeout=3.0)
        assert not bridge._thread.is_alive()
        with pytest.raises(KeyError):
            publisher.installed_version("eof-sub")


def test_bridge_close_unblocks_waiting_receive_and_cleans_up():
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "close-sub")
        remote = RemoteWeightSubscriber(SocketMessageChannel(right), _EchoFetcher())
        assert bridge.authenticated.wait(5.0)
        remote.request()

        errors: list[BaseException] = []

        def receiver():
            try:
                remote.receive(timeout=None)
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=receiver, daemon=True)
        t.start()
        time.sleep(0.05)
        bridge.close(timeout=3.0)
        t.join(timeout=3.0)
        assert not bridge._thread.is_alive()
        assert not t.is_alive()
        assert errors and isinstance(errors[0], WeightTransportClosed)
        with pytest.raises(KeyError):
            publisher.installed_version("close-sub")
        remote.close()


# --------------------------------------------------------------------------- #17: publish / _begin_pull error handling


def test_publisher_publish_releases_handle_when_describe_raises():
    released: list[object] = []

    class _FailingDescribeBackend:
        def stage(self, version, snapshot):
            return ("handle", version, snapshot)

        def describe(self, handle):
            raise RuntimeError("describe boom")

        def serve(self, handle, offer_id, timeout):
            del handle, offer_id, timeout

        def release(self, handle):
            released.append(handle)

    with WeightPublisher(_FailingDescribeBackend()) as publisher:
        with pytest.raises(RuntimeError, match="describe boom"):
            publisher.publish(1, "w1")
        assert len(released) == 1
        assert released[0] == ("handle", 1, "w1")
        assert publisher.latest_version is None
        assert publisher.staged_version is None


def test_publisher_begin_pull_error_leaves_offer_offered_and_supersedable():
    serve_calls = 0

    class _FlakyServeBackend:
        def stage(self, version, snapshot):
            return snapshot

        def describe(self, handle):
            return {"kind": "flaky"}

        def serve(self, handle, offer_id, timeout):
            nonlocal serve_calls
            serve_calls += 1
            if serve_calls == 1:
                raise RuntimeError("transient serve failure")

        def release(self, handle):
            del handle

    with WeightPublisher(_FlakyServeBackend()) as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1")
        offer1 = sub.receive(timeout=1.0)
        assert offer1 is not None
        with pytest.raises(RuntimeError, match="transient serve failure"):
            sub.pull(offer1)
        # Offer must remain in "offered" state: retrying pull succeeds.
        assert sub.pull(offer1) == "w1"
        sub.ack(offer1)

        # Also verify a failed serve can be superseded cleanly by a newer publish.
        serve_calls = 0
        sub.request()
        publisher.publish(2, "w2")
        offer2 = sub.receive(timeout=1.0)
        assert offer2 is not None
        with pytest.raises(RuntimeError, match="transient serve failure"):
            sub.pull(offer2)
        publisher.publish(3, "w3")
        assert (offer2.offer_id, 2, "s0", OfferOutcome.SUPERSEDED) in publisher.settlements
        offer3 = sub.receive(timeout=1.0)
        assert offer3 is not None and offer3.version == 3
        assert sub.pull(offer3) == "w3"
        sub.ack(offer3)


# --------------------------------------------------------------------------- #18: bridge _offers bounded & ack error


def test_bridge_offers_bounded_and_ack_before_begin_does_not_wedge():
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "sub")
        remote = RemoteWeightSubscriber(SocketMessageChannel(right), _EchoFetcher())
        assert bridge.authenticated.wait(5.0)

        # Supersede 15 offers without pulling them; _offers must hold at most 1 active descriptor.
        remote.request()
        for version in range(1, 16):
            publisher.publish(version, f"w{version}", metadata={"v": version})
            offer = remote.receive(timeout=2.0)
            assert offer is not None and offer.version == version
            assert len(bridge._offers) <= 1

        # Calling ack() before begin()/pull() fails validation but MUST NOT delete the offer from bridge._offers.
        with pytest.raises(WeightTransportError, match="must be pulled"):
            remote.ack(offer)
        assert offer.offer_id in bridge._offers

        # Subsequent pull + ack must still succeed cleanly.
        assert remote.pull(offer) == {"v": 15}
        remote.ack(offer)
        assert remote.installed_version == 15
        assert len(bridge._offers) == 0

        remote.close()
        bridge.join(timeout=3.0)


# --------------------------------------------------------------------------- #19: JaxTransferServerBackend / Client


def test_jax_transfer_server_backend_enforces_finite_timeout_and_close():
    await_calls: list[tuple[int, object]] = []
    closed_flag = []

    class _FakeServer:
        def address(self):
            return "127.0.0.1:9999"

        def await_pull(self, uuid, arrays, timeout=None):
            await_calls.append((uuid, timeout))

        def close(self):
            closed_flag.append(True)

    with pytest.raises(ValueError, match="default_pull_timeout"):
        JaxTransferServerBackend(server=_FakeServer(), default_pull_timeout=0.0)
    with pytest.raises(ValueError, match="default_pull_timeout"):
        JaxTransferServerBackend(server=_FakeServer(), default_pull_timeout=-5.0)

    backend = JaxTransferServerBackend(server=_FakeServer(), default_pull_timeout=15.0)
    # Publisher has pull_timeout=None, yet backend.serve() MUST pass a finite datetime timeout to await_pull.
    with WeightPublisher(backend, pull_timeout=None) as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, {"a": jnp.ones((2,), dtype=jnp.float32)})
        offer = sub.receive(timeout=1.0)
        assert offer is not None
        sub.pull(offer)
        sub.ack(offer)

    assert len(await_calls) == 1
    assert isinstance(await_calls[0][1], datetime.datetime)

    backend.close()
    backend.close()  # idempotent
    assert closed_flag == [True]
    with pytest.raises(WeightTransportClosed):
        _ = backend.address
    with pytest.raises(WeightTransportClosed):
        backend.stage(2, {"a": jnp.ones((2,), dtype=jnp.float32)})


def test_jax_transfer_client_forwards_timeout_evicts_connection_and_closes():
    connect_count = 0
    seen_timeouts: list[object] = []
    fail_first = True

    class _FakeConnection:
        def __init__(self, idx):
            self.idx = idx
            self.closed = False

        def pull(self, uuid, targets, timeout=None):
            nonlocal fail_first
            del uuid
            seen_timeouts.append(timeout)
            if fail_first:
                fail_first = False
                raise RuntimeError("broken pipe")
            return [jnp.zeros(t.shape, dtype=t.dtype) for t in targets]

        def close(self):
            self.closed = True

    created_conns: list[_FakeConnection] = []

    class _FakeServer:
        def connect(self, address):
            nonlocal connect_count
            del address
            connect_count += 1
            conn = _FakeConnection(connect_count)
            created_conns.append(conn)
            return conn

    # Build a valid jax_transfer descriptor using JaxTransferServerBackend.stage.
    class _StubBackendServer:
        def address(self):
            return "127.0.0.1:8888"

        def await_pull(self, uuid, arrays, timeout=None):
            del uuid, arrays, timeout

    backend = JaxTransferServerBackend(server=_StubBackendServer())
    handle = backend.stage(1, {"w": jnp.ones((3,), dtype=jnp.float32)})
    offer = WeightOffer(offer_id=1, subscriber="s0", version=1, predecessor=None, descriptor=backend.describe(handle))

    client = JaxTransferClient(server=_FakeServer())
    # First fetch fails -> must pass datetime timeout and evict broken connection #1.
    with pytest.raises(WeightTransportError, match="failed"):
        client.fetch(offer, timeout=5.0)
    assert len(seen_timeouts) == 1 and isinstance(seen_timeouts[0], datetime.datetime)
    assert connect_count == 1
    assert created_conns[0].closed is True
    assert "127.0.0.1:8888" not in client._connections

    # Second fetch reconnects with fresh connection #2 and succeeds.
    result = client.fetch(offer, timeout=5.0)
    assert connect_count == 2
    assert isinstance(seen_timeouts[1], datetime.datetime)
    np.testing.assert_array_equal(np.asarray(result["w"]), np.zeros((3,), dtype=np.float32))

    # Already-expired timeout raises WeightTransferStalled without calling pull().
    with pytest.raises(WeightTransferStalled):
        client.fetch(offer, timeout=0.0)
    assert len(seen_timeouts) == 2

    client.close()
    client.close()
    with pytest.raises(WeightTransportClosed):
        client.fetch(offer, timeout=5.0)


# --------------------------------------------------------------------------- #38, #39, #40, #41, #42, #51


def test_peer_is_local_handles_ipv4_mapped_ipv6_and_unknown_families():
    def _fake_sock(family, peer=None):
        return SimpleNamespace(family=family, getpeername=lambda: (peer, 1234))

    assert _peer_is_local(_fake_sock(socket.AF_INET, "127.0.0.1")) is True
    assert _peer_is_local(_fake_sock(socket.AF_INET, "10.0.0.1")) is False
    assert _peer_is_local(_fake_sock(socket.AF_INET6, "::1")) is True
    assert _peer_is_local(_fake_sock(socket.AF_INET6, "::ffff:127.0.0.1")) is True
    assert _peer_is_local(_fake_sock(socket.AF_INET6, "::ffff:192.168.1.5")) is False
    if hasattr(socket, "AF_UNIX"):
        assert _peer_is_local(_fake_sock(socket.AF_UNIX)) is True
    assert _peer_is_local(_fake_sock(socket.AF_UNSPEC)) is False
    assert _peer_is_local(_fake_sock(99999)) is False


def test_authenticate_enforces_total_handshake_deadline_against_slow_loris(monkeypatch):
    monkeypatch.setattr(weight_transport, "_HANDSHAKE_TIMEOUT", 0.25)
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "slow", auth_token="secret")
        stop = threading.Event()

        def slow_loris():
            # Send a 200-byte frame header, then trickle 1 byte every 0.04s (~8s total if per-recv timeout resets).
            try:
                right.sendall(struct.pack(">I", 200))
                for _ in range(200):
                    if stop.wait(0.04):
                        return
                    right.sendall(b" ")
            except OSError:
                return

        attacker = threading.Thread(target=slow_loris, daemon=True)
        attacker.start()
        t0 = time.monotonic()
        bridge.join(timeout=3.0)
        elapsed = time.monotonic() - t0
        stop.set()
        right.close()
        attacker.join(timeout=2.0)

        assert not bridge._thread.is_alive()
        assert elapsed < 1.5, f"slow-loris bypassed total handshake deadline ({elapsed:.2f}s)"
        assert isinstance(bridge.auth_error, WeightTransportError)
        assert not bridge.authenticated.is_set()


def test_publisher_validates_watchdog_interval_and_survives_release_errors():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="watchdog_interval"):
            WeightPublisher(pull_timeout=1.0, watchdog_interval=bad)

    released: list[str] = []

    def flaky_release(snap):
        released.append(snap)
        if snap == "w1":
            raise RuntimeError("release callback exploded")

    with WeightPublisher(pull_timeout=0.04, watchdog_interval=0.01) as publisher:
        sub1 = publisher.subscribe("s1")
        sub1.request()
        publisher.publish(1, "w1", release=flaky_release)
        sub1.pull(sub1.receive(timeout=1.0))
        publisher.publish(2, "w2", release=flaky_release)

        deadline = time.monotonic() + 3.0
        while "w1" not in released and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "w1" in released
        assert publisher._watchdog_thread is not None and publisher._watchdog_thread.is_alive()

        # Drain the first stall error, then trigger a second stall to prove the watchdog survived.
        with pytest.raises(WeightTransferStalled):
            publisher.check_watchdog()

        sub2 = publisher.subscribe("s2")
        sub2.request()
        offer2 = sub2.receive(timeout=1.0)
        assert offer2 is not None
        sub2.pull(offer2)
        publisher.publish(3, "w3", release=flaky_release)

        deadline = time.monotonic() + 3.0
        while "w2" not in released and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "w2" in released


def test_publisher_receive_while_pulling_raises_instead_of_hanging():
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1")
        offer = sub.receive(timeout=1.0)
        assert offer is not None
        sub.pull(offer)
        # Active offer is now "pulling"; receive(timeout=None) must raise immediately instead of hanging.
        with pytest.raises(WeightTransportError, match="pulling"):
            sub.receive(timeout=None)


def test_describe_sharding_rejects_unconstrained_and_unsupported_multidevice():
    dev = jax.local_devices()[0]
    mesh = Mesh(np.asarray([dev]), ("dp",))

    assert _describe_sharding(None) is None
    assert _describe_sharding(SingleDeviceSharding(dev)) is None
    described = _describe_sharding(NamedSharding(mesh, PartitionSpec("dp")))
    assert described == {"kind": "named", "axis_names": ["dp"], "mesh_shape": [1], "spec": ["dp"]}

    unconstrained = NamedSharding(mesh, PartitionSpec(PartitionSpec.UNCONSTRAINED))
    with pytest.raises(WeightTransportError, match="UNCONSTRAINED"):
        _describe_sharding(unconstrained)

    fake_multidevice = SimpleNamespace(device_set={0, 1})
    with pytest.raises(WeightTransportError, match="unsupported multi-device sharding"):
        _describe_sharding(fake_multidevice)


def test_bridge_handles_malformed_json_and_oversized_frame_post_auth_cleanly():
    thread_exceptions: list[threading.ExceptHookArgs] = []
    orig_hook = threading.excepthook
    threading.excepthook = thread_exceptions.append
    try:
        with WeightPublisher() as publisher:
            # Case 1: malformed JSON post-auth
            left, right = socket.socketpair()
            bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "bad-json")
            assert bridge.authenticated.wait(5.0)
            payload = b"{not valid json"
            right.sendall(struct.pack(">I", len(payload)) + payload)
            bridge.join(timeout=3.0)
            right.close()
            assert not bridge._thread.is_alive()
            with pytest.raises(KeyError):
                publisher.installed_version("bad-json")

            # Case 2: oversized frame post-auth
            left2, right2 = socket.socketpair()
            bridge2 = RemoteSubscriberBridge(publisher, SocketMessageChannel(left2), "oversized")
            assert bridge2.authenticated.wait(5.0)
            right2.sendall(struct.pack(">I", weight_transport._MAX_FRAME_BYTES + 1))
            bridge2.join(timeout=3.0)
            right2.close()
            assert not bridge2._thread.is_alive()
            with pytest.raises(KeyError):
                publisher.installed_version("oversized")
    finally:
        threading.excepthook = orig_hook
    assert thread_exceptions == [], f"Unexpected unhandled bridge daemon exceptions: {thread_exceptions}"


# --------------------------------------------------------------------------- #5, #6, #43, #52: eSurge regressions


def _make_async_pipeline_shell():
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="r",
        prompt="p",
        prompt_token_ids=[1],
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    registry.register("r", RequestRecord(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=16)), output)
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=Detokenizer(_CharTokenizer(), context_window=0),
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=1000.0,
        on_stop_strings=lambda matches: None,
        on_activity=lambda: None,
        on_fatal=lambda *args: pytest.fail(f"Output pipeline failed: {args}"),
    )
    return registry, pipeline, output


def _make_abort_shell(registry: RequestRegistry, pipeline: OutputPipeline, stale_kv_ids: list[str]):
    engine = object.__new__(esurge_engine.eSurge)
    engine._scheduler_lock = threading.RLock()
    engine._request_lock = registry.request_lock
    engine._output_lock = registry.output_lock
    engine._active_requests = registry.records
    engine._request_outputs = registry.outputs
    engine._request_events = registry.events
    engine._output_pipeline = pipeline
    engine._weight_version = 1
    engine._request_plane = None
    engine.core = SimpleNamespace(
        stale_kv_parent_ids=lambda min_v, cur_v: list(stale_kv_ids),
    )

    def fake_abort(request_id: str) -> None:
        with engine._scheduler_lock, engine._request_lock, engine._output_lock:
            engine._active_requests.pop(request_id, None)
            ro = engine._request_outputs.get(request_id)
            if ro is not None:
                ro.finished = True
                for comp in ro.outputs:
                    if comp.finish_reason is None:
                        comp.finish_reason = "abort"

    engine.abort_request = fake_abort
    return engine


def test_abort_requests_before_version_flushes_kv_versions_and_queued_v0_tokens():
    # Case (a): _KVVersionUpdate queued behind a slow item must land BEFORE abort_request() clears _active_requests.
    registry, pipeline, output = _make_async_pipeline_shell()
    release_blocker = threading.Event()
    entered_blocker = threading.Event()
    orig_apply = pipeline._apply_kv_versions

    def gated_apply(item):
        if "blocker" in item.versions:
            entered_blocker.set()
            assert release_blocker.wait(5.0)
        return orig_apply(item)

    pipeline._apply_kv_versions = gated_apply
    pipeline.start()
    try:
        pipeline.submit_kv_versions({"blocker": 0})
        assert entered_blocker.wait(2.0)
        pipeline.submit_kv_versions({"r": 0})
        engine = _make_abort_shell(registry, pipeline, stale_kv_ids=["r"])

        # Release blocker shortly after abort_requests_before_version starts flushing.
        threading.Timer(0.05, release_blocker.set).start()
        aborted = engine.abort_requests_before_version(1)
        assert aborted == ["r"]
        assert output.finished is True
        assert output.outputs[0].finish_reason == "abort"
        assert output.outputs[0].oldest_kv_weight_version == 0
    finally:
        release_blocker.set()
        pipeline.stop()

    # Case (b): request emitted a v0 token into OutputPipeline.queue and was preempted (stale_kv_ids == []).
    registry2, pipeline2, output2 = _make_async_pipeline_shell()
    release_blocker2 = threading.Event()
    entered_blocker2 = threading.Event()
    orig_apply2 = pipeline2._apply_kv_versions

    def gated_apply2(item):
        if "blocker" in item.versions:
            entered_blocker2.set()
            assert release_blocker2.wait(5.0)
        return orig_apply2(item)

    pipeline2._apply_kv_versions = gated_apply2
    pipeline2.start()
    try:
        pipeline2.submit_kv_versions({"blocker": 0})
        assert entered_blocker2.wait(2.0)
        bundle = {0: EngineCoreOutputs(outputs=[EngineCoreOutput("r", [65])], timestamp=1.0)}
        pipeline2.submit(bundle, weight_version=0)
        # Simulate preemption having cleared core._kv_versions (stale_kv_ids=[]).
        engine2 = _make_abort_shell(registry2, pipeline2, stale_kv_ids=[])
        threading.Timer(0.05, release_blocker2.set).start()
        aborted2 = engine2.abort_requests_before_version(1)
        assert aborted2 == ["r"]
        assert output2.outputs[0].finish_reason == "abort"
        assert output2.oldest_weight_version == 0
    finally:
        release_blocker2.set()
        pipeline2.stop()


def test_apply_kv_versions_updates_output_even_after_direct_abort_clears_active_requests():
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="parent",
        prompt="p",
        prompt_token_ids=[1],
        outputs=[CompletionOutput(index=0, text="", token_ids=[]), CompletionOutput(index=1, text="", token_ids=[])],
    )
    registry.register("parent", RequestRecord(prompt_token_ids=[1], sampling_params=SamplingParams(max_tokens=16)), output)
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=Detokenizer(_CharTokenizer(), context_window=0),
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=1000.0,
        on_stop_strings=lambda matches: None,
        on_activity=lambda: None,
        on_fatal=lambda *args: pytest.fail(f"Output pipeline failed: {args}"),
    )
    # Simulate abort_request clearing _active_requests before _KVVersionUpdate runs.
    registry.records.clear()
    pipeline.submit_kv_versions({"parent": 0, "parent-1": 2})
    assert output.outputs[0].oldest_kv_weight_version == 0
    assert output.outputs[1].oldest_kv_weight_version == 2


def test_install_weights_between_steps_waits_for_inflight_step_during_terminate():
    in_step = threading.Event()
    release_step = threading.Event()
    emitted_versions: list[int] = []
    swap_ran_during_step = False
    step_running = False

    engine = object.__new__(esurge_engine.eSurge)
    engine._closed = False
    engine.cache_config = SimpleNamespace(enable_prefix_caching=False)
    engine._step_coordinator = None
    engine._request_plane = None
    engine._scheduler_lock = threading.RLock()
    engine._weight_version = 0
    engine._info = lambda *args, **kwargs: None
    engine._resolve_weight_update_components = lambda model, gd, gs, go: (model, gs, go)
    engine._output_pipeline = SimpleNamespace(
        submit_kv_versions=lambda versions: None,
        submit=lambda outputs, weight_version=None: emitted_versions.append(weight_version),
    )

    scheduled_once = False

    def fake_schedule():
        nonlocal scheduled_once
        if scheduled_once:
            return SimpleNamespace(total_num_scheduled_tokens=0, num_scheduled_tokens={})
        scheduled_once = True
        return SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"r": 1})

    engine.core = SimpleNamespace(
        schedule=fake_schedule,
        commit=lambda so, mo: {0: SimpleNamespace(outputs=[])},
        mark_kv_weight_version=lambda v: {},
        closed_error=RuntimeError,
    )

    def fake_execute_model(so):
        nonlocal step_running
        del so
        step_running = True
        in_step.set()
        assert release_step.wait(5.0)
        step_running = False
        return SimpleNamespace()

    def fake_update_weights(*, graphdef, graphstate, graphother, reset_state):
        nonlocal swap_ran_during_step
        del graphdef, graphstate, graphother, reset_state
        if step_running:
            swap_ran_during_step = True

    engine.runner = SimpleNamespace(
        executor_manager=SimpleNamespace(has_compiled_variants=lambda: True),
        execute_model=fake_execute_model,
        update_model_weights=fake_update_weights,
    )

    loop = EngineLoop(
        core=engine.core,
        runner=engine.runner,
        coordinator=None,
        scheduler_lock=engine._scheduler_lock,
        emit_outputs=engine._enqueue_engine_outputs,
        drain_parser_stops=lambda: None,
        on_fatal=lambda exc: pytest.fail(f"Loop fatal: {exc}"),
        heartbeat=lambda: None,
        handle_profiling_step=lambda: None,
        is_nonrecoverable=lambda exc: False,
        overlap_execution=False,
        async_scheduling=False,
        runner_verbose=False,
        info=lambda *args: None,
    )
    engine._engine_loop = loop
    loop.start()
    try:
        assert in_step.wait(5.0)
        # Simulate terminate() clearing _scheduler_running BEFORE joining the loop thread while step is in flight.
        engine._scheduler_running = False
        assert loop.thread is not None and loop.thread.is_alive()

        result: dict[str, object] = {}

        def do_swap():
            result["version"] = engine.update_model_weights("new-model", keep_requests=True)

        swap_thread = threading.Thread(target=do_swap, daemon=True)
        swap_thread.start()
        # Give swap_thread time to attempt the swap; it MUST wait for the in-flight step instead of running inline.
        time.sleep(0.08)
        assert swap_thread.is_alive(), "_swap() ran inline concurrently with an in-flight model step!"
        assert engine.weight_version == 0

        # Let the in-flight step finish and emit outputs (under old version 0), then loop exits and runs _swap().
        release_step.set()
        swap_thread.join(timeout=5.0)
        loop.join(timeout=5.0)

        assert not swap_ran_during_step
        assert emitted_versions == [0]
        assert result["version"] == 1
        assert engine.weight_version == 1

        # Closed engine rejects weight updates cleanly.
        engine._closed = True
        with pytest.raises(RuntimeError, match="closed"):
            engine.update_model_weights("after-close", keep_requests=True)
    finally:
        release_step.set()
        loop.request_stop()
        loop.join(timeout=5.0)


def test_pre_token_preemption_clears_oldest_kv_version_while_post_token_preserves_it():
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    # 6 pages of 4 tokens: "post" (4-token prompt) decodes and forces preemption of
    # "pre" (16-token prompt) while "pre" is still mid-prefill with 0 emitted tokens.
    core = EngineCoreDriver(
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec, layer_names=None)],
        max_num_seqs=2,
        max_num_batched_tokens=6,
        max_model_len=64,
        num_pages=6,
        enable_prefix_caching=False,
        async_scheduling=False,
        chunked_prefill_enabled=True,
    )
    registry = RequestRegistry()
    pre_prompt = list(range(1, 17))
    post_prompt = list(range(1, 5))
    pre_out = RequestOutput(
        request_id="pre",
        prompt="p",
        prompt_token_ids=pre_prompt,
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    post_out = RequestOutput(
        request_id="post",
        prompt="q",
        prompt_token_ids=post_prompt,
        outputs=[CompletionOutput(index=0, text="", token_ids=[])],
    )
    registry.register("pre", RequestRecord(prompt_token_ids=pre_prompt, sampling_params=SamplingParams(max_tokens=32)), pre_out)
    registry.register("post", RequestRecord(prompt_token_ids=post_prompt, sampling_params=SamplingParams(max_tokens=32)), post_out)
    pipeline = OutputPipeline(
        registry=registry,
        detokenizer_client=Detokenizer(_CharTokenizer(), context_window=0),
        eos_token_ids=[],
        decode_interval_tokens=1,
        decode_interval_secs=1000.0,
        on_stop_strings=lambda matches: None,
        on_activity=lambda: None,
        on_fatal=lambda *args: pytest.fail(f"Output pipeline failed: {args}"),
    )
    try:
        core.add_requests(
            [
                AdmissionRecord("post", post_prompt, SamplingParams(max_tokens=32, ignore_eos=True), eos_token_id=0),
                AdmissionRecord("pre", pre_prompt, SamplingParams(max_tokens=32, ignore_eos=True), eos_token_id=0),
            ]
        )
        # Step 1 under v0: post finishes prefill and samples a token; pre only prefills 4 of 20 prompt tokens.
        so = core.schedule()
        rids = list(so.num_scheduled_tokens)
        sampled = [[65] if core.request_state(r)[1] >= core.request_state(r)[2] else [] for r in rids]
        outs = core.commit(
            so,
            ModelRunnerOutput(
                req_ids=rids,
                req_id_to_index={r: i for i, r in enumerate(rids)},
                sampled_token_ids=sampled,
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        pipeline.clear_kv_versions(core.drain_preempted_kv_requests())
        pipeline.submit(outs, weight_version=0)

        # Swap v0 -> v1: both hold v0 KV.
        marked = core.mark_kv_weight_version(0)
        assert marked == {"post": 0, "pre": 0}
        pipeline.submit_kv_versions(marked)
        assert pre_out.outputs[0].oldest_kv_weight_version == 0
        assert pre_out.outputs[0].weight_version_segments is None
        assert post_out.outputs[0].oldest_kv_weight_version == 0
        assert post_out.outputs[0].weight_version_segments == [(0, 0)]

        # Run steps under v1 until "pre" is preempted (while still mid-prefill with 0 tokens emitted).
        pre_was_preempted = False
        for _ in range(30):
            so = core.schedule()
            rids = list(so.num_scheduled_tokens)
            sampled = [[66] if core.request_state(r)[1] >= core.request_state(r)[2] else [] for r in rids]
            outs = core.commit(
                so,
                ModelRunnerOutput(
                    req_ids=rids,
                    req_id_to_index={r: i for i, r in enumerate(rids)},
                    sampled_token_ids=sampled,
                    spec_token_ids=None,
                    logprobs=None,
                    prompt_logprobs_dict={},
                ),
            )
            cleared = core.drain_preempted_kv_requests()
            if "pre" in cleared:
                pre_was_preempted = True
            pipeline.clear_kv_versions(cleared)
            pipeline.submit(outs, weight_version=1)
            if pre_was_preempted:
                break

        assert pre_was_preempted, "Expected 'pre' to be preempted in tiny page pool"
        assert list(pre_out.outputs[0].token_ids) == []
        # "pre" was preempted with 0 tokens emitted -> stale v0 KV provenance MUST be cleared!
        assert pre_out.outputs[0].oldest_kv_weight_version is None
        # Also verify clearing a post-token request preserves its v0 provenance.
        pipeline.clear_kv_versions({"post"})
        assert post_out.outputs[0].oldest_kv_weight_version == 0

        # Once "pre" later emits its first token under v1, oldest_kv_weight_version becomes 1 (not 0!).
        pipeline.submit({0: EngineCoreOutputs(outputs=[EngineCoreOutput("pre", [67])], timestamp=9.0)}, weight_version=1)
        assert pre_out.outputs[0].oldest_kv_weight_version == 1
        assert pre_out.outputs[0].weight_version_segments == [(0, 1)]
    finally:
        core.shutdown()


def test_stale_kv_parent_ids_deduplicates_n_gt_1_children_in_order():
    spec = FullAttentionSpec(page_size=4, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    core = EngineCoreDriver(
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec, layer_names=None)],
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=64,
        num_pages=16,
        enable_prefix_caching=False,
        async_scheduling=False,
    )
    try:
        core.add_requests(
            [
                AdmissionRecord("p1-0", [1, 2], SamplingParams(max_tokens=4), eos_token_id=0, parent_request_id="p1"),
                AdmissionRecord("p2-0", [1, 2], SamplingParams(max_tokens=4), eos_token_id=0, parent_request_id="p2"),
                AdmissionRecord("p1-1", [1, 2], SamplingParams(max_tokens=4), eos_token_id=0, parent_request_id="p1"),
                AdmissionRecord("solo", [1, 2], SamplingParams(max_tokens=4), eos_token_id=0, parent_request_id=None),
            ]
        )
        so = core.schedule()
        rids = list(so.num_scheduled_tokens)
        core.commit(
            so,
            ModelRunnerOutput(
                req_ids=rids,
                req_id_to_index={r: i for i, r in enumerate(rids)},
                sampled_token_ids=[[3] for _ in rids],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
            ),
        )
        assert core.stale_kv_parent_ids(1, 0) == ["p1", "p2", "solo"]
    finally:
        core.shutdown()


def test_remote_engine_stream_preserves_weight_version_provenance():
    registry = RequestRegistry()
    output = RequestOutput(
        request_id="req-1",
        prompt="hi",
        prompt_token_ids=[1, 2],
        outputs=[
            CompletionOutput(
                index=0,
                text="ok",
                token_ids=[10, 11],
                finish_reason="length",
                weight_version_segments=[(0, 2), (1, 3)],
                oldest_kv_weight_version=1,
            )
        ],
        finished=True,
        accumulated_text="ok",
        delta_text="ok",
        num_generated_tokens=2,
        update_seq=1,
    )
    event = threading.Event()
    event.set()
    registry.outputs["req-1"] = output
    registry.events["req-1"] = event

    remote = object.__new__(RemoteEngineHandle)
    remote._registry = registry
    remote._fatal = None
    remote._channel = SimpleNamespace(alive=True, dead_reason=None)
    remote._plane = SimpleNamespace(notify_abort=lambda rid: None)
    remote._admission = SimpleNamespace(
        prepare_sampling_params_for_request=lambda sp, **kw: sp,
        add_request=lambda *args, **kwargs: None,
    )

    yielded = list(remote.stream("hi", SamplingParams(max_tokens=2), request_id="req-1"))
    assert len(yielded) == 1
    comp = yielded[0].outputs[0]
    assert comp.weight_version_segments == [(0, 2), (1, 3)]
    assert comp.oldest_kv_weight_version == 1
    assert yielded[0].oldest_weight_version == 2
    assert yielded[0].newest_weight_version == 3
    assert yielded[0].oldest_kv_weight_version == 1


# --------------------------------------------------------------------------- #22 & #49: BoundedAsyncRolloutCoordinator


@pytest.mark.parametrize("wait_flag", [False, True])
def test_coordinator_close_wait_true_and_false_wakes_blocked_consumers_cleanly(wait_flag):
    gate = threading.Event()
    started = threading.Event()

    def worker(_snapshot, payload):
        started.set()
        gate.wait(timeout=2.0)
        return payload

    coordinator = BoundedAsyncRolloutCoordinator(worker, max_workers=1, max_outstanding=2)
    coordinator.submit("slow", snapshot=PolicySnapshot(version=0, state=None))
    assert started.wait(timeout=2.0)

    errors: list[BaseException] = []

    def consumer():
        try:
            coordinator.next_completed(timeout=None)
        except BaseException as exc:
            errors.append(exc)

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    # Wait until consumer registers as a waiter.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with coordinator._lock:
            if coordinator._waiters == 1:
                break
        time.sleep(0.005)

    if wait_flag:
        # Even with wait=True, consumer must wake BEFORE executor.shutdown finishes (before gate is set).
        closer = threading.Thread(target=lambda: coordinator.close(wait=True, cancel_pending=True), daemon=True)
        closer.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "next_completed(timeout=None) blocked on close(wait=True)!"
        gate.set()
        closer.join(timeout=2.0)
        assert not closer.is_alive()
    else:
        coordinator.close(wait=False, cancel_pending=True)
        t.join(timeout=2.0)
        gate.set()
        assert not t.is_alive()

    assert len(errors) == 1 and isinstance(errors[0], RuntimeError) and "closed" in str(errors[0])
    assert coordinator.credits_in_use == 0
    assert coordinator.outstanding_count == 0


def test_coordinator_next_completed_enforces_monotonic_deadline_despite_stale_ids():
    gate = threading.Event()
    coordinator = BoundedAsyncRolloutCoordinator(
        lambda _s, p: gate.wait(timeout=5.0) and p,
        max_workers=1,
        max_outstanding=2,
    )
    try:
        coordinator.submit("slow", snapshot=PolicySnapshot(version=0, state=None))
        stop_injector = threading.Event()

        def inject_stale():
            while not stop_injector.wait(0.02):
                coordinator._completion_ids.put_nowait(999_999)

        injector = threading.Thread(target=inject_stale, daemon=True)
        injector.start()
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            coordinator.next_completed(timeout=0.12)
        elapsed = time.monotonic() - t0
        stop_injector.set()
        injector.join(timeout=2.0)
        assert elapsed < 0.6, f"stale dequeued IDs reset next_completed timeout ({elapsed:.2f}s)"
    finally:
        gate.set()
        coordinator.close(wait=True)


# --------------------------------------------------------------------------- #10, #20, #21, #50: AsyncRolloutMixin


@pytest.mark.parametrize("use_hook", [True, False])
def test_async_mixin_refreshes_behavior_parity_gate_on_consumer_thread(monkeypatch, use_hook):
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(jax, "process_count", lambda: 1)

    seen_gates: list[int] = []
    hook_calls = 0

    def preprocess(*, state, batch, produced_at_step):
        # Simulate background lookahead worker stamping behavior_parity_gate=1 before step 0 completes.
        del state
        out_batch = dict(batch)
        out_batch["behavior_parity_gate"] = jnp.asarray(1, dtype=jnp.int32)
        return AsyncRolloutResult(out_batch, {}, produced_at_step, 0.0)

    def execute(*, state, batch, informations):
        del informations
        gate_val = batch["behavior_parity_gate"]
        assert isinstance(gate_val, jax.Array) and gate_val.dtype == jnp.int32
        seen_gates.append(int(np.asarray(gate_val)))
        return (
            SimpleNamespace(step=state.step + 1, esurge_cache_scope_key="test"),
            SimpleNamespace(loss=0.0, accuracy=None),
            None,
        )

    def on_step_end(*, state, metrics, step):
        del step
        trainer._startup_behavior_parity_checked = True
        return state, metrics

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=3,
        arguments=SimpleNamespace(
            async_mode=True,
            max_inflight_tasks=4,
            queue_maxsize=4,
            max_staleness=4,
            weight_sync_steps=100,
            request_timeout=2.0,
            heartbeat_stale_after_s=2.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _startup_behavior_parity_checked=False,
        _async_grpo_train_compiled_once=True,
        _get_next_batch=lambda it, ds: (next(it), it),
        _sync_rollout_snapshot=lambda *a, **kw: AsyncRolloutMixin._sync_rollout_snapshot(trainer, *a, **kw),
        _take_rollout_snapshot=lambda state, *, policy_step, cache_scope_key: SimpleNamespace(
            state=SimpleNamespace(step=policy_step),
            policy_step=policy_step,
            nbytes=0,
            release=lambda: None,
        ),
        _preprocess_async_rollout=preprocess,
        _execute_preprocessed_train_step=execute,
        on_step_start=lambda **kwargs: kwargs["state"],
        on_step_end=on_step_end,
        train_tracker=SimpleNamespace(trace_compilation=_NullContext),
        _maybe_start_profiler=lambda *args: None,
        apply_training_hooks=lambda **kwargs: kwargs["metrics"],
        scheduler=None,
        _backward_flops_per_token=0,
        _extra_backward_flops_per_token=0,
        training_batch_size=1,
        log_metrics=lambda **kwargs: None,
        _should_save_tpu_preemption_checkpoint=lambda *args: False,
        log_weight_distribution=lambda **kwargs: None,
        log_watchers=lambda **kwargs: None,
        maybe_generate=lambda **kwargs: None,
        maybe_benchmark=lambda **kwargs: None,
        _save_checkpoint_for_step=lambda **kwargs: None,
        _should_run_evaluation=lambda *args: False,
        _profiler_should_block_until_ready=lambda: False,
    )
    if use_hook:

        def refresh_hook(batch):
            nonlocal hook_calls
            hook_calls += 1
            updated = dict(batch)
            pending = not bool(trainer._startup_behavior_parity_checked)
            updated["behavior_parity_gate"] = jnp.asarray(int(pending), dtype=jnp.int32)
            return updated

        trainer._refresh_behavior_parity_gate_for_step = refresh_hook

    state, error, _ = AsyncRolloutMixin._train_epoch_async(
        trainer,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": i} for i in range(3)),
        SimpleNamespace(update=lambda **kw: (0.0, None)),
        SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=3,
    )
    assert error is None
    assert state.step == 3
    assert seen_gates == [1, 0, 0], f"use_hook={use_hook}: prefetched lookahead kept stale parity gate {seen_gates}"
    if use_hook:
        assert hook_calls == 3


def test_async_mixin_execute_preprocessed_step_delegates_to_dispatch_and_buckets():
    # Case (a): delegates through _dispatch_train_step (supports execution-mode gradient accumulation).
    dispatch_calls: list[tuple[object, object]] = []

    class _DispatchTrainer(AsyncRolloutMixin):
        pruning_module = None
        _has_buckets = False

        def _dispatch_train_step(self, state, batch):
            dispatch_calls.append((state, batch))
            return state.replace(step=state.step + 1), LossMetrics(loss=jnp.asarray(1.5))

        def sharded_training_step_function(self, *args, **kwargs):
            raise AssertionError("should delegate through _dispatch_train_step, not sharded_training_step_function")

    class _FakeState:
        def __init__(self, step=0, graphdef="base-gd"):
            self.step = step
            self.graphdef = graphdef

        def replace(self, **kwargs):
            return _FakeState(step=kwargs.get("step", self.step), graphdef=kwargs.get("graphdef", self.graphdef))

    t1 = _DispatchTrainer()
    s_out, m_out, err = t1._execute_preprocessed_train_step(_FakeState(0), {"x": jnp.ones((2,))}, {"k": 7})
    assert err is None
    assert len(dispatch_calls) == 1
    assert s_out.step == 1
    assert m_out.other_metrics == {"k": 7}

    # Case (b): _has_buckets swaps bucket graphdef before call and restores base graphdef afterwards.
    seen_bucket_graphdefs: list[str] = []

    class _BucketTrainer(AsyncRolloutMixin):
        pruning_module = None
        _has_buckets = True
        _bucket_for_step = 1

        def __init__(self):
            self._bucket_graphdefs = ["gd-0", "gd-1"]
            self._bucket_extra_args = [(), ("extra",)]
            self._bucket_static_args = [(), ("static",)]

            def fn1(state, batch, extra, static):
                seen_bucket_graphdefs.append(state.graphdef)
                assert extra == "extra" and static == "static"
                return state.replace(step=state.step + 1), LossMetrics(loss=jnp.asarray(0.5))

            self._bucket_step_fns = [None, fn1]

    t2 = _BucketTrainer()
    s2, _m2, err2 = t2._execute_preprocessed_train_step(_FakeState(0, "base-gd"), {"x": jnp.ones((2,))}, {})
    assert err2 is None
    assert seen_bucket_graphdefs == ["gd-1"]
    assert s2.graphdef == "base-gd"
    assert s2.step == 1


def test_async_mixin_control_exceptions_caught_in_lookahead_fetch_and_stale_drop(monkeypatch):
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(jax, "process_count", lambda: 1)

    # Case (a): EasyDeLPreemptionSignal during lookahead fetch_batch() is returned cleanly, not raised.
    fetch_calls = 0

    def get_batch_lookahead_preempt(it, ds):
        nonlocal fetch_calls
        fetch_calls += 1
        if fetch_calls == 2:
            raise EasyDeLPreemptionSignal("preempt in lookahead fetch")
        return next(it), it

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=4,
        arguments=SimpleNamespace(
            async_mode=True,
            max_inflight_tasks=2,
            queue_maxsize=2,
            max_staleness=2,
            weight_sync_steps=100,
            request_timeout=2.0,
            heartbeat_stale_after_s=2.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=get_batch_lookahead_preempt,
        _sync_rollout_snapshot=lambda *a, **kw: AsyncRolloutMixin._sync_rollout_snapshot(trainer, *a, **kw),
        _take_rollout_snapshot=lambda state, *, policy_step, cache_scope_key: SimpleNamespace(
            state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, release=lambda: None
        ),
        _preprocess_async_rollout=lambda *, state, batch, produced_at_step: AsyncRolloutResult(
            batch, {}, produced_at_step, 0.0
        ),
        on_step_start=lambda **kwargs: kwargs["state"],
    )
    state, exc, _ = AsyncRolloutMixin._train_epoch_async(
        trainer,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": i} for i in range(4)),
        SimpleNamespace(update=lambda **kw: (0.0, None)),
        SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=4,
    )
    assert isinstance(exc, EasyDeLPreemptionSignal)
    assert state.step == 0

    # Case (b): EasyDeLBreakRequest during stale-drop recovery is returned cleanly.
    preprocess_count = 0

    def preprocess_force_stale(*, state, batch, produced_at_step):
        nonlocal preprocess_count
        preprocess_count += 1
        if preprocess_count == 1:
            # Force staleness > max_staleness (=0) so stale-drop recovery runs.
            return AsyncRolloutResult(batch, {}, produced_at_step - 5, 0.0)
        raise EasyDeLBreakRequest("break during stale-drop")

    trainer2 = SimpleNamespace(
        data_collator=None,
        max_training_steps=4,
        arguments=SimpleNamespace(
            async_mode=True,
            max_inflight_tasks=1,
            queue_maxsize=1,
            max_staleness=0,
            weight_sync_steps=1,
            request_timeout=2.0,
            heartbeat_stale_after_s=2.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=lambda it, ds: (next(it), it),
        _sync_rollout_snapshot=lambda *a, **kw: AsyncRolloutMixin._sync_rollout_snapshot(trainer2, *a, **kw),
        _take_rollout_snapshot=lambda state, *, policy_step, cache_scope_key: SimpleNamespace(
            state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, release=lambda: None
        ),
        _preprocess_async_rollout=preprocess_force_stale,
        on_step_start=lambda **kwargs: kwargs["state"],
    )
    state2, exc2, _ = AsyncRolloutMixin._train_epoch_async(
        trainer2,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": i} for i in range(4)),
        SimpleNamespace(update=lambda **kw: (0.0, None)),
        SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=4,
    )
    assert isinstance(exc2, EasyDeLBreakRequest)
    assert state2.step == 0


def test_async_mixin_release_generation_runtime_waits_for_retiring_workers(monkeypatch):
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(jax, "process_count", lambda: 1)

    worker_running = threading.Event()
    release_worker = threading.Event()
    parent_released = threading.Event()
    parent_release_calls: list[bool] = []

    class _BaseStub:
        def release_generation_runtime(self, *, state=None, extra_states=(), clear_esurge_compiled_cache=False):
            del state, extra_states, clear_esurge_compiled_cache
            # Worker MUST already be idle when parent release runs!
            parent_release_calls.append(bool(getattr(self, "_async_grpo_rollout_idle", False)))
            parent_released.set()

    class _TrainerStub(AsyncRolloutMixin, _BaseStub):
        def __init__(self):
            self.data_collator = None
            self.max_training_steps = 4
            self.arguments = SimpleNamespace(
                async_mode=True,
                max_inflight_tasks=2,
                queue_maxsize=2,
                max_staleness=4,
                weight_sync_steps=100,
                request_timeout=2.0,
                heartbeat_stale_after_s=2.0,
            )
            self._async_grpo_train_compiled_once = True
            self.train_tracker = SimpleNamespace(trace_compilation=_NullContext)

        def _get_next_batch(self, it, ds):
            del ds
            return next(it), it

        def _take_rollout_snapshot(self, state, *, policy_step, cache_scope_key):
            del state, cache_scope_key
            return SimpleNamespace(state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, release=lambda: None)

        def _preprocess_async_rollout(self, *, state, batch, produced_at_step):
            del state
            if batch["index"] == 1:
                worker_running.set()
                assert release_worker.wait(5.0)
            return AsyncRolloutResult(batch, {}, produced_at_step, 0.0)

        def _execute_preprocessed_train_step(self, *, state, batch, informations):
            del batch, informations
            assert worker_running.wait(2.0)
            return state, SimpleNamespace(), EasyDeLBreakRequest("early exit")

        def on_step_start(self, *, state, step):
            del step
            return state

    trainer = _TrainerStub()
    try:
        _, err, _ = trainer._train_epoch_async(
            SimpleNamespace(step=0, esurge_cache_scope_key="test"),
            None,
            iter({"index": i} for i in range(4)),
            None,
            SimpleNamespace(start_step=lambda: None),
            None,
            0,
            epoch_start_step=0,
            epoch_end_step=4,
        )
        assert isinstance(err, EasyDeLBreakRequest)
        assert trainer._async_rollout_idle_event.is_set() is False

        # Start release_generation_runtime on another thread; it must WAIT until the retiring worker finishes.
        teardown = threading.Thread(target=lambda: trainer.release_generation_runtime(clear_esurge_compiled_cache=True), daemon=True)
        teardown.start()
        time.sleep(0.08)
        assert teardown.is_alive() and not parent_released.is_set()

        release_worker.set()
        teardown.join(timeout=3.0)
        assert not teardown.is_alive()
        assert parent_released.is_set()
        assert parent_release_calls == [True]
        assert trainer._async_rollout_idle_event.is_set() is True
    finally:
        release_worker.set()


def test_async_mixin_next_rollout_scheduled_zero_when_deferred_and_sync_time_not_replicated(monkeypatch):
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(jax, "process_count", lambda: 1)

    consumed_infos: list[dict] = []

    def execute(*, state, batch, informations):
        del batch
        consumed_infos.append(dict(informations))
        return (
            SimpleNamespace(step=state.step + 1, esurge_cache_scope_key="test"),
            SimpleNamespace(loss=0.0, accuracy=None),
            None,
        )

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=4,
        arguments=SimpleNamespace(
            async_mode=True,
            max_inflight_tasks=4,
            queue_maxsize=4,
            max_staleness=1,
            weight_sync_steps=100,
            request_timeout=2.0,
            heartbeat_stale_after_s=2.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=lambda it, ds: (next(it), it),
        _sync_rollout_snapshot=lambda *a, **kw: AsyncRolloutMixin._sync_rollout_snapshot(trainer, *a, **kw),
        _take_rollout_snapshot=lambda state, *, policy_step, cache_scope_key: SimpleNamespace(
            state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, release=lambda: None
        ),
        _preprocess_async_rollout=lambda *, state, batch, produced_at_step: AsyncRolloutResult(
            batch, {}, produced_at_step, 0.0
        ),
        _execute_preprocessed_train_step=execute,
        on_step_start=lambda **kwargs: kwargs["state"],
        on_step_end=lambda **kwargs: (kwargs["state"], kwargs["metrics"]),
        train_tracker=SimpleNamespace(trace_compilation=_NullContext),
        _maybe_start_profiler=lambda *args: None,
        apply_training_hooks=lambda **kwargs: kwargs["metrics"],
        scheduler=None,
        _backward_flops_per_token=0,
        _extra_backward_flops_per_token=0,
        training_batch_size=1,
        log_metrics=lambda **kwargs: None,
        _should_save_tpu_preemption_checkpoint=lambda *args: False,
        log_weight_distribution=lambda **kwargs: None,
        log_watchers=lambda **kwargs: None,
        maybe_generate=lambda **kwargs: None,
        maybe_benchmark=lambda **kwargs: None,
        _save_checkpoint_for_step=lambda **kwargs: None,
        _should_run_evaluation=lambda *args: False,
        _profiler_should_block_until_ready=lambda: False,
    )

    state, err, _ = AsyncRolloutMixin._train_epoch_async(
        trainer,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": i} for i in range(4)),
        SimpleNamespace(update=lambda **kw: (0.0, None)),
        SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=4,
    )
    assert err is None and state.step == 4
    # Step 0 (v0): schedules step 1 (v0) -> next_rollout_scheduled == 1
    # Step 1 (v0): step 2 would have lag 2 > max_staleness=1 -> admission defers ALL lookahead -> MUST be 0!
    # Step 2 (v2): schedules step 3 (v2) -> next_rollout_scheduled == 1
    # Step 3 (v2): final step -> next_rollout_scheduled == 0
    assert [info["async_rollout/next_rollout_scheduled"] for info in consumed_infos] == [1, 0, 1, 0]


def test_async_mixin_stale_drop_does_not_replicate_sync_time_to_resubmitted_items(monkeypatch):
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(jax, "process_count", lambda: 1)

    sync_calls = 0
    forced_stale = False
    consumed_sync_times: list[float] = []

    def fake_sync(snapshot, policy_state, policy_step, *, force=False, can_release=True, cache_scope_key="test"):
        nonlocal sync_calls
        del policy_state, can_release, cache_scope_key
        if snapshot is not None and not force:
            return snapshot, 0.0
        sync_calls += 1
        snap = SimpleNamespace(
            state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, release=lambda: None
        )
        # Assign a distinctive sync time on the forced stale-drop resync.
        return snap, 2.5 if force else 0.1

    def preprocess(*, state, batch, produced_at_step):
        nonlocal forced_stale
        del state
        # On batch 1 (consumed at step 1 while batch 2 and batch 3 are already queued),
        # return an artificially stale produced_at_step so stale-drop recovery triggers.
        if batch["index"] == 1 and not forced_stale:
            forced_stale = True
            return AsyncRolloutResult(batch, {}, -10, 0.0)
        return AsyncRolloutResult(batch, {}, produced_at_step, 0.0)

    def execute(*, state, batch, informations):
        del batch
        consumed_sync_times.append(float(informations["async_rollout/policy_sync_time"]))
        return (
            SimpleNamespace(step=state.step + 1, esurge_cache_scope_key="test"),
            SimpleNamespace(loss=0.0, accuracy=None),
            None,
        )

    trainer = SimpleNamespace(
        data_collator=None,
        max_training_steps=4,
        arguments=SimpleNamespace(
            async_mode=True,
            max_inflight_tasks=4,
            queue_maxsize=4,
            max_staleness=4,
            weight_sync_steps=100,
            request_timeout=2.0,
            heartbeat_stale_after_s=2.0,
            learning_rate=0.1,
            max_length=4,
            merge_lora_before_save=False,
        ),
        _async_grpo_train_compiled_once=True,
        _get_next_batch=lambda it, ds: (next(it), it),
        _sync_rollout_snapshot=fake_sync,
        _preprocess_async_rollout=preprocess,
        _execute_preprocessed_train_step=execute,
        on_step_start=lambda **kwargs: kwargs["state"],
        on_step_end=lambda **kwargs: (kwargs["state"], kwargs["metrics"]),
        train_tracker=SimpleNamespace(trace_compilation=_NullContext),
        _maybe_start_profiler=lambda *args: None,
        apply_training_hooks=lambda **kwargs: kwargs["metrics"],
        scheduler=None,
        _backward_flops_per_token=0,
        _extra_backward_flops_per_token=0,
        training_batch_size=1,
        log_metrics=lambda **kwargs: None,
        _should_save_tpu_preemption_checkpoint=lambda *args: False,
        log_weight_distribution=lambda **kwargs: None,
        log_watchers=lambda **kwargs: None,
        maybe_generate=lambda **kwargs: None,
        maybe_benchmark=lambda **kwargs: None,
        _save_checkpoint_for_step=lambda **kwargs: None,
        _should_run_evaluation=lambda *args: False,
        _profiler_should_block_until_ready=lambda: False,
    )

    state, err, _ = AsyncRolloutMixin._train_epoch_async(
        trainer,
        SimpleNamespace(step=0, esurge_cache_scope_key="test"),
        None,
        iter({"index": i} for i in range(4)),
        SimpleNamespace(update=lambda **kw: (0.0, None)),
        SimpleNamespace(start_step=lambda: None, calculate=lambda **kw: {}),
        None,
        0,
        epoch_start_step=0,
        epoch_end_step=4,
    )
    assert err is None and state.step == 4
    # Step 1 triggered stale-drop resync (2.5s); resubmitted steps 2 and 3 MUST have 0.0 sync time (not replicated!).
    assert consumed_sync_times == [0.1, 2.5, 0.0, 0.0]


def test_async_mixin_retiring_worker_calling_release_generation_runtime_does_not_self_deadlock(monkeypatch):
    monkeypatch.setattr(jax, "device_get", lambda value: value)
    monkeypatch.setattr(jax, "process_count", lambda: 1)

    worker_running = threading.Event()
    main_in_release = threading.Event()
    worker_called_release = threading.Event()
    parent_release_threads: list[tuple[int, bool]] = []
    main_thread_id = threading.get_ident()

    class _BaseStub:
        def release_generation_runtime(self, *, state=None, extra_states=(), clear_esurge_compiled_cache=False):
            del state, extra_states, clear_esurge_compiled_cache
            parent_release_threads.append((threading.get_ident(), bool(getattr(self, "_async_grpo_rollout_idle", False))))

    class _TrainerStub(AsyncRolloutMixin, _BaseStub):
        def __init__(self):
            self.data_collator = None
            self.max_training_steps = 4
            self.arguments = SimpleNamespace(
                async_mode=True,
                max_inflight_tasks=2,
                queue_maxsize=2,
                max_staleness=4,
                weight_sync_steps=100,
                request_timeout=3.0,
                heartbeat_stale_after_s=3.0,
            )
            self._async_grpo_train_compiled_once = True
            self.train_tracker = SimpleNamespace(trace_compilation=_NullContext)

        def _get_next_batch(self, it, ds):
            del ds
            return next(it), it

        def _take_rollout_snapshot(self, state, *, policy_step, cache_scope_key):
            del state, cache_scope_key
            return SimpleNamespace(
                state=SimpleNamespace(step=policy_step), policy_step=policy_step, nbytes=0, release=lambda: None
            )

        def _preprocess_batch_input(self, *, state, batch, is_train=True):
            del is_train
            if batch["index"] == 1:
                assert self._async_grpo_force_old_logps is True
                worker_running.set()
                assert main_in_release.wait(3.0)
                # Simulates BaseTrainer.rollout(..., release_runtime_after_generation=True)
                # calling self.release_generation_runtime(state=state) on the retiring worker thread.
                self.release_generation_runtime(state=state)
                worker_called_release.set()
            return dict(batch), {}

        def _execute_preprocessed_train_step(self, *, state, batch, informations):
            del batch, informations
            assert worker_running.wait(2.0)
            return state, SimpleNamespace(), EasyDeLBreakRequest("early exit")

        def on_step_start(self, *, state, step):
            del step
            return state

    trainer = _TrainerStub()
    try:
        _, err, _ = trainer._train_epoch_async(
            SimpleNamespace(step=0, esurge_cache_scope_key="test"),
            None,
            iter({"index": i} for i in range(4)),
            None,
            SimpleNamespace(start_step=lambda: None),
            None,
            0,
            epoch_start_step=0,
            epoch_end_step=4,
        )
        assert isinstance(err, EasyDeLBreakRequest)
        assert trainer._async_rollout_idle_event.is_set() is False

        # Signal the retiring worker right after main thread enters release_generation_runtime().
        threading.Timer(0.03, main_in_release.set).start()
        t0 = time.monotonic()
        trainer.release_generation_runtime(clear_esurge_compiled_cache=True)
        elapsed = time.monotonic() - t0

        assert worker_called_release.is_set()
        assert elapsed < 1.5, f"Self-deadlock detected in release_generation_runtime ({elapsed:.2f}s)"
        # Only the main thread ran super().release_generation_runtime(), and only once workers were idle.
        assert parent_release_threads == [(main_thread_id, True)]
        assert trainer._async_rollout_idle_event.is_set() is True
    finally:
        main_in_release.set()


def test_jax_transfer_inline_pull_preserves_keyboard_interrupt_and_backend_close_race_safe():
    class _InterruptConn:
        def pull(self, uuid, targets, timeout=None):
            del uuid, targets, timeout
            raise KeyboardInterrupt("ctrl-c")

    class _Server:
        def address(self):
            return "127.0.0.1:7777"

        def connect(self, address):
            del address
            return _InterruptConn()

        def await_pull(self, uuid, arrays, timeout=None):
            del uuid, arrays, timeout

    backend = JaxTransferServerBackend(server=_Server())
    handle = backend.stage(1, {"a": jnp.ones((2,), dtype=jnp.float32), "b": jnp.ones((2,), dtype=jnp.float32)})
    offer = WeightOffer(offer_id=1, subscriber="s0", version=1, predecessor=None, descriptor=backend.describe(handle))

    # KeyboardInterrupt during inline deadline=None pull must NOT be swallowed into WeightTransportError.
    client = JaxTransferClient(server=_Server())
    with pytest.raises(KeyboardInterrupt, match="ctrl-c"):
        client.fetch(offer, timeout=None)

    # Concurrent close() during backend.serve() raises WeightTransportClosed, not AttributeError.
    call_count = 0

    class _ClosingMidServeServer:
        def address(self):
            return "127.0.0.1:7778"

        def await_pull(self, uuid, arrays, timeout=None):
            nonlocal call_count
            del uuid, arrays, timeout
            call_count += 1
            if call_count == 1:
                backend2.close()

    backend2 = JaxTransferServerBackend(server=_ClosingMidServeServer())
    handle2 = backend2.stage(1, {"a": jnp.ones((2,), dtype=jnp.float32), "b": jnp.ones((2,), dtype=jnp.float32)})
    with pytest.raises(WeightTransportClosed):
        backend2.serve(handle2, offer_id=1, timeout=5.0)


def test_socket_channel_deadline_restores_previous_socket_timeout():
    left, right = socket.socketpair()
    try:
        ch_left = SocketMessageChannel(left)
        ch_right = SocketMessageChannel(right)
        for initial_timeout in (None, 12.5):
            left.settimeout(initial_timeout)
            right.settimeout(initial_timeout)
            ch_left.send({"op": "ping"}, deadline=time.monotonic() + 2.0)
            msg = ch_right.recv(deadline=time.monotonic() + 2.0)
            assert msg == {"op": "ping"}
            assert left.gettimeout() == initial_timeout
            assert right.gettimeout() == initial_timeout
    finally:
        left.close()
        right.close()


def test_output_pipeline_flush_returns_false_when_stopped_before_barrier():
    _, pipeline, _ = _make_async_pipeline_shell()
    entered = threading.Event()
    release = threading.Event()
    orig_apply = pipeline._apply_kv_versions

    def gated_apply(item):
        if "blocker" in item.versions:
            entered.set()
            assert release.wait(5.0)
        return orig_apply(item)

    pipeline._apply_kv_versions = gated_apply
    pipeline.start()
    try:
        pipeline.submit_kv_versions({"blocker": 0})
        assert entered.wait(2.0)
        # Queue _STOP_SENTINEL before flush() queues its _Barrier, so worker exits before reaching the barrier.
        pipeline.queue.put(None)
        results: list[bool] = []
        flusher = threading.Thread(target=lambda: results.append(pipeline.flush(timeout=3.0)), daemon=True)
        flusher.start()
        time.sleep(0.05)
        release.set()
        flusher.join(timeout=3.0)
        assert not flusher.is_alive()
        assert results == [False]
    finally:
        release.set()
        pipeline.stop()
