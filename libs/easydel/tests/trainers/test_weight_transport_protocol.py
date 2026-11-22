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
"""Pure-python protocol tests for the versioned learner-to-sampler weight transport."""

import collections
import socket
import threading
import time

import pytest
from easydel.trainers._online_rl.weight_transport import (
    OfferOutcome,
    RemoteSubscriberBridge,
    RemoteWeightSubscriber,
    SocketMessageChannel,
    WeightOffer,
    WeightOfferSettled,
    WeightPublisher,
    WeightTransferStalled,
    WeightTransportError,
    WeightVersionChainError,
)

pytestmark = pytest.mark.cpu_ok


class ReleaseLog:
    def __init__(self):
        self.counts = collections.Counter()
        self.lock = threading.Lock()

    def __call__(self, snapshot):
        with self.lock:
            self.counts[snapshot] += 1


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_no_offer_without_request_and_newest_offered_on_request():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        publisher.publish(1, "w1", release=releases)
        publisher.publish(2, "w2", release=releases)
        # No request outstanding: nothing offered, and the superseded v1 snapshot is already released.
        assert releases.counts == {"w1": 1}
        assert publisher.staged_version == 2
        with pytest.raises(WeightTransportError, match="no outstanding request"):
            sub.receive(timeout=0.01)
        sub.request()
        offer = sub.receive(timeout=1.0)
        assert (offer.version, offer.predecessor) == (2, None)
        assert sub.pull(offer) == "w2"
        sub.ack(offer)
        assert sub.installed_version == 2
        assert publisher.installed_version("s0") == 2
        # A second request waits: nothing newer than the installed version exists.
        sub.request()
        assert sub.receive(timeout=0.05) is None


def test_single_outstanding_request_token():
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        with pytest.raises(WeightTransportError, match="outstanding request"):
            sub.request()
        publisher.publish(1, "w1")
        offer = sub.receive(timeout=1.0)
        with pytest.raises(WeightTransportError, match="unsettled offer"):
            sub.request()
        assert sub.decline(offer) is True
        sub.request()  # settled offer frees the token


def test_request_blocks_until_publish_from_another_thread():
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        got = []
        receiver = threading.Thread(target=lambda: got.append(sub.receive(timeout=5.0)))
        receiver.start()
        time.sleep(0.05)
        publisher.publish(7, "w7")
        receiver.join(timeout=5.0)
        assert got and got[0].version == 7


def test_version_chain_names_predecessor_and_explicit_jumps():
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0", installed_version=3)
        publisher.publish(4, "w4")
        sub.request()
        offer = sub.receive(timeout=1.0)
        assert (offer.predecessor, offer.skipped_versions) == (3, 0)
        sub.pull(offer)
        sub.ack(offer)
        publisher.publish(5, "w5")
        publisher.publish(9, "w9")
        sub.request()
        offer = sub.receive(timeout=1.0)
        assert (offer.version, offer.predecessor, offer.skipped_versions) == (9, 4, 4)


def test_version_chain_gap_is_detected_by_subscriber():
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0", installed_version=1)
        # The sampler's own view diverges from the publisher's (e.g. it reloaded older weights).
        sub._installed_version = 0
        publisher.publish(2, "w2")
        sub.request()
        with pytest.raises(WeightVersionChainError, match="predecessor 1"):
            sub.receive(timeout=1.0)
    offer = WeightOffer(offer_id=1, subscriber="x", version=3, predecessor=3)
    from easydel.trainers._online_rl.weight_transport import verify_offer_chain

    with pytest.raises(WeightVersionChainError, match="does not advance"):
        verify_offer_chain(offer, 3)


def test_publish_requires_increasing_versions():
    with WeightPublisher() as publisher:
        publisher.publish(2, "w2")
        with pytest.raises(ValueError, match="must increase"):
            publisher.publish(2, "again")


def test_supersede_unpulled_offer_releases_old_snapshot_and_reoffers():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1", release=releases)
        first = sub.receive(timeout=1.0)
        assert first.version == 1
        publisher.publish(2, "w2", release=releases)
        assert releases.counts == {"w1": 1}
        assert (first.offer_id, 1, "s0", OfferOutcome.SUPERSEDED) in publisher.settlements
        with pytest.raises(WeightOfferSettled) as err:
            sub.pull(first)
        assert err.value.outcome is OfferOutcome.SUPERSEDED
        second = sub.receive(timeout=1.0)
        assert (second.version, second.predecessor) == (2, None)
        assert sub.pull(second) == "w2"
        sub.ack(second)
    assert releases.counts == {"w1": 1, "w2": 1}


def test_in_flight_pull_keeps_snapshot_alive_across_newer_publish():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1", release=releases)
        offer = sub.receive(timeout=1.0)
        assert sub.pull(offer) == "w1"
        publisher.publish(2, "w2", release=releases)
        assert releases.counts == {}  # the pull still references w1; it is not superseded
        sub.ack(offer)
        assert releases.counts == {"w1": 1}
        assert publisher.installed_version("s0") == 1
    assert releases.counts == {"w1": 1, "w2": 1}


def test_exactly_once_settlement_under_decline_and_close():
    releases = ReleaseLog()
    publisher = WeightPublisher()
    subs = [publisher.subscribe(f"s{i}") for i in range(3)]
    for sub in subs:
        sub.request()
    publisher.publish(1, "w1", release=releases)
    offers = [sub.receive(timeout=1.0) for sub in subs]
    assert subs[0].decline(offers[0]) is True
    assert subs[0].decline(offers[0]) is False  # second decline is a no-op
    subs[1].pull(offers[1])  # in flight at close
    publisher.close()
    publisher.close()  # idempotent
    outcomes = {offer_id: outcome for offer_id, _, _, outcome in publisher.settlements}
    assert len(publisher.settlements) == 3
    assert outcomes == {
        offers[0].offer_id: OfferOutcome.DECLINED,
        offers[1].offer_id: OfferOutcome.CLOSED,
        offers[2].offer_id: OfferOutcome.CLOSED,
    }
    assert releases.counts == {"w1": 1}
    with pytest.raises(WeightOfferSettled):
        subs[1].ack(offers[1])
    assert len(publisher.settlements) == 3


def test_concurrent_ack_and_decline_settle_once():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        for round_index in range(50):
            sub = publisher.subscribe(f"s{round_index}")
            sub.request()
            publisher.publish(round_index + 1, f"w{round_index}", release=releases)
            offer = sub.receive(timeout=1.0)
            sub.pull(offer)
            barrier = threading.Barrier(2)
            results = []

            def do_ack(sub=sub, offer=offer, barrier=barrier, results=results):
                barrier.wait()
                try:
                    sub.ack(offer)
                    results.append("ack")
                except WeightOfferSettled:
                    results.append("ack-lost")

            def do_decline(sub=sub, offer=offer, barrier=barrier, results=results):
                barrier.wait()
                results.append("decline" if sub.decline(offer) else "decline-lost")

            threads = [threading.Thread(target=do_ack), threading.Thread(target=do_decline)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5.0)
            assert sorted(results) in (["ack", "decline-lost"], ["ack-lost", "decline"])
            matching = [s for s in publisher.settlements if s[0] == offer.offer_id]
            assert len(matching) == 1
            sub.close()
    # every superseded staged snapshot was released exactly once; the last is released at close
    assert all(count == 1 for count in releases.counts.values())
    assert len(releases.counts) == 50


def test_unsubscribe_settles_open_offer_as_closed():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1", release=releases)
        offer = sub.receive(timeout=1.0)
        sub.close()
        assert publisher.settlements[-1] == (offer.offer_id, 1, "s0", OfferOutcome.CLOSED)
        publisher.publish(2, "w2", release=releases)
        assert releases.counts == {"w1": 1}


def test_watchdog_settles_stalled_pull_and_raises():
    clock = FakeClock()
    releases = ReleaseLog()
    with WeightPublisher(pull_timeout=10.0, clock=clock) as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1", release=releases)
        offer = sub.receive(timeout=1.0)
        sub.pull(offer)
        clock.now = 5.0
        publisher.check_watchdog()  # within budget
        clock.now = 10.5
        with pytest.raises(WeightTransferStalled, match=r"pull_timeout=10\.0"):
            publisher.check_watchdog()
        assert publisher.settlements[-1] == (offer.offer_id, 1, "s0", OfferOutcome.STALLED)
        with pytest.raises(WeightTransferStalled):
            sub.ack(offer)
        publisher.publish(2, "w2", release=releases)  # error already reported once; publishing continues
        assert releases.counts == {"w1": 1}
        assert publisher.installed_version("s0") is None


def test_watchdog_error_surfaces_at_publish_without_taking_snapshot():
    clock = FakeClock()
    releases = ReleaseLog()
    with WeightPublisher(pull_timeout=1.0, clock=clock) as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1", release=releases)
        sub.pull(sub.receive(timeout=1.0))
        clock.now = 2.0
        with pytest.raises(WeightTransferStalled):
            publisher.publish(2, "w2", release=releases)
        assert (publisher.latest_version, publisher.staged_version) == (1, 1)
        # The stalled offer dropped its reference, but w1 is still the staged snapshot; w2 stays the caller's.
        assert releases.counts == {}
    assert releases.counts == {"w1": 1}


def test_background_watchdog_releases_stalled_snapshot():
    releases = ReleaseLog()
    with WeightPublisher(pull_timeout=0.05, watchdog_interval=0.01) as publisher:
        sub = publisher.subscribe("s0")
        sub.request()
        publisher.publish(1, "w1", release=releases)
        sub.pull(sub.receive(timeout=1.0))
        publisher.publish(2, "w2", release=releases)
        deadline = time.monotonic() + 5.0
        while releases.counts["w1"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert releases.counts["w1"] == 1
        with pytest.raises(WeightTransferStalled):
            publisher.check_watchdog()


class EchoFetcher:
    """Fetcher stand-in returning the descriptor's payload; the control path is what is under test."""

    def fetch(self, offer, *, timeout):
        del timeout
        return dict(offer.metadata)


def _bridge_pair(publisher, name, installed_version=None):
    left, right = socket.socketpair()
    bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), name, installed_version=installed_version)
    remote = RemoteWeightSubscriber(SocketMessageChannel(right), EchoFetcher(), installed_version=installed_version)
    return bridge, remote


def test_remote_subscriber_round_trip_over_socket_channel():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        bridge, remote = _bridge_pair(publisher, "remote")
        remote.request()
        publisher.publish(1, "w1", release=releases, metadata={"payload": "v1"})
        offer = remote.receive(timeout=2.0)
        assert (offer.version, offer.predecessor, offer.descriptor["kind"]) == (1, None, "in_process")
        assert remote.pull(offer) == {"payload": "v1"}
        remote.ack(offer)
        assert remote.installed_version == 1
        assert publisher.installed_version("remote") == 1
        # superseded offer is reported as such across the wire
        remote.request()
        publisher.publish(2, "w2", release=releases)
        stale = remote.receive(timeout=2.0)
        publisher.publish(3, "w3", release=releases)
        with pytest.raises(WeightOfferSettled) as err:
            remote.pull(stale)
        assert err.value.outcome is OfferOutcome.SUPERSEDED
        assert releases.counts == {"w1": 1, "w2": 1}
        remote.close()
        bridge.join(timeout=5.0)
        with pytest.raises(WeightTransportError):
            remote.request()


def test_remote_subscriber_detects_chain_gap():
    with WeightPublisher() as publisher:
        bridge, remote = _bridge_pair(publisher, "remote", installed_version=4)
        remote._installed_version = 2  # sampler restarted from an older checkpoint
        remote.request()
        publisher.publish(5, "w5")
        with pytest.raises(WeightVersionChainError):
            remote.receive(timeout=2.0)
        remote.close()
        bridge.join(timeout=5.0)


def test_bridge_settles_offer_when_remote_disconnects():
    releases = ReleaseLog()
    with WeightPublisher() as publisher:
        bridge, remote = _bridge_pair(publisher, "remote")
        remote.request()
        publisher.publish(1, "w1", release=releases)
        offer = remote.receive(timeout=2.0)
        remote._channel.close()  # abrupt disconnect, no close RPC
        bridge.join(timeout=5.0)
        assert publisher.settlements[-1] == (offer.offer_id, 1, "remote", OfferOutcome.CLOSED)
        publisher.publish(2, "w2", release=releases)
        assert releases.counts == {"w1": 1}


def test_token_authenticated_round_trip():
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote", auth_token="s3cret")
        remote = RemoteWeightSubscriber(SocketMessageChannel(right), EchoFetcher(), auth_token="s3cret")
        assert bridge.authenticated.wait(5.0) and bridge.auth_error is None
        remote.request()
        publisher.publish(1, "w1", metadata={"payload": "v1"})
        offer = remote.receive(timeout=2.0)
        assert remote.pull(offer) == {"payload": "v1"}
        remote.ack(offer)
        assert publisher.installed_version("remote") == 1
        remote.close()
        bridge.join(timeout=5.0)


@pytest.mark.parametrize("bridge_token,remote_token", [("right", "wrong"), ("right", None), (None, "right")])
def test_mismatched_tokens_fail_closed_and_serve_nothing(bridge_token, remote_token):
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote", auth_token=bridge_token)
        with pytest.raises(WeightTransportError):
            remote = RemoteWeightSubscriber(SocketMessageChannel(right), EchoFetcher(), auth_token=remote_token)
            # A tokenless sampler skips the handshake; its first RPC must then fail instead of being served.
            remote.request()
            publisher.publish(1, "w1")
            remote.receive(timeout=2.0)
        bridge.join(timeout=5.0)
        assert not bridge._thread.is_alive()
        # Nothing was acked, and the failed bridge unsubscribed its sampler.
        assert all(settled[3] is not OfferOutcome.ACKED for settled in publisher.settlements)
        with pytest.raises(KeyError):
            publisher.installed_version("remote")


def test_handshake_timeout_closes_a_silent_peer(monkeypatch):
    from easydel.trainers._online_rl import weight_transport

    monkeypatch.setattr(weight_transport, "_HANDSHAKE_TIMEOUT", 0.5)
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote", auth_token="s3cret")
        start = time.monotonic()
        bridge.join(timeout=5.0)  # the peer never answers the hello
        assert time.monotonic() - start < 4.0
        assert not bridge._thread.is_alive()
        assert isinstance(bridge.auth_error, WeightTransportError)
        assert not bridge.authenticated.is_set()
        right.close()


def test_reflected_nonce_or_mac_is_rejected():
    """A peer without the token echoes the bridge's own nonce, or its own MAC, back; both must fail."""
    for mode in ("nonce", "mac"):
        with WeightPublisher() as publisher:
            left, right = socket.socketpair()
            bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote", auth_token="s3cret")
            attacker = SocketMessageChannel(right)
            hello = attacker.recv()
            if mode == "nonce":
                attacker.send(hello)
            else:
                attacker.send({"op": "hello", "nonce": "ab" * 32})
                attacker.send(attacker.recv())  # reflect the bridge's MAC as our proof
            bridge.join(timeout=5.0)
            assert isinstance(bridge.auth_error, WeightTransportError), mode
            assert not bridge.authenticated.is_set()
            attacker.close()


def test_replayed_proof_from_an_earlier_session_is_rejected():
    from easydel.trainers._online_rl.weight_transport import _handshake_mac

    with WeightPublisher() as publisher:
        # Proof computed for an earlier session's nonces, as a passive observer could have captured it.
        stale = _handshake_mac(b"s3cret", "sampler", b"cd" * 32, b"ef" * 32)
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote", auth_token="s3cret")
        attacker = SocketMessageChannel(right)
        attacker.recv()
        attacker.send({"op": "hello", "nonce": "cd" * 32})
        attacker.recv()
        attacker.send({"op": "auth", "mac": stale})
        bridge.join(timeout=5.0)
        assert isinstance(bridge.auth_error, WeightTransportError)
        assert not bridge.authenticated.is_set()
        attacker.close()


def test_non_local_peer_without_token_is_refused(monkeypatch):
    from easydel.trainers._online_rl import weight_transport

    monkeypatch.setattr(weight_transport, "_peer_is_local", lambda sock: False)
    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote")
        bridge.join(timeout=5.0)
        assert isinstance(bridge.auth_error, WeightTransportError)
        assert "auth_token" in str(bridge.auth_error)
        assert not bridge.authenticated.is_set()
        with pytest.raises(WeightTransportError):
            RemoteWeightSubscriber(SocketMessageChannel(right), EchoFetcher())


def test_oversized_frame_before_authentication_is_rejected():
    import struct

    with WeightPublisher() as publisher:
        left, right = socket.socketpair()
        bridge = RemoteSubscriberBridge(publisher, SocketMessageChannel(left), "remote", auth_token="s3cret")
        right.sendall(struct.pack(">I", 32 << 20))  # claims a 32 MiB "hello"; must not be buffered
        bridge.join(timeout=5.0)
        assert isinstance(bridge.auth_error, WeightTransportError)
        assert not bridge.authenticated.is_set()
        right.close()
