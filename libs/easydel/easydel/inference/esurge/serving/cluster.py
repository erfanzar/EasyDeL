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

"""Per-host facts for a multi-host serving world: addresses and the shared coordination token.

The JAX world itself is joined by ``import easydel`` (TPU slices are detected from GCE metadata), so
nothing here starts processes or numbers ranks.
"""

from __future__ import annotations

import secrets
import typing as tp

from .config import INTERNAL_ADDRESS

_TOKEN_KEY = "easydel/esurge/serving/coordination_token/{port}"


class _KeyValueClient(tp.Protocol):
    def key_value_set(self, key: str, value: str, allow_overwrite: bool = False) -> None: ...

    def blocking_key_value_get(self, key: str, timeout_in_ms: int) -> str: ...


def _distributed_client() -> _KeyValueClient | None:
    from jax._src.distributed import global_state

    return getattr(global_state, "client", None)


def shared_coordination_token(
    control_port: int,
    *,
    is_leader: bool,
    timeout_s: float,
    client: _KeyValueClient | None = None,
) -> str:
    """Return the ZMQ coordination token every process of this launch agrees on.

    The leader generates a fresh 256-bit token and publishes it in the JAX coordination service's key-value
    store; workers block until it appears. The token lives only in process memory and that store, so it changes
    every launch and is never on disk. Anything able to talk to the JAX coordination service can read it, which
    is the same trust boundary as joining the JAX world itself: keep both on internal addresses.

    Args:
        control_port: Control-plane port, part of the key so engines on different ports do not collide.
        is_leader: Whether this process is the coordination leader (``jax.process_index() == 0``).
        timeout_s: Seconds a worker waits for the leader.
        client: Key-value client; defaults to the initialized ``jax.distributed`` client.

    Raises:
        RuntimeError: When ``jax.distributed`` is not initialized or the leader never published.
    """
    client = client if client is not None else _distributed_client()
    if client is None:
        raise RuntimeError("jax.distributed is not initialized; a multi-host serving world needs it")
    key = _TOKEN_KEY.format(port=int(control_port))
    if is_leader:
        token = secrets.token_urlsafe(32)
        client.key_value_set(key, token, allow_overwrite=False)
        return token
    try:
        return str(client.blocking_key_value_get(key, max(1, int(float(timeout_s) * 1000))))
    except Exception as exc:
        raise RuntimeError(f"the coordination leader did not publish a token within {timeout_s:.0f}s") from exc


def resolve_bind_host(value: str, *, internal_ip: tp.Callable[[], str | None] | None = None) -> str:
    """Turn a configured bind address into a concrete one; ``internal`` becomes this VM's internal IP.

    Raises:
        RuntimeError: For ``internal`` when the internal address cannot be determined (not on GCE).
    """
    if value != INTERNAL_ADDRESS:
        return value
    if internal_ip is None:
        from eray.gcp import detect_internal_ip as internal_ip
    address = internal_ip()
    if not address:
        raise RuntimeError("bind address `internal` needs the GCE metadata server; set an explicit address instead")
    return address


def public_urls(
    bind_host: str,
    port: int,
    *,
    internal_ip: tp.Callable[[], str | None] | None = None,
    external_ip: tp.Callable[[], str | None] | None = None,
) -> dict[str, str]:
    """Base URLs clients can use for a server bound to ``bind_host:port``.

    ``internal`` is always reported; ``external`` only when the server listens on all interfaces and the VM
    has an external address (it is reachable only if a firewall rule allows the port).
    """
    if internal_ip is None or external_ip is None:
        from eray.gcp import detect_external_ip, detect_internal_ip

        internal_ip = internal_ip or detect_internal_ip
        external_ip = external_ip or detect_external_ip
    wildcard = bind_host in {"0.0.0.0", "::", ""}
    internal = (internal_ip() if wildcard else bind_host) or bind_host
    urls = {"internal": f"http://{internal}:{port}/v1"}
    if wildcard:
        external = external_ip()
        if external:
            urls["external"] = f"http://{external}:{port}/v1"
    return urls
