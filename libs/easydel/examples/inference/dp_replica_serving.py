# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Data-parallel eSurge serving — N independent replicas behind one router.

When a model fits comfortably on a chip subset, running several independent
replicas and load-balancing across them beats sharding one copy over the
whole pod. Each replica is its own JAX world exposing the eSurge request
plane; a :class:`RoutedEngine` fronts them with prefix-affinity +
join-shortest-queue routing, and one OpenAI API server serves the group.

Two ways to run it:

1. ONE COMMAND (recommended). Let eLarge launch the replicas and the router
   for you — see ``dp_replica_serve.yaml`` next to this file:

       python -m easydel.scripts.elarge --config dp_replica_serve.yaml

   It carves the host's chips into ``replicas.count`` engines, each on its
   own chip subset, and starts the routed API server in front. This is the
   path validated on a v5p-8 (two replicas × two chips).

2. PROGRAMMATIC (this script). Connect to replica engines that are already
   running their request plane (launched with ``elarge serve_replica`` on
   each host/slice, or with any process that built an eSurge engine with
   ``distributed={"coordination": "zmq", ...}``), then route across them.
   Use this when you own replica placement (multiple pods, a custom
   gateway, autoscaling) rather than a single-host chip split.
"""

from easydel.inference import eSurgeApiServer
from easydel.inference.esurge.distributed import RemoteEngineHandle, RoutedEngine

# Control-plane endpoints of already-running replica engines. Each replica
# was started with a ZMQ request plane, e.g.:
#
#   python -m easydel.scripts.elarge --config replica.yaml
#     # where replica.yaml's serve_replica action binds one control_port
#
# or programmatically with distributed={"coordination": "zmq",
# "distributed_control_port": <port>, "distributed_auth_token": AUTH}.
REPLICAS = [
    ("10.0.0.11", 19760),
    ("10.0.0.12", 19760),
    ("10.0.0.13", 19760),
    ("10.0.0.14", 19760),
]
AUTH = "change-me-same-on-all-replicas"
TOKENIZER = "Qwen/Qwen3.5-4B"


def main():
    # A client-side handle per replica: it speaks the request plane like an
    # in-pod rank would, renders tokens locally, and exposes the engine
    # surface (generate/stream/chat/limits). Construction blocks until the
    # replica's plane answers the handshake.
    handles = [
        RemoteEngineHandle(host, port, auth_token=AUTH, tokenizer_source=TOKENIZER)
        for host, port in REPLICAS
    ]

    # Front the replicas with prefix-affinity + join-shortest-queue routing.
    # RoutedEngine is duck-typed as an engine, so the API server accepts it
    # directly. Dead replicas are excluded from routing automatically.
    router = RoutedEngine(handles)

    ed_server = eSurgeApiServer({router.esurge_name: router})
    try:
        ed_server.fire()
    finally:
        router.close()


if __name__ == "__main__":
    main()
