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

"""DP-replica gate: RemoteEngineHandle + RoutedEngine over live engines.

Two independent tiny eSurge replicas run in this process, each exposing
the request plane through a workerless (``world_size == 1``) ZMQ leader.
Remote handles attach with the client role and must behave like engines:
token-identical generation versus a direct engine call, coherent
streaming, abort rendering. The router in front must pin repeat prefixes
to the member that served them, join-shortest-queue on the heartbeat
stats otherwise, and fail over when a replica dies.

Ordering matters: the replica-death test kills engine B, so it runs last.
"""

from __future__ import annotations

import socket
import time

import pytest

TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
AUTH = "routed-replica-test-token"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_engine(seed: int, control_port: int):
    import easydel as ed
    import jax
    import spectrax as spx
    from easydel.inference.esurge.esurge_engine import eSurge
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = ed.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sharding_axis_dims=(1, 1, 1, 1, jax.device_count(), 1),
    )
    with config.mesh:
        model = ed.LlamaForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(seed),
        )
    return eSurge(
        model=model,
        processor=tokenizer,
        runtime={"max_model_len": 256, "max_num_seqs": 4, "async_scheduling": False, "overlap_execution": False},
        cache={"page_size": 16, "hbm_utilization": 0.05, "max_cache_tokens": 4096, "enable_prefix_caching": False},
        parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
        distributed={
            "coordination": "zmq",
            "distributed_auth_token": AUTH,
            "distributed_control_bind_host": "127.0.0.1",
            "distributed_control_port": control_port,
            "distributed_heartbeat_interval_s": 0.2,
        },
    )


class _Replicas:
    def __init__(self):
        from easydel.inference.esurge.distributed.remote_engine import RemoteEngineHandle
        from easydel.inference.esurge.distributed.routed_engine import RoutedEngine

        port_a, port_b = _free_port(), _free_port()
        # Different seeds: the replicas are distinguishable by their outputs.
        self.engine_a = _build_engine(0, port_a)
        self.engine_b = _build_engine(1, port_b)
        self.handle_a = RemoteEngineHandle(
            "127.0.0.1",
            port_a,
            auth_token=AUTH,
            tokenizer_source=TOKENIZER,
            heartbeat_timeout_s=10.0,
        )
        self.handle_b = RemoteEngineHandle(
            "127.0.0.1",
            port_b,
            auth_token=AUTH,
            tokenizer_source=TOKENIZER,
            heartbeat_timeout_s=10.0,
        )
        self.router = RoutedEngine([self.handle_a, self.handle_b])

    def close(self):
        for handle in (self.handle_a, self.handle_b):
            try:
                handle.close()
            except Exception:
                pass
        for engine in (self.engine_a, self.engine_b):
            try:
                engine.terminate()
            except Exception:
                pass


@pytest.fixture(scope="module")
def replicas():
    world = _Replicas()
    try:
        yield world
    finally:
        world.close()


def _greedy(max_tokens=8):
    from easydel.inference.sampling_params import SamplingParams

    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)


PROMPT = "the quick brown fox jumps over the lazy dog"


def test_handle_generation_matches_direct_engine_call(replicas):
    direct = replicas.engine_a.generate([PROMPT], sampling_params=_greedy(), use_tqdm=False)
    remote = replicas.handle_a.generate([PROMPT], sampling_params=_greedy())
    direct_tokens = [int(t) for t in direct[0].outputs[0].token_ids]
    remote_tokens = [int(t) for t in remote[0].outputs[0].token_ids]
    assert len(direct_tokens) == 8
    assert remote_tokens == direct_tokens, "client-role ingress must be token-identical to local ingress"
    assert remote[0].outputs[0].finish_reason == "length"
    assert remote[0].accumulated_text == direct[0].accumulated_text


def test_handle_streaming_renders_coherent_deltas(replicas):
    deltas = []
    final_text = None
    finished = False
    for update in replicas.handle_a.stream(PROMPT + " streamed", sampling_params=_greedy()):
        if update.delta_text:
            deltas.append(update.delta_text)
        final_text = update.accumulated_text
        finished = update.finished
        if update.finished:
            break
    assert finished
    assert final_text
    assert "".join(deltas) == final_text


def test_handle_abort_renders_terminal_state(replicas):
    request_id = "routed-abort-req"
    saw_tokens = False
    finish_reason = None
    for update in replicas.handle_a.stream(
        PROMPT + " to be aborted", sampling_params=_greedy(max_tokens=64), request_id=request_id
    ):
        if not saw_tokens and update.num_generated_tokens >= 1 and not update.finished:
            saw_tokens = True
            replicas.handle_a.abort_request(request_id)
        if update.finished:
            finish_reason = update.outputs[0].finish_reason
            break
    assert saw_tokens
    assert finish_reason == "abort"


def test_router_pins_repeat_prefixes_to_the_same_member(replicas):
    prompt = PROMPT + " affinity check with a distinctive tail"
    first = replicas.router.generate([prompt], sampling_params=_greedy())
    pinned, _depth = replicas.router.affinity.lookup(prompt)
    assert pinned is not None
    second = replicas.router.generate([prompt], sampling_params=_greedy())
    still_pinned, _depth = replicas.router.affinity.lookup(prompt)
    assert still_pinned == pinned, "a repeat prefix must stay on the member that served it"
    first_tokens = [int(t) for t in first[0].outputs[0].token_ids]
    second_tokens = [int(t) for t in second[0].outputs[0].token_ids]
    assert first_tokens == second_tokens, "same member + greedy sampling => identical tokens"


def test_router_joins_shortest_queue_on_stats(replicas):
    from easydel.inference.esurge.distributed.wire import QueueStats

    original = replicas.handle_b._channel.queue_stats
    try:
        replicas.handle_b._channel.queue_stats = QueueStats(
            num_waiting=32, num_running=4, pending_prefill_tokens=100_000
        )
        chosen = replicas.router.select_member("a brand new prompt with no affinity history")
        assert chosen == 0, "JSQ must avoid the loaded replica"
    finally:
        replicas.handle_b._channel.queue_stats = original


def test_zz_dead_replica_fails_over(replicas):
    # Kill replica B; its leader broadcasts Shutdown to attached clients.
    replicas.engine_b.terminate()
    deadline = time.monotonic() + 15.0
    while replicas.handle_b.alive and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not replicas.handle_b.alive, "handle must observe the owner shutdown"

    from easydel.inference.esurge.distributed.request_plane import RequestPlaneError

    with pytest.raises(RequestPlaneError):
        replicas.handle_b.generate([PROMPT], sampling_params=_greedy())

    # The router must route around the dead member for any prompt.
    for probe in ("x", PROMPT, "another prompt entirely"):
        assert replicas.router.select_member(probe) == 0
    outputs = replicas.router.generate([PROMPT + " after failover"], sampling_params=_greedy())
    assert outputs[0].finished
    assert outputs[0].outputs[0].finish_reason == "length"
