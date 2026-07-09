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

"""Two-process CPU integration test for the ZMQ step-coordination plane.

This is the direct regression test for the multi-host serving deadlock:
requests are admitted on rank 0 only (single ingress, like the HTTP
server), while rank 1 never sees them — under a 2-process
``jax.distributed`` runtime the jitted step is a global collective, so
without the coordinator rank 0 would block forever. With
``coordination="zmq"`` rank 0 mirrors its runner-call stream to rank 1 and
generation completes, byte-equal to a single-process reference.

Heavy (spawns JAX subprocesses); gated behind ``EASYDEL_MULTIHOST_TEST=1``
and marked slow.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("EASYDEL_MULTIHOST_TEST") != "1",
        reason="set EASYDEL_MULTIHOST_TEST=1 to run the multi-process coordination test",
    ),
]

_WORKER_SCRIPT = r"""
import json
import os
import sys

process_id = int(sys.argv[1])
num_processes = int(sys.argv[2])
coordinator = sys.argv[3]
control_port = int(sys.argv[4])
out_path = sys.argv[5]

import jax

if num_processes > 1:
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=num_processes,
        process_id=process_id,
    )

import easydel as ed
import spectrax as spx
from jax import numpy as jnp
from transformers import AutoTokenizer

from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.sampling_params import SamplingParams

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = ed.LlamaConfig(
    vocab_size=len(tokenizer),
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    max_position_embeddings=128,
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
        rngs=spx.Rngs(0),
    )

distributed = {}
if num_processes > 1:
    distributed = {
        "coordination": "zmq",
        "distributed_auth_token": "integration-test-token",
        "distributed_leader_addr": "127.0.0.1",
        "distributed_control_port": control_port,
        "distributed_ready_timeout_s": 600.0,
        "distributed_step_timeout_s": 120.0,
        "distributed_heartbeat_timeout_s": 120.0,
        "distributed_verify_digest_interval": 1,
    }

engine = eSurge(
    model=model,
    processor=tokenizer,
    runtime={
        "max_model_len": 128,
        "max_num_seqs": 2,
        "async_scheduling": False,
        "overlap_execution": False,
    },
    cache={
        "page_size": 16,
        "hbm_utilization": 0.05,
        "max_cache_tokens": 2048,
        "enable_prefix_caching": False,
    },
    parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    distributed=distributed,
)

if process_id == 0:
    outputs = engine.generate(
        ["the quick brown fox jumps over the lazy dog"],
        sampling_params=SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
        use_tqdm=False,
    )
    token_ids = [int(t) for t in outputs[0].outputs[0].token_ids]
    with open(out_path, "w") as f:
        json.dump({"token_ids": token_ids}, f)
    engine.terminate()
else:
    # Worker rank: the replay thread serves the leader until shutdown.
    engine._worker_replay_thread.join(timeout=600.0)
    engine.terminate()
print(f"rank {process_id} done", flush=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_world(num_processes: int, tmp: Path, label: str) -> list[int]:
    script = tmp / f"{label}_runner.py"
    script.write_text(_WORKER_SCRIPT)
    out_path = tmp / f"{label}_tokens.json"
    coordinator = f"127.0.0.1:{_free_port()}"
    control_port = _free_port()

    env = dict(os.environ)
    env.update(
        {
            "ENABLE_DISTRIBUTED_INIT": "0",
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        }
    )
    procs = []
    log_paths = []
    log_files = []
    for pid in range(num_processes):
        # stdout goes to a file, NOT a pipe: the engine spawns
        # tokenizer/detokenizer worker subprocesses that inherit stdio, and
        # an inherited pipe would keep communicate() blocked long after the
        # rank process itself exits.
        log_path = tmp / f"{label}_rank{pid}.log"
        log_file = open(log_path, "w")
        log_paths.append(log_path)
        log_files.append(log_file)
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    str(pid),
                    str(num_processes),
                    coordinator,
                    str(control_port),
                    str(out_path),
                ],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        )
    try:
        for pid, proc in enumerate(procs):
            try:
                proc.wait(timeout=900)
            except subprocess.TimeoutExpired as exc:
                for p in procs:
                    p.kill()
                log = log_paths[pid].read_text()[-4000:]
                raise AssertionError(f"{label}: rank {pid} timed out — the coordination plane hung\n{log}") from exc
            assert proc.returncode == 0, (
                f"{label}: rank {pid} exited {proc.returncode}\n{log_paths[pid].read_text()[-4000:]}"
            )
    finally:
        for log_file in log_files:
            log_file.close()
    assert out_path.exists(), f"{label}: leader produced no output\n{log_paths[0].read_text()[-4000:]}"
    return json.loads(out_path.read_text())["token_ids"]


def test_two_process_generation_completes_and_is_deterministic():
    """The deadlock regression test plus cross-run determinism.

    Completion alone is the deadlock proof: requests enter at rank 0 only,
    and without step replication rank 0 would block forever in the first
    global collective. Leader/worker agreement is enforced *inside* each run
    by the coordinator's sampled-token digest check at interval=1 — any
    divergence aborts the run with a nonzero exit.

    Token equality is asserted between two runs of the same 2-process
    topology. A single-process reference is intentionally NOT compared
    against: a different mesh layout (TP=1 vs TP=2) changes float reduction
    orders, which legally flips near-tied argmaxes on a randomly initialized
    tiny model.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        first = _run_world(2, tmp, "dual_a")
        second = _run_world(2, tmp, "dual_b")
        assert len(first) == 8
        assert first == second, (
            f"2-process coordinated generation is not reproducible:\n  first:  {first}\n  second: {second}"
        )
