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

"""Two-process CPU integration test for request-plane coalescing.

The replicated-trainer contract: every rank calls ``generate()`` with
identical prompts and auto-generated ids, and the plane must schedule ONE
physical request per logical admission slot — whichever rank admits first
creates the group, the other attaches as a subscriber and receives the
same token deltas (live or as a catch-up replay). Both ranks must observe
byte-identical outputs, a straggler attaching after the group finished
must be replayed from the retained log, and one rank's abort must detach
only that subscriber while the other rank streams to completion.

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
        reason="set EASYDEL_MULTIHOST_TEST=1 to run the multi-process coalescing test",
    ),
]

_WORKER_SCRIPT = r"""
import json
import os
import sys
import time

process_id = int(sys.argv[1])
num_processes = int(sys.argv[2])
coordinator = sys.argv[3]
control_port = int(sys.argv[4])
work_dir = sys.argv[5]

import jax

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
        rngs=spx.Rngs(0),
    )

engine = eSurge(
    model=model,
    processor=tokenizer,
    runtime={
        "max_model_len": 256,
        "max_num_seqs": 4,
        "async_scheduling": False,
        "overlap_execution": False,
    },
    cache={
        "page_size": 16,
        "hbm_utilization": 0.05,
        "max_cache_tokens": 4096,
        "enable_prefix_caching": False,
    },
    parsing={"silent_mode": True, "tool_parser": "", "reasoning_parser": ""},
    distributed={
        "coordination": "zmq",
        "distributed_auth_token": "integration-test-token",
        "distributed_leader_addr": "127.0.0.1",
        "distributed_control_port": control_port,
        "distributed_ready_timeout_s": 600.0,
        "distributed_step_timeout_s": 120.0,
        "distributed_heartbeat_timeout_s": 120.0,
        "distributed_admit_timeout_s": 300.0,
        "distributed_verify_digest_interval": 1,
    },
)

PROMPT = "the quick brown fox jumps over the lazy dog"
GREEDY = dict(temperature=0.0, ignore_eos=True)
results = {}


def barrier(name, timeout=600.0):
    mine = os.path.join(work_dir, f"barrier_{name}_rank{process_id}")
    other = os.path.join(work_dir, f"barrier_{name}_rank{1 - process_id}")
    with open(mine, "w") as f:
        f.write("x")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(other):
            return
        time.sleep(0.1)
    raise TimeoutError(f"barrier {name}: peer never arrived")


# Slot 1 — concurrent identical generate on BOTH ranks: one scheduled
# group, byte-identical outputs.
barrier("slot1")
outputs = engine.generate(
    [PROMPT],
    sampling_params=SamplingParams(max_tokens=8, **GREEDY),
    use_tqdm=False,
)
results["slot1_tokens"] = [int(t) for t in outputs[0].outputs[0].token_ids]
results["slot1_text"] = outputs[0].outputs[0].text

# Slot 2 — late attach: rank 0 generates and finishes, rank 1 admits the
# identical content only afterwards and must be served from the replay log.
if process_id == 0:
    outputs = engine.generate(
        [PROMPT + " again"],
        sampling_params=SamplingParams(max_tokens=8, **GREEDY),
        use_tqdm=False,
    )
    results["slot2_tokens"] = [int(t) for t in outputs[0].outputs[0].token_ids]
    with open(os.path.join(work_dir, "slot2_done"), "w") as f:
        f.write("x")
else:
    deadline = time.time() + 600
    while not os.path.exists(os.path.join(work_dir, "slot2_done")):
        if time.time() > deadline:
            raise TimeoutError("slot2_done never appeared")
        time.sleep(0.1)
    outputs = engine.generate(
        [PROMPT + " again"],
        sampling_params=SamplingParams(max_tokens=8, **GREEDY),
        use_tqdm=False,
    )
    results["slot2_tokens"] = [int(t) for t in outputs[0].outputs[0].token_ids]

# Slot 3 — one-rank abort: both ranks stream the same content; rank 1
# aborts after the first token, rank 0 must still run to completion.
barrier("slot3")
stream_tokens = 0
finish_reason = None
request_id = None
for update in engine.stream(
    PROMPT + " thrice",
    sampling_params=SamplingParams(max_tokens=32, **GREEDY),
):
    request_id = update.request_id
    stream_tokens = update.num_generated_tokens
    if process_id == 1 and stream_tokens >= 1 and not update.finished:
        engine.abort_request(request_id)
    if update.finished:
        finish_reason = update.outputs[0].finish_reason
        break
results["slot3_tokens"] = stream_tokens
results["slot3_finish"] = finish_reason

if process_id == 0:
    plane_stats = dict(engine._request_plane.stats)
    results["stats"] = plane_stats

with open(os.path.join(work_dir, f"results_rank{process_id}.json"), "w") as f:
    json.dump(results, f)

# Both ranks stay alive until each side has written its results, so the
# owner keeps serving any in-flight remote work.
barrier("teardown")
engine.terminate()
print(f"rank {process_id} done", flush=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_world(tmp: Path) -> tuple[dict, dict]:
    script = tmp / "coalesce_runner.py"
    script.write_text(_WORKER_SCRIPT)
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
    for pid in range(2):
        # stdout goes to a file, NOT a pipe: the engine spawns worker
        # subprocesses that inherit stdio and would hold a pipe open.
        log_path = tmp / f"coalesce_rank{pid}.log"
        log_file = open(log_path, "w")
        log_paths.append(log_path)
        log_files.append(log_file)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(script), str(pid), "2", coordinator, str(control_port), str(tmp)],
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
                raise AssertionError(f"rank {pid} timed out — coalescing hung\n{log}") from exc
            assert proc.returncode == 0, f"rank {pid} exited {proc.returncode}\n{log_paths[pid].read_text()[-4000:]}"
    finally:
        for log_file in log_files:
            log_file.close()

    rank0_path = tmp / "results_rank0.json"
    rank1_path = tmp / "results_rank1.json"
    assert rank0_path.exists(), f"rank 0 produced no results\n{log_paths[0].read_text()[-4000:]}"
    assert rank1_path.exists(), f"rank 1 produced no results\n{log_paths[1].read_text()[-4000:]}"
    return json.loads(rank0_path.read_text()), json.loads(rank1_path.read_text())


def test_lockstep_ranks_coalesce_to_one_scheduled_request_each():
    with tempfile.TemporaryDirectory() as tmpdir:
        rank0, rank1 = _run_world(Path(tmpdir))

        # Slot 1: concurrent identical admissions — byte-identical outputs.
        assert len(rank0["slot1_tokens"]) == 8
        assert rank0["slot1_tokens"] == rank1["slot1_tokens"], (
            f"coalesced ranks diverged:\n  rank0: {rank0['slot1_tokens']}\n  rank1: {rank1['slot1_tokens']}"
        )
        assert rank0["slot1_text"] == rank1["slot1_text"]

        # Slot 2: the late rank is served from the retained replay log.
        assert rank0["slot2_tokens"] == rank1["slot2_tokens"]

        # Slot 3: rank 1 aborted its subscription; rank 0 ran to completion.
        assert rank1["slot3_finish"] == "abort"
        assert rank0["slot3_finish"] == "length"
        assert rank0["slot3_tokens"] == 32

        # The owner scheduled each logical slot exactly once; the twin
        # admission attached instead of double-scheduling.
        stats = rank0["stats"]
        assert stats["scheduled_groups"] == 3, stats
        assert stats["coalesced_attaches"] == 3, stats
