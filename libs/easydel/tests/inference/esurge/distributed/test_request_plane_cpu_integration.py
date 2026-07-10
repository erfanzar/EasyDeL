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

"""Two-process CPU integration test for the unified request plane.

The "ingress anywhere" contract, end to end: rank 1 — a plane *origin* that
runs no scheduler — calls ``generate``/``stream``/``abort_request`` exactly
like a single-host caller. Admissions forward to the rank-0 owner, raw
token deltas stream back, and rank 1's own output pipeline renders them, so
rank 1's greedy generation must be token-identical to rank 0's direct call
on the same engine. Streaming deltas, ``n>1`` fan-out, stop-string
trimming, and mid-flight aborts all exercise origin-side rendering paths.

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
        reason="set EASYDEL_MULTIHOST_TEST=1 to run the multi-process request-plane test",
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


def wait_for(path, timeout=600.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {path}")


owner_tokens_path = os.path.join(work_dir, "owner_tokens.json")
origin_results_path = os.path.join(work_dir, "origin_results.json")
origin_done_path = os.path.join(work_dir, "origin_done")

if process_id == 0:
    # Owner-local ingress: the reference tokens for the parity check.
    outputs = engine.generate(
        [PROMPT],
        sampling_params=SamplingParams(max_tokens=8, **GREEDY),
        use_tqdm=False,
    )
    with open(owner_tokens_path, "w") as f:
        json.dump(
            {
                "token_ids": [int(t) for t in outputs[0].outputs[0].token_ids],
                "text": outputs[0].outputs[0].text,
            },
            f,
        )
    # Keep serving until the origin rank finishes its scenarios.
    wait_for(origin_done_path)
    engine.terminate()
else:
    wait_for(owner_tokens_path)
    with open(owner_tokens_path) as f:
        owner_ref = json.load(f)
    results = {}

    # 1. Parity: remote ingress must be token-identical to the owner's
    #    direct call (same engine, same greedy step stream).
    outputs = engine.generate(
        [PROMPT],
        sampling_params=SamplingParams(max_tokens=8, **GREEDY),
        use_tqdm=False,
    )
    results["parity_tokens"] = [int(t) for t in outputs[0].outputs[0].token_ids]
    results["parity_text"] = outputs[0].outputs[0].text
    results["parity_finish_reason"] = outputs[0].outputs[0].finish_reason

    # 2. Streaming: origin-side rendering must produce a coherent delta
    #    stream that concatenates to the final text.
    deltas = []
    final_text = None
    finished = False
    for update in engine.stream(PROMPT, sampling_params=SamplingParams(max_tokens=8, **GREEDY)):
        if update.delta_text:
            deltas.append(update.delta_text)
        final_text = update.accumulated_text
        finished = update.finished
        if update.finished:
            break
    results["stream_concat"] = "".join(deltas)
    results["stream_final"] = final_text
    results["stream_finished"] = finished

    # 3. n>1 fan-out: two samples, both rendered on the origin. Greedy
    #    forbids n>1, so sample with temperature; only counts are asserted.
    outputs = engine.generate(
        [PROMPT],
        sampling_params=SamplingParams(max_tokens=6, n=2, temperature=0.7, ignore_eos=True),
        use_tqdm=False,
    )
    results["n2_token_counts"] = [len(comp.token_ids) for comp in outputs[0].outputs]
    results["n2_finish_reasons"] = [comp.finish_reason for comp in outputs[0].outputs]

    # 4. Stop strings: matched and trimmed on the origin, rows freed on the
    #    owner via the forwarded StopHit.
    stop_str = owner_ref["text"][2:5] if len(owner_ref["text"]) >= 5 else ""
    if stop_str:
        outputs = engine.generate(
            [PROMPT],
            sampling_params=SamplingParams(max_tokens=8, stop=[stop_str], **GREEDY),
            use_tqdm=False,
        )
        results["stop_str"] = stop_str
        results["stop_text"] = outputs[0].outputs[0].text
        results["stop_finish_reason"] = outputs[0].outputs[0].finish_reason
    else:
        results["stop_str"] = ""

    # 5. Abort mid-flight: local terminal state renders immediately; the
    #    owner frees the scheduled rows via the forwarded AbortReq.
    abort_id = "origin-abort-request"
    saw_tokens = False
    abort_finish = None
    for update in engine.stream(
        PROMPT,
        sampling_params=SamplingParams(max_tokens=200, **GREEDY),
        request_id=abort_id,
    ):
        if not saw_tokens and update.num_generated_tokens >= 1:
            saw_tokens = True
            engine.abort_request(abort_id)
        if update.finished:
            abort_finish = update.outputs[0].finish_reason
            break
    results["abort_saw_tokens"] = saw_tokens
    results["abort_finish_reason"] = abort_finish

    with open(origin_results_path, "w") as f:
        json.dump(results, f)
    with open(origin_done_path, "w") as f:
        f.write("done")
    engine.terminate()

print(f"rank {process_id} done", flush=True)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_world(tmp: Path) -> tuple[dict, dict]:
    script = tmp / "plane_runner.py"
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
        # stdout goes to a file, NOT a pipe: the engine spawns
        # tokenizer/detokenizer worker subprocesses that inherit stdio, and
        # an inherited pipe would keep communicate() blocked long after the
        # rank process itself exits.
        log_path = tmp / f"plane_rank{pid}.log"
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
                raise AssertionError(f"rank {pid} timed out — the request plane hung\n{log}") from exc
            assert proc.returncode == 0, f"rank {pid} exited {proc.returncode}\n{log_paths[pid].read_text()[-4000:]}"
    finally:
        for log_file in log_files:
            log_file.close()

    owner_path = tmp / "owner_tokens.json"
    origin_path = tmp / "origin_results.json"
    assert owner_path.exists(), f"owner produced no reference output\n{log_paths[0].read_text()[-4000:]}"
    assert origin_path.exists(), f"origin produced no results\n{log_paths[1].read_text()[-4000:]}"
    return json.loads(owner_path.read_text()), json.loads(origin_path.read_text())


def test_origin_rank_generates_through_the_request_plane():
    """Rank-1 ingress equals rank-0 ingress, plus the origin-side render paths.

    One 2-process world runs five scenarios back to back (startup is the
    expensive part). Parity is the core assertion: the owner schedules both
    calls on the same engine with greedy sampling, so the origin's tokens
    must match the owner's exactly — any divergence means origin-side
    rendering drifted from the local pipeline.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        owner_ref, results = _run_world(Path(tmpdir))

        # Parity: byte-identical tokens through the plane.
        assert len(owner_ref["token_ids"]) == 8
        assert results["parity_tokens"] == owner_ref["token_ids"], (
            f"remote ingress diverged from owner-local ingress:\n"
            f"  owner:  {owner_ref['token_ids']}\n"
            f"  origin: {results['parity_tokens']}"
        )
        assert results["parity_text"] == owner_ref["text"]
        assert results["parity_finish_reason"] == "length"

        # Streaming: deltas concatenate to the final text.
        assert results["stream_finished"] is True
        assert results["stream_final"] == owner_ref["text"]
        assert results["stream_concat"] == results["stream_final"]

        # n>1: both samples fully generated on the origin.
        assert results["n2_token_counts"] == [6, 6]
        assert all(reason == "length" for reason in results["n2_finish_reasons"])

        # Stop strings: trimmed before the match, finished as a stop.
        if results["stop_str"]:
            assert results["stop_str"] not in results["stop_text"]
            assert owner_ref["text"].startswith(results["stop_text"])
            assert results["stop_finish_reason"] == "stop"

        # Abort: terminal state rendered locally after tokens started.
        assert results["abort_saw_tokens"] is True
        assert results["abort_finish_reason"] == "abort"
