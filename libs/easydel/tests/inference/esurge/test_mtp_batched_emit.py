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

"""Batched greedy verify EMISSION parity (host-loop batching).

The eSurge runner used to project + argmax + int()-sync the verify rows of every
spec request ONE AT A TIME in the per-request emission loop (``N`` LM-head matmuls
and ``N`` blocking device->host argmax transfers per decode step). The
batched-emission pre-pass in ``eSurgeRunner._execute_model_impl`` projects and
argmaxes the verify rows of the WHOLE running batch in ONE matmul + ONE transfer,
then the per-request loop consumes host integers.

WHY THIS TEST IS STRUCTURED THE WAY IT IS: a naive full-runner token comparison of
"batched emission" vs "per-request emission" is NOT a valid parity test, for two
independent reasons that both had to be removed:

1. Batching the LM-head matmul changes the matmul M-dimension, so on REAL logits
   the greedy argmax can flip on a near-tie (the ~1 ULP sensitivity every
   batched-decode engine has; see ``test_mtp_batched_draft.py``).
2. The CPU model forward is itself NOT bit-reproducible run-to-run (multi-threaded
   matmul reductions), so even the per-request path gives different tokens across
   two separate generations -- verified directly during development
   (``perreq_run_a != perreq_run_b``). Any token comparison across two forwards is
   therefore meaningless.

So this test isolates the EMISSION logic from BOTH: within a SINGLE process it makes
every stochastic/noisy input deterministic and batch-invariant -- the model forward
output (``_hidden_states`` + sampled tokens) is replaced by a fixed function of
(flat token index, step); ``project_hidden_rows`` is replaced by a batch-invariant
one-hot keyed off which hidden feature is largest; and the drafter is replaced by a
deterministic stub -- then runs the SAME generation twice, once with batched
emission and once with per-request emission, and asserts identical emitted tokens.
Because every input is identical and deterministic, the ONLY thing that varies is
the emission batching (flat gather / offset+count slicing / greedy acceptance /
seed-hidden selection / sequence-buffer writes / spec-token scatter); identical
tokens prove it is exactly equivalent to the per-request loop. (The real LM-head
near-tie of reason 1 is the documented, accepted difference in production.)
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.esurge.runners.spec import strategy as _strategy
from easydel.inference.esurge.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import Qwen3_5MTPDrafter

try:
    from ._common import make_tiny_qwen35 as make_tiny_model
except ImportError:  # standalone `python test_x.py`
    from _common import make_tiny_qwen35 as make_tiny_model

_VOCAB = 256
_HIDDEN = 128




def _det_hidden(shape, dtype, step):
    """Deterministic forward hidden: a fixed function of (flat token index, feature,
    step) -- no dependence on the noisy real forward, and per-row distinct so the
    argmax-of-features projection yields a non-trivial per-row token."""
    t = jnp.arange(shape[0], dtype=jnp.float32)[:, None]
    d = jnp.arange(shape[1], dtype=jnp.float32)[None, :]
    return jnp.sin(t * 0.7 + d * 0.13 + float(step) * 1.1).astype(dtype)


def _fake_project(_self, hidden_rows):
    """Batch-invariant one-hot projection keyed off the row's dominant feature.
    ``argmax`` over the feature axis is a per-row reduction, so a row projected alone
    is bit-identical to the same row inside a batched gather -- no near-tie."""
    hr = jnp.asarray(hidden_rows)
    feat = jnp.argmax(hr.astype(jnp.float32), axis=-1).astype(jnp.int32)
    tok = jnp.mod(feat * 7 + 3, _VOCAB).astype(jnp.int32)
    return jax.nn.one_hot(tok, _VOCAB, dtype=jnp.float32) * 30.0


def _fake_draft_next(self, *, req_id, seed_token, seed_position, seed_hidden, req_state, row_pos=None, **kw):
    """Deterministic drafts from host ints (no drafter forward -> no ULP noise)."""
    k = int(self.num_speculative_tokens)
    base = int(seed_token) * 3 + int(seed_position) + 1
    return [int((base + i * 5) % _VOCAB) for i in range(k)]


def _fake_draft_next_batched(self, seeds):
    return {
        str(rid): _fake_draft_next(
            self, req_id=rid, seed_token=tok, seed_position=pos, seed_hidden=h, req_state=None
        )
        for (rid, _row, tok, pos, h) in seeds
    }


def _fake_begin_batched_draft(self, seeds, *, seed_hidden_matrix=None):
    """Early-dispatch fake: compute the same deterministic drafts eagerly.

    The runner's early batched-draft pre-pass passes seed hidden as INT rows
    into ``seed_hidden_matrix``; the deterministic fake only consumes the host
    ints (seed token/position), so both call shapes produce identical drafts.
    """
    del seed_hidden_matrix
    return {"fake": _fake_draft_next_batched(self, seeds)}


def _fake_finish_batched_draft(self, handle):
    return handle["fake"]


def _run(model, *, n, k, disable_emit, prompts, project_rows):
    """Fully-deterministic generation (faked forward/project/drafter); returns
    {rid: emitted tokens}. Only the emission batching differs by ``disable_emit``."""
    saved = {
        "proj": _strategy.DrafterSpeculation.project_hidden_rows,
        "dn": _strategy.DrafterSpeculation.draft_next,
        "dnb": _strategy.DrafterSpeculation.draft_next_batched,
        "bbd": _strategy.DrafterSpeculation.begin_batched_draft,
        "fbd": _strategy.DrafterSpeculation.finish_batched_draft,
        "emit": os.environ.get("EASYDEL_DISABLE_BATCHED_EMIT"),
        "persist": os.environ.get("EASYDEL_MTP_PERSIST_KV"),
    }

    def _spy_project(self, hr):
        project_rows.append(int(jnp.asarray(hr).shape[0]))
        return _fake_project(self, hr)

    _strategy.DrafterSpeculation.project_hidden_rows = _spy_project
    _strategy.DrafterSpeculation.draft_next = _fake_draft_next
    _strategy.DrafterSpeculation.draft_next_batched = _fake_draft_next_batched
    _strategy.DrafterSpeculation.begin_batched_draft = _fake_begin_batched_draft
    _strategy.DrafterSpeculation.finish_batched_draft = _fake_finish_batched_draft
    os.environ["EASYDEL_DISABLE_BATCHED_EMIT"] = "1" if disable_emit else "0"
    os.environ["EASYDEL_MTP_PERSIST_KV"] = "1"
    try:
        runner = eSurgeRunner(
            model=model, hbm_utilization=0.05, page_size=16, max_cache_tokens=4096,
            max_model_len=64, max_num_batched_tokens=64, min_input_pad=1, min_token_pad=16,
            max_num_seqs=n, max_num_seq_buckets=[n], async_scheduling=False, use_aot_forward=False,
            verbose=False, enable_overlap_execution=False, enable_sampler_metrics=False,
            drafter=Qwen3_5MTPDrafter(model, num_draft_tokens=k, persist_kv=True),
        )
        runner.compile(max_num_batched_tokens=64)

        # Replace the (non-reproducible) real forward outputs with deterministic ones.
        real_execute = runner.executor_manager.execute
        step_counter = types.SimpleNamespace(v=0)

        def _det_execute(**kw):
            out_tokens, valid, iib, pib, hidden, logits, metrics = real_execute(**kw)
            s = step_counter.v
            step_counter.v += 1
            if hidden is not None:
                # Replace values deterministically but keep the original device
                # sharding so downstream jit'd consumers see the expected layout.
                hidden = jax.device_put(_det_hidden(hidden.shape, hidden.dtype, s), hidden.sharding)
            if out_tokens is not None:
                r = out_tokens.shape[0]
                det_t = jnp.mod(jnp.arange(r, dtype=jnp.int32) * 7 + s * 13 + 5, _VOCAB).astype(out_tokens.dtype)
                out_tokens = jax.device_put(det_t, out_tokens.sharding)
            return out_tokens, valid, iib, pib, hidden, logits, metrics

        runner.executor_manager.execute = _det_execute

        sch = Scheduler.from_runner(
            runner, max_num_batched_tokens=64, enable_prefix_caching=False,
            async_scheduling=False, num_speculative_tokens=runner.num_speculative_tokens,
        )
        reqs = []
        for i in range(n):
            r = EngineRequest(
                request_id=f"r-{i}", prompt_token_ids=list(prompts[i]),
                sampling_params=SamplingParams(max_tokens=14, temperature=0.0, ignore_eos=True),
                eos_token_id=None,
            )
            sch.add_request(r)
            reqs.append(r)
        for _ in range(4000):
            so = sch.schedule()
            out = runner.execute_model(so)
            sch.update_from_output(so, out)
            if all(r.is_finished() for r in reqs):
                break
        else:
            raise AssertionError("generation did not finish")
        toks = {r.request_id: list(r.output_token_ids) for r in reqs}
        runner.shutdown()
        runner.executor_manager.kv_pages = None
        return toks
    finally:
        _strategy.DrafterSpeculation.project_hidden_rows = saved["proj"]
        _strategy.DrafterSpeculation.draft_next = saved["dn"]
        _strategy.DrafterSpeculation.draft_next_batched = saved["dnb"]
        _strategy.DrafterSpeculation.begin_batched_draft = saved["bbd"]
        _strategy.DrafterSpeculation.finish_batched_draft = saved["fbd"]
        for key, val in (("EASYDEL_DISABLE_BATCHED_EMIT", saved["emit"]), ("EASYDEL_MTP_PERSIST_KV", saved["persist"])):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def test_batched_emit_equals_per_request_b3_k3():
    """B=3, k=3: batched emission == per-request emission, token-exact, on identical
    deterministic inputs."""
    model = make_tiny_model()
    n, k = 3, 3
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6], [7, 7, 1, 2, 3, 8, 8, 4, 10, 5], [10, 20, 30, 40, 11, 21, 31]]
    bcalls, pcalls = [], []
    bat = _run(model, n=n, k=k, disable_emit=False, prompts=prompts, project_rows=bcalls)
    per = _run(model, n=n, k=k, disable_emit=True, prompts=prompts, project_rows=pcalls)
    assert set(bat) == {"r-0", "r-1", "r-2"}
    for rid in bat:
        assert len(bat[rid]) == 14, f"{rid} incomplete: {bat[rid]}"
        assert bat[rid] == per[rid], f"{rid}: batched {bat[rid]} != per-request {per[rid]}"
    # Batched pre-pass engaged (a projection covered >= 2 requests' verify rows);
    # per-request only ever projects k+1 rows.
    assert any(c > (k + 1) for c in bcalls), f"batched emission never engaged: {sorted(set(bcalls))}"
    assert all(c <= (k + 1) for c in pcalls), f"per-request unexpectedly batched: {sorted(set(pcalls))}"


def test_batched_emit_equals_per_request_b3_k2():
    """Parity for a different k (B=3, k=2): different pad bucket (3*3) + accept pattern."""
    model = make_tiny_model()
    n, k = 3, 2
    prompts = [[3, 1, 4, 1, 5, 9, 2, 6], [7, 7, 1, 2, 3, 8, 8, 4, 10, 5], [10, 20, 30, 40, 11, 21, 31]]
    bcalls, pcalls = [], []
    bat = _run(model, n=n, k=k, disable_emit=False, prompts=prompts, project_rows=bcalls)
    per = _run(model, n=n, k=k, disable_emit=True, prompts=prompts, project_rows=pcalls)
    assert set(bat) == {"r-0", "r-1", "r-2"}
    for rid in bat:
        assert bat[rid] == per[rid], f"{rid}: batched {bat[rid]} != per-request {per[rid]}"
    assert any(c > (k + 1) for c in bcalls), f"batched emission never engaged: {sorted(set(bcalls))}"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([os.path.abspath(__file__), "-v", "-s"]))
