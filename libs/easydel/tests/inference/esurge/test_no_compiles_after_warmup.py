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

"""eSurge compiles every per-step program in ``runner.compile``: serving afterwards compiles nothing.

On a multi-host mesh a program that compiles on the first step reaching its shapes holds every other host in the next
collective while that one host compiles. This serves a tiny hybrid (KDA + DSA) model through the engine core with async
scheduling: requests that finish at different steps, more requests than rows (admission and row moves), prefix-cache
hits on a shared prompt prefix, greedy and sampled requests. No XLA program may compile after ``runner.compile``, and
the compile warm-up must leave the live RNG key and penalty state bit-identical. Computation test: the suite's conftest
refuses it on a CPU backend.
"""

import collections
import gc

import jax
import numpy as np
from easydel.inference.esurge.benchmarking import zero_schedule_needs_update
from easydel.inference.esurge.runners import eSurgeRunner
from easydel.inference.sampling_params import SamplingParams
from jax import monitoring

try:
    from ._common import make_core_for_runner, served_request
    from .test_recurrent_row_windows import _make_tiny_glm5_next
except ImportError:  # standalone `python test_x.py`
    from _common import make_core_for_runner, served_request
    from test_recurrent_row_windows import _make_tiny_glm5_next

_COMPILE_EVENTS = frozenset(
    {"/jax/core/compile/backend_compile_duration", "/jax/compilation_cache/cache_retrieval_time_sec"}
)
PREFIX = [(7 * i + 3) % 120 + 1 for i in range(40)]


def _runner(model) -> eSurgeRunner:
    return eSurgeRunner(
        model=model,
        hbm_utilization=0.02,
        page_size=16,
        max_cache_tokens=2048,
        max_model_len=128,
        max_num_batched_tokens=64,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=4,
        async_scheduling=True,
        use_aot_forward=True,
        verbose=False,
        enable_overlap_execution=False,
        enable_sampler_metrics=False,
    )


def _live_sampler_state(manager) -> tuple:
    """Host copies of the live RNG key and penalty state, plus whether their buffers are still alive."""
    key = manager.rng_key
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    runtime = manager.sampler_runtime
    counts = runtime._sampler_token_counts
    return (
        np.asarray(key),
        np.asarray(counts),
        bool(runtime._sampler_penalty_state_dirty),
        bool(runtime._sampler_penalty_state_ready),
        manager.rng_key.is_deleted() or counts.is_deleted(),
    )


def _serve(runner, scheduler, specs) -> dict[str, list[int]]:
    """Drive the runner as the engine's overlap loop does; returns the output tokens per request."""
    requests = []
    for spec in specs:
        request = served_request(
            request_id=spec["id"],
            prompt_token_ids=list(spec["prompt"]),
            sampling_params=SamplingParams(
                max_tokens=spec["max_tokens"], temperature=spec["temperature"], ignore_eos=True
            ),
            eos_token_id=None,
        )
        scheduler.add_request(request)
        requests.append(request)

    def commit(so, output):
        if hasattr(output, "get_output"):
            output = output.get_output()
        scheduler.commit(so, output)

    def run(so):
        if runner.can_dispatch_next_before_async_drain(so):
            return so, runner.execute_model_async(so)
        commit(so, runner.execute_model(so))
        return None

    def run_empty(so):
        if zero_schedule_needs_update(so):
            commit(so, runner.execute_model(so))

    pending = None
    for _ in range(2000):
        if pending is None and all(request.is_finished() for request in requests):
            break
        if pending is not None:
            prev, handle = pending
            pending = None
            if runner.can_dispatch_next_before_async_drain(prev):
                nxt = scheduler.schedule()
                if int(nxt.total_num_scheduled_tokens) > 0:
                    ahead = (
                        (nxt, runner.execute_model_async(nxt))
                        if runner.can_dispatch_next_before_async_drain(nxt)
                        else None
                    )
                    commit(prev, handle)
                    pending = ahead if ahead is not None else run(nxt)
                    continue
                commit(prev, handle)
                run_empty(nxt)
                continue
            commit(prev, handle)
        so = scheduler.schedule()
        if int(so.total_num_scheduled_tokens) == 0:
            run_empty(so)
            continue
        pending = run(so)
    else:
        raise AssertionError("generation did not finish")
    return {request.request_id: [int(t) for t in request.output_token_ids] for request in requests}


def test_serving_compiles_nothing_after_runner_compile():
    model = _make_tiny_glm5_next()
    runner = _runner(model)
    manager = runner.executor_manager
    warm_up = manager._warm_step_programs
    seen: dict = {}

    def recorded_warm_up(**kwargs):
        before = _live_sampler_state(manager)
        warm_up(**kwargs)
        seen["state"] = (before, _live_sampler_state(manager))
        seen["buckets"] = kwargs

    manager._warm_step_programs = recorded_warm_up
    used = collections.Counter()
    for name in ("permute_recurrent_slots", "clear_recurrent_slots", "save_recurrent_snapshots"):
        method = getattr(manager, name)

        def counted(*args, _method=method, _name=name, **kwargs):
            used[_name] += 1
            return _method(*args, **kwargs)

        setattr(manager, name, counted)
    build_handoff = runner._build_device_token_handoff

    def counted_handoff(**kwargs):
        handoff = build_handoff(**kwargs)
        used["handoff"] += handoff is not None
        return handoff

    runner._build_device_token_handoff = counted_handoff

    compiled: list[tuple[str, str]] = []

    def on_duration(event, duration_secs, **kwargs):
        del duration_secs
        if event in _COMPILE_EVENTS:
            compiled.append((event.rsplit("/", 1)[-1], str(kwargs.get("fun_name", ""))))

    try:
        runner.compile(max_num_batched_tokens=64)
        before, after = seen["state"]
        assert before[4] is False and after[4] is False, "the warm-up consumed a live buffer"
        for live_before, live_after in zip(before[:2], after[:2], strict=True):
            assert live_before.dtype == live_after.dtype and np.array_equal(live_before, live_after)
        assert before[2:4] == after[2:4]
        assert seen["buckets"]["logits_buckets"] == [1, 2, 4], seen["buckets"]

        scheduler = make_core_for_runner(runner, max_num_batched_tokens=64, async_scheduling=True)
        monitoring.register_event_duration_secs_listener(on_duration)
        try:
            # One request fills the prefix cache, then six (two more than the rows) share its prefix and finish at
            # different steps; greedy and sampled requests alternate.
            first = _serve(runner, scheduler, [{"id": "p0", "prompt": PREFIX, "max_tokens": 4, "temperature": 0.0}])
            assert len(first["p0"]) == 4
            lengths = [2, 9, 5, 14, 3, 11]
            specs = [
                {
                    "id": f"r{i}",
                    "prompt": [*PREFIX, *(((11 * i + 5 * j) % 120) + 1 for j in range(3 + i))],
                    "max_tokens": n,
                    "temperature": 0.0 if i % 2 == 0 else 0.7,
                }
                for i, n in enumerate(lengths)
            ]
            tokens = _serve(runner, scheduler, specs)
        finally:
            monitoring.unregister_event_duration_listener(on_duration)
        assert [len(tokens[f"r{i}"]) for i in range(len(lengths))] == lengths
        assert used["handoff"] > 0 and used["permute_recurrent_slots"] + used["clear_recurrent_slots"] > 0, used
        assert not compiled, f"compiled after runner.compile ({dict(used)}): {collections.Counter(compiled)}"
    finally:
        runner.shutdown()
        manager.kv_pages = None
        gc.collect()
