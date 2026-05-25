#!/usr/bin/env python3
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3.5-9B eSurge throughput benchmark, no speculative decoding.

This is the EasyDeL side of an apples-to-apples TPU comparison against
``vllm bench throughput``:

  - Qwen/Qwen3.5-9B
  - 4-way tensor parallelism
  - max_model_len=131072
  - max_num_seqs=32
  - hbm_utilization=0.80
  - no drafter / no MTP

The harness uses token-id prompts directly so prompt length is exact and no
tokenizer/text detour changes the batch shape. The script loads the model
from a local snapshot (defaulting to ``/dev/shm/easydel_ckpts/...``), spins up
an :class:`eSurgeRunner` with the configured page size / cache budget, then
drives prompts through the scheduler while optionally overlapping the next
schedule with the previous async forward. Per-step phase profiles
(``forward``, ``prep``, ``sample``, ``post``) are aggregated by total token
count and emitted into the JSON summary written to ``--json-out``.

Side effects:
    - Requires a TPU backend; refuses to run otherwise.
    - Reads from ``--model`` path.
    - Writes a JSON results document to ``--json-out``.
    - Optionally writes a JAX profiler trace to ``--xprof-dir`` for one
      selected timed trial.

Usage:
    python scripts/bench_qwen35_9b_131k_no_mtp.py --num-prompts 32 \\
        --prompt-len 1024 --output-len 256 --trials 1 \\
        --json-out /tmp/easydel_qwen35_9b_131k_32_1024x256.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "tpu")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "64")

DEFAULT_MODEL_PATH = (
    "/dev/shm/easydel_ckpts/hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
SHARDING_AXIS_DIMS = (1, 1, 1, 1, -1, 1)
SHARDING_AXIS_NAMES = ("pp", "dp", "fsdp", "ep", "tp", "sp")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the throughput benchmark.

    Returns:
        argparse.Namespace: Parsed flags. Notable groups: ``--model`` /
            ``--prompt-len`` / ``--output-len`` / ``--num-prompts`` shape the
            workload; ``--max-model-len`` / ``--max-num-seqs`` /
            ``--max-num-batched-tokens`` / ``--hbm-utilization`` / ``--page-size``
            configure the runner; ``--warmups`` / ``--trials`` / ``--seed`` /
            ``--json-out`` drive the measurement loop; ``--no-async`` /
            ``--no-overlap`` / ``--use-aot-forward`` / ``--verbose-runner``
            toggle runtime behaviour; the ``--xprof-*`` family controls
            optional JAX profiler capture.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--hbm-utilization", type=float, default=0.80)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="/tmp/easydel_qwen35_9b_131k_32_1024x256.json")
    parser.add_argument("--no-async", action="store_true")
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--use-aot-forward", action="store_true")
    parser.add_argument("--verbose-runner", action="store_true")
    parser.add_argument("--xprof-dir", default=None, help="Optional JAX profiler output directory for one timed trial.")
    parser.add_argument("--xprof-trial", type=int, default=1, help="1-based timed trial index to profile.")
    parser.add_argument("--xprof-host-level", type=int, default=1)
    parser.add_argument("--xprof-python-level", type=int, default=0)
    return parser.parse_args()


def configure_tpu_process() -> None:
    """Best-effort cleanup of leftover TPU process state before initializing JAX.

    Inspects ``/dev/vfio/0`` for current users via ``fuser`` (informational
    only) and removes any stale ``/tmp/libtpu_lockfile`` so a previous TPU
    runtime that died ungracefully does not block this process from acquiring
    the device.

    Returns:
        None.
    """
    vfio = Path("/dev/vfio/0")
    if vfio.exists():
        subprocess.run(["fuser", str(vfio)], check=False)
    lockfile = Path("/tmp/libtpu_lockfile")
    if lockfile.exists():
        lockfile.unlink()


def make_prompts(*, count: int, prompt_len: int, vocab_size: int, seed: int) -> list[list[int]]:
    """Generate ``count`` random token-ID prompts of exactly ``prompt_len`` tokens.

    Samples integer tokens uniformly from ``[1000, min(vocab_size-1, 100000))``
    to avoid special tokens at the very low end of the vocabulary while
    keeping prompts well inside the model's regular vocab range.

    Args:
        count: Number of prompts to produce.
        prompt_len: Token length of each prompt.
        vocab_size: Vocabulary size of the target model.
        seed: NumPy PRNG seed for reproducibility.

    Returns:
        list[list[int]]: ``count`` lists of ``prompt_len`` Python ints.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    lo = 1000
    hi = max(lo + 1, min(int(vocab_size) - 1, 100000))
    return [[int(tok) for tok in rng.integers(lo, hi, size=prompt_len, dtype=np.int32)] for _ in range(count)]


def all_finished(requests) -> bool:
    """Return ``True`` when every request in ``requests`` has finished.

    Args:
        requests: Iterable of :class:`EngineRequest` objects.

    Returns:
        bool: ``True`` if every request is finished.
    """
    return all(req.is_finished() for req in requests)


def zero_schedule_needs_update(scheduler_output) -> bool:
    """Return ``True`` if a zero-token schedule still requires a model update.

    A schedule with zero scheduled tokens can still carry side effects
    (finished requests to release, preempted requests to spill, new requests
    to admit, or cached requests that need a no-op pass). When any of those
    bookkeeping fields are non-empty the model still needs to be invoked.

    Args:
        scheduler_output: Output of :meth:`Scheduler.schedule`.

    Returns:
        bool: ``True`` when a follow-up ``execute_model`` is required.
    """
    return bool(
        scheduler_output.finished_req_ids
        or scheduler_output.preempted_req_ids
        or scheduler_output.scheduled_new_reqs
        or scheduler_output.scheduled_cached_reqs.num_reqs
    )


def reset_runner(runner) -> None:
    """Reset a runner's state between benchmark batches without zeroing KV pages.

    Calls ``runner.reset_state()`` and then walks the cache views, resetting
    linear-attention recurrent / convolutional states (small, must start
    clean) while leaving the full-attention ragged KV pages alone. Zeroing
    the ragged pages at 131k context would be enormously expensive and is
    safe to skip because the prefill phase overwrites the live slots and the
    page table never reads stale pages outside the active range.

    Args:
        runner: An :class:`eSurgeRunner` instance.

    Returns:
        None. Mutates ``runner.executor_manager.kv_pages`` in place when any
        view changed.
    """
    import jax

    runner.reset_state()
    cache = runner.executor_manager.kv_pages
    if cache is None or not hasattr(cache, "views"):
        return

    new_views = []
    changed = False
    for view in cache.views:
        if view is None:
            new_views.append(None)
            continue

        if hasattr(view, "kv_pages") and not hasattr(view, "recurrent_state"):
            new_views.append(view)
            continue

        if hasattr(view, "recurrent") and getattr(view, "recurrent", None) is not None:
            recurrent = view.recurrent.reset()
            new_views.append(
                view.replace(
                    recurrent=recurrent,
                    conv_state=getattr(recurrent, "conv_state", None),
                    recurrent_state=getattr(recurrent, "recurrent_state", None),
                    positions=getattr(recurrent, "positions", None),
                    seqlen_offset=getattr(recurrent, "seqlen_offset", None),
                )
            )
            changed = True
            continue

        if hasattr(view, "recurrent_state") or hasattr(view, "conv_state"):
            new_views.append(view.reset())
            changed = True
            continue

        new_views.append(view)

    if changed:
        runner.executor_manager.kv_pages = cache.replace(views=new_views)
        jax.block_until_ready(runner.executor_manager.kv_pages)


def run_batch(runner, prompts: list[list[int]], output_len: int, max_num_batched_tokens: int) -> dict:
    """Drive a single batch of prompts through the runner and collect throughput.

    Resets the runner, builds a fresh :class:`Scheduler`, adds one
    :class:`EngineRequest` per prompt with greedy decoding (``temperature=0``,
    ``ignore_eos=True``) capped at ``output_len`` tokens, then schedules and
    executes the model in a loop. When async scheduling is enabled the next
    schedule is dispatched before the previous async forward drains so the
    benchmark mirrors the live overlap path. After the loop, per-step
    profiling records on ``runner._perf_phase_history`` are aggregated by
    ``total_tokens`` bucket.

    Args:
        runner: Initialized :class:`eSurgeRunner`.
        prompts: Token-ID prompts produced by :func:`make_prompts`.
        output_len: Maximum decode length per request.
        max_num_batched_tokens: Scheduler token budget per step.

    Returns:
        dict: Summary with ``wallclock_s``, ``generated_tokens``,
            ``output_tok_s``, ``per_request_generated``, ``profile_steps``,
            and the per-token-bucket ``profile_by_total_tokens`` aggregation.

    Raises:
        RuntimeError: If the benchmark loop exits before every request has
            finished generating.
    """
    import jax
    import jax.numpy as jnp

    from easydel.inference.esurge.request import EngineRequest
    from easydel.inference.esurge.scheduler import Scheduler
    from easydel.inference.sampling_params import SamplingParams

    reset_runner(runner)
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=False,
        async_scheduling=bool(getattr(runner, "async_scheduling", False)),
        num_speculative_tokens=0,
    )

    sampling = SamplingParams(max_tokens=output_len, temperature=0.0, top_p=1.0, ignore_eos=True)
    requests = []
    for idx, prompt in enumerate(prompts):
        req = EngineRequest(
            request_id=f"bench-{idx}",
            prompt_token_ids=list(prompt),
            sampling_params=sampling.clone(),
            eos_token_id=None,
        )
        scheduler.add_request(req)
        requests.append(req)

    def drain(pending) -> None:
        """Wait on an in-flight async execution and apply its output to the scheduler.

        Args:
            pending: ``(future, scheduler_output)`` tuple returned by a prior
                async dispatch.

        Returns:
            None.
        """
        future, pending_scheduler_output = pending
        model_output = runner.wait_for_execution(future)
        scheduler.update_from_output(pending_scheduler_output, model_output)

    def execute_positive(scheduler_output):
        """Execute a non-empty schedule either async-dispatched or sync.

        When async scheduling is on and the runner permits dispatching the
        next step before the previous one drains, returns a pending tuple to
        be drained later. Otherwise, runs the model synchronously and
        immediately commits the output back to the scheduler.

        Args:
            scheduler_output: A schedule with ``total_num_scheduled_tokens > 0``.

        Returns:
            ``(future, scheduler_output)`` tuple if dispatched async, else
            ``None`` after a synchronous execute + update.
        """
        if (
            runner.async_scheduling
            and scheduler_output.total_num_scheduled_tokens > 0
            and runner.can_dispatch_next_before_async_drain(scheduler_output)
        ):
            return runner.execute_model_async(scheduler_output), scheduler_output
        output = runner.execute_model(scheduler_output)
        scheduler.update_from_output(scheduler_output, output)
        return None

    profile_start = len(getattr(runner, "_perf_phase_history", []))
    t0 = time.perf_counter()
    pending = None
    max_iters = len(prompts) * (len(prompts[0]) + output_len + 32)
    for _ in range(max_iters):
        if pending is not None:
            future, prev_scheduler_output = pending
            if runner.can_dispatch_next_before_async_drain(prev_scheduler_output):
                scheduler_output = scheduler.schedule()
                if scheduler_output.total_num_scheduled_tokens > 0:
                    next_pending = None
                    if runner.can_dispatch_next_before_async_drain(scheduler_output):
                        next_pending = runner.execute_model_async(scheduler_output), scheduler_output
                    drain((future, prev_scheduler_output))
                    if next_pending is None:
                        next_pending = execute_positive(scheduler_output)
                    pending = next_pending
                    if all_finished(requests) and pending is None:
                        break
                    continue
                drain((future, prev_scheduler_output))
                pending = None
                if zero_schedule_needs_update(scheduler_output):
                    output = runner.execute_model(scheduler_output)
                    scheduler.update_from_output(scheduler_output, output)
                if all_finished(requests):
                    break
                continue

            drain((future, prev_scheduler_output))
            pending = None
            if all_finished(requests):
                break

        scheduler_output = scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            if zero_schedule_needs_update(scheduler_output):
                output = runner.execute_model(scheduler_output)
                scheduler.update_from_output(scheduler_output, output)
            if all_finished(requests):
                break
            continue
        pending = execute_positive(scheduler_output)
        if all_finished(requests) and pending is None:
            break

    if pending is not None:
        drain(pending)
    if not all_finished(requests):
        unfinished = [req.request_id for req in requests if not req.is_finished()]
        raise RuntimeError(f"benchmark did not finish; unfinished={unfinished[:8]}")

    jax.block_until_ready(jnp.zeros(()))
    wallclock = time.perf_counter() - t0
    generated = sum(len(req.output_token_ids) for req in requests)
    profile_records = [dict(r) for r in list(getattr(runner, "_perf_phase_history", []))[profile_start:]]
    profile_by_tokens: dict[str, dict] = {}
    for record in profile_records:
        key = str(int(record.get("total_tokens", 0) or 0))
        bucket = profile_by_tokens.setdefault(
            key,
            {
                "steps": 0,
                "wallclock_s": 0.0,
                "forward_s": 0.0,
                "prep_s": 0.0,
                "sample_s": 0.0,
                "post_s": 0.0,
                "scheduled_reqs": {},
                "new_reqs": {},
                "cached_reqs": {},
                "token_buckets": {},
                "req_buckets": {},
            },
        )
        bucket["steps"] += 1
        bucket["wallclock_s"] += float(record.get("total_time", 0.0) or 0.0)
        bucket["forward_s"] += float(record.get("forward_time", 0.0) or 0.0)
        bucket["prep_s"] += float(record.get("prep_time", 0.0) or 0.0)
        bucket["sample_s"] += float(record.get("sample_time", 0.0) or 0.0)
        bucket["post_s"] += float(record.get("post_time", 0.0) or 0.0)
        scheduled_req_key = str(int(record.get("num_scheduled_reqs", 0) or 0))
        new_req_key = str(int(record.get("num_new", 0) or 0))
        cached_req_key = str(int(record.get("num_cached", 0) or 0))
        bucket["scheduled_reqs"][scheduled_req_key] = bucket["scheduled_reqs"].get(scheduled_req_key, 0) + 1
        bucket["new_reqs"][new_req_key] = bucket["new_reqs"].get(new_req_key, 0) + 1
        bucket["cached_reqs"][cached_req_key] = bucket["cached_reqs"].get(cached_req_key, 0) + 1
        token_key = ",".join(str(v) for v in record.get("token_buckets", [])) or "?"
        req_key = ",".join(str(v) for v in record.get("req_buckets", [])) or "?"
        bucket["token_buckets"][token_key] = bucket["token_buckets"].get(token_key, 0) + 1
        bucket["req_buckets"][req_key] = bucket["req_buckets"].get(req_key, 0) + 1
    return {
        "wallclock_s": wallclock,
        "generated_tokens": generated,
        "output_tok_s": generated / wallclock if wallclock > 0 else 0.0,
        "per_request_generated": [len(req.output_token_ids) for req in requests],
        "profile_steps": len(profile_records),
        "profile_by_total_tokens": profile_by_tokens,
    }


def main() -> int:
    """CLI entry point for the Qwen3.5-9B no-MTP eSurge throughput benchmark.

    Parses arguments, validates the TPU backend, loads the model, builds the
    runner, runs warmups and timed trials, and writes a JSON results document
    to ``--json-out``.

    Returns:
        int: Process exit code (``0`` on success).

    Raises:
        SystemExit: If the JAX backend is not TPU.
        FileNotFoundError: If ``--model`` points at a path that does not exist.
    """
    args = parse_args()
    configure_tpu_process()

    import jax
    import jax.numpy as jnp

    import easydel as ed
    from easydel.inference.esurge.runners import eSurgeRunner

    if jax.default_backend() != "tpu":
        raise SystemExit(f"Refusing to benchmark on {jax.default_backend()!r}; TPU is required.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    print("EasyDeL eSurge no-MTP throughput benchmark")
    print(f"model       : {model_path}")
    print(f"devices     : {jax.devices()}")
    print(
        "shape       : "
        f"prompt_len={args.prompt_len} output_len={args.output_len} "
        f"num_prompts={args.num_prompts} max_model_len={args.max_model_len}"
    )
    print(
        "runtime     : "
        f"TP=4 hbm={args.hbm_utilization} page_size={args.page_size} "
        f"max_num_batched_tokens={args.max_num_batched_tokens} "
        f"async={not args.no_async} overlap={not args.no_overlap} aot={args.use_aot_forward}"
    )

    partition_axis = ed.PartitionAxis()
    partition_axis.hidden_state_axis = None
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=str(model_path),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        config_kwargs={
            "attn_mechanism": "ragged_page_attention_v3",
            "decode_attn_mechanism": "ragged_page_attention_v3",
            "attn_dtype": jnp.bfloat16,
            "kvdtype": jnp.bfloat16,
            "freq_max_position_embeddings": args.max_model_len,
            "mask_max_position_embeddings": args.max_model_len,
        },
        sharding_axis_dims=SHARDING_AXIS_DIMS,
        sharding_axis_names=SHARDING_AXIS_NAMES,
        partition_axis=partition_axis,
        auto_shard_model=True,
    )
    text_config = model.config.get_text_config()
    vocab_size = int(getattr(text_config, "vocab_size", getattr(model.config, "vocab_size", 151936)))
    prompts = make_prompts(
        count=args.num_prompts,
        prompt_len=args.prompt_len,
        vocab_size=vocab_size,
        seed=args.seed,
    )

    runner = eSurgeRunner(
        model=model,
        hbm_utilization=args.hbm_utilization,
        page_size=args.page_size,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        min_input_pad=args.max_num_seqs,
        min_token_pad=16,
        max_num_seqs=args.max_num_seqs,
        max_num_seq_buckets=[args.max_num_seqs],
        async_scheduling=not args.no_async,
        use_aot_forward=args.use_aot_forward,
        verbose=args.verbose_runner,
        enable_overlap_execution=not args.no_overlap,
        enable_window_aware_runtime_cap=False,
        kernel_tile_policy="auto",
        drafter=None,
    )
    print(f"runner page_size={runner.page_size} token_buckets={runner.num_tokens_paddings}")
    print(f"runner request_buckets={runner.max_num_seq_buckets}")
    runner.compile(max_num_batched_tokens=args.max_num_batched_tokens)

    for warmup in range(args.warmups):
        result = run_batch(runner, prompts, args.output_len, args.max_num_batched_tokens)
        print(
            f"warmup {warmup + 1}: generated={result['generated_tokens']} "
            f"wallclock={result['wallclock_s']:.3f}s tok/s={result['output_tok_s']:.2f}"
        )

    trial_results = []
    for trial in range(args.trials):
        do_xprof = bool(args.xprof_dir) and (trial + 1) == args.xprof_trial
        if do_xprof:
            xprof_path = Path(args.xprof_dir)
            xprof_path.mkdir(parents=True, exist_ok=True)
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = args.xprof_host_level
            options.python_tracer_level = args.xprof_python_level
            print(f"xprof start: {xprof_path}")
            jax.profiler.start_trace(str(xprof_path), profiler_options=options)
        try:
            if do_xprof:
                with jax.profiler.StepTraceAnnotation("easydel.qwen35_9b_no_mtp", step_num=trial + 1):
                    result = run_batch(runner, prompts, args.output_len, args.max_num_batched_tokens)
            else:
                result = run_batch(runner, prompts, args.output_len, args.max_num_batched_tokens)
        finally:
            if do_xprof:
                jax.profiler.stop_trace()
                print(f"xprof stop : {xprof_path}")
        trial_results.append(result)
        print(
            f"trial {trial + 1}: generated={result['generated_tokens']} "
            f"wallclock={result['wallclock_s']:.3f}s tok/s={result['output_tok_s']:.2f}"
        )

    tok_s = [r["output_tok_s"] for r in trial_results]
    payload = {
        "engine": "easydel-esurge",
        "model": str(model_path),
        "no_mtp": True,
        "config": vars(args),
        "runner": {
            "page_size": runner.page_size,
            "token_buckets": list(runner.num_tokens_paddings),
            "request_buckets": list(runner.max_num_seq_buckets),
            "max_pages_per_req": int(runner.max_pages_per_req),
        },
        "trials": trial_results,
        "summary": {
            "median_output_tok_s": statistics.median(tok_s),
            "mean_output_tok_s": statistics.mean(tok_s),
        },
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("summary    : " + json.dumps(payload["summary"], sort_keys=True))
    print(f"json_out   : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
