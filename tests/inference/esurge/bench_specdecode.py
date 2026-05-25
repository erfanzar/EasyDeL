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

"""Single-request eSurge speculative-decoding benchmark (Qwen3.5 MTP).

Codex's earlier benchmark crashed: it issued 6 concurrent prompts and the
eSurge engine was not precompiled for the resulting `model_step(128, 6)`
bucket (runtime compilation disabled). This benchmark uses the SINGLE-
REQUEST runner path (`max_num_seqs=1`, `max_num_seq_buckets=[1]`) — the
bucket configuration that `tests/inference/esurge/test_spec_decode.py`
already proves compiles and runs — so it avoids that failure entirely.

It loads the real `Qwen/Qwen3.5-2B` checkpoint (which ships pretrained
`mtp.*` weights), drives the real `eSurgeRunner` + `Scheduler` loop, and
times generation WITH and WITHOUT the MTP drafter.

Run:  /home/erfan/EasyDeL/.venv/bin/python tests/inference/esurge/bench_specdecode.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ["JAX_PLATFORM_NAME"] = "tpu"

CKPT = "Qwen/Qwen3.5-2B"
CKPT_DIR = "/dev/shm/easydel_ckpts/hf"
FALLBACK_PROMPT = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5]
PROMPT_TEXT = os.environ.get(
    "EASYDEL_BENCH_PROMPT_TEXT",
    "Write a concise technical explanation of speculative decoding for transformer inference, "
    "including how draft tokens are verified and why acceptance rate affects throughput.",
)
PROMPT: list[int] = []
MAX_NEW = int(os.environ.get("EASYDEL_BENCH_MAX_NEW", "96"))
MAX_MODEL_LEN = int(os.environ.get("EASYDEL_BENCH_MAX_MODEL_LEN", "256"))
MAX_BATCHED = int(os.environ.get("EASYDEL_BENCH_MAX_BATCHED", "192"))
MIN_TOKEN_PAD = max(16, int(os.environ.get("EASYDEL_BENCH_MIN_TOKEN_PAD", "16")))
USE_AOT_FORWARD = os.environ.get("EASYDEL_BENCH_USE_AOT", "0").lower() in {"1", "true", "yes"}
VERBOSE_RUNNER = os.environ.get("EASYDEL_BENCH_VERBOSE", "0").lower() in {"1", "true", "yes"}
BENCH_TRIALS = int(os.environ.get("EASYDEL_BENCH_TRIALS", "3"))
BENCH_WARMUPS = int(os.environ.get("EASYDEL_BENCH_WARMUPS", "3"))
BENCH_MIN_SPEC_WARMUPS = 2
PRINT_TRIALS = os.environ.get("EASYDEL_BENCH_PRINT_TRIALS", "0").lower() in {"1", "true", "yes"}
PROFILE_JSON = os.environ.get("EASYDEL_BENCH_PROFILE_JSON")
XPROF_DIR = os.environ.get("EASYDEL_BENCH_XPROF_DIR")
XPROF_LABELS = {
    part.strip() for part in os.environ.get("EASYDEL_BENCH_XPROF_LABELS", "baseline-greedy").split(",") if part.strip()
}
XPROF_TRIAL = int(os.environ.get("EASYDEL_BENCH_XPROF_TRIAL", "1"))
XPROF_HOST_TRACER_LEVEL = int(os.environ.get("EASYDEL_BENCH_XPROF_HOST_LEVEL", "2"))
XPROF_PYTHON_TRACER_LEVEL = int(os.environ.get("EASYDEL_BENCH_XPROF_PYTHON_LEVEL", "0"))
PAGE_SIZE = int(os.environ.get("EASYDEL_BENCH_PAGE_SIZE", "16"))
HBM_UTILIZATION = float(os.environ.get("EASYDEL_BENCH_HBM_UTILIZATION", "0.25"))
KERNEL_TILE_POLICY = os.environ.get("EASYDEL_BENCH_KERNEL_TILE_POLICY", "auto")
OVERLAP_EXECUTION = os.environ.get("EASYDEL_BENCH_OVERLAP", "0").lower() in {"1", "true", "yes"}
ASYNC_SCHEDULING = os.environ.get("EASYDEL_BENCH_ASYNC", "0").lower() in {"1", "true", "yes"}
REPLICATE_HIDDEN_TP = os.environ.get("EASYDEL_BENCH_REPLICATE_HIDDEN_TP", "1").lower() in {"1", "true", "yes"}
BENCH_MODES = {
    part.strip().lower()
    for part in os.environ.get(
        "EASYDEL_BENCH_MODES",
        "baseline-greedy,spec-greedy,baseline-sampling,spec-sampling",
    ).split(",")
    if part.strip()
}
REJECT_BACKOFF = 0
MTP_DRAFT_TOKENS = 1
BENCH_SHARDING_AXIS_DIMS = tuple(
    int(part.strip())
    for part in os.environ.get("EASYDEL_BENCH_SHARDING_AXIS_DIMS", "1,1,1,1,-1,1").split(",")
    if part.strip()
)
BENCH_SHARDING_AXIS_NAMES = tuple(
    part.strip()
    for part in os.environ.get("EASYDEL_BENCH_SHARDING_AXIS_NAMES", "pp,dp,fsdp,ep,tp,sp").split(",")
    if part.strip()
)

if len(BENCH_SHARDING_AXIS_DIMS) != len(BENCH_SHARDING_AXIS_NAMES):
    raise ValueError(
        "EASYDEL_BENCH_SHARDING_AXIS_DIMS and EASYDEL_BENCH_SHARDING_AXIS_NAMES "
        f"must have matching lengths, got {BENCH_SHARDING_AXIS_DIMS} and {BENCH_SHARDING_AXIS_NAMES}."
    )


PROFILE_TIME_FIELDS = (
    "total_time",
    "runner_time",
    "copy_enqueue_time",
    "token_materialize_time",
    "prep_time",
    "prep_host_time",
    "prep_put_time",
    "prep_extra_put_time",
    "prep_batch_metadata_time",
    "prep_handoff_time",
    "prep_sampler_window_time",
    "prep_ensure_variants_time",
    "prep_pack_inputs_time",
    "forward_time",
    "model_enqueue_time",
    "sampler_enqueue_time",
    "logits_wait_time",
    "exec_enqueue_time",
    "exec_wait_time",
    "sample_time",
    "sampler_wait_time",
    "execute_overhead_time",
    "metrics_time",
    "prev_async_time",
    "step_time",
    "step_gap_time",
    "sync_time",
    "post_time",
    "spec_project_time",
    "spec_draft_time",
    "spec_suffix_time",
    "spec_replay_time",
    "spec_commit_time",
    "misc_time",
)

BENCH_RESULTS: dict[str, dict] = {}


def _local_snapshots() -> list[Path]:
    snapshot_root = Path(CKPT_DIR) / "models--Qwen--Qwen3.5-2B" / "snapshots"
    return sorted(p for p in snapshot_root.glob("*/") if (p / "config.json").is_file())


def _preflight_environment() -> None:
    if not Path("/dev/vfio/0").exists():
        raise SystemExit("TPU device /dev/vfio/0 was not found; cannot run the TPU benchmark in this environment.")
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)


_preflight_environment()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

if jax.default_backend() != "tpu":
    raise SystemExit(f"Refusing to benchmark on {jax.default_backend()!r}; this benchmark is TPU-only.")
if not all(device.platform == "tpu" for device in jax.devices()):
    raise SystemExit(f"Refusing mixed/non-TPU devices: {jax.devices()}")

import easydel as ed  # noqa: E402
from easydel.inference.esurge.request import EngineRequest  # noqa: E402
from easydel.inference.esurge.runners import eSurgeRunner  # noqa: E402
from easydel.inference.esurge.scheduler import Scheduler  # noqa: E402
from easydel.inference.sampling_params import SamplingParams  # noqa: E402


def assert_tpu_backend() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    if backend != "tpu":
        raise SystemExit(f"Refusing to benchmark on {backend!r}; this benchmark is TPU-only.")
    if not devices or not all(device.platform == "tpu" for device in devices):
        raise SystemExit(f"Refusing mixed/non-TPU devices: {devices}")


def load_model():
    """Load Qwen3.5-2B through AutoEasyDeL; report whether MTP loaded."""
    from huggingface_hub import snapshot_download

    partition_axis = ed.PartitionAxis()
    if REPLICATE_HIDDEN_TP:
        partition_axis.hidden_state_axis = None

    local_snapshots = _local_snapshots()
    if local_snapshots:
        path = str(local_snapshots[-1])
    else:
        path = snapshot_download(
            CKPT,
            cache_dir=CKPT_DIR,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
        )
    print(f"  checkpoint: {path}")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        sharding_axis_dims=BENCH_SHARDING_AXIS_DIMS,
        sharding_axis_names=BENCH_SHARDING_AXIS_NAMES,
        partition_axis=partition_axis,
        auto_shard_model=True,
    )
    has_mtp = bool(getattr(model, "has_mtp", lambda: False)())
    print(f"  model loaded: {type(model).__name__}  has_mtp={has_mtp}")
    print(
        "  model mesh: "
        f"dims={getattr(model.config, 'sharding_axis_dims', None)} "
        f"names={getattr(model.config, 'sharding_axis_names', None)} "
        f"hidden_state_axis={getattr(getattr(model.config, 'partition_axis', None), 'hidden_state_axis', None)!r} "
        f"mesh={getattr(model.config, 'mesh', None)}"
    )
    if not has_mtp:
        raise RuntimeError("loaded model has no MTP head — cannot benchmark spec-decode")
    return model, path


def resolve_prompt_tokens(checkpoint_path: str) -> list[int]:
    """Use a real tokenizer prompt by default; keep synthetic IDs as fallback."""
    raw_tokens = os.environ.get("EASYDEL_BENCH_PROMPT_TOKENS")
    if raw_tokens:
        tokens = [int(tok.strip()) for tok in raw_tokens.split(",") if tok.strip()]
        if not tokens:
            raise ValueError("EASYDEL_BENCH_PROMPT_TOKENS was set but no token IDs were parsed.")
        return tokens

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True, trust_remote_code=True)
        tokens = tokenizer(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
        if tokens:
            return [int(tok) for tok in tokens]
    except Exception as exc:
        print(f"  tokenizer prompt failed ({exc!r}); falling back to synthetic token IDs")

    return list(FALLBACK_PROMPT)


def build_runner(model, drafter):
    """Single-request eSurge runner (the bucket config known to compile)."""
    async_scheduling = bool(ASYNC_SCHEDULING and drafter is None)
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=HBM_UTILIZATION,
        page_size=PAGE_SIZE,
        max_cache_tokens=4096,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_BATCHED,
        min_input_pad=1,
        min_token_pad=MIN_TOKEN_PAD,
        max_num_seqs=1,
        max_num_seq_buckets=[1],
        async_scheduling=async_scheduling,
        use_aot_forward=USE_AOT_FORWARD,
        verbose=VERBOSE_RUNNER,
        enable_overlap_execution=OVERLAP_EXECUTION,
        kernel_tile_policy=KERNEL_TILE_POLICY,
        drafter=drafter,
    )
    runner.compile(max_num_batched_tokens=MAX_BATCHED)
    if runner.max_model_len >= 16 and runner.num_tokens_paddings[0] < 16:
        raise RuntimeError(f"decode token bucket floor must be >=16, got {runner.num_tokens_paddings}")
    return runner


def reset_runner_for_request(runner):
    """Clear per-request runner state between independent benchmark trials."""
    runner.reset_state()
    if runner.executor_manager.kv_pages is not None:
        runner.executor_manager.kv_pages = runner.executor_manager.kv_pages.reset()
        jax.block_until_ready(runner.executor_manager.kv_pages)


def make_sampling_params(*, max_new_tokens: int, temperature: float, top_p: float) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        ignore_eos=True,
    )


def generate_once(
    runner,
    *,
    run_id: int,
    temperature: float,
    top_p: float,
    max_new_tokens=MAX_NEW,
):
    """Drive scheduler+runner for one request; return (tokens, wallclock_s)."""
    assert_tpu_backend()
    reset_runner_for_request(runner)
    async_scheduling = bool(getattr(runner, "async_scheduling", False))
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=MAX_BATCHED,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        num_speculative_tokens=runner.num_speculative_tokens,
    )
    request = EngineRequest(
        request_id=f"bench-{run_id}",
        prompt_token_ids=list(PROMPT),
        sampling_params=make_sampling_params(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        ),
        eos_token_id=None,
    )
    scheduler.add_request(request)

    def _zero_schedule_needs_update(scheduler_output) -> bool:
        return bool(
            scheduler_output.finished_req_ids
            or scheduler_output.preempted_req_ids
            or scheduler_output.scheduled_new_reqs
            or scheduler_output.scheduled_cached_reqs.num_reqs
        )

    def _drain_async(pending) -> None:
        future, pending_scheduler_output = pending
        model_output = runner.wait_for_execution(future)
        scheduler.update_from_output(pending_scheduler_output, model_output)

    def _execute_positive(scheduler_output):
        if (
            async_scheduling
            and scheduler_output.total_num_scheduled_tokens > 0
            and runner.can_dispatch_next_before_async_drain(scheduler_output)
        ):
            with jax.profiler.TraceAnnotation("easydel.execute_model_async"):
                return runner.execute_model_async(scheduler_output), scheduler_output
        with jax.profiler.TraceAnnotation("easydel.execute_model"):
            output = runner.execute_model(scheduler_output)
        scheduler.update_from_output(scheduler_output, output)
        return None

    t0 = time.time()
    if not async_scheduling:
        for _ in range(max_new_tokens + 8):
            with jax.profiler.TraceAnnotation("easydel.scheduler.schedule"):
                scheduler_output = scheduler.schedule()
            with jax.profiler.TraceAnnotation("easydel.execute_model"):
                output = runner.execute_model(scheduler_output)
            with jax.profiler.TraceAnnotation("easydel.scheduler.update_from_output"):
                scheduler.update_from_output(scheduler_output, output)
            if request.is_finished():
                break
    else:
        pending = None
        for _ in range(max_new_tokens + 16):
            if pending is not None:
                future, prev_scheduler_output = pending
                if runner.can_dispatch_next_before_async_drain(prev_scheduler_output):
                    with jax.profiler.TraceAnnotation("easydel.scheduler.schedule"):
                        scheduler_output = scheduler.schedule()
                    if scheduler_output.total_num_scheduled_tokens > 0:
                        next_pending = None
                        if runner.can_dispatch_next_before_async_drain(scheduler_output):
                            with jax.profiler.TraceAnnotation("easydel.execute_model_async"):
                                next_pending = runner.execute_model_async(scheduler_output), scheduler_output
                        _drain_async((future, prev_scheduler_output))
                        if next_pending is None:
                            next_pending = _execute_positive(scheduler_output)
                        pending = next_pending
                        if request.is_finished() and pending is None:
                            break
                        continue
                    _drain_async((future, prev_scheduler_output))
                    pending = None
                    if _zero_schedule_needs_update(scheduler_output):
                        with jax.profiler.TraceAnnotation("easydel.execute_model"):
                            output = runner.execute_model(scheduler_output)
                        with jax.profiler.TraceAnnotation("easydel.scheduler.update_from_output"):
                            scheduler.update_from_output(scheduler_output, output)
                    if request.is_finished():
                        break
                    continue

                _drain_async((future, prev_scheduler_output))
                pending = None
                if request.is_finished():
                    break

            with jax.profiler.TraceAnnotation("easydel.scheduler.schedule"):
                scheduler_output = scheduler.schedule()
            if scheduler_output.total_num_scheduled_tokens == 0:
                if _zero_schedule_needs_update(scheduler_output):
                    with jax.profiler.TraceAnnotation("easydel.execute_model"):
                        output = runner.execute_model(scheduler_output)
                    with jax.profiler.TraceAnnotation("easydel.scheduler.update_from_output"):
                        scheduler.update_from_output(scheduler_output, output)
                if request.is_finished():
                    break
                continue
            pending = _execute_positive(scheduler_output)
            if request.is_finished() and pending is None:
                break

        if pending is not None:
            _drain_async(pending)
    jax.block_until_ready(jnp.zeros(()))  # flush
    wallclock = time.time() - t0
    return list(request.output_token_ids), wallclock


def reset_spec_counters(runner) -> None:
    runner.spec_decode_num_drafts_generated = 0
    runner.spec_decode_num_drafts_accepted = 0
    runner.spec_decode_num_verify_steps = 0
    if hasattr(runner, "spec_decode_debug_traces"):
        runner.spec_decode_debug_traces.clear()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _phase_value(record, field: str) -> float:
    value = record.get(field, 0.0)
    return float(value or 0.0)


def _summarize_group(records: list[dict]) -> dict:
    token_buckets: dict[str, int] = {}
    req_buckets: dict[str, int] = {}
    for record in records:
        token_key = ",".join(str(v) for v in record.get("token_buckets", [])) or "?"
        req_key = ",".join(str(v) for v in record.get("req_buckets", [])) or "?"
        token_buckets[token_key] = token_buckets.get(token_key, 0) + 1
        req_buckets[req_key] = req_buckets.get(req_key, 0) + 1

    timings_ms = {}
    for field in PROFILE_TIME_FIELDS:
        values_ms = [_phase_value(record, field) * 1e3 for record in records if field in record]
        timings_ms[field] = {
            "mean": statistics.mean(values_ms) if values_ms else None,
            "median": statistics.median(values_ms) if values_ms else None,
            "p95": _percentile(values_ms, 95.0),
        }

    total_times = [_phase_value(record, "total_time") for record in records]
    median_total = statistics.median(total_times) if total_times else 0.0
    return {
        "steps": len(records),
        "total_tokens": sum(int(record.get("total_tokens", 0) or 0) for record in records),
        "median_step_tps": (1.0 / median_total) if median_total > 0 else None,
        "token_bucket_counts": token_buckets,
        "req_bucket_counts": req_buckets,
        "timings_ms": timings_ms,
    }


def summarize_runner_profile(runner, start_index: int) -> dict:
    records = [dict(record) for record in list(getattr(runner, "_perf_phase_history", []))[start_index:]]
    decode_records = [record for record in records if int(record.get("total_tokens", 0) or 0) == 1]
    prefill_records = [record for record in records if int(record.get("total_tokens", 0) or 0) > 1]
    return {
        "records": records,
        "summary": {
            "all": _summarize_group(records),
            "prefill": _summarize_group(prefill_records),
            "decode": _summarize_group(decode_records),
        },
    }


def bench(runner, label, *, temperature: float, top_p: float):
    """Warm up once (excluded), then timed runs; return median tok/s."""
    toks: list[int] = []
    warmups = max(0, BENCH_WARMUPS)
    if label.startswith("spec-"):
        warmups = max(warmups, max(0, BENCH_MIN_SPEC_WARMUPS))
    for warmup_idx in range(warmups):
        toks, _ = generate_once(
            runner,
            run_id=-(warmup_idx + 1),
            temperature=temperature,
            top_p=top_p,
        )  # warmup (compile / lazy XLA runtime init)
    reset_spec_counters(runner)
    times = []
    profile_start = len(getattr(runner, "_perf_phase_history", []))
    for i in range(BENCH_TRIALS):
        do_xprof = XPROF_DIR and label in XPROF_LABELS and (i + 1) == XPROF_TRIAL
        if do_xprof:
            xprof_path = Path(XPROF_DIR) / label
            xprof_path.mkdir(parents=True, exist_ok=True)
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = XPROF_HOST_TRACER_LEVEL
            options.python_tracer_level = XPROF_PYTHON_TRACER_LEVEL
            print(f"    xprof start: {xprof_path}")
            jax.profiler.start_trace(str(xprof_path), profiler_options=options)
        try:
            with jax.profiler.StepTraceAnnotation(f"easydel.{label}", step_num=i + 1):
                toks, dt = generate_once(runner, run_id=i + 1, temperature=temperature, top_p=top_p)
        finally:
            if do_xprof:
                jax.block_until_ready(jnp.zeros(()))
                jax.profiler.stop_trace()
                print(f"    xprof stop : {xprof_path}")
        times.append(dt)
        if PRINT_TRIALS:
            print(
                f"    trial {i + 1}: generated={len(toks)} wallclock={dt:.3f}s tok/s={(len(toks) / dt if dt > 0 else 0.0):.2f}"
            )
    med = statistics.median(times)
    n = len(toks)
    tps = n / med if med > 0 else 0.0
    print(f"  [{label}] generated={n} median_wallclock={med:.3f}s tok/s={tps:.2f}")
    BENCH_RESULTS[label] = {
        "generated_tokens": n,
        "median_wallclock_s": med,
        "tok_s": tps,
        "trial_wallclock_s": list(times),
        "temperature": temperature,
        "top_p": top_p,
    }
    if PROFILE_JSON:
        profile = summarize_runner_profile(runner, profile_start)
        BENCH_RESULTS[label]["profile"] = profile
        decode = profile["summary"]["decode"]
        total_ms = decode["timings_ms"]["total_time"]["median"]
        forward_ms = decode["timings_ms"]["forward_time"]["median"]
        sample_ms = decode["timings_ms"]["sample_time"]["median"]
        prep_ms = decode["timings_ms"]["prep_time"]["median"]
        if total_ms is not None:
            forward_ms = 0.0 if forward_ms is None else forward_ms
            sample_ms = 0.0 if sample_ms is None else sample_ms
            prep_ms = 0.0 if prep_ms is None else prep_ms
            print(
                "    profile decode median: "
                f"total={total_ms:.2f}ms fwd={forward_ms:.2f}ms "
                f"sample={sample_ms:.2f}ms prep={prep_ms:.2f}ms"
            )
    return tps, n, toks


def report(
    mode: str, base_tps: float, spec_tps: float, spec_runner, *, spec_toks, base_toks, max_new_tokens: int
) -> None:
    gen = spec_runner.spec_decode_num_drafts_generated
    acc = spec_runner.spec_decode_num_drafts_accepted
    steps = spec_runner.spec_decode_num_verify_steps
    accept_rate = (acc / gen) if gen else 0.0
    speedup = (spec_tps / base_tps) if base_tps else 0.0

    print("\n" + "=" * 78)
    print(f"RESULTS — Qwen3.5-2B, eSurge, single request, {mode}, {max_new_tokens} new tokens")
    print("=" * 78)
    print(f"  baseline       : {base_tps:7.2f} tok/s")
    print(f"  spec-decode    : {spec_tps:7.2f} tok/s")
    print(f"  WALLCLOCK SPEEDUP: {speedup:.2f}x")
    print(f"  drafts generated/accepted: {gen}/{acc}  (acceptance rate {accept_rate:.1%})")
    print(f"  verify steps   : {steps}")
    if mode == "greedy":
        print(f"  correctness    : spec == baseline tokens -> {spec_toks == base_toks}")
        if spec_toks != base_toks:
            print("  WARNING: greedy output differs — spec-decode must be exact. Investigate.")
    traces = getattr(spec_runner, "spec_decode_debug_traces", [])
    if traces:
        print("  debug traces:")
        for trace in traces[:5]:
            print(
                "    "
                f"source={trace['source']} start={trace['start_pos']} real={trace['real_count']} "
                f"accepted={trace['accepted']} corrected={trace['corrected_token']}"
            )
            for row in trace["rows"]:
                print(
                    "      "
                    f"i={row['draft_index']} target_row={row['target_local_index']} "
                    f"draft_pos={row['draft_token_position']} sched={row['scheduled_draft_token']} "
                    f"buf={row['buffer_draft_token']} match={row['scheduler_buffer_match']} "
                    f"argmax={row.get('target_argmax')} rank={row.get('draft_rank_under_target')}"
                )
    print("=" * 78)


def main():
    print("=" * 78)
    print("eSurge speculative-decoding benchmark — Qwen3.5-2B, single request")
    print("=" * 78)
    print(
        "config: "
        f"max_new={MAX_NEW} max_model_len={MAX_MODEL_LEN} max_batched={MAX_BATCHED} "
        f"min_token_pad={MIN_TOKEN_PAD} "
        f"use_aot={USE_AOT_FORWARD} mtp_draft_tokens={MTP_DRAFT_TOKENS} reject_backoff={REJECT_BACKOFF} "
        f"warmups={BENCH_WARMUPS} min_spec_warmups={BENCH_MIN_SPEC_WARMUPS} "
        f"trials={BENCH_TRIALS} modes={sorted(BENCH_MODES)} verbose={VERBOSE_RUNNER} "
        f"page_size={PAGE_SIZE} hbm={HBM_UTILIZATION} tile_policy={KERNEL_TILE_POLICY} "
        f"overlap={OVERLAP_EXECUTION} async={ASYNC_SCHEDULING} "
        f"sharding_axis_dims={BENCH_SHARDING_AXIS_DIMS} sharding_axis_names={BENCH_SHARDING_AXIS_NAMES}"
    )
    print(f"devices: {jax.devices()}")
    global PROMPT
    model, checkpoint_path = load_model()
    PROMPT = resolve_prompt_tokens(checkpoint_path)
    print(f"prompt_tokens: {len(PROMPT)}")
    if len(PROMPT) + MAX_NEW >= MAX_MODEL_LEN:
        raise RuntimeError(
            f"prompt length ({len(PROMPT)}) + max_new ({MAX_NEW}) must be < max_model_len ({MAX_MODEL_LEN})."
        )

    base_runner = None
    spec_runner = None
    base_tps = spec_tps = sample_base_tps = sample_spec_tps = None
    base_toks: list[int] = []
    spec_toks: list[int] = []
    sample_base_toks: list[int] = []
    sample_spec_toks: list[int] = []

    if BENCH_MODES & {"baseline-greedy", "baseline-sampling"}:
        print("\n[runner] baseline (no drafter)")
        base_runner = build_runner(model, drafter=None)

    if "baseline-greedy" in BENCH_MODES and base_runner is not None:
        print("\n[1/4] baseline greedy (no drafter)")
        base_tps, _base_n, base_toks = bench(base_runner, "baseline-greedy", temperature=0.0, top_p=1.0)

    if BENCH_MODES & {"spec-greedy", "spec-sampling"}:
        print("\n[runner] speculative (MTP drafter)")
        drafter = model.drafter(method="mtp", num_draft_tokens=MTP_DRAFT_TOKENS)
        spec_runner = build_runner(model, drafter=drafter)

    if "spec-greedy" in BENCH_MODES and spec_runner is not None:
        print("\n[2/4] speculative greedy (MTP drafter)")
        spec_tps, _spec_n, spec_toks = bench(spec_runner, "spec-greedy", temperature=0.0, top_p=1.0)
        BENCH_RESULTS["spec-greedy"].update(
            {
                "drafts_generated": spec_runner.spec_decode_num_drafts_generated,
                "drafts_accepted": spec_runner.spec_decode_num_drafts_accepted,
                "verify_steps": spec_runner.spec_decode_num_verify_steps,
            }
        )
        if base_tps is not None:
            report(
                "greedy",
                base_tps,
                spec_tps,
                spec_runner,
                spec_toks=spec_toks,
                base_toks=base_toks,
                max_new_tokens=MAX_NEW,
            )

    if "baseline-sampling" in BENCH_MODES and base_runner is not None:
        print("\n[3/4] baseline sampling (temperature=0.8, top_p=0.95)")
        sample_base_tps, _sample_base_n, sample_base_toks = bench(
            base_runner,
            "baseline-sampling",
            temperature=0.8,
            top_p=0.95,
        )

    if "spec-sampling" in BENCH_MODES and spec_runner is not None:
        print("\n[4/4] speculative sampling (temperature=0.8, top_p=0.95)")
        sample_spec_tps, _sample_spec_n, sample_spec_toks = bench(
            spec_runner,
            "spec-sampling",
            temperature=0.8,
            top_p=0.95,
        )
        BENCH_RESULTS["spec-sampling"].update(
            {
                "drafts_generated": spec_runner.spec_decode_num_drafts_generated,
                "drafts_accepted": spec_runner.spec_decode_num_drafts_accepted,
                "verify_steps": spec_runner.spec_decode_num_verify_steps,
            }
        )
        if sample_base_tps is not None:
            report(
                "sampling temperature=0.8 top_p=0.95",
                sample_base_tps,
                sample_spec_tps,
                spec_runner,
                spec_toks=sample_spec_toks,
                base_toks=sample_base_toks,
                max_new_tokens=MAX_NEW,
            )
    if PROFILE_JSON:
        payload = {
            "benchmark": "bench_specdecode.py",
            "checkpoint": checkpoint_path,
            "prompt_tokens": len(PROMPT),
            "config": {
                "max_new": MAX_NEW,
                "max_model_len": MAX_MODEL_LEN,
                "max_batched": MAX_BATCHED,
                "min_token_pad": MIN_TOKEN_PAD,
                "page_size": PAGE_SIZE,
                "hbm_utilization": HBM_UTILIZATION,
                "kernel_tile_policy": KERNEL_TILE_POLICY,
                "use_aot_forward": USE_AOT_FORWARD,
                "overlap_execution": OVERLAP_EXECUTION,
                "async_scheduling": ASYNC_SCHEDULING,
                "warmups": BENCH_WARMUPS,
                "min_spec_warmups": BENCH_MIN_SPEC_WARMUPS,
                "trials": BENCH_TRIALS,
                "modes": sorted(BENCH_MODES),
                "sharding_axis_dims": BENCH_SHARDING_AXIS_DIMS,
                "sharding_axis_names": BENCH_SHARDING_AXIS_NAMES,
            },
            "results": BENCH_RESULTS,
        }
        profile_path = Path(PROFILE_JSON)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nprofile_json: {profile_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
