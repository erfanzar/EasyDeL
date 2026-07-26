#!/usr/bin/env python3
"""A/B benchmark: eSurge batched serving with bf16 vs quantized (W8/NF4) weights.

Copy of scripts/bench_spec_ab.py extended with weight-quantization knobs:
the model can be quantized on load (``--quantization int8|nf4|affine``) via
EasyDeL's quantize-on-load path (from_pretrained ``quantization_config`` +
``apply_quantization=True``), and the ejkernel quantized-matmul runtime can be
steered with ``--qmm-platform`` / ``--qmm-tpu-path``. Decode-only tok/s and
per-phase decode breakdowns come from the same measurement loop as the spec
A/B harness so numbers are directly comparable with bf16 baselines produced by
scripts/bench_spec_ab.py --modes off.

Example (Qwen3.6-27B W8 on a v5p-8, tp=4):

    python scripts/bench_quant_ab.py \
        --model /dev/shm/easydel_ckpts/qwen36_27b_spx \
        --tokenizer Qwen/qwen3.6-27b \
        --quantization int8 --concurrency 8,16,32 --modes off

Smoke mode (coherence check, no timing): add ``--smoke`` to generate a few
real prompts and print the decoded continuations.

Original harness docstring follows.

A/B benchmark: eSurge batched serving WITH vs WITHOUT MTP speculative decoding.

Measures aggregate AND decode-only output tok/s at several concurrency levels
from compiled runners, spec-OFF vs spec-ON (the model's built-in MTP head),
with per-step phase breakdowns (verify forward / draft / emit / misc) and
acceptance (tok/step). Prompts are real chat-templated text by default —
random-token prompts depress MTP acceptance and give misleading ratios.

Example (Qwen3.6-27B on a v5p-8, tp=4):

    python scripts/bench_spec_ab.py \
        --model /dev/shm/easydel_ckpts/qwen36_27b_spx \
        --tokenizer Qwen/qwen3.6-27b \
        --concurrency 8,16,32 --num-spec 4

Play knobs: --num-spec (K), --concurrency, --no-persist (disable MTP KV
persistence across verify windows), --prompt-source random (see acceptance
collapse), --modes on (skip the OFF baseline), --prompt-len/--output-len.

Measurement only. No source changes. Greedy sampling (the optimized spec path).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "tpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "tpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="EasyDeL checkpoint path or HF id")
    p.add_argument("--tokenizer", default=None, help="tokenizer id/path (default: --model)")
    p.add_argument(
        "--model-class",
        default="image-text-to-text",
        choices=["image-text-to-text", "causal-lm"],
        help="Auto class used to load the model (Qwen3.5/3.6 VLM checkpoints are image-text-to-text)",
    )
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--prompt-source", default="real", choices=["real", "random"])
    p.add_argument(
        "--prompt-text-file",
        default=None,
        help="long text file to slice real prompts from (default: this repo's WORKSPACE.md/README.md)",
    )
    p.add_argument("--concurrency", default="8,16,32")
    p.add_argument("--off-conc", default=None, help="override concurrency list for spec-OFF")
    p.add_argument("--on-conc", default=None, help="override concurrency list for spec-ON")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--seq-buckets", default="8,16,32")
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--hbm-utilization", type=float, default=0.65)
    p.add_argument("--max-cache-tokens", type=int, default=40960)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--num-spec", type=int, default=4, help="K draft tokens per verify window")
    p.add_argument("--no-persist", action="store_true", help="disable MTP KV persistence across windows")
    p.add_argument("--sharding-axis-dims", default="1,1,1,1,-1,1", help="pp,dp,fsdp,ep,tp,sp")
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--modes", default="off", help="comma list: off,on")
    p.add_argument("--json-out", default="bench_quant_ab.json")
    p.add_argument(
        "--quantization",
        default="none",
        choices=["none", "int8", "nf4", "affine"],
        help="weight quantization applied on load (none = bf16 baseline)",
    )
    p.add_argument("--quant-group-size", type=int, default=64)
    p.add_argument("--quant-bits", type=int, default=None, help="bits for affine mode (int8=8 fixed)")
    p.add_argument(
        "--quant-pattern",
        default=r"^(?!.*(?:embedding|norm|lm_head|in_proj_b|in_proj_a|visual)).*$",
        help=(
            "regex selecting layers to quantize. Default = framework default "
            "(exclude embedding/norm/lm_head) plus the tiny GDN gating projections "
            "in_proj_b / in_proj_a (out_features == num_v_heads: too small to group, "
            "perf-irrelevant, gating-sensitive) plus the vision tower (visual: "
            "irrelevant for text decode; some vision helpers assume dense weights)."
        ),
    )
    p.add_argument(
        "--qmm-platform",
        default=None,
        choices=[None, "xla", "pallas", "auto"],
        help="config-level qmm_platform_override for ejkernel quantized matmul",
    )
    p.add_argument(
        "--qmm-tpu-path",
        default=None,
        choices=[None, "hybrid", "packed", "predecode"],
        help="config-level qmm_tpu_path_override (implies fused dispatch when set)",
    )
    p.add_argument("--smoke", action="store_true", help="generate a few prompts, print decoded text, skip timing")
    p.add_argument("--smoke-n", type=int, default=4)
    return p.parse_args()


def reset_runner(runner):
    runner.reset_state()
    cache = runner.executor_manager.kv_pages
    if cache is None or not hasattr(cache, "views"):
        return
    new_views, changed = [], False
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


def run_batch(runner, prompts, output_len, max_num_batched_tokens):
    """Continuous-batching loop (mirrors scripts/bench_esurge.run_batch) with the
    correct num_draft_tokens for the runner."""
    from easydel.inference.esurge.benchmarking import all_finished
    from easydel.inference.esurge.benchmarking import zero_schedule_needs_update as zero_needs_update
    from easydel.inference.esurge.request import EngineRequest
    from easydel.inference.esurge.scheduler import Scheduler
    from easydel.inference.sampling_params import SamplingParams

    reset_runner(runner)
    # Speculative decoding drives draft/verify/commit inside the synchronous
    # execute_model path; the async dispatch path does not advance it.
    use_async = bool(getattr(runner, "async_scheduling", False)) and int(runner.num_draft_tokens) == 0
    scheduler = Scheduler.from_runner(
        runner,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=False,
        async_scheduling=use_async,
        num_draft_tokens=runner.num_draft_tokens,
    )
    sampling = SamplingParams(max_tokens=output_len, temperature=0.0, top_p=1.0, ignore_eos=True)
    requests = []
    for idx, prompt in enumerate(prompts):
        req = EngineRequest(
            request_id=f"b-{idx}",
            prompt_token_ids=list(prompt),
            sampling_params=sampling.clone(),
            eos_token_id=None,
        )
        scheduler.add_request(req)
        requests.append(req)

    def drain(pending):
        future, pso = pending
        mo = runner.wait_for_execution(future)
        scheduler.update_from_output(pso, mo)

    def execute_positive(so):
        if use_async and so.total_num_scheduled_tokens > 0 and runner.can_dispatch_next_before_async_drain(so):
            return runner.execute_model_async(so), so
        out = runner.execute_model(so)
        scheduler.update_from_output(so, out)
        return None

    profile_start = len(getattr(runner, "_perf_phase_history", []))
    t0 = time.perf_counter()
    pending = None
    max_iters = len(prompts) * (len(prompts[0]) + output_len + 32)
    for _ in range(max_iters):
        if pending is not None:
            future, prev_so = pending
            if runner.can_dispatch_next_before_async_drain(prev_so):
                so = scheduler.schedule()
                if so.total_num_scheduled_tokens > 0:
                    nxt = None
                    if runner.can_dispatch_next_before_async_drain(so):
                        nxt = (runner.execute_model_async(so), so)
                    drain((future, prev_so))
                    if nxt is None:
                        nxt = execute_positive(so)
                    pending = nxt
                    if all_finished(requests) and pending is None:
                        break
                    continue
                drain((future, prev_so))
                pending = None
                if zero_needs_update(so):
                    out = runner.execute_model(so)
                    scheduler.update_from_output(so, out)
                if all_finished(requests):
                    break
                continue
            drain((future, prev_so))
            pending = None
            if all_finished(requests):
                break
        so = scheduler.schedule()
        if so.total_num_scheduled_tokens == 0:
            if zero_needs_update(so):
                out = runner.execute_model(so)
                scheduler.update_from_output(so, out)
            if all_finished(requests):
                break
            continue
        pending = execute_positive(so)
        if all_finished(requests) and pending is None:
            break
    if pending is not None:
        drain(pending)
    if not all_finished(requests):
        unfinished = [r.request_id for r in requests if not r.is_finished()]
        raise RuntimeError(f"did not finish: {unfinished[:8]}")
    jax.block_until_ready(jnp.zeros(()))
    wall = time.perf_counter() - t0
    generated = sum(len(r.output_token_ids) for r in requests)
    records = [dict(r) for r in list(getattr(runner, "_perf_phase_history", []))[profile_start:]]
    return {"wall": wall, "generated": generated, "records": records, "requests": requests}


def decode_stats(records):
    """Median phase times (ms) over pure-decode steps (num_new==0)."""
    dec = [r for r in records if int(r.get("num_new", 0) or 0) == 0 and int(r.get("total_tokens", 0) or 0) > 0]
    pre = [r for r in records if int(r.get("num_new", 0) or 0) > 0]

    def med(rs, k):
        vals = [float(r.get(k, 0.0) or 0.0) for r in rs if k in r]
        return statistics.median(vals) * 1e3 if vals else None

    tok_buckets = {}
    for r in dec:
        tb = ",".join(str(v) for v in r.get("token_buckets", [])) or "?"
        tok_buckets[tb] = tok_buckets.get(tb, 0) + 1
    return {
        "decode_steps": len(dec),
        "decode_fwd_ms": med(dec, "forward_time"),
        "decode_total_ms": med(dec, "total_time"),
        "decode_prep_ms": med(dec, "prep_time"),
        "decode_sample_ms": med(dec, "sample_time"),
        "decode_spec_draft_ms": med(dec, "spec_draft_time"),
        "decode_spec_project_ms": med(dec, "spec_project_time"),
        "decode_spec_argmax_sync_ms": med(dec, "spec_argmax_sync_time"),
        "decode_spec_meta_ms": med(dec, "spec_meta_time"),
        "decode_spec_emit_write_ms": med(dec, "spec_emit_write_time"),
        "decode_spec_commit_ms": med(dec, "spec_commit_time"),
        "decode_post_ms": med(dec, "post_time"),
        "decode_sync_ms": med(dec, "sync_time"),
        "decode_misc_ms": med(dec, "misc_time"),
        "decode_token_buckets": tok_buckets,
        "prefill_steps": len(pre),
        "prefill_fwd_ms": med(pre, "forward_time"),
    }


def run_conc(runner, n, args):
    from easydel.inference.esurge.benchmarking import make_prompts, make_real_prompts

    if args.prompt_source == "random":
        prompts = make_prompts(count=n, prompt_len=args.prompt_len, vocab_size=runner._vocab_size)
    else:
        prompts = make_real_prompts(
            count=n,
            prompt_len=args.prompt_len,
            tokenizer_name=args.tokenizer or args.model,
            text_file=args.prompt_text_file,
            doc_paths=(REPO_ROOT / "WORKSPACE.md", REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"),
        )
    for _ in range(args.warmups):
        run_batch(runner, prompts, args.output_len, args.max_num_batched_tokens)
    trials = []
    last = None
    for _ in range(args.trials):
        res = run_batch(runner, prompts, args.output_len, args.max_num_batched_tokens)
        trials.append(res)
        last = res
    walls = [t["wall"] for t in trials]
    gens = [t["generated"] for t in trials]
    med_wall = statistics.median(walls)
    gen = gens[0]
    tok_s = gen / med_wall if med_wall > 0 else 0.0
    try:
        ds = decode_stats(last["records"])
    except Exception as exc:  # never lose a measured run to post-processing
        ds = {"decode_stats_error": repr(exc)}
    # Acceptance: emitted decode tokens per request per decode step (equal-length
    # prompts/outputs => every request rides every step; prefill emits one token
    # per request outside the decode-step count).
    dec_steps = int(ds.get("decode_steps") or 0)
    tok_per_step = ((gen - n) / (dec_steps * n)) if dec_steps > 0 else None
    ds["tok_per_step"] = tok_per_step
    dtot = ds.get("decode_total_ms")
    if dtot and tok_per_step:
        ds["decode_tok_s"] = n * tok_per_step / (dtot / 1e3)
    else:
        ds["decode_tok_s"] = (n / (dtot / 1e3)) if dtot else None
    return {
        "n": n,
        "generated": gen,
        "median_wall_s": med_wall,
        "tok_s": tok_s,
        "trial_tok_s": [g / w if w > 0 else 0.0 for g, w in zip(gens, walls)],
        "sample_output_ids": [int(t) for t in list(last["requests"][0].output_token_ids)[:48]],
        **ds,
    }


def build_runner(model, *, drafter, args):
    from easydel.inference.esurge.runners import eSurgeRunner

    seq_buckets = [int(x) for x in args.seq_buckets.split(",") if x.strip()]
    runner = eSurgeRunner(
        model=model,
        hbm_utilization=args.hbm_utilization,
        page_size=args.page_size,
        max_cache_tokens=args.max_cache_tokens,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        min_input_pad=1,
        min_token_pad=16,
        max_num_seqs=args.max_num_seqs,
        max_num_seq_buckets=seq_buckets,
        use_aot_forward=True,
        verbose=False,
        kernel_tile_policy="auto",
        drafter=drafter,
    )
    tc = model.config.get_text_config()
    runner._vocab_size = int(getattr(tc, "vocab_size", 151936))
    print(
        f"  runner: token_buckets={runner.num_tokens_paddings} req_buckets={runner.max_num_seq_buckets} "
        f"num_spec={runner.num_draft_tokens}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner.compile(max_num_batched_tokens=args.max_num_batched_tokens)
    print(f"  compiled in {time.perf_counter() - t0:.1f}s", flush=True)
    return runner


def free_runner(runner):
    try:
        runner.executor_manager.kv_pages = None
    except Exception:
        pass
    del runner
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def main():
    args = parse_args()
    if jax.default_backend() != "tpu":
        raise SystemExit(f"need TPU, got {jax.default_backend()}")
    import easydel as ed
    from easydel.utils import set_inference_mode

    print(f"devices: {jax.devices()}", flush=True)
    conc = [int(x) for x in args.concurrency.split(",") if x.strip()]
    off_conc = [int(x) for x in args.off_conc.split(",")] if args.off_conc else conc
    on_conc = [int(x) for x in args.on_conc.split(",")] if args.on_conc else conc
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    axis_dims = tuple(int(x) for x in args.sharding_axis_dims.split(","))

    auto_cls = (
        ed.AutoEasyDeLModelForImageTextToText
        if args.model_class == "image-text-to-text"
        else ed.AutoEasyDeLModelForCausalLM
    )
    partition_axis = ed.PartitionAxis()
    partition_axis.hidden_state_axis = None

    quant_kwargs = {}
    config_kwargs = None
    if args.quantization != "none":
        from easydel.layers.quantization import QuantizationConfig

        quant_kwargs = {
            "quantization_config": QuantizationConfig(
                dtype=args.quantization,
                group_size=args.quant_group_size,
                bits=args.quant_bits,
                pattern=args.quant_pattern,
            ),
            "apply_quantization": True,
        }
        config_kwargs = {}
        if args.qmm_platform is not None:
            config_kwargs["qmm_platform_override"] = args.qmm_platform
        if args.qmm_tpu_path is not None:
            config_kwargs["qmm_tpu_path_override"] = args.qmm_tpu_path
        config_kwargs = config_kwargs or None
        print(
            f"quantization: {args.quantization} group={args.quant_group_size} bits={args.quant_bits} "
            f"qmm_platform={args.qmm_platform} qmm_tpu_path={args.qmm_tpu_path}",
            flush=True,
        )

    t0 = time.perf_counter()
    with set_inference_mode():
        model = auto_cls.from_pretrained(
            pretrained_model_name_or_path=args.model,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            sharding_axis_dims=axis_dims,
            sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
            partition_axis=partition_axis,
            auto_shard_model=True,
            config_kwargs=config_kwargs,
            **quant_kwargs,
        )
    has_mtp = bool(getattr(model, "has_mtp", lambda: False)())
    print(f"model loaded in {time.perf_counter() - t0:.1f}s  has_mtp={has_mtp}  type={type(model).__name__}", flush=True)
    if "on" in modes and not has_mtp:
        raise SystemExit("model has no MTP head; spec-ON needs one (or drop --modes to 'off')")

    os.environ["EASYDEL_MTP_PERSIST_KV"] = "0" if args.no_persist else "1"
    results = {"off": {}, "on": {}}

    def _brk(r):
        return (
            f"fwd={r.get('decode_fwd_ms'):.1f} draft={r.get('decode_spec_draft_ms'):.1f} "
            f"post={r.get('decode_post_ms'):.1f} misc={r.get('decode_misc_ms'):.1f} "
            f"prep={r.get('decode_prep_ms'):.1f}"
        )

    tokenizer = None

    def _decode_sample(ids):
        nonlocal tokenizer
        try:
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
            return tokenizer.decode(ids)
        except Exception as exc:
            return f"<decode failed: {exc!r}>"

    if args.smoke:
        print(f"\n=== SMOKE (quantization={args.quantization}, n={args.smoke_n}) ===", flush=True)
        from easydel.inference.esurge.benchmarking import make_real_prompts

        runner = build_runner(model, drafter=None, args=args)
        prompts = make_real_prompts(
            count=args.smoke_n,
            prompt_len=args.prompt_len,
            tokenizer_name=args.tokenizer or args.model,
            text_file=args.prompt_text_file,
            doc_paths=(REPO_ROOT / "WORKSPACE.md", REPO_ROOT / "README.md", REPO_ROOT / "CLAUDE.md"),
        )
        res = run_batch(runner, prompts, args.output_len, args.max_num_batched_tokens)
        for i, req in enumerate(res["requests"]):
            out_ids = list(req.output_token_ids)
            text = _decode_sample(out_ids)
            print(f"\n--- smoke[{i}] prompt tail: {_decode_sample(list(prompts[i])[-48:])!r}", flush=True)
            print(f"--- smoke[{i}] output ({len(out_ids)} toks): {text!r}", flush=True)
        free_runner(runner)
        print("\nSMOKE_DONE", flush=True)
        return 0

    if "off" in modes:
        print("\n=== spec-OFF runner ===", flush=True)
        runner = build_runner(model, drafter=None, args=args)
        for n in off_conc:
            r = run_conc(runner, n, args)
            if args.prompt_source == "real" and r.get("sample_output_ids"):
                r["sample_output_text"] = _decode_sample(r["sample_output_ids"])
            results["off"][str(n)] = r
            print(
                f"  [off] n={n:2d} agg={r['tok_s']:8.1f}t/s dec={r.get('decode_tok_s'):8.1f}t/s "
                f"dec_fwd={r.get('decode_fwd_ms'):.1f}ms dec_tot={r.get('decode_total_ms'):.1f}ms",
                flush=True,
            )
            if r.get("sample_output_text"):
                print(f"  [off] n={n:2d} sample: {r['sample_output_text'][:200]!r}", flush=True)
        free_runner(runner)

    if "on" in modes:
        print(f"\n=== spec-ON runner (MTP K={args.num_spec}, persist={not args.no_persist}) ===", flush=True)
        drafter = model.drafter(method="mtp", num_draft_tokens=args.num_spec)
        runner = build_runner(model, drafter=drafter, args=args)
        for n in on_conc:
            r = run_conc(runner, n, args)
            results["on"][str(n)] = r
            print(
                f"  [on ] n={n:2d} agg={r['tok_s']:8.1f}t/s dec={r.get('decode_tok_s'):8.1f}t/s "
                f"tok/step={r.get('tok_per_step'):.2f} dec_tot={r.get('decode_total_ms'):.1f}ms | {_brk(r)}",
                flush=True,
            )
            try:
                if args.prompt_source == "real" and r.get("sample_output_ids"):
                    from transformers import AutoTokenizer

                    _tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
                    print(f"  [on ] n={n:2d} sample: {_tok.decode(r['sample_output_ids'])!r}", flush=True)
            except Exception as _exc:
                print(f"  sample decode failed: {_exc!r}", flush=True)
        free_runner(runner)

    print("\nRESULTS_TABLE_BEGIN", flush=True)
    hdr = (
        f"{'n':>4} | {'OFF agg':>8} | {'OFF dec':>8} | {'ON agg':>8} | {'ON dec':>8} | "
        f"{'ratio(dec)':>10} | {'tok/step':>8} | {'verify ms':>9} | {'draft ms':>8}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for n in sorted(set(off_conc) | set(on_conc)):
        off = results["off"].get(str(n))
        on = results["on"].get(str(n))

        def g(d, k):
            return d.get(k) if d else None

        off_d, on_d = g(off, "decode_tok_s"), g(on, "decode_tok_s")
        ratio = (on_d / off_d) if (off_d and on_d) else None

        def f(x, w, p=1):
            return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"

        print(
            f"{n:>4} | {f(g(off, 'tok_s'), 8)} | {f(off_d, 8)} | {f(g(on, 'tok_s'), 8)} | {f(on_d, 8)} | "
            f"{f(ratio, 10, 3)} | {f(g(on, 'tok_per_step'), 8, 2)} | {f(g(on, 'decode_fwd_ms'), 9, 1)} | "
            f"{f(g(on, 'decode_spec_draft_ms'), 8, 1)}",
            flush=True,
        )
    print("RESULTS_TABLE_END", flush=True)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": results}, indent=2, sort_keys=True, default=str))
    print(f"json_out: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
