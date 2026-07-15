"""Sweep MTP num_draft_tokens K for eSurge spec-decode; reuse model+baseline.

Loads the model once, compiles the baseline runner once, then for each K in
SWEEP_KS rebuilds only the MTP spec runner and measures greedy + sampling
throughput, acceptance, accepted-tokens-per-step, per-step time, and the
verify(forward)-vs-MTP-draft cost breakdown from the runner perf history.
"""

import os
import statistics
import sys

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ.setdefault("EASYDEL_BENCH_CKPT", "/dev/shm/easydel_ckpts/qwen36_27b_spx")
os.environ.setdefault("EASYDEL_BENCH_TOKENIZER", "Qwen/qwen3.6-27b")
os.environ.setdefault("EASYDEL_BENCH_MODEL_CLASS", "image_text_to_text")
os.environ.setdefault("EASYDEL_BENCH_MAX_NEW", "160")
os.environ.setdefault("EASYDEL_BENCH_MAX_MODEL_LEN", "512")
os.environ.setdefault("EASYDEL_BENCH_MAX_BATCHED", "512")
os.environ.setdefault("EASYDEL_BENCH_WARMUPS", "1")
os.environ.setdefault("EASYDEL_BENCH_TRIALS", "2")
os.environ.setdefault("EASYDEL_BENCH_PRINT_TRIALS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gc

import bench_specdecode as B
from transformers import AutoTokenizer

KS = [int(x) for x in os.environ.get("SWEEP_KS", "1,2,4,8").split(",")]
TRIALS = B.BENCH_TRIALS


def step_break(runner, ntrials, verify_steps, k):
    """Median per-phase ms over the timed-trial decode (verify) steps."""
    hist = list(runner._perf_phase_history)
    # each trial = 1 prefill record + (verify_steps/ntrials) verify records
    tail = hist[-(int(verify_steps) + ntrials + 2):] if verify_steps else hist
    dec = [r for r in tail if 1 <= int(r.get("total_tokens", 0)) <= k + 2]

    def med(field):
        vals = [float(r.get(field, 0) or 0.0) * 1e3 for r in dec]
        return statistics.median(vals) if vals else 0.0

    return {
        "n": len(dec),
        "total": med("total_time"),
        "fwd": med("forward_time"),
        "draft": med("spec_draft_time"),
        "project": med("spec_project_time"),
        "post": med("post_time"),
        "prep": med("prep_time"),
        "suffix": med("spec_suffix_time"),
        "commit": med("spec_commit_time"),
    }


def main():
    print("=" * 90)
    print("MTP num_draft_tokens SWEEP — Qwen3.6-27B tp=4 bf16 fast-path (default)")
    print(f"KS={KS} max_new={B.MAX_NEW} max_model_len={B.MAX_MODEL_LEN} trials={TRIALS}")
    print("=" * 90)
    model, ckpt = B.load_model()
    B.PROMPT = B.resolve_prompt_tokens(ckpt)
    print(f"prompt_tokens: {len(B.PROMPT)}  devices: {len(B.__dict__.get('jax').devices())}")
    tok = AutoTokenizer.from_pretrained(os.environ["EASYDEL_BENCH_TOKENIZER"], trust_remote_code=True)

    print("\n[baseline] build+compile")
    base = B.build_runner(model, drafter=None)
    bg_tps, _, bg_toks = B.bench(base, "baseline-greedy", temperature=0.0, top_p=1.0)
    bg_break = step_break(base, TRIALS, len(bg_toks) * TRIALS, 0)
    bs_tps, _, bs_toks = B.bench(base, "baseline-sampling", temperature=0.8, top_p=0.95)
    print(f"\nSAMPLE[baseline-greedy]: {tok.decode(bg_toks)[:220]!r}")

    rows = []
    for k in KS:
        print(f"\n{'#' * 30} K={k} {'#' * 30}")
        drafter = model.drafter(method="mtp", num_draft_tokens=k)
        sr = B.build_runner(model, drafter=drafter)

        sg_tps, _, sg_toks = B.bench(sr, "spec-greedy", temperature=0.0, top_p=1.0)
        g_gen = sr.spec_decode_num_drafts_generated
        g_acc = sr.spec_decode_num_drafts_accepted
        g_steps = sr.spec_decode_num_verify_steps
        g_break = step_break(sr, TRIALS, g_steps, k)
        g_med_wall = (len(sg_toks) / sg_tps) if sg_tps else 0.0
        g_steps_pt = (g_steps / TRIALS) if g_steps else 0.0
        g_sample = tok.decode(sg_toks)

        ss_tps, _, ss_toks = B.bench(sr, "spec-sampling", temperature=0.8, top_p=0.95)
        s_gen = sr.spec_decode_num_drafts_generated
        s_acc = sr.spec_decode_num_drafts_accepted
        s_steps = sr.spec_decode_num_verify_steps

        rows.append(
            dict(
                k=k,
                g_tps=sg_tps,
                g_spd=(sg_tps / bg_tps) if bg_tps else 0.0,
                g_acc_rate=(g_acc / g_gen) if g_gen else 0.0,
                g_acc_ps=(g_acc / g_steps) if g_steps else 0.0,
                g_tok_ps=(len(sg_toks) / g_steps_pt) if g_steps_pt else 0.0,
                g_step_ms=(g_med_wall / g_steps_pt * 1e3) if g_steps_pt else 0.0,
                g_break=g_break,
                g_sample=g_sample,
                g_match=(sg_toks == bg_toks),
                s_tps=ss_tps,
                s_spd=(ss_tps / bs_tps) if bs_tps else 0.0,
                s_acc_rate=(s_acc / s_gen) if s_gen else 0.0,
                s_acc_ps=(s_acc / s_steps) if s_steps else 0.0,
            )
        )
        del sr, drafter
        gc.collect()

    print("\n\n" + "=" * 118)
    print("SWEEP RESULTS — Qwen3.6-27B tp=4 bf16 fast-path")
    print("=" * 118)
    print(f"baseline greedy  : {bg_tps:7.2f} tok/s   per-step total={bg_break['total']:.2f}ms fwd={bg_break['fwd']:.2f}ms")
    print(f"baseline sampling: {bs_tps:7.2f} tok/s")
    print("-" * 118)
    hdr = (
        f"{'K':>3} | {'spec_g':>8} {'speedup':>8} {'accept':>7} {'acc/stp':>8} {'tok/stp':>8} "
        f"{'step_ms':>8} {'verify':>8} {'draft':>8} {'proj':>7} {'post':>7} | {'spec_s':>8} {'speedup':>8} {'accept':>7}"
    )
    print(hdr)
    print("-" * 118)
    for r in rows:
        b = r["g_break"]
        print(
            f"{r['k']:>3} | {r['g_tps']:>8.2f} {r['g_spd']:>7.2f}x {r['g_acc_rate']:>7.1%} {r['g_acc_ps']:>8.2f} "
            f"{r['g_tok_ps']:>8.2f} {r['g_step_ms']:>8.2f} {b['fwd']:>8.2f} {b['draft']:>8.2f} {b['project']:>7.2f} "
            f"{b['post']:>7.2f} | {r['s_tps']:>8.2f} {r['s_spd']:>7.2f}x {r['s_acc_rate']:>7.1%}"
        )
    print("=" * 118)
    print("cols: verify=forward_time(target verify fwd)  draft=spec_draft_time(recursive MTP draft)  proj=spec_project_time")
    for r in rows:
        print(f"\n--- K={r['k']} greedy coherence (match_baseline={r['g_match']}) ---\n{r['g_sample'][:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
