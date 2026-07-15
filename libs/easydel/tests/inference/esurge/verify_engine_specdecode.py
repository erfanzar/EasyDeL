"""Verify #2 — full eSurge ENGINE + real chat requests, MTP drafter on vs off.

Loads Qwen3.6-27B (spx, VLM class) once, builds the real ``eSurge`` engine
(NOT the raw runner+scheduler) with the MTP drafter enabled and, separately,
with no drafter, sends real chat requests through each, and reports decoded
text (coherence) + tokens/sec (speedup).
"""

from __future__ import annotations

import gc
import os
import time

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ["JAX_PLATFORMS"] = "tpu"
os.environ["JAX_PLATFORM_NAME"] = "tpu"

CKPT = os.environ.get("EASYDEL_BENCH_CKPT", "/dev/shm/easydel_ckpts/qwen36_27b_spx")
TOKENIZER_ID = os.environ.get("EASYDEL_BENCH_TOKENIZER", "Qwen/qwen3.6-27b")
MAX_NEW = int(os.environ.get("VERIFY_MAX_NEW", "96"))
MAX_MODEL_LEN = int(os.environ.get("VERIFY_MAX_MODEL_LEN", "256"))
MAX_BATCHED = int(os.environ.get("VERIFY_MAX_BATCHED", "192"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

if jax.default_backend() != "tpu":
    raise SystemExit(f"Refusing to run on {jax.default_backend()!r}; TPU-only.")

import easydel as ed  # noqa: E402
from easydel.inference import eSurge  # noqa: E402
from easydel.inference.esurge.config import (  # noqa: E402
    eSurgeCacheRuntimeConfig,
    eSurgeParsingConfig,
    eSurgeRuntimeConfig,
)
from easydel.inference.sampling_params import SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

PROMPTS = [
    "Explain speculative decoding in two sentences.",
    "Write a haiku about TPUs.",
    "What is the capital of France?",
    "List three prime numbers greater than 10.",
]


def load_model():
    partition_axis = ed.PartitionAxis()
    partition_axis.hidden_state_axis = None
    model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=CKPT,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        sharding_axis_dims=(1, 1, 1, 1, -1, 1),
        sharding_axis_names=("pp", "dp", "fsdp", "ep", "tp", "sp"),
        partition_axis=partition_axis,
        auto_shard_model=True,
    )
    print(f"  model={type(model).__name__} has_mtp={bool(getattr(model, 'has_mtp', lambda: False)())}")
    return model


def build_engine(model, tokenizer, drafter):
    return eSurge(
        model=model,
        processor=tokenizer,
        runtime=eSurgeRuntimeConfig.from_dict(
            max_model_len=MAX_MODEL_LEN,
            min_input_pad=1,
            max_num_seqs=1,
            max_num_seq_buckets=[1],
            max_num_batched_tokens=MAX_BATCHED,
            async_scheduling=False,
            overlap_execution=False,
            runner_verbose=False,
        ),
        cache=eSurgeCacheRuntimeConfig.from_dict(
            hbm_utilization=0.25,
            page_size=16,
            max_cache_tokens=4096,
            enable_prefix_caching=False,
        ),
        parsing=eSurgeParsingConfig.from_dict(silent_mode=True),
        drafter=drafter,
    )


def format_prompt(tokenizer, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    for kwargs in ({"enable_thinking": False}, {}):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
        except Exception:
            continue
    return text


def run_engine(engine, tokenizer, label: str) -> dict:
    engine.initiate()
    sp = SamplingParams(max_tokens=MAX_NEW, temperature=0.0, top_p=1.0)
    results = []
    # warmup (compile) on the first prompt, excluded from timing
    _ = engine.generate(format_prompt(tokenizer, PROMPTS[0]), sp, use_tqdm=False)
    total_tokens = 0
    total_time = 0.0
    print(f"\n{'=' * 78}\n[{label}] decoded responses\n{'=' * 78}")
    for i, prompt in enumerate(PROMPTS):
        t0 = time.time()
        outs = engine.generate(format_prompt(tokenizer, prompt), sp, use_tqdm=False)
        dt = time.time() - t0
        out = outs[0]
        text = out.get_text()
        ntok = int(getattr(out, "num_generated_tokens", 0) or 0)
        gen_time = float(getattr(out, "time_spent_generating", 0.0) or 0.0)
        total_tokens += ntok
        total_time += gen_time if gen_time > 0 else dt
        tps = ntok / gen_time if gen_time > 0 else (ntok / dt if dt > 0 else 0.0)
        results.append({"prompt": prompt, "text": text, "ntok": ntok, "tps": tps, "engine_tps": out.tokens_per_second})
        print(f"\n--- prompt {i + 1}: {prompt}")
        print(f"    [{ntok} tok, wall={dt:.2f}s gen={gen_time:.2f}s, {tps:.2f} tok/s, engine_tps={out.tokens_per_second:.2f}]")
        print(f"    {text!r}")
    # Primary metric: mean of the engine's own per-request tokens/sec. This
    # excludes first-use XLA compilation stalls (a warm prompt can trigger a
    # multi-minute bucket compile that would otherwise dominate a wallclock
    # aggregate and make the comparison meaningless).
    engine_tps_vals = [r["engine_tps"] for r in results if r["engine_tps"] and r["engine_tps"] > 0]
    mean_engine_tps = (sum(engine_tps_vals) / len(engine_tps_vals)) if engine_tps_vals else 0.0
    agg = total_tokens / total_time if total_time > 0 else 0.0
    print(
        f"\n[{label}] mean engine tok/s (compile-excluded): {mean_engine_tps:.2f}  "
        f"| wallclock aggregate (compile-polluted): {agg:.2f}"
    )
    engine.terminate()
    return {
        "label": label,
        "mean_engine_tps": mean_engine_tps,
        "agg_tps": agg,
        "total_tokens": total_tokens,
        "total_time": total_time,
        "results": results,
    }


def main():
    print("=" * 78)
    print("Verify #2 — full eSurge engine + real requests (Qwen3.6-27B, tp=4)")
    print("=" * 78)
    print(f"devices: {jax.devices()}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    model = load_model()

    print("\n>>> building drafter-ON engine (MTP)")
    drafter = model.drafter(method="mtp", num_draft_tokens=1)
    spec_engine = build_engine(model, tokenizer, drafter)
    spec = run_engine(spec_engine, tokenizer, "engine+MTP-drafter")
    del spec_engine
    gc.collect()

    print("\n>>> building drafter-OFF engine (baseline)")
    base_engine = build_engine(model, tokenizer, None)
    base = run_engine(base_engine, tokenizer, "engine-baseline")
    del base_engine
    gc.collect()

    print("\n" + "=" * 78)
    print("ENGINE THROUGHPUT COMPARISON")
    print("=" * 78)
    b = base["mean_engine_tps"]
    s = spec["mean_engine_tps"]
    print(f"  baseline (no drafter): {b:.2f} tok/s  (mean engine tok/s, compile-excluded)")
    print(f"  spec-decode (MTP)    : {s:.2f} tok/s  (mean engine tok/s, compile-excluded)")
    if b > 0:
        print(f"  ENGINE SPEEDUP       : {s / b:.2f}x")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
