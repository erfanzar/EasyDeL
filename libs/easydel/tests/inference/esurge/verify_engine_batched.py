"""Batched eSurge ENGINE throughput: MTP drafter ON vs OFF at batch N.

Batched decode is memory-bandwidth bound on weight loads; speculative decoding
amortizes those loads across a draft window, so this is the regime where spec
decode is expected to win. Loads Qwen3.6-27B once, builds the real ``eSurge``
engine with ``max_num_seqs=N``, sends N concurrent requests through the
drafter-ON and drafter-OFF engines, and compares aggregate tokens/sec
(total generated tokens / batch wallclock, compile excluded via a warmup batch).
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
MAX_MODEL_LEN = int(os.environ.get("VERIFY_MAX_MODEL_LEN", "512"))
MAX_BATCHED = int(os.environ.get("VERIFY_MAX_BATCHED", "512"))
BATCH = int(os.environ.get("VERIFY_BATCH", "4"))
NUM_DRAFT = int(os.environ.get("VERIFY_NUM_DRAFT", "1"))

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

_BASE_PROMPTS = [
    "Explain speculative decoding for transformer inference in a few sentences.",
    "Describe how paged attention manages the key-value cache during serving.",
    "What are the main trade-offs between tensor parallelism and data parallelism?",
    "Summarize why memory bandwidth limits autoregressive decoding throughput.",
    "Explain what a mixture-of-experts layer does and why it saves compute.",
    "How does continuous batching improve inference server utilization?",
    "What is the role of a draft model in speculative decoding?",
    "Describe the difference between prefill and decode phases in LLM serving.",
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
            max_num_seqs=BATCH,
            max_num_seq_buckets=[BATCH],
            max_num_batched_tokens=MAX_BATCHED,
            async_scheduling=False,
            overlap_execution=False,
            runner_verbose=False,
        ),
        cache=eSurgeCacheRuntimeConfig.from_dict(
            hbm_utilization=0.30,
            page_size=16,
            max_cache_tokens=BATCH * MAX_MODEL_LEN,
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
    # ignore_eos keeps every request generating exactly MAX_NEW tokens so the
    # batch stays full at BATCH rows for the whole run (stable compile bucket,
    # clean aggregate throughput).
    sp = SamplingParams(max_tokens=MAX_NEW, temperature=0.0, top_p=1.0, ignore_eos=True)
    prompts = [format_prompt(tokenizer, _BASE_PROMPTS[i % len(_BASE_PROMPTS)]) for i in range(BATCH)]

    # Warmup batch (compile) — excluded from timing.
    _ = engine.generate(prompts, sp, use_tqdm=False)

    t0 = time.time()
    outs = engine.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0

    total_tokens = sum(int(getattr(o, "num_generated_tokens", 0) or 0) for o in outs)
    engine_tps_vals = [o.tokens_per_second for o in outs if o.tokens_per_second and o.tokens_per_second > 0]
    mean_engine_tps = (sum(engine_tps_vals) / len(engine_tps_vals)) if engine_tps_vals else 0.0
    agg = total_tokens / wall if wall > 0 else 0.0

    print(f"\n{'=' * 78}\n[{label}] batch={BATCH} draft={NUM_DRAFT}\n{'=' * 78}")
    print(f"  total_tokens={total_tokens} batch_wall={wall:.2f}s")
    print(f"  aggregate throughput = {agg:.2f} tok/s")
    print(f"  sum(engine per-req tok/s) = {sum(engine_tps_vals):.2f}  mean per-req = {mean_engine_tps:.2f}")
    print(f"  sample response[0]: {outs[0].get_text()[:240]!r}")
    engine.terminate()
    return {
        "label": label,
        "agg_tps": agg,
        "sum_engine_tps": sum(engine_tps_vals),
        "total_tokens": total_tokens,
        "wall": wall,
    }


def main():
    print("=" * 78)
    print(f"Batched eSurge engine — Qwen3.6-27B tp=4, batch={BATCH}, draft={NUM_DRAFT}, gen={MAX_NEW}")
    print("=" * 78)
    print(f"devices: {jax.devices()}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
    model = load_model()

    print("\n>>> drafter-ON engine (MTP)")
    drafter = model.drafter(method="mtp", num_draft_tokens=NUM_DRAFT)
    spec = run_engine(build_engine(model, tokenizer, drafter), tokenizer, "engine+MTP-drafter")
    gc.collect()

    print("\n>>> drafter-OFF engine (baseline)")
    base = run_engine(build_engine(model, tokenizer, None), tokenizer, "engine-baseline")
    gc.collect()

    print("\n" + "=" * 78)
    print(f"BATCHED ENGINE THROUGHPUT COMPARISON (batch={BATCH}, draft={NUM_DRAFT})")
    print("=" * 78)
    print(f"  baseline aggregate : {base['agg_tps']:.2f} tok/s")
    print(f"  spec-decode agg    : {spec['agg_tps']:.2f} tok/s")
    if base["agg_tps"] > 0:
        print(f"  SPEEDUP (aggregate): {spec['agg_tps'] / base['agg_tps']:.2f}x")
    if base["sum_engine_tps"] > 0:
        print(f"  SPEEDUP (sum engine tps): {spec['sum_engine_tps'] / base['sum_engine_tps']:.2f}x")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
