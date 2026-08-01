"""Check that eSurge terminates on EOS/stop for this model.

Every SAO rollout returns ``finish_reason='length'`` at 1024, 2048, 8192 and
16384-token budgets. Two very different causes produce that: the model
genuinely never finishes reasoning, or the serving path never honours the stop
tokens. Trivial prompts separate them -- a model that cannot answer "What is
2+2?" inside 512 tokens is not deliberating, it is failing to stop.

Runs the same eSurge path the trainer uses, with thinking disabled so the
reply is a direct answer.
"""

from __future__ import annotations

import json

import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoTokenizer

from easydel.inference.sampling_params import SamplingParams

MODEL_ID = "Qwen/Qwen3.5-4B"
QUESTIONS = ["What is 2+2?", "Name three primary colors.", "Say the word 'hello' and nothing else."]


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in QUESTIONS
    ]

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        auto_shard_model=True,
    )
    engine = model.get_esurge(
        tokenizer=tokenizer,
        max_model_len=1536,
        max_num_seqs=len(prompts),
        max_num_batched_tokens=1024,
        max_cache_tokens=1536 * len(prompts),
        hbm_utilization=None,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        async_scheduling=False,
        overlap_execution=False,
        runner_verbose=False,
    )

    outputs = engine.generate(prompts, SamplingParams(max_tokens=512, temperature=0.6, top_p=0.95, top_k=64), use_tqdm=False)

    records = []
    for question, out in zip(QUESTIONS, outputs, strict=False):
        completion = out.outputs[0] if out.outputs else None
        rec = {
            "question": question,
            "tokens": int(out.num_generated_tokens or 0),
            "finish_reason": getattr(completion, "finish_reason", None),
            "stopped_early": int(out.num_generated_tokens or 0) < 512,
            "text": (out.accumulated_text or "")[:300],
        }
        records.append(rec)
        print(f"Q: {question!r}\n  tokens={rec['tokens']} finish={rec['finish_reason']} "
              f"stopped_early={rec['stopped_early']}\n  text={rec['text']!r}\n")

    verdict = "eSurge honours stop/EOS" if all(r["stopped_early"] for r in records) else "STOP TOKENS NOT HONOURED"
    print("VERDICT:", verdict)
    with open("/dev/shm/stop_token_probe.json", "w") as fh:
        json.dump({"verdict": verdict, "records": records}, fh, indent=2)
    engine.terminate()


if __name__ == "__main__":
    main()
