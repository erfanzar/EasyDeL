"""Print what the SAO rollout policy actually generates.

The RL run reports ``completion_length`` pinned exactly at the cap with
``ExecutionReward=0`` at both 1024 and 2048 budgets, which no amount of extra
budget explains on its own. The metrics cannot distinguish "coherent but
verbose reasoning" from "degenerate output that never terminates", so this
decodes real rollouts and shows the text.

Uses env.py's own dataset preparation so the prompt is byte-identical to the
one training feeds the policy (same chat template, same enable_thinking
prefill), and generates through eSurge exactly as the trainer does.
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--num-prompts", type=int, default=3)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--json-out", default="/dev/shm/rollout_text_diagnosis.json")
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    args = ap.parse_args()

    sys.path.insert(0, "/home/erfan/EasyDeL")
    import env as sao

    import easydel as ed
    import jax
    from jax import numpy as jnp

    from easydel.inference.sampling_params import SamplingParams

    # Rebuild the exact training prompts via env.py's own pipeline.
    prep_args = sao.build_parser().parse_args(
        [
            "validate-data",
            "--local-files-only",
            "--dataset-limit",
            "64",
            "--max-prompt-length",
            str(args.prompt_len),
            "--max-tests-per-completion",
            "2",
            "--runner-command",
            "/usr/bin/false",
        ]
    )
    tokenizer = sao.load_tokenizer(prep_args)
    train_dataset, _hidden, _summary = sao.prepare_dataset(prep_args, tokenizer)

    prompts = []
    for i in range(min(args.num_prompts, len(train_dataset))):
        ids = list(train_dataset[i]["input_ids"])
        prompts.append(tokenizer.decode(ids, skip_special_tokens=False))

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        auto_shard_model=True,
    )

    engine = model.get_esurge(
        tokenizer=tokenizer,
        max_model_len=args.prompt_len + args.max_new_tokens,
        max_num_seqs=len(prompts),
        max_num_batched_tokens=args.prompt_len * len(prompts),
        max_cache_tokens=(args.prompt_len + args.max_new_tokens) * len(prompts),
        hbm_utilization=None,
        page_size=32,
        enable_prefix_caching=True,
        data_parallelism_axis="dp",
        async_scheduling=False,
        overlap_execution=False,
        runner_verbose=False,
    )

    # Same sampling as SAOConfig.
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                              presence_penalty=args.presence_penalty)
    outputs = engine.generate(prompts, sampling, use_tqdm=False)

    records = []
    for i, out in enumerate(outputs):
        # accumulated_text is POST reasoning/tool separation: an unclosed
        # <think> block leaves it empty even when tokens were produced. The raw
        # field is what actually distinguishes coherent reasoning from garbage.
        raw = getattr(out, "raw_accumulated_text", "") or ""
        text = out.accumulated_text or ""
        completion = out.outputs[0] if out.outputs else None
        detok = tokenizer.decode(getattr(completion, "token_ids", None) or [], skip_special_tokens=False)
        rec = {
            "index": i,
            "raw_text_head": raw[:1500],
            "raw_text_tail": raw[-500:],
            "raw_len_chars": len(raw),
            "detokenized_head": detok[:1500],
            "detokenized_len_chars": len(detok),
            "reasoning_content_head": (getattr(completion, "reasoning_content", None) or "")[:800],
            "num_generated_tokens": int(out.num_generated_tokens or 0),
            "finish_reason": getattr(completion, "finish_reason", None),
            "hit_cap": int(out.num_generated_tokens or 0) >= args.max_new_tokens,
            "has_open_think": "<think>" in text and "</think>" not in text,
            "has_close_think": "</think>" in text,
            "has_python_fence": "```python" in text,
            "has_closing_fence": text.count("```") >= 2,
            "text_len_chars": len(text),
            "text_head": text[:1200],
            "text_tail": text[-600:],
        }
        records.append(rec)
        print("=" * 100)
        print(f"ROLLOUT {i}: tokens={rec['num_generated_tokens']} finish={rec['finish_reason']} "
              f"hit_cap={rec['hit_cap']} open_think={rec['has_open_think']} "
              f"close_think={rec['has_close_think']} py_fence={rec['has_python_fence']}")
        print("-" * 100)
        print("RAW HEAD:", rec["raw_text_head"][:900])
        print("DETOK HEAD:", rec["detokenized_head"][:900])
        print("-" * 40, "TAIL", "-" * 40)
        print(rec["text_tail"])

    with open(args.json_out, "w") as fh:
        json.dump({"model": args.model_id, "max_new_tokens": args.max_new_tokens, "rollouts": records}, fh, indent=2)
    print("\nwrote", args.json_out)
    engine.terminate()


if __name__ == "__main__":
    main()
