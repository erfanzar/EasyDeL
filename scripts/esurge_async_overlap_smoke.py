#!/usr/bin/env python3
"""Smoke-test eSurge async scheduling with overlap execution on a real model."""

from __future__ import annotations

import argparse
import time

from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="Model name or path to load.",
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in one short sentence.",
        help="User message to generate from.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--hbm-utilization", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = eSurge(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        hbm_utilization=args.hbm_utilization,
        async_scheduling=True,
        overlap_execution=True,
        enable_prefix_caching=True,
    )
    engine.initiate()

    started = time.monotonic()
    try:
        output = engine.chat(
            [{"role": "user", "content": args.prompt}],
            sampling_params=SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            ),
        )
    finally:
        engine.terminate()

    elapsed = time.monotonic() - started
    text = output.outputs[0].text
    token_ids = output.outputs[0].token_ids
    finish_reason = output.outputs[0].finish_reason
    print(f"prompt: {args.prompt!r}")
    print(f"generated: {text!r}")
    print(f"token_ids: {token_ids!r}")
    print(f"finish_reason: {finish_reason!r}")
    print(f"elapsed_s: {elapsed:.2f}")

    if not text.strip() and not token_ids:
        raise RuntimeError("Generation completed but returned no text and no token IDs.")


if __name__ == "__main__":
    main()
