#!/usr/bin/env python3
"""Smoke-test eSurge async scheduling with overlap execution on a real model.

Loads a small instruction model (default
``HuggingFaceTB/SmolLM2-1.7B-Instruct``), spins up an :class:`eSurge` engine
with ``async_scheduling=True`` and ``overlap_execution=True`` plus prefix
caching, and runs a single chat completion against it. Useful as a quick
end-to-end check that nothing in the runtime regressed for the most common
async + overlap configuration.

The script asserts that something actually came back from the engine and
prints prompt, generated text, token ids, finish reason, and wall-clock time.

Side effects:
    - Downloads the model weights via Hugging Face on first run.
    - Initializes and terminates an :class:`eSurge` engine.

Usage:
    python scripts/esurge_async_overlap_smoke.py \\
        --model HuggingFaceTB/SmolLM2-1.7B-Instruct --prompt "hi" \\
        --max-tokens 16
"""

from __future__ import annotations

import argparse
import time

from easydel.inference.esurge.config import eSurgeCacheRuntimeConfig, eSurgeRuntimeConfig
from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed flags (``model``, ``prompt``,
            ``max_model_len``, ``max_num_seqs``, ``max_tokens``,
            ``temperature``, ``top_p``, ``hbm_utilization``).
    """
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
    """Run the async-overlap smoke test end-to-end.

    Builds the engine, calls :meth:`eSurge.chat` with the supplied prompt, and
    prints the resulting text / token ids / finish reason / wall-clock time.
    Guarantees ``engine.terminate()`` runs even when generation raises.

    Returns:
        None.

    Raises:
        RuntimeError: If the engine returns no text and no token ids.
    """
    args = parse_args()
    engine = eSurge(
        model=args.model,
        runtime=eSurgeRuntimeConfig.from_dict(
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            async_scheduling=True,
            overlap_execution=True,
        ),
        cache=eSurgeCacheRuntimeConfig.from_dict(
            hbm_utilization=args.hbm_utilization,
            enable_prefix_caching=True,
        ),
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
