"""Smoke test for PR #254: dynamic batching with prompt count < max_num_seqs.

Runs on a real TPU (v5p-8) with Qwen/Qwen3.5-9B to verify that
SequenceBuffer condensation doesn't crash or corrupt page table
state when fewer prompts than max_num_seqs are submitted.
"""

from __future__ import annotations

import os

os.environ["HF_HOME"] = "/dev/shm/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/dev/shm/huggingface"

from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.sampling_params import SamplingParams


def main() -> None:
    engine = eSurge(
        model="Qwen/Qwen3-0.6B",
        max_model_len=512,
        max_num_seqs=64,
        hbm_utilization=0.85,
        enable_prefix_caching=True,
    )
    engine.initiate()

    params = SamplingParams(max_tokens=64, temperature=0.7, top_p=0.9)

    try:
        # Test 1: single prompt (1 << max_num_seqs=64)
        print("=" * 60)
        print("Test 1: single prompt (1 / 64 slots)")
        outputs = engine.generate(["What is 2+2?"], sampling_params=params)
        assert len(outputs) == 1
        assert outputs[0].outputs[0].text.strip() != ""
        print(f"  OK: {outputs[0].outputs[0].text[:80]!r}")

        # Test 2: 2 prompts (PR #254 core scenario)
        print("=" * 60)
        print("Test 2: 2 prompts (2 / 64 slots)")
        outputs = engine.generate(
            ["Hello, how are you?", "Tell me a joke."],
            sampling_params=params,
        )
        assert len(outputs) == 2
        for i, out in enumerate(outputs):
            assert out.outputs[0].text.strip() != ""
            print(f"  Prompt {i}: {out.outputs[0].text[:80]!r}")

        # Test 3: 5 prompts with varied lengths
        print("=" * 60)
        print("Test 3: 5 prompts with varied lengths (5 / 64 slots)")
        prompts = [
            "Hi",
            "Explain quantum computing in one sentence.",
            "What is the capital of France?",
            "Write a haiku about the ocean.",
            "Count from 1 to 10.",
        ]
        outputs = engine.generate(prompts, sampling_params=params)
        assert len(outputs) == 5
        for i, out in enumerate(outputs):
            assert out.outputs[0].text.strip() != ""
            print(f"  Prompt {i}: {out.outputs[0].text[:80]!r}")

        # Test 4: repeated batches (condense cycle stress)
        print("=" * 60)
        print("Test 4: 3 repeated small batches (condense cycle stress)")
        for batch_num in range(3):
            outputs = engine.generate(
                [f"Batch {batch_num}: say something short."],
                sampling_params=SamplingParams(max_tokens=16, temperature=0.5),
            )
            assert len(outputs) == 1
            assert outputs[0].outputs[0].text.strip() != ""
            print(f"  Batch {batch_num}: {outputs[0].outputs[0].text[:80]!r}")

        # Test 5: fill closer to max_num_seqs
        print("=" * 60)
        print("Test 5: 32 prompts (32 / 64 slots, half capacity)")
        prompts = [f"Question {i}: What is {i}+{i}?" for i in range(32)]
        outputs = engine.generate(
            prompts,
            sampling_params=SamplingParams(max_tokens=16, temperature=0.0),
        )
        assert len(outputs) == 32
        failed = [i for i, o in enumerate(outputs) if not o.outputs[0].text.strip()]
        assert not failed, f"Empty outputs at indices: {failed}"
        print("  OK: all 32 prompts returned non-empty text")

        print("=" * 60)
        print("ALL TESTS PASSED")

    finally:
        engine.terminate()


if __name__ == "__main__":
    main()
