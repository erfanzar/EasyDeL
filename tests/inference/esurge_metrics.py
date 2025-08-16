#!/usr/bin/env python3
"""Test that eSurge returns consistent output with metrics for all generation methods."""

import asyncio

from easydel.inference.esurge import eSurge
from easydel.inference.sampling_params import SamplingParams


def print_output_details(output, method_name):
    """Print detailed output information."""
    print(f"\n{method_name} Output:")
    print("-" * 50)
    print(f"Request ID: {output.request_id}")
    print(f"Prompt: {output.prompt[:50]}...")
    print(f"Generated Text: {output.get_text()}")
    print(f"Accumulated Text: {output.accumulated_text}")
    print(f"Tokens Generated: {output.num_generated_tokens}")
    print(f"Time Spent: {output.time_spent_generating:.2f}s")
    print(f"Tokens/Second: {output.tokens_per_second:.2f}")
    print(f"Finished: {output.finished}")
    print(f"Finish Reason: {output.outputs[0].finish_reason if output.outputs else None}")

    # Print summary
    print("\nSummary:")
    summary = output.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def test_generate():
    """Test synchronous generation."""
    print("\n" + "=" * 60)
    print("Testing Synchronous Generation (generate)")
    print("=" * 60)

    # Initialize eSurge with small model
    engine = eSurge(
        model="microsoft/phi-2",
        max_model_len=128,
        max_num_seqs=4,
        hbm_utilization=0.3,
    )

    # Test generation
    sampling_params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = engine.generate("Once upon a time", sampling_params=sampling_params, use_tqdm=False)

    # Check output
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finished
    assert output.tokens_per_second > 0
    assert output.num_generated_tokens > 0
    assert output.time_spent_generating > 0
    assert output.accumulated_text == output.get_text()

    print_output_details(output, "generate()")
    return True


async def test_agenerate():
    """Test async generation."""
    print("\n" + "=" * 60)
    print("Testing Async Generation (agenerate)")
    print("=" * 60)

    # Initialize eSurge
    engine = eSurge(
        model="microsoft/phi-2",
        max_model_len=128,
        max_num_seqs=4,
        hbm_utilization=0.3,
    )

    # Test async generation
    sampling_params = SamplingParams(max_tokens=20, temperature=0.7)
    outputs = await engine.agenerate(
        "The future of AI is",
        sampling_params=sampling_params,
    )

    # Check output
    assert len(outputs) == 1
    output = outputs[0]
    assert output.finished
    assert output.tokens_per_second > 0
    assert output.num_generated_tokens > 0
    assert output.time_spent_generating > 0
    assert output.accumulated_text == output.get_text()

    print_output_details(output, "agenerate()")
    return True


def test_stream():
    """Test streaming generation."""
    print("\n" + "=" * 60)
    print("Testing Streaming Generation (stream)")
    print("=" * 60)

    # Initialize eSurge
    engine = eSurge(
        model="microsoft/phi-2",
        max_model_len=128,
        max_num_seqs=4,
        hbm_utilization=0.3,
    )

    # Test streaming
    sampling_params = SamplingParams(max_tokens=20, temperature=0.7)

    last_output = None
    for output in engine.stream("Hello world", sampling_params=sampling_params):
        # Check that metrics are being updated during streaming
        assert output.num_generated_tokens >= 0
        assert output.time_spent_generating >= 0
        if output.num_generated_tokens > 0:
            assert output.tokens_per_second > 0
        last_output = output

    # Check final output
    assert last_output is not None
    assert last_output.finished
    assert last_output.tokens_per_second > 0
    assert last_output.num_generated_tokens > 0
    assert last_output.accumulated_text == last_output.get_text()

    print_output_details(last_output, "stream()")
    return True


async def test_astream():
    """Test async streaming generation."""
    print("\n" + "=" * 60)
    print("Testing Async Streaming Generation (astream)")
    print("=" * 60)

    # Initialize eSurge
    engine = eSurge(
        model="microsoft/phi-2",
        max_model_len=128,
        max_num_seqs=4,
        hbm_utilization=0.3,
    )

    # Test async streaming
    sampling_params = SamplingParams(max_tokens=20, temperature=0.7)

    last_output = None
    async for output in engine.astream("In the beginning", sampling_params=sampling_params):
        # Check that metrics are being updated during streaming
        assert output.num_generated_tokens >= 0
        assert output.time_spent_generating >= 0
        if output.num_generated_tokens > 0:
            assert output.tokens_per_second > 0
        last_output = output

    # Check final output
    assert last_output is not None
    assert last_output.finished
    assert last_output.tokens_per_second > 0
    assert last_output.num_generated_tokens > 0
    assert last_output.accumulated_text == last_output.get_text()

    print_output_details(last_output, "astream()")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing eSurge Output Consistency")
    print("All methods should return RequestOutput with TPS, accumulated text, etc.")
    print("=" * 70)

    try:
        # Test synchronous methods
        test_generate()
        test_stream()

        # Test async methods
        asyncio.run(test_agenerate())
        asyncio.run(test_astream())

        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("eSurge returns consistent output format with all metrics:")
        print("  - accumulated_text: The complete generated text")
        print("  - tokens_per_second: Real-time generation speed")
        print("  - num_generated_tokens: Total tokens generated")
        print("  - time_spent_generating: Total generation time")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
