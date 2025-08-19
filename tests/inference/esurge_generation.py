"""Example usage of the eSurge engine for text generation."""

import asyncio

from easydel.inference.esurge.esurge_engine import eSurge
from easydel.inference.sampling_params import SamplingParams


def run_all_examples():
    """Run all examples with a single engine instance to avoid OOM."""
    print("=" * 60)
    print("Initializing eSurge Engine")
    print("=" * 60)

    # Create a single engine instance
    engine = eSurge(
        model="meta-llama/Llama-3.2-3B-Instruct",
        max_model_len=512,
        max_num_seqs=8,
        hbm_utilization=0.4,
        enable_prefix_caching=True,
    )

    # Simple generation
    print("\n" + "=" * 60)
    print("Simple Generation Example")
    print("=" * 60)

    prompt = "The future of artificial intelligence is"
    outputs = engine.generate(
        prompt,
        sampling_params=SamplingParams(
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
        ),
    )

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()

    # Batch generation
    print("=" * 60)
    print("Batch Generation Example")
    print("=" * 60)

    prompts = [
        "Write a haiku about programming:",
        "Explain quantum computing in simple terms:",
        "What are the benefits of exercise?",
        "How do neural networks work?",
    ]

    outputs = engine.generate(
        prompts,
        sampling_params=SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        ),
    )

    for i, output in enumerate(outputs, 1):
        print(f"[{i}] Prompt: {output.prompt}")
        print(f"    Response: {output.outputs[0].text}")
        print()

    # Streaming generation
    print("=" * 60)
    print("Streaming Generation Example")
    print("=" * 60)

    prompt = "Tell me a short story about a robot:"

    print(f"Prompt: {prompt}")
    print("Streaming response: ", end="", flush=True)

    last_text = ""
    for output in engine.stream(
        prompt,
        sampling_params=SamplingParams(max_tokens=150),
    ):
        # Print only new tokens
        if output.outputs[0].text:
            new_text = output.outputs[0].text[len(last_text) :]
            print(new_text, end="", flush=True)
            last_text = output.outputs[0].text

    print("\n")

    # Custom sampling parameters
    print("=" * 60)
    print("Custom Sampling Parameters Example")
    print("=" * 60)

    prompt = "Generate creative names for a new programming language:"

    # Different sampling strategies
    sampling_configs = [
        ("Creative", SamplingParams(max_tokens=30, temperature=1.2, top_p=0.95)),
        ("Balanced", SamplingParams(max_tokens=30, temperature=0.5, top_p=0.9)),
        ("Conservative", SamplingParams(max_tokens=30, temperature=0.1, top_k=10)),
    ]

    for name, params in sampling_configs:
        outputs = engine.generate(prompt, sampling_params=params)
        print(f"{name} (temp={params.temperature}):")
        print(f"  {outputs[0].outputs[0].text}")
        print()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


async def run_async_examples():
    """Run async examples with a single engine instance."""
    print("\n" + "=" * 60)
    print("Async Examples")
    print("=" * 60)

    engine = eSurge(
        model="meta-llama/Llama-3.2-3B-Instruct",
        max_model_len=256,
        max_num_seqs=4,
        hbm_utilization=0.4,
    )

    print("\nAsync Generation Example")
    print("-" * 40)

    prompts = [
        "What is machine learning?",
        "Explain deep learning:",
        "What are transformers in AI?",
    ]

    tasks = [
        engine.agenerate(
            prompt,
            sampling_params=SamplingParams(max_tokens=80),
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)

    for outputs in results:
        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Response: {output.outputs[0].text}")
            print()

    print("Async Streaming Example")
    print("-" * 40)

    prompt = "Explain the concept of recursion:"

    print(f"Prompt: {prompt}")
    print("Streaming: ", end="", flush=True)

    last_text = ""
    async for output in engine.astream(
        prompt,
        sampling_params=SamplingParams(max_tokens=100),
    ):
        if output.outputs[0].text:
            new_text = output.outputs[0].text[len(last_text) :]
            print(new_text, end="", flush=True)
            last_text = output.outputs[0].text

    print("\n")


def main():
    """Run all examples."""
    run_all_examples()
    asyncio.run(run_async_examples())


if __name__ == "__main__":
    main()
