eSurge Examples
===============

This document provides practical examples of using the eSurge inference engine.

Basic Generation
----------------

Simple Text Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    os.environ["EASYDEL_AUTO"] = "1"

    from jax import numpy as jnp
    import easydel as ed

    # Initialize engine with a model
    engine = ed.eSurge(
        model="meta-llama/Llama-3.2-3B-Instruct",
        max_model_len=512,
        max_num_seqs=8,
        hbm_utilization=0.4,
        enable_prefix_caching=True,
    )

    # Simple generation
    outputs = engine.generate(
        "The future of artificial intelligence is",
        sampling_params=ed.SamplingParams(
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
        ),
    )

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")

Batch Generation
~~~~~~~~~~~~~~~~

.. code-block:: python

    prompts = [
        "Write a haiku about programming:",
        "Explain quantum computing in simple terms:",
        "What are the benefits of exercise?",
        "How do neural networks work?",
    ]

    outputs = engine.generate(
        prompts,
        sampling_params=ed.SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        ),
    )

    for i, output in enumerate(outputs, 1):
        print(f"[{i}] Prompt: {output.prompt}")
        print(f"    Response: {output.outputs[0].text}")
        print()

Streaming Generation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    prompt = "Tell me a short story about a robot:"

    print(f"Prompt: {prompt}")
    print("Streaming response: ", end="", flush=True)

    last_text = ""
    for output in engine.stream(
        prompt,
        sampling_params=ed.SamplingParams(max_tokens=150),
    ):
        # Print only new tokens
        if output.outputs[0].text:
            new_text = output.outputs[0].text[len(last_text):]
            print(new_text, end="", flush=True)
            last_text = output.outputs[0].text

    print("\n")

Async Generation
----------------

Async with asyncio
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import asyncio

    async def generate_async():
        engine = ed.eSurge(
            model="meta-llama/Llama-3.2-3B-Instruct",
            max_model_len=256,
            max_num_seqs=4,
            hbm_utilization=0.4,
        )

        prompts = [
            "What is machine learning?",
            "Explain deep learning:",
            "What are transformers in AI?",
        ]

        tasks = [
            engine.agenerate(
                prompt,
                sampling_params=ed.SamplingParams(max_tokens=80),
            )
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks)

        for outputs in results:
            for output in outputs:
                print(f"Prompt: {output.prompt}")
                print(f"Response: {output.outputs[0].text}")
                print()

    # Run async example
    asyncio.run(generate_async())

Async Streaming
~~~~~~~~~~~~~~~

.. code-block:: python

    async def stream_async():
        prompt = "Explain the concept of recursion:"

        print(f"Prompt: {prompt}")
        print("Streaming: ", end="", flush=True)

        last_text = ""
        async for output in engine.astream(
            prompt,
            sampling_params=ed.SamplingParams(max_tokens=100),
        ):
            if output.outputs[0].text:
                new_text = output.outputs[0].text[len(last_text):]
                print(new_text, end="", flush=True)
                last_text = output.outputs[0].text

        print("\n")

    asyncio.run(stream_async())

Custom Sampling
---------------

Different Sampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    prompt = "Generate creative names for a new programming language:"

    # Different sampling strategies
    sampling_configs = [
        ("Creative", ed.SamplingParams(max_tokens=30, temperature=1.2, top_p=0.95)),
        ("Balanced", ed.SamplingParams(max_tokens=30, temperature=0.5, top_p=0.9)),
        ("Conservative", ed.SamplingParams(max_tokens=30, temperature=0.1, top_k=10)),
    ]

    for name, params in sampling_configs:
        outputs = engine.generate(prompt, sampling_params=params)
        print(f"{name} (temp={params.temperature}):")
        print(f"  {outputs[0].outputs[0].text}")
        print()

Advanced Configuration
----------------------

Loading Custom Models
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from easydel import AutoEasyDeLModelForCausalLM, EasyDeLBaseConfigDict
    from easydel.layers.attention import AttentionMechanisms
    from jax import lax
    from transformers import AutoTokenizer

    # Load model with custom configuration
    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        "your-model-id",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=EasyDeLBaseConfigDict(
            freq_max_position_embeddings=16384,
            mask_max_position_embeddings=16384,
            attn_mechanism=AttentionMechanisms.RAGGED_PAGE_ATTENTION,
            attn_dtype=jnp.bfloat16,
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained("your-model-id")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create engine with preloaded model
    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=16384,
        max_num_seqs=64,
        hbm_utilization=0.9,
        page_size=128,
        esurge_name="custom-model",
    )

API Server Integration
----------------------

Starting the Server
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize engine
    engine = ed.eSurge(
        model="meta-llama/Llama-3.2-3B-Instruct",
        max_model_len=2048,
        max_num_seqs=16,
        hbm_utilization=0.85,
    )

    # Enable monitoring
    engine.start_monitoring()

    # Launch API server
    server = ed.eSurgeApiServer(engine)
    server.fire(host="0.0.0.0", port=8000)

Using with OpenAI Client
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import openai

    # Configure client
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-required",
    )

    # Chat completion
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing"}
        ],
        temperature=0.7,
        max_tokens=200,
        stream=True,
    )

    # Stream response
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

Performance Monitoring
----------------------

Metrics Collection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Start monitoring
    urls = engine.start_monitoring(
        dashboard_port=8080,
        prometheus_port=9090,
        enable_dashboard=True,
        enable_prometheus=True,
        enable_console=False,
    )

    print(f"Dashboard: {urls['dashboard']}")
    print(f"Prometheus: {urls['prometheus']}")

    # Generate some requests
    for i in range(10):
        engine.generate(
            f"Question {i}: What is {i}?",
            sampling_params=ed.SamplingParams(max_tokens=50),
        )

    # Get metrics summary
    metrics = engine.get_metrics_summary()
    print(f"Requests/sec: {metrics['requests_per_second']:.2f}")
    print(f"Avg latency: {metrics['average_latency']:.3f}s")
    print(f"Avg throughput: {metrics['average_throughput']:.1f} tokens/s")

Error Handling
--------------

Request Management
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Generate with request ID tracking
    request_id = "custom-request-123"

    try:
        outputs = engine.generate(
            "Tell me about space exploration",
            sampling_params=ed.SamplingParams(max_tokens=100),
            request_id=request_id,
        )
        print(outputs[0].outputs[0].text)
    except Exception as e:
        print(f"Generation failed: {e}")
        # Abort the request if needed
        engine.abort_request(request_id)

    # Check engine status
    print(f"Pending requests: {engine.num_pending_requests}")
    print(f"Running requests: {engine.num_running_requests}")

Complete Example Application
-----------------------------

Chat Application
~~~~~~~~~~~~~~~~

.. code-block:: python

    import os
    os.environ["EASYDEL_AUTO"] = "1"

    from jax import numpy as jnp
    import easydel as ed

    class ChatBot:
        def __init__(self, model_id="microsoft/phi-2"):
            self.engine = ed.eSurge(
                model=model_id,
                max_model_len=2048,
                max_num_seqs=8,
                hbm_utilization=0.8,
                esurge_name="chatbot",
            )
            self.conversation = []

        def format_prompt(self, messages):
            """Format conversation for model."""
            prompt = ""
            for msg in messages:
                role = msg["role"].capitalize()
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant: "
            return prompt

        def chat(self, user_input):
            """Process user input and return response."""
            self.conversation.append({"role": "user", "content": user_input})

            prompt = self.format_prompt(self.conversation)

            # Stream response
            response_text = ""
            print("Assistant: ", end="", flush=True)

            for output in self.engine.stream(
                prompt,
                sampling_params=ed.SamplingParams(
                    max_tokens=200,
                    temperature=0.7,
                    stop=["\nUser:", "\n\n"],
                )
            ):
                if output.delta_text:
                    print(output.delta_text, end="", flush=True)
                    response_text += output.delta_text

            print()  # New line after response
            self.conversation.append({"role": "assistant", "content": response_text})
            return response_text

        def reset(self):
            """Reset conversation history."""
            self.conversation = []

    # Use the chatbot
    if __name__ == "__main__":
        bot = ChatBot()

        print("ChatBot initialized. Type 'quit' to exit, 'reset' to clear history.")

        while True:
            user_input = input("\nYou: ")

            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'reset':
                bot.reset()
                print("Conversation reset.")
                continue

            bot.chat(user_input)

Best Practices
--------------

1. **Resource Management**

   .. code-block:: python

       # Always terminate the engine when done
       try:
           outputs = engine.generate(prompt)
       finally:
           engine.terminate()
           if engine.monitoring_active:
               engine.stop_monitoring()

2. **Batch Processing**

   .. code-block:: python

       # Process in batches for better throughput
       def process_large_dataset(prompts, batch_size=16):
           results = []
           for i in range(0, len(prompts), batch_size):
               batch = prompts[i:i+batch_size]
               outputs = engine.generate(
                   batch,
                   sampling_params=ed.SamplingParams(max_tokens=100)
               )
               results.extend(outputs)
           return results

3. **Streaming with Progress**

   .. code-block:: python

       from tqdm import tqdm

       def stream_with_progress(prompt, max_tokens=200):
           pbar = tqdm(total=max_tokens, desc="Generating")

           for output in engine.stream(
               prompt,
               sampling_params=ed.SamplingParams(max_tokens=max_tokens)
           ):
               new_tokens = output.num_generated_tokens - pbar.n
               if new_tokens > 0:
                   pbar.update(new_tokens)

               if output.finished:
                   break

           pbar.close()
           return output.outputs[0].text

See Also
--------

- :doc:`esurge` - Main eSurge documentation
- :doc:`api_docs/inference_esurge` - API reference
- :doc:`vinference_api` - Alternative inference engine
