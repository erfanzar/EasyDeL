eSurge Inference Engine
=======================

eSurge is a high-performance inference engine designed for efficient batched text generation with advanced caching and scheduling capabilities. It provides fine-grained control over memory management and request scheduling, optimized for TPU, GPU, and CPU platforms.

Overview
--------

eSurge offers a production-ready serving solution with state-of-the-art performance characteristics:

- **Advanced KV Cache Management**: Page-based allocation with prefix caching support
- **Multiple Attention Patterns**: Full, sliding window, and chunked attention support
- **Flexible Request Scheduling**: FCFS and priority-based scheduling algorithms
- **Continuous Batching**: Dynamic batching for improved throughput
- **Multi-Backend Support**: Optimized for TPU, GPU, and CPU with platform-specific kernels
- **Streaming Generation**: Low-latency streaming with delta text updates
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints

Key Features
------------

Performance Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Paged Attention**: Efficient memory management with page-based KV cache allocation
- **Prefix Caching**: Reuse common prefixes across requests for improved efficiency
- **JIT Compilation**: Fully JIT-compiled forward passes for ~20% speedup
- **Background Scheduling**: Thread-safe request processing with background scheduler
- **Optimized Token Decoding**: Interval-based batching for efficient token processing

Memory Management
~~~~~~~~~~~~~~~~~

- **HBM Utilization Control**: Fine-grained control over high-bandwidth memory usage
- **Dynamic Page Allocation**: Automatic memory management with page pooling
- **Configurable Page Sizes**: Adjust page size for optimal memory utilization
- **Cache Eviction Policies**: Smart eviction strategies for long-running services

Request Handling
~~~~~~~~~~~~~~~~

- **Priority Scheduling**: Support for request prioritization
- **Request Preemption**: Ability to preempt lower-priority requests
- **Batched Processing**: Efficient batching of multiple requests
- **Stream-First Design**: Optimized for real-time streaming applications

Installation
------------

For TPU-specific optimizations:

.. code-block:: bash

    pip install easydel[tpu]

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

Here's a simple example to get started with eSurge:

.. code-block:: python

    import os
    os.environ["EASYDEL_AUTO"] = "1"  # Enable automatic sharding

    from jax import lax
    from jax import numpy as jnp
    from transformers import AutoTokenizer
    import easydel as ed

    # Load model and tokenizer
    model_id = "your-model-id"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION,
            attn_dtype=jnp.bfloat16,
        ),
    )

    # Initialize eSurge engine
    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=8192,
        max_num_seqs=16,
        hbm_utilization=0.9,
        page_size=64,
    )

    # Generate text
    response = engine.generate(
        "What is the meaning of life?",
        sampling_params=ed.SamplingParams(
            max_tokens=100,
            temperature=0.7,
        )
    )
    print(response[0].outputs[0].text)

Streaming Generation
~~~~~~~~~~~~~~~~~~~~

eSurge supports streaming generation for real-time applications:

.. code-block:: python

    # Stream tokens as they're generated
    for output in engine.stream(
        "Tell me a story about a robot",
        sampling_params=ed.SamplingParams(
            max_tokens=200,
            temperature=0.8,
        )
    ):
        print(output.delta_text, end="", flush=True)

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple prompts efficiently:

.. code-block:: python

    prompts = [
        "Explain quantum computing",
        "What is machine learning?",
        "How does the internet work?",
    ]

    # Process multiple prompts
    outputs = engine.generate(
        prompts,
        sampling_params=ed.SamplingParams(
            max_tokens=100,
            temperature=0.7,
        )
    )

    for output in outputs:
        print(f"Request {output.request_id}: {output.outputs[0].text}")

Configuration
-------------

Engine Configuration
~~~~~~~~~~~~~~~~~~~~

eSurge provides extensive configuration options:

.. code-block:: python

    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,

        # Model configuration
        max_model_len=16384,          # Maximum sequence length
        max_num_seqs=32,               # Maximum concurrent sequences

        # Memory configuration
        hbm_utilization=0.9,           # HBM utilization ratio (0.0-1.0)
        page_size=64,                  # KV cache page size

        # Performance options
        runner_verbose=False,          # Verbose runner logging
        min_input_pad=32,              # Minimum input padding

        # Naming
        esurge_name="my-engine",       # Engine instance name
    )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For fine-grained control, use configuration objects:

.. code-block:: python

    from easydel.inference.esurge import (
        Config,
        SchedulerConfig,
        CacheConfig,
    )

    # Advanced configuration is typically handled through engine parameters
    # The engine internally creates the appropriate configuration

    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=8192,
        max_num_seqs=16,
        max_num_batched_tokens=2048,  # Optional
        enable_prefix_caching=True,
        page_size=16,
        hbm_utilization=0.9,
    )

API Server
----------

eSurge includes an OpenAI-compatible API server for easy deployment:

Starting the Server
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize engine (as shown above)
    engine = ed.eSurge(model=model, tokenizer=tokenizer, ...)

    # Start monitoring (optional)
    engine.start_monitoring()

    # Launch API server
    server = ed.eSurgeApiServer(engine)
    server.fire(host="0.0.0.0", port=8000)

The server provides OpenAI-compatible endpoints:

- ``POST /v1/chat/completions`` - Chat completions
- ``POST /v1/completions`` - Text completions
- ``GET /v1/models`` - List available models
- ``GET /health`` - Health check endpoint
- ``GET /metrics`` - Prometheus metrics (if monitoring enabled)

Using the API
~~~~~~~~~~~~~

Once the server is running, you can use any OpenAI-compatible client:

.. code-block:: python

    import openai

    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-required",  # eSurge doesn't require API keys
    )

    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        temperature=0.7,
        max_tokens=100,
        stream=True,  # Streaming is supported
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, end="")

Monitoring and Metrics
----------------------

eSurge provides comprehensive monitoring capabilities:

Console Monitoring
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Start console monitoring
    engine.start_monitoring()

    # The console will display:
    # - Request throughput
    # - Token generation rate
    # - Memory utilization
    # - Cache hit rates
    # - Active request count

Web Dashboard
~~~~~~~~~~~~~

Launch an interactive web dashboard:

.. code-block:: python

    from easydel.inference.esurge import eSurgeWebDashboard

    dashboard = eSurgeWebDashboard(engine)
    dashboard.launch(port=7860)

Prometheus Metrics
~~~~~~~~~~~~~~~~~~

Export metrics for Prometheus:

.. code-block:: python

    from easydel.inference.esurge import start_monitoring_server

    # Start Prometheus metrics server
    monitoring_server = start_monitoring_server(engine, port=9090)

    # Access metrics at http://localhost:9090/metrics

Performance Tuning
------------------

TPU Optimization
~~~~~~~~~~~~~~~~

For best performance on TPU:

.. code-block:: python

    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=8192,
        max_num_seqs=64,        # TPUs handle larger batches well
        page_size=128,          # Larger pages for TPU
        use_aot_forward=True,   # Essential for TPU performance
        hbm_utilization=0.95,   # TPUs have abundant HBM
    )

GPU Optimization
~~~~~~~~~~~~~~~~

For NVIDIA GPUs:

.. code-block:: python

    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=4096,
        max_num_seqs=32,
        page_size=64,
        hbm_utilization=0.85,       # Leave headroom for kernels
    )

Memory-Constrained Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For systems with limited memory:

.. code-block:: python

    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=2048,     # Reduce max length
        max_num_seqs=8,         # Fewer concurrent sequences
        page_size=32,           # Smaller pages
        hbm_utilization=0.7,    # Conservative memory usage
    )

Advanced Features
-----------------

Prefix Caching
~~~~~~~~~~~~~~

Enable prefix caching for improved efficiency with common prefixes:

.. code-block:: python

    from easydel.inference.esurge import CacheConfig

    cache_config = CacheConfig(
        enable_prefix_caching=True,
        prefix_cache_capacity=0.3,  # 30% of cache for prefixes
    )

    engine = ed.eSurge(
        model=model,
        tokenizer=tokenizer,
        cache_config=cache_config,
    )

Custom Scheduling
~~~~~~~~~~~~~~~~~

Implement custom scheduling logic:

.. code-block:: python

    from easydel.inference.esurge import PriorityRequestQueue

    # Priority scheduling is configured in the scheduler
    # The scheduler automatically handles request prioritization
    # based on the scheduling policy (FCFS or Priority)

    # Note: Direct priority setting is handled internally
    # by the scheduler based on arrival time and request characteristics

Function Calling
~~~~~~~~~~~~~~~~

eSurge supports function calling for tool use:

.. code-block:: python

    # Function calling is available through the API server
    # When using the API server with OpenAI-compatible clients,
    # function calling follows the OpenAI format:

    # client.chat.completions.create(
    #     model="default",
    #     messages=[{"role": "user", "content": "What's the weather?"}],
    #     functions=[{...}],  # OpenAI function schema
    #     function_call="auto"
    # )

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Out of Memory Errors**

Reduce memory usage by:

- Decreasing ``max_num_seqs``
- Reducing ``max_model_len``
- Lowering ``hbm_utilization``
- Using smaller ``page_size``

**Slow Generation**

Improve performance by:

- Enabling ``use_aot_forward`` for TPU
- Increasing ``page_size`` for better memory access patterns
- Adjusting ``max_num_batched_tokens``
- Using appropriate precision (``bfloat16`` recommended)

**Request Timeouts**

Handle long-running requests:

- Increase timeout values in sampling parameters
- Monitor request queue with dashboard
- Implement request preemption for fairness

Best Practices
--------------

1. **Memory Management**

   - Monitor HBM utilization and adjust ``hbm_utilization`` parameter
   - Use prefix caching for repetitive prompts
   - Enable request preemption for long-running services

2. **Performance Optimization**

   - Use platform-specific settings (TPU vs GPU)
   - Enable JIT compilation with ``use_aot_forward``
   - Batch similar-length sequences together

3. **Production Deployment**

   - Enable monitoring and metrics collection
   - Implement health checks and auto-restart
   - Use load balancing for multiple instances
   - Set up proper logging and alerting

4. **Scaling**

   - Horizontal scaling with multiple engine instances
   - Use model parallelism for large models
   - Implement request routing based on load

API Reference
-------------

For detailed API documentation, see:

- :doc:`api_docs/inference_esurge` - Core eSurge APIs
- :doc:`api_docs/inference_esurge_engine` - Engine implementation
- :doc:`api_docs/inference_esurge_scheduler` - Scheduling components
- :doc:`api_docs/inference_esurge_server` - API server implementation

Example Applications
--------------------

Chat Application
~~~~~~~~~~~~~~~~

Complete example of a chat application:

.. code-block:: python

    import os
    os.environ["EASYDEL_AUTO"] = "1"

    from jax import lax, numpy as jnp
    from transformers import AutoTokenizer
    import easydel as ed

    def create_chat_engine(model_id="microsoft/phi-2"):
        """Create an eSurge engine for chat."""
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            model_id,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            config_kwargs=ed.EasyDeLBaseConfigDict(
                attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION,
            ),
        )

        engine = ed.eSurge(
            model=model,
            tokenizer=tokenizer,
            max_model_len=4096,
            max_num_seqs=16,
            hbm_utilization=0.9,
            page_size=64,
            esurge_name="chat-engine",
        )

        return engine

    def chat_loop(engine):
        """Interactive chat loop."""
        conversation = []

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            conversation.append({"role": "user", "content": user_input})

            # Format conversation for model
            prompt = format_conversation(conversation)

            # Generate response
            print("Assistant: ", end="", flush=True)
            response_text = ""

            for output in engine.stream(
                prompt,
                sampling_params=ed.SamplingParams(
                    max_tokens=200,
                    temperature=0.7,
                    stop=["</s>", "\n\nYou:"],
                )
            ):
                print(output.delta_text, end="", flush=True)
                response_text += output.delta_text

            print()  # New line after response
            conversation.append({"role": "assistant", "content": response_text})

    def format_conversation(messages):
        """Format messages for model input."""
        formatted = ""
        for msg in messages:
            role = msg["role"].capitalize()
            formatted += f"{role}: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted

    if __name__ == "__main__":
        engine = create_chat_engine()
        engine.start_monitoring()
        print("Chat engine ready! Type 'quit' to exit.")
        chat_loop(engine)

Batch Processing Service
~~~~~~~~~~~~~~~~~~~~~~~~

Example of a batch processing service:

.. code-block:: python

    import asyncio
    from typing import List, Dict
    import easydel as ed

    class BatchProcessor:
        """Batch text processing service."""

        def __init__(self, engine: ed.eSurge):
            self.engine = engine
            self.pending_requests = {}

        async def process_batch(
            self,
            texts: List[str],
            sampling_params: ed.SamplingParams = None,
        ) -> List[str]:
            """Process a batch of texts."""
            if sampling_params is None:
                sampling_params = ed.SamplingParams(
                    max_new_tokens=100,
                    temperature=0.7,
                )

            # Submit all requests
            request_ids = []
            for text in texts:
                request_id = self.engine.add_request(
                    prompt=text,
                    sampling_params=sampling_params,
                )
                request_ids.append(request_id)
                self.pending_requests[request_id] = None

            # Wait for all completions
            results = {}
            while len(results) < len(request_ids):
                outputs = self.engine.step()

                for output in outputs:
                    if output.finished:
                        results[output.request_id] = output.outputs[0].text
                        self.pending_requests.pop(output.request_id, None)

                await asyncio.sleep(0.01)  # Small delay

            # Return in original order
            return [results[rid] for rid in request_ids]

        async def process_stream(
            self,
            texts: List[str],
            callback=None,
        ):
            """Process texts with streaming callbacks."""
            for text in texts:
                request_id = self.engine.add_request(
                    prompt=text,
                    sampling_params=ed.SamplingParams(
                        max_new_tokens=100,
                        temperature=0.7,
                    ),
                )

                # Stream this request to completion
                while True:
                    outputs = self.engine.step()

                    for output in outputs:
                        if output.request_id == request_id:
                            if callback:
                                await callback(output)

                            if output.finished:
                                break
                    else:
                        await asyncio.sleep(0.01)
                        continue
                    break

    # Usage example
    async def main():
        engine = create_chat_engine()  # From previous example
        processor = BatchProcessor(engine)

        # Batch processing
        prompts = [
            "Explain quantum computing in simple terms",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
        ]

        results = await processor.process_batch(prompts)
        for prompt, result in zip(prompts, results):
            print(f"Q: {prompt}\nA: {result}\n")

    if __name__ == "__main__":
        asyncio.run(main())

Migration Guide
---------------

From vInference to eSurge
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're migrating from vInference to eSurge:

.. code-block:: python

    # Old vInference code
    from easydel.inference import vInference

    engine = vInference(
        model=model,
        processor=tokenizer,
        generation_config=config,
    )

    # New eSurge code
    from easydel.inference.esurge import eSurge

    engine = eSurge(
        model=model,
        tokenizer=tokenizer,
        max_model_len=config.max_length,
        max_num_seqs=16,  # New parameter for concurrency
        hbm_utilization=0.9,  # Better memory control
    )

Key differences:

- eSurge uses ``tokenizer`` instead of ``processor``
- More granular configuration options
- Built-in support for continuous batching
- Improved memory management with paged attention

Performance Comparison
----------------------

eSurge vs Other Engines
~~~~~~~~~~~~~~~~~~~~~~~~

Performance characteristics compared to other inference engines:

+------------------+----------+----------+-----------+-------------+
| Metric           | eSurge   | vLLM     | TGI       | vInference  |
+==================+==========+==========+===========+=============+
| TPU Support      | ✅ Native| ⚠️ Limited  | ❌ None   | ✅ Native   |
+------------------+----------+----------+-----------+-------------+
| Paged Attention  | ✅ Yes   | ✅ Yes   | ✅ Yes    | ❌ No       |
+------------------+----------+----------+-----------+-------------+
| Prefix Caching   | ✅ Yes   | ✅ Yes   | ⚠️ Limited| ❌ No       |
+------------------+----------+----------+-----------+-------------+
| JAX/Flax Native  | ✅ Yes   | ❌ No    | ❌ No     | ✅ Yes      |
+------------------+----------+----------+-----------+-------------+
| Continuous Batch | ✅ Yes   | ✅ Yes   | ✅ Yes    | ⚠️ Limited  |
+------------------+----------+----------+-----------+-------------+

Future Roadmap
--------------

Planned features for future releases:

- **Speculative Decoding**: Faster generation with draft models
- **Multi-LoRA Support**: Serve multiple LoRA adapters efficiently
- **Tensor Parallelism**: Scale to larger models
- **Quantization Methods**: Additional quantization schemes (AWQ, GPTQ)
- **Model Hotswapping**: Switch models without restarting
- **Request Caching**: Cache and reuse common request patterns
- **Pipeline Parallelism**: Better scaling across multiple devices

Contributing
------------

We welcome contributions to eSurge! Areas where help is appreciated:

- Performance optimizations for specific hardware
- Additional scheduling algorithms
- New attention mechanisms
- Documentation and examples
- Bug reports and fixes

See the main EasyDeL contributing guide for more information.

Support
-------

For issues and questions:

- GitHub Issues: https://github.com/erfanzar/EasyDeL/issues
- Discord: Join the EasyDeL community
- Documentation: https://easydel.readthedocs.io

License
-------

eSurge is part of EasyDeL and is licensed under the Apache License 2.0.
