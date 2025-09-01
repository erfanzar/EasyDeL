vSurge Inference Engine Example
===============================

The `vSurge` component in EasyDeL provides a flexible and efficient inference engine for large language models. It is designed to handle both streaming and non-streaming text generation. This example demonstrates how to use `vSurge` with the `oDrive` engine backend, based on a typical inference script.

Import necessary modules
------------------------

First, import the required libraries: `jax`, `jax.numpy`, `transformers.AutoTokenizer`, and `easydel`.

.. code-block:: python

    import jax
    from jax import numpy as jnp
    from transformers import AutoTokenizer

    import easydel as ed

Load the model and tokenizer
----------------------------

Load the pretrained model and tokenizer using EasyDeL's `AutoEasyDeLModelForCausalLM` and Hugging Face's `AutoTokenizer`. Configure the model with necessary parameters like sharding, data types, and attention mechanisms.

-   `pretrained_model_name_or_path`: Specifies the model to load from Hugging Face or a local path.
-   `dtype` and `param_dtype`: Define the data types for computations and model parameters, respectively. `jnp.bfloat16` is commonly used for efficiency.
-   `max_length`: The maximum sequence length the model can handle.
-   `prefill_length`: The maximum length for the initial prompt processing (prefill stage).
-   `partition_axis`: An EasyDeL utility for defining sharding configurations across devices.
-   `processor`: The tokenizer loaded using `AutoTokenizer`. Padding side is set to "left" and pad token is set to EOS token for batched inference.
-   `model`: The EasyDeL model loaded using `AutoEasyDeLModelForCausalLM`.
    -   `auto_shard_model`: Automatically shards the model parameters across available devices.
    -   `sharding_axis_dims`: Defines the sharding dimensions for the model parameters. `(1, 1, 1, -1, 1)` is a common configuration.
    -   `config_kwargs`: Allows passing additional configuration parameters to the model's underlying configuration object using `EasyDeLBaseConfigDict`.
        -   `freq_max_position_embeddings` and `mask_max_position_embeddings`: Related to rotary embeddings and attention masks, set to `max_length`.
        -   `kv_cache_quantization_method`: Specifies the quantization method for the KV cache. `ed.EasyDeLQuantizationMethods.NONE` means no quantization.
        -   `gradient_checkpointing`: Controls gradient checkpointing behavior. `ed.EasyDeLGradientCheckPointers.NONE` disables it.
        -   `attn_mechanism`: Specifies the attention mechanism to use. `ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION` is crucial for efficient KV cache management in vSurge.
    -   `quantization_method`: Specifies the quantization method for model weights. `ed.EasyDeLQuantizationMethods.NONE` means no quantization.
    -   `precision`: Controls the precision of computations. `jax.lax.Precision.Precision.DEFAULT` uses the default precision for the chosen dtype.

.. code-block:: python

    pretrained_model_name_or_path = (
        "Qwen/Qwen3-8B"
    )
    dtype = param_dtype = jnp.bfloat16
    max_length = 8192
    prefill_length = 4096
    partition_axis = ed.PartitionAxis()
    processor = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    processor.padding_side = "left"
    processor.pad_token_id = processor.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            decode_attn_mechanism=ed.AttentionMechanisms.REGRESSIVE_DECODE,
            kvdtype=jnp.bfloat16,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=param_dtype,
        dtype=dtype,
        partition_axis=partition_axis,
        precision=jax.lax.Precision.DEFAULT,
    )

Create the vSurge instance
--------------------------

Instantiate the `vSurge` engine using the `create_odriver` class method. This method sets up `vSurge` to use the `oDrive` engine backend. Provide the loaded model, processor, and configuration parameters.

-   `model`: The loaded EasyDeL model.
-   `processor`: The loaded tokenizer.
-   `max_concurrent_prefill`: The maximum number of prefill requests that can be processed concurrently.
-   `max_concurrent_decodes`: The maximum number of decode requests (token generation) that can be processed concurrently.
-   `seed`: A random seed for reproducibility.

.. code-block:: python

    max_concurrent_decodes = 64
    max_concurrent_prefill = 1

    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        max_prefill_length=prefill_length,
        max_concurrent_prefill=max_concurrent_prefill,
        max_concurrent_decodes=max_concurrent_decodes,
        seed=877,
    )

Start and Compile the Engine
----------------------------

Before performing inference, the `vSurge` engine needs to be started and compiled. The `start()` method initializes the engine, and the `compile()` method compiles the necessary JAX functions for efficient execution.

.. code-block:: python

    surge.compile()
    surge.start()

Non-Streaming Generation
------------------------

For non-streaming generation, call the `generate` method with `stream=False`. Provide a list of prompts and corresponding sampling parameters. The method will return a list of final results once generation is complete for all prompts.

-   `prompts`: A list of input strings for which to generate text.
-   `sampling_params`: A list of `ed.SamplingParams` objects, one for each prompt, specifying parameters like:
    -   `max_tokens`: The maximum number of tokens to generate.
    -   `temperature`: Controls the randomness of the output. Higher values mean more randomness.
    -   `top_p`: The cumulative probability threshold for nucleus sampling.

.. code-block:: python

    non_streaming_prompts = [
        "USER:What is the capital of France?\nASSISTANT:",
        "USER:Explain the concept of recursion\nASSISTANT:",
    ]
    non_streaming_sampling_params = [
        ed.SamplingParams(max_tokens=30, temperature=0.1),
        ed.SamplingParams(max_tokens=80, temperature=0.6, top_p=0.9),
    ]

    # For non-streaming, the generate method returns a list of final results
    # Note: generate is an async method, so it should be awaited in an async context.
    import asyncio

    async def run_non_streaming():
        final_results = await surge.generate(
            prompts=non_streaming_prompts,
            sampling_params=non_streaming_sampling_params,
            stream=False,
        )

        # final_results is a list of ReturnSample objects (one per prompt)
        for i, result in enumerate(final_results):
            print(f"Non-Streaming Result for Prompt {i + 1}:")
            print(f"  Generated Text: {result.text}")
            print(f"  Tokens per second: {result.tokens_per_second}")

    # To run this in a script:
    # asyncio.run(run_non_streaming())

Iterate through the results to access the generated text and other information like tokens per second.

Streaming Generation
--------------------

The `vSurge` engine also supports streaming generation, which is useful for applications that need to display tokens as they are generated (e.g., chatbots). To perform streaming inference, you call the `generate` method with `stream=True`. This method returns an asynchronous iterator that yields `ReturnSample` objects as tokens are generated for each prompt.

You would typically iterate through this asynchronous iterator to process the incoming tokens.

.. code-block:: python

    streaming_prompts = [
        "USER:Tell me a short story about a cat.\nASSISTANT:",
        "USER:Describe the process of photosynthesis.\nASSISTANT:",
    ]
    streaming_sampling_params = [
        ed.SamplingParams(max_tokens=50, temperature=0.7),
        ed.SamplingParams(max_tokens=100, temperature=0.5),
    ]

    async def run_streaming():
        # generate with stream=True returns an async iterator
        async for request_output in surge.generate(
            prompts=streaming_prompts,
            sampling_params=streaming_sampling_params,
            stream=True,
        ):
            # request_output is a list of ReturnSample objects, one for each prompt
            for i, sample in enumerate(request_output):
                # The text field in streaming provides the cumulative generated text so far
                print(f"Streaming Update for Prompt {i + 1}: {sample.text}")
            # Add a small delay to simulate processing time if needed
            # await asyncio.sleep(0.01)

    # To run this in a script:
    # asyncio.run(run_streaming())

Stop the Engine
---------------

After completing inference, stop the `vSurge` engine to release resources.

.. code-block:: python

    surge.stop()
