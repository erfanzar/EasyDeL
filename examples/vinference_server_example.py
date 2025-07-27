#!/usr/bin/env python3
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example of using the vInference API Server for language models.

This script demonstrates:
1. How to load a text-based language model like LLaMA
2. How to configure the vInference for the model
3. How to start the API server
"""

import argparse
import os

# Add the parent directory to the path so we can import easydel
import jax
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed


def main():
    parser = argparse.ArgumentParser(description="Run the EasyDeL vInference API server")

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name or path from HuggingFace",
    )

    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")

    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type for model",
    )

    args = parser.parse_args()

    # Set up sharding and device config
    sharding_axis_dims = (1, 1, 1, 1, -1)  # You may adjust this based on your hardware
    partition_axis = ed.PartitionAxis()

    # Map dtype string to actual type
    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16, "float32": jnp.float32}
    dtype = dtype_map[args.dtype]

    # Load the model
    print(f"Loading model: {args.model}")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        args.model,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=args.max_length,
            mask_max_position_embeddings=args.max_length,
            attn_dtype=dtype,
            attn_softmax_dtype=dtype,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=dtype,
        dtype=dtype,
        partition_axis=partition_axis,
        precision=jax.lax.Precision.DEFAULT,
    )

    model.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Create the inference object
    inference = ed.vInference(
        model=model,
        processor_class=tokenizer,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=2048,
            streaming_chunks=64,
            num_return_sequences=1,
            sampling_params=ed.SamplingParams(
                temperature=0.7,
                top_p=0.95,
                top_k=50,
            ),
        ),
        inference_name=os.path.basename(args.model),
    )

    # Precompile for better performance
    inference.precompile(
        ed.vInferencePreCompileConfig(
            batch_size=1,
            prefill_length=1024,
        )
    )

    # Print information about the server
    model_id = inference.inference_name
    print(f"\nStarting vInference API Server with model: {model_id}")
    print(f"Server will be available at: http://0.0.0.0:{args.port}")
    print("\nExample curl request:")
    print("------------------------------------------")
    print("curl -X POST http://0.0.0.0:" + str(args.port) + "/v1/chat/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print("     -d '{")
    print(f'  "model": "{model_id}",')
    print('  "messages": [')
    print("    {")
    print('      "role": "user",')
    print('      "content": "Explain quantum computing in simple terms"')
    print("    }")
    print("  ],")
    print('  "temperature": 0.7,')
    print('  "max_tokens": 500')
    print("}'")
    print("------------------------------------------")
    print("\nUse Ctrl+C to stop the server.\n")

    # Start the API server
    ed.vInferenceApiServer(inference).fire(port=args.port)


if __name__ == "__main__":
    main()
