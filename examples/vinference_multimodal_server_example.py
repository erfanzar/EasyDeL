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
Example of using the vInference API Server for multimodal models.

This script demonstrates:
1. How to load a vision-language model like LLaVA
2. How to configure the vInference for multimodal processing
3. How to start the API server with vision capabilities
"""

import argparse

# Add the parent directory to the path so we can import easydel
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

import easydel as ed


def main():
    parser = argparse.ArgumentParser(description="Run the EasyDeL vInference API server for multimodal models")

    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Model name or path from HuggingFace",
    )

    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")

    parser.add_argument("--prefill_length", type=int, default=2048, help="Prefill sequence length")

    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens to generate")

    parser.add_argument("--vision_height", type=int, default=336, help="Image height for vision processing")

    parser.add_argument("--vision_width", type=int, default=336, help="Image width for vision processing")

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type for model weights",
    )

    parser.add_argument(
        "--param_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Data type for model parameters",
    )

    args = parser.parse_args()

    # Set up sharding and device config
    sharding_axis_dims = (1, 1, 1, -1, 1)  # You may adjust this based on your hardware

    # Map dtype string to actual type
    dtype_map = {"float16": jnp.float16, "bfloat16": jnp.bfloat16, "float32": jnp.float32}
    dtype = dtype_map[args.dtype]
    param_dtype = dtype_map[args.param_dtype]

    # Set up max_length based on prefill and max_new_tokens
    max_length = args.prefill_length + args.max_new_tokens
    partition_axis = ed.PartitionAxis()

    # Load processor for the multimodal model
    print(f"Loading processor for model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    processor.padding_side = "left"

    # Load the model
    print(f"Loading model: {args.model}")
    model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
        args.model,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            attn_dtype=param_dtype,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=param_dtype,
        dtype=dtype,
        partition_axis=partition_axis,
        precision=jax.lax.Precision.DEFAULT,
    )

    # Create inference configuration
    inference = ed.vInference(
        model=model,
        processor_class=processor,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=args.max_new_tokens,
            sampling_params=ed.SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                top_k=10,
            ),
            eos_token_id=model.generation_config.eos_token_id,
            streaming_chunks=32,
            num_return_sequences=1,
        ),
        inference_name="multimodal",
    )

    # Precompile with vision settings for better performance
    inference.precompile(
        ed.vInferencePreCompileConfig(
            batch_size=1,
            prefill_length=args.prefill_length,
            vision_included=True,
            vision_batch_size=1,
            vision_channels=3,
            vision_height=args.vision_height,
            vision_width=args.vision_width,
        )
    )

    # Print information about the server
    print(f"\nStarting vInference API Server with model: {args.model}")
    print(f"Server will be available at: http://0.0.0.0:{args.port}")
    print("\nExample curl request:")
    print("------------------------------------------")
    print("curl -X POST http://0.0.0.0:" + str(args.port) + "/v1/chat/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print("     -d '{")
    print('  "model": "multimodal",')
    print('  "messages": [')
    print("    {")
    print('      "role": "user",')
    print('      "content": [')
    print("        {")
    print('          "type": "image",')
    print(
        '          "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llama2-architecture.png"'
    )
    print("        },")
    print("        {")
    print('          "type": "text",')
    print('          "text": "Explain what this image shows in detail."')
    print("        }")
    print("      ]")
    print("    }")
    print("  ],")
    print('  "temperature": 0.8,')
    print('  "max_tokens": 500')
    print("}'")
    print("------------------------------------------")
    print("\nUse Ctrl+C to stop the server.\n")

    # Start the server
    ed.vInferenceApiServer(inference, max_workers=1).fire(port=args.port)


if __name__ == "__main__":
    main()
