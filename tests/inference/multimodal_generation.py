import os
# Disable Ray auto-initialization
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

import ray

# Initialize Ray explicitly with minimal settings
if not ray.is_initialized():
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level="ERROR",
        include_dashboard=False,
    )

import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

def main():
    # Setup parameters
    prefill_length = 2048
    max_new_tokens = 1024
    max_length = max_new_tokens + prefill_length
    model_name = "llava-hf/llava-1.5-7b-hf"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    processor.padding_side = "left"

    model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
        model_name,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
        ),
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    model.eval()

    # Prepare input with image and text
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        },
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        messages,
        return_tensors="jax",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )

    # Initialize inference
    inference = ed.vInference(
        model=model,
        processor_class=processor,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=max_new_tokens,
            sampling_params=ed.SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                top_k=10,
            ),
            eos_token_id=model.generation_config.eos_token_id,
            streaming_chunks=32,
            num_return_sequences=1,
        ),
    )

    # Precompile for specific dimensions to optimize performance
    inference.precompile(
        ed.vInferencePreCompileConfig(
            batch_size=1,
            prefill_length=prefill_length,
            vision_included=True,  # Important for vision models
            vision_batch_size=1,   # Number of images
            vision_channels=3,     # RGB channels
            vision_height=336,     # Image height
            vision_width=336,      # Image width
        )
    )

    # Generate response
    for response in inference.generate(**inputs):
        pass  # Process streaming tokens if needed

    # Get the final result
    result = processor.batch_decode(
        response.sequences[..., response.padded_length:],
        skip_special_tokens=True,
    )[0]
    print(result)

    # Shutdown Ray cleanly
    ray.shutdown()

if __name__ == "__main__":
    main()