import asyncio

import jax
from huggingface_hub import HfApi
from jax import lax, sharding
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


async def main():
    sharding_axis_dims = (1, 1, 1, 1, -1)
    max_length = 6144
    num_devices = len(jax.devices())
    input_shape = (num_devices, max_length)
    pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    dtype = jnp.float16
    partition_axis = ed.PartitionAxis()
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        input_shape=input_shape,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            use_scan_mlp=False,
            partition_axis=partition_axis,
            attn_dtype=jnp.float16,
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
        ),
        quantization_method="8bit",
        platform="triton",
        partition_axis=partition_axis,
        param_dtype=dtype,
        dtype=dtype,
        precision=lax.Precision("fastest"),
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inference = ed.vInference(
        model=model,
        processor_class=tokenizer,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=1024,
            sampling_params=ed.SamplingParams(
                max_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                top_k=10,
            ),
            eos_token_id=model.generation_config.eos_token_id,
            streaming_chunks=64,
        ),
    )
    await inference.async_precompile(1)
    print(inference.inference_name)

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Generate a long random story"}],
        return_tensors="np",
        return_dict=True,
        max_length=inference.model_prefill_length,
        padding="max_length",
        add_generation_prompt=True,
    )

    input_ids, attention_mask = ids["input_ids"], ids["attention_mask"]
    pad_seq = inference.model_prefill_length
    with jax.profiler.trace("/tmp/tensorboard"):
        print("FIRST ATTEMPT 1")
        async for response in inference.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ):
            next_slice = slice(
                pad_seq,
                pad_seq + inference.generation_config.streaming_chunks,
            )
            pad_seq += inference.generation_config.streaming_chunks
            print(
                tokenizer.decode(response.sequences[0][next_slice], skip_special_tokens=True),
                end="",
            )

        print(f"\nTPS : {response.tokens_per_second}")

        print("FIRST ATTEMPT 2")
        async for response in inference.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ):
            next_slice = slice(
                pad_seq,
                pad_seq + inference.generation_config.streaming_chunks,
            )
            pad_seq += inference.generation_config.streaming_chunks
            print(
                tokenizer.decode(response.sequences[0][next_slice], skip_special_tokens=True),
                end="",
            )

        print(f"\nTPS : {response.tokens_per_second}")


if __name__ == "__main__":
    asyncio.run(main())
