import os

import jax
import transformers
from jax import numpy as jnp

import easydel as ed


def main():
    sharding_axis_dims = (1, 1, 1, 1, -1)

    prefill_length = 4096
    max_new_tokens = 2048
    max_length = prefill_length + max_new_tokens

    pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"

    if jax.default_backend() == "gpu":
        dtype = param_dtype = jnp.float16
    else:
        dtype = param_dtype = jnp.bfloat16

    partition_axis = ed.PartitionAxis()
    processor = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    processor.padding_side = "left"
    processor.pad_token_id = processor.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
            attn_dtype=param_dtype,
            attn_softmax_dtype=jnp.float32,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=param_dtype,
        dtype=dtype,
        partition_axis=partition_axis,
        precision=jax.lax.Precision.DEFAULT,
    )

    if os.getenv("APPED_LORA_TEST", "false") in ["true", "yes"]:
        model = model.apply_lora_to_layers(32, ".*(q_proj|k_proj).*")
    sampling_params = ed.SamplingParams(
        max_tokens=max_new_tokens + 540,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.7,
        top_p=0.95,
        top_k=4,
        min_p=0.0,
    )
    print(sampling_params.sampling_type)
    inference = ed.vInference(
        model=model,
        processor_class=processor,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=max_new_tokens,
            eos_token_id=model.generation_config.eos_token_id,
            streaming_chunks=32,
            num_return_sequences=1,
            sampling_params=sampling_params,
        ),
    )

    print(model.model_task, model.model_type)

    inference.precompile(ed.vInferencePreCompileConfig(batch_size=1, prefill_length=prefill_length))

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "write long story about why you love EasyDeL"},
    ]

    inputs = processor.apply_chat_template(
        messages,
        return_tensors="jax",
        return_dict=True,
        add_generation_prompt=True,
    )

    print("Stage 1 => Start Generation Process.")
    for response in inference.generate(**inputs, sampling_params=sampling_params):  # noqa
        ...

    sequences = response.sequences[..., response.padded_length :]
    print(processor.batch_decode(sequences, skip_special_tokens=True)[0])

    print(response.tokens_per_second)

    print("Stage 2 => Start Generation Process.")
    sampling_params.top_p = 0.8
    for response in inference.generate(**inputs, sampling_params=sampling_params):  # noqa
        ...

    sequences = response.sequences[..., response.padded_length :]
    print(processor.batch_decode(sequences, skip_special_tokens=True)[0])

    print(response.tokens_per_second)


if __name__ == "__main__":
    main()
