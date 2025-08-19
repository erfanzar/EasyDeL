import jax
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding
from transformers import AutoTokenizer

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
    sharding_axis_dims = (1, 1, 1, 1, -1)
    max_length = 4096
    pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    partition_axis = ed.PartitionAxis()
    dtype = jnp.float16
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
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
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    inference = ed.vInference(
        model=model,
        processor_class=tokenizer,
        generation_config=ed.vInferenceConfig(
            max_new_tokens=2048,
            streaming_chunks=64,
            num_return_sequences=1,
        ),
        inference_name="LLaMA",
    )

    inference.precompile(ed.vInferencePreCompileConfig(batch_size=1, prefill_length=[1024]))
    print(inference.inference_name)
    ed.vInferenceApiServer(inference).run(port=11557)


if __name__ == "__main__":
    main()


# curl -X POST http://0.0.0.0:11557/v1/chat/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#   "model": "LLaMA",
#   "messages": [
#     {
#       "role": "user",
#       "content": "hi"
#     }
#   ],
#   "function_call": "none",
#   "temperature": 0.5,
#   "top_p": 1,
#   "n": 1,
#   "stream": false,
#   "stop": "string",
#   "max_tokens": 16,
#   "presence_penalty": 0,
#   "frequency_penalty": 0,
#   "logit_bias": {
#     "additionalProp1": 0,
#     "additionalProp2": 0,
#     "additionalProp3": 0
#   },
#   "user": "string"
# }'
