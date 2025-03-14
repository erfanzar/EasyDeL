import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import easydel as ed
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding
from transformers import AutoTokenizer

import jax

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 4096
	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
	partition_axis = ed.PartitionAxis()
	dtype = jnp.bfloat16
	model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		auto_shard_model=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			attn_dtype=dtype,
			attn_softmax_dtype=jnp.float32,
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
			temperature=model.generation_config.temperature,
			top_p=model.generation_config.top_p,
			top_k=model.generation_config.top_k,
			eos_token_id=model.generation_config.eos_token_id,
			pad_token_id=model.generation_config.pad_token_id,
			bos_token_id=model.generation_config.bos_token_id,
			streaming_chunks=64,
			num_return_sequences=1,
		),
	)

	inference.precompile(
		ed.vInferencePreCompileConfig(
			batch_size=1,
			prefill_length=[1024],
		)
	)
	print(inference.inference_name)
	ed.vInferenceApiServer(inference).fire()


if __name__ == "__main__":
	main()


# curl -X POST http://0.0.0.0:11556/v1/chat/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#   "model": "$MODEL_ID",
#   "messages": [
#     {
#       "role": "user",
#       "content": "hi"
#     }
#   ],
#   "function_call": "none",
#   "temperature": 1,
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
