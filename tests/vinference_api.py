import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["EASYDEL_AUTO"] = "true"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding
from transformers import AutoTokenizer

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 2048
	pretrained_model_name_or_path = "EasyDeL/EasyDeL-Llama-3.2-1B-Instruct"
	dtype = jnp.float16
	partition_axis = ed.PartitionAxis()
	model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		input_shape=(len(jax.devices()), max_length),
		auto_shard_params=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=dict(
			use_scan_mlp=False,
			partition_axis=partition_axis,
			attn_dtype=jnp.float16,
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.A8BIT,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.A8BIT,
		platform=ed.EasyDeLPlatforms.TRITON,
		partition_axis=partition_axis,
		param_dtype=dtype,
		dtype=dtype,
	)
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	tokenizer.padding_side = "left"
	tokenizer.pad_token_id = tokenizer.eos_token_id
	inference = ed.vInference(
		model=model,
		params=params,
		tokenizer=tokenizer,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=256,
			temperature=model.generation_config.temperature,
			top_p=model.generation_config.top_p,
			top_k=model.generation_config.top_k,
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=32,
		),
		inference_name="llama3-1B",
	)

	inference.precompile()
	print(inference.inference_name)
	ed.vInferenceApiServer({inference.inference_name: inference}).fire()


if __name__ == "__main__":
	main()
