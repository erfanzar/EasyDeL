import os
import sys

import jax
import torch

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["EASYDEL_AUTO"] = "true"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding
from transformers import AutoTokenizer

import easydel as ed

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
			gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		platform=ed.EasyDeLPlatforms.TRITON,
		param_dtype=dtype,  # jnp.float8_e5m2,
		dtype=dtype,
		torch_dtype=torch.float16,
		partition_axis=partition_axis,
		precision=jax.lax.Precision("fastest"),
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
		),
	)

	inference.precompile()
	print(inference.inference_name)
	ed.vInferenceApiServer({inference.inference_name: inference}).fire()


if __name__ == "__main__":
	main()
