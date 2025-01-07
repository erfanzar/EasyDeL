import os
import sys

import transformers

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
from jax import numpy as jnp

import easydel as ed


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
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		),
		platform=ed.EasyDeLPlatforms.JAX,
		param_dtype=dtype,
		dtype=dtype,
		partition_axis=partition_axis,
		precision=jax.lax.Precision("fastest"),
	)

	tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	print(model)
	ids = tokenizer.encode("this might be a test? But I", return_tensors="np")
	output = model(ids)
	next_token = jnp.argmax(jax.nn.softmax(output.logits[0, -1, :])).reshape(1, 1)
	print(output.logits)
	print(next_token)
	print(tokenizer.decode(jnp.concatenate([ids, next_token], axis=-1)[0]))
	# <|begin_of_text|>this might be a test? But I'm


if __name__ == "__main__":
	main()
