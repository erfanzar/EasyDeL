import inspect
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

import time
from typing import Callable, List

from easydel import (
	FlaxLlamaForCausalLM,
	GenerationPipeline,
	GenerationPipelineConfig,
	LlamaConfig,
)
from easydel.modules.flax_modeling_utils import quantize_params
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer


def get_bool_inputs(func: Callable) -> List[str]:
	"""
	Inspect a function and return a list of its boolean input variables.

	Args:
	    func (Callable): The function to inspect.

	Returns:
	    List[str]: A list of parameter names that are annotated as bool.
	"""
	signature = inspect.signature(func)
	bool_params = []

	for param_name, param in signature.parameters.items():
		if param.annotation is bool:
			bool_params.append(param_name)

	return bool_params


def main():
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
	tokenizer.padding_side = "left"
	tokenizer.pad_token = tokenizer.eos_token
	max_position_embeddings = 512
	config = LlamaConfig(
		hidden_size=512,
		intermediate_size=1024,
		num_hidden_layers=4,
		max_position_embeddings=max_position_embeddings + 128,
		use_scan_mlp=False,
		axis_dims=(1, -1, 1, 1),
		quantize_kv_cache=True,
		q_block=32,
		k_block=32,
		pallas_runtime=True,
		# attn_mechanism="pallas_flash"
	)
	model = FlaxLlamaForCausalLM(
		config=config,
		dtype=jnp.float16,
		param_dtype=jnp.float16,
		precision=lax.Precision("fastest"),
		input_shape=(2, 2),
		_do_init=True,
		seed=81,
	)
	tokenizer.padding_side = "left"
	tokens = tokenizer(
		"SOME TEXT",
		return_tensors="np",
		max_length=max_position_embeddings,
		padding="max_length",
	)
	input_ids = tokens["input_ids"]
	attention_mask = tokens["attention_mask"]
	params = quantize_params(model.params, method="8bit") if False else model.params
	pipeline = GenerationPipeline(
		model=model,
		params=params,
		tokenizer=tokenizer,
		generation_config=GenerationPipelineConfig(
			max_new_tokens=128,
			temprature=0.8,
			top_p=0.95,
			top_k=10,
			eos_token_id=23070,
			length_penalty=1.2,
			repetition_penalty=1.2,
		),
		# parameters_are_quantized=True,
	)

	time_start = None
	for gen, token in enumerate(pipeline.generate(input_ids, attention_mask)):  # noqa: B007
		if time_start is None:
			time_start = time.time()
		print(token, end="")
	print("\nTPS : %f \n" % ((gen + 1) / (time.time() - time_start)))
	# print("\n")
	# print("*" * 50)
	# for token in pipeline.generate(input_ids, attention_mask):
	# 	print(token, end="")


if __name__ == "__main__":
	main()
