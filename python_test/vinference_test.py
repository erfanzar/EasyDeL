import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))


import jax

import os
import time

import easydel as ed
from huggingface_hub import HfApi
from jax import lax, sharding
from jax import numpy as jnp
from transformers import AutoTokenizer

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 8192
	num_devices = len(jax.devices())
	input_shape = (num_devices, max_length)
	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"

	dtype = jnp.float16
	partition_axis = ed.PartitionAxis()
	model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		input_shape=input_shape,
		device_map="auto",
		auto_shard_params=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=dict(
			use_scan_mlp=False,
			partition_axis=partition_axis,
			attn_dtype=jnp.float16,
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			q_block=64,
			k_block=128,
			attn_mechanism=ed.AttentionMechanisms.flash_attn2,
			quantize_kv_cache=True,
		),
		platform="triton",
		quantization_method="8bit",
		partition_axis=partition_axis,
		param_dtype=dtype,
		dtype=dtype,
		precision=lax.Precision("fastest"),
	)
	tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

	tokenizer.padding_side = "left"
	tokenizer.pad_token_id = tokenizer.eos_token_id
	infernece = ed.vInference(
		model=model,
		params=params,
		tokenizer=tokenizer,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=512,
			temperature=model.generation_config.temperature,
			top_p=model.generation_config.top_p,
			top_k=model.generation_config.top_k,
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=32,
		),
	)
	ids = tokenizer.apply_chat_template(
		[{"role": "user", "content": "COMP"}],
		return_tensors="np",
		return_dict=True,
		max_length=1024,
		padding="max_length",
	)
	infernece.precompile(*ids["input_ids"].shape)
	while True:
		ids = tokenizer.apply_chat_template(
			[{"role": "user", "content": input("> ")}],
			return_tensors="np",
			return_dict=True,
			max_length=1024,
			padding="max_length",
			add_generation_prompt=True,
		)

		start = time.time()

		input_ids, attention_mask = ids["input_ids"], ids["attention_mask"]
		pad_seq = input_ids.shape[-1]

		for response in infernece.generate(
			input_ids=input_ids, attention_mask=attention_mask
		):
			next_slice = slice(
				pad_seq, pad_seq + infernece.generation_config.streaming_chunks
			)
			pad_seq += infernece.generation_config.streaming_chunks
			print(
				tokenizer.decode(response.sequences[0][next_slice], skip_special_tokens=True),
				end="",
			)
		print()
		end = time.time()

		print(
			"TPS :",
			sum(response.sequences[0][input_ids.shape[-1] :] != tokenizer.eos_token_id)
			/ (end - start),
		)


if __name__ == "__main__":
	main()
