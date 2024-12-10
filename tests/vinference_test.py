import os
import sys

import transformers

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


import jax
import torch
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 6144

	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
	dtype = jnp.float16
	partition_axis = ed.PartitionAxis()

	dtype = jnp.float16

	model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		auto_shard_model=False,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			attn_dtype=jnp.float16,
			gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		platform=ed.EasyDeLPlatforms.TRITON,
		param_dtype=dtype,
		dtype=dtype,
		torch_dtype=torch.float16,
		partition_axis=partition_axis,
		precision=jax.lax.Precision("fastest"),
	)
	tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	tokenizer.padding_side = "left"
	tokenizer.pad_token_id = tokenizer.eos_token_id
	model.eval()
	model = model.shard_model()
	inference = ed.vInference(
		model=model,
		tokenizer=tokenizer,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=1024,
			temperature=0.1,
			top_p=model.generation_config.top_p,
			top_k=model.generation_config.top_k,
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=16,
		),
	)

	print(model.model_task)
	print(model.model_type)
	print("Compiling")
	inference.precompile(1, inference.model_prefill_length)
	print("Done Compiling")

	prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

	messages = [
		{
			"role": "system",
			"content": "Please reason step by step, and put your final answer within \\boxed{}. and give 3 different responses",
		},
		{"role": "user", "content": prompt},
	]

	ids = tokenizer.apply_chat_template(
		messages,
		return_tensors="jax",
		return_dict=True,
		max_length=inference.model_prefill_length,
		padding="max_length",
		add_generation_prompt=True,
	)

	pad_seq = inference.model_prefill_length

	print("Start Generation Process.")
	with jax.profiler.trace("tmp-files/vinference"):
		for response in inference.generate(**ids):
			next_slice = slice(
				pad_seq,
				pad_seq + inference.generation_config.streaming_chunks,
			)
			pad_seq += inference.generation_config.streaming_chunks
			print(
				tokenizer.decode(response.sequences[0][next_slice], skip_special_tokens=True),
				end="",
			)

		print()
		print(response.generated_tokens)
		print("TPS :", response.tokens_pre_second)


if __name__ == "__main__":
	main()
