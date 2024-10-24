import asyncio
import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["EASYDEL_AUTO"] = "1"
# os.environ["EKERNEL_OPS"] = "true"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))


import os
import time

import easydel as ed
import jax
from huggingface_hub import HfApi
from jax import lax, sharding
from jax import numpy as jnp
from transformers import AutoTokenizer

PartitionSpec, api = sharding.PartitionSpec, HfApi()


async def main():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 6144
	num_devices = len(jax.devices())
	input_shape = (num_devices, max_length)
	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
	dtype = jnp.float16
	partition_axis = ed.PartitionAxis()
	model, params = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		input_shape=input_shape,
		auto_shard_params=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=dict(
			use_scan_mlp=False,
			partition_axis=partition_axis,
			attn_dtype=jnp.float16,
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			block_q=32,
			block_k=128,
			attn_mechanism=ed.AttentionMechanisms.FLASH_ATTN2,
		),
		# quantization_method="8bit",
		quantization_platform="jax",
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
		params=params,
		tokenizer=tokenizer,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=1024,
			temperature=model.generation_config.temperature,
			top_p=model.generation_config.top_p,
			top_k=model.generation_config.top_k,
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=32,
		),
	)
	await inference.async_precompile(1)
	print(inference.inference_name)
	conversation = []
	while True:
		conversation.append({"role": "user", "content": input("USER > ")})
		ids = tokenizer.apply_chat_template(
			conversation,
			return_tensors="np",
			return_dict=True,
			max_length=inference.model_prefill_length,
			padding="max_length",
			add_generation_prompt=True,
		)

		start = time.time()
		input_ids, attention_mask = ids["input_ids"], ids["attention_mask"]
		start_length = inference.model_prefill_length
		pad_seq = inference.model_prefill_length
		print("ASSISTANT > ", end="")
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
				tokenizer.decode(
					response.sequences[0][next_slice],
					skip_special_tokens=True,
				),
				end="",
			)

		print()
		end = time.time()
		final_response = tokenizer.decode(
			response.sequences[0][start_length:pad_seq],
			skip_special_tokens=True,
		)
		conversation.append({"role": "user", "content": final_response})
		print(inference.count_tokens(conversation))
		print(
			"TPS :",
			sum(response.sequences[0][input_ids.shape[-1] :] != tokenizer.eos_token_id)
			/ (end - start),
		)


if __name__ == "__main__":
	asyncio.run(main())
