import json
import time
import jax
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed
from datasets import load_dataset


async def main():
	pretrained_model_name_or_path = "Qwen/Qwen3-8B"
	dtype = param_dtype = jnp.bfloat16
	dataset = load_dataset("openai/gsm8k", "main", split="train")
	max_length = 8192
	prefill_length = 4096
	partition_axis = ed.PartitionAxis()
	processor = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	processor.padding_side = "left"
	processor.pad_token_id = processor.eos_token_id

	model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		auto_shard_model=True,
		sharding_axis_dims=(1, 1, -1, 1),
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
			attn_mechanism=ed.AttentionMechanisms.AUTO,
			decode_attn_mechanism=ed.AttentionMechanisms.REGRESSIVE_DECODE,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		param_dtype=param_dtype,
		dtype=dtype,
		partition_axis=partition_axis,
		precision=jax.lax.Precision.DEFAULT,
	) 
	max_concurrent_decodes = 8

	surge = ed.vSurge.create_vdriver(
		model=model,
		processor=processor,
		max_prefill_length=prefill_length,
		prefill_lengths=[prefill_length], 
		max_concurrent_decodes=max_concurrent_decodes,
		seed=877,
	)
	surge.start()
	surge.compile()

	non_streaming_prompts = dataset["question"][0:10]

	sampling_params = ed.SamplingParams(max_tokens=1024, temperature=0.2, top_p=0.95)

	start = time.time()
	final_results = await surge.generate(
		prompts=non_streaming_prompts,
		sampling_params=sampling_params,
		stream=False,
	)
	results = {}
	num_generated_tokens_overall = 0
	for i, result_list in enumerate(final_results):
		num_generated_tokens_overall += result_list.num_generated_tokens
		print(result_list.text)
		results[f"{i}"] = {
			"generated_content": result_list.text,
			"TPS": result_list.tokens_per_second,
			"num_generated_tokens": result_list.num_generated_tokens,
		}
	end = time.time()
	time_took = end - start
	over_all_tps = num_generated_tokens_overall / time_took
	print("over_all_tps", over_all_tps)
	print("num_generated_tokens_overall", num_generated_tokens_overall)
	print("time_took", time_took)
	json.dump(dict(results), open("results.json", "w"))
	surge.stop()


if __name__ == "__main__":
	import asyncio

	asyncio.run(main())
