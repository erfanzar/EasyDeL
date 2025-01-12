import os
import sys

os.environ["LOGGING_LEVEL_ED"] = "DEBUG"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import easydel as ed
import transformers
import jax
from jax import numpy as jnp


def main():
	if jax.device_count() > 4:
		sharding_axis_dims = (1, 1, 1, -1)
	else:
		sharding_axis_dims = (1, 1, 1, -1)

	max_length = 4096

	# if jax.default_backend() == "gpu":
	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
	# else:
	# 	pretrained_model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"

	dtype = jnp.bfloat16
	extra = {}
	if jax.default_backend() == "gpu":
		import torch

		extra = {"torch_dtype": torch.float16}
		param_dtype = jnp.float8_e5m2
		if os.environ.get("APPED_LORA_TEST", "false") in ["true", "yes"]:
			param_dtype = jnp.bfloat16
	else:
		param_dtype = jnp.bfloat16

	partition_axis = ed.PartitionAxis()
	tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	tokenizer.padding_side = "left"
	tokenizer.pad_token_id = tokenizer.eos_token_id
	print("TOKENIZER LOADED")
	print("LOADING MODEL ... ")
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
		param_dtype=param_dtype,
		dtype=dtype,
		partition_axis=partition_axis,
		precision=jax.lax.Precision("fastest"),
		**extra,
	)
	print("MODEL LOADED")

	if os.environ.get("APPED_LORA_TEST", "false") in ["true", "yes"]:
		model = model.apply_lora_to_layers(32, ".*(q_proj|k_proj).*")
	print("CREATING vInference")

	inference = ed.vInference(
		model=model,
		processor_class=tokenizer,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=1024,
			temperature=0.0,
			do_sample=False,
			top_p=0.95,
			top_k=10,
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=32,
		),
	)

	print(model.model_task)
	print(model.model_type)
	if os.environ.get("USE_AOT", "true") in ["true", "yes", "on", "1"]:
		print("Compiling")
		inference.precompile(1, inference.model_prefill_length)
		print("Done Compiling")
	messages = [
		{
			"role": "system",
			"content": "You are a helpful AI assistant.",
		},
		{
			"role": "user",
			"content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
		},
		{
			"role": "assistant",
			"content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
		},
		{
			"role": "user",
			"content": "What about solving an 2x + 3 = 7 equation?",
		},
	]
	ed.utils.helpers.get_logger(name=__name__).info("Applying Chat Template")
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
