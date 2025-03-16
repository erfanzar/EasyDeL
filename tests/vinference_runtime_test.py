import os
import sys

# os.environ["LOGGING_LEVEL_ED"] = "DEBUG"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
import transformers
from jax import numpy as jnp

import easydel as ed


def main():
	if jax.device_count() > 4:
		sharding_axis_dims = (1, 1, 1, -1)
	else:
		sharding_axis_dims = (1, 1, 1, -1)

	prefill_length = 4096
	max_new_tokens = 2048
	max_length = prefill_length + max_new_tokens

	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
	# pretrained_model_name_or_path = "google/gemma-3-1b-it"

	if jax.default_backend() == "gpu":
		dtype = jnp.float16
		param_dtype = jnp.bfloat16
		attn_kwargs = dict(
			attn_dtype=jnp.float16,
			attn_softmax_dtype=jnp.float32,
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		)
	else:
		dtype = jnp.bfloat16
		param_dtype = jnp.bfloat16
		attn_kwargs = dict(attn_mechanism=ed.AttentionMechanisms.AUTO)

	partition_axis = ed.PartitionAxis()
	processor = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
	processor.padding_side = "left"
	processor.pad_token_id = processor.eos_token_id

	model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		auto_shard_model=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			**attn_kwargs,
		),
		quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		param_dtype=param_dtype,
		dtype=dtype,
		partition_axis=partition_axis,
		precision=jax.lax.Precision.DEFAULT,
	)

	if os.environ.get("APPED_LORA_TEST", "false") in ["true", "yes"]:
		model = model.apply_lora_to_layers(32, ".*(q_proj|k_proj).*")

	inference = ed.vInference(
		model=model,
		processor_class=processor,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=max_new_tokens,
			temperature=0.7,
			do_sample=True,
			top_p=0.95,
			top_k=10,
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=32,
			num_return_sequences=1,
		),
	)

	print(model.model_task, model.model_type)

	inference.precompile(
		ed.vInferencePreCompileConfig(
			batch_size=1,
			prefill_length=prefill_length,
		)
	)

	messages = [
		{"role": "system", "content": "You are a helpful AI assistant."},
		{"role": "user", "content": "write 10 lines story about why you love EasyDeL"},
	]

	inputs = processor.apply_chat_template(
		messages,
		return_tensors="jax",
		return_dict=True,
		add_generation_prompt=True,
	)

	print("Start Generation Process.")
	for response in inference.generate(**inputs):
		...
	sequences = response.sequences[..., response.padded_length :]

	print(processor.batch_decode(sequences, skip_special_tokens=True)[0])
	print(response.tokens_pre_second)


if __name__ == "__main__":
	main()
