import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import easydel as ed

import jax
from jax import numpy as jnp
from transformers import AutoProcessor, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


def main():
	sharding_axis_dims = (1, 1, -1, 1)

	prefill_length = 2048
	max_new_tokens = 128

	max_length = max_new_tokens + prefill_length
	pretrained_model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"

	min_pixels = 256 * 28 * 28
	max_pixels = min_pixels
	resized_height, resized_width = 420, 420

	dtype = jnp.float16
	param_dtype = jnp.bfloat16
	partition_axis = ed.PartitionAxis()

	processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(
		pretrained_model_name_or_path,
		min_pixels=min_pixels,
		max_pixels=max_pixels,
		resized_height=resized_height,
		resized_width=resized_width,
	)
	processor.padding_side = "left"
	processor.eos_token_id = processor.tokenizer.eos_token_id
	processor.pad_token_id = processor.tokenizer.pad_token_id

	model: ed.Gemma3ForConditionalGeneration = (
		ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
			pretrained_model_name_or_path,
			auto_shard_model=True,
			sharding_axis_dims=sharding_axis_dims,
			config_kwargs=ed.EasyDeLBaseConfigDict(
				freq_max_position_embeddings=max_length,
				mask_max_position_embeddings=max_length,
				kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
				gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
				attn_dtype=jnp.bfloat16,
				attn_mechanism=ed.AttentionMechanisms.VANILLA,
			),
			quantization_method=ed.EasyDeLQuantizationMethods.NONE,
			param_dtype=param_dtype,
			dtype=dtype,
			partition_axis=partition_axis,
			precision=jax.lax.Precision.DEFAULT,
		)
	)
	messages = [
		{"role": "system", "content": "You are a helpful AI assistant."},
		{
			"role": "user",
			"content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
		},
		{
			"role": "assistant",
			"content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey.",
		},
		# {"role": "user", "content": "Can you code?"},
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": "https://picsum.photos/seed/picsum/200/300",
					"min_pixels": min_pixels,
					"max_pixels": max_pixels,
					"resized_height": resized_height,
					"resized_width": resized_width,
				},
				{"type": "text", "text": "what are these metrics indication exacly"},
			],
		},
	]
	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=[processor.apply_chat_template(messages, add_generation_prompt=True)],
		images=image_inputs,
		videos=video_inputs,
		max_length=prefill_length,
		padding="max_length",
		return_tensors="jax",
	)
	inference = ed.vInference(
		model=model,
		processor_class=processor,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=max_new_tokens,
			sampling_params=ed.SamplingParams(
				max_tokens=max_new_tokens,
				temperature=0.8,
				top_p=0.95,
				top_k=10,
			),
			eos_token_id=model.generation_config.eos_token_id,
			streaming_chunks=32,
			num_return_sequences=1,
		),
	)
	print("Start Generation Process.")
	print(inputs.keys())
	print(inputs["input_ids"].shape)
	for response in inference.generate(**inputs):
		...
	print(
		processor.batch_decode(
			response.sequences[..., response.padded_length :],
			skip_special_tokens=True,
		)[0]
	)
	print("\n" + ("*" * 5))
	print("TPS  :", response.tokens_pre_second)
	print("Loss :", model.compute_loss(**inputs)[-1].loss)  # 21.4800


if __name__ == "__main__":
	main()
