# fmt:off
import os
import sys
import threading

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import time
from functools import partial

import flax
import flax.nnx

# fmt:on
import jax
import torch
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLProcessor
from transformers import Qwen2VLForConditionalGeneration as hfmodel_cond

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def log_mem():
	while True:
		ed.utils.analyze_memory.SMPMemoryMonitor(5).print_current_status()
		time.sleep(5)


threading.Thread(target=log_mem)  # .start()


def to_np(x):
	return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x


def print_errors(left, right, prefix="", n_diff=5):
	left = to_np(left)
	right = to_np(right)
	prefix = f"{prefix} - " if prefix else ""

	diff_mask = ~jnp.isclose(left, right, atol=0.125, rtol=0)
	diff_indices = jnp.where(diff_mask.reshape(-1))[0]
	n_fails = jnp.sum(diff_mask)
	correct_percentage = 100 * (1 - jnp.mean(diff_mask))
	mean_error = jnp.mean(left) - jnp.mean(right)
	is_passed = n_fails == 0
	print(f"\n{prefix}Comparison Results:")
	print("=" * (len(prefix) + 20))
	if n_fails > 0:
		print(f"\n{prefix}Found {n_fails} differences")
		print(f"{prefix}Showing first {min(n_diff, len(diff_indices))} differences:")
		for idx in diff_indices[:n_diff]:
			print(
				f"{prefix}  Index {idx:6d}: {left.reshape(-1)[idx]:10.6f} vs {right.reshape(-1)[idx]:10.6f} "
				f"(diff: {left.reshape(-1)[idx] - right.reshape(-1)[idx]:10.6f})"
			)
	print(f"\n{prefix}Summary Statistics:")
	print(f"{prefix}  Pass/Fail: {'✓ PASSED' if is_passed else '✗ FAILED'}")
	print(f"{prefix}  Correct Elements: {correct_percentage:6.2f}%")
	print(f"{prefix}  Mean Error: {mean_error:10.6f}")

	print(f"\n{prefix}Last 5 Elements Comparison:")
	print(f"{prefix}  Left:  {left.reshape(-1)[-5:]}")
	print(f"{prefix}  Right: {right.reshape(-1)[-5:]}")
	print("\n" + "=" * (len(prefix) + 20))


def comparing_torch_to_easydel():
	config = ed.Qwen2VLConfig(
		hidden_size=256,
		intermediate_size=256,
		num_hidden_layers=2,
		num_attention_heads=2,
		num_key_value_heads=2,
		rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]},
		vision_config=dict(
			depth=2,
			embed_dim=1280,
			hidden_size=3584,
			hidden_act="quick_gelu",
			mlp_ratio=4,
			num_heads=16,
			in_channels=3,
			patch_size=14,
			spatial_merge_size=2,
			temporal_patch_size=2,
		),
	)

	hfm = hfmodel_cond(config)
	_, _, transform_function = ed.get_modules_by_type(
		"qwen2_vl",
		ed.TaskType.IMAGE_TEXT_TO_TEXT,
	)
	model_tree = transform_function(
		state_dict=hfm.state_dict(),
		device=jax.devices("cpu")[0],
		remove_state_dict=True,
	)

	model = ed.Qwen2VLForConditionalGeneration.lazy_init(
		config=config,
		dtype=jnp.float16,
		param_dtype=jnp.float16,
		rngs=flax.nnx.Rngs(0),
	)

	model = ed.traversals.merge_model_and_tree(model, model_tree)
	print("model created")
	min_pixels = 6 * 28 * 28
	max_pixels = min_pixels
	processor = AutoProcessor.from_pretrained(
		"Qwen/Qwen2-VL-2B-Instruct",
		min_pixels=min_pixels,
		max_pixels=max_pixels,
	)
	messages = [
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": "https://picsum.photos/seed/picsum/200/300",
				},
				{"type": "text", "text": "what are these metrics indication exacly"},
			],
		},
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": "https://picsum.photos/seed/picsum/200/300",
				},
				{"type": "text", "text": "Check Rope"},
			],
		},
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": "https://picsum.photos/seed/picsum/200/300",
				},
				{"type": "text", "text": "Check Rope 2"},
			],
		},
	]
	text = processor.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)
	image_inputs, video_inputs = process_vision_info(messages)
	processor_kw = dict(
		images=image_inputs,
		videos=video_inputs,
		padding="max_length",
		max_length=2048,
	)

	@partial(jax.jit, static_argnames=model.static_arguments)
	def call(**kwr):
		return model(**kwr, return_dict=True)

	inputs = model.prepare_inputs_for_call(
		**processor(text=[text], **processor_kw, return_tensors="np"), drop_ids=False
	)
	hf_res = hfm(**processor(text=[text], **processor_kw, return_tensors="pt"))
	res = call(**inputs)
	hr = hf_res.logits
	er = res.logits
	print_errors(er, hr, "Result")
	# Excepted
	# Result - Comparison Results:
	# =============================

	# Result - Summary Statistics:
	# Result -   Pass/Fail: ✓ PASSED
	# Result -   Correct Elements: 100.00%
	# Result -   Mean Error:  -0.000000

	# Result - Last 5 Elements Comparison:
	# Result -   Left:  [-0.1342   0.1746  -0.2764  -0.05704  0.349  ]
	# Result -   Right: [-0.13453704  0.17414932 -0.27545053 -0.05645031  0.3492928 ]

	# =============================


def generate():
	model = hfmodel_cond.from_pretrained(
		"Qwen/Qwen2-VL-2B-Instruct",
		torch_dtype=torch.bfloat16,
		attn_implementation="sdpa",
		device_map="auto",
	)

	min_pixels = 256 * 28 * 28
	max_pixels = min_pixels
	processor = AutoProcessor.from_pretrained(
		"Qwen/Qwen2-VL-2B-Instruct",
		min_pixels=min_pixels,
		max_pixels=max_pixels,
	)

	# messages = [
	#   {
	#     "role": "user",
	#     "content": [
	#       {
	#         "type": "image",
	#         "image": "https://picsum.photos/seed/picsum/200/300",
	#       },
	#       {"type": "text", "text": "what are these metrics indication exacly"},
	#     ],
	#   }
	# ]
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
		{"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
	]

	text = processor.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=True,
	)

	image_inputs, video_inputs = process_vision_info(messages)
	inputs = processor(
		text=[text],
		images=image_inputs,
		videos=video_inputs,
		max_length=512,
		padding="max_length",
		return_tensors="pt",
	)
	inputs = inputs.to("cuda")
	for k, v in inputs.items():
		print(k, v.shape)
	start = time.time()
	generated_ids = model.generate(**inputs, max_new_tokens=64)
	generated_ids_trimmed = [
		out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
	]
	start = time.time() - start
	print(len(generated_ids_trimmed[0]) / start)
	output_text = processor.batch_decode(
		generated_ids_trimmed,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)
	print(output_text[0])


def easydel_generate():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 1024

	pretrained_model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
	dtype = jnp.float16
	partition_axis = ed.PartitionAxis()

	dtype = jnp.float16

	model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
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
		param_dtype=jnp.float16,  # float8_e4m3fn,
		dtype=dtype,
		torch_dtype=torch.float16,
		partition_axis=partition_axis,
		precision=jax.lax.Precision("fastest"),
	)

	min_pixels = 256 * 28 * 28
	max_pixels = min_pixels

	processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(
		pretrained_model_name_or_path,
		min_pixels=min_pixels,
		max_pixels=max_pixels,
	)
	processor.padding_side = "left"
	processor.eos_token_id = processor.tokenizer.eos_token_id
	processor.pad_token_id = processor.tokenizer.pad_token_id

	model.eval()
	inference = ed.vInference(
		model=model,
		processor_class=processor,
		generation_config=ed.vInferenceConfig(
			max_new_tokens=128,
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
	print("Compiling")
	inference.precompile(
		ed.vInferencePreCompileConfig(
			batch_size=1,
			prefill_length=inference.model_prefill_length,
		)
	)
	print("Done Compiling")

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
		# {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
		{
			"role": "user",
			"content": [
				{
					"type": "image",
					"image": "https://picsum.photos/seed/picsum/200/300",
				},
				{"type": "text", "text": "what are these metrics indication exacly"},
			],
		},
	]
	image_inputs, video_inputs = process_vision_info(messages)

	pad_seq = inference.model_prefill_length
	ids = processor(
		text=[processor.apply_chat_template(messages, add_generation_prompt=True)],
		images=image_inputs,
		videos=video_inputs,
		max_length=pad_seq,
		padding="max_length",
		return_tensors="jax",
	)
	for k, v in ids.items():
		print(k, type(v), v.shape if hasattr(v, "shape") else None)
	print("Start Generation Process.")
	with jax.profiler.trace("tmp-files/vinference"):
		for response in inference.generate(**ids):
			next_slice = slice(
				pad_seq,
				pad_seq + inference.generation_config.streaming_chunks,
			)
			pad_seq += inference.generation_config.streaming_chunks
			print(
				processor.decode(response.sequences[0][next_slice], skip_special_tokens=True),
				end="",
			)

		print()
		print(response.generated_tokens)
		print("TPS :", response.tokens_pre_second)


if __name__ == "__main__":
	comparing_torch_to_easydel()
	# generate()
	# easydel_generate()
