import re

from datasets import Dataset, load_dataset
from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer
import easydel as ed

repo_id = "qwen/qwen2-0.5b-instruct"
total_batch_size = 8
num_return_sequences = 8
max_prompt_length = 512
max_completion_length = 256
max_sequence_length = max_completion_length + max_prompt_length

processor = AutoTokenizer.from_pretrained(repo_id)

model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
	repo_id,
	sharding_axis_dims=(1, -1, 1, 1),
	auto_shard_model=True,
	dtype=jnp.bfloat16,
	param_dtype=jnp.bfloat16,
	precision=lax.Precision.DEFAULT,
	config_kwargs=ed.EasyDeLBaseConfigDict(
		attn_dtype=jnp.bfloat16,
		attn_softmax_dtype=jnp.float32,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
		freq_max_position_embeddings=max_sequence_length,
		mask_max_position_embeddings=max_sequence_length,
	),
	quantize_tensors=False,
	quantization_method=ed.EasyDeLQuantizationMethods.NONE,
)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
	answer = text.split("<answer>")[-1]
	answer = answer.split("</answer>")[0]
	return answer.strip()


def extract_hash_answer(text: str):
	if "####" not in text:
		return None
	return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
	data = load_dataset("openai/gsm8k", "main")[split]
	data = data.map(
		lambda x: {
			"prompt": [
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": x["question"]},
			],
			"answer": extract_hash_answer(x["answer"]),
		}
	)
	return data


def correctness_reward_func(prompts, completions, batch, **kwargs) -> list[float]:
	responses = [completion[0]["content"] for completion in completions]
	extracted_responses = [extract_xml_answer(r) for r in responses]
	answer = processor.batch_decode(batch["answer_ids"]) * num_return_sequences
	return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
	responses = [completion[0]["content"] for completion in completions]
	extracted_responses = [extract_xml_answer(r) for r in responses]
	return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
	"""Reward function that checks if the completion has a specific format."""
	pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
	responses = [completion[0]["content"] for completion in completions]
	matches = [re.match(pattern, r) for r in responses]
	return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
	"""Reward function that checks if the completion has a specific format."""
	pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
	responses = [completion[0]["content"] for completion in completions]
	matches = [re.match(pattern, r) for r in responses]
	return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
	count = 0.0
	if text.count("<reasoning>\n") == 1:
		count += 0.125
	if text.count("\n</reasoning>\n") == 1:
		count += 0.125
	if text.count("\n<answer>\n") == 1:
		count += 0.125
		count -= len(text.split("\n</answer>\n")[-1]) * 0.001
	if text.count("\n</answer>") == 1:
		count += 0.125
		count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
	return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
	contents = [completion[0]["content"] for completion in completions]
	return [count_xml(c) for c in contents]


arguments = ed.GRPOConfig(
	save_directory="/home/erfan/outputs",
	max_prompt_length=max_prompt_length,
	max_completion_length=max_completion_length,
	num_train_epochs=1,
	total_batch_size=total_batch_size,
	log_steps=10,
	use_wandb=False,
	report_steps=5,
	save_steps=100,
	progress_bar_type="json",
	save_optimizer_state=False,
	do_eval=False,
)

train_dataset = get_gsm8k_questions("train")
test_dataset = get_gsm8k_questions("test")
vinference = ed.vInference(
	model=model,
	processor_class=processor,
	generation_config=ed.vInferenceConfig(
		bos_token_id=processor.bos_token_id,
		eos_token_id=processor.eos_token_id,
		pad_token_id=processor.pad_token_id,
		do_sample=True,
		max_new_tokens=max_completion_length,
		streaming_chunks=max_completion_length,
		top_k=10,
		top_p=0.95,
		num_return_sequences=num_return_sequences,
	),
	seed=84,
)

vinference.precompile(total_batch_size, max_prompt_length)


def data_tokenize_fn(batch, tokenizer, tools):
	ids = tokenizer(
		batch["prompt"],
		return_tensors="np",
		padding="max_length",
		padding_side="left",
		max_length=arguments.max_prompt_length,
		truncation=True,
		add_special_tokens=False,
	)
	ans = tokenizer(
		batch["answer"],
		return_tensors="np",
		padding="max_length",
		padding_side="left",
		max_length=arguments.max_prompt_length,
		truncation=True,
		add_special_tokens=False,
		return_attention_mask=False,
	)
	ids.update({"answer_ids": ans["input_ids"]})
	return ids


trainer = ed.GRPOTrainer(
	model=model,
	reward_funcs=[
		xmlcount_reward_func,
		soft_format_reward_func,
		strict_format_reward_func,
		int_reward_func,
		correctness_reward_func,
	],
	processing_class=processor,
	eval_dataset=test_dataset,
	train_dataset=train_dataset,
	arguments=arguments,
	vinference=vinference,
	data_tokenize_fn=data_tokenize_fn,
)

trainer.train()
