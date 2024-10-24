import functools
import os
import sys

import fjformer
import jax.experimental

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../..",
	)
)
import jax

jax.config.update("jax_platform_name", "cpu")  # CPU Test !

import flax.core
from datasets import load_dataset
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from transformers import AutoTokenizer

from easydel import FlaxMistralForCausalLM, MistralConfig, TrainingArguments
from easydel.trainers import conversations_formatting_function
from easydel.trainers.supervised_fine_tuning_trainer import SFTTrainer


def main():
	sequence_length = 128
	config = MistralConfig(
		hidden_size=128,
		num_attention_heads=8,
		num_key_value_heads=4,
		num_hidden_layers=4,
		intermediate_size=256,
		gradient_checkpointing="",
		max_position_embeddings=sequence_length,
	)

	tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

	def prompter(sample):
		return [
			conversations_formatting_function(tokenizer, messages_field="messages")(sample)
		]

	train_dataset = load_dataset("HuggingFaceH4/deita-10k-v0-sft", split="train_sft")

	model = FlaxMistralForCausalLM(config=config, _do_init=True)
	params = model.shard_params(model.params)

	def _sh(*a, **kw):
		return (
			("model/embed_tokens/embedding", PartitionSpec(("fsdp", "sp"))),
			(
				"self_attn/(q_proj|k_proj|v_proj)/kernel",
				PartitionSpec(("fsdp", "sp")),
			),
			("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "sp"))),
			("mlp/gate_proj/kernel", PartitionSpec(("fsdp", "sp"))),
			("mlp/down_proj/kernel", PartitionSpec(("fsdp", "sp"))),
			("mlp/up_proj/kernel", PartitionSpec(("fsdp", "sp"))),
			("input_layernorm/kernel", PartitionSpec(None)),
			("post_attention_layernorm/kernel", PartitionSpec(None)),
			("model/norm/kernel", PartitionSpec(None)),
			("lm_head/kernel", PartitionSpec(("fsdp", "sp"))),
			(".*", PartitionSpec(("fsdp", "sp"))),
		)

	model.config.get_partition_rules = _sh
	dtype = jnp.float32
	trainer = SFTTrainer(
		arguments=TrainingArguments(
			model_name="SFTTrainer_TEST",
			num_train_epochs=3,
			total_batch_size=2,
			gradient_accumulation_steps=2,
			use_wandb=False,
			model_class=type(model),
			do_train=True,
			do_eval=False,
			max_sequence_length=sequence_length,
			configs_to_initialize_model_class={
				"config": model.config,
				"input_shape": (1, 1),
				"dtype": dtype,
				"param_dtype": dtype,
			},
			dtype=dtype,
			param_dtype=dtype,
			track_memory=False,
			learning_rate=5e-4,
			label_smoothing_factor=0.1,
			z_loss=0.0001,
			train_on_inputs=True,
			save_steps=500,
			save_total_limit=1,
			do_last_save=False,
			pruning_module=fjformer.jaxpruner.MagnitudePruning(
				sparsity_distribution_fn=functools.partial(
					fjformer.jaxpruner.sparsity_distributions.uniform, sparsity=0.4
				),
				scheduler=fjformer.jaxpruner.sparsity_schedules.OneShotSchedule(0),
			),
			sparsify_module=True,
			sparse_module_type="bcsr",
		),
		train_dataset=train_dataset,
		eval_dataset=None,  # we don't have eval dataset rn :)
		tokenizer=tokenizer,
		dataset_text_field=None,
		formatting_func=prompter,
		packing=True,
		num_of_sequences=1024,
		chars_per_token=2.1,
		dataset_num_proc=32,
	)
	return trainer.train(model_parameters=flax.core.FrozenDict({"params": params}))


if __name__ == "__main__":
	res = main()
