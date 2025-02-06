import easydel as ed
from jax import numpy as jnp
import jax
from transformers import AutoTokenizer
from datasets import load_dataset

repo_id = "Qwen/Qwen2-0.5B-Instruct"

processing_class = AutoTokenizer.from_pretrained(repo_id)
model = ed.AutoEasyDeLModelForSequenceClassification.from_pretrained(
	repo_id,
	auto_shard_model=True,
	sharding_axis_dims=(1, 1, 1, -1),
	config_kwargs=ed.EasyDeLBaseConfigDict(
		freq_max_position_embeddings=2048,
		mask_max_position_embeddings=2048,
		attn_dtype=jnp.bfloat16,
		attn_softmax_dtype=jnp.float32,
		gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
		kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
		attn_mechanism=ed.AttentionMechanisms.VANILLA,
	),
	quantization_method=ed.EasyDeLQuantizationMethods.NONE,
	platform=ed.EasyDeLPlatforms.JAX,
	param_dtype=jnp.bfloat16,
	dtype=jnp.bfloat16,
	precision=jax.lax.Precision.HIGH,
	partition_axis=ed.PartitionAxis(),
)

model.config.pad_token_id = processing_class.pad_token_id

dataset = load_dataset("trl-lib/ultrafeedback_binarized")

trainer = ed.RewardTrainer(
	model=model,
	processing_class=processing_class,
	arguments=ed.RewardConfig(
		max_length=1024,
		max_sequence_length=1024,
		num_train_epochs=1,
		total_batch_size=8,
		log_steps=1,
		do_last_save=True,
		use_wandb=True,
		save_optimizer_state=False,
		progress_bar_type="json",
		save_steps=1000,
		save_total_limit=1,
	),
	train_dataset=dataset["train"],
	eval_dataset=None,
)
trainer.train()
