import re
from dataclasses import field

import jax
from datasets import load_dataset
from eformer.aparser import DataClassArgumentParser
from eformer.pytree import auto_pytree
from jax import numpy as jnp
from math_verify import LatexExtractionConfig, parse, verify  # type:ignore
from transformers import AutoConfig, AutoTokenizer

import easydel as ed
from easydel.infra.factory import registry
from easydel.modules import *  # noqa # init


@auto_pytree
class RunTimeConfig:
    """
    Configuration class for runtime settings.

    Attributes:
        repo_id (str): The repository ID.
        processor_repo_id (str, optional): The repository ID for the processor. If None, defaults to repo_id.
        refrence_model_repo_id (str, optional): The repository ID for the reference model. If None, defaults to repo_id.
        sharding_axis (Tuple[int]): The sharding axis. Defaults to (1, -1, 1, 1, 1).
        attn_mechanism (ed.AttentionMechanisms): The attention mechanism to use. Defaults to
            ed.AttentionMechanisms.VANILLA.
        param_dtype (jnp.dtype): The data type for model parameters. Defaults to jnp.bfloat16.
        dtype (jnp.dtype): The data type for general computation. Defaults to jnp.bfloat16.
        attn_dtype (jnp.dtype): The data type for attention computation. Defaults to jnp.bfloat16.
        attn_softmax_dtype (jnp.dtype): The data type for attention softmax computation. Defaults to jnp.float32.
    """

    repo_id: str = field(
        metadata={"help": "The repository ID."},
    )

    processor_repo_id: str | None = field(
        default=None,
        metadata={"help": "The repository ID for the processor. If None, defaults to repo_id."},
    )
    kv_cache_quantization: ed.EasyDeLQuantizationMethods = field(default=ed.EasyDeLQuantizationMethods.NONE)

    dataset_use_rate: int = field(
        default=100,
        metadata={"help": "split in train or test dataset"},
    )

    sharding_axis: str = field(
        default="1, -1, 1, 1, 1",
        metadata={"help": "The sharding axis."},
    )
    attn_mechanism: ed.AttentionMechanisms = field(
        default=ed.AttentionMechanisms.AUTO,
        metadata={"help": "The attention mechanism to use."},
    )
    param_dtype: jnp.dtype = field(
        default=jnp.bfloat16,
        metadata={"help": "The data type for model parameters."},
    )
    dtype: jnp.dtype = field(
        default=jnp.bfloat16,
        metadata={"help": "The data type for general computation."},
    )
    attn_dtype: jnp.dtype = field(
        default=jnp.bfloat16,
        metadata={"help": "The data type for attention computation."},
    )
    attn_softmax_dtype: jnp.dtype = field(
        default=jnp.float32,
        metadata={"help": "The data type for attention softmax computation."},
    )

    def __post_init__(self):
        """Post-initialization to set dependent parameters."""
        if self.processor_repo_id is None:
            self.processor_repo_id = self.repo_id
        if isinstance(self.sharding_axis, str):
            self.sharding_axis = tuple(map(int, self.sharding_axis.split(",")))


parser = DataClassArgumentParser((ed.GRPOConfig, RunTimeConfig))
grpo_config, runtime_config = parser.parse_args_into_dataclasses()

runtime_config: RunTimeConfig
grpo_config: ed.GRPOConfig

if jax.process_index() == 0:
    print("Training Arguments\n----------------------")
    print(grpo_config)
    print("----------------------")


def main():
    processor = AutoTokenizer.from_pretrained(runtime_config.processor_repo_id)
    processor.padding_side = "left"

    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    max_prompt_length = grpo_config.max_prompt_length
    max_completion_length = grpo_config.max_completion_length
    max_sequence_length = max_completion_length + max_prompt_length

    hf_config = AutoConfig.from_pretrained(runtime_config.repo_id)

    avails = [v.module.__name__ for v in registry.task_registry[ed.TaskType.IMAGE_TEXT_TO_TEXT].values()]

    if hf_config.architectures and any(arch in avails for arch in hf_config.architectures):
        load_module = ed.AutoEasyDeLModelForImageTextToText
    else:
        load_module = ed.AutoEasyDeLModelForCausalLM

    model = load_module.from_pretrained(
        runtime_config.repo_id,
        auto_shard_model=True,
        sharding_axis_dims=runtime_config.sharding_axis,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=runtime_config.attn_dtype,
            attn_softmax_dtype=runtime_config.attn_softmax_dtype,
            kv_cache_quantization_method=runtime_config.kv_cache_quantization,
            attn_mechanism=runtime_config.attn_mechanism,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,  # change this if u go OOM
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=runtime_config.param_dtype,
        dtype=runtime_config.dtype,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        rewards_list = [1.0 if match else 0.0 for match in matches]
        return rewards_list

    def accuracy_reward(prompts, completions, batch, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        # solutions = kwargs["solution"]
        solutions = processor.batch_decode(batch["solution_ids"]) * grpo_config.num_return_sequences
        completion_contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for content, solution in zip(completion_contents, solutions, strict=False):
            gold_parsed = parse(
                solution,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            answer_parsed = parse(
                content,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) != 0:
                try:
                    rewards.append(float(verify(answer_parsed, gold_parsed)))
                except Exception:
                    rewards.append(0.0)
            else:
                rewards.append(1.0)
        return rewards

    SYSTEM_PROMPT = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant"
        " first thinks about the think process in the mind and then provides the user with the answer. The think "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> think process here </think><answer> answer here </answer>"
    )

    dataset_id = "AI-MO/NuminaMath-TIR"
    train_dataset, test_dataset = load_dataset(
        dataset_id,
        split=[
            f"train[:{runtime_config.dataset_use_rate}%]",
            f"test[:{runtime_config.dataset_use_rate}%]",
        ],
    )

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    train_dataset = train_dataset.map(make_conversation, remove_columns=["messages"])
    test_dataset = test_dataset.map(make_conversation, remove_columns=["messages"])

    def data_tokenize_fn(batch, tokenizer, tools):
        ids = tokenizer(
            batch["prompt"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",
            max_length=grpo_config.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        ans = tokenizer(
            batch["solution"],
            return_tensors="np",
            padding="max_length",
            padding_side="left",
            max_length=grpo_config.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        ids.update({"solution_ids": ans["input_ids"]})
        return ids

    trainer = ed.GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward],
        processing_class=processor,
        eval_dataset=test_dataset,
        train_dataset=train_dataset,
        arguments=grpo_config,
        data_tokenize_fn=data_tokenize_fn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
