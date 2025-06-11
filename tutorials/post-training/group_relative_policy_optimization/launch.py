# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

import ray
from eformer.executor.ray import TpuAcceleratorConfig, execute

ray.init()


@execute(TpuAcceleratorConfig("v4-64"))
@ray.remote
def main():
    import easydel as ed  # noqa

    import jax
    from datasets import load_dataset
    from jax import numpy as jnp
    from math_verify import LatexExtractionConfig, parse, verify
    from transformers import AutoTokenizer

    model_id = "CohereLabs/aya-expanse-8b"
    processor = AutoTokenizer.from_pretrained(model_id)
    processor.padding_side = "left"

    total_batch_size = 8
    num_return_sequences = 4
    top_k = 50
    top_p = 0.95
    temperature = 0.7

    grpo_config = ed.GRPOConfig(
        total_batch_size=total_batch_size,
        max_prompt_length=2048,
        max_completion_length=2048,
        learning_rate=1e-6,
        learning_rate_end=6e-7,
        log_steps=5,
        report_steps=10,
        progress_bar_type="json",
        num_train_epochs=3,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        do_last_save=True,
        track_memory=False,
        save_steps=1000,
        save_total_limit=0,  # 0 means remove everything and then save
        save_optimizer_state=False,
        per_epoch_training_steps=None,
        per_epoch_evaluation_steps=None,
        use_wandb=True,
        clip_grad=1.0,
        weight_distribution_log_steps=100,
        warmup_steps=0,
        beta=0.04,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    max_prompt_length = grpo_config.max_prompt_length
    max_completion_length = grpo_config.max_completion_length
    max_sequence_length = max_completion_length + max_prompt_length

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        auto_shard_model=True,
        sharding_axis_dims=(1, -1, 1, 1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_sequence_length,
            mask_max_position_embeddings=max_sequence_length,
            attn_dtype=jnp.bfloat16,
            attn_softmax_dtype=jnp.bfloat16,
            kvdtype=jnp.bfloat16,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.AUTO,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        precision=jax.lax.Precision.DEFAULT,
        partition_axis=ed.PartitionAxis(),
    )

    def format_reward(completions, **kwargs):
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>.*?</think>.*?"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content) for content in completion_contents]
        rewards_list = [1.0 if match else 0.0 for match in matches]
        return rewards_list

    def accuracy_reward(prompts, completions, batch, **kwargs):
        """Reward function that checks if the completion is the same as the ground truth."""
        # solutions = kwargs["solution"]
        solutions = processor.batch_decode(batch["solution_ids"]) * num_return_sequences
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
        "process and answer are enclosed within <think> </think> and answer needs not tag tags, respectively, i.e., "
        "<think> think process here </think> answer here"
    )

    dataset_id = "AI-MO/NuminaMath-TIR"
    train_dataset, test_dataset = load_dataset(
        dataset_id,
        split=["train[:100%]", "test[:100%]"],
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
