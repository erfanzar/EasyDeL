# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

import json
import typing as tp

import jax
from eformer.aparser import DataClassArgumentParser
from eformer.pytree import auto_pytree, field
from jax import numpy as jnp
from lm_eval import evaluator
from transformers import AutoTokenizer

import easydel as ed
from easydel.inference.evals import vSurgeLMEvalAdapter


@auto_pytree
class EvaluationConfig:
    """
    Configuration class for EasyDeL model evaluation.

    This dataclass holds all the necessary parameters to configure and run
    model evaluations using the EasyDeL framework and lm-eval harness.

    Attributes:
        model (str): Model name or path to load for evaluation.
        tasks (List[str]): List of task names (from lm-eval) to evaluate the model on.
        runner_driver (tp.Literal["vdriver", "odriver"]): The driver type to use for the vSurge engine.
        attn_mechanism (ed.AttentionMechanisms): The attention mechanism configuration for the model.
        output (str): Path to save the evaluation results (in JSON format).
        sharding_axis (str | tuple): String representation or tuple of sharding dimensions for model parallelism.
        max_length (int): Maximum sequence length the model can handle.
        prefill_length (int): Maximum length for the prefill phase in the vSurge engine.
        page_size (int): Page size for the paged attention mechanism (if used).
        hbm_utilization (float): Fraction of HBM memory to utilize (relevant for oDriver).
        max_concurrent_decodes (int): Maximum number of concurrent decode operations in the vSurge engine.
        num_fewshot (int): Number of few-shot examples to use for tasks that support them.
    """

    model: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "Model name or path."},
    )
    tasks: list[str] = field(
        default_factory=lambda: ["gsm8k"],
        metadata={"help": "Tasks to evaluate on."},
    )

    runner_driver: tp.Literal["vdriver", "odriver"] = field(
        default="odriver",
        metadata={"help": "The driver to use for evaluation."},
    )

    attn_mechanism: ed.AttentionMechanisms = field(
        default=ed.AttentionMechanisms.AUTO,
        metadata={"help": "The attention mechanism to use."},
    )
    output: str = field(
        default="evaluation_results.json",
        metadata={"help": "Output file path."},
    )
    sharding_axis: str = field(
        default="1, 1, -1, 1, 1",
        metadata={"help": "The sharding axis."},
    )
    max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length."},
    )
    prefill_length: int = field(
        default=4096,
        metadata={"help": "Prefill length."},
    )
    page_size: int = field(
        default=128,
        metadata={"help": "Page size."},
    )
    hbm_utilization: float = field(
        default=0.875,
        metadata={"help": "HBM utilization."},
    )
    max_concurrent_decodes: int = field(
        default=64,
        metadata={"help": "Maximum concurrent decodes."},
    )
    num_fewshot: int = field(
        default=0,
        metadata={"help": "Number of few-shot examples."},
    )

    def __post_init__(self):
        """
        Post-initialization method to process configuration values.

        Specifically, this method converts the `sharding_axis` string
        representation into a tuple of integers if it is provided as a string.
        """
        if isinstance(self.sharding_axis, str):
            self.sharding_axis = tuple(map(int, self.sharding_axis.split(",")))


def main():
    """
    Main function to set up and run model evaluations using EasyDeL and lm-eval.

    This function performs the following steps:
    1. Parses evaluation configuration from command-line arguments.
    2. Determines the appropriate data type based on the JAX backend.
    3. Initializes the tokenizer and sets padding side and pad token ID.
    4. Loads the EasyDeL model with specified configurations and sharding.
    5. Creates a vSurge engine instance (vdriver or odriver) based on the configuration.
    6. Wraps the model and engine in an vSurgeLMEvalAdapter adapter for compatibility with lm-eval.
    7. Runs the evaluation using `lm_eval.evaluator.simple_evaluate` on the specified tasks.
    8. Saves the evaluation results to a JSON file.
    9. Prints a summary of the evaluation results.
    10. Ensures the vSurge engine is stopped upon completion or error.
    """
    parser = DataClassArgumentParser(EvaluationConfig)
    eval_config = parser.parse_args_into_dataclasses()[0]
    print(eval_config)
    print(f"Creating vSurgeLMEvalAdapter adapter for {eval_config.model}")
    if jax.default_backend() == "tpu":
        dtype = param_dtype = jnp.bfloat16
    else:
        dtype = param_dtype = jnp.float16

    partition_axis = ed.PartitionAxis()
    processor = AutoTokenizer.from_pretrained(eval_config.model)
    processor.padding_side = "left"
    if processor.pad_token_id is None:
        processor.pad_token_id = processor.eos_token_id

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        eval_config.model,
        auto_shard_model=True,
        sharding_axis_dims=eval_config.sharding_axis,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=eval_config.max_length,
            mask_max_position_embeddings=eval_config.max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            attn_dtype=param_dtype,
            attn_mechanism=eval_config.attn_mechanism,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=param_dtype,
        dtype=dtype,
        partition_axis=partition_axis,
        precision=jax.lax.Precision.DEFAULT,
    )

    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        max_prefill_length=eval_config.prefill_length,
        max_length=eval_config.max_length,
        prefill_lengths=[eval_config.prefill_length],
        max_concurrent_decodes=eval_config.max_concurrent_decodes,
    )

    model = vSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=eval_config.max_length,
        max_new_tokens=eval_config.max_length - eval_config.prefill_length,
    )

    try:
        print(f"Starting evaluation on tasks: {eval_config.tasks}")
        results = evaluator.simple_evaluate(
            model=model,
            tasks=eval_config.tasks,
            num_fewshot=eval_config.num_fewshot,
            batch_size=eval_config.max_concurrent_decodes,
            device="cpu",
        )

        with open(eval_config.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Evaluation results saved to {eval_config.output}")
        print("Summary of results:")
        for task, metrics in results["results"].items():
            print(f"{task}: {metrics}")

    finally:
        model.stop()

    return results


if __name__ == "__main__":
    main()
