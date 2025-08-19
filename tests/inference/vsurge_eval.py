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

from jax import lax
from jax import numpy as jnp
from lm_eval import evaluator
from transformers import AutoTokenizer

import easydel as ed


def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tasks = ["gsm8k"]
    max_decode_length = 2048
    max_prefill_length = 2048
    max_concurrent_decodes = 32

    max_length = max_prefill_length + max_decode_length

    processor = AutoTokenizer.from_pretrained(model_id)
    processor.pad_token_id = processor.eos_token_id
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.HIGH,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.SDPA,
            attn_dtype=jnp.bfloat16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,  # change this if u go OOM
        ),
        partition_axis=ed.PartitionAxis(kv_head_axis="tp"),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
    )

    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        prefill_lengths=[
            max_prefill_length // 8,
            max_prefill_length // 4,
            max_prefill_length // 2,
            max_prefill_length,
        ],
        max_concurrent_decodes=max_concurrent_decodes,
        max_prefill_length=max_prefill_length,
        max_length=max_length,
        vsurge_name="xe-0",
        verbose=False,
        seed=48,
    )

    runner = ed.vSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=max_length,
        max_new_tokens=max_decode_length,
    )
    try:
        print(f"Starting evaluation on tasks: {tasks}")
        results = evaluator.simple_evaluate(
            model=runner,
            tasks=tasks,
            num_fewshot=5,
            batch_size=max_concurrent_decodes,
            device="cpu",
        )

        with open("eval-result.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Summary of results:")
        for task, metrics in results["results"].items():
            print(f"{task}: {metrics}")

    finally:
        runner.stop()


if __name__ == "__main__":
    main()
