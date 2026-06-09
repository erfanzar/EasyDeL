# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

from jax import numpy as jnp
from lm_eval import evaluator  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer

import easydel as ed
from easydel.inference.esurge.config import eSurgeContextConfig, eSurgeRuntimeConfig


def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tasks = ["gsm8k"]
    max_decode_length = 2048
    max_prefill_length = 2048
    max_num_seqs = 32  # eSurge parameter for max concurrent sequences

    max_model_len = max_prefill_length + max_decode_length

    processor = AutoTokenizer.from_pretrained(model_id)
    processor.pad_token_id = processor.eos_token_id

    # Create eSurge instance for evaluation
    surge = ed.eSurge(
        model=model_id,
        processor=processor,
        loading_kwargs={"dtype": jnp.bfloat16},
        runtime=eSurgeRuntimeConfig.from_dict(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            compile_runner=True,
            esurge_name="esurge-eval",
            runner_verbose=False,
        ),
        context=eSurgeContextConfig.from_dict(
            reserve_tokens=800,
            auto_truncate_prompt=True,
            auto_cap_new_tokens=True,
        ),
    )

    # Create eSurge adapter
    runner = ed.eSurgeLMEvalAdapter(
        surge=surge,
        processor=processor,
        max_length=max_model_len,
        max_new_tokens=max_decode_length,
        batch_size=max_num_seqs,
    )

    try:
        print(f"Starting evaluation on tasks: {tasks}")
        results = evaluator.simple_evaluate(
            model=runner,
            tasks=tasks,
            num_fewshot=5,
            batch_size=max_num_seqs,
            device="cpu",
        )

        with open("esurge-eval-result.json", "w") as f:
            json.dump(results, f, indent=2)

        print("Summary of results:")
        for task, metrics in results["results"].items():
            print(f"{task}: {metrics}")

    finally:
        runner.stop()


if __name__ == "__main__":
    main()
