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

import easydel as ed


def main():
    tokenizer_id = model_id = "Qwen/Qwen3.5-27B"
    max_concurrent_decodes = 32
    max_length = 2**16

    elm = ed.eLargeModel(
        {
            "model": {
                "name_or_path": model_id,
                "tokenizer": tokenizer_id,
                "task": "auto-bind",
            },
            "loader": {
                "dtype": "bfloat16",
                "param_dtype": "bfloat16",
                "precision": "fastest",
                "verbose": True,
            },
            "sharding": {
                "axis_dims": (1, 1, 1, -1, 1),
                "axis_names": ("dp", "fsdp", "ep", "tp", "sp"),
                "auto_shard_model": True,
            },
            "base_config": {
                "values": {
                    "freq_max_position_embeddings": max_length,
                    "mask_max_position_embeddings": max_length,
                    "attn_mechanism": ed.AttentionMechanisms.AUTO,
                    "attn_dtype": "bf16",
                    "gradient_checkpointing": ed.EasyDeLGradientCheckPointers.NONE,
                    "moe_method": ed.MoEMethods.FUSED_MOE,
                }
            },
            "esurge": {
                "runtime": {
                    "max_model_len": max_length,
                    "max_num_seqs": max_concurrent_decodes,
                    "min_input_pad": 8,
                    "max_num_batched_tokens": 2048,
                    "use_aot_forward": True,
                    "runner_verbose": False,
                },
                "cache": {
                    "hbm_utilization": 0.9,
                    "page_size": 128,
                    "enable_prefix_caching": True,
                    "data_parallelism_axis": "fsdp",
                },
            },
            "eval": {"max_new_tokens": 8192},
        }
    )
    esurge = elm.build_esurge()

    for out in esurge.chat(
        [{"role": "user", "content": "code fibo in c"}],
        sampling_params=ed.SamplingParams(max_tokens=2048),
        stream=True,
    ):
        if out.delta_text is not None:
            print(f"\033[1;38;5;78m{out.delta_text}", end="")

        if out.delta_reasoning_content is not None:
            print(f"\033[1;38;5;99m{out.delta_reasoning_content}", end="")
    print(f"\n\nTPS:{out.tokens_per_second}")


if __name__ == "__main__":
    main()
