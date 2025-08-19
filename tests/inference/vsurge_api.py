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


from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed
from easydel.inference.openai_api_modules import FunctionCallFormat


def main():
    # model_id = "Qwen/Qwen3-8B"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    max_decode_length = 2048
    max_prefill_length = 6144
    max_concurrent_decodes = 4

    max_length = max_prefill_length + max_decode_length

    processor = AutoTokenizer.from_pretrained(model_id)
    processor.pad_token_id = processor.eos_token_id
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_id,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision.DEFAULT,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.SDPA,
            attn_dtype=jnp.bfloat16,
            kvdtype=jnp.float8_e5m2,
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
        vsurge_name="surge",
        verbose=True,
        seed=48,
    )

    surge.compile()
    surge.start()
    ed.vSurgeApiServer(
        surge,
        enable_function_calling=True,
        default_function_format=FunctionCallFormat.OPENAI,
    ).run(port=8888)


if __name__ == "__main__":
    main()
