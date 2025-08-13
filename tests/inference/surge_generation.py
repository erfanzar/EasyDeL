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

import asyncio

from jax import lax
from jax import numpy as jnp
from transformers import AutoTokenizer

import easydel as ed


def main():
    prompts = [
        "USER:code fibo in c++\nASSIST:",
        "USER:code fibo in go\nASSIST:",
        "USER:code fibo in rust\nASSIST:",
    ]
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    max_decode_length = 2048
    max_prefill_length = 2048
    max_concurrent_decodes = 16

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
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
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
        vsurge_name="llama",
        verbose=True,
        seed=48,
    )

    surge.compile()
    surge.start()

    async def _run(vsg: ed.vSurge):
        context = await vsg.generate(
            prompts,
            ed.SamplingParams(max_tokens=32, top_p=0.95, temperature=0.7, n=3),
            stream=True,
        )
        conct = ""
        async for response in context:
            out = response[0]
            if out is not None:
                # print(out.text[0], flush=True, end="")
                conct += out.text[0]
        print(conct)
        response_1 = response[0]
        print(response_1.tokens_per_second[0], end="\n============\n")  # prompt 1 gen 1
        print(response_1.tokens_per_second[1], end="\n============\n")  # prompt 1 gen 2
        print(response_1.tokens_per_second[2], end="\n============\n")  # prompt 1 gen 3

    asyncio.run(_run(surge))
    exit(0)


if __name__ == "__main__":
    main()
