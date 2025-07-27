# fmt:off



# fmt:on
import jax
import torch
from huggingface_hub import HfApi
from jax import numpy as jnp
from jax import sharding

import easydel as ed

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def save():
    sharding_axis_dims = (1, 1, 1, 1, -1)
    max_length = 6144

    pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    dtype = jnp.float16
    partition_axis = ed.PartitionAxis()

    dtype = jnp.float16

    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        auto_shard_model=True,
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_length,
            mask_max_position_embeddings=max_length,
            attn_dtype=jnp.float16,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        platform=ed.EasyDeLPlatforms.TRITON,
        param_dtype=dtype,
        dtype=dtype,
        torch_dtype=torch.float16,
        partition_axis=partition_axis,
        precision=jax.lax.Precision("fastest"),
    )

    model.eval()
    model = model.quantize(
        method=ed.EasyDeLQuantizationMethods.A8BIT,
        block_size=128,
        quantization_pattern=".*(gate_proj|up_proj).*",
    )
    model.save_pretrained("tmp-files/qq")


def load():
    model = ed.EasyDeLBaseModule.from_pretrained(
        "tmp-files/qq",
        model_task=ed.TaskType.CAUSAL_LM,
    )
    print(
        f"{model.config.quantization_method=}\n"
        f"{model.config.quantization_block_size=}\n"
        f"{model.config.quantization_pattern=}\n"
    )
    print("Loaded")


if __name__ == "__main__":
    save()
    load()
