from typing import Optional, Sequence, Tuple

import jax
from easydel import AutoEasyDeLModelForCausalLM, PartitionAxis


def load_model(
        pretrained_model_name_or_path: str,
        device=jax.devices('cpu')[0],  # Device to be used in order to Load Model on (Offload device)
        dtype: jax.numpy.dtype = jax.numpy.float32,
        param_dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: Optional[jax.lax.Precision] = None,
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        partition_axis: PartitionAxis = PartitionAxis(),
        shard_attention_computation: bool = True,
        input_shape: Tuple[int, int] = (1, 1),
        backend: Optional[str] = None,
):
    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        sharding_axis_names=sharding_axis_names,
        sharding_axis_dims=sharding_axis_dims,
        partition_axis=partition_axis,
        shard_attention_computation=shard_attention_computation,
        input_shape=input_shape,
        backend=backend

    )
