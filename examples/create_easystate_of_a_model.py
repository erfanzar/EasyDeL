from easydel import EasyDeLState
import jax
from jax.sharding import PartitionSpec
from typing import Sequence, Optional


def load_model(
        pretrained_model_name_or_path: str,
        device=jax.devices('cpu')[0],  # Device to be used in order to Load Model on (Offload device)
        dtype: jax.numpy.dtype = jax.numpy.float32,
        param_dtype: jax.numpy.dtype = jax.numpy.float32,
        precision: Optional[jax.lax.Precision] = jax.lax.Precision("fastest"),
        sharding_axis_dims: Sequence[int] = (1, -1, 1, 1),
        sharding_axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
        query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        generation_query_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "tp", None, None),
        key_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        value_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
        generation_bias_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), None, None, None),
        attention_partition_spec: PartitionSpec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
        shard_attention_computation: bool = True,
        input_shape: Sequence[int] = (1, 1),
        backend: Optional[str] = None,
):
    state = EasyDeLState.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        device=device,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        sharding_axis_names=sharding_axis_names,
        sharding_axis_dims=sharding_axis_dims,
        query_partition_spec=query_partition_spec,
        generation_query_partition_spec=generation_query_partition_spec,
        generation_bias_partition_spec=generation_bias_partition_spec,
        key_partition_spec=key_partition_spec,
        value_partition_spec=value_partition_spec,
        bias_partition_spec=bias_partition_spec,
        attention_partition_spec=attention_partition_spec,
        shard_attention_computation=shard_attention_computation,
        input_shape=input_shape,
        backend=backend,
    )

    print(state)

    # Save State

    state.save_state(
        filename="SAVED_MODEL.easy",
    )
