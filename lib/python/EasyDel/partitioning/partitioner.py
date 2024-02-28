from jax.sharding import PartitionSpec
from dataclasses import dataclass


@dataclass
class EasyDelPartitions:
    query_partition_spec: PartitionSpec
    key_partition_spec: PartitionSpec
    value_partition_spec: PartitionSpec
    bias_partition_spec: PartitionSpec
    attention_partition_spec: PartitionSpec


def get_partitions(
        jax_attn_format: bool = True,
        fsdp_on_batch: bool = True
) -> EasyDelPartitions:

    """
    The get_partitions function is a helper function that returns an EasyDelPartitions object.
    The EasyDelPartitions object contains the PartitionSpec objects for each of the five tensors in
    the attention computation: query, key, value, bias and attention. The PartitionSpec objects are
    used to specify how each tensor should be partitioned across devices (i.e., which dimensions of
    each tensor should be split across devices). For example, if we want to split the batch dimension
    of all five tensors across two devices then we would set ``query_partition_spec=key_partition_spec=value_partition_spec=

    :param jax_attn_format: bool: Specify whether the attention
    :param fsdp_on_batch: bool: Determine whether the batch dimension is partitioned
    :return: A easydelpartitions object
    """
    if jax_attn_format:
        if fsdp_on_batch:
            query_partition_spec = PartitionSpec("fsdp", None, "sp", None)
            key_partition_spec = PartitionSpec("fsdp", None, "sp", None)
            value_partition_spec = PartitionSpec("fsdp", None, "sp", None)
            bias_partition_spec = PartitionSpec("fsdp", None, "sp", None)
            attention_partition_spec = PartitionSpec("fsdp", None, "sp", None)
        else:
            query_partition_spec = PartitionSpec("dp", "fsdp", "tp", "sp", None)
            key_partition_spec = PartitionSpec("dp", "fsdp", "tp", "sp", None)
            value_partition_spec = PartitionSpec("dp", "fsdp", "tp", "sp", None)
            bias_partition_spec = PartitionSpec("dp", None, "fsdp", None)
            attention_partition_spec = PartitionSpec("dp", "fsdp", "tp", "sp", None)
    else:
        if fsdp_on_batch:
            query_partition_spec = PartitionSpec("fsdp", "sp", None, None)
            key_partition_spec = PartitionSpec("fsdp", "sp", None, None)
            value_partition_spec = PartitionSpec("fsdp", "sp", None, None)
            bias_partition_spec = PartitionSpec("fsdp", "sp", None, None)
            attention_partition_spec = PartitionSpec("fsdp", "sp", None, None)
        else:
            query_partition_spec = PartitionSpec("dp", "sp", "fsdp", None)
            key_partition_spec = PartitionSpec("dp", "sp", "fsdp", None)
            value_partition_spec = PartitionSpec("dp", "sp", "fsdp", None)
            bias_partition_spec = PartitionSpec("dp", "fsdp", None, None)
            attention_partition_spec = PartitionSpec("dp", "sp", "fsdp", None)
    return EasyDelPartitions(
        query_partition_spec=query_partition_spec,
        key_partition_spec=key_partition_spec,
        value_partition_spec=value_partition_spec,
        bias_partition_spec=bias_partition_spec,
        attention_partition_spec=attention_partition_spec
    )
