from jax.sharding import PartitionSpec
from dataclasses import dataclass


@dataclass
class EasyDelPartitions:
    q_ps: PartitionSpec
    k_ps: PartitionSpec
    v_ps: PartitionSpec
    b_ps: PartitionSpec
    a_ps: PartitionSpec


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
    of all five tensors across two devices then we would set ``q_ps=k_ps=v_ps=

    :param jax_attn_format: bool: Specify whether the attention
    :param fsdp_on_batch: bool: Determine whether the batch dimension is partitioned
    :return: A easydelpartitions object
    """
    if jax_attn_format:
        if fsdp_on_batch:
            q_ps = PartitionSpec("fsdp", None, "sp", None)
            k_ps = PartitionSpec("fsdp", None, "sp", None)
            v_ps = PartitionSpec("fsdp", None, "sp", None)
            b_ps = PartitionSpec("fsdp", None, "sp", None)
            a_ps = PartitionSpec("fsdp", None, "sp", None)
        else:
            q_ps = PartitionSpec("dp", "fsdp", "tp", "sp", None)
            k_ps = PartitionSpec("dp", "fsdp", "tp", "sp", None)
            v_ps = PartitionSpec("dp", "fsdp", "tp", "sp", None)
            b_ps = PartitionSpec("dp", None, "fsdp", None)
            a_ps = PartitionSpec("dp", "fsdp", "tp", "sp", None)
    else:
        if fsdp_on_batch:
            q_ps = PartitionSpec("fsdp", "sp", None, None)
            k_ps = PartitionSpec("fsdp", "sp", None, None)
            v_ps = PartitionSpec("fsdp", "sp", None, None)
            b_ps = PartitionSpec("fsdp", "sp", None, None)
            a_ps = PartitionSpec("fsdp", "sp", None, None)
        else:
            q_ps = PartitionSpec("dp", "sp", "fsdp", None)
            k_ps = PartitionSpec("dp", "sp", "fsdp", None)
            v_ps = PartitionSpec("dp", "sp", "fsdp", None)
            b_ps = PartitionSpec("dp", "fsdp", None, None)
            a_ps = PartitionSpec("dp", "sp", "fsdp", None)
    return EasyDelPartitions(
        q_ps=q_ps,
        k_ps=k_ps,
        v_ps=v_ps,
        b_ps=b_ps,
        a_ps=a_ps
    )
