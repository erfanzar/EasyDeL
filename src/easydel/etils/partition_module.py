from typing import NamedTuple, Optional, Tuple, Union

AxisType = Optional[Union[Tuple[str, ...], str]]


class PartitionAxis(NamedTuple):
    """
    A NamedTuple representing different axes of partitioning in a model.

    Each field represents an axis and its corresponding partitioning strategy.
    The value of each field can be:

    * None: The axis is not partitioned.
    * str: The name of the single mesh dimension across which the axis is partitioned.
    * Tuple[str, ...]: A tuple of mesh dimension names, indicating a sharding strategy
      where the axis is split across multiple mesh dimensions.

    Attributes:
        batch_axis: Partitioning strategy for the batch dimension. Defaults to ("fsdp", "dp").
        sequence_axis: Partitioning strategy for the sequence dimension. Defaults to "sp".
        query_sequence_axis: Partitioning strategy for the query sequence dimension. Defaults to "sp".
        head_axis: Partitioning strategy for the attention head dimension. Defaults to "tp".
        key_sequence_axis: Partitioning strategy for the key sequence dimension. Defaults to "sp".
        hidden_state_axis: Partitioning strategy for the hidden state dimension. Defaults to "tp".
        attention_dim_axis: Partitioning strategy for the attention dimension. Defaults to None.
        bias_head_sequence_axis: Partitioning strategy for the bias head sequence dimension. Defaults to None.
        bias_key_sequence_axis: Partitioning strategy for the bias key sequence dimension. Defaults to None.
        generation_query_sequence_axis: Partitioning strategy for the query sequence dimension during generation.
            Defaults to None.
        generation_head_axis: Partitioning strategy for the attention head dimension during generation.
            Defaults to "tp".
        generation_key_sequence_axis: Partitioning strategy for the key sequence dimension during generation.
            Defaults to "sp".
        generation_attention_dim_axis: Partitioning strategy for the attention dimension during generation.
            Defaults to None.
    """

    batch_axis: AxisType = ("fsdp", "dp")
    sequence_axis: AxisType = "sp"
    query_sequence_axis: AxisType = "sp"
    head_axis: AxisType = "tp"
    key_sequence_axis: AxisType = "sp"
    hidden_state_axis: AxisType = "tp"
    attention_dim_axis: AxisType = None
    bias_head_sequence_axis: AxisType = None
    bias_key_sequence_axis: AxisType = None

    generation_query_sequence_axis: AxisType = None
    generation_head_axis: AxisType = "tp"
    generation_key_sequence_axis: AxisType = "sp"
    generation_attention_dim_axis: AxisType = None
