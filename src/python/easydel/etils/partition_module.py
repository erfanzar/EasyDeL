from typing import NamedTuple, Tuple, Union, Optional

AxisType = Optional[Union[Tuple[str, ...], str]]


class PartitionAxis(NamedTuple):
    batch_axis: AxisType = ("fsdp", "dp")
    query_sequence_axis: AxisType = "sp"
    head_axis: AxisType = "tp"
    key_sequence_axis: AxisType = "sp"
    hidden_state_axis: AxisType = "sp"
    attention_dim_axis: AxisType = None
    bias_head_sequence_axis: AxisType = None
    bias_key_sequence_axis: AxisType = None

    generation_query_sequence_axis: AxisType = None
    generation_head_axis: AxisType = "tp"
    generation_key_sequence_axis: AxisType = "sp"
    generation_attention_dim_axis: AxisType = None
