from typing import Optional, Sequence

import jax
from jax.sharding import Sharding, Mesh, PartitionSpec


class PartitionerConfig:
    def __init__(
            self,
            axis_dims: Sequence[int] = (1, -1, 1, 1),
            axis_names: Sequence[str] = ("dp", "fsdp", "tp", "sp"),
            backend: Optional[None] = jax.default_backend(),
            input_ids_partition_spec: PartitionSpec = PartitionSpec("dp", "fsdp"),
            attention_mask_partition_spec: PartitionSpec = PartitionSpec("dp", "fsdp"),
    ):
        self.axis_dims = axis_dims
        self.axis_names = axis_names
        self.backend = backend
        self.input_ids_partition_spec = input_ids_partition_spec
        self.attention_mask_partition_spec = attention_mask_partition_spec

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__ + "(" + "".join("\n\t" + k for k, v in self.__dict__.items()) + "\n)"
