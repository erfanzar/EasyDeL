import logging
import os
import time
import typing

import numpy as np
from jax._src.mesh import Mesh

os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
from fjformer import get_jax_mesh, get_mem, initialise_tracking, make_shard_and_gather_fns, match_partition_rules
from EasyDel.modules import FlaxLlamaForCausalLM, LlamaConfig
from typing import Union, Tuple, Sequence, Optional
from jax.sharding import PartitionSpec


def main():
    initialise_tracking(interval=0.1)
    mesh = get_jax_mesh('1,-1,1', ("dp", "fsdp", "sp"))
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8
    )

    model = FlaxLlamaForCausalLM(config=config, _do_init=False)
    with jax.default_device(jax.devices('cpu')[4]):
        params = model.init_weights(jax.random.PRNGKey(0), (1, 1))
    partition_rules = (

        ("model/embed_tokens/embedding", PartitionSpec(("tp", "fsdp"))),

        ("self_attn/(q_proj|k_proj|v_proj)/kernel", PartitionSpec("sp", ("fsdp", "tp"))),
        ("self_attn/o_proj/kernel", PartitionSpec(("fsdp", "tp"))),

        ("mlp/gate_proj/kernel", PartitionSpec("tp", "fsdp")),
        ("mlp/down_proj/kernel", PartitionSpec("fsdp", "tp")),
        ("mlp/up_proj/kernel", PartitionSpec("tp", "fsdp")),

        ("input_layernorm/kernel", PartitionSpec(("fsdp", "tp"))),
        ("post_attention_layernorm/kernel", PartitionSpec(("fsdp", "tp"))),

        ("model/norm/kernel", PartitionSpec(("fsdp", "tp"))),
        ("lm_head/kernel", PartitionSpec(("fsdp", "tp"))),
        (".*", PartitionSpec(("fsdp", "sp"))),
    )
    partition_specs = match_partition_rules(partition_rules, params=params)
    with mesh:
        shard_fns, _ = make_shard_and_gather_fns(partition_specs=partition_specs)
        new_params = jax.tree_util.tree_map(lambda f, p: f(p), shard_fns, params)
    del params
    time.sleep(0.2)
    print(get_mem())


if __name__ == "__main__":
    main()
