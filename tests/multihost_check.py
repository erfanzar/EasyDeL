import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
from jax import numpy as jnp

import easydel as ed
from jax.experimental import multihost_utils
from flax import nnx as nn


def main():
	sharding_axis_dims = (1, 1, 1, -1)
	max_length = 4096
	pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
	partition_axis = ed.PartitionAxis()
	dtype = jnp.bfloat16
	model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
		pretrained_model_name_or_path,
		auto_shard_model=True,
		sharding_axis_dims=sharding_axis_dims,
		config_kwargs=ed.EasyDeLBaseConfigDict(
			freq_max_position_embeddings=max_length,
			mask_max_position_embeddings=max_length,
			attn_dtype=dtype, 
			attn_mechanism=ed.AttentionMechanisms.VANILLA,
		), 
		platform=ed.EasyDeLPlatforms.JAX,
		param_dtype=dtype,
		dtype=dtype,
		partition_axis=partition_axis,
		precision=jax.lax.Precision("fastest"),
	)
	pytree = nn.to_tree(model)
	tr = []
	for ar in jax.tree_util.tree_leaves(pytree):
		tr.append(isinstance(ar, jax.Array))
	print(set(tr))
	multihost_utils.assert_equal(jax.tree_util.tree_leaves(pytree), "failed!")


if __name__ == "__main__":
	main()
