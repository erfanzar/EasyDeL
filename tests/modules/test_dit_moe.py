import jax
import jax.numpy as jnp
from flax import nnx as nn

from easydel.modules.dit_moe.dit_moe_configuration import DiTMoEConfig
from easydel.modules.dit_moe.modeling_dit_moe import DiTMoE


def test_dit_moe_forward_shape():
	config = DiTMoEConfig(
		hidden_size=32,
		intermediate_size=64,
		moe_intermediate_size=64,
		num_attention_heads=4,
		n_routed_experts=8,
		num_experts_per_tok=2,
		n_shared_experts=1,
		n_group=2,
		topk_group=1,
	)
	module = DiTMoE(
		config=config,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		rngs=nn.Rngs(params=jax.random.PRNGKey(0)),
	)
	inputs = jax.random.normal(jax.random.PRNGKey(1), (2, 4, config.hidden_size))
	outputs = module(inputs)
	assert outputs.shape == inputs.shape
