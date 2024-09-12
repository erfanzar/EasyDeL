import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))


from easydel import (
	FlaxLlamaForCausalLM,
	LlamaConfig,
)
from jax import lax
from jax import numpy as jnp


def main():
	model = FlaxLlamaForCausalLM(
		config=LlamaConfig(
			hidden_size=512,
			intermediate_size=1024,
			num_hidden_layers=4,
			max_position_embeddings=512,
			use_scan_mlp=False,
			axis_dims=(1, -1, 1, 1),
			quantize_kv_cache=True,
			q_block=32,
			k_block=32,
			pallas_runtime=True,
			attn_mechanism="flash_attn2",
		),
		dtype=jnp.float16,
		param_dtype=jnp.float16,
		precision=lax.Precision("fastest"),
		input_shape=(2, 2),
		_do_init=True,
		seed=81,
	)
	print(model.init_cache(1, 128))
	print(model.init_cache(1, 128)["model"]["layers"]["0"]["self_attn"]["cached_key"])
	print(
		model.init_cache(1, 128)["model"]["layers"]["0"]["self_attn"][
			"cached_key"
		].__class__.__name__
	)


if __name__ == "__main__":
	main()
