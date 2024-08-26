import os
import sys

import jax

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)
sys.path.append(
	os.path.join(
		dirname,
		"../src",
	)
)

from easydel import EasyDeLState, FlaxLlamaForCausalLM, LlamaConfig
from easydel.etils.auto_tx import get_optimizer_and_scheduler


def main():
	config = LlamaConfig(
		hidden_size=512,
		intermediate_size=512,
		num_hidden_layers=1,
		num_attention_heads=8,
		num_key_value_heads=4,
		use_scan_mlp=False,
	)

	module = FlaxLlamaForCausalLM(
		config=config,
		input_shape=(8, 8),
		dtype=jax.numpy.float32,
		param_dtype=jax.numpy.float32,
		_do_init=True,
	)
	# print(module.params)
	state = module.to_easydel_state(module.params)
	# state = state.to_8bit()
	state.save_state("state.easy")


def load():
	state = EasyDeLState.load_state(
		"state.easy",
		init_optimizer_state=False,
		verbose=True,
		param_dtype=jax.numpy.float16,
	)
	# print(jax.eval_shape(lambda: state))
	# print(state)
	# print("-" * 50)
	# state.serialize()
	# print(state)
	# print("-" * 50)
	# state.un_serialize()
	# print(state)


def eval_shape_create_test():
	config = LlamaConfig(
		hidden_size=512,
		intermediate_size=512,
		num_hidden_layers=4,
		num_attention_heads=8,
		num_key_value_heads=4,
		use_scan_mlp=False,
	)

	module = FlaxLlamaForCausalLM(config=config, input_shape=(8, 8), _do_init=True)

	tx_init = dict(optimizer="adamw", scheduler="none", learning_rate=1e-5, steps=5000)

	def create_state():
		state = EasyDeLState.create(
			module_config=config,
			params=module.params,
			tx_init=tx_init,
			apply_fn=module.__call__,
			tx=get_optimizer_and_scheduler(**tx_init)[0],
			hyperparameters=EasyDeLState.create_hyperparameters(model_type=config.model_type),
			module=module,
			module_config_args=None,
		)
		return state

	print(jax.eval_shape(create_state))


if __name__ == "__main__":
	main()
	load()
	# eval_shape_create_test()
