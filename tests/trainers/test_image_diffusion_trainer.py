import jax
import jax.numpy as jnp
import optax
from flax import nnx as nn

from easydel.modules.dit import DiTConfig, DiTForImageDiffusion
from easydel.trainers.image_diffusion_trainer._fn import training_step


def _build_state():
	config = DiTConfig(
		image_size=8,
		patch_size=2,
		in_channels=3,
		hidden_size=32,
		num_hidden_layers=2,
		num_attention_heads=4,
		num_classes=4,
		class_dropout_prob=0.0,
		use_conditioning=True,
	)
	model = DiTForImageDiffusion(
		config=config,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		rngs=nn.Rngs(params=jax.random.PRNGKey(0)),
	)
	state = model.to_state()
	tx = optax.adam(1e-3)
	return state.replace(tx=tx, opt_state=tx.init(state.graphstate))


def test_image_diffusion_training_step_runs():
	state = _build_state()
	batch = {
		"pixel_values": jax.random.normal(jax.random.PRNGKey(1), (2, 8, 8, 3)),
		"labels": jnp.array([0, 1], dtype=jnp.int32),
		"rng_keys": jax.random.split(jax.random.PRNGKey(2), 2),
	}
	new_state, metrics = training_step(
		state,
		batch,
		num_train_timesteps=16,
		prediction_type="velocity",
		gradient_accumulation_steps=1,
	)
	assert jnp.isfinite(metrics.loss)
	assert int(new_state.step) == int(state.step) + 1
