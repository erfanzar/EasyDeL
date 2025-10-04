import types

import jax
import jax.numpy as jnp
import optax
from flax import nnx as nn

from easydel.modules.unet2d import UNet2DConditionModel, UNet2DConfig
from easydel.modules.vae import AutoencoderKL, VAEConfig
from easydel.trainers.stable_diffusion_trainer._fn import stable_diffusion_training_step
from easydel.trainers.stable_diffusion_trainer.stable_diffusion_config import StableDiffusionConfig


class DummyScheduler:
	def __init__(self, num_train_timesteps: int = 10):
		self.config = types.SimpleNamespace(
			num_train_timesteps=num_train_timesteps,
			prediction_type="epsilon",
		)
		alphas = jnp.linspace(0.1, 0.9, num_train_timesteps)
		self.alphas_cumprod = alphas

	def add_noise(self, latents, noise, timesteps):
		return latents + noise

	def get_velocity(self, latents, noise, timesteps):
		return noise


def _build_states():
	unet_cfg = UNet2DConfig(
		sample_size=4,
		in_channels=4,
		out_channels=4,
		down_block_types=["DownBlock2D"],
		up_block_types=["UpBlock2D"],
		block_out_channels=[32],
		layers_per_block=1,
		cross_attention_dim=768,
	)
	unet = UNet2DConditionModel(
		config=unet_cfg,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		rngs=nn.Rngs(params=jax.random.PRNGKey(0)),
	)
	unet_state = unet.to_state()
	unet_tx = optax.adamw(1e-3)
	unet_state = unet_state.replace(tx=unet_tx, opt_state=unet_tx.init(unet_state.graphstate))

	vae_cfg = VAEConfig(
		in_channels=3,
		out_channels=3,
		down_block_types=["DownEncoderBlock2D"],
		up_block_types=["UpDecoderBlock2D"],
		block_out_channels=[32],
		layers_per_block=1,
		latent_channels=4,
		scaling_factor=0.18215,
	)
	vae = AutoencoderKL(
		config=vae_cfg,
		dtype=jnp.float32,
		param_dtype=jnp.float32,
		rngs=nn.Rngs(params=jax.random.PRNGKey(1)),
	)
	vae_state = vae.to_state()
	return unet_state, vae_state


def test_stable_diffusion_training_step_uses_precomputed_text_embeddings():
	unet_state, vae_state = _build_states()
	config = StableDiffusionConfig(
		resolution=32,
		scaling_factor=0.18215,
		conditioning_dropout_prob=0.0,
	)
	scheduler = optax.constant_schedule(1e-3)
	batch = {
		"pixel_values": jax.random.normal(jax.random.PRNGKey(2), (2, 3, 32, 32)),
		"encoder_hidden_states": jax.random.normal(jax.random.PRNGKey(3), (2, 77, 768)),
	}
	noise_scheduler = DummyScheduler(num_train_timesteps=10)
	new_state, _, metrics, _ = stable_diffusion_training_step(
		unet_state,
		vae_state,
		text_encoder_state=None,
		batch=batch,
		noise_scheduler=noise_scheduler,
		config=config,
		learning_rate_fn=scheduler,
		partition_spec=None,
		gradient_accumulation_steps=1,
		rng=jax.random.PRNGKey(4),
	)
	assert jnp.isfinite(metrics.loss)
	assert int(new_state.step) == int(unet_state.step) + 1
