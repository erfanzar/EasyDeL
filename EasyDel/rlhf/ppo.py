import jax.numpy as jnp
from flax.training import train_state

import flax.linen as nn
import gym

# Define the environment.
env = gym.make("CartPole-v0")

# Define the policy network.
policy = nn.Sequential(
    nn.Dense(128),
    nn.relu,
    nn.Dense(env.action_space.n),
)

