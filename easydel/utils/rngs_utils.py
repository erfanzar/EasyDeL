# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for JAX."""

from typing import Tuple, Union

from jax import random as jrandom


def next_rng(
	*args,
	**kwargs,
) -> Union[jrandom.PRNGKey, Tuple[jrandom.PRNGKey, ...], dict]:
	"""Provides access to the global JaxRNG and splits the key based on arguments.

	This function wraps the global `jax_utils_rng` instance and calls its `__call__` method,
	passing through any arguments provided. This provides a convenient way to access and
	split the global random number generator key.

	Args:
	    *args: Positional arguments passed to the `jax_utils_rng` instance's `__call__` method.
	    **kwargs: Keyword arguments passed to the `jax_utils_rng` instance's `__call__` method.

	Returns:
	    The split PRNGKey(s) from the global `jax_utils_rng` instance.
	"""
	global jax_utils_rng
	return jax_utils_rng(*args, **kwargs)


class JaxRNG:
	"""A wrapper around JAX's PRNGKey that simplifies key splitting."""

	def __init__(self, rng: jrandom.PRNGKey):
		"""Initializes the JaxRNG with a PRNGKey.

		Args:
		    rng: A JAX PRNGKey.
		"""
		self.rng = rng

	@classmethod
	def from_seed(cls, seed: int) -> "JaxRNG":
		"""Creates a JaxRNG instance from a seed.

		Args:
		    seed: The seed to use for the random number generator.

		Returns:
		    A JaxRNG instance.
		"""
		return cls(jrandom.PRNGKey(seed))

	def __call__(
		self, keys: Union[int, Tuple[str, ...]] = None
	) -> Union[jrandom.PRNGKey, Tuple[jrandom.PRNGKey, ...], dict]:
		"""Splits the internal PRNGKey and returns new keys.

		Args:
		    keys:  If None, returns a single split key and updates the internal RNG.
		           If an int, splits the key into `keys + 1` parts, updates the internal RNG,
		           and returns the last `keys` parts as a tuple.
		           If a tuple of strings, splits the key into `len(keys) + 1` parts,
		           updates the internal RNG, and returns a dictionary mapping the strings
		           to their corresponding key parts.

		Returns:
		    The split PRNGKey(s) based on the `keys` argument.
		"""
		if keys is None:
			self.rng, split_rng = jrandom.split(self.rng)
			return split_rng
		elif isinstance(keys, int):
			split_rngs = jrandom.split(self.rng, num=keys + 1)
			self.rng = split_rngs[0]
			return tuple(split_rngs[1:])
		else:
			split_rngs = jrandom.split(self.rng, num=len(keys) + 1)
			self.rng = split_rngs[0]
			return {key: val for key, val in zip(keys, split_rngs[1:])}  # noqa:B905


class GenerateRNG:
	"""An infinite generator of JAX PRNGKeys, useful for iterating over seeds."""

	def __init__(self, seed: int = 0):
		"""Initializes the generator with a starting seed.

		Args:
		    seed: The seed to use for the initial PRNGKey.
		"""
		self.seed = seed
		self._rng = jrandom.PRNGKey(seed)

	def __next__(self) -> jrandom.PRNGKey:
		"""Generates and returns the next PRNGKey in the sequence.

		Returns:
		    The next PRNGKey derived from the internal state.
		"""
		self._rng, key = jrandom.split(self._rng)
		return key

	@property
	def rng(self) -> jrandom.PRNGKey:
		"""Provides access to the next PRNGKey without advancing the generator.

		Returns:
		    The next PRNGKey in the sequence.
		"""
		return next(self)
