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

import jax.numpy as jnp

from easydel.layers.rotary_embedding import (
	DeepseekScalingRotaryEmbedding,
	DynamicNTKScalingRotaryEmbedding,
	LinearScalingRotaryEmbedding,
	Llama3RotaryEmbedding,
	Phi3LongRoPEScaledRotaryEmbedding,
	RotaryEmbedding,
	YaRNScalingRotaryEmbedding,
	get_rope,
)

run_batch_size = 2
run_nheads = 32
head_size = 128
rotary_dim = 128
max_position = 8192
run_seq_len = 4096
base = 10000
is_neox_style = True
dtype = jnp.float32


def test_rotary_embedding():
	rotary_emb = RotaryEmbedding(
		head_size,
		rotary_dim,
		max_position,
		base,
		is_neox_style,
		dtype,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type}")


def test_linear_scaling_rotary_embedding():
	scaling_factor = [2.0]
	rotary_emb = LinearScalingRotaryEmbedding(
		head_size=head_size,
		rotary_dim=rotary_dim,
		max_position_embeddings=max_position,
		base=base,
		is_neox_style=is_neox_style,
		scaling_factors=scaling_factor,
		dtype=dtype,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type}")


def test_dynamic_ntk_scaling_rotary_embedding():
	scaling_factor = 2.0
	rotary_emb = DynamicNTKScalingRotaryEmbedding(
		head_size=head_size,
		rotary_dim=rotary_dim,
		max_position_embeddings=max_position,
		base=base,
		is_neox_style=is_neox_style,
		scaling_factor=scaling_factor,
		dtype=dtype,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type}")


def test_yarn_scaling_rotary_embedding():
	scaling_factor = 2.0
	rope_scaling = {
		"extrapolation_factor": 1.0,
		"attn_factor": 1.0,
		"beta_fast": 32,
		"beta_slow": 1,
	}
	rotary_emb = YaRNScalingRotaryEmbedding(
		scaling_factor=scaling_factor,
		head_size=head_size,
		rotary_dim=rotary_dim,
		base=base,
		is_neox_style=is_neox_style,
		dtype=dtype,
		max_position_embeddings=max_position,
		**rope_scaling,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type}")


def test_deepseek_yarn_scaling_rotary_embedding():
	head_size = 64
	rotary_dim = 64
	max_position_embeddings = 2048
	base = 10000
	is_neox_style = True
	scaling_factor = 2.0
	dtype = jnp.float32

	rotary_emb = DeepseekScalingRotaryEmbedding(
		head_size=head_size,
		rotary_dim=rotary_dim,
		max_position_embeddings=max_position_embeddings,
		base=base,
		is_neox_style=is_neox_style,
		scaling_factor=scaling_factor,
		dtype=dtype,
		extrapolation_factor=1.0,
		attn_factor=1.0,
		beta_fast=32,
		beta_slow=1,
		mscale=1.0,
		mscale_all_dim=0.0,
	)

	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))

	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query.shape == query_rot.shape
	assert key_rot.shape == key_rot.shape
	print(f"Pass {rotary_emb._type}")


def test_llama3_rotary_embedding():
	scaling_factor = 2.0
	low_freq_factor = 1.0
	high_freq_factor = 1.0
	original_max_position = 1024
	rotary_emb = Llama3RotaryEmbedding(
		head_size,
		rotary_dim,
		max_position,
		base,
		is_neox_style,
		dtype,
		scaling_factor,
		low_freq_factor,
		high_freq_factor,
		original_max_position,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type}")


def test_phi3_long_rope_scaled_rotary_embedding():
	original_max_position = 1024
	short_factor = [1.0]
	long_factor = [2.0]
	rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
		head_size,
		rotary_dim,
		max_position,
		original_max_position,
		base,
		is_neox_style,
		dtype,
		short_factor,
		long_factor,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type}")


def test_get_rope():
	rope_scaling = {
		"rope_type": "yarn",
		"factor": 2.0,
		"original_max_position_embeddings": 1024,
		"extrapolation_factor": 1.0,
		"attn_factor": 1.0,
		"beta_fast": 32,
		"beta_slow": 1,
	}
	rotary_emb = get_rope(
		head_size,
		rotary_dim,
		max_position,
		base,
		is_neox_style,
		rope_scaling,
		dtype,
	)
	positions = jnp.arange(run_seq_len).reshape(1, -1).repeat(run_batch_size, 0)
	query = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	key = jnp.ones((run_batch_size, run_seq_len, run_nheads, head_size))
	query_rot, key_rot = rotary_emb(positions, query, key)
	assert query_rot.shape == query.shape
	assert key_rot.shape == key.shape
	print(f"Pass {rotary_emb._type} (get_rope)")


if __name__ == "__main__":
	test_rotary_embedding()
	test_linear_scaling_rotary_embedding()
	test_dynamic_ntk_scaling_rotary_embedding()
	test_yarn_scaling_rotary_embedding()
	test_llama3_rotary_embedding()
	test_deepseek_yarn_scaling_rotary_embedding()
	test_phi3_long_rope_scaled_rotary_embedding()
	test_get_rope()
