import functools
import os
import sys
import typing as tp

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from easydel.kernels.gpu_ops.triton_gqa_flash_attention_2 import (
	triton_gqa_flash_attention2_gpu,
)
import jax
import jaxlib
import triton
from jax import nn
from jax import numpy as jnp
from jax import random as jrnd
from fjformer import create_mesh
from jax.sharding import NamedSharding, PartitionSpec

mesh = create_mesh(
	(1, 1, 1, -1),
	("dp", "fsdp", "tp", "sp"),
)


def _get_inputs(
	B,
	QH,
	KVH,
	QS,
	KS,
	D,
	provider: tp.Literal["triton", "cudnn_sdpa"],
	USE_BIAS=True,
):
	q_key, k_key, v_key = jrnd.split(jrnd.PRNGKey(8), 3)
	qkv_spec = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
	b_spec = PartitionSpec(("dp", "fsdp"), "tp", None, None)
	q = jax.nn.initializers.normal(2)(q_key, (B, QS, QH, D), dtype=jnp.float16)
	k = jax.nn.initializers.normal(2)(k_key, (B, KS, KVH, D), dtype=jnp.float16)
	v = jax.nn.initializers.normal(2)(v_key, (B, KS, KVH, D), dtype=jnp.float16)
	b = (
		jnp.where(
			jrnd.randint(v_key, (B, 1, QS, KS), 0, 4) > 2,
			jnp.finfo(jnp.float16).min,
			0,
		)
		if USE_BIAS
		else None
	)
	if provider == "triton":
		q, k, v = map(lambda x: jax.device_put(x, NamedSharding(mesh, qkv_spec)), [q, k, v])
		if USE_BIAS:
			b = jax.device_put(b, NamedSharding(mesh, b_spec))
	return q, k, v, b


benchmark_configs = []
for mode in ["fwd"]:
	for batch_size in [1, 2]:
		for bias in [True, False]:
			for headdim in [64, 128, 256]:
				for num_heads in [8, 16, 32]:
					benchmark_configs.append(
						triton.testing.Benchmark(
							x_names=["S"],
							x_vals=[1024, 2048, 4096, 6144, 8192],
							line_arg="provider",
							line_vals=["triton", "cudnn_sdpa"],
							line_names=["Triton", "cuDNN SDPA"],
							styles=[("green", "-"), ("blue", ":")],
							ylabel="MS",
							plot_name=f"batch_size={batch_size}-bias={bias}-headdim={headdim}-num_heads={num_heads}-mode={mode}",
							args={
								"B": batch_size,
								"H": num_heads,
								"D": headdim,
								"mode": mode,
								"BIAS": bias,
							},
						)
					)


@triton.testing.perf_report(benchmark_configs)
def mha_attention_benchmark(
	B,
	S,
	H,
	D,
	mode,
	BIAS,
	provider,
):
	# try:
	query, key, value, bias = _get_inputs(B, H, H, S, S, D, provider, BIAS)
	if mode == "fwd":
		if provider == "triton":
			fn = lambda: triton_gqa_flash_attention2_gpu(query, key, value, bias)
		elif provider == "cudnn_sdpa":
			_fn = jax.jit(
				functools.partial(
					nn.dot_product_attention,
					implementation="cudnn",
				)
			)
			fn = lambda: _fn(query, key, value, bias).block_until_ready()
	elif mode == "bwd":
		if provider == "triton":
			fn = lambda: jax.grad(lambda *x: triton_gqa_flash_attention2_gpu(*x).sum())(
				query, key, value, bias
			)
		elif provider == "cudnn_sdpa":
			_fn = jax.jit(
				functools.partial(
					nn.dot_product_attention,
					implementation="cudnn",
				)
			)
			fn = lambda: jax.grad(lambda *x: _fn(*x).sum())(
				query, key, value, bias
			).block_until_ready()
	try:
		ms = triton.testing.do_bench(fn)
	except jaxlib.xla_extension.XlaRuntimeError:
		ms = 1000.0000
	return ms
	# except:  # noqa
	# 	return 500.0000


if __name__ == "__main__":
	mha_attention_benchmark.run(
		print_data=True,
		save_path="/home/erfan/PycharmProjects/bench/",
	)
