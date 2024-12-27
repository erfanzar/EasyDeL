import easydel as ed

import jax
import torch

from transformers import LlamaForCausalLM
from flax import nnx as nn
from jax import numpy as jnp


def to_np(x):
	return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x


def print_errors(left, right, prefix="", n_diff=5):
	left = to_np(left)
	right = to_np(right)
	prefix = f"{prefix} - " if prefix else ""

	diff_mask = ~jnp.isclose(left, right, atol=0.125, rtol=0)
	diff_indices = jnp.where(diff_mask.reshape(-1))[0]
	n_fails = jnp.sum(diff_mask)
	correct_percentage = 100 * (1 - jnp.mean(diff_mask))
	mean_error = jnp.mean(left) - jnp.mean(right)
	is_passed = n_fails == 0
	print(f"\n{prefix}Comparison Results:")
	print("=" * (len(prefix) + 20))
	if n_fails > 0:
		print(f"\n{prefix}Found {n_fails} differences")
		print(f"{prefix}Showing first {min(n_diff, len(diff_indices))} differences:")
		for idx in diff_indices[:n_diff]:
			print(
				f"{prefix}  Index {idx:6d}: {left.reshape(-1)[idx]:10.6f} vs {right.reshape(-1)[idx]:10.6f} "
				f"(diff: {left.reshape(-1)[idx] - right.reshape(-1)[idx]:10.6f})"
			)
	print(f"\n{prefix}Summary Statistics:")
	print(f"{prefix}  Pass/Fail: {'✓ PASSED' if is_passed else '✗ FAILED'}")
	print(f"{prefix}  Correct Elements: {correct_percentage:6.2f}%")
	print(f"{prefix}  Mean Error: {mean_error:10.6f}")

	print(f"\n{prefix}Last 5 Elements Comparison:")
	print(f"{prefix}  Left:  {left.reshape(-1)[-5:]}")
	print(f"{prefix}  Right: {right.reshape(-1)[-5:]}")
	print("\n" + "=" * (len(prefix) + 20))


def main():
	config = ed.LlamaConfig(
		hidden_size=128,
		intermediate_size=256,
		num_hidden_layers=4,
		num_attention_heads=4,
	)

	hf_model = LlamaForCausalLM(config=config)

	model = ed.LlamaForCausalLM.lazy_init(
		config=config,
		dtype=jnp.float16,
		param_dtype=jnp.float16,
		rngs=nn.Rngs(0),
	)
	ids = jax.random.randint(jax.random.key(42), (1, 64), 0, 400, "i4")
	model = ed.traversals.merge_model_and_tree(
		model,
		model.transform_fn(hf_model.state_dict()),
	)
	model.apply_lora_to_layers(32, ".*(gate_proj|up_proj).*")
	lora_output = model(ids)
	model.unwrap_lora_to_layers()
	unwa_output = model(ids)
	print_errors(lora_output.logits, unwa_output.logits)


if __name__ == "__main__":
	main()
