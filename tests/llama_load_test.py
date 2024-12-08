import os
import sys


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


import easydel as ed
from huggingface_hub import HfApi
import jax

from jax import sharding
from flax import nnx as nn

PartitionSpec, api = sharding.PartitionSpec, HfApi()


def main():
	rngs = nn.Rngs(0)
	config = ed.LlamaConfig(32000, 128, 512, 4, 4, 1, 64)
	config.attn_mechanism = "vanilla"
	config.gradient_checkpointing = ""
	model = ed.LlamaForCausalLM(config=config, rngs=rngs)
	input_ids = jax.random.randint(rngs.params(), (1, 128), 0, 32000, "i4")
	outputs = model(input_ids)
	print(outputs)
	kwargs = model.prepare_inputs_for_generation(input_ids=input_ids, max_length=128)
	outputs = model(input_ids)


if __name__ == "__main__":
	main()
