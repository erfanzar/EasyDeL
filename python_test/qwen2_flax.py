import os

# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import copy
import jax

try:
    from lib.python.EasyDel import Qwen2Config, FlaxQwen2ForCausalLM, get_modules_by_type
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel import Qwen2Config, FlaxQwen2ForCausalLM, get_modules_by_type
from jax import numpy as jnp
from transformers import Qwen2ForCausalLM
import torch
import numpy as np
from fjformer.partition_utils import make_shard_and_gather_fns, match_partition_rules, create_mesh


def main():
    torch.manual_seed(42)
    seq_len = 128
    config = Qwen2Config(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=4,
        intermediate_size=256,
        gradient_checkpointing="",
        max_position_embeddings=seq_len,
    )

    torch_model = Qwen2ForCausalLM(
        config=copy.deepcopy(config)
    )
    *_, trf_ = get_modules_by_type(model_type="qwen2")
    params = {"params": trf_(torch_model.state_dict(), device=jax.devices("cpu")[0])}
    batch_size = len(jax.devices())
    np_random_input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids = torch.from_numpy(np_random_input_ids).reshape(batch_size, -1).to(torch.long)
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(batch_size, -1)
    torch_output = torch_model(
        input_ids=input_ids
    )
    config.add_jax_args()
    config.add_basic_configurations(
        use_shard_map=True
    )
    mesh = config.jax_mesh()
    with mesh:
        partition_specs = match_partition_rules(config.get_partition_rules(True), params)
        shard, _ = make_shard_and_gather_fns(partition_specs, jnp.float32)

        params = jax.tree_map(lambda p, f: f(p), params, shard)

        flax_model = FlaxQwen2ForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=False,
            input_shape=(batch_size, seq_len)
        )
        flax_output = flax_model(
            input_ids=flax_input_ids,
            params=params,

        )
        res = jnp.allclose(torch_output.logits.cpu().detach().numpy(), flax_output.logits, atol=1e-5)
        print(
            "Qwen2 Huggingface Predictions :\n", torch_output.logits.cpu().detach().numpy(),
            "\nEasyDel Predictions: \n", flax_output.logits
        )
        if res:  # A Little Bit of humor
            print("\033[1;36mTest Passed Unfortunately 🥳")
        else:
            print("\033[1;31mTest Failed Successfully  🤕")
        error = jnp.mean(torch_output.logits.cpu().detach().numpy() - flax_output.logits)
        print("Error : ", error)


if __name__ == "__main__":
    main()
