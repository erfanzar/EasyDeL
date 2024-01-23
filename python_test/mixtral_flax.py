import copy

import os

os.environ["JAX_TRACEBACK_FILTERING"] = 'off'
import jax

try:
    from lib.python.EasyDel import MixtralConfig, FlaxMixtralForCausalLM
    from lib.python.EasyDel.transform.easydel_transform import huggingface_to_easydel
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel import MixtralConfig, FlaxMixtralForCausalLM
    from lib.python.EasyDel.transform.easydel_transform import huggingface_to_easydel
from jax import numpy as jnp
from transformers import MixtralForCausalLM
import torch
import numpy as np


def main():
    torch.manual_seed(42)
    seq_len = 128
    config = MixtralConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=1,
        num_key_value_heads=4,
        intermediate_size=512,
        num_local_experts=8,
        max_position_embeddings=seq_len
    )
    batch_size = len(jax.devices())

    torch_model = MixtralForCausalLM(
        config=copy.deepcopy(config)
    )
    params = {"params":
        huggingface_to_easydel(
            torch_model.state_dict(),
            embedding_layer_names=["embed_tokens"],
            device=jax.devices('cpu')[0]
        )
    }

    np_random_input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids = torch.from_numpy(np_random_input_ids).reshape(batch_size, -1).to(torch.long)
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(batch_size, -1)
    torch_output = torch_model(
        input_ids=input_ids
    )
    torch_output = torch_output.logits.cpu().detach().numpy()
    config.add_jax_args()
    config.add_basic_configurations(
        use_shard_map=True
    )

    try:
        flax_model = FlaxMixtralForCausalLM(
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
        res = jnp.allclose(torch_output, flax_output.logits, atol=1e-5)
        print('Mistral Huggingface Predictions :\n', torch_output,
              '\nEasyDel Predictions: \n', flax_output.logits)
        if res:  # A Little Bit of humor
            print('\033[1;36mTest Passed Unfortunately 🥳')
        else:
            print('\033[1;31mTest Failed Successfully  🤕')
        error = jnp.mean(torch_output - flax_output.logits)
        print("Error : ", error)
    except TypeError as e:
        print(e.__str__())


if __name__ == '__main__':
    main()
