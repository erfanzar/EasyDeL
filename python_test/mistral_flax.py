import copy

import os

os.environ["JAX_TRACEBACK_FILTERING"] = 'off'
import jax

try:
    from lib.python.EasyDel import MistralConfig, FlaxMistralForCausalLM
    from lib.python.EasyDel.transform import mistral_convert_hf_to_flax
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel import MistralConfig, FlaxMistralForCausalLM
    from lib.python.EasyDel.transform import mistral_convert_hf_to_flax
from jax import numpy as jnp
from transformers import MistralForCausalLM
import torch
import numpy as np


def main():
    torch.manual_seed(42)

    config = MistralConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_hidden_layers=16,
        intermediate_size=3072,
        gradient_checkpointing='',
    )
    print('Model Config :\n', config)

    torch_model = MistralForCausalLM(
        config=copy.deepcopy(config)
    )
    params = {"params": mistral_convert_hf_to_flax(torch_model.state_dict(), config, jax.devices('cpu')[0])}

    np_random_input_ids = np.random.randint(0, config.vocab_size, (1, 128))
    input_ids = torch.from_numpy(np_random_input_ids).reshape(1, -1).to(torch.long)
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(1, -1)
    torch_output = torch_model(
        input_ids=input_ids
    )
    config.add_jax_args()
    try:

        flax_model = FlaxMistralForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=False, input_shape=(1, 6)
        )
        flax_output = flax_model(
            input_ids=flax_input_ids,
            params=params,

        )
        res = jnp.allclose(torch_output.logits.cpu().detach().numpy(), flax_output.logits, atol=1e-5)
        print('Mistral Huggingface Predictions :\n', torch_output.logits.cpu().detach().numpy(),
              '\nEasyDel Predictions: \n', flax_output.logits)
        if res:  # A Little Bit of humor
            print('\033[1;36mTest Passed Unfortunately 🥳')
        else:
            print('\033[1;31mTest Failed Successfully  🤕')
        error = jnp.mean(torch_output.logits.cpu().detach().numpy() - flax_output.logits)
        print("Error : ", error)
    except TypeError as e:
        print(e.__str__())


if __name__ == '__main__':
    main()
