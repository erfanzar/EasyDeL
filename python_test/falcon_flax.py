import copy

import flax.traverse_util
import jax

try:
    from lib.python.EasyDel import get_modules_by_type
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel import get_modules_by_type
from jax import numpy as jnp
from transformers import FalconForCausalLM
import torch
import numpy as np


def main():
    torch.manual_seed(42)
    FalconConfig, FlaxFalconForCausalLM, transform_fn = get_modules_by_type("falcon")
    config = FalconConfig(
        vocab_size=1200,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=1,
        gradient_checkpointing="",
        alibi=False
    )

    torch_model = FalconForCausalLM(
        config=copy.deepcopy(config)
    )
    input_shape = (1, 128)
    params = {"params":
        transform_fn(
            state_dict=torch_model.state_dict(),
            device=jax.devices("cpu")[0],
        )
    }
    np_random_input_ids = np.random.randint(0, config.vocab_size, input_shape)
    input_ids = torch.from_numpy(np_random_input_ids).reshape(1, -1).to(torch.long)
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(1, -1)
    torch_output = torch_model(
        input_ids=input_ids,
        attention_mask=torch.ones(*input_shape)
    )

    flax_model = FlaxFalconForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        _do_init=False,
        input_shape=input_shape
    )
    flax_output = flax_model(
        input_ids=flax_input_ids,
        params=params,
        return_dict=True

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
    # except TypeError as e:
    #     print(e.__str__())


if __name__ == '__main__':
    main()
