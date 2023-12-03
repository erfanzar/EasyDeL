import copy
import jax
import transformers
from flax.traverse_util import flatten_dict, unflatten_dict

try:
    from lib.python.EasyDel.modules.phi import PhiConfig, FlaxPhiForCausalLM
    from lib.python.EasyDel.transform.easydel_transform import huggingface_to_easydel
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel.modules.phi import PhiConfig, FlaxPhiForCausalLM
    from lib.python.EasyDel.transform.easydel_transform import huggingface_to_easydel
from jax import numpy as jnp
from transformers import AutoModelForCausalLM
import torch
import numpy as np


def main():
    torch.manual_seed(42)

    torch_model = AutoModelForCausalLM.from_pretrained(
        'microsoft/phi-1_5',
        trust_remote_code=True
    )
    _config: transformers.PretrainedConfig = torch_model.config
    print(f"Config:\n{_config}")
    params = {"params": unflatten_dict(huggingface_to_easydel(
        torch_model.state_dict(),
        'wte',
        device=jax.devices('cpu')[0]
    ))}
    print(params)
    np_random_input_ids = np.random.randint(0, _config.vocab_size, (1, 128))
    input_ids = torch.from_numpy(np_random_input_ids).reshape(1, -1).to(torch.long)
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(1, -1)

    torch_output = torch_model(
        input_ids=input_ids
    )
    config = PhiConfig()
    for k, v in _config.__dict__.items():
        setattr(config, k, v)
    config.add_jax_args()
    # try:

    flax_model = FlaxPhiForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        _do_init=False,
        input_shape=(1, 6)
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

    # except TypeError as e:
    #     print(e.__str__())


if __name__ == '__main__':
    main()
