import copy
import jax
import transformers
from flax.traverse_util import flatten_dict, unflatten_dict

try:
    from lib.python.EasyDel.modules.qwen7b import Qwen7BConfig, FlaxQwen7BForCausalLM
    from lib.python.EasyDel.transform.easydel_transform import huggingface_to_easydel
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel.modules.qwen7b import Qwen7BConfig, FlaxQwen7BForCausalLM
    from lib.python.EasyDel.transform.easydel_transform import huggingface_to_easydel
from jax import numpy as jnp
from transformers import AutoModelForCausalLM, AutoConfig
import torch
import numpy as np


def main():
    torch.manual_seed(42)

    torch_config = AutoConfig.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        trust_remote_code=True
    )
    torch_config.hidden_size = 256
    torch_config.intermediate_size = 200
    torch_config.max_position_embeddings = 128
    torch_config.num_hidden_layers = 2
    torch_config.num_attention_heads = 2
    torch_config.cv_channels = 64

    torch_model = AutoModelForCausalLM.from_config(
        config=torch_config,
        trust_remote_code=True
    )

    torch_model.eval()
    params = {
        "params": huggingface_to_easydel(
            torch_model.state_dict(),
            embedding_layer_names=["wte"],
            device=jax.devices("cpu")[0],
        )
    }

    np_random_input_ids = np.random.randint(0, torch_config.vocab_size, (1, 8))
    input_ids = torch.from_numpy(np_random_input_ids).reshape(1, -1).to(torch.long)
    flax_input_ids = jnp.asarray(np_random_input_ids, dtype=jnp.int32).reshape(1, -1)

    torch_output = torch_model(
        input_ids=input_ids
    )
    config = Qwen7BConfig()
    for k, v in torch_config.__dict__.items():
        setattr(config, k, v)
    config.add_jax_args()
    flax_model = FlaxQwen7BForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        _do_init=False,
        input_shape=(1, 6)
    )
    flax_output = flax_model(
        input_ids=flax_input_ids,
        params=params,
        add_params_field=False,
        train=False,
        return_dict=True
    )
    torch_output = torch_output.logits.cpu().detach().numpy()
    res = jnp.allclose(torch_output, flax_output.logits, atol=1e-5)
    print("PHI Huggingface Predictions :\n", torch_output,
          "\nEasyDel Predictions: \n", flax_output.logits)
    if res:
        print("\033[1;36mTest Passed Unfortunately 🥳")
    else:
        print("\033[1;31mTest Failed Successfully  🤕")
    error = jnp.mean(torch_output - flax_output.logits)
    print("Error : ", error)


if __name__ == "__main__":
    main()
