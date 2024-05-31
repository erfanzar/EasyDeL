# Example of loading model across mutiple devices
import copy

import flax.traverse_util
# Need the latest version 0.0.43 or git+https://github.com/erfanzar/EasyDeL.git

import jax
import torch

try:
    from src.python.easydel import get_modules_by_type, AutoShardAndGatherFunctions
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from src.python.easydel import get_modules_by_type, AutoShardAndGatherFunctions

from fjformer import make_shard_and_gather_fns, match_partition_rules
from transformers import FalconForCausalLM


def main():
    torch.manual_seed(42)
    FalconConfig, FlaxFalconForCausalLM, transform_fn = get_modules_by_type("falcon")
    config = FalconConfig(
        vocab_size=1200,
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=2,
        gradient_checkpointing="",
        alibi=False,
    )

    torch_model = FalconForCausalLM(
        config=copy.deepcopy(config)
    )
    easy_model = FlaxFalconForCausalLM(
        config=config
    )

    shard_fns, gather_fns = AutoShardAndGatherFunctions.from_config(
        config=config,
        flatten=True
    )

    pytorch_dict = torch_model.state_dict()
    with config.get_mesh():
        params = transform_fn(
            pytorch_dict,
            device=jax.devices("cpu")[0],  # This got no use but incase that some key missmatch and not getting
            # Kwargs req error we just pass that (No any params will be load on CPU for suer :) )
            shard_fns=shard_fns
        )
    print("Sharded Successfully")


if __name__ == "__main__":
    main()
