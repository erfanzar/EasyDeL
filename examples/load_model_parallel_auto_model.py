# Example of loading model across mutiple devices

# Need the latest version 0.0.43 or git+https://github.com/erfanzar/EasyDeL.git

import jax

try:
    from lib.python.EasyDel import AutoEasyDelModelForCausalLM, AutoEasyDelConfig, get_modules_by_type, \
        AutoShardAndGatherFunctions
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from lib.python.EasyDel import AutoEasyDelModelForCausalLM, AutoEasyDelConfig, get_modules_by_type, \
        AutoShardAndGatherFunctions

from fjformer import make_shard_and_gather_fns, match_partition_rules


def main():
    model_id = "erfanzar/LLamaStory-70M"
    config = AutoEasyDelConfig.from_pretrained(
        pretrained_model_name_or_path=model_id
    )
    _, module, _ = get_modules_by_type(
        config.model_type
    )

    dummy_model = module(
        config=config,
        _do_init=False
    )
    shard_fns, gather_fns = AutoShardAndGatherFunctions.from_config(
        config=config
    )
    model, params = AutoEasyDelModelForCausalLM.from_pretrained(model_id, shard_fns=shard_fns)


if __name__ == "__main__":
    main()
