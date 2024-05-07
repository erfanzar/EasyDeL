# Example of loading model across mutiple devices
import jax

try:
    from src.python.easydel import AutoEasyDeLModelForCausalLM, AutoEasyDeLConfig, get_modules_by_type
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    cp = Path.cwd().__str__()
    sys.path.append(cp)
    from src.python.easydel import AutoEasyDeLModelForCausalLM, AutoEasyDeLConfig, get_modules_by_type

from fjformer import make_shard_and_gather_fns, match_partition_rules


def main():
    model_id = "erfanzar/LLamaStory-70M"
    config = AutoEasyDeLConfig.from_pretrained(
        pretrained_model_name_or_path=model_id
    )
    _, module, _ = get_modules_by_type(
        config.model_type
    )

    dummy_model = module(
        config=config,
        _do_init=False
    )
    partition_specs = match_partition_rules(config.get_partition_rules(True), dummy_model.params_shape_tree)
    shard_fns, gather_fns = make_shard_and_gather_fns(
        partition_specs=partition_specs,
        dtype_specs=jax.numpy.float16
    )

    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(model_id, shard_fns=shard_fns)


if __name__ == "__main__":
    main()
