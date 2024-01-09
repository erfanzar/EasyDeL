import jax

from lib.python.EasyDel import LlamaConfig, FlaxLlamaForCausalLM
from lib.python.EasyDel import EasyDelState
from lib.python.EasyDel.etils.auto_tx import get_optimizer_and_scheduler


def main():
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        use_sacn_mlp=False,
    )

    module = FlaxLlamaForCausalLM(
        config=config,
        input_shape=(8, 8),
        _do_init=True
    )

    tx_init = dict(
        optimizer="adamw",
        scheduler="none",
        learning_rate=1e-5,
        steps=5000
    )
    tx, sc = get_optimizer_and_scheduler(
        **tx_init
    )
    state = EasyDelState.create(
        params=module.params,
        apply_fn=module.__call__,
        tx=tx,
        tx_init=tx_init,
        hyperparameters={
            "model_type_is_llama": 1
        },
        module_config=config,
    )

    state.save_state(
        filename="state.easy", verbose=True
    )


def load():
    state = EasyDelState.load_state("state.easy", init_optimizer_state=False, verbose=True)
    state = state.shard_params()
    print(jax.eval_shape(lambda: state))


def eval_shape_create_test():
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        use_sacn_mlp=False,
    )

    module = FlaxLlamaForCausalLM(
        config=config,
        input_shape=(8, 8),
        _do_init=True
    )

    tx_init = dict(
        optimizer="adamw",
        scheduler="none",
        learning_rate=1e-5,
        steps=5000
    )

    def create_state():
        state = EasyDelState.create(
            module_config=config,
            params=module.params,
            tx_init=tx_init,
            apply_fn=module.__call__,
            tx=get_optimizer_and_scheduler(**tx_init)[0],
            hyperparameters={
                "model_type_is_llama": 1
            },
            module=module,
            module_config_args=None
        )
        print(state)
        return state

    print(jax.eval_shape(create_state))


if __name__ == "__main__":
    # main()
    # load()
    eval_shape_create_test()
