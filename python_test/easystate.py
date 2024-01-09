from fjformer.optimizers import get_adamw_with_linear_scheduler
from lib.python.EasyDel import LlamaConfig, FlaxLlamaForCausalLM
from lib.python.EasyDel import EasyDelState


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
        steps=5000
    )
    tx, sc = get_adamw_with_linear_scheduler(
        **tx_init
    )
    state = EasyDelState.create(
        params=module.params,
        apply_fn=module.__call__,
        tx=tx,
        tx_init=tx_init,
        tx_name="adamw",
        sc_name="linear",
        model_type="llama",
        hyperparameters={
            "some": 2
        },
        module_config=config,
    )

    state.save_state(
        filename="state.easy", verbose=True
    )


def load():
    state = EasyDelState.load_state("state.easy", init_optimizer_state=False, verbose=True)
    print(state)


if __name__ == "__main__":
    main()
    load()
