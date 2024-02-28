from tqdm import tqdm
import jax
import jax.numpy as jnp
import optax
from lib.python.EasyDel import AutoEasyDelModelForCausalLM

from fjformer.xrapture.xrapture import XRapTure, XRapTureConfig


def main():
    model, params = AutoEasyDelModelForCausalLM.from_pretrained('erfanzar/LLamaStory-70M')

    tx = optax.adamw(learning_rate=1e-4, weight_decay=1e-4)
    rab_config = XRapTureConfig(
        64,
        fully_fine_tune_parameters=["embed_tokens"],
        lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],
        verbose=True
    )
    rab = XRapTure(
        config=rab_config
    )

    rapture_modules = rab.apply_lora(
        module=model,
        parameters={"params": params},
        tx=tx
    )

    tx = rapture_modules.lora_tx
    lora_module = rapture_modules.lora_module
    lora_parameters = rapture_modules.lora_parameters
    lora_opt_state = rapture_modules.lora_opt_state

    def loss_fn(lora_params, batch):
        return -jnp.mean(
            jnp.take_along_axis(
                jax.nn.log_softmax(
                    lora_module(batch[:, :-1], params=lora_params).logits
                ), batch[:, 1:, None], axis=-1
            )
        )

    @jax.jit
    def update_fn(
            params,
            opt_state,
            batch
    ):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = tx.update(grads, opt_state, params=params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    example_data = jax.random.randint(jax.random.PRNGKey(0), (4, 128), 0, 32000)

    iterations = 300

    pbar = tqdm(range(iterations))

    pbar.set_description("LoRA Tuning ")

    for _ in pbar:
        lora_parameters, lora_opt_state, loss = update_fn(
            lora_parameters,
            lora_opt_state,
            example_data
        )
        pbar.set_postfix(loss=loss, total_iterations=iterations)

    final_predictions = lora_module(
        example_data,
        params=lora_parameters
    ).logits

    merged_params = rab.merge_parameters(
        lora_parameters
    )


if __name__ == '__main__':
    main()
