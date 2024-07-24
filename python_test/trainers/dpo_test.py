import os
import sys

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dirname)  # noqa: E402
sys.path.append(
    os.path.join(
        dirname,
        "../../src",
    )
)  # noqa: E402
# jax.config.update("jax_platform_name", "cpu")  # CPU Test !
from typing import Dict, Optional  # noqa: E402

import easydel as ed  # noqa
import jax  # noqa: E402  # noqa
from datasets import Dataset, load_dataset  # noqa: E402
from easydel import (  # noqa: E402
    DPOTrainer,
    EasyDeLState,
    FlaxLlamaForCausalLM,
    LlamaConfig,
    TrainArguments,
)
from easydel.modules.flax_modeling_utils import quantize_params_8bit  # noqa
from jax import numpy as jnp  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(
    split: str,
    sanity_check: bool = False,
    silent: bool = False,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        split=split,
        cache_dir=cache_dir,
    )
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)


def main():
    num_devices = len(jax.devices())
    sharding_axis_dims = (1, 1, 1, -1)

    max_length = 512
    max_target_length = 256
    max_prompt_length = 256
    input_shape = (num_devices, max_length)
    dtype = jnp.bfloat16

    # assert len(jax.devices("cpu")) == 8, "XLA Device manipulation failed."
    with jax.default_device(jax.devices("cpu")[0]):
        model_name_or_path = "erfanzar/LLamaStory-70M"
        conf = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=512,
            use_scan_mlp=False,
            attention_bias=False,
        )
        arguments = TrainArguments(
            num_train_epochs=4,
            model_name="DPO_TEST",
            total_batch_size=8,
            use_wandb=False,
            learning_rate=7e-5,
            learning_rate_end=9e-6,
            warmup_steps=100,
            optimizer=ed.EasyDeLOptimizers.ADAMW,
            scheduler=ed.EasyDeLSchedulers.WARM_UP_COSINE,
            weight_decay=0.02,
            max_sequence_length=max_length,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
            sharding_array=sharding_axis_dims,
            gradient_accumulation_steps=1,
            init_input_shape=input_shape,
            dtype=dtype,
            param_dtype=dtype,
            step_start_point=0,
            do_last_save=False,
            training_time="7H",
            force_batch_and_gradient_accumulation_steps_calculation=False,
            track_memory=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        train_dataset = get_hh("train[:10%]", sanity_check=True)
        eval_dataset = get_hh("test[:10%]", sanity_check=True)

        module = FlaxLlamaForCausalLM(
            config=conf,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=True,
            input_shape=(8, 8),
        )
        ref_module = FlaxLlamaForCausalLM(
            config=conf,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            _do_init=True,
            input_shape=(8, 8),
        )

        state = EasyDeLState.load(
            module=module,
            apply_fn=module.__call__,
            params={"params": module.shard_params(module.params)},
        )
        ref_state = EasyDeLState.load(
            module=ref_module,
            apply_fn=ref_module.__call__,
            params={
                "params": quantize_params_8bit(
                    ref_module.shard_params(ref_module.params),
                )
            },
        )

        dpo_trainer = DPOTrainer(
            model_state=state,
            ref_model_state=ref_state,
            beta=0.1,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            arguments=arguments,
            max_length=max_length,
            max_target_length=max_target_length,
            max_prompt_length=max_prompt_length,
            # dataset_map_arguments={"num_proc": os.cpu_count()},
        )

        dpo_trainer.train()


if __name__ == "__main__":
    main()
