import jax
from datasets import load_dataset
from easydel import (
    AutoEasyDeLModelForCausalLM,
    CausalLanguageModelTrainer,
    EasyDeLGradientCheckPointers,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLState,
    EasyDeLXRapTureConfig,
    TrainArguments,
    easystate_to_huggingface_model,
    get_models_by_type,
)
from flax.core import FrozenDict, unfreeze
from jax import numpy as jnp
from transformers import AutoTokenizer
from transformers import GemmaForCausalLM as ModuleTorch


def main(use_lora=False):
    pretrained_model_name_or_path = "google/gemma-2b-it"

    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices("cpu")[0],
        input_shape=(1, 1),
        device_map="auto",
        sharding_axis_dims=(1, -1, 1, 1),
    )

    config = model.config

    model_parameters = FrozenDict({"params": params})

    dtype = jnp.bfloat16
    config.add_basic_configurations(
        attn_mechanism="normal",
        block_b=1,
        block_q=128,
        block_k=128,
        block_k_major=128,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )

    max_length = 4096

    configs_to_initialize_model_class = {
        "config": config,
        "dtype": dtype,
        "param_dtype": dtype,
        "input_shape": (1, max_length),
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    rapture_config = (
        EasyDeLXRapTureConfig(
            model_parameters,
            lora_dim=64,
            fully_fine_tune_parameters=["embed_tokens"],
            lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],
            verbose=True,
        )
        if use_lora
        else None
    )

    dataset = load_dataset(
        "erfanzar/Zeus-v0.1-Llama",
        split="train",
    )

    def gemma_prompt(x):
        return (
            x.replace("[/INST]", "<end_of_turn>\n<start_of_turn>model\n")
            .replace("</s><s>[INST]", "<end_of_turn>\n")
            .replace("<s>[INST] <<SYS>>\n", "<start_of_turn>system\n")
            .replace("<s>[INST]", "<start_of_turn>user\n")
            .replace("<</SYS>>\n", "<end_of_turn>\n")
            .replace("<end_of_turn>\n\n", "<end_of_turn>\n")
        )

    def tokenization_process(data_chunk) -> dict:
        return tokenizer(
            gemma_prompt(data_chunk["prompt"]),
            add_special_tokens=False,
            max_length=max_length,
            padding="max_length",
        )

    dataset = dataset.map(
        tokenization_process, num_proc=18, remove_columns=dataset.column_names
    )

    train_args = TrainArguments(
        model_class=get_models_by_type(config.model_type)[1],
        configs_to_initialize_model_class=configs_to_initialize_model_class,
        custom_rule=config.get_partition_rules(True),
        model_name="Gemma-FineTune",
        num_train_epochs=2,
        learning_rate=5e-5,
        learning_rate_end=7e-6,
        warmup_steps=200,
        optimizer=EasyDeLOptimizers.ADAMW,
        scheduler=EasyDeLSchedulers.LINEAR,
        weight_decay=0.02,
        total_batch_size=64,
        max_sequence_length=max_length,
        gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=(1, -1, 1, 1),
        gradient_accumulation_steps=1,
        init_input_shape=(1, max_length),
        dtype=dtype,
        param_dtype=dtype,
        step_start_point=0,
        training_time="7H",
        rapture_config=rapture_config,
        wandb_entity=None,
    )

    trainer = CausalLanguageModelTrainer(
        train_args, dataset.shuffle().shuffle().shuffle(), checkpoint_path=None
    )

    model_parameters = model_parameters if not use_lora else None

    output = trainer.train(model_parameters=model_parameters, state=None)
    params = {
        "params": unfreeze(output.state.params)["params"]
        | {
            "lm_head": {
                "kernel": output.state.params["params"]["model"]["embed_tokens"][
                    "embedding"
                ].T
            }
        }
    }

    output.state = output.state.replace(params=FrozenDict(params))
    output.state.save_state("Jupyter-State.easy")
    with jax.default_device(jax.devices("cpu")[0]):
        model = easystate_to_huggingface_model(
            state=EasyDeLState.load_state("Jupyter-State.easy"),
            base_huggingface_module=ModuleTorch,
            config=config,
        )

    model = model.half()
    model.push_to_hub("Gemma-2B-Fine-tuned")
    tokenizer.push_to_hub("Gemma-2B-Fine-tuned")


if __name__ == "__main__":
    main()
