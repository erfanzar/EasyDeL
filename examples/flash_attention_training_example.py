from EasyDel import (
    AutoEasyDelModelForCausalLM,
    TrainArguments,
    CausalLanguageModelTrainer,
    EasyDelOptimizers,
    EasyDelSchedulers,
    EasyDelGradientCheckPointers,
    get_modules_by_type
)
from datasets import load_dataset
from huggingface_hub import HfApi
from flax.core import FrozenDict
from transformers import AutoTokenizer
from jax import numpy as jnp
import jax
from fjformer import GenerateRNG

rng_g = GenerateRNG()
api = HfApi()


def launch():
    pretrained_model_name_or_path = "openlm-research/open_llama_3b_v2"  # You can change this to any other models
    # that you want like falcon mistral gpt etc...
    device_num = len(jax.devices())
    model, params = AutoEasyDelModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=(device_num, 1),
        device_map="auto"
    )

    config = model.config

    model_parameters = FrozenDict({"params": params})

    config.add_basic_configurations(
        attn_mechanism="flash",
        block_b=1,
        block_q=512,
        block_k=512,
        block_k_major=512
    )

    original_max_position_embeddings = config.max_position_embeddings
    config.freq_max_position_embeddings = config.max_position_embeddings
    config.max_position_embeddings = 2048
    config.c_max_position_embeddings = config.max_position_embeddings
    config.use_pjit_attention_force = False

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True
    )

    max_length = config.max_position_embeddings

    configs_to_init_model_class = {
        'config': config,
        'dtype': jnp.bfloat16,
        'param_dtype': jnp.bfloat16,
        'input_shape': (device_num, config.block_q)
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "YOUR_DATASET_HERE",
        split="train",
    )
    tokenization_process = lambda data_chunk: tokenizer(
        data_chunk["prompt"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length"
    )
    dataset = dataset.map(
        tokenization_process,
        num_proc=18,
        remove_columns=dataset.column_names
    )
    train_args = TrainArguments(
        model_class=get_modules_by_type(config.model_type)[1],
        configs_to_init_model_class=configs_to_init_model_class,
        custom_rule=config.get_partition_rules(True),
        model_name="FlashAttentionTest",
        num_train_epochs=2,
        learning_rate=8e-5,
        learning_rate_end=5e-05,
        warmup_steps=200,
        optimizer=EasyDelOptimizers.ADAMW,
        scheduler=EasyDelSchedulers.LINEAR,
        weight_decay=0.02,
        total_batch_size=16,
        max_length=max_length,
        gradient_checkpointing=EasyDelGradientCheckPointers.NOTHING_SAVEABLE,
        sharding_array=(1, -1, 1, 1),
        use_pjit_attention_force=False,
        gradient_accumulation_steps=2,
        dtype=jnp.bfloat16,
        init_input_shape=(8, config.block_q),
        step_start_point=0,
        training_time="7H"
    )

    trainer = CausalLanguageModelTrainer(
        train_args,
        dataset.shuffle().shuffle().shuffle(),
        checkpoint_path=None
    )

    output = trainer.train(
        model_parameters=model_parameters,
        state=None
    )


if __name__ == "__main__":
    launch()
