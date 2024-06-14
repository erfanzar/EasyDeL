import transformers

import easydel
from easydel import (
    AutoEasyDeLModelForCausalLM,
    TrainArguments,
    EasyDeLOptimizers,
    EasyDeLSchedulers,
    EasyDeLGradientCheckPointers,
    EasyDeLState,
    EasyDeLXRapTureConfig,
    SFTTrainer,
    get_modules_by_type,
    easystate_to_huggingface_model,
    conversations_formatting_function
)
from datasets import load_dataset
from flax.core import FrozenDict
from transformers import AutoTokenizer
from jax import numpy as jnp, sharding
import jax
from transformers import AutoConfig
from huggingface_hub import HfApi
from easydel.etils import define_flags_with_default

PartitionSpec = sharding.PartitionSpec
api = HfApi()

FLAGS, DEF_FLAGS = define_flags_with_default(
    pretrained_model_name_or_path="",
    pretrained_model_name_or_path_tokenizer="",
    new_repo_id="",
    train_dataset="",
    model_name="SFT-EasyDeL",
    sharding_axis_dims=(1, -1, 1, 1),
    max_length=2048,
    input_shape=(8, 2048),
    use_lora=False,
    block_size=512,
    attn_mechanism="sharded_vanilla",
    weight_decay=0.02,
    total_batch_size=24,
    gradient_accumulation_steps=1,
    step_start_point=0,
    num_train_epochs=3,
    learning_rate=2e-5,
    learning_rate_end=9e-6,
    warmup_steps=7,
    optimizer=EasyDeLOptimizers.ADAMW,
    scheduler=EasyDeLSchedulers.WARM_UP_COSINE,
    gradient_checkpointing=EasyDeLGradientCheckPointers.NOTHING_SAVEABLE,
    lora_dim=64,
    fully_fine_tune_parameters=["embed_tokens"],
    lora_fine_tune_parameters=["q_proj", "v_proj", "k_proj", "o_proj"],
    packing_sft=True,
    messages_field="conversation",
    training_time="90H",
    _required_fields=["pretrained_model_name_or_path", "new_repo_id", "train_dataset"]
)


def main():
    pretrained_model_name_or_path_tokenizer = (
        FLAGS.pretrained_model_name_or_path_tokenizer if (
                FLAGS.pretrained_model_name_or_path_tokenizer != ""
        ) else FLAGS.pretrained_model_name_or_path
    )

    dtype = jnp.bfloat16
    sharding_axis_dims = eval(FLAGS.sharding_axis_dims)
    qps = PartitionSpec(("dp", "fsdp"), "sp", None, "tp")
    kps = PartitionSpec(("dp", "fsdp"), "sp", None, "tp")
    vps = PartitionSpec(("dp", "fsdp"), "sp", None, "tp")
    bps = PartitionSpec(("dp", "fsdp"), "sp", None, None)
    aps = PartitionSpec(("dp", "fsdp"), "sp", None, "tp")

    partition_axis = easydel.PartitionAxis(
        batch_axis=("dp", "fsdp"),
        query_sequence_axis="sp",
        key_sequence_axis="sp",
        head_axis="tp"
    )

    model, params = AutoEasyDeLModelForCausalLM.from_pretrained(
        FLAGS.pretrained_model_name_or_path,
        device=jax.devices('cpu')[0],
        input_shape=FLAGS.input_shape,
        device_map="auto",
        sharding_axis_dims=sharding_axis_dims,
        config_kwargs=dict(
            use_scan_mlp=False,
            attn_mechanism=FLAGS.attn_mechanism,
            partition_axis=partition_axis
        ),
        partition_axis=partition_axis
    )

    config = model.config

    model_use_tie_word_embedding = config.tie_word_embeddings

    model_parameters = FrozenDict({"params": params})

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path_tokenizer,
        trust_remote_code=True
    )

    config.add_basic_configurations(
        attn_mechanism=FLAGS.attn_mechanism,
        shard_attention_computation=True,
        partition_axis=partition_axis
    )

    configs_to_initialize_model_class = {
        "config": config,
        "dtype": dtype,
        "param_dtype": dtype,
        "input_shape": FLAGS.input_shape
    }

    if tokenizer.pad_token == None:
        tokenizer.pad_token = tokenizer.eos_token

    rapture_config = EasyDeLXRapTureConfig(
        model_parameters,
        lora_dim=FLAGS.lora_dim,
        fully_fine_tune_parameters=FLAGS.fully_fine_tune_parameters,
        lora_fine_tune_parameters=FLAGS.lora_fine_tune_parameters,
        verbose=True
    ) if FLAGS.use_lora else None

    train_dataset = load_dataset(FLAGS.train_dataset, split="train")

    train_arguments = TrainArguments(
        model_class=get_modules_by_type(config.model_type)[1],
        configs_to_initialize_model_class=configs_to_initialize_model_class,
        custom_rule=config.get_partition_rules(True),

        num_train_epochs=FLAGS.num_train_epochs,
        learning_rate=FLAGS.learning_rate,
        learning_rate_end=FLAGS.learning_rate_end,
        warmup_steps=FLAGS.warmup_steps,
        optimizer=FLAGS.optimizer,
        scheduler=FLAGS.scheduler,
        weight_decay=FLAGS.weight_decay,
        total_batch_size=FLAGS.total_batch_size,
        init_input_shape=FLAGS.input_shape,
        max_sequence_length=FLAGS.max_length,
        model_name=FLAGS.model_name,
        training_time=FLAGS.training_time,
        gradient_checkpointing=FLAGS.gradient_checkpointing,
        sharding_array=sharding_axis_dims,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        step_start_point=FLAGS.step_start_point,

        dtype=dtype,
        param_dtype=dtype,

        force_batch_and_gradient_accumulation_steps_calculation=False,
        rapture_config=rapture_config,
        track_memory=True
    )

    trainer = SFTTrainer(
        arguments=train_arguments,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        dataset_text_field=None,
        formatting_func=lambda x: [
            conversations_formatting_function(tokenizer, messages_field=FLAGS.messages_field)(x)],
        packing=FLAGS.packing_sft,
        num_of_sequences=FLAGS.max_length,
    )

    output = trainer.train(
        model_parameters=model_parameters if not FLAGS.use_lora else None,
        state=None
    )

    api.create_repo(FLAGS.new_repo_id, exist_ok=True)

    api.create_repo(FLAGS.new_repo_id, exist_ok=True)
    file_path = "/".join(output.checkpoint_path.split("/")[:-1])
    output.state.module.save_pretrained(file_path, output.state.params, float_dtype=dtype)

    api.upload_folder(
        repo_id=FLAGS.new_repo_id,
        folder_path=file_path,
    )

    with jax.default_device(jax.devices("cpu")[0]):
        state = EasyDeLState.load_state(
            output.checkpoint_path,
            input_shape=FLAGS.input_shape,
        )

        if model_use_tie_word_embedding:
            state_new_params = {
                "params": state.params["params"] | {
                    "lm_head": {
                        "kernel": state.params["params"]["model"]["embed_tokens"]["embedding"].T
                    }
                }
            }

            state = state.replace(params=state_new_params)

    config = AutoConfig.from_pretrained(FLAGS.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_model_name_or_path)

    with jax.default_device(jax.devices("cpu")[0]):
        model = easystate_to_huggingface_model(
            state=state,
            base_huggingface_module=type(transformers.AutoModelForCausalLM.from_config(config)),
            config=config
        )

    half_model = model.half()

    tokenizer.push_to_hub(FLAGS.new_repo_id)
    half_model.push_to_hub(FLAGS.new_repo_id)


if __name__ == "__main__":
    main()
