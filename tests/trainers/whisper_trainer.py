from datasets import Audio, load_dataset
from jax import lax
from jax import numpy as jnp
from transformers import WhisperProcessor, WhisperTokenizer

import easydel as ed

dataset_train = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "fa",
    split="train",
    streaming=True,
    trust_remote_code=True,
)
dataset_eval = None
REPO_ID = "openai/whisper-large-v3-turbo"

tokenizer = WhisperTokenizer.from_pretrained(
    REPO_ID,
    language="persian",
    task="transcribe",
)
processor = WhisperProcessor.from_pretrained(
    REPO_ID,
    language="persian",
    task="transcribe",
)

processor.tokenizer.padding_side = "right"


def _prepare_example(example):
    """Prepare a single example for training."""
    audio = example["audio"]
    example["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    transcription = example["sentence"]
    example["labels"] = processor.tokenizer(transcription).input_ids
    return example


def _post_processing(features):
    batch = processor.feature_extractor.pad(
        [{"input_features": features["input_features"]}],
        return_tensors="jax",
    )
    labels_batch = processor.tokenizer.pad(
        [{"input_ids": features["labels"]}],
        return_tensors="jax",
        padding="max_length",
        max_length=448,
    )
    labels = jnp.asarray(
        jnp.where(
            labels_batch["attention_mask"] == 1,
            labels_batch["input_ids"],
            -100,
        )
    )
    if labels[:, 0] == 50285:
        labels = labels[:, 1:]
    batch["labels"] = jnp.asarray(labels).squeeze(0)
    batch["decoder_attention_mask"] = labels_batch["attention_mask"].squeeze(0)
    batch["input_features"] = batch["input_features"].squeeze(0)
    return batch


def _is_valid_audio_length(length):
    """Check if audio length is within acceptable range."""
    return 5 < length < 30


columns_to_remove = [
    "accent",
    "age",
    "client_id",
    "down_votes",
    "gender",
    "locale",
    "path",
    "segment",
    "up_votes",
]
dataset_train = dataset_train.remove_columns(columns_to_remove)
dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))

dataset_train = dataset_train.map(
    _prepare_example,
    remove_columns=dataset_train.column_names,
)
dataset_train = dataset_train.filter(
    _is_valid_audio_length,
    input_columns=["input_length"],
)
dataset_train = dataset_train.map(
    _post_processing,
    remove_columns=["input_length"],
)


def main():
    model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
        REPO_ID,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        precision=lax.Precision("highest"),
        sharding_axis_dims=(1, -1, 1, 1, 1),
        sharding_axis_names=("dp", "fsdp", "ep", "tp", "sp"),
        partition_axis=ed.PartitionAxis(),
        auto_shard_model=True,
        config_kwargs=ed.EasyDeLBaseConfigDict(
            attn_dtype=jnp.float32,
            attn_mechanism=ed.AttentionMechanisms.VANILLA,
            use_scan_mlp=False,
        ),
    )
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    train_arguments = ed.TrainingArguments(
        total_batch_size=64,
        num_train_epochs=10,
        do_eval=False,
        learning_rate=1e-5,
        learning_rate_end=1e-6,
        optimizer=ed.EasyDeLOptimizers.ADAMW,
        scheduler=ed.EasyDeLSchedulers.LINEAR,
        gradient_accumulation_steps=1,
        do_last_save=True,
        save_steps=100,
        max_training_steps=100_000,
        save_total_limit=5,
        save_directory="/home/erfan/ASR-runtime",
        model_name="WhisperNu",
        use_wandb=True,
        log_steps=10,
        progress_bar_type="json",
    )
    trainer = ed.Trainer(
        model=model,
        arguments=train_arguments,
        dataset_train=dataset_train,
        dataset_eval=dataset_eval,
    )
    trainer.train()


if __name__ == "__main__":
    main()
