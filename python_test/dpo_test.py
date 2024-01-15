import jax
import termcolor
from datasets import load_dataset, Dataset

from lib.python.EasyDel.reinforcement_learning.trainer.dpo_trainer import DPOTrainer, PartitionerConfig, TrainArguments
from absl.app import FLAGS, run
from transformers import AutoTokenizer
from lib.python.EasyDel import AutoEasyDelModelForCausalLM, EasyDelState


def dpo_data():
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="test_prefs",
        use_auth_token=True
    )

    original_columns: list[str] | dict[str, list[str]] = dataset.column_names

    def return_prompt_and_responses(samples) -> dict:
        return {
            "prompt": [prompt for prompt in samples["prompt"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )
    return dataset


def main(argv):
    with jax.default_device(jax.devices("cpu")[0]):
        arguments = TrainArguments(
            num_train_epochs=4,
            model_name="DPO_TEST"
        )
        model = EasyDelState.from_pretrained(
            pretrained_model_name_or_path="erfanzar/LLamaStory-70M"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "ahxt/LiteLlama-460M-1T"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_df = dpo_data().to_pandas()
        train_df["chosen"] = train_df["chosen"].apply(lambda x: x[1]["content"])
        train_df["rejected"] = train_df["rejected"].apply(lambda x: x[1]["content"])
        train_df = train_df.dropna()
        val_df = train_df.sample(10)
        train_data = Dataset.from_pandas(train_df)
        val_data = Dataset.from_pandas(val_df)
        dpo_trainer = DPOTrainer(
            model,
            beta=0.1,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            arguments=arguments,
            max_length=512,
            max_target_length=256,
            max_prompt_length=256
        )

        print(dpo_trainer)


if __name__ == "__main__":
    run(main)
