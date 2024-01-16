import jax
import termcolor
from datasets import load_dataset, Dataset

from lib.python.EasyDel.reinforcement_learning.trainer.dpo_trainer import DPOTrainer, TrainArguments
from absl.app import FLAGS, run
from transformers import AutoTokenizer
from lib.python.EasyDel import EasyDelState


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
            # "prompt": samples["prompt"],
            "chosen": [chosen[0]["content"] for chosen in samples["chosen"]],
            "rejected": [rejected[0]["content"] for rejected in samples["rejected"]],
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
        state = EasyDelState.from_pretrained(
            pretrained_model_name_or_path="gpt2"
        )

        ref_state = EasyDelState.from_pretrained(
            pretrained_model_name_or_path="gpt2"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dpo_trainer = DPOTrainer(
            state,
            ref_state,
            beta=0.1,
            train_dataset=dpo_data(),
            tokenizer=tokenizer,
            arguments=arguments,
            max_length=512,
            max_target_length=256,
            max_prompt_length=256
        )

        print(dpo_trainer)
        dpo_trainer.train()


if __name__ == "__main__":
    run(main)
