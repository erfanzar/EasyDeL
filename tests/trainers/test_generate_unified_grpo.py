"""
Lightweight smoke for GRPO generate_unified alignment without running full training.

This uses the GRPO trainer with a tiny config and a minimal tokenizer/model stub
available from the GRPO test helpers. It checks that prompt_ids/prompt_mask are
preserved and that completion_prompts length matches completions.
"""

import numpy as np

import easydel as ed
from tests.trainers._common import (
    dummy_reward_fn,
    get_tokenizer,
    load_causal_lm_model,
    load_preference_dataset,
    make_config,
)


def test_generate_unified_matches_prompt_ids():
    tokenizer = get_tokenizer()
    model = load_causal_lm_model()
    dataset = load_preference_dataset().select(range(2))

    prompts: list[str] = []
    for sample in dataset.select(range(2)):
        if "prompt" in sample:
            prompts.append(str(sample["prompt"]))
        elif "chosen" in sample:
            prompts.append(str(sample["chosen"]))
        else:
            prompts.append("Tell me a joke about cats.")

    tokenized = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
        return_attention_mask=True,
    )

    input_ids = np.asarray(tokenized["input_ids"], dtype=np.int32)
    attention_mask = np.asarray(tokenized["attention_mask"], dtype=np.int32)

    args = make_config(
        ed.GRPOConfig,
        "group-relative-policy-optimization",
        overrides={
            "max_prompt_length": 128,
            "max_completion_length": 32,
            "max_sequence_length": 200,
            "num_return_sequences": 8,
            "use_esurge_generation": True,
            "generation_max_new_tokens": 16,
            "generation_top_p": 0.9,
            "generation_temperature": 0.7,
            "use_wandb": False,
        },
    )

    trainer = ed.GRPOTrainer(
        arguments=args,
        model=model,
        reward_funcs=dummy_reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    results = trainer.generate_unified(
        input_ids=input_ids,
        attention_mask=attention_mask,
        state=trainer.model_state,
        use_esurge=args.use_esurge_generation,  # exercise eSurge path by default
        shard_inputs=False,
    )

    # Prompt tensors unchanged
    assert np.array_equal(np.asarray(results.prompt_ids), input_ids)
    assert np.array_equal(np.asarray(results.prompt_mask), attention_mask)

    # Completion prompts align with completion rows
    if results.completion_prompts is not None:
        assert len(results.completion_prompts) == results.completion_ids.shape[0]


if __name__ == "__main__":
    test_generate_unified_matches_prompt_ids()
