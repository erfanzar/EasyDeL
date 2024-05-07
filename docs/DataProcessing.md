## Data Processing

here in this case you will see an example data required by EasyDeL to pre-train or fine-tune models

```python
from datasets import load_dataset
from easydel.data_preprocessing import DataProcessor, DataProcessorArguments
from transformers import LlamaTokenizerFast


def main():
    tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    dataset = load_dataset("erfanzar/orca-lite")
    print(dataset)

    #     DatasetDict({
    #         train: Dataset({
    #             features: ['user', 'gpt', 'system', 'llama_2_prompt_style', 'prompt_length'],
    #             num_rows: 101397
    #         })
    #     })

    processor_arguments = DataProcessorArguments(
        max_position_embeddings=2048,
        num_proc=6,
        prompt_field='llama_2_prompt_style',

    )

    easydel_dataset = DataProcessor.process_data(
        data=dataset['train'],
        tokenizer=tokenizer,
        arguments=processor_arguments,
        field='train'
    )
    print(easydel_dataset)
    # DatasetDict({
    #     train: Dataset({
    #         features: ['input_ids', 'attention_mask'],
    #         num_rows: 101397
    #     })
    # })


if __name__ == "__main__":
    main()
```

now you can pass this data to Trainer and train your model 😇.