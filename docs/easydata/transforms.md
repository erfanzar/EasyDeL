# Transforms

Transforms are functions that process examples one at a time. EasyData provides a rich set of transforms for tokenization, chat templates, and field manipulation.

## Transform Interface

All transforms implement a simple callable interface:

```python
class Transform(Protocol):
    def __call__(self, example: dict) -> dict: ...
```

## Chat Template Transforms

### ChatTemplateTransform

Converts conversational data to formatted text using tokenizer's chat template.

```python
from easydel.data import ChatTemplateTransform
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
transform = ChatTemplateTransform(tokenizer)

# Input
example = {
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
}

# Output
result = transform(example)
# {"text": "<s>[INST] Hello! [/INST] Hi there! </s>"}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | Any | Required | HuggingFace tokenizer with chat template |
| `messages_field` | str | `"messages"` | Field containing messages list |
| `output_field` | str | `"text"` | Field to store formatted text |
| `tools` | list | None | Tools for function calling templates |
| `convert_from_value_format` | bool | True | Auto-convert from/value to role/content |
| `drop_messages` | bool | True | Remove original messages field |

### MaybeApplyChatTemplate

Conditionally applies chat template only if data is conversational.

```python
from easydel.data import MaybeApplyChatTemplate

transform = MaybeApplyChatTemplate(tokenizer)

# Conversational - template applied
result = transform({"messages": [{"role": "user", "content": "Hi"}]})

# Non-conversational - passed through unchanged
result = transform({"text": "Hello world"})
```

### ConvertToChatML

Converts from/value format (ShareGPT style) to role/content (ChatML) format.

```python
from easydel.data import ConvertToChatML

transform = ConvertToChatML(
    role_mapping={"human": "user", "gpt": "assistant"},
)

# Input (ShareGPT format)
example = {
    "conversations": [
        {"from": "human", "value": "Hello!"},
        {"from": "gpt", "value": "Hi there!"},
    ]
}

# Output (ChatML format)
result = transform(example)
# {"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
```

## Tokenization Transforms

### TokenizedShardedSource

Wraps a source with on-the-fly tokenization.

```python
from easydel.data import TokenizedShardedSource, TokenizerConfig, JsonShardedSource
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")
source = JsonShardedSource("data/*.jsonl")

config = TokenizerConfig(
    name_or_path="meta-llama/Llama-2-7b",
    max_length=2048,
    truncation=True,
    padding=False,
    add_special_tokens=True,
)

tokenized = TokenizedShardedSource(
    source=source,
    tokenizer=tokenizer,
    tokenizer_config=config,
    content_field="text",
    additional_fields=["metadata"],  # Fields to preserve
)

for example in tokenized.open_shard(tokenized.shard_names[0]):
    print(example["input_ids"])
```

### TokenizerConfig

Configuration for tokenization behavior:

```python
from easydel.data import TokenizerConfig

config = TokenizerConfig(
    name_or_path="meta-llama/Llama-2-7b",
    max_length=2048,
    truncation=True,
    padding=False,  # or "max_length", "longest"
    add_special_tokens=True,
    return_attention_mask=True,
    trust_remote_code=True,
)
```

### TokenizerManager

Manages tokenizers with caching:

```python
from easydel.data import TokenizerManager, TokenizerConfig

manager = TokenizerManager()

config = TokenizerConfig(name_or_path="meta-llama/Llama-2-7b")
tokenizer = manager.get_tokenizer(config)

# Tokenize text
result = manager.tokenize_text(tokenizer, "Hello world", config)
# {"input_ids": [...], "attention_mask": [...]}

# Batch tokenization
results = manager.tokenize_batch(tokenizer, ["Hello", "World"], config)
# {"input_ids": [[...], [...]], "attention_mask": [[...], [...]]}
```

## Field Operations

### SelectFields

Keep only specified fields.

```python
from easydel.data import SelectFields

transform = SelectFields(["input_ids", "attention_mask"])

example = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "extra": "removed"}
result = transform(example)
# {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
```

### DropFields

Remove specified fields.

```python
from easydel.data import DropFields

transform = DropFields(["__source__", "metadata"])

example = {"text": "hello", "__source__": "ds1", "metadata": {...}}
result = transform(example)
# {"text": "hello"}
```

### RenameFields

Rename fields.

```python
from easydel.data import RenameFields

transform = RenameFields({"content": "text", "label": "target"})

example = {"content": "hello", "label": 1}
result = transform(example)
# {"text": "hello", "target": 1}
```

### AddField

Add a constant field.

```python
from easydel.data import AddField

transform = AddField("dataset_name", "alpaca")

example = {"text": "hello"}
result = transform(example)
# {"text": "hello", "dataset_name": "alpaca"}
```

### MapField

Transform a field with a function.

```python
from easydel.data import MapField

transform = MapField("text", lambda x: x.lower())

example = {"text": "HELLO WORLD"}
result = transform(example)
# {"text": "hello world"}
```

### ExtractField

Extract a field from nested structure.

```python
from easydel.data import ExtractField

transform = ExtractField("response.content", "answer")

example = {"response": {"content": "42", "metadata": {...}}}
result = transform(example)
# {"answer": "42", ...}
```

### CombineFields

Combine multiple fields.

```python
from easydel.data import CombineFields

transform = CombineFields(
    fields=["prompt", "response"],
    output_field="text",
    separator="\n\n",
)

example = {"prompt": "Question:", "response": "Answer"}
result = transform(example)
# {"text": "Question:\n\nAnswer", ...}
```

## Filter Transforms

### FilterByField

Filter based on field value.

```python
from easydel.data import FilterByField

transform = FilterByField("language", lambda x: x == "en")

# Returns example if condition met, None otherwise
result = transform({"text": "hello", "language": "en"})  # Returns example
result = transform({"text": "hola", "language": "es"})   # Returns None
```

### FilterNonEmpty

Filter out examples with empty fields.

```python
from easydel.data import FilterNonEmpty

transform = FilterNonEmpty(["text", "response"])

result = transform({"text": "hello", "response": "world"})  # Returns example
result = transform({"text": "", "response": "world"})       # Returns None
```

## Chaining Transforms

### ChainedTransform

Chain multiple transforms:

```python
from easydel.data import ChainedTransform, ChatTemplateTransform, SelectFields

pipeline = ChainedTransform([
    ConvertToChatML(role_mapping={"human": "user", "gpt": "assistant"}),
    ChatTemplateTransform(tokenizer),
    SelectFields(["text"]),
])

result = pipeline(example)
```

### TransformedShardedSource

Apply transforms to a source:

```python
from easydel.data import TransformedShardedSource, JsonShardedSource

source = JsonShardedSource("data/*.jsonl")

transformed = TransformedShardedSource(
    source=source,
    transform=ChatTemplateTransform(tokenizer),
)

for example in transformed.open_shard(transformed.shard_names[0]):
    print(example["text"])
```

## Trainer-Specific Transforms

EasyDeL provides trainer-specific transforms that handle the full preprocessing pipeline:

### SFTPreprocessTransform

```python
from easydel.trainers.transforms import SFTPreprocessTransform

transform = SFTPreprocessTransform(
    tokenizer=tokenizer,
    max_length=2048,
)
```

### DPOPreprocessTransform

```python
from easydel.trainers.transforms import DPOPreprocessTransform

transform = DPOPreprocessTransform(
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_completion_length=512,
)
```

### KTOPreprocessTransform

```python
from easydel.trainers.transforms import KTOPreprocessTransform

transform = KTOPreprocessTransform(
    tokenizer=tokenizer,
    max_prompt_length=512,
    max_completion_length=512,
)
```

### GRPOPreprocessTransform

```python
from easydel.trainers.transforms import GRPOPreprocessTransform

transform = GRPOPreprocessTransform(
    tokenizer=tokenizer,
    max_prompt_length=1024,
)
```

## Custom Transforms

Create your own transforms:

```python
from easydel.data.transforms import Transform, Example

class MyTransform(Transform):
    def __init__(self, param: str):
        self._param = param

    def __call__(self, example: Example) -> Example:
        result = dict(example)
        result["custom_field"] = self._param
        return result

    def __repr__(self) -> str:
        return f"MyTransform(param={self._param!r})"

# Use it
transform = MyTransform("value")
result = transform({"text": "hello"})
# {"text": "hello", "custom_field": "value"}
```

## Best Practices

1. **Chain transforms efficiently**: Put filtering transforms early to reduce processing
2. **Use batched tokenization**: For large datasets, use `batched_tokenize_iterator`
3. **Preserve fields explicitly**: Use `additional_fields` to keep needed metadata
4. **Test transforms individually**: Verify each transform before chaining

## Next Steps

- [Token Packing](pipeline.md#packing) - Pack tokenized sequences
- [Dataset Mixing](mixing.md) - Combine transformed datasets
- [Pipeline API](pipeline.md) - Full pipeline with transforms
