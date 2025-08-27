# Vision-Language Models

EasyDeL provides comprehensive support for vision-language models (VLMs), enabling you to work with multimodal systems that process both images and text. This page demonstrates how to use various vision-language models with EasyDeL.

## Supported Models

EasyDeL supports a wide range of vision-language models, including:

- **LLaVA**: Large Language and Vision Assistant
- **Gemma3**: Google's multimodal model with vision capabilities
- **Aya Vision**: A powerful vision-language model from Cohere
- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model
- **SigLIP**: Google's Sign and Language Image Pre-training model
- **Qwen2VL**: Qwen's vision-language model

## Basic Usage Pattern

Most vision-language models in EasyDeL follow a similar pattern:

1. Load the processor/tokenizer for handling text and images
2. Initialize the model with appropriate configuration
3. Create inputs by applying a chat template with images
4. Use the model directly or through vInference for generation

## LLaVA Model Example

[LLaVA](https://github.com/haotian-liu/LLaVA) (Large Language and Vision Assistant) is a popular open-source vision-language model that connects a vision encoder with a language model.

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

# Setup parameters
prefill_length = 2048
max_new_tokens = 1024
max_length = max_new_tokens + prefill_length
model_name = "llava-hf/llava-1.5-7b-hf"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_name)
processor.padding_side = "left"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    model_name,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        freq_max_position_embeddings=max_length,
        mask_max_position_embeddings=max_length,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    ),
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
)

# Prepare input with image and text
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]

# Process inputs
inputs = processor.apply_chat_template(
    messages,
    return_tensors="jax",
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
)

# Initialize inference
inference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=max_new_tokens,
        sampling_params=ed.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=10,
        ),
        eos_token_id=model.generation_config.eos_token_id,
        streaming_chunks=32,
        num_return_sequences=1,
    ),
)

# Precompile for specific dimensions to optimize performance
inference.precompile(
    ed.vInferencePreCompileConfig(
        batch_size=1,
        prefill_length=prefill_length,
        vision_included=True,  # Important for vision models
        vision_batch_size=1,   # Number of images
        vision_channels=3,     # RGB channels
        vision_height=336,     # Image height
        vision_width=336,      # Image width
    )
)

# Generate response
for response in inference.generate(**inputs):
    pass  # Process streaming tokens if needed

# Get the final result
result = processor.batch_decode(
    response.sequences[..., response.padded_length:],
    skip_special_tokens=True,
)[0]
print(result)
```

## CLIP Image-Text Matching

[CLIP](https://openai.com/research/clip) (Contrastive Language-Image Pre-training) can be used for zero-shot image classification, image-text similarity, and more:

```python
import easydel as ed
import jax
from transformers import CLIPProcessor
from PIL import Image
import requests

# Load model and processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = ed.AutoEasyDeLModelForZeroShotImageClassification.from_pretrained(
    "openai/clip-vit-base-patch32"
)

# Load an image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # cat image
image = Image.open(requests.get(url, stream=True).raw)

# Process inputs
inputs = processor(
    text=[
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a person",
        "a photo of a car",
    ],
    images=image,
    return_tensors="np",
    padding=True,
)

# Get predictions
with model.mesh:
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = jax.nn.softmax(logits_per_image, axis=1)

    print(f"Prediction probabilities: {probs[0]}")
    # Should show highest probability for "a photo of a cat"
```

## SigLIP Image-Text Matching

[SigLIP](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/image_text/siglip) (Sign and Language Image Pre-training) is Google's vision-language model:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from PIL import Image
import requests
from transformers import AutoProcessor

# Load image
image = Image.open(
    requests.get(
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        stream=True
    ).raw
)

# Load processor and models
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model = ed.AutoEasyDeLModel.from_pretrained(
    "google/siglip-base-patch16-224",
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
)

# Prepare inputs
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
inputs = processor(
    text=texts,
    images=image,
    padding="max_length",
    return_tensors="jax"
)

# Get predictions
with model.mesh:
    outputs = model(**inputs)

# Process results
probs = jax.nn.sigmoid(outputs.logits_per_image)
for i, text in enumerate(texts):
    print(f"{probs[0][i]:.1%} probability that the image is '{text}'")
```

## Aya Vision

[Aya Vision](https://aya.for.ai/) is a powerful open multilingual VLM:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

# Load processor and model
processor = AutoProcessor.from_pretrained("CohereForAI/aya-vision-8b")
processor.padding_side = "left"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    "CohereForAI/aya-vision-8b",
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    ),
    quantization_method=ed.EasyDeLQuantizationMethods.NF4,  # Quantization for efficiency
    param_dtype=jnp.float16,
    dtype=jnp.float16,
)

# Prepare input with image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]

# Process inputs and generate
inputs = processor.apply_chat_template(
    messages,
    return_tensors="jax",
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
)

inference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=1024,
        streaming_chunks=32,
    ),
)

# Generate response
inference.precompile(
    ed.vInferencePreCompileConfig(
        vision_included=True,
        vision_height=364,
        vision_width=364,
    )
)

result = inference.generate_text(**inputs)
print(result)
```

## Gemma3 Multimodal

Google's [Gemma3](https://blog.google/technology/developers/gemma-3/) supports multimodal inputs:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

# Load processor and model
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
processor.padding_side = "left"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    "google/gemma-3-4b-it",
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    ),
    param_dtype=jnp.bfloat16,
    dtype=jnp.float16,
)

# Prepare input with image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    },
]

# Process inputs
inputs = processor.apply_chat_template(
    messages,
    return_tensors="jax",
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
)

# Initialize inference
inference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=1024,
        sampling_params=ed.SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=10,
        ),
    ),
)

# Precompile for specific dimensions
inference.precompile(
    ed.vInferencePreCompileConfig(
        vision_included=True,
        vision_batch_size=1,
        vision_height=896,
        vision_width=896,
    )
)

# Generate and get result
result = inference.generate_text(**inputs)
print(result)
```

## Qwen2VL Model

[Qwen2VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) is Alibaba's vision-language model:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

# Configuration
min_pixels = 256 * 28 * 28
resized_height, resized_width = 420, 420

# Load processor and model
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=min_pixels,
    max_pixels=min_pixels,
    resized_height=resized_height,
    resized_width=resized_width,
)
processor.padding_side = "left"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    auto_shard_model=True,
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    ),
    param_dtype=jnp.bfloat16,
    dtype=jnp.float16,
)

# Prepare conversation with image
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://picsum.photos/seed/picsum/200/300",
                "min_pixels": min_pixels,
                "max_pixels": min_pixels,
                "resized_height": resized_height,
                "resized_width": resized_width,
            },
            {"type": "text", "text": "Describe what you see in this image."},
        ],
    },
]

# For Qwen2VL, special processing is required
from qwen_vl_utils import process_vision_info  # Import the utility
image_inputs, video_inputs = process_vision_info(messages)

# Process inputs
inputs = processor(
    text=[processor.apply_chat_template(messages, add_generation_prompt=True)],
    images=image_inputs,
    videos=video_inputs,
    max_length=2048,
    padding="max_length",
    return_tensors="jax",
)

# Initialize inference
inference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=128,
        sampling_params=ed.SamplingParams(
            temperature=0.8,
            top_p=0.95,
        ),
    ),
)

# Generate response
result = inference.generate_text(**inputs)
print(result)
```

## Advanced Features

### Quantization Options

To reduce memory footprint without significant quality loss, you can apply quantization:

```python
model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_method=ed.EasyDeLQuantizationMethods.NF4,  # Use NF4 quantization
    # Other parameters...
)
```

Supported quantization methods:

- `NF4`: 4-bit quantization for efficient inference
- `A8BIT`: 8-bit quantization
- `NONE`: No quantization (default)

### Memory Optimization with Attention Mechanisms

For large vision-language models, choose the appropriate attention mechanism:

```python
model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    "llava-hf/llava-1.5-13b-hf",  # Larger model
    config_kwargs=ed.EasyDeLBaseConfigDict(
        attn_mechanism=ed.AttentionMechanisms.FLASH_ATTENTION,  # More efficient attention
        # Other parameters...
    ),
    # Other parameters...
)
```

### Custom Image Dimensions

When working with different image sizes, make sure to precompile with the correct dimensions:

```python
inference.precompile(
    ed.vInferencePreCompileConfig(
        vision_included=True,
        vision_batch_size=1,
        vision_channels=3,
        vision_height=512,  # Custom height
        vision_width=768,   # Custom width
    )
)
```

## Performance Tips

1. **Use Quantization**: For large VLMs, use `NF4` or `A8BIT` quantization
2. **Optimize Attention**: Choose `FLASH_ATTENTION` for GPU or `SPLASH_ATTENTION` for TPU
3. **Precompile**: Always precompile with `inference.precompile()` for best performance
4. **Image Size**: Use the smallest image dimensions that give good results
5. **Batch Processing**: Process multiple images together when possible
6. **System Memory**: VLMs use more memory than text-only models; adjust your batch size accordingly
