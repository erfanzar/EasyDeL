# EasyDeL vInference API Server

The EasyDeL vInference API Server provides an OpenAI-compatible API for serving language models and multimodal models through JAX and EasyDeL. It supports both text-only and text+image interactions, with streaming, token counting, and other advanced features.

## Key Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API clients
- **Text & Multimodal Support**: Works with both text-only LLMs and vision-language models
- **Streaming Responses**: Progressive generation with minimal latency
- **Token Counting**: Calculate token usage for inputs
- **Model Management**: Serve multiple models from a single server
- **Performance Metrics**: Optional Prometheus-compatible metrics
- **Hardware Optimization**: JAX-powered acceleration on GPU/TPU

## API Endpoints

### Chat Completions API

```markdown
POST /v1/chat/completions
```

Generate a model response for the given chat conversation.

#### Text-Only Example Request

```json
{
  "model": "LLaMA",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": false
}
```

#### Multimodal Example Request

```json
{
  "model": "multimodal",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "image": "https://example.com/image.jpg"
        },
        {
          "type": "text",
          "text": "Describe this image in detail."
        }
      ]
    }
  ],
  "temperature": 0.8,
  "max_tokens": 300,
  "stream": false
}
```

#### Response Format

```json
{
  "id": "chat-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "LLaMA",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a field that uses quantum mechanics to perform calculations..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 120,
    "total_tokens": 134,
    "tokens_per_second": 42.5,
    "processing_time": 2.82
  }
}
```

### Token Counting API

```markdown
POST /v1/count_tokens
```

Count the number of tokens in a given text or conversation.

#### Request Example

```json
{
  "model": "LLaMA",
  "conversation": "Explain quantum computing in simple terms"
}
```

#### Response Format

```json
{
  "model": "LLaMA",
  "count": 7
}
```

### Models API

```markdown
GET /v1/models
```

Get the list of available models on the server.

#### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "id": "LLaMA",
      "object": "model",
      "owned_by": "easydel",
      "permission": []
    },
    {
      "id": "multimodal",
      "object": "model",
      "owned_by": "easydel",
      "permission": []
    }
  ]
}
```

### Health Check APIs

```markdown
GET /liveness
GET /readiness
```

Check if the API server is running and ready to receive requests.

## Setup Guide

### Prerequisites

- JAX with appropriate GPU/TPU backend
- EasyDeL
- FastAPI and dependencies

### Starting a Text-Only Model Server

```python
import jax
import easydel as ed
from jax import numpy as jnp
from transformers import AutoTokenizer

# Load the model
model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    # Model loading configuration...
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Create inference object
inference = ed.vInference(
    model=model,
    processor_class=tokenizer,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=2048,
        streaming_chunks=64,
        num_return_sequences=1,
    ),
    inference_name="LLaMA",
)

# Precompile for better performance
inference.precompile(
    ed.vInferencePreCompileConfig(
        batch_size=1,
        prefill_length=1024,
    )
)

# Start the API server
ed.vInferenceApiServer(inference).fire(
    port=8000
)
```

### Starting a Multimodal Model Server

```python
import jax
import easydel as ed
from jax import numpy as jnp
from transformers import AutoProcessor

# Load the processor and model
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.padding_side = "left"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    # Model loading configuration...
)

# Create inference object
inference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=1024,
        streaming_chunks=32,
        num_return_sequences=1,
    ),
    inference_name="multimodal",
)

# Precompile with vision settings
inference.precompile(
    ed.vInferencePreCompileConfig(
        batch_size=1,
        prefill_length=2048,
        vision_included=True,
        vision_batch_size=1,
        vision_channels=3,
        vision_height=336,
        vision_width=336,
    )
)

# Start the API server
ed.vInferenceApiServer(inference).fire(
    port=8000
)
```

## Client Usage Examples

### Text-Only Chat Completion

```python
import requests

api_url = "http://localhost:8000"
model_id = "LLaMA"
prompt = "Explain quantum computing in simple terms"

response = requests.post(
    f"{api_url}/v1/chat/completions",
    json={
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Streaming Chat Completion

```python
import json
import requests

api_url = "http://localhost:8000"
model_id = "LLaMA"
prompt = "Write a short story about a robot learning to paint"

response = requests.post(
    f"{api_url}/v1/chat/completions",
    json={
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        if line_text.startswith("data: "):
            data = line_text[6:]
            if data == "[DONE]":
                break
            try:
                json_data = json.loads(data)
                if "choices" in json_data and json_data["choices"]:
                    delta = json_data["choices"][0]["delta"]
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
            except json.JSONDecodeError:
                pass
```

### Multimodal Chat Completion

```python
import base64
import requests

api_url = "http://localhost:8000"
model_id = "multimodal"
text_prompt = "Describe this image in detail."
image_path = "path/to/image.jpg"

# Read and encode image as base64
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = requests.post(
    f"{api_url}/v1/chat/completions",
    json={
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{base64_image}"
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ],
        "temperature": 0.8,
        "max_tokens": 300
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Token Counting

```python
import requests

api_url = "http://localhost:8000"
model_id = "LLaMA"
text = "Explain quantum computing in simple terms"

response = requests.post(
    f"{api_url}/v1/count_tokens",
    json={
        "model": model_id,
        "conversation": text
    }
)

result = response.json()
print(f"Token count: {result['count']}")
```

### Listing Available Models

```python
import requests

api_url = "http://localhost:8000"

response = requests.get(f"{api_url}/v1/models")
result = response.json()

print("Available models:")
for model in result["data"]:
    print(f"- {model['id']}")
```

## Advanced Configuration

### Creating a Server with Multiple Models

```python
import easydel as ed

# Initialize your model inference objects
inference1 = ed.vInference(
    # Configuration for first model
    inference_name="model1"
)

inference2 = ed.vInference(
    # Configuration for second model
    inference_name="model2"
)

# Create a server with multiple models
server = ed.vInferenceApiServer({
    "model1": inference1,
    "model2": inference2
})

# Start the server
server.fire(port=8000)
```

### Customizing Sampling Parameters

```python
import easydel as ed

# Create inference with custom sampling parameters
inference = ed.vInference(
    # Other configuration...
    generation_config=ed.vInferenceConfig(
        max_new_tokens=2048,
        streaming_chunks=64,
        sampling_params=ed.SamplingParams(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )
    )
)
```

## Performance Considerations

- JAX compilation happens on the first inference, expect higher latency initially
- Precompilation improves first inference latency
- For multimodal models, optimize vision preprocessing parameters
- Adjust batch size and max_workers based on available hardware
- Streaming generation may have slightly higher total latency but better perceived responsiveness
