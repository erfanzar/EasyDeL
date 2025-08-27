# Multimodal Inference Serving

EasyDeL provides powerful tools for deploying and serving multimodal models through its vInference API. This page explains how to set up and use the vInference API Server for multimodal models, enabling efficient and scalable deployment of vision-language and audio-language models.

## Overview

The vInference API Server in EasyDeL lets you expose multimodal models through a REST API compatible with the OpenAI API format. This makes it easy to integrate these models into applications, websites, and services.

## Setting Up a Multimodal API Server

Here's how to set up a multimodal API server with LLaVA as an example:

```python
import easydel as ed
import jax
from jax import numpy as jnp
from transformers import AutoProcessor

# Configuration
prefill_length = 2048
max_new_tokens = 1024
model_name = "llava-hf/llava-1.5-7b-hf"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_name)
processor.padding_side = "left"

model = ed.AutoEasyDeLModelForImageTextToText.from_pretrained(
    model_name,
    auto_shard_model=True,
    sharding_axis_dims=(1, 1, 1, -1, 1),
    config_kwargs=ed.EasyDeLBaseConfigDict(
        freq_max_position_embeddings=prefill_length + max_new_tokens,
        mask_max_position_embeddings=prefill_length + max_new_tokens,
        attn_mechanism=ed.AttentionMechanisms.VANILLA,
    ),
    param_dtype=jnp.bfloat16,
    dtype=jnp.float16,
)

# Create vInference instance
inference = ed.vInference(
    model=model,
    processor_class=processor,
    generation_config=ed.vInferenceConfig(
        max_new_tokens=max_new_tokens,
        streaming_chunks=32,
        sampling_params=ed.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=10,
        ),
        eos_token_id=model.generation_config.eos_token_id,
    ),
    inference_name="mmprojector",  # Name for the API endpoint
)

# Precompile for maximum performance
inference.precompile(
    ed.vInferencePreCompileConfig(
        batch_size=1,
        prefill_length=prefill_length,
        vision_included=True,  # Enable vision processing
        vision_batch_size=1,
        vision_channels=3,
        vision_height=336,
        vision_width=336,
    )
)

# Start the API server
ed.vInferenceApiServer(inference, max_workers=1).fire(
    host="0.0.0.0",
    port=8000
)
```

## API Request Format

The API follows an OpenAI-compatible format. Here's an example request for the LLaVA model:

```json
{
    "model": "mmprojector",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ]
        }
    ],
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 1024,
    "stream": false
}
```

## Streaming Responses

For interactive applications, you can enable streaming to get tokens as they're generated:

```json
{
    "model": "mmprojector",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                },
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                }
            ]
        }
    ],
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 1024,
    "stream": true
}
```

## Serving Multiple Models

You can serve multiple models simultaneously with the vInferenceApiServer:

```python
# Setup a LLaVA model
llava_inference = ed.vInference(
    model=llava_model,
    processor_class=llava_processor,
    generation_config=llava_config,
    inference_name="llava",  # Name for the first model
)

# Setup a Whisper model
gemma_inference = ed.vInference(
    model=gemma_model,
    processor_class=gemma_processor,
    generation_config=gemma_config,
    inference_name="gemma3",  # Name for the second model
)

# Serve both models on the same server
ed.vInferenceApiServer(
    {
        "llava": llava_inference,
        "gemma3": gemma_inference
    },  # Dictionary mapping model names to inference engines
    max_workers=4
).fire(
    host="0.0.0.0",
    port=8000
)

# Alternatively, for a single model with automatic naming:
# ed.vInferenceApiServer(whisper_inference).fire(port=8000)
```

## Audio Model API Usage

For Whisper models, you can make requests with audio files:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@speech.mp3" \
  -F "model=whisper" \
  -F "response_format=json" \
  -F "language=en"
```

The response will be a JSON object with the transcription:

```json
{
  "text": "This is the transcribed text from the audio file."
}
```

## Advanced Configuration

### Load Balancing and Scaling

For high-traffic applications, configure multiple workers:

```python
ed.vInferenceApiServer(
    inference,
    max_workers=8           # More workers for parallel requests
).fire(
    host="0.0.0.0",
    port=8000
)
```

### Setting Metrics Port for Monitoring

To enable monitoring with Prometheus metrics:

```python
ed.vInferenceApiServer(inference).fire(
    port=8000,
    log_level="info"        # Logging level (debug, info, warning, error)
)
```

### Secure HTTPS Configuration

Enable HTTPS with SSL certificates:

```python
ed.vInferenceApiServer(inference).fire(
    port=443,
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem"
)
```

## Authentication and Security

For API key authentication, you'll need to implement custom middleware with FastAPI. Here's a simplified example:

```python
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery

API_KEYS = ["sk-abcdef1234567890", "sk-qwerty0987654321"]
API_KEY_NAME = "Authorization"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name=API_KEY_NAME, auto_error=False)

app = APP  # The FastAPI app used by vInferenceApiServer

async def get_api_key(
    api_key_header: str = Security(api_key_header),
    api_key_query: str = Security(api_key_query),
):
    if api_key_header and api_key_header.startswith("Bearer "):
        api_key = api_key_header.replace("Bearer ", "")
        if api_key in API_KEYS:
            return api_key
    if api_key_query in API_KEYS:
        return api_key_query
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

# Add the dependency to your chat completions endpoint
app.dependency_overrides[vInferenceApiServer.chat_completions] = lambda: Depends(get_api_key)
```

## Docker Deployment

For production deployment, use Docker. Example Dockerfile:

```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install JAX, EasyDeL and dependencies
RUN pip install --no-cache-dir easydel[all]

# Copy your model code
WORKDIR /app
COPY serve_model.py .

# Expose API port
EXPOSE 8000

# Run the server
CMD ["python3", "serve_model.py"]
```

## Client Integration Examples

### Python Client

```python
import requests
import base64
from PIL import Image
import io

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Encode the image
image_path = "path/to/image.jpg"
base64_image = encode_image(image_path)

# Prepare the API request
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-your-api-key"  # If API key is enabled
}
payload = {
    "model": "mmprojector",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": base64_image
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ]
        }
    ],
    "temperature": 0.7,
    "max_tokens": 300
}

# Make the API request
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

### JavaScript/TypeScript Client

```javascript
async function queryMultimodalModel() {
    // Convert the image to base64
    const imageFile = document.getElementById('imageInput').files[0];
    const base64Image = await convertToBase64(imageFile);

    // Prepare the API request
    const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer sk-your-api-key'  // If API key is enabled
        },
        body: JSON.stringify({
            model: 'mmprojector',
            messages: [
                {
                    role: 'user',
                    content: [
                        {
                            type: 'image',
                            image: base64Image
                        },
                        {
                            type: 'text',
                            text: 'What is in this image?'
                        }
                    ]
                }
            ],
            temperature: 0.7,
            max_tokens: 300
        })
    });

    const result = await response.json();
    document.getElementById('result').textContent = result.choices[0].message.content;
}

// Helper function to convert file to base64
function convertToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            let encoded = reader.result.toString().replace(/^data:(.*,)?/, '');
            resolve(encoded);
        };
        reader.onerror = error => reject(error);
    });
}
```

## Performance Tips

1. **Precompile with Exact Dimensions**: Always precompile with the exact image dimensions you'll use in production
2. **Quantization**: Use appropriate quantization methods for your model size and hardware
3. **Batch Size**: Adjust batch_size in precompile settings based on your expected traffic patterns
4. **Streaming Chunks**: Fine-tune the `streaming_chunks` parameter in vInferenceConfig to balance memory usage and compilation overhead
5. **Worker Count**: Set max_workers based on your CPU cores and expected concurrent requests
