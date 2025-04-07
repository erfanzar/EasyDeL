# EasyDeL Whisper API Server

The EasyDeL Whisper API Server provides an API compatible with OpenAI's Whisper API for audio transcription and translation. This allows you to run your own Whisper service using JAX/Flax via EasyDeL.

## Installation

The server requires FastAPI and other dependencies. Install them with:

```bash
pip install "fastapi[all]" uvicorn easydel
```

## Running the Server

### Option 1: Using the CLI

EasyDeL provides a built-in CLI to start the Whisper server:

```bash
python -m easydel.inference.vwhisper.cli --model "openai/whisper-large-v3-turbo" --port 8000
```

### Option 2: Using the Server Module

```python
import easydel as ed
from jax import numpy as jnp

ed.inference.vwhisper.run_server(
    model_name="openai/whisper-large-v3-turbo",
    host="0.0.0.0",
    port=8000,
    dtype=jnp.bfloat16
)
```

### Option 3: Using the Example Script

```bash
python examples/whisper_server_example.py --model "openai/whisper-large-v3-turbo" --port 8000
```

## API Endpoints

The server provides the following OpenAI-compatible endpoints:

### Transcription

`POST /v1/audio/transcriptions`

Transcribes audio into the same language as the audio.

**Parameters**:

- `file`: The audio file to transcribe (required)
- `model`: Model name (ignored, but required for OpenAI API compatibility)
- `prompt`: Optional text to guide the model's style
- `response_format`: The format of the transcript output (`json`, `text`, `srt`, `verbose_json`, `vtt`)
- `temperature`: Sampling temperature (0-1)
- `language`: Language code of the input audio
- `timestamp_granularities`: When returning timestamps, specify granularity (`word`, `segment`)

### Translation

`POST /v1/audio/translations`

Translates audio into English.

**Parameters**:

- `file`: The audio file to translate (required)
- `model`: Model name (ignored, but required for OpenAI API compatibility)
- `prompt`: Optional text to guide the model's style
- `response_format`: The format of the transcript output (`json`, `text`, `srt`, `verbose_json`, `vtt`)
- `temperature`: Sampling temperature (0-1)
- `timestamp_granularities`: When returning timestamps, specify granularity (`word`, `segment`)

## Client Usage

### Python Requests

```python
import requests

# For transcription
files = {"file": open("audio.mp3", "rb")}
data = {"model": "whisper-large-v3-turbo", "response_format": "json"}
response = requests.post("http://localhost:8000/v1/audio/transcriptions", files=files, data=data)
result = response.json()
print(result["text"])

# For translation
files = {"file": open("audio.mp3", "rb")}
data = {"model": "whisper-large-v3-turbo", "response_format": "json"}
response = requests.post("http://localhost:8000/v1/audio/translations", files=files, data=data)
result = response.json()
print(result["text"])
```

### Using the Example Client

```bash
python examples/whisper_client_example.py audio.mp3 --server http://localhost:8000 --mode transcribe --language en --timestamps
```

## Response Format

The server returns responses in a format compatible with the OpenAI Whisper API:

```json
{
  "text": "Transcribed or translated text",
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Text segment with timestamp"
    },
    ...
  ]
}
```

## Model Support

The server supports any Whisper model available on Hugging Face, including:

- `openai/whisper-tiny`
- `openai/whisper-base`
- `openai/whisper-small`
- `openai/whisper-medium`
- `openai/whisper-large-v3-turbo`
- `openai/whisper-large-v3`

## Performance Considerations

- The first request will be slower as the model is loaded and compiled
- Subsequent requests will be much faster
- Using smaller models can significantly improve speed
- Using bfloat16 provides a good balance of speed and accuracy

## Compatibility with OpenAI API

This API implementation is designed to be drop-in compatible with the [OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio). You can use existing clients by just changing the base URL.

## Limitations

- Some advanced OpenAI features may not be fully supported
- Performance will depend on your hardware (GPU/TPU recommended)
