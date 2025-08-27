# Audio-Language Models

EasyDeL provides robust support for audio-language models, allowing you to process speech and audio with powerful pre-trained models. This page demonstrates how to use audio processing capabilities in EasyDeL, with a particular focus on Whisper for speech recognition and transcription.

## Whisper Speech Recognition

[Whisper](https://github.com/openai/whisper) is OpenAI's versatile speech recognition model that can transcribe and translate audio in multiple languages. EasyDeL offers optimized JAX/Flax implementations of Whisper for efficient inference.

## Basic Whisper Usage

Here's a simple example demonstrating how to use Whisper for speech transcription:

```python
import easydel as ed
from jax import numpy as jnp
from transformers import WhisperProcessor, WhisperTokenizer

# Load model and processors
model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo",  # Latest Whisper model
    dtype=jnp.bfloat16,               # Mixed precision for efficiency
    param_dtype=jnp.bfloat16
)

# Load tokenizer and processor
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

# Initialize vWhisperInference
inference = ed.vWhisperInference(
    model=model,
    tokenizer=tokenizer,
    processor=processor
)

# Transcribe audio from a URL
result = inference(
    "https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/F_0126_6y9m_1.wav",
    return_timestamps=True  # Include timestamps in the result
)

print(result)
```

## Advanced Configuration

For more control over the transcription process, you can configure the Whisper inference engine with additional parameters:

```python
import easydel as ed
from jax import numpy as jnp
from transformers import WhisperProcessor, WhisperTokenizer

# Load model with quantization for efficiency
model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3",
    dtype=jnp.bfloat16,
    param_dtype=jnp.bfloat16,
    quantization_method=ed.EasyDeLQuantizationMethods.A8BIT  # 8-bit quantization
)

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Configure vWhisperInference with specific Whisper configuration
inference = ed.vWhisperInference(
    model=model,
    tokenizer=tokenizer,
    processor=processor,
    inference_config=ed.vWhisperInferenceConfig(
        batch_size=1,
        max_length=256,
        return_timestamps=True,
        task="transcribe",         # Options: "transcribe" or "translate"
        language="en",             # Target language
        is_multilingual=True       # Enable multilingual support
    ),
    dtype=jnp.bfloat16
)

# Transcribe from a local file
result = inference(
    "path/to/local/audio.mp3",
    chunk_length_s=30,                # Process audio in 30-second chunks
    stride_length_s=5,                # Overlap between chunks
    batch_size=1
)

print(result)
```

## Processing Different Audio Formats

Whisper can handle various audio formats:

```python
# From a URL
result_url = inference("https://example.com/audio.wav")

# From a local file path
result_file = inference("path/to/local/audio.mp3")

# From a byte array (e.g., from an uploaded file)
with open("path/to/audio.wav", "rb") as f:
    audio_bytes = f.read()
result_bytes = inference(audio_bytes)
```

## Long-Form Audio Processing

For long audio files, Whisper automatically segments the audio:

```python
# Process a long podcast or lecture
result = inference(
    "https://example.com/long_podcast.mp3",
    chunk_length_s=30,        # Process in 30-second chunks
    stride_length_s=5,        # 5-second overlap between chunks
)
```

## Multilingual Support

Whisper can transcribe and translate multiple languages:

```python
# Transcribe French audio
fr_result = inference(
    "https://example.com/french_speech.wav",
    language="fr",            # Specify source language for better results
    task="transcribe"         # Keep original language
)

# Translate Spanish audio to English
es_en_result = inference(
    "https://example.com/spanish_speech.wav",
    language="es",            # Source language
    task="translate"          # Translate to English
)
```

## Running the Whisper API Server

EasyDeL provides a dedicated API server for Whisper, compatible with OpenAI's Whisper API:

```python
import easydel as ed
from jax import numpy as jnp

# Run the server with the specified model
ed.inference.vwhisper.run_server(
    model_name="openai/whisper-large-v3",
    host="0.0.0.0",
    port=8000,
    dtype=jnp.bfloat16
)
```

Alternatively, you can use the CLI tool:

```bash
python -m easydel.inference.vwhisper.cli --model openai/whisper-large-v3 --port 8000
```

## Using the Whisper API

Once the server is running, you can make requests to transcribe or translate audio:

```bash
# Transcription request
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@speech.mp3" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=json" \
  -F "language=en"

# Translation request
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@spanish-speech.mp3" \
  -F "model=openai/whisper-large-v3" \
  -F "response_format=json"
```

## API Response Format

The API returns responses in the following format:

```json
{
  "text": "This is the transcribed text from the audio file."
}
```

For more detailed output with timestamps:

```json
{
  "text": "This is the transcribed text with timestamps.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "This is the first segment"
    },
    {
      "id": 1,
      "start": 3.5,
      "end": 7.2,
      "text": "This is the second segment"
    }
  ]
}
```

## Running Whisper and vInference Together

You can run both Whisper and vInference API servers together by using the respective server implementations:

```python
import easydel as ed
import threading
from jax import numpy as jnp

# Run the Whisper server in a separate thread
whisper_thread = threading.Thread(
    target=ed.inference.vwhisper.run_server,
    kwargs={
        "model_name": "openai/whisper-large-v3",
        "host": "0.0.0.0",
        "port": 8000,
        "dtype": jnp.bfloat16
    }
)
whisper_thread.start()

# Run the vInference server for other models
llava_inference = ed.vInference(
    model=llava_model,
    processor_class=llava_processor,
    generation_config=ed.vInferenceConfig(max_new_tokens=1024),
    inference_name="llava"
)

ed.vInferenceApiServer(llava_inference, max_workers=4).fire(
    host="0.0.0.0",
    port=8001  # Different port from the Whisper server
)
```

## Performance Optimization

To optimize Whisper inference performance:

1. **Quantization**: Use `quantization_method=ed.EasyDeLQuantizationMethods.A8BIT` for faster inference with minimal quality loss

2. **Mixed Precision**: Use `dtype=jnp.bfloat16` and `param_dtype=jnp.bfloat16` for efficient computation

3. **Chunking**: Adjust `chunk_length_s` based on your audio length and memory constraints

4. **Batch Processing**: Process multiple shorter audio files in a batch for higher throughput

5. **Language Specification**: When the audio language is known, specify it with the `language` parameter for better results

## Tips for Best Results

1. **Audio Quality**: Higher quality audio yields better transcription accuracy

2. **Model Size**: Larger Whisper models (large-v3) provide better results but require more memory

3. **Language Hints**: Specifying the language helps for non-English audio

4. **Timestamps**: Use `return_timestamps=True` for audio-text alignment in applications

5. **Translation**: The "translate" task works well for converting non-English speech to English text
