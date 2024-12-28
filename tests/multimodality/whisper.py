from jax import numpy as jnp
from transformers import (
	WhisperProcessor,
	WhisperTokenizer,
)

import easydel as ed


def main():
	REPO_ID = "openai/whisper-large-v3-turbo"
	model = ed.AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
		REPO_ID,
		dtype=jnp.float16,
		param_dtype=jnp.float16,
	)
	tokenizer = WhisperTokenizer.from_pretrained(REPO_ID)
	processor = WhisperProcessor.from_pretrained(REPO_ID)
	model = model.quantize()
	inference = ed.vWhisperInference(
		model=model,
		tokenizer=tokenizer,
		processor=processor,
	)
	result_url = inference(
		"https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
		return_timestamps=True,
	)
	print(result_url)


if __name__ == "__main__":
	main()
